import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import os

import torch
import torch.nn as torch_nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

# Fix for VS Code to save plots as images
matplotlib.use('Agg')

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper for colored tqdm postfix stats
def color_stat(label, val, good, bad, fmt='.1f'):
    if (good < bad and val <= good) or (good > bad and val >= good):
        c = '\033[92m'  # green
    elif (good < bad and val >= bad) or (good > bad and val <= bad):
        c = '\033[91m'  # red
    else:
        c = '\033[93m'  # yellow
    reset = '\033[0m'
    return f'{label}={c}{val:{fmt}}{reset}'

# CUSTOM MEC ENVIRONMENT
class MECOffloadingEnv:
    def __init__(self):
        self.num_locations = 3               # 0=Local, 1=Cloud, 2=MEC
        self.cpu_levels = [0.25, 0.5, 1.0]
        self.num_cpu_levels = len(self.cpu_levels)
        self.action_space_n = self.num_locations * self.num_cpu_levels
        self.observation_space_n = 5

        # Mobility settings
        self.area_size = 1000.0
        self.mec_pos = np.array([500.0, 500.0])
        self.user_pos = np.array([np.random.uniform(0, self.area_size), np.random.uniform(0, self.area_size)])
        self.max_speed = 5.0

        # Wireless channel
        self.bandwidth = 20e6
        self.p_tx = 0.5
        self.noise_power = 1e-10

        # Compute & energy params
        self.f_l = 1.5e9
        self.f_e = 10e9
        self.f_c = 50e9
        self.P_tx = 0.5
        self.T_bh = 0.2
        self.b_bh = 100e6
        self.zeta = 1e-27
        self.e_e = 5e-10
        self.e_c = 2e-10
        self.P_bh = 0.2

        # Reward params
        self.r_local = 0.0
        self.r_cloud = 0.3
        self.r_mec = 0.6
        self.lam = 5.0

        self.current_step = 0
        self.max_steps_per_episode = 50
        self.last_action = 0

    def reset(self):
        self.current_step = 0
        self.last_action = 0
        self.user_pos = np.array([np.random.uniform(0, self.area_size), np.random.uniform(0, self.area_size)])
        self.Ret = self.update_mobility_and_rate()
        return self._generate_state()

    def _generate_state(self):
        i_t = random.uniform(10e6, 20e6)
        c_t = random.uniform(0.8e9, 1.5e9)
        d_t = random.uniform(0.5, 1.5)
        Re_t = self.Ret
        state = np.array([i_t, c_t, d_t, Re_t, self.last_action], dtype=np.float32)
        return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    def decode_action(self, action):
        location = action // self.num_cpu_levels
        cpu_idx = action % self.num_cpu_levels
        rho = self.cpu_levels[cpu_idx]
        return location, cpu_idx, rho

    def update_mobility_and_rate(self):
        move = np.random.uniform(-self.max_speed, self.max_speed, 2)
        self.user_pos += move
        self.user_pos = np.clip(self.user_pos, 0, self.area_size)
        distance = np.linalg.norm(self.user_pos - self.mec_pos)
        distance = max(distance, 1.0)
        channel_gain = 1e-4 / (distance ** 2.5)
        snr = (self.p_tx * channel_gain) / self.noise_power
        return self.bandwidth * np.log2(1 + snr)

    def step(self, action, state):
        self.current_step += 1
        i_t = state[0][0].item()
        c_t = state[0][1].item()
        d_t = state[0][2].item()
        Re_t = state[0][3].item()

        location, cpu_idx, rho = self.decode_action(action)
        T_t, E_t = 0.0, 0.0

        if location == 0:  # Local
            f_exec = max(rho * self.f_l, 1e-9)
            T_t = c_t / f_exec
            E_t = c_t * self.zeta * (f_exec ** 2)
        elif location == 1:  # Cloud
            f_exec = max(rho * self.f_c, 1e-9)
            T_t = (i_t / Re_t) + self.T_bh + (c_t / f_exec)
            E_t = (self.P_tx * (i_t / Re_t)) + (self.P_bh * (i_t / self.b_bh)) + (c_t * self.e_c)
        elif location == 2:  # MEC
            f_exec = max(rho * self.f_e, 1e-9)
            T_t = (i_t / Re_t) + (c_t / f_exec)
            E_t = (self.P_tx * (i_t / Re_t)) + (c_t * self.e_e)

        deadline_penalty = max(0.0, T_t - d_t)
        reward_val = - (E_t) - (self.lam * deadline_penalty)
        reward = torch.tensor([reward_val], dtype=torch.float32, device=device)

        self.Ret = self.update_mobility_and_rate()
        done = (self.current_step >= self.max_steps_per_episode)
        self.last_action = action
        next_state = None if done else self._generate_state()

        info = {
            'energy': E_t,
            'delay': T_t,
            'deadline_violated': int(T_t > d_t),
            'location': location,
            'cpu_idx': cpu_idx,
            'rho': rho,
        }
        return next_state, reward, done, info


# DQN
class DQN(torch_nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = torch_nn.Linear(n_observations, 256)
        self.layer2 = torch_nn.Linear(256, 256)
        self.layer3 = torch_nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)


# Shared hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.001
LR = 5e-5
NUM_EPISODES = 1000


def smooth(x, w=20):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0))
    out = (c[w:] - c[:-w]) / w
    return np.concatenate([np.full(w-1, out[0]), out])

def rolling_std(x, w=20):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = np.std(x[lo:i+1])
    return out


def run_training(config_name, eps_start, eps_end, eps_decay, out_dir):
    """Train one DQN agent with the given exploration config and save dashboard."""
    os.makedirs(out_dir, exist_ok=True)

    env = MECOffloadingEnv()
    policy_net = DQN(env.observation_space_n, env.action_space_n).to(device)
    target_net = DQN(env.observation_space_n, env.action_space_n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = [0]  # list so closures can mutate

    def select_action(state):
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done[0] / eps_decay)
        steps_done[0] += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(env.action_space_n)]], device=device, dtype=torch.long)

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        if len(non_final_next_states_list) == 0:
            return
        non_final_next_states = torch.cat(non_final_next_states_list)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = torch_nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    episode_rewards = []
    episode_energy = []
    episode_delay = []
    episode_violation_rate = []
    episode_action_dist = []
    episode_eps = []

    print(f"\n=== Training [{config_name}] | EPS_START={eps_start} EPS_END={eps_end} EPS_DECAY={eps_decay} ===")
    pbar = tqdm(
        range(NUM_EPISODES),
        desc=f"{config_name}",
        ncols=130,
        colour='green',
        bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}'
    )

    for i_episode in pbar:
        state = env.reset()
        total_reward = 0
        ep_energies, ep_delays, ep_violations = [], [], []
        ep_locations = [0, 0, 0]

        for t in count():
            action = select_action(state)
            next_state, reward, done, info = env.step(action.item(), state)
            total_reward += reward.item()

            ep_energies.append(info['energy'])
            ep_delays.append(info['delay'])
            ep_violations.append(info['deadline_violated'])
            ep_locations[info['location']] += 1

            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_rewards.append(total_reward)
                episode_energy.append(np.mean(ep_energies))
                episode_delay.append(np.mean(ep_delays))
                episode_violation_rate.append(np.mean(ep_violations))
                total_acts = sum(ep_locations)
                episode_action_dist.append([c / total_acts for c in ep_locations])
                cur_eps = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done[0] / eps_decay)
                episode_eps.append(cur_eps)

                window = min(50, len(episode_rewards))
                pbar.set_postfix_str(
                    f"{color_stat('R', np.mean(episode_rewards[-window:]), good=-50, bad=-150)} "
                    f"{color_stat('viol%', np.mean(episode_violation_rate[-window:])*100, good=10, bad=40, fmt='.0f')} "
                    f"{color_stat('E(mJ)', np.mean(episode_energy[-window:])*1000, good=1, bad=5, fmt='.2f')} "
                    f"eps={cur_eps:.2f}"
                )
                break

    print(f'[{config_name}] training complete.')

    # Save raw metrics
    np.savez(
        os.path.join(out_dir, 'metrics.npz'),
        rewards=np.array(episode_rewards),
        energy=np.array(episode_energy),
        delay=np.array(episode_delay),
        violation_rate=np.array(episode_violation_rate),
        action_dist=np.array(episode_action_dist),
        eps=np.array(episode_eps),
    )

    # Dashboard
    episodes_x = np.arange(len(episode_rewards))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f'Phase 2 Dashboard — {config_name}', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    mean_r = smooth(episode_rewards, 20)
    std_r = rolling_std(episode_rewards, 20)
    ax.fill_between(episodes_x, mean_r - std_r, mean_r + std_r, alpha=0.25, color='steelblue', label='±1σ (20 ep)')
    ax.plot(episodes_x, mean_r, color='steelblue', linewidth=2, label='20-ep mean')
    ax.set_title('Episode Reward')
    ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(episodes_x, smooth(episode_violation_rate, 20) * 100, color='crimson', linewidth=2)
    ax.set_title('Deadline Violation Rate')
    ax.set_xlabel('Episode'); ax.set_ylabel('% of steps violating')
    ax.set_ylim(0, 100); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    dist = np.array(episode_action_dist)
    dist_smooth = np.stack([smooth(dist[:, i], 20) for i in range(3)], axis=1)
    ax.stackplot(episodes_x, dist_smooth.T * 100,
                 labels=['Local', 'Cloud', 'MEC'],
                 colors=['#888888', '#4a90d9', '#50c878'], alpha=0.85)
    ax.set_title('Action Location Distribution')
    ax.set_xlabel('Episode'); ax.set_ylabel('% of actions')
    ax.set_ylim(0, 100); ax.legend(loc='upper right'); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(episodes_x, smooth(np.array(episode_energy) * 1000, 20), color='darkorange', linewidth=2)
    ax.set_title('Average Energy per Step')
    ax.set_xlabel('Episode'); ax.set_ylabel('Energy (mJ)')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(episodes_x, smooth(episode_delay, 20), color='purple', linewidth=2)
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='avg deadline')
    ax.set_title('Average Delay per Step')
    ax.set_xlabel('Episode'); ax.set_ylabel('Delay (s)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(episodes_x, episode_eps, color='teal', linewidth=2)
    ax.set_title('Exploration (ε)')
    ax.set_xlabel('Episode'); ax.set_ylabel('ε')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    dash_path = os.path.join(out_dir, 'dashboard.png')
    plt.savefig(dash_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[{config_name}] dashboard saved → {dash_path}')

    return {
        'name': config_name,
        'rewards': episode_rewards,
        'violation_rate': episode_violation_rate,
        'eps': episode_eps,
    }


# EXPLORATION CONFIGS
CONFIGS = [
    # name,         eps_start, eps_end, eps_decay
    ('fast_decay',  1.0, 0.05, 2000),    # baseline — exploration dies fast
    ('slow_decay',  1.0, 0.05, 15000),   # smooth long decay
    ('persistent',  1.0, 0.15, 15000),   # slow decay + high floor
]

if __name__ == '__main__':
    base_dir = 'runs'
    os.makedirs(base_dir, exist_ok=True)

    results = []
    for name, eps_s, eps_e, eps_d in CONFIGS:
        out_dir = os.path.join(base_dir, name)
        res = run_training(name, eps_s, eps_e, eps_d, out_dir)
        results.append(res)

    # COMPARISON PLOT
    print("\nGenerating comparison plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Exploration Schedule Comparison', fontsize=14, fontweight='bold')

    colors = ['#e74c3c', '#3498db', '#2ecc71']

    ax = axes[0]
    for res, c in zip(results, colors):
        ax.plot(smooth(res['rewards'], 20), label=res['name'], color=c, linewidth=2)
    ax.set_title('Episode Reward (20-ep mean)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    for res, c in zip(results, colors):
        ax.plot(smooth(np.array(res['violation_rate']) * 100, 20), label=res['name'], color=c, linewidth=2)
    ax.set_title('Deadline Violation Rate')
    ax.set_xlabel('Episode'); ax.set_ylabel('% violating')
    ax.set_ylim(0, 100); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    for res, c in zip(results, colors):
        ax.plot(res['eps'], label=res['name'], color=c, linewidth=2)
    ax.set_title('Exploration (ε) Schedule')
    ax.set_xlabel('Episode'); ax.set_ylabel('ε')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    cmp_path = os.path.join(base_dir, 'comparison.png')
    plt.savefig(cmp_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Comparison plot saved → {cmp_path}')
    print('\nAll done.')