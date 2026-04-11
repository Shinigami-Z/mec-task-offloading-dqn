import math
from datetime import timedelta
import random

import time
from datetime import timedelta

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

def get_unique_run_dir(base_dir, name):
    """Return a run dir that doesn't clobber existing ones. Appends -Run(1), -Run(2)..."""
    candidate = os.path.join(base_dir, name)
    if not os.path.exists(candidate):
        return candidate
    i = 1
    while True:
        candidate = os.path.join(base_dir, f'{name}-Run({i})')
        if not os.path.exists(candidate):
            return candidate
        i += 1

def save_run_yaml(out_dir, cfg):
    """Write a human-readable YAML of every variable used in this run."""
    path = os.path.join(out_dir, 'config.yaml')
    with open(path, 'w') as f:
        f.write('# Auto-generated run configuration\n')
        for section, items in cfg.items():
            f.write(f'\n{section}:\n')
            for k, v in items.items():
                # format floats nicely, scientific for big/small
                if isinstance(v, float):
                    if abs(v) >= 1e5 or (0 < abs(v) < 1e-3):
                        vs = f'{v:.3e}'
                    else:
                        vs = f'{v:g}'
                elif isinstance(v, str):
                    vs = f'"{v}"'
                else:
                    vs = str(v)
                f.write(f'  {k}: {vs}\n')
    return path

def fmt_eta(seconds):
    return str(timedelta(seconds=int(seconds)))


def math_sanity_check():
    """Run one env.step under every action with a fixed synthetic state and print the breakdown.
    Lets you verify energy / delay / migration / privacy / reward math by eye."""
    print('\n' + '='*72)
    print(' MATH SANITY CHECK — fixed state, all 9 actions')
    print('='*72)

    env = MECOffloadingEnv()
    env.reset()
    # force a known state so the numbers are reproducible
    # physical units (we'll feed normalized into step as it expects)
    i_t, c_t, d_t, Re_t = 15e6, 1.0e9, 1.0, 5e7
    import torch as _t
    fake_state = _t.tensor([[i_t/1e7, c_t/1e9, d_t, Re_t/1e8, 0.0]],
                           dtype=_t.float32, device=device)

    print(f'\nFixed inputs:')
    print(f'  i_t (data bits)        = {i_t:.3e}')
    print(f'  c_t (cpu cycles)       = {c_t:.3e}')
    print(f'  d_t (deadline, s)      = {d_t}')
    print(f'  R_e(t) (uplink bits/s) = {Re_t:.3e}')
    print(f'  reward_scale           = {env.reward_scale}')
    print(f'  lambda                 = {env.lam}')
    print(f'  privacy: local={env.r_local} cloud={env.r_cloud} mec={env.r_mec}')
    print()

    loc_names = ['Local', 'Cloud', 'MEC']
    print(f"{'act':>3} {'loc':>6} {'rho':>5} {'T(s)':>10} {'E(J)':>12} "
          f"{'viol':>5} {'priv':>5} {'reward':>12}")
    print('-'*72)

    for a in range(env.action_space_n):
        # reset prev_location each time so no migration triggers
        env.prev_location = None
        env.current_step = 0
        _, reward, _, info = env.step(a, fake_state)
        loc = loc_names[info['location']]
        print(f"{a:>3} {loc:>6} {info['rho']:>5} "
              f"{info['delay']:>10.4f} {info['energy']:>12.4e} "
              f"{info['deadline_violated']:>5} {info['privacy']:>5.2f} "
              f"{reward.item():>12.5f}")

    # Now a migration test
    print('\nMigration test (Cloud → MEC with same state):')
    env.prev_location = 1  # pretend we were on Cloud
    env.current_step = 0
    _, reward, _, info = env.step(6, fake_state)  # action 6 = MEC, rho=0.25
    print(f"  migrated={info['migrated']}  T={info['delay']:.4f}s  "
          f"E={info['energy']:.4e}J  reward={reward.item():.5f}")
    print('='*72 + '\n')




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
        self.reward_scale = 1000.0   # divisor for reward normalization

        # Migration cost params (MEC <-> Cloud switching)
        self.b_mig = 100e6   # migration bandwidth (bits/s)
        self.j_mig = 1.0     # migration power (W)
        self.prev_location = None  # track last remote server for migration detection

        self.current_step = 0
        self.max_steps_per_episode = 50
        self.last_action = 0

    def reset(self):
        self.current_step = 0
        self.last_action = 0
        self.prev_location = None
        self.user_pos = np.array([np.random.uniform(0, self.area_size), np.random.uniform(0, self.area_size)])
        self.Ret = self.update_mobility_and_rate()
        return self._generate_state()

    def _generate_state(self):
        i_t = random.uniform(10e6, 20e6)
        c_t = random.uniform(0.8e9, 1.5e9)
        d_t = random.uniform(0.5, 1.5)
        Re_t = self.Ret
        # normalize to comparable scales so the network sees well-conditioned inputs
        state = np.array([
            i_t / 1e7,
            c_t / 1e9,
            d_t,
            Re_t / 1e8,
            self.last_action / self.action_space_n,
        ], dtype=np.float32)
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
        # un-normalize state values back to physical units for the physics calc
        i_t  = state[0][0].item() * 1e7
        c_t  = state[0][1].item() * 1e9
        d_t  = state[0][2].item()
        Re_t = state[0][3].item() * 1e8

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

        # Migration cost: only when switching between remote servers (MEC <-> Cloud)
        T_mig, E_mig = 0.0, 0.0
        migrated = 0
        if (self.prev_location in (1, 2)) and (location in (1, 2)) and (location != self.prev_location):
            T_mig = i_t / self.b_mig
            E_mig = self.j_mig * T_mig
            migrated = 1
        T_t += T_mig
        E_t += E_mig

        # Privacy penalty (higher for edge than cloud, zero for local)
        if location == 0:
            privacy = self.r_local
        elif location == 1:
            privacy = self.r_cloud
        else:
            privacy = self.r_mec

        deadline_penalty = max(0.0, T_t - d_t)
        reward_val = - E_t - (self.lam * deadline_penalty) - privacy
        reward_val = reward_val / self.reward_scale   # normalize
        reward = torch.tensor([reward_val], dtype=torch.float32, device=device)

        self.Ret = self.update_mobility_and_rate()
        done = (self.current_step >= self.max_steps_per_episode)
        self.last_action = action
        self.prev_location = location
        next_state = None if done else self._generate_state()

        info = {
            'energy': E_t,
            'delay': T_t,
            'deadline_violated': int(T_t > d_t),
            'location': location,
            'cpu_idx': cpu_idx,
            'rho': rho,
            'migrated': migrated,
            'privacy': privacy,
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


# Shared hyperparameters (defaults — can be overridden per-run)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.001
LR = 5e-5
NUM_EPISODES = 1000
REPLAY_CAPACITY = 100000


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


def print_menu():
    bar = '=' * 60
    print(f'\n╔{bar}╗')
    print(f'║{"MEC DQN Offloading — Trainer":^60}║')
    print(f'╠{bar}╣')
    print(f'║  [1] Run ALL configs (full comparison)                     ║')
    print(f'║  [2] fast_decay  only                                      ║')
    print(f'║  [3] slow_decay  only                                      ║')
    print(f'║  [4] persistent  only                                      ║')
    print(f'║  [5] Custom run (choose your own params)                   ║')
    print(f'║  [6] Math sanity check (verify physics)                    ║')
    print(f'║  [7] Quit                                                  ║')
    print(f'╚{bar}╝')

def ask(prompt, default, cast=float):
    raw = input(f"  {prompt} [default: {default}]: ").strip()
    if raw == '':
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"  ! Invalid input. Using default value: {default}")
        return default

def custom_run_config():
    """Interactively gather params for a custom run. Returns (name, eps_s, eps_e, eps_d, overrides)."""
    print('\n  --- Custom Run Setup (press Enter to accept default) ---')
    name    = input('  Run name [custom]: ').strip() or 'custom'

    print('\n  Exploration:')
    eps_s   = ask('eps_start',  1.0)
    eps_e   = ask('eps_end',    0.05)
    eps_d   = ask('eps_decay',  15000, int)

    print('\n  Training:')
    episodes = ask('num_episodes',          NUM_EPISODES,    int)
    steps    = ask('max_steps_per_episode', 50,              int)
    batch    = ask('batch_size',            BATCH_SIZE,      int)
    buf      = ask('replay_capacity',       REPLAY_CAPACITY, int)
    lr       = ask('learning_rate',         LR)
    gamma    = ask('gamma',                 GAMMA)
    tau      = ask('tau',                   TAU)

    print('\n  Environment / reward:')
    lam         = ask('lambda (deadline penalty weight)', 5.0)
    reward_scl  = ask('reward_scale (divisor)',           1000.0)
    r_mec       = ask('privacy penalty MEC  (r_e)',       0.6)
    r_cloud     = ask('privacy penalty Cloud (r_c)',      0.3)
    b_mig       = ask('migration bandwidth (bits/s)',     100e6)
    j_mig       = ask('migration power (W)',              1.0)

    repeats = ask('\nHow many times to run this config', 1, int)
    if repeats < 1: 
        repeats = 1

    overrides = {
        'num_episodes': episodes,
        'max_steps_per_episode': steps,
        'batch_size': batch,
        'replay_capacity': buf,
        'lr': lr,
        'gamma': gamma,
        'tau': tau,
        'lam': lam,
        'reward_scale': reward_scl,
        'r_mec': r_mec,
        'r_cloud': r_cloud,
        'b_mig': b_mig,
        'j_mig': j_mig,
    }

    print('\n  --- Config summary ---')
    print(f'  name={name} | episodes={episodes} | steps/ep={steps}')
    print(f'  eps: {eps_s} → {eps_e} (decay={eps_d})')
    print(f'  batch={batch} buf={buf} lr={lr} γ={gamma} τ={tau}')
    print(f'  λ={lam} reward_scale={reward_scl}')
    print(f'  privacy: r_mec={r_mec} r_cloud={r_cloud}')
    print(f'  migration: b={b_mig} j={j_mig}')
    confirm = input('\n  Start training? [Y/n]: ').strip().lower()
    if confirm == 'n':
        return None
    return (name, eps_s, eps_e, eps_d, overrides, repeats)


def run_training(config_name, eps_start, eps_end, eps_decay, out_dir, overrides=None):
    """Train one DQN agent with the given exploration config and save dashboard."""
    os.makedirs(out_dir, exist_ok=True)

    run_start_time = time.time()

    # Build config dict for YAML dump
    ov_local = overrides or {}
    yaml_cfg = {
        'meta': {
            'config_name': config_name,
            'out_dir': out_dir,
        },
        'exploration': {
            'eps_start': eps_start,
            'eps_end':   eps_end,
            'eps_decay': eps_decay,
        },
        'training': {
            'num_episodes':    ov_local.get('num_episodes',    NUM_EPISODES),
            'batch_size':      ov_local.get('batch_size',      BATCH_SIZE),
            'replay_capacity': ov_local.get('replay_capacity', REPLAY_CAPACITY),
            'lr':              ov_local.get('lr',              LR),
            'gamma':           ov_local.get('gamma',           GAMMA),
            'tau':             ov_local.get('tau',             TAU),
            'max_steps_per_episode': ov_local.get('max_steps_per_episode', 50),
        },
        'environment': {
            'lam':          ov_local.get('lam',          5.0),
            'reward_scale': ov_local.get('reward_scale', 1000.0),
            'r_local':      0.0,
            'r_cloud':      ov_local.get('r_cloud',      0.3),
            'r_mec':        ov_local.get('r_mec',        0.6),
            'b_mig':        ov_local.get('b_mig',        100e6),
            'j_mig':        ov_local.get('j_mig',        1.0),
        },
    }
    
    save_run_yaml(out_dir, yaml_cfg)



    # Resolve hyperparams (allow per-run overrides)
    ov = overrides or {}
    batch_size = ov.get('batch_size',      BATCH_SIZE)
    gamma      = ov.get('gamma',           GAMMA)
    tau        = ov.get('tau',             TAU)
    lr         = ov.get('lr',              LR)
    num_eps    = ov.get('num_episodes',    NUM_EPISODES)
    buf_cap    = ov.get('replay_capacity', REPLAY_CAPACITY)

    env = MECOffloadingEnv()
    # apply env overrides
    if 'max_steps_per_episode' in ov: env.max_steps_per_episode = ov['max_steps_per_episode']
    if 'lam' in ov: 
        env.lam = ov['lam']
    if 'reward_scale' in ov: 
        env.reward_scale = ov['reward_scale']
    if 'r_mec' in ov: 
        env.r_mec = ov['r_mec']
    if 'r_cloud' in ov: 
        env.r_cloud = ov['r_cloud']
    if 'b_mig' in ov: 
        env.b_mig = ov['b_mig']
    if 'j_mig' in ov: 
        env.j_mig = ov['j_mig']

    policy_net = DQN(env.observation_space_n, env.action_space_n).to(device)
    target_net = DQN(env.observation_space_n, env.action_space_n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    memory = ReplayMemory(buf_cap)

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
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size)
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
        next_state_values = torch.zeros(batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * gamma) + reward_batch

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
        range(num_eps),
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
                target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
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
                    f"{color_stat('R', np.mean(episode_rewards[-window:]), good=-0.05, bad=-0.15, fmt='.3f')} "
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


    elapsed = time.time() - run_start_time
    # append elapsed into the YAML after training
    with open(os.path.join(out_dir, 'config.yaml'), 'a') as f:
        f.write(f'\nresults:\n')
        f.write(f'  elapsed_sec: {elapsed:.2f}\n')
        f.write(f'  elapsed_human: "{fmt_eta(elapsed)}"\n')
        f.write(f'  final_reward_mean_last50: {float(np.mean(episode_rewards[-50:])):.5f}\n')
        f.write(f'  final_violation_rate_last50: {float(np.mean(episode_violation_rate[-50:])):.5f}\n')
    print(f'[{config_name}] took {fmt_eta(elapsed)} — config + dashboard in {out_dir}')


    return {
        'name': config_name,
        'rewards': episode_rewards,
        'violation_rate': episode_violation_rate,
        'eps': episode_eps,
        'elapsed': elapsed,
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

    while True:
        print_menu()
        choice = input('  Select » ').strip()

        if choice == '7' or choice.lower() in ('q', 'quit', 'exit'):
            print('Bye.')
            break

        if choice == '6':
            math_sanity_check()
            continue

        # Build (name, eps_s, eps_e, eps_d, overrides, repeats) tuples
        if choice == '1':
            reps = ask('how many times to run each config', 1, int)
            selected = [(n, s, e, d, None, reps) for (n, s, e, d) in CONFIGS]
        elif choice == '2':
            n, s, e, d = CONFIGS[0]
            reps = ask('how many times to run', 1, int)
            selected = [(n, s, e, d, None, reps)]
        elif choice == '3':
            n, s, e, d = CONFIGS[1]
            reps = ask('how many times to run', 1, int)
            selected = [(n, s, e, d, None, reps)]
        elif choice == '4':
            n, s, e, d = CONFIGS[2]
            reps = ask('how many times to run', 1, int)
            selected = [(n, s, e, d, None, reps)]
        elif choice == '5':
            cfg = custom_run_config()
            if cfg is None:
                print('  Cancelled.')
                continue
            selected = [cfg]  # already has repeats in the tuple
        else:
            print('  ! Invalid choice, try again.')
            continue

        # Flatten into an execution queue, one entry per run
        run_queue = []
        for item in selected:
            name, eps_s, eps_e, eps_d, ov, reps = item
            for _ in range(reps):
                run_queue.append((name, eps_s, eps_e, eps_d, ov))

        total_runs = len(run_queue)
        print(f'\n  → Queued {total_runs} run(s) total.')

        results = []
        batch_start = time.time()
        per_run_times = []

        for idx, (name, eps_s, eps_e, eps_d, ov) in enumerate(run_queue, start=1):
            # Unique dir so same-name runs don't overwrite
            out_dir = get_unique_run_dir(base_dir, name)

            # Overall ETA printout
            if per_run_times:
                avg = sum(per_run_times) / len(per_run_times)
                remaining = avg * (total_runs - idx + 1)
                eta_str = fmt_eta(remaining)
            else:
                eta_str = 'estimating after first run...'
            print(f'\n╭─ Run {idx}/{total_runs}: {os.path.basename(out_dir)}')
            print(f'│  Batch elapsed: {fmt_eta(time.time() - batch_start)}   Batch ETA: {eta_str}')
            print(f'╰─')

            t0 = time.time()
            res = run_training(name, eps_s, eps_e, eps_d, out_dir, overrides=ov)
            per_run_times.append(time.time() - t0)
            results.append(res)

        batch_elapsed = time.time() - batch_start
        print(f'\n  ✓ All {total_runs} run(s) finished in {fmt_eta(batch_elapsed)}')

        if len(results) >= 2:
            print("\nGenerating comparison plot...")
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle('Exploration Schedule Comparison', fontsize=14, fontweight='bold')
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                      '#1abc9c', '#e67e22', '#34495e', '#c0392b', '#8e44ad']

            ax = axes[0]
            for i, res in enumerate(results):
                ax.plot(smooth(res['rewards'], 20),
                        label=f"{res['name']}#{i+1}",
                        color=colors[i % len(colors)], linewidth=2)
            ax.set_title('Episode Reward (20-ep mean)')
            ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
            ax.legend(fontsize=8); ax.grid(alpha=0.3)

            ax = axes[1]
            for i, res in enumerate(results):
                ax.plot(smooth(np.array(res['violation_rate']) * 100, 20),
                        label=f"{res['name']}#{i+1}",
                        color=colors[i % len(colors)], linewidth=2)
            ax.set_title('Deadline Violation Rate')
            ax.set_xlabel('Episode'); ax.set_ylabel('% violating')
            ax.set_ylim(0, 100); ax.legend(fontsize=8); ax.grid(alpha=0.3)

            ax = axes[2]
            for i, res in enumerate(results):
                ax.plot(res['eps'],
                        label=f"{res['name']}#{i+1}",
                        color=colors[i % len(colors)], linewidth=2)
            ax.set_title('Exploration (ε) Schedule')
            ax.set_xlabel('Episode'); ax.set_ylabel('ε')
            ax.legend(fontsize=8); ax.grid(alpha=0.3)

            plt.tight_layout()
            cmp_path = os.path.join(base_dir, f'comparison_{int(time.time())}.png')
            plt.savefig(cmp_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f'Comparison plot saved → {cmp_path}')

        print('\nDone. Returning to menu.')