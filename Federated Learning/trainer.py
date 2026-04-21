"""
MEC DQN Offloading — Phase 2 trainer with Federated Learning.
Made by Yousif Iskander.

Supports two training modes:
  • Single-agent DQN — one device, one policy (original Phase 1/2 setup)
  • Federated DQN    — multiple devices, collaborative training via FedAvg,
                       raw task data never leaves the device

FL here addresses a different privacy threat than the existing R(t) term:
  • R(t) — location-leakage at inference (what the agent reveals by its
           offloading choice when deployed)
  • FL   — data-leakage at training (the agent never has to upload its
           task history to a central server; only policy weights travel)
"""
import math
import os
import random
import time
from collections import namedtuple, deque
from datetime import timedelta
from itertools import count

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as torch_nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

matplotlib.use('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cuDNN autotuner — picks the fastest conv/matmul algorithms for a fixed input
# shape. Our tensors are fixed-size so this is a free speedup on GPU.
torch.backends.cudnn.benchmark = True


# ================================================================== #
# SECTION 1 — Utilities                                               #
# ================================================================== #
def set_seed(seed):
    """Seed every RNG we touch so runs are reproducible given the same seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def color_stat(label, val, good, bad, fmt='.1f'):
    """Colorize a tqdm postfix stat: green=good, red=bad, yellow=in between."""
    if (good < bad and val <= good) or (good > bad and val >= good):
        c = '\033[92m'
    elif (good < bad and val >= bad) or (good > bad and val <= bad):
        c = '\033[91m'
    else:
        c = '\033[93m'
    return f'{label}={c}{val:{fmt}}\033[0m'


def get_unique_run_dir(base_dir, name):
    """runs/<name>/Run(N) — auto-increments N to keep repeats grouped."""
    parent = os.path.join(base_dir, name)
    os.makedirs(parent, exist_ok=True)
    existing = [d for d in os.listdir(parent)
                if os.path.isdir(os.path.join(parent, d)) and d.startswith('Run(')]
    i = len(existing) + 1
    while os.path.exists(os.path.join(parent, f'Run({i})')):
        i += 1
    return os.path.join(parent, f'Run({i})')


def delete_all_runs(base_dir):
    """Wipe the entire runs/ directory after a confirmation prompt."""
    import shutil
    if not os.path.isdir(base_dir) or not os.listdir(base_dir):
        print(f'  {base_dir}/ is already empty. Nothing to delete.')
        return
    entries = os.listdir(base_dir)
    print(f'\n  ⚠  About to delete {len(entries)} item(s) from {base_dir}/:')
    for e in entries[:10]:
        print(f'     - {e}')
    if len(entries) > 10:
        print(f'     ... and {len(entries) - 10} more')
    confirm = input('\n  Type "DELETE" to confirm (anything else cancels): ').strip()
    if confirm != 'DELETE':
        print('  Cancelled.')
        return
    shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    print(f'  ✓ {base_dir}/ wiped.')


def save_run_yaml(out_dir, cfg):
    """Human-readable YAML dump of every hyperparameter for this run."""
    path = os.path.join(out_dir, 'config.yaml')
    with open(path, 'w') as f:
        f.write('# Auto-generated run configuration\n')
        for section, items in cfg.items():
            f.write(f'\n{section}:\n')
            for k, v in items.items():
                if isinstance(v, float):
                    vs = f'{v:.3e}' if (abs(v) >= 1e5 or (0 < abs(v) < 1e-3)) else f'{v:g}'
                elif isinstance(v, str):
                    vs = f'"{v}"'
                else:
                    vs = str(v)
                f.write(f'  {k}: {vs}\n')
    return path


def fmt_eta(seconds):
    return str(timedelta(seconds=int(seconds)))


def launch_tensorboard(base_dir, port=6006):
    """Fire up a TensorBoard server pointing at runs/. Non-blocking."""
    import subprocess
    import shutil
    import webbrowser

    if shutil.which('tensorboard') is None:
        print('  ! tensorboard not found. Install it with:  pip install tensorboard')
        return

    url = f'http://localhost:{port}'
    print(f'\n  Launching TensorBoard on {url}  (logdir={base_dir})')
    try:
        proc = subprocess.Popen(
            ['tensorboard', '--logdir', base_dir, '--port', str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f'  → PID {proc.pid}. Opening browser in 2s...')
        time.sleep(2)
        try:
            webbrowser.open(url)
        except Exception:
            print(f'  (could not auto-open browser — visit {url} manually)')
    except Exception as e:
        print(f'  ! Failed to launch TensorBoard: {e}')


def update_latest_symlink(base_dir, target):
    """Best-effort runs/latest symlink — fails silently on Windows without admin."""
    link = os.path.join(base_dir, 'latest')
    try:
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.abspath(target), link)
    except OSError:
        pass


# ================================================================== #
# SECTION 2 — Environment                                             #
# ================================================================== #
class MECOffloadingEnv:
    """Phase 2 MEC env. 3 execution locations × 3 CPU ratios = 9 actions.
    Mobility-driven uplink rate, migration cost between remote servers,
    privacy penalty that grows with location reveal."""

    def __init__(self):
        self.num_locations = 3               # 0=Local, 1=Cloud, 2=MEC
        self.cpu_levels = [0.25, 0.5, 1.0]
        self.num_cpu_levels = len(self.cpu_levels)
        self.action_space_n = self.num_locations * self.num_cpu_levels
        self.observation_space_n = 5

        # Mobility
        self.area_size = 1000.0
        self.mec_pos = np.array([500.0, 500.0])
        self.user_pos = np.array([np.random.uniform(0, self.area_size),
                                  np.random.uniform(0, self.area_size)])
        self.max_speed = 5.0

        # Wireless channel — paper Table 1, except We kept at 20 MHz
        # to match the paper's own numerical examples (Sec. 4), which
        # implicitly assume Re≈20 Mbps at typical distances. Strict
        # We=1 MHz makes offloading physically infeasible within the
        # 1s deadline from any position in a 1000×1000 m area.
        self.bandwidth = 20e6          # We = 20 MHz (deviation from Table 1)
        self.p_tx = 0.1                # P  = 0.1 W  for channel calc (Table 1)
        self.noise_power = 1e-9        # φ² = 1e-9   (Table 1)
        self.channel_h = 1.0           # h  = 1      (Table 1)
        self.path_loss_alpha = 3.0     # α  = 3      (Table 1)

        # Compute & energy params (from project spec — Table 1)
        self.f_l = 1.5e9
        self.f_e = 10e9
        self.f_c = 50e9
        self.P_tx = 0.5                # device Tx power for uplink energy (Table 1)
        self.T_bh = 0.2
        self.b_bh = 100e6
        self.zeta = 1e-27
        self.e_e = 5e-10
        self.e_c = 2e-10
        self.P_bh = 0.2

        # Reward params — paper eq. (1):  r = −(ω·E + (1−ω)·R) − λ·max(0, T−d)
        self.r_local = 0.0
        self.r_cloud = 0.3
        self.r_mec = 0.6
        self.omega = 0.7               # energy↔privacy trade-off (Phase 2 default)
        self.lam = 10.0                # deadline-violation weight (was 5.0)
        self.reward_scale = 1000.0

        # Migration cost (MEC ↔ Cloud switching only)
        self.b_mig = 100e6
        self.j_mig = 1.0
        self.prev_location = None

        self.current_step = 0
        self.max_steps_per_episode = 50
        self.last_action = 0

    def reset(self):
        self.current_step = 0
        self.last_action = 0
        self.prev_location = None
        self.user_pos = np.array([np.random.uniform(0, self.area_size),
                                  np.random.uniform(0, self.area_size)])
        self.Ret = self.update_mobility_and_rate()
        return self._generate_state()

    def _generate_state(self):
        # Task distribution from paper Table 1:
        #   i(t) ∈ [1, 5] MB → [8e6, 40e6] bits
        #   c(t) ∈ [0.5, 2] × 10⁹ cycles
        #   d(t) = 1 s  (fixed)
        i_t = random.uniform(8e6, 40e6)
        c_t = random.uniform(0.5e9, 2.0e9)
        d_t = 1.0
        Re_t = self.Ret
        # Normalize every feature to O(1) — keeps network inputs well-conditioned
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
        """Paper eq: R_e(t) = W · log2(1 + P·h·r^(-α) / φ²)"""
        move = np.random.uniform(-self.max_speed, self.max_speed, 2)
        self.user_pos += move
        self.user_pos = np.clip(self.user_pos, 0, self.area_size)
        distance = max(np.linalg.norm(self.user_pos - self.mec_pos), 1.0)
        channel_gain = self.channel_h / (distance ** self.path_loss_alpha)
        snr = (self.p_tx * channel_gain) / self.noise_power
        return self.bandwidth * np.log2(1 + snr)

    def step(self, action, state):
        self.current_step += 1
        # Un-normalize state back to physical units for the physics calc
        i_t  = state[0][0].item() * 1e7
        c_t  = state[0][1].item() * 1e9
        d_t  = state[0][2].item()
        Re_t = state[0][3].item() * 1e8

        location, cpu_idx, rho = self.decode_action(action)
        T_t, E_t = 0.0, 0.0

        if location == 0:    # Local execution
            f_exec = max(rho * self.f_l, 1e-9)
            T_t = c_t / f_exec
            E_t = c_t * self.zeta * (f_exec ** 2)
        elif location == 1:  # Cloud execution (uplink + backhaul + remote compute)
            f_exec = max(rho * self.f_c, 1e-9)
            T_t = (i_t / Re_t) + self.T_bh + (c_t / f_exec)
            E_t = (self.P_tx * (i_t / Re_t)) + (self.P_bh * (i_t / self.b_bh)) + (c_t * self.e_c)
        else:                # MEC execution (uplink + edge compute)
            f_exec = max(rho * self.f_e, 1e-9)
            T_t = (i_t / Re_t) + (c_t / f_exec)
            E_t = (self.P_tx * (i_t / Re_t)) + (c_t * self.e_e)

        # Migration cost only when switching between remote servers (MEC ↔ Cloud)
        T_mig, E_mig, migrated = 0.0, 0.0, 0
        if (self.prev_location in (1, 2)) and (location in (1, 2)) and (location != self.prev_location):
            T_mig = i_t / self.b_mig
            E_mig = self.j_mig * T_mig
            migrated = 1
        T_t += T_mig
        E_t += E_mig

        # Reward — paper eq. (1):  r = −(ω·E + (1−ω)·R) − λ·max(0, T−d)
        privacy = [self.r_local, self.r_cloud, self.r_mec][location]
        deadline_penalty = max(0.0, T_t - d_t)
        cost_EP = self.omega * E_t + (1.0 - self.omega) * privacy
        reward_val = (-cost_EP - self.lam * deadline_penalty) / self.reward_scale
        reward = torch.tensor([reward_val], dtype=torch.float32, device=device)

        self.Ret = self.update_mobility_and_rate()
        done = (self.current_step >= self.max_steps_per_episode)
        self.last_action = action
        self.prev_location = location
        next_state = None if done else self._generate_state()

        info = {
            'energy': E_t, 'delay': T_t,
            'deadline_violated': int(T_t > d_t),
            'location': location, 'cpu_idx': cpu_idx, 'rho': rho,
            'migrated': migrated, 'privacy': privacy,
        }
        return next_state, reward, done, info


class HeterogeneousMECEnv(MECOffloadingEnv):
    """MEC env with a per-client task-size multiplier. Used to simulate non-IID
    data across FL clients — some devices run heavier workloads than others
    while the physics and action space stay identical so the learned policy
    is still comparable across clients. Everything else is unchanged from
    MECOffloadingEnv."""

    def __init__(self, task_scale=1.0):
        super().__init__()
        self.task_scale = task_scale

    def _generate_state(self):
        # Same distribution as base env, scaled per client (non-IID)
        i_t = random.uniform(8e6, 40e6) * self.task_scale
        c_t = random.uniform(0.5e9, 2.0e9) * self.task_scale
        d_t = 1.0
        Re_t = self.Ret
        state = np.array([
            i_t / 1e7, c_t / 1e9, d_t, Re_t / 1e8,
            self.last_action / self.action_space_n,
        ], dtype=np.float32)
        return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)


# ================================================================== #
# SECTION 3 — DQN network + replay buffer                             #
# ================================================================== #
class DQN(torch_nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        # Paper Table 2: 2 hidden layers of 128 units
        self.layer1 = torch_nn.Linear(n_observations, 128)
        self.layer2 = torch_nn.Linear(128, 128)
        self.layer3 = torch_nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ================================================================== #
# SECTION 4 — Defaults                                                #
# ================================================================== #
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005              # raised from 0.001 to compensate for less-frequent soft updates
LR = 5e-5
NUM_EPISODES = 1000
REPLAY_CAPACITY = 100000
EVAL_EVERY = 50          # run greedy eval every N training episodes
EVAL_EPISODES = 10       # number of episodes per eval pass (averaged)
CHECKPOINT_EVERY = 200   # save resumable checkpoint every N episodes
TRAIN_FREQ = 4           # optimize every N env steps (standard DQN, ~3–4× speedup)
TB_STEP_LOG_FREQ = 50    # throttle per-step TB writes (loss/grad_norm)


def smooth(x, w=20):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0))
    out = (c[w:] - c[:-w]) / w
    return np.concatenate([np.full(w - 1, out[0]), out])


def rolling_std(x, w=20):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = np.std(x[lo:i + 1])
    return out


# ================================================================== #
# SECTION 5 — Sanity check                                            #
# ================================================================== #
def math_sanity_check():
    """Run every action on a fixed synthetic state and print the breakdown."""
    print('\n' + '=' * 72)
    print(' MATH SANITY CHECK')
    print('=' * 72)
    env = MECOffloadingEnv()
    env.reset()
    i_t, c_t, d_t, Re_t = 15e6, 1.0e9, 1.0, 5e7
    fake_state = torch.tensor([[i_t/1e7, c_t/1e9, d_t, Re_t/1e8, 0.0]],
                              dtype=torch.float32, device=device)
    loc_names = ['Local', 'Cloud', 'MEC']
    print(f"{'act':>3} {'loc':>6} {'rho':>5} {'T(s)':>10} {'E(J)':>12} "
          f"{'viol':>5} {'priv':>5} {'reward':>12}")
    print('-' * 72)
    for a in range(env.action_space_n):
        env.prev_location = None
        env.current_step = 0
        _, reward, _, info = env.step(a, fake_state)
        loc = loc_names[info['location']]
        print(f"{a:>3} {loc:>6} {info['rho']:>5} "
              f"{info['delay']:>10.4f} {info['energy']:>12.4e} "
              f"{info['deadline_violated']:>5} {info['privacy']:>5.2f} "
              f"{reward.item():>12.5f}")
    print('=' * 72 + '\n')


# ================================================================== #
# SECTION 6 — Greedy evaluation (shared by both modes)                #
# ================================================================== #
@torch.no_grad()
def run_eval(policy_net, env, n_episodes=10):
    """Run N greedy (ε=0) episodes on a separate env to measure true policy
    quality. Training reward is noisy because of exploration; eval reward shows
    what the learned policy would actually do in deployment."""
    policy_net.eval()
    rewards, viols, energies = [], [], []
    loc_counts = [0, 0, 0]
    for _ in range(n_episodes):
        state = env.reset()
        total_r = 0.0
        for _ in count():
            q = policy_net(state)
            action = q.max(1)[1].view(1, 1)
            next_state, reward, done, info = env.step(action.item(), state)
            total_r += reward.item()
            viols.append(info['deadline_violated'])
            energies.append(info['energy'])
            loc_counts[info['location']] += 1
            state = next_state
            if done:
                rewards.append(total_r)
                break
    policy_net.train()
    total = max(sum(loc_counts), 1)
    return {
        'reward': float(np.mean(rewards)),
        'violation_rate': float(np.mean(viols)),
        'energy': float(np.mean(energies)),
        'loc_dist': [c / total for c in loc_counts],
    }


# ================================================================== #
# SECTION 7 — Federated Learning: Client + FedAvg                     #
# ================================================================== #
class FederatedClient:
    """One mobile device. Owns env, policy/target networks, optimizer,
    and replay buffer. Replay buffer NEVER leaves the client — only
    policy-net weights are uploaded to the server during aggregation."""

    def __init__(self, client_id, obs_size, action_size, env, overrides=None):
        self.id = client_id
        self.env = env
        ov = overrides or {}
        self.batch_size = ov.get('batch_size', BATCH_SIZE)
        self.gamma      = ov.get('gamma', GAMMA)
        self.tau        = ov.get('tau', TAU)
        self.lr         = ov.get('lr', LR)
        self.train_freq = ov.get('train_freq', TRAIN_FREQ)

        self.policy_net = DQN(obs_size, action_size).to(device)
        self.target_net = DQN(obs_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(ov.get('replay_capacity', REPLAY_CAPACITY))
        self.criterion = torch_nn.SmoothL1Loss()

        # ε-greedy schedule advances with this client's own experience
        self.steps_done = 0
        self.global_step = 0

        # Per-client histories (for dashboards)
        self.episode_rewards = []
        self.episode_violations = []
        self.episode_energies = []

    # ---- FL communication primitives ----
    def set_weights(self, state_dict):
        """Receive the global model. Also resync target net — otherwise there's
        a large policy/target offset right after aggregation which destabilizes
        the TD update."""
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def get_weights(self):
        """Return CPU copies of local policy-net weights for aggregation.
        CPU avoids keeping GPU tensors alive in the aggregator; .clone()
        detaches from the autograd graph."""
        return {k: v.detach().cpu().clone() for k, v in self.policy_net.state_dict().items()}

    # ---- DQN internals (same math as the single-agent loop) ----
    def select_action(self, state, eps):
        if random.random() > eps:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        return torch.tensor([[random.randrange(self.env.action_space_n)]],
                            device=device, dtype=torch.long)

    def optimize_step(self):
        if len(self.memory) < self.batch_size:
            return None, None
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state),
                                       device=device, dtype=torch.bool)
        non_final_next = [s for s in batch.next_state if s is not None]
        if not non_final_next:
            return None, None
        non_final_next_states = torch.cat(non_final_next)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        sav = self.policy_net(state_batch).gather(1, action_batch)
        next_vals = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_vals[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected = (next_vals * self.gamma) + reward_batch

        loss = self.criterion(sav, expected.unsqueeze(1))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item(), grad_norm.item()

    def soft_update(self):
        with torch.no_grad():
            for tp, p in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)

    def local_train(self, num_episodes, eps_start, eps_end, eps_decay):
        """Run N episodes locally. ε decays smoothly across rounds because
        steps_done persists. Returns metrics + number of transitions collected
        (used as the FedAvg weight)."""
        rewards, vios, energies = [], [], []
        transitions_this_round = 0

        for _ in range(num_episodes):
            state = self.env.reset()
            total_r = 0.0
            ep_vio, ep_en = [], []

            for _ in count():
                eps = eps_end + (eps_start - eps_end) * math.exp(-self.steps_done / eps_decay)
                action = self.select_action(state, eps)
                next_state, reward, done, info = self.env.step(action.item(), state)
                total_r += reward.item()
                ep_vio.append(info['deadline_violated'])
                ep_en.append(info['energy'])

                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.steps_done += 1
                transitions_this_round += 1

                if self.global_step % self.train_freq == 0:
                    loss, _ = self.optimize_step()
                    if loss is not None:
                        self.soft_update()
                self.global_step += 1

                if done:
                    rewards.append(total_r)
                    vios.append(float(np.mean(ep_vio)))
                    energies.append(float(np.mean(ep_en)))
                    break

        self.episode_rewards.extend(rewards)
        self.episode_violations.extend(vios)
        self.episode_energies.extend(energies)

        return {
            'reward': float(np.mean(rewards)),
            'violation_rate': float(np.mean(vios)),
            'energy': float(np.mean(energies)),
            'transitions': transitions_this_round,
        }


def federated_average(client_weights, sample_counts=None):
    """Weighted FedAvg over a list of state_dicts.

    If sample_counts is given, each client's contribution is weighted by the
    number of transitions it collected — matches the McMahan et al. FedAvg
    formulation (clients with more data pull the global model harder).
    If None, falls back to uniform averaging.

    The DQN has no BatchNorm/running stats, so plain element-wise mean is
    sufficient; with BN you'd need to handle running_mean/var separately.
    """
    n = len(client_weights)
    if sample_counts is None or sum(sample_counts) == 0:
        weights = [1.0 / n] * n
    else:
        total = sum(sample_counts)
        weights = [c / total for c in sample_counts]

    avg = {}
    for key in client_weights[0].keys():
        ref_dtype = client_weights[0][key].dtype
        stacked = torch.stack([w[key].float() for w in client_weights])
        shape = [n] + [1] * (stacked.dim() - 1)
        w_t = torch.tensor(weights, dtype=stacked.dtype).view(*shape)
        avg[key] = (stacked * w_t).sum(0).to(ref_dtype)
    return avg


# ================================================================== #
# SECTION 8 — Menu / input helpers                                    #
# ================================================================== #
def print_menu():
    bar = '=' * 60
    print(f'\n╔{bar}╗')
    print(f'║{"MEC DQN Offloading — Trainer":^60}║')
    print(f'╠{bar}╣')
    print(f'║  Single-Agent DQN:                                         ║')
    print(f'║    [1]  Run ALL configs (full comparison)                  ║')
    print(f'║    [2]  fast_decay  only                                   ║')
    print(f'║    [3]  slow_decay  only                                   ║')
    print(f'║    [4]  persistent  only                                   ║')
    print(f'║    [5]  Custom run                                         ║')
    print(f'║                                                            ║')
    print(f'║  Federated Learning:                                       ║')
    print(f'║    [F1] Quick FL run (3 clients, 10 rounds)                ║')
    print(f'║    [F2] Standard FL run (5 clients, 50 rounds)             ║')
    print(f'║    [F3] Custom FL run                                      ║')
    print(f'║                                                            ║')
    print(f'║  Utilities:                                                ║')
    print(f'║    [6]  Math sanity check                                  ║')
    print(f'║    [7]  Launch TensorBoard                                 ║')
    print(f'║    [8]  Delete ALL runs                                    ║')
    print(f'║    [9]  Quit                                               ║')
    print(f'╚{bar}╝')


def ask(prompt, default, cast=float):
    raw = input(f"  {prompt} [default: {default}]: ").strip()
    if raw == '':
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"  ! Invalid input. Using default: {default}")
        return default


def custom_run_config():
    """Interactive wizard for a custom single-agent run."""
    print('\n  --- Custom Run Setup ---')
    name = input('  Run name [custom]: ').strip() or 'custom'

    print('\n  Exploration:')
    eps_s = ask('eps_start', 1.0)
    eps_e = ask('eps_end', 0.05)
    eps_d = ask('eps_decay', 15000, int)

    print('\n  Training:')
    episodes = ask('num_episodes', NUM_EPISODES, int)
    steps = ask('max_steps_per_episode', 50, int)
    batch = ask('batch_size', BATCH_SIZE, int)
    buf = ask('replay_capacity', REPLAY_CAPACITY, int)
    lr = ask('learning_rate', LR)
    gamma = ask('gamma', GAMMA)
    tau = ask('tau', TAU)
    train_freq = ask('train_freq (optimize every N steps)', TRAIN_FREQ, int)

    print('\n  Environment / reward:')
    lam = ask('lambda (deadline penalty)', 10.0)
    omega = ask('omega (E ↔ privacy, 1=energy only)', 0.7)
    reward_scl = ask('reward_scale', 1000.0)
    r_mec = ask('privacy MEC', 0.6)
    r_cloud = ask('privacy Cloud', 0.3)
    b_mig = ask('migration bandwidth', 100e6)
    j_mig = ask('migration power', 1.0)

    repeats = max(1, ask('\nrepeats', 1, int))

    overrides = {
        'num_episodes': episodes, 'max_steps_per_episode': steps,
        'batch_size': batch, 'replay_capacity': buf,
        'lr': lr, 'gamma': gamma, 'tau': tau, 'train_freq': train_freq,
        'lam': lam, 'omega': omega, 'reward_scale': reward_scl,
        'r_mec': r_mec, 'r_cloud': r_cloud,
        'b_mig': b_mig, 'j_mig': j_mig,
    }

    if input('\n  Start? [Y/n]: ').strip().lower() == 'n':
        return None
    return (name, eps_s, eps_e, eps_d, overrides, repeats)


def custom_fl_config():
    """Interactive wizard for a custom FL run."""
    print('\n  --- Custom FL Setup ---')
    name = input('  Run name [fl_custom]: ').strip() or 'fl_custom'

    print('\n  Federation:')
    n_clients = ask('n_clients', 5, int)
    n_rounds = ask('n_rounds', 50, int)
    local_eps = ask('local_episodes per round', 20, int)
    client_frac = ask('client_fraction (1.0 = all selected each round)', 1.0)
    hetero = ask('heterogeneous [1=yes/0=no]', 1, int) == 1

    print('\n  Exploration:')
    eps_s = ask('eps_start', 1.0)
    eps_e = ask('eps_end', 0.05)
    eps_d = ask('eps_decay', 15000, int)

    print('\n  Training:')
    batch = ask('batch_size', BATCH_SIZE, int)
    buf = ask('replay_capacity', REPLAY_CAPACITY, int)
    lr = ask('learning_rate', LR)
    gamma = ask('gamma', GAMMA)
    tau = ask('tau', TAU)
    train_freq = ask('train_freq', TRAIN_FREQ, int)

    overrides = {
        'batch_size': batch, 'replay_capacity': buf,
        'lr': lr, 'gamma': gamma, 'tau': tau, 'train_freq': train_freq,
    }

    if input('\n  Start? [Y/n]: ').strip().lower() == 'n':
        return None
    return {
        'name': name,
        'n_clients': n_clients, 'n_rounds': n_rounds,
        'local_episodes': local_eps, 'client_fraction': client_frac,
        'heterogeneous': hetero,
        'eps_start': eps_s, 'eps_end': eps_e, 'eps_decay': eps_d,
        'overrides': overrides,
    }


# ================================================================== #
# SECTION 9 — Single-agent training (baseline)                        #
# ================================================================== #
def run_training(config_name, eps_start, eps_end, eps_decay, out_dir,
                 overrides=None, seed=0):
    """Train one DQN agent. Saves config, metrics, dashboard, TB logs, best
    model, and periodic checkpoints."""
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    writer = SummaryWriter(log_dir=out_dir)
    run_start_time = time.time()

    ov = overrides or {}
    batch_size = ov.get('batch_size', BATCH_SIZE)
    gamma = ov.get('gamma', GAMMA)
    tau = ov.get('tau', TAU)
    lr = ov.get('lr', LR)
    num_eps = ov.get('num_episodes', NUM_EPISODES)
    buf_cap = ov.get('replay_capacity', REPLAY_CAPACITY)
    train_freq = ov.get('train_freq', TRAIN_FREQ)

    yaml_cfg = {
        'meta': {
            'mode': 'single_agent',
            'config_name': config_name,
            'out_dir': out_dir,
            'seed': seed,
            'device': str(device),
        },
        'exploration': {
            'eps_start': eps_start, 'eps_end': eps_end, 'eps_decay': eps_decay,
        },
        'training': {
            'num_episodes': num_eps, 'batch_size': batch_size,
            'replay_capacity': buf_cap, 'lr': lr, 'gamma': gamma, 'tau': tau,
            'train_freq': train_freq,
            'max_steps_per_episode': ov.get('max_steps_per_episode', 50),
        },
        'environment': {
            'lam': ov.get('lam', 10.0),
            'omega': ov.get('omega', 0.7),
            'reward_scale': ov.get('reward_scale', 1000.0),
            'r_local': 0.0,
            'r_cloud': ov.get('r_cloud', 0.3),
            'r_mec': ov.get('r_mec', 0.6),
            'b_mig': ov.get('b_mig', 100e6),
            'j_mig': ov.get('j_mig', 1.0),
        },
    }
    save_run_yaml(out_dir, yaml_cfg)

    env = MECOffloadingEnv()
    eval_env = MECOffloadingEnv()
    for key in ['max_steps_per_episode', 'lam', 'omega', 'reward_scale',
                'r_mec', 'r_cloud', 'b_mig', 'j_mig']:
        if key in ov:
            setattr(env, key, ov[key])
            setattr(eval_env, key, ov[key])

    policy_net = DQN(env.observation_space_n, env.action_space_n).to(device)
    target_net = DQN(env.observation_space_n, env.action_space_n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    memory = ReplayMemory(buf_cap)
    criterion = torch_nn.SmoothL1Loss()

    steps_done = [0]
    global_step = [0]
    optimize_calls = [0]

    def select_action(state):
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done[0] / eps_decay)
        steps_done[0] += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1, 1)
        return torch.tensor([[random.randrange(env.action_space_n)]],
                            device=device, dtype=torch.long)

    def optimize_model():
        if len(memory) < batch_size:
            return None, None
        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state),
                                       device=device, dtype=torch.bool)
        non_final_next = [s for s in batch.next_state if s is not None]
        if not non_final_next:
            return None, None
        non_final_next_states = torch.cat(non_final_next)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        sav = policy_net(state_batch).gather(1, action_batch)
        next_vals = torch.zeros(batch_size, device=device)
        with torch.no_grad():
            next_vals[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        expected = (next_vals * gamma) + reward_batch

        loss = criterion(sav, expected.unsqueeze(1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 100)
        optimizer.step()
        return loss.item(), grad_norm.item()

    def soft_update():
        with torch.no_grad():
            for tp, p in zip(target_net.parameters(), policy_net.parameters()):
                tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

    episode_rewards, episode_energy, episode_delay = [], [], []
    episode_violation_rate, episode_action_dist, episode_eps = [], [], []
    best_eval_reward = -float('inf')
    eval_history = []

    print(f"\n=== Training [{config_name}] seed={seed} | ε: {eps_start}→{eps_end} "
          f"decay={eps_decay} | train_freq={train_freq} ===")
    pbar = tqdm(range(num_eps), desc=f"{config_name}", ncols=130, colour='green',
                bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}')

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

            if global_step[0] % train_freq == 0:
                loss_val, gn = optimize_model()
                if loss_val is not None:
                    optimize_calls[0] += 1
                    soft_update()
                    if optimize_calls[0] % TB_STEP_LOG_FREQ == 0:
                        writer.add_scalar('train/loss', loss_val, global_step[0])
                        writer.add_scalar('train/grad_norm', gn, global_step[0])
            global_step[0] += 1

            if done:
                episode_rewards.append(total_reward)
                episode_energy.append(np.mean(ep_energies))
                episode_delay.append(np.mean(ep_delays))
                episode_violation_rate.append(np.mean(ep_violations))
                total_acts = sum(ep_locations)
                episode_action_dist.append([c / total_acts for c in ep_locations])
                cur_eps = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done[0] / eps_decay)
                episode_eps.append(cur_eps)

                writer.add_scalar('reward/episode', total_reward, i_episode)
                writer.add_scalar('reward/mean_50', np.mean(episode_rewards[-50:]), i_episode)
                writer.add_scalar('env/violation_rate', np.mean(ep_violations), i_episode)
                writer.add_scalar('env/energy_mJ', np.mean(ep_energies) * 1000, i_episode)
                writer.add_scalar('env/delay_s', np.mean(ep_delays), i_episode)
                writer.add_scalar('train/eps', cur_eps, i_episode)
                writer.add_scalar('action_dist/local', ep_locations[0] / total_acts, i_episode)
                writer.add_scalar('action_dist/cloud', ep_locations[1] / total_acts, i_episode)
                writer.add_scalar('action_dist/mec',   ep_locations[2] / total_acts, i_episode)

                window = min(50, len(episode_rewards))
                pbar.set_postfix_str(
                    f"{color_stat('R', np.mean(episode_rewards[-window:]), -0.05, -0.15, '.3f')} "
                    f"{color_stat('viol%', np.mean(episode_violation_rate[-window:])*100, 10, 40, '.0f')} "
                    f"{color_stat('E(mJ)', np.mean(episode_energy[-window:])*1000, 1, 5, '.2f')} "
                    f"eps={cur_eps:.2f}"
                )
                break

        if (i_episode + 1) % EVAL_EVERY == 0:
            ev = run_eval(policy_net, eval_env, EVAL_EPISODES)
            ev['episode'] = i_episode
            eval_history.append(ev)
            writer.add_scalar('eval/reward', ev['reward'], i_episode)
            writer.add_scalar('eval/violation_rate', ev['violation_rate'], i_episode)
            writer.add_scalar('eval/energy_mJ', ev['energy'] * 1000, i_episode)
            if ev['reward'] > best_eval_reward:
                best_eval_reward = ev['reward']
                torch.save({
                    'policy_net': policy_net.state_dict(),
                    'episode': i_episode,
                    'eval': ev,
                }, os.path.join(out_dir, 'best_model.pt'))

        if (i_episode + 1) % CHECKPOINT_EVERY == 0:
            torch.save({
                'policy_net': policy_net.state_dict(),
                'target_net': target_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'steps_done': steps_done[0],
                'episode': i_episode,
            }, os.path.join(out_dir, 'checkpoint.pt'))

    writer.close()

    np.savez(os.path.join(out_dir, 'metrics.npz'),
             rewards=np.array(episode_rewards),
             energy=np.array(episode_energy),
             delay=np.array(episode_delay),
             violation_rate=np.array(episode_violation_rate),
             action_dist=np.array(episode_action_dist),
             eps=np.array(episode_eps))

    # ---- Dashboard ------------------------------------------------
    episodes_x = np.arange(len(episode_rewards))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f'Dashboard — {config_name} (seed={seed})', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    mean_r = smooth(episode_rewards, 20)
    std_r = rolling_std(episode_rewards, 20)
    ax.fill_between(episodes_x, mean_r - std_r, mean_r + std_r, alpha=0.25, color='steelblue')
    ax.plot(episodes_x, mean_r, color='steelblue', linewidth=2, label='train (20-ep mean)')
    if eval_history:
        ex = [e['episode'] for e in eval_history]
        er = [e['reward'] for e in eval_history]
        ax.plot(ex, er, 'o-', color='darkorange', linewidth=2, markersize=5, label='eval (greedy)')
    ax.set_title('Episode Reward'); ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(episodes_x, smooth(episode_violation_rate, 20) * 100, color='crimson', linewidth=2)
    ax.set_title('Deadline Violation Rate'); ax.set_xlabel('Episode'); ax.set_ylabel('%')
    ax.set_ylim(0, 100); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    dist = np.array(episode_action_dist)
    dist_smooth = np.stack([smooth(dist[:, i], 20) for i in range(3)], axis=1)
    ax.stackplot(episodes_x, dist_smooth.T * 100,
                 labels=['Local', 'Cloud', 'MEC'],
                 colors=['#888888', '#4a90d9', '#50c878'], alpha=0.85)
    ax.set_title('Action Location Distribution'); ax.set_xlabel('Episode'); ax.set_ylabel('%')
    ax.set_ylim(0, 100); ax.legend(loc='upper right'); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(episodes_x, smooth(np.array(episode_energy) * 1000, 20), color='darkorange', linewidth=2)
    ax.set_title('Average Energy per Step'); ax.set_xlabel('Episode'); ax.set_ylabel('mJ')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(episodes_x, smooth(episode_delay, 20), color='purple', linewidth=2)
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='avg deadline')
    ax.set_title('Average Delay per Step'); ax.set_xlabel('Episode'); ax.set_ylabel('s')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(episodes_x, episode_eps, color='teal', linewidth=2)
    ax.set_title('Exploration (ε)'); ax.set_xlabel('Episode'); ax.set_ylabel('ε')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    dash_path = os.path.join(out_dir, 'dashboard.png')
    plt.savefig(dash_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    elapsed = time.time() - run_start_time
    final_r = float(np.mean(episode_rewards[-50:]))
    final_v = float(np.mean(episode_violation_rate[-50:]))
    with open(os.path.join(out_dir, 'config.yaml'), 'a') as f:
        f.write('\nresults:\n')
        f.write(f'  elapsed_sec: {elapsed:.2f}\n')
        f.write(f'  elapsed_human: "{fmt_eta(elapsed)}"\n')
        f.write(f'  final_reward_mean_last50: {final_r:.5f}\n')
        f.write(f'  final_violation_rate_last50: {final_v:.5f}\n')
        f.write(f'  best_eval_reward: {best_eval_reward:.5f}\n')
        f.write(f'  total_env_steps: {global_step[0]}\n')
        f.write(f'  total_optimize_calls: {optimize_calls[0]}\n')
    print(f'[{config_name}] done in {fmt_eta(elapsed)} → {out_dir}')

    return {
        'name': config_name, 'seed': seed,
        'rewards': episode_rewards, 'violation_rate': episode_violation_rate,
        'energy': episode_energy, 'eps': episode_eps,
        'elapsed': elapsed, 'out_dir': out_dir,
        'final_reward': final_r, 'final_violation': final_v,
        'best_eval_reward': best_eval_reward,
    }


# ================================================================== #
# SECTION 10 — Federated training                                     #
# ================================================================== #
def run_federated_training(name, n_clients, n_rounds, local_episodes,
                           eps_start, eps_end, eps_decay, out_dir,
                           overrides=None, base_seed=0,
                           heterogeneous=True, client_fraction=1.0):
    """FedAvg + DQN across n_clients devices. Each round:
       1. Broadcast global weights to selected clients
       2. Each client trains locally for `local_episodes` episodes
       3. Server performs weighted FedAvg on collected policy-net weights
       4. Aggregated model evaluated on a held-out eval env
    """
    os.makedirs(out_dir, exist_ok=True)
    set_seed(base_seed)
    writer = SummaryWriter(log_dir=out_dir)
    run_start = time.time()

    ov = overrides or {}
    obs_size = 5
    action_size = 9  # 3 locations × 3 CPU levels

    yaml_cfg = {
        'meta': {
            'mode': 'federated',
            'config_name': name,
            'out_dir': out_dir,
            'seed': base_seed,
            'device': str(device),
        },
        'federation': {
            'n_clients': n_clients,
            'n_rounds': n_rounds,
            'local_episodes_per_round': local_episodes,
            'client_fraction': client_fraction,
            'heterogeneous': heterogeneous,
            'aggregation': 'FedAvg (weighted by transitions)',
        },
        'exploration': {
            'eps_start': eps_start, 'eps_end': eps_end, 'eps_decay': eps_decay,
        },
        'training': {
            'batch_size':      ov.get('batch_size', BATCH_SIZE),
            'lr':              ov.get('lr', LR),
            'gamma':           ov.get('gamma', GAMMA),
            'tau':             ov.get('tau', TAU),
            'train_freq':      ov.get('train_freq', TRAIN_FREQ),
            'replay_capacity': ov.get('replay_capacity', REPLAY_CAPACITY),
        },
    }
    save_run_yaml(out_dir, yaml_cfg)

    # ---- Build clients --------------------------------------------
    rng = np.random.default_rng(base_seed + 10_000)
    clients = []
    profiles = []
    for cid in range(n_clients):
        if heterogeneous:
            profile = {
                'task_scale': float(rng.uniform(0.6, 1.4)),
                'r_mec':      float(rng.uniform(0.3, 0.9)),
                'r_cloud':    float(rng.uniform(0.1, 0.4)),
                'max_speed':  float(rng.uniform(2.0, 8.0)),
            }
        else:
            profile = {'task_scale': 1.0, 'r_mec': 0.6, 'r_cloud': 0.3, 'max_speed': 5.0}
        profiles.append(profile)

        env = HeterogeneousMECEnv(task_scale=profile['task_scale'])
        env.r_mec     = profile['r_mec']
        env.r_cloud   = profile['r_cloud']
        env.max_speed = profile['max_speed']
        for key in ['max_steps_per_episode', 'lam', 'omega', 'reward_scale', 'b_mig', 'j_mig']:
            if key in ov:
                setattr(env, key, ov[key])

        clients.append(FederatedClient(cid, obs_size, action_size, env, overrides=ov))

    # Global model — what clients receive at the start of each round
    global_net = DQN(obs_size, action_size).to(device)

    # Sync all clients to the initial random weights
    init_weights = {k: v.detach().cpu().clone() for k, v in global_net.state_dict().items()}
    for c in clients:
        c.set_weights(init_weights)

    # Held-out eval env — matches no single client, tests generalization
    eval_env = MECOffloadingEnv()
    for key in ['max_steps_per_episode', 'lam', 'omega', 'reward_scale']:
        if key in ov:
            setattr(eval_env, key, ov[key])

    # ---- Console banner -------------------------------------------
    print(f"\n=== Federated Training [{name}] ===")
    print(f"    Clients: {n_clients} | Rounds: {n_rounds} | Local episodes/round: {local_episodes}")
    print(f"    Heterogeneous: {heterogeneous} | Client fraction: {client_fraction}")
    print(f"    Aggregation: weighted FedAvg | ε: {eps_start}→{eps_end} decay={eps_decay}")
    if heterogeneous:
        print("    Client profiles:")
        for cid, p in enumerate(profiles):
            print(f"      C{cid}: task×{p['task_scale']:.2f}  "
                  f"speed={p['max_speed']:.1f}  r_mec={p['r_mec']:.2f}  r_cloud={p['r_cloud']:.2f}")

    # ---- Training loop --------------------------------------------
    round_metrics = []
    eval_history = []
    best_eval_reward = -float('inf')
    n_selected = max(1, int(round(n_clients * client_fraction)))

    pbar = tqdm(range(n_rounds), desc=f"{name}", ncols=130, colour='cyan',
                bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}')

    for round_idx in pbar:
        # 1. Select participating clients for this round
        if client_fraction < 1.0:
            selected = random.sample(clients, n_selected)
        else:
            selected = clients

        # 2. Broadcast current global weights
        current_weights = {k: v.detach().cpu().clone() for k, v in global_net.state_dict().items()}
        for c in selected:
            c.set_weights(current_weights)

        # 3. Local training on each selected client
        client_weights = []
        client_samples = []
        client_mets = []
        for c in selected:
            m = c.local_train(local_episodes, eps_start, eps_end, eps_decay)
            client_weights.append(c.get_weights())
            client_samples.append(m['transitions'])
            client_mets.append(m)

        # 4. Aggregate via weighted FedAvg
        aggregated = federated_average(client_weights, sample_counts=client_samples)
        global_net.load_state_dict(aggregated)

        # 5. Round-level TB logging
        mean_r = float(np.mean([m['reward']         for m in client_mets]))
        mean_v = float(np.mean([m['violation_rate'] for m in client_mets]))
        mean_e = float(np.mean([m['energy']         for m in client_mets]))
        writer.add_scalar('fl/round_reward_mean', mean_r, round_idx)
        writer.add_scalar('fl/round_violation_mean', mean_v, round_idx)
        writer.add_scalar('fl/round_energy_mJ_mean', mean_e * 1000, round_idx)
        writer.add_scalar('fl/n_participants', len(selected), round_idx)
        for c, m in zip(selected, client_mets):
            writer.add_scalar(f'client/{c.id}/reward', m['reward'], round_idx)
            writer.add_scalar(f'client/{c.id}/violation', m['violation_rate'], round_idx)

        round_metrics.append({
            'reward': mean_r, 'violation': mean_v, 'energy': mean_e,
            'participants': len(selected),
        })

        # 6. Evaluate the aggregated global model (greedy, held-out env)
        ev = run_eval(global_net, eval_env, EVAL_EPISODES)
        ev['round'] = round_idx
        eval_history.append(ev)
        writer.add_scalar('eval/global_reward', ev['reward'], round_idx)
        writer.add_scalar('eval/global_violation', ev['violation_rate'], round_idx)
        writer.add_scalar('eval/global_energy_mJ', ev['energy'] * 1000, round_idx)
        writer.add_scalar('eval/global_local_pct', ev['loc_dist'][0] * 100, round_idx)
        writer.add_scalar('eval/global_cloud_pct', ev['loc_dist'][1] * 100, round_idx)
        writer.add_scalar('eval/global_mec_pct',   ev['loc_dist'][2] * 100, round_idx)

        if ev['reward'] > best_eval_reward:
            best_eval_reward = ev['reward']
            torch.save({
                'global_net': global_net.state_dict(),
                'round': round_idx,
                'eval': ev,
            }, os.path.join(out_dir, 'best_global_model.pt'))

        pbar.set_postfix_str(
            f"{color_stat('R_eval', ev['reward'], -0.05, -0.15, '.3f')} "
            f"{color_stat('viol%', ev['violation_rate']*100, 10, 40, '.0f')} "
            f"best={best_eval_reward:.3f}"
        )

    writer.close()

    # ---- Persist raw metrics --------------------------------------
    np.savez(os.path.join(out_dir, 'fl_metrics.npz'),
             round_rewards=np.array([m['reward']    for m in round_metrics]),
             round_violations=np.array([m['violation'] for m in round_metrics]),
             round_energies=np.array([m['energy']   for m in round_metrics]),
             eval_rewards=np.array([e['reward']         for e in eval_history]),
             eval_violations=np.array([e['violation_rate'] for e in eval_history]),
             eval_energies=np.array([e['energy']        for e in eval_history]))

    _make_fl_dashboard(clients, round_metrics, eval_history, profiles,
                       out_dir, heterogeneous, name)

    elapsed = time.time() - run_start

    with open(os.path.join(out_dir, 'config.yaml'), 'a') as f:
        f.write('\nresults:\n')
        f.write(f'  elapsed_sec: {elapsed:.2f}\n')
        f.write(f'  elapsed_human: "{fmt_eta(elapsed)}"\n')
        f.write(f'  final_round_train_reward: {round_metrics[-1]["reward"]:.5f}\n')
        f.write(f'  final_round_violation: {round_metrics[-1]["violation"]:.5f}\n')
        f.write(f'  final_eval_reward: {eval_history[-1]["reward"]:.5f}\n')
        f.write(f'  best_eval_reward: {best_eval_reward:.5f}\n')

    print(f"\n[FL {name}] done in {fmt_eta(elapsed)} → {out_dir}")
    print(f"     Best eval reward: {best_eval_reward:.4f}")
    print(f"     Final eval reward: {eval_history[-1]['reward']:.4f}")
    print(f"     Final eval violation: {eval_history[-1]['violation_rate']*100:.1f}%")

    return {
        'name': name, 'seed': base_seed, 'out_dir': out_dir,
        'elapsed': elapsed, 'round_metrics': round_metrics,
        'eval_history': eval_history, 'best_eval_reward': best_eval_reward,
        'global_net': global_net,
    }


def _make_fl_dashboard(clients, round_metrics, eval_history, profiles,
                       out_dir, heterogeneous, name):
    """Six-panel FL dashboard: global eval curves, per-client trajectories,
    heterogeneity visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f'Federated Learning — {name}', fontsize=14, fontweight='bold')
    rounds = np.arange(len(round_metrics))

    ax = axes[0, 0]
    eval_r = [e['reward'] for e in eval_history]
    ax.plot(rounds, eval_r, 'o-', color='darkorange', linewidth=2, markersize=4)
    ax.set_title('Global Model Eval Reward (greedy)')
    ax.set_xlabel('FL round'); ax.set_ylabel('Reward'); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    eval_v = [e['violation_rate'] * 100 for e in eval_history]
    ax.plot(rounds, eval_v, 'o-', color='crimson', linewidth=2, markersize=4)
    ax.set_title('Global Eval Violation Rate')
    ax.set_xlabel('FL round'); ax.set_ylabel('%')
    ax.set_ylim(0, 100); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    if eval_history:
        dist = eval_history[-1]['loc_dist']
        bars = ax.bar(['Local', 'Cloud', 'MEC'], [d * 100 for d in dist],
                      color=['#888888', '#4a90d9', '#50c878'])
        for b, d in zip(bars, dist):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    f'{d*100:.0f}%', ha='center', fontsize=10)
    ax.set_title('Final Global Action Distribution')
    ax.set_ylabel('%'); ax.set_ylim(0, 100); ax.grid(alpha=0.3, axis='y')

    ax = axes[1, 0]
    for c in clients:
        if len(c.episode_rewards) > 0:
            window = min(20, max(1, len(c.episode_rewards) // 4))
            r = smooth(c.episode_rewards, window)
            ax.plot(r, alpha=0.7, linewidth=1.2, label=f'C{c.id}')
    ax.set_title('Per-Client Training Reward')
    ax.set_xlabel('Local episode (cumulative)'); ax.set_ylabel('Reward')
    if len(clients) <= 8:
        ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    mean_r = [m['reward'] for m in round_metrics]
    ax.plot(rounds, mean_r, '-', color='steelblue', linewidth=2, label='Round mean (train)')
    ax.plot(rounds, eval_r, '--', color='darkorange', linewidth=2, label='Global eval')
    ax.set_title('Training vs Eval Reward')
    ax.set_xlabel('FL round'); ax.set_ylabel('Reward')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    if heterogeneous and profiles:
        cids = list(range(len(profiles)))
        ts = [p['task_scale'] for p in profiles]
        ax.bar(cids, ts, color='tab:purple', alpha=0.7)
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.4, label='baseline')
        ax.set_title('Client Task-Size Scale (non-IID)')
        ax.set_xlabel('Client ID'); ax.set_ylabel('task_scale')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Homogeneous\n(all clients identical)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fl_dashboard.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


# ================================================================== #
# SECTION 11 — Aggregate comparison / summary                         #
# ================================================================== #
def aggregate_comparison_plot(results, base_dir):
    """Paper-quality comparison plot: one line per config with shaded ±std band."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[r['name']].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Config Comparison (mean ± std across seeds)', fontsize=14, fontweight='bold')
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    for ci, (name, rs) in enumerate(groups.items()):
        col = colors[ci % len(colors)]
        min_len = min(len(r['rewards']) for r in rs)
        rew = np.array([smooth(r['rewards'][:min_len], 20) for r in rs])
        vio = np.array([smooth(np.array(r['violation_rate'][:min_len]) * 100, 20) for r in rs])
        eps_ = np.array([r['eps'][:min_len] for r in rs])
        x = np.arange(min_len)

        for ax, data, title, ylab in [
            (axes[0], rew, 'Reward', 'R'),
            (axes[1], vio, 'Violation Rate', '%'),
            (axes[2], eps_, 'ε Schedule', 'ε'),
        ]:
            m, s = data.mean(0), data.std(0)
            ax.plot(x, m, color=col, linewidth=2, label=f'{name} (n={len(rs)})')
            ax.fill_between(x, m - s, m + s, color=col, alpha=0.2)
            ax.set_title(title); ax.set_xlabel('Episode'); ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)

    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(base_dir, f'comparison_{int(time.time())}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def write_summary_md(results, base_dir):
    """Post-batch markdown summary table."""
    path = os.path.join(base_dir, 'summary.md')
    with open(path, 'w') as f:
        f.write('# Run Summary\n\n')
        f.write('| config | seed | final_R | viol% | E(mJ) | best_eval | time |\n')
        f.write('|---|---|---|---|---|---|---|\n')
        for r in results:
            e_mj = float(np.mean(r['energy'][-50:])) * 1000
            f.write(f"| {r['name']} | {r['seed']} | {r['final_reward']:.4f} | "
                    f"{r['final_violation']*100:.1f} | {e_mj:.2f} | "
                    f"{r['best_eval_reward']:.4f} | {fmt_eta(r['elapsed'])} |\n")
    return path


# ================================================================== #
# SECTION 12 — Configs + main loop                                    #
# ================================================================== #
CONFIGS = [
    # name,         eps_start, eps_end, eps_decay
    ('fast_decay',  1.0, 0.05, 2000),    # baseline — exploration dies fast
    ('slow_decay',  1.0, 0.05, 15000),   # smooth long decay
    ('persistent',  1.0, 0.15, 15000),   # slow decay + high ε floor
]

# Preset FL configs. fl_quick is for smoke-testing; fl_standard uses the same
# slow_decay ε schedule as the single-agent config for fair comparison.
FL_PRESETS = {
    'fl_quick':    dict(n_clients=3, n_rounds=10, local_episodes=5,
                        eps_start=1.0, eps_end=0.1, eps_decay=2000,
                        heterogeneous=True, client_fraction=1.0),
    'fl_standard': dict(n_clients=5, n_rounds=50, local_episodes=20,
                        eps_start=1.0, eps_end=0.05, eps_decay=15000,
                        heterogeneous=True, client_fraction=1.0),
}


if __name__ == '__main__':
    base_dir = 'runs'
    os.makedirs(base_dir, exist_ok=True)
    print(f'Device: {device}')

    while True:
        print_menu()
        choice = input('  Select » ').strip().lower()

        # ---- Utilities ----
        if choice == '9' or choice in ('q', 'quit', 'exit'):
            print('Bye.')
            break
        if choice == '6':
            math_sanity_check()
            continue
        if choice == '7':
            port = ask('port', 6006, int)
            launch_tensorboard(base_dir, port=port)
            continue
        if choice == '8':
            delete_all_runs(base_dir)
            continue

        # ---- Single-agent DQN modes ----
        if choice in ('1', '2', '3', '4', '5'):
            if choice == '1':
                reps = ask('repeats per config', 1, int)
                selected = [(n, s, e, d, None, reps) for (n, s, e, d) in CONFIGS]
            elif choice in ('2', '3', '4'):
                n, s, e, d = CONFIGS[int(choice) - 2]
                reps = ask('repeats', 1, int)
                selected = [(n, s, e, d, None, reps)]
            else:  # choice == '5'
                cfg = custom_run_config()
                if cfg is None:
                    print('  Cancelled.')
                    continue
                selected = [cfg]

            base_seed = ask('base seed', 0, int)

            run_queue = []
            for name, eps_s, eps_e, eps_d, ov, reps in selected:
                for i in range(reps):
                    run_queue.append((name, eps_s, eps_e, eps_d, ov, base_seed + i))

            total = len(run_queue)
            print(f'\n  → Queued {total} run(s).')

            results = []
            batch_start = time.time()
            per_run_times = []

            for idx, (name, eps_s, eps_e, eps_d, ov, seed) in enumerate(run_queue, 1):
                out_dir = get_unique_run_dir(base_dir, name)
                rel = os.path.relpath(out_dir, base_dir)
                eta_str = (fmt_eta((sum(per_run_times) / len(per_run_times)) * (total - idx + 1))
                           if per_run_times else 'estimating...')
                print(f'\n╭─ Run {idx}/{total}: {rel} (seed={seed})')
                print(f'│  Elapsed: {fmt_eta(time.time() - batch_start)}   ETA: {eta_str}')
                print(f'╰─')

                t0 = time.time()
                res = run_training(name, eps_s, eps_e, eps_d, out_dir,
                                   overrides=ov, seed=seed)
                per_run_times.append(time.time() - t0)
                results.append(res)
                update_latest_symlink(base_dir, out_dir)

            print(f'\n  ✓ All {total} run(s) finished in {fmt_eta(time.time() - batch_start)}')

            if len(results) >= 2:
                print('Generating aggregate comparison + summary...')
                cp = aggregate_comparison_plot(results, base_dir)
                sp = write_summary_md(results, base_dir)
                print(f'  → {cp}')
                print(f'  → {sp}')

            print('\nTip: menu [7] to launch TensorBoard.\n')
            continue

        # ---- Federated Learning modes ----
        if choice in ('f1', 'f2', 'f3'):
            if choice == 'f1':
                cfg = dict(FL_PRESETS['fl_quick'])
                cfg['name'] = 'fl_quick'
                cfg['overrides'] = None
            elif choice == 'f2':
                cfg = dict(FL_PRESETS['fl_standard'])
                cfg['name'] = 'fl_standard'
                cfg['overrides'] = None
            else:
                cfg = custom_fl_config()
                if cfg is None:
                    print('  Cancelled.')
                    continue

            reps = ask('repeats', 1, int)
            base_seed = ask('base seed', 0, int)

            results = []
            batch_start = time.time()
            for i in range(reps):
                seed = base_seed + i
                out_dir = get_unique_run_dir(base_dir, cfg['name'])
                rel = os.path.relpath(out_dir, base_dir)
                print(f'\n╭─ FL Run {i+1}/{reps}: {rel} (seed={seed})')
                print(f'│  Elapsed: {fmt_eta(time.time() - batch_start)}')
                print(f'╰─')

                res = run_federated_training(
                    name=cfg['name'],
                    n_clients=cfg['n_clients'],
                    n_rounds=cfg['n_rounds'],
                    local_episodes=cfg['local_episodes'],
                    eps_start=cfg['eps_start'],
                    eps_end=cfg['eps_end'],
                    eps_decay=cfg['eps_decay'],
                    out_dir=out_dir,
                    overrides=cfg.get('overrides'),
                    base_seed=seed,
                    heterogeneous=cfg.get('heterogeneous', True),
                    client_fraction=cfg.get('client_fraction', 1.0),
                )
                results.append(res)
                update_latest_symlink(base_dir, out_dir)

            print(f'\n  ✓ All {reps} FL run(s) finished in {fmt_eta(time.time() - batch_start)}')
            print('\nTip: menu [7] to launch TensorBoard.\n')
            continue

        print('  ! Invalid choice.')
