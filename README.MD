# DQN for Task Offloading in MEC — Energy, Delay & Privacy

A Deep Q-Network that learns an energy-efficient, privacy-aware task offloading
policy in a Mobile Edge Computing (MEC) system. At each time slot the agent
decides **where** to run a task (local device / edge server / cloud) and
**how much CPU** to allocate, trading off energy, deadline violations, and
privacy leakage.

> Phase 2 implementation — includes user mobility, dynamic uplink rate,
> migration cost, and a simplified privacy penalty on top of the base
> Phase 1 DQN pipeline.

**Made by Yousif Iskander.**

---

## Table of Contents

1. [What this project does](#what-this-project-does)
2. [The model](#the-model)
3. [Installation](#installation)
4. [Running it](#running-it)
5. [Menu options](#menu-options)
6. [Output structure](#output-structure)
7. [TensorBoard](#tensorboard)
8. [Configuration reference](#configuration-reference)
9. [Code map](#code-map)
10. [Tips & troubleshooting](#tips--troubleshooting)

---

## What this project does

The agent is placed in a simulated MEC environment. Every time step:

1. A new task arrives with random input size, CPU cycles, and deadline.
2. The agent picks one of **9 actions** = 3 execution locations × 3 CPU ratios:
   - `Local` / `Cloud` / `MEC (Edge)`
   - CPU ratio: `0.25`, `0.5`, `1.0`
3. The environment computes delay, energy, privacy penalty, and migration
   cost (if the agent switched between remote servers).
4. The agent gets a negative reward proportional to total cost.
5. The user's position drifts, which changes the wireless uplink rate to the
   MEC server, so the optimal action shifts over time.

Over many episodes, the DQN learns a policy that balances these costs.

---

## The model

### State (5-dim, normalized)

```
s_t = [ i(t)/1e7,     # input data size, bits
        c(t)/1e9,     # required CPU cycles
        d(t),         # deadline, seconds
        R_e(t)/1e8,   # current uplink bitrate (mobility-dependent)
        a_{t-1}/|A| ] # previous action (helps with migration awareness)
```

### Action space (9 discrete actions)

| idx | location | ρ (CPU ratio) |
|-----|----------|---------------|
| 0–2 | Local    | 0.25 / 0.5 / 1.0 |
| 3–5 | Cloud    | 0.25 / 0.5 / 1.0 |
| 6–8 | MEC      | 0.25 / 0.5 / 1.0 |

### Reward

```
r_t = -( E(t) + λ · max(0, T(t) − d(t)) + R(t) ) / reward_scale
```

where:
- `E(t)` = energy (J), including migration energy if the agent switched MEC↔Cloud
- `T(t)` = total delay (s), including transmission + compute + migration
- `R(t)` = privacy penalty: 0 local, 0.3 cloud, 0.6 MEC
- `λ` = deadline-violation weight (default 5.0)
- `reward_scale` = divisor to keep rewards well-conditioned (default 1000)

### Network

A 3-layer MLP: `5 → 256 → 256 → 9`, ReLU activations, SmoothL1Loss,
AdamW optimizer with target network soft-updates (Polyak averaging, τ=0.001).

---

## Installation

```bash
# Python 3.9+ recommended
pip install torch numpy matplotlib tqdm tensorboard
```

GPU is auto-detected. No GPU? It falls back to CPU without any changes.

---

## Running it

```bash
python train.py
```

You'll land in a menu. Pick an option and it'll prompt for any extra info
(repeats, seed, etc.). Everything gets saved under `runs/`.

---

## Menu options

```
╔════════════════════════════════════════════════════════════╗
║               MEC DQN Offloading — Trainer                 ║
╠════════════════════════════════════════════════════════════╣
║  [1] Run ALL configs (full comparison)                     ║
║  [2] fast_decay  only                                      ║
║  [3] slow_decay  only                                      ║
║  [4] persistent  only                                      ║
║  [5] Custom run                                            ║
║  [6] Math sanity check                                     ║
║  [7] Launch TensorBoard (view runs in browser)             ║
║  [8] Quit                                                  ║
╚════════════════════════════════════════════════════════════╝
```

**[1] Run all configs** — trains all three exploration schedules in sequence.
Great for producing the comparison plot.

**[2–4] Single config** — trains just one. Asks how many repeats you want
(each repeat uses a different seed, auto-incremented from your base seed).

**[5] Custom run** — interactive wizard that lets you override any
hyperparameter (episodes, batch size, LR, γ, τ, λ, reward scale, privacy
weights, migration cost…). Shows a confirmation summary before starting.

**[6] Math sanity check** — runs every action on a fixed synthetic state
and prints the delay/energy/reward breakdown. Use this to verify the physics
math by eye before a long training run.

**[7] TensorBoard** — spawns `tensorboard --logdir runs/` in the background,
opens your browser. Non-blocking — you can go right back to the menu.

**[8] Quit** — also accepts `q`, `quit`, `exit`.

### Exploration schedules

| Name | ε_start | ε_end | ε_decay | Behavior |
|------|---------|-------|---------|----------|
| `fast_decay` | 1.0 | 0.05 | 2000  | exploration dies fast — aggressive exploit |
| `slow_decay` | 1.0 | 0.05 | 15000 | smooth long decay — balanced |
| `persistent` | 1.0 | 0.15 | 15000 | slow decay with high floor — never stops exploring |

---

## Output structure

Every run writes to its own directory under `runs/`:

```
runs/
├── latest → fast_decay-Run(3)/       # symlink to most recent run
├── fast_decay/
│   ├── config.yaml         # every hyperparameter used, plus results summary
│   ├── metrics.npz         # raw numpy arrays (rewards, energy, delay, …)
│   ├── dashboard.png       # 6-panel training dashboard
│   ├── best_model.pt       # weights with highest eval reward
│   ├── checkpoint.pt       # resumable full state (every 200 eps)
│   └── events.out.tfevents.*  # TensorBoard logs
├── slow_decay/…
├── persistent/…
├── comparison_1712952841.png   # cross-config mean ± std plot
└── summary.md                  # markdown table of all runs
```

### `config.yaml` — what was used + what happened

Auto-generated at run start with every param, then appended with results
when training finishes:

```yaml
meta:
  config_name: "slow_decay"
  seed: 0
exploration:
  eps_start: 1
  eps_end: 0.05
  eps_decay: 15000
training:
  num_episodes: 1000
  batch_size: 128
  lr: 5.000e-05
  gamma: 0.99
  tau: 0.001
environment:
  lam: 5
  reward_scale: 1000
  r_mec: 0.6
  r_cloud: 0.3

results:
  elapsed_sec: 251.43
  elapsed_human: "0:04:11"
  final_reward_mean_last50: -0.07214
  final_violation_rate_last50: 0.08420
  best_eval_reward: -0.06811
```

### `dashboard.png` — 6-panel summary

| Panel | What it shows |
|-------|---------------|
| Reward | Training reward (20-ep mean ± σ band) + eval reward (orange dots) |
| Violation Rate | % of steps that missed their deadline |
| Action Distribution | Stacked area of Local/Cloud/MEC usage over time |
| Energy | Avg mJ per step |
| Delay | Avg seconds per step, vs. 1.0s deadline reference line |
| ε | Exploration schedule actually followed |

### `summary.md` — batch table

After a batch of runs finishes, you get:

```
| config      | seed | final_R  | viol% | E(mJ) | best_eval | time    |
|-------------|------|----------|-------|-------|-----------|---------|
| fast_decay  | 0    | -0.0821  | 11.8  | 1.42  | -0.0711   | 0:04:12 |
| slow_decay  | 0    | -0.0722  |  8.4  | 1.21  | -0.0681   | 0:04:15 |
| persistent  | 0    | -0.0755  |  9.1  | 1.28  | -0.0702   | 0:04:18 |
```

Paste-ready for a report.

---

## TensorBoard

Hit **[7]** in the menu, or run manually:

```bash
tensorboard --logdir runs/
```

**Scalars you get:**
- `reward/episode`, `reward/mean_50`
- `env/violation_rate`, `env/energy_mJ`, `env/delay_s`
- `eval/reward`, `eval/violation_rate`, `eval/energy_mJ` (greedy eval, every 50 eps)
- `train/loss`, `train/grad_norm`, `train/eps`
- `action_dist` (grouped: local / cloud / mec fractions on one chart)

TensorBoard makes multi-run comparison trivial — just toggle which runs
you want overlaid in the left sidebar.

---

## Configuration reference

All defaults live at the top of `train.py`:

```python
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.001
LR = 5e-5
NUM_EPISODES = 1000
REPLAY_CAPACITY = 100000
EVAL_EVERY = 50          # eval cadence (episodes)
EVAL_EPISODES = 10       # greedy episodes per eval
CHECKPOINT_EVERY = 200   # resumable snapshot cadence
```

Environment parameters live inside `MECOffloadingEnv.__init__` and map
directly to the project spec (`f_l`, `f_e`, `f_c`, `P_tx`, `T_bh`, `b_bh`,
`ζ`, `e_e`, `e_c`, `P_bh`, privacy weights, migration cost, etc.).

Any of these can be overridden per-run through the **Custom** menu without
touching the source.

---

## Code map

| Section | Purpose |
|---------|---------|
| `set_seed` | Reproducibility — seeds `random`, `numpy`, `torch` (incl. CUDA) |
| `MECOffloadingEnv` | The environment: physics, mobility, reward |
| `DQN` | 3-layer MLP, 256 hidden units |
| `ReplayMemory` | Vanilla uniform-sampling replay buffer (deque-backed) |
| `run_eval` | Greedy-policy evaluation — ε=0 for N episodes |
| `math_sanity_check` | Prints per-action delay/energy/reward on a fixed state |
| `run_training` | One full training run (core loop + logging + plots) |
| `aggregate_comparison_plot` | Groups repeats by config, plots mean±std |
| `write_summary_md` | Post-batch markdown summary table |
| `launch_tensorboard` | Non-blocking TensorBoard launcher with browser-open |
| `__main__` loop | Menu dispatch, run queue, batch ETA |

### Key design choices

- **Normalized state** — inputs to the network are all O(1)-scale to keep
  the optimizer well-conditioned; physics is un-normalized inside `step()`.
- **Reward scaling** — dividing by 1000 keeps reward around O(0.1), which
  plays nicely with SmoothL1 loss and the default Adam learning rate.
- **In-place soft update** — `tp.data.mul_(1-τ).add_(p.data, α=τ)` instead
  of the state-dict copy loop. Runs every step, so the speedup adds up.
- **Separate eval env** — so eval doesn't perturb the training env's
  mobility state or RNG stream.
- **Unique run dirs** — `name`, `name-Run(1)`, `name-Run(2)`… so repeats
  never clobber each other.

---

## Tips & troubleshooting

**"Rewards plateau at a weird value"** — check your reward scale. If
`-0.05` feels like a ceiling, try `reward_scale=100` so gradients are larger.

**"Violation rate never drops"** — λ might be too low. Try `λ=10` or `20`
in the Custom menu. Also check the math sanity output (menu [6]) to
confirm the deadline-breaching actions do get penalized more.

**"Action distribution collapses to all-local immediately"** — exploration
died too fast. Use `slow_decay` or `persistent`, or bump `EPS_DECAY` higher
in a custom run.

**"TensorBoard launch fails"** — install it (`pip install tensorboard`)
and confirm `tensorboard` is on your PATH. You can also run the command
manually in a second terminal.

**"Loss explodes / NaN"** — check `train/grad_norm` in TensorBoard. If it
spikes, lower the LR or reduce `tau` so the target net moves more slowly.

**"`runs/latest` symlink didn't update on Windows"** — symlinks need
admin on older Windows. Harmless — everything else still works; you just
have to cd into the timestamped dir manually.

**"I want to resume a crashed run"** — `checkpoint.pt` has everything you
need (policy net, target net, optimizer state, steps_done). Resume logic
isn't wired into the menu yet, but the data is there.

---

## Project references

Based on the project spec *"DQN for Task Offloading in MEC
(Energy–Delay–Privacy)"* — Phase 2, which adds user mobility, dynamic
uplink rate, migration cost, and the simplified privacy leakage model to
the base Phase 1 DQN pipeline.

---

*Made by Yousif Iskander.*