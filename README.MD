# MEC DQN Offloading — with Federated Learning

DQN-based task offloading for Mobile Edge Computing, extended with **Federated
Learning (FedAvg)** so multiple mobile devices can train collaboratively
without ever uploading their raw task history.

> CSCI4052U — Neural Computation
> Made by Yousif Iskander

---

## What it does

At each timestep a device gets a task `(i_t, c_t, d_t)` — input size, required
CPU cycles, deadline — and must pick **where to run it** and **how much CPU to
allocate**. The action space is:

- 3 execution locations: **Local**, **Cloud**, **MEC** (mobile edge server)
- 3 CPU ratios: **0.25**, **0.5**, **1.0**
- Total: **9 discrete actions**

The agent is rewarded for minimizing energy, meeting deadlines, and preserving
privacy:

```
R = −( E_t  +  λ · max(0, T_t − d_t)  +  privacy_cost ) / reward_scale
```

where `privacy_cost` is `0.0 / 0.3 / 0.6` for Local / Cloud / MEC respectively
(MEC reveals more location info than the cloud).

This project covers **two training modes**:

| Mode           | Devices | Privacy                                    | Use case                          |
|---------------:|:-------:|:-------------------------------------------|:----------------------------------|
| Single-agent   | 1       | Inference-time only (privacy term `R(t)`) | Baseline — one user, one policy   |
| **Federated**  | N       | **Inference + training-time**              | N users share a policy, data stays local |

The two privacy layers are complementary. `R(t)` stops the **deployed** agent
from leaking location via its choices. FL stops the **training process** from
ever collecting a central dataset of users' tasks.

---

## Installation

```bash
pip install torch numpy matplotlib tqdm tensorboard
```

Tested on CPU and CUDA. No GPU required — the default config finishes in a few
minutes on CPU for baseline runs.

## Quickstart

```bash
python3 trainer.py
```

You'll see the menu:

```
╔════════════════════════════════════════════════════════════╗
║                MEC DQN Offloading — Trainer                ║
╠════════════════════════════════════════════════════════════╣
║  Single-Agent DQN:                                         ║
║    [1]  Run ALL configs (full comparison)                  ║
║    [2]  fast_decay  only                                   ║
║    [3]  slow_decay  only                                   ║
║    [4]  persistent  only                                   ║
║    [5]  Custom run                                         ║
║                                                            ║
║  Federated Learning:                                       ║
║    [F1] Quick FL run (3 clients, 10 rounds)                ║
║    [F2] Standard FL run (5 clients, 50 rounds)             ║
║    [F3] Custom FL run                                      ║
║                                                            ║
║  Utilities:                                                ║
║    [6]  Math sanity check                                  ║
║    [7]  Launch TensorBoard                                 ║
║    [8]  Delete ALL runs                                    ║
║    [9]  Quit                                               ║
╚════════════════════════════════════════════════════════════╝
```

### Menu options

| Key  | What it does |
|------|--------------|
| `1`  | Run **all three** baseline configs (`fast_decay`, `slow_decay`, `persistent`) with chosen seed/repeats. Produces a side-by-side comparison plot. |
| `2–4`| Run a single named baseline config. |
| `5`  | Custom baseline — wizard prompts for every hyperparameter. |
| `F1` | **Quick FL run** — 3 clients × 10 rounds × 5 local episodes. Smoke test, ~1 min. |
| `F2` | **Standard FL run** — 5 clients × 50 rounds × 20 local episodes. The headline experiment. |
| `F3` | **Custom FL run** — wizard for `n_clients`, `n_rounds`, IID/non-IID, ε schedule, etc. |
| `6`  | Print reward/energy/delay for every action on a fixed state. Useful for checking reward scaling. |
| `7`  | Launch TensorBoard pointed at `runs/`. |
| `8`  | Nuke the `runs/` directory (asks for `DELETE` confirmation). |
| `9`  | Quit. |

---

## Project structure

```
trainer.py              ← Everything. Single entry point.
README.md               ← This file.
runs/                   ← Auto-created. All outputs live here.
  ├─ fast_decay/
  │   └─ Run(1)/
  │       ├─ config.yaml         ← Every hyperparameter for this run
  │       ├─ dashboard.png       ← 6-panel training dashboard
  │       ├─ metrics.npz         ← Raw episode-level metrics
  │       ├─ best_model.pt       ← Best eval-reward checkpoint
  │       ├─ checkpoint.pt       ← Resumable state every 200 eps
  │       └─ events.out.tfevents.*  (TensorBoard)
  ├─ slow_decay/Run(1)/
  ├─ persistent/Run(1)/
  ├─ fl_standard/
  │   └─ Run(1)/
  │       ├─ config.yaml            ← FL-mode config + per-client profiles
  │       ├─ fl_dashboard.png       ← 6-panel FL dashboard
  │       ├─ fl_metrics.npz         ← Round-level + eval metrics
  │       ├─ best_global_model.pt   ← Best aggregated global model
  │       └─ events.out.tfevents.*
  ├─ comparison_<ts>.png     ← Cross-config comparison (baseline menu [1])
  ├─ summary.md              ← Markdown results table
  └─ latest → <most recent run>   (symlink)
```

### Output files explained

- **`config.yaml`** — complete hyperparameter dump + final results. Everything
  needed to reproduce or write up a run.
- **`dashboard.png` / `fl_dashboard.png`** — training curves, violation rate,
  action distribution, energy, delay, ε schedule (baseline) or per-client
  curves + global eval + heterogeneity (FL).
- **`metrics.npz`** — raw arrays (`rewards`, `energy`, `delay`,
  `violation_rate`, `action_dist`, `eps`). Load with `np.load(...)`.
- **`best_model.pt`** — the DQN checkpoint with the best greedy-eval reward.
  For FL runs this is `best_global_model.pt` (the aggregated server model).
- **TensorBoard logs** — everything above, streamed live. Launch with menu [7].

---

## File structure inside `trainer.py`

Everything lives in **one file** organized into numbered sections:

| Section | Purpose |
|--------:|---------|
| 1 | Utilities (seed, colored tqdm, run-dir management, TB launcher) |
| 2 | `MECOffloadingEnv` + `HeterogeneousMECEnv` — the env + non-IID variant |
| 3 | `DQN` network, `Transition`, `ReplayMemory` |
| 4 | Default hyperparameters + smoothing helpers |
| 5 | `math_sanity_check()` |
| 6 | `run_eval()` — shared greedy evaluator |
| 7 | **`FederatedClient`** + **`federated_average()`** |
| 8 | Menu/input wizards (`custom_run_config`, `custom_fl_config`) |
| 9 | `run_training()` — single-agent loop |
| 10 | **`run_federated_training()`** + `_make_fl_dashboard()` |
| 11 | Cross-run comparison plots + markdown summary |
| 12 | `CONFIGS`, `FL_PRESETS`, main menu loop |

---

## How Federated Learning works here

### The setup

Each FL client is a complete DQN agent:

- Its own `HeterogeneousMECEnv` (possibly with different task sizes, mobility,
  privacy weights — this is the non-IID part)
- Its own `policy_net`, `target_net`, optimizer, replay buffer
- Its own ε-greedy counter that persists across rounds

A **single global model** lives on the server. Clients never see each other's
data — only the global model weights.

### One FL round

```
┌───────────────────────────────────────────────────────────────┐
│ Step 1: Server broadcasts global weights → selected clients   │
│ Step 2: Each client does `local_episodes` episodes of DQN     │
│         training on its own env. Replay buffer stays local.   │
│ Step 3: Clients upload only policy_net weights                │
│ Step 4: Server computes weighted FedAvg:                      │
│                                                                │
│                     w_global = Σ_k (n_k / N) · w_k            │
│                                                                │
│         where n_k = transitions collected by client k         │
│         and   N   = total transitions this round              │
│ Step 5: Global model evaluated on held-out env (greedy)       │
└───────────────────────────────────────────────────────────────┘
```

### Design decisions worth mentioning in the writeup

- **Weighted FedAvg** (McMahan et al.) — clients with more experience pull
  harder. Important here because under non-IID settings some clients collect
  more transitions per round than others.
- **Target net resynced to global weights** after each broadcast. Without
  this, the client has a stale target net from before aggregation, and the TD
  error spikes dramatically. Easy bug to miss.
- **ε counter persists across rounds** — so exploration decays smoothly over
  the whole run, not per-round.
- **Replay buffers never leave clients** — this is the whole privacy point.
  Only weights cross the wire.
- **Held-out eval env uses default parameters** — matches no single client.
  This tests whether the aggregated policy **generalizes**, not whether it
  memorized one client's distribution.

### Non-IID client profiles

When `heterogeneous=True` (default), each client gets a random profile:

| Parameter      | Range          | What it does                              |
|----------------|----------------|-------------------------------------------|
| `task_scale`   | [0.6, 1.4]     | Multiplies input size & CPU demand        |
| `max_speed`    | [2.0, 8.0] m/s | Faster = noisier uplink rate              |
| `r_mec`        | [0.3, 0.9]     | Client's sensitivity to MEC privacy cost  |
| `r_cloud`      | [0.1, 0.4]     | Client's sensitivity to Cloud privacy cost|

The profiles are printed to console at the start of every FL run and saved
into `config.yaml`.

---

## Configuration reference

### Shared DQN hyperparameters

| Param           | Default | Notes                                         |
|-----------------|---------|-----------------------------------------------|
| `BATCH_SIZE`    | 128     | Replay minibatch                              |
| `GAMMA`         | 0.99    | Discount                                      |
| `TAU`           | 0.005   | Soft target update                            |
| `LR`            | 5e-5    | AdamW learning rate                           |
| `REPLAY_CAPACITY` | 100000 | Per-client in FL mode                        |
| `TRAIN_FREQ`    | 4       | Optimize every N env steps                    |
| `EVAL_EVERY`    | 50      | Episodes between greedy evals (single-agent)  |

### Single-agent presets (`CONFIGS`)

| Name         | ε_start | ε_end | decay | Notes                           |
|--------------|---------|-------|-------|---------------------------------|
| `fast_decay` | 1.0     | 0.05  | 2000  | Exploration dies fast           |
| `slow_decay` | 1.0     | 0.05  | 15000 | Smooth long decay (recommended) |
| `persistent` | 1.0     | 0.15  | 15000 | Slow decay + high ε floor       |

### FL presets (`FL_PRESETS`)

| Name          | clients | rounds | local_eps | decay  | Purpose       |
|---------------|---------|--------|-----------|--------|---------------|
| `fl_quick`    | 3       | 10     | 5         | 2000   | Smoke test    |
| `fl_standard` | 5       | 50     | 20        | 15000  | Main FL run. Matches `slow_decay` schedule for direct comparison with single-agent. |

### FL-specific parameters

| Param             | Default | Notes                                                          |
|-------------------|---------|----------------------------------------------------------------|
| `n_clients`       | 5       | Number of simulated mobile devices                             |
| `n_rounds`        | 50      | FedAvg aggregation rounds                                      |
| `local_episodes`  | 20      | Episodes each client trains per round                          |
| `client_fraction` | 1.0     | Fraction of clients sampled per round (1.0 = all)              |
| `heterogeneous`   | True    | Non-IID client profiles. Set False for an IID control run.     |

---

## Suggested experiments for the writeup

1. **Baseline vs FL at matched env steps.** Single-agent trains on
   ~50 000 steps (1000 eps × 50); FL-standard trains on 5 clients ×
   50 rounds × 20 eps × 50 steps = 250 000 total env steps, but those are
   split across 5 agents. Report reward / violation rate / energy side-by-side.

2. **IID vs non-IID FL.** Run `F3` twice with `heterogeneous=1` and
   `heterogeneous=0`, holding everything else equal. Non-IID typically
   converges slower and to a slightly worse global optimum — this is the
   well-known FedAvg weakness and worth showing.

3. **Client-count sweep.** Run `F3` with `n_clients ∈ {2, 5, 10}` keeping
   total env steps roughly constant. Shows communication-vs-sample tradeoff.

4. **Partial participation.** Set `client_fraction < 1.0` (e.g. 0.5) to
   simulate realistic networks where not every device is reachable every
   round.

---

## Tips

- For a first run, pick **F1** to confirm everything works (~1 minute).
- For writeup results, use **F2** (`fl_standard`) with `repeats=3` and
  compare against baseline option `3` (`slow_decay`) also with `repeats=3`.
  Multiple seeds give mean±std error bars, which look much more credible
  than single-seed numbers.
- All curves are also available live in TensorBoard — hit `7` while a run
  is happening to inspect in real time.
