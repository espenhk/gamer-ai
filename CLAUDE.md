# CLAUDE.md

Trackmania Nations Forever RL agent. Drives autonomously via hill-climbing / evolutionary / CMA-ES / Q-learning trained against live TMInterface session.

**Runtime Windows-only**: `pywin32`, `mss` window grab, `tminterface` bind to live game process.

---

## Repository Structure

```
tmnf-ai/
├── main.py                 # Entry point — python main.py <experiment_name>
├── grid_search.py          # Grid search over param combinations (supports --distribute)
├── analytics.py            # Experiment result plots and summary tables
├── param_explorer.py       # Interactive weight/param exploration tool
├── policies.py             # Backward-compat shim → framework + games/tmnf policies
├── setup_and_run.ps1       # Windows bootstrap script
├── pyproject.toml
├── framework/              # Game-agnostic training loop, obs spec, analytics, base policies
├── games/
│   └── tmnf/               # TMNF-specific env, reward, lidar, steering, policies, clients, tools
├── clients/                # Backward-compat shim → games/tmnf/clients
├── rl/                     # Backward-compat shim + PPO/pretrain experiments
├── distributed/            # Coordinator, worker, protocol for distributed grid search
├── infrastructure/         # Terraform: auth, remote_state, environment (Azure VMs)
├── config/                 # Master configs + grid-search templates
├── experiments/            # Per-experiment results (git-ignored)
├── tests/                  # Unit tests
├── tracks/, replays/       # Centerline .npy files and TMNF replay .Gbx files
└── runs/, plans/           # Saved run metadata and planning notes
```

---

## Running

```bash
# Single experiment
python main.py <experiment_name> [--no-interrupt] [--re-initialize]

# Grid search over param combinations
python grid_search.py config/my_grid.yaml [--no-interrupt]

# Tests
python -m pytest tests/
```

First run with new name: `experiments/<track>/<name>/` created, both master configs copied in. Edit experiment copies to tune without affecting others. `--re-initialize` ignores existing weights file, re-runs probe + cold-start.

---

## Policies

All policies live in `policies.py`, inherit `BasePolicy`. Active policy set via `policy_type` in `training_params.yaml`.

| `policy_type` | Class | Algorithm |
|---|---|---|
| `hill_climbing` | `WeightedLinearPolicy` | Mutate-and-keep. Includes probe + cold-start phases. |
| `neural_net` | `NeuralNetPolicy` | MLP (pure numpy). Mutate-and-keep on network weights. |
| `epsilon_greedy` | `EpsilonGreedyPolicy` | Tabular Q-learning, ε-greedy exploration, ε decays per episode. |
| `mcts` | `MCTSPolicy` | UCT-style online Q-learner (UCB1). No env cloning — builds value table over real episodes. |
| `genetic` | `GeneticPolicy` | Population of `WeightedLinearPolicy` instances. Evolutionary selection + crossover + mutation. |
| `cmaes` | `CMAESPolicy` | `(μ/μ_w, λ)-CMA-ES` (Hansen 2016) over flat `WeightedLinearPolicy` weights. Automatic step-size + covariance adaptation. |

`SimplePolicy` = non-trainable hand-coded PD baseline (see `steering.py`).

### WeightedLinearPolicy

Three independent linear heads (steer, accel, brake), each `dot(weights, normalised_obs)`. Weights stored in YAML. `mutated(scale, share)` adds Gaussian noise to random `share` fraction of weights; features pre-normalised so all contribute equally per mutation step. Auto-migrates existing weight files when new observation features added.

### GeneticPolicy

Maintains population of `WeightedLinearPolicy` instances. Each generation: evaluate all individuals (`eval_episodes` episodes each, averaged), keep top `elite_k` unchanged, breed rest via uniform crossover between two random elites + mutation. Best individual ever seen = champion, saved to YAML for inference.

### CMAESPolicy

Implements `(μ/μ_w, λ)-CMA-ES` (Hansen 2016) over concatenated `[steer | accel | brake]` weight vector of `WeightedLinearPolicy` (~63 dimensions for base observation space).

**Training loop** (called from `_greedy_loop_cmaes`):
1. `sample_population()` — draws λ offspring from `N(mean, σ²·C)` using cached eigen-factorization `C = B D² Bᵀ`
2. Evaluate each offspring for `eval_episodes` episodes → average reward → reward vector
3. `update_distribution(rewards)` — weighted mean recombination (top μ = λ//2 elites), cumulative step-size adaptation (CSA) for σ, rank-1 + rank-μ covariance update

**Key properties**: `population_size` (λ), `sigma` (current σ), `champion_reward`.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `population_size` | `20` | λ — offspring sampled per generation |
| `initial_sigma` | `0.3` | Starting step size (adapts via CSA each generation) |
| `eval_episodes` | `1` | Episodes per individual per generation (averaged for fitness) |

`n_sims` controls generations; total episodes = `n_sims × population_size × eval_episodes`. No `mutation_scale` tuning needed — σ adapts automatically.

> **Budget note**: `eval_episodes > 1` multiplies total episode count by that factor. For `GeneticPolicy` the same formula applies: `n_sims × population_size × eval_episodes`. Keep `eval_episodes: 1` in grid-search templates to preserve comparability with existing runs unless you are explicitly studying variance reduction vs episode budget.

`save()` writes champion in `WeightedLinearPolicy` YAML format so analytics, weight heatmaps, inference work unchanged.

---

## Training Phases

Only `hill_climbing` runs probe and cold-start. All others go straight to greedy.

**1. Probe** (no weights file, or `--re-initialize`)
Runs 6 fixed-action episodes (brake/accel × left/straight/right, `probe_s` seconds each). Establishes reward floor for cold-start comparison.

**2. Cold-start search**
Up to `cold_restarts` rounds of random-init hill-climbing, `cold_sims` simulations each. Stops early if any restart beats probe floor. Best policy saved, used as greedy starting point.

**3. Greedy optimisation**
`n_sims` iterations (or generations for `genetic`). Best weights saved after each improvement.

---

## Configuration

### `config/training_params.yaml`

| Parameter | Default | Description |
|---|---|---|
| `track` | `a03_centerline` | Stem of `.npy` file in `tracks/` |
| `speed` | `10.0` | Game speed multiplier (TMInterface max 10×) |
| `in_game_episode_s` | `30.0` | In-game seconds per episode |
| `n_sims` | `100` | Greedy simulations / generations |
| `mutation_scale` | `0.05` | Std-dev of Gaussian noise per mutation |
| `mutation_share` | `1.0` | Fraction of weights perturbed per mutation (1.0 = all) |
| `probe_s` | `15.0` | In-game seconds per probe action run |
| `cold_restarts` | `20` | Max random restarts in cold-start search |
| `cold_sims` | `5` | Hill-climb sims per cold-start restart |
| `n_lidar_rays` | `8` | LIDAR rays appended to observation (0 = disabled) |
| `policy_type` | `genetic` | Algorithm (see Policies table above) |
| `policy_params` | `{}` | Type-specific hyperparams |

### `config/reward_config.yaml`

| Parameter | Default | Description |
|---|---|---|
| `progress_weight` | `10000.0` | Primary signal — proportional to track progress delta |
| `centerline_weight` | `-0.1` | Lateral offset penalty coefficient |
| `centerline_exp` | `2.0` | Exponent for centerline penalty (2 = quadratic) |
| `speed_weight` | `0.05` | Bonus per m/s (tie-breaker) |
| `step_penalty` | `-0.05` | Per-tick time cost |
| `finish_bonus` | `5000.0` | One-time bonus at `track_progress >= 1.0` |
| `finish_time_weight` | `-5.0` | Penalty/bonus relative to `par_time_s` |
| `par_time_s` | `60.0` | Reference lap time in seconds |
| `accel_bonus` | `0.5` | Flat reward per step when throttle pressed |
| `airborne_penalty` | `-1.0` | Applied when ≤1 wheel contact AND `vertical_offset <= 0` |
| `lidar_wall_weight` | `-5.0` | Wall proximity: `weight * (1 - min_ray)^2` |
| `crash_threshold_m` | `25.0` | Terminates episode when `|lateral_offset| > threshold` |

---

## RL Environment (`rl/env.py`)

### Observation (15 + n_lidar_rays floats, float32)

Defined in `obs_spec.py` — single source of truth for feature names, scales, descriptions.

| Index | Name | Scale | Description |
|-------|------|-------|-------------|
| 0 | `speed_ms` | 50.0 | Vehicle speed in m/s |
| 1 | `lateral_offset_m` | 5.0 | Metres from centreline (neg=left, pos=right) |
| 2 | `vertical_offset_m` | 2.0 | Metres above (+) / below (-) centreline |
| 3 | `yaw_error_rad` | π | Track heading minus car heading, [−π, π] |
| 4 | `pitch_rad` | 0.3 | Nose-up/down rotation |
| 5 | `roll_rad` | 0.3 | Tilt left/right |
| 6 | `track_progress` | 1.0 | Fraction of track completed, [0, 1] |
| 7 | `turning_rate` | 65536.0 | Raw TMInterface steer value, ±65536 |
| 8–11 | `wheel_N_contact` | 1.0 | Ground contact per wheel (0 or 1) |
| 12–14 | `angular_vel_N` | 5.0 | Angular velocity x/y/z (rad/s) |
| 15+ | `lidar_i` | 1.0 | Wall distance rays ~[0, 1] (if `n_lidar_rays > 0`) |

### Action Space

`Box([-1, 0, 0], [1, 1, 1], shape=(3,), dtype=float32)`

| Index | Name | Range | Notes |
|-------|------|-------|-------|
| 0 | steer | [−1, 1] | Maps to [−65536, 65536] in-game |
| 1 | accel | [0, 1] | Thresholded at 0.5 → bool |
| 2 | brake | [0, 1] | Thresholded at 0.5 → bool; can fire simultaneously with accel |

Policies using Discrete(9) abstraction ({brake, coast, accel} × {left, straight, right}) convert internally via `ACTIONS` in `clients/rl_client.py`.

### Termination

- **Finished:** `track_progress >= 1.0`
- **Crashed:** `|lateral_offset| > crash_threshold_m`
- **Truncated:** elapsed time exceeded

### Episode Warmup

First 100 steps force full-throttle straight (`accel + straight, no brake`) regardless of policy. Covers braking-start phase so weights/Q-tables not updated during forced behaviour.

---

## LIDAR (`lidar.py`)

Set `n_lidar_rays > 0` to append wall-distance observations. `LidarSensor`:
1. Captures game window via MSS
2. Converts to 128×32 binary edge image (grayscale → threshold → Canny → dilate → blur)
3. Raycasts `n_lidar_rays` evenly spaced angles from 0 to π, returning normalised distances ~[0, 1]

LIDAR rays appended to observation. All policies handle variable-length observations; `WeightedLinearPolicy` auto-migrates weight files to add new keys (initialised to 0.0).

Requires `mss`, `opencv-python`, `pywin32`.

---

## Grid Search (`grid_search.py`)

```bash
python grid_search.py config/my_grid.yaml
```

Set any param to list to sweep it:

```yaml
base_name: "gs_v1"
training_params:
  mutation_scale: [0.05, 0.1, 0.2]   # 3-way sweep
  n_sims: 50
reward_params:
  centerline_weight: [-0.1, -0.5]    # 2-way sweep
```

Creates one experiment per Cartesian-product combination (3 × 2 = 6 here). Names encode only varied params: `gs_v1__ms0.05__cw_n0.1`.

---

## Distributed training (`distributed/`)

Scale grid search across multiple Windows VMs by splitting combinations over coordinator + worker pool.

- `distributed/coordinator.py` — HTTP work-queue server. Bearer-token auth; heartbeat-based re-queue of stalled jobs.
- `distributed/worker.py` — polls `/work`, runs `train_rl()` locally against its TMInterface session, posts `ExperimentData` back to `/result`.
- `distributed/protocol.py` — `ComboSpec` / `ResultPayload` dataclasses + JSON (de)serialization shared by both sides.

Entry point: `python grid_search.py <config> --distribute` (coordinator mode). Workers launched independently on each VM.

---

## Infrastructure (Azure)

Three-stage Terraform stack under `infrastructure/` provisions distributed training fleet.

- `auth/` — service principal + role assignments.
- `remote_state/` — storage account for shared Terraform state.
- `environment/` — Windows 11 Pro VMs (1 coordinator + N workers), Key Vault for admin passwords, NSG allows RDP only from single configured IP.

See `infrastructure/README.md` for operational commands (plan/apply, start/stop/deallocate, worker scaling).

---

## Analytics (`analytics.py`)

Called automatically at end of each experiment/grid-search run. Writes plots and summary JSON to `experiments/<track>/<name>/results/`. Skipped phases produce no output files.

---

## Threading Model

TMInterface callback-driven (`on_run_step`); RL loop step-driven (`env.step()`). `RLClient` bridges with:
- `_action` queue (RL thread → game thread)
- `_state_queue` (game thread → RL thread, maxsize=1; drain before put)
- `_episode_ready` event (signals env reset complete)

Daemon keepalive thread keeps `iface.running` alive. `on_registered` sets event that `TMNFEnv.__init__` waits on before returning.

---

## Dependencies

Managed by Poetry. Run `poetry install` from repo root.

`tminterface` and `pygbx` not on PyPI — install from source before `poetry install`.

Core runtime deps: `numpy`, `scipy`, `gymnasium`, `pyyaml`, `matplotlib`, `opencv-python`, `mss`, `pywin32`, `tminterface`, `pygbx`.