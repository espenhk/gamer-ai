# CLAUDE.md

Trackmania Nations Forever RL agent. Drives autonomously via hill-climbing / evolutionary / CMA-ES / Q-learning trained against live TMInterface session.

**Runtime per game**:
- TMNF: Windows-only (`pywin32`, `mss` window grab, `tminterface` bind to live game process).
- TORCS: Linux/Windows via `gym_torcs`.
- SC2: Linux/Windows/Mac via PySC2; runs headless on Linux against the Blizzard SC2 binary.

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
│   ├── tmnf/               # TMNF-specific env, reward, lidar, steering, policies, clients, tools
│   ├── torcs/              # TORCS racing simulator integration
│   └── sc2/                # StarCraft 2 integration via PySC2 (Linux-friendly, headless)
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
# Single experiment (TMNF — default)
python main.py <experiment_name> [--no-interrupt] [--re-initialize]

# Run on a different game
python main.py <experiment_name> --game torcs
python main.py <experiment_name> --game sc2

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
| `neural_dqn` | `NeuralDQNPolicy` | Deep Q-learning with replay buffer + target network. |
| `reinforce` | `REINFORCEPolicy` | Monte Carlo policy gradient (optional running-mean baseline). |
| `lstm` | `LSTMEvolutionPolicy` | Recurrent (LSTM) policy, trained by evolutionary search over network weights. |

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

### NeuralDQNPolicy

MLP Q-network (pure numpy) with experience replay and a periodically-synced target network. Action space is the 9-element discrete set (`DISCRETE_ACTIONS`). ε-greedy exploration decays linearly from `epsilon_start` to `epsilon_end` over `epsilon_decay_steps` environment steps.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `hidden_sizes` | `[64, 64]` | Hidden layer widths of the Q-network MLP |
| `replay_buffer_size` | `10000` | Max transitions stored in the replay buffer |
| `batch_size` | `64` | Transitions sampled per gradient update |
| `min_replay_size` | `500` | Buffer must reach this size before training starts |
| `target_update_freq` | `200` | Steps between copying online weights → target network |
| `learning_rate` | `0.001` | Adam-style gradient step size |
| `epsilon_start` | `1.0` | Initial exploration rate |
| `epsilon_end` | `0.05` | Final exploration rate |
| `epsilon_decay_steps` | `5000` | Steps over which ε decays linearly |
| `gamma` | `0.99` | Discount factor |

### REINFORCEPolicy

Monte Carlo policy-gradient over the 9-element discrete action set. Collects full episodes, computes discounted returns, optionally subtracts a running-mean baseline, then updates the softmax policy network via gradient ascent.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `hidden_sizes` | `[64, 64]` | Hidden layer widths of the policy MLP |
| `learning_rate` | `0.001` | Gradient step size |
| `gamma` | `0.99` | Discount factor for return computation |
| `entropy_coeff` | `0.01` | Entropy regularisation weight (encourages exploration) |
| `baseline` | `"running_mean"` | Return baseline: `"running_mean"` or `"none"` |

### LSTMEvolutionPolicy

LSTM recurrent policy trained by CMA-ES-style evolutionary search over flattened network weights. The hidden state is reset each episode; at each step the LSTM receives the current observation and emits logits over the 9-element discrete action set.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `hidden_size` | `32` | LSTM hidden/cell state dimensionality |
| `population_size` | `20` | λ — offspring evaluated per generation |
| `initial_sigma` | `0.05` | Starting perturbation scale (smaller than CMAESPolicy because the LSTM weight space is larger) |

`n_sims` controls generations; total episodes = `n_sims × population_size`. Saved champion weights are incompatible across different `hidden_size` or `n_lidar_rays` values — changing either requires `--re-initialize`.

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

Policies using the Discrete(25) abstraction ({full brake, half brake, coast, half accel, full accel} × {full left, half left, straight, half right, full right}) convert internally via `ACTIONS` in `clients/rl_client.py`.

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

## StarCraft 2 (`games/sc2/`)

**Scope**: 7 standard PySC2 minigames and full 1v1 RL training against a built-in bot on any ladder map (e.g. `Simple64`). Fog-of-war belief machinery is deferred to a future issue.

### Setup

1. Install the Blizzard StarCraft 2 binary. On Linux use the [headless build](https://github.com/Blizzard/s2client-proto#linux-packages); on Windows/Mac the regular client works.
2. Set `SC2PATH` to the install root (Linux), or place the install in `~/StarCraftII/` (the PySC2 default).
3. Download the [PySC2 maps](https://github.com/Blizzard/s2client-proto#downloads) (mini-games + ladder maps) and unzip them into `Maps/` under the install root.
4. Install the optional sc2 dependency group:
   ```bash
   poetry install --with sc2
   ```

### Running

```bash
# Default minigame (MoveToBeacon)
python main.py myrun --game sc2

# Choose a different minigame: edit games/sc2/config/training_params.yaml
# (or the per-experiment copy under experiments/) and set map_name.

# 1v1 ladder training against a very_easy bot on Simple64:
#   1. Edit experiments/sc2_Simple64/<name>/training_params.yaml:
#        map_name: Simple64
#        in_game_episode_s: 600.0
#        policy_type: sc2_genetic    # or cmaes, neural_dqn, reinforce, lstm
#   2. Edit experiments/sc2_Simple64/<name>/reward_config.yaml:
#        score_weight: 0.0
#        economy_weight: 0.001
#   3. Run:
python main.py myrun --game sc2 --no-interrupt
```

The first run creates `experiments/sc2_<map>/<name>/` and copies both master configs in. Edit the experiment-local copies to tune without affecting other runs.

### Config knobs

| Key | Default | Notes |
|---|---|---|
| `map_name` | `MoveToBeacon` | Any of the 7 minigame names or a ladder map (e.g. `Simple64`). |
| `agent_race` | `random` | `protoss` / `terran` / `zerg` / `random`. |
| `bot_difficulty` | `very_easy` | Only for ladder maps. |
| `step_mul` | `8` | Game ticks per env step (~0.5 s real-time). |
| `screen_size`, `minimap_size` | `64` | Square feature-layer resolutions. |
| `in_game_episode_s` | `120.0` | Seconds before truncation; use `600.0` for ladder maps. |

### Supported policies

SC2-specific policies and tabular framework policies work on both the 13-dim
minigame and 21-dim ladder observation spaces.  The framework's generic linear
policies (`hill_climbing`, `neural_net`, and the base `genetic`) are **not**
compatible with SC2 because their output encoding clips `fn_idx` to `[-1, 1]`
and thresholds `x`/`y` to binary — use `sc2_genetic` instead.

| `policy_type` | Algorithm | Notes |
|---|---|---|
| `sc2_genetic` | Evolutionary (population of `SC2LinearPolicy`) | Default; recommended for first runs. |
| `epsilon_greedy` | Tabular Q-learning over `DISCRETE_ACTIONS` | |
| `mcts` | UCT-style Q-learning over `DISCRETE_ACTIONS` | |
| `cmaes` | (μ/μ_w, λ)-CMA-ES over `SC2LinearPolicy` weights | `games/sc2/policies.py` |
| `neural_dqn` | Deep Q-network with experience replay | `games/sc2/policies.py` |
| `reinforce` | Monte Carlo policy gradient | `games/sc2/policies.py` |
| `lstm` | LSTM policy trained by evolutionary search | `games/sc2/policies.py` |

### Reward tuning (`reward_config.yaml`)

| Reward param | Default | Description |
|---|---|---|
| `score_weight` | `1.0` | Coefficient on the PySC2 cumulative `score` delta. |
| `win_bonus` / `loss_penalty` | `100` / `-100` | Terminal reward on `player_outcome > 0` / `< 0` (ladder maps only). |
| `step_penalty` | `-0.001` | Per-tick time cost. |
| `idle_penalty` | `0.0` | Per-step penalty when `army_count == 0 and food_used < food_cap` (BuildMarines / economy maps). |
| `idle_bonus` | `0.0` | Per-step bonus when the agent issues `no_op` AND friendly units are within combat range of an enemy on screen.  Issue #127 — opt-in, default `0.0`.  Useful for combat minigames (DefeatRoaches, DefeatZerglingsAndBanelings) where standing still lets units shoot. |
| `economy_weight` | `0.0` | Coefficient on (minerals + vespene) delta — recommended `0.001` for ladder maps. |

For ladder maps (`Simple64` etc.) the recommended preset is:

```yaml
score_weight: 0.0
win_bonus: 100.0
loss_penalty: -100.0
step_penalty: -0.001
economy_weight: 0.001
```

`win_bonus` and `loss_penalty` are always active for ladder maps regardless of `score_weight`.

### Action space (issues #122, #127)

Continuous `Box([fn_idx, x, y, queue], shape=(4,))`.  `DISCRETE_ACTIONS` (used by tabular policies) is now ``[no_op, select_army, Move_screen × N×N]`` where ``N = SCREEN_GRID_RESOLUTION`` (default 8 → 64-cell grid, 66 rows total).

- Row 0 is `no_op` so tabular policies can elect to do nothing (issue #127) — necessary for "stand still and shoot" tactics.
- Row 1 is `select_army` (also the warmup action).
- Rows 2..N-1 are `Move_screen` calls at cell centres of an 8×8 grid covering the screen at one-cell-per-8-pixels granularity.

`SC2MultiHeadLinearPolicy` (the `sc2_genetic` default) now emits **continuous** `(x, y) ∈ [0, 1]²` coordinates via a sigmoid head (issue #122).  Pre-#122 weight files (with `spatial_{0..8}_weights` keys) load with the new `x_weights` / `y_weights` defaulting to zero.

### Observation space (issue #126)

Three preset specs, opt-in via the `obs_spec_preset` training param:

| Preset | Dim | Default for | Notes |
|---|---|---|---|
| `minigame` | 13 | All minigame names | Player totals, selected-unit summary, screen `player_relative` summary. |
| `ladder` | 43 | All non-minigame maps | Adds `food_workers`/`food_army`, idle worker / warp gate / larva counts, minimap stats (`minimap_self_count`, `minimap_enemy_count`, `minimap_visible_frac`, `minimap_explored_frac`, `minimap_camera_x/y`, `game_loop`), the 13 PySC2 score-cumulative entries, screen unit-density / mean-HP, and top-K enemy counts (`topk_enemy_within_8`, `topk_enemy_within_24`). |
| `rich`  | 80 | Opt-in only | Adds 8 per-unit-type counts (Marine / SCV / Zergling / Drone / Probe / Stalker / Roach / Mutalisk), screen quadrant counts (NE/NW/SE/SW × self/enemy), top-3 closest enemies' (rel_x, rel_y, hp_ratio), available-actions binary mask, and last-action one-hot. |

Set `obs_spec_preset: rich` in `training_params.yaml` to opt into the rich preset on any map.  Existing weight files migrate via the standard "missing key → 0.0" path — old champions can be loaded under any preset, and the new feature weights default to zero.

---

## Dependencies

Managed by Poetry. Run `poetry install` from repo root.

`tminterface` and `pygbx` not on PyPI — install from source before `poetry install`.

Core runtime deps: `numpy`, `scipy`, `gymnasium`, `pyyaml`, `matplotlib`, `opencv-python`, `mss`, `pywin32`, `tminterface`, `pygbx`.

Optional groups:
- `--with torcs` — `gym_torcs` for TORCS support.
- `--with sc2` — `pysc2` for StarCraft 2 support.