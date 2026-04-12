# TMNF — Trackmania Nations Forever RL

Hill-climbing / evolutionary / CMA-ES / Q-learning agent for A03. See the root `CLAUDE.md` for full architecture documentation.

---

## Quick start

```bash
# Single experiment
python main.py <experiment_name> [--no-interrupt] [--re-initialize]

# Grid search over multiple param combinations
python grid_search.py config/my_grid.yaml [--no-interrupt]
```

Results land in `experiments/<track>/<name>/results/`.

`--re-initialize` ignores any existing weights file and reruns probe + cold-start from scratch.

---

## Configuring a run

On first run, `config/training_params.yaml` is copied into `experiments/<track>/<name>/training_params.yaml`. Edit the experiment copy to tune without affecting other experiments.

```yaml
track: a03_centerline
speed: 10.0
in_game_episode_s: 30.0
n_sims: 100
mutation_scale: 0.05
mutation_share: 1.0      # fraction of weights perturbed per mutation
probe_s: 15.0
cold_restarts: 20
cold_sims: 5
n_lidar_rays: 8          # 0 = disabled

policy_type: genetic     # see Policy types below

policy_params:
  # type-specific hyperparams (see below)
```

---

## Policy types

Set `policy_type` in `training_params.yaml`. Each type uses the same `n_sims` budget but runs a different training loop.

| `policy_type` | Algorithm | Notes |
|---|---|---|
| `hill_climbing` | Mutate-and-keep `WeightedLinearPolicy` | Default. Includes probe + cold-start phases. |
| `neural_net` | Mutate-and-keep `NeuralNetPolicy` (MLP) | Pure numpy, no framework needed. Configure `hidden_sizes`. |
| `epsilon_greedy` | Tabular Q-learning, ε-greedy exploration | ε decays per episode. Q-table is in-memory only. |
| `mcts` | UCT-style online Q-learner (UCB1) | Approximation — no env cloning, builds value table over real episodes. |
| `genetic` | Population of `WeightedLinearPolicy`, evolutionary | `n_sims` = number of generations; total episodes = `n_sims × population_size`. |
| `cmaes` | CMA-ES over flat `WeightedLinearPolicy` weights (Hansen 2016) | Adapts full covariance matrix; automatic step-size control via CSA. |

### Policy-specific params

```yaml
# neural_net
policy_params:
  hidden_sizes: [16, 16]

# epsilon_greedy
policy_params:
  n_bins: 3
  epsilon: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.05
  alpha: 0.1
  gamma: 0.99

# mcts
policy_params:
  n_bins: 3
  c: 1.41        # UCB1 exploration constant
  alpha: 0.1
  gamma: 0.99

# genetic
policy_params:
  population_size: 20
  elite_k: 3
  # mutation_scale inherited from top-level

# cmaes
policy_params:
  population_size: 20   # λ — offspring per generation
  initial_sigma: 0.3    # starting step size (adapts automatically via CSA)
```

#### CMA-ES details

`CMAESPolicy` implements `(μ/μ_w, λ)-CMA-ES` (Hansen 2016) over the concatenated `[steer | accel | brake]` weight vector of a `WeightedLinearPolicy` (~63 dimensions for the base observation space).

Each generation:
1. **Sample** — draw λ offspring from `N(mean, σ²·C)` using the cached eigen-factorization `C = B D² Bᵀ`
2. **Evaluate** — run one episode per offspring
3. **Update** — weighted mean recombination, cumulative step-size adaptation (CSA) for σ, and rank-1 + rank-μ covariance update

The champion (best individual seen across all generations) is saved in standard `WeightedLinearPolicy` YAML format so analytics, heatmaps, and inference work without modification. Step-size σ is reported per generation in logs and adapts automatically — no `mutation_scale` tuning needed.

`n_sims` controls the number of generations; total episodes = `n_sims × population_size`.

### Training phases (hill_climbing only)

1. **Probe** — runs 6 fixed-action episodes (brake/accel × left/straight/right; coast skipped) for `probe_s` seconds each to establish a reward floor.
2. **Cold-start search** — up to `cold_restarts` rounds of random-init hill-climbing, `cold_sims` sims each. Stops early if the floor is beaten.
3. **Greedy** — `n_sims` iterations of mutate-and-keep.

All other policy types skip probe and cold-start and go straight to greedy.

### Episode warmup

The first 100 steps of every episode force full-throttle straight (`accel + straight, no brake`) regardless of the policy. This covers the braking-start phase so the policy's Q-table / weights are not poisoned by forced behaviour.

---

## LIDAR

Set `n_lidar_rays > 0` to append wall-distance observations from a screenshot-based LIDAR sensor. The `LidarSensor` class:

- Captures the game window via MSS
- Converts to a 128×32 binary edge image (grayscale → threshold → Canny → dilate → blur)
- Raycasts `n_lidar_rays` evenly spaced angles from 0 to π, returning normalised distances in ~[0, 1]

LIDAR rays are appended to the observation vector. All policies handle variable-length observations automatically; `WeightedLinearPolicy` auto-migrates existing weights files to add new LIDAR keys (initialised to 0.0).

Requires: `mss`, `opencv-python`, `pywin32`.

---

## Grid search

Copy `config/grid_search_template.yaml`, set any param to a list to sweep it:

```yaml
base_name: "gs_v1"
training_params:
  mutation_scale: [0.05, 0.1, 0.2]   # 3-way sweep
  n_sims: 50
  ...
reward_params:
  centerline_weight: [-0.1, -0.5]    # 2-way sweep
  ...
```

This creates one experiment per unique combination (3 × 2 = 6 here). Experiment names encode only the varied params: `gs_v1__ms0.05__cw_n0.1`, etc.

---

## Reward & parameter tuning reference

### When a reward param needs changing

| Param | Symptom |
|---|---|
| `progress_weight` | All runs score similarly (low reward variance) → scale too high. Car ignores all other signals entirely → reduce. |
| `centerline_weight` | Car wanders 10+ m but still improves → too weak. Car hugs centerline but drives slowly / avoids steering → too strong. |
| `centerline_exp` | Car oscillates wide/narrow alternately → increase to 3–4 to punish large drifts more sharply without tightening near-center tolerance. |
| `accel_bonus` | >90% accel steps but car still crashes or never steers → too high; reduce. Coasting dominates after cold-start → too low; increase. |
| `step_penalty` | Car times out without finishing when it clearly could → too small; increase to −0.1 or more. |
| `finish_bonus` | Car never reaches the finish despite good driving → too small; increase to 5000–10000. Also check `in_game_episode_s` is long enough to allow finishing. |
| `finish_time_weight` | Car finishes but brakes before the line → weight too small; increase magnitude. |
| `airborne_penalty` | Car bounces/falls frequently but penalty has no effect → fires rarely or too small vs other signals; increase. |

### When a training param needs changing

| Param | Symptom |
|---|---|
| `mutation_scale` | Hill-climbing never improves (all candidates regress) → too large, mutations destroy good weights. Improvement rate high early then stalls → too small to escape local minima; increase. |
| `n_sims` | Best reward plateaus in first 30% of sims → wasted budget; reduce and run more experiments. Large variance in final reward across independent runs → increase. |
| `in_game_episode_s` | Most episodes truncate (not crash/finish) → too short; car never reaches the finish. Most episodes crash very early → may be too long; reduce or tighten `crash_threshold_m`. |
| `crash_threshold_m` | Too many mid-episode crashes introducing noise → lower threshold to terminate faster. Too many truncated episodes when car is actually driving okay → threshold may be fine; check `in_game_episode_s`. |
| `cold_restarts` / `cold_sims` | Cold-start never beats probe floor → increase restarts or sims per restart. Cold-start always wins on restart 1 → too many restarts; reduce. |
| `speed` | Car shows oscillation or instability at 10× but not at 5× → reduce game speed. |
| `probe_s` | Probe rewards are all near-zero (car barely moves) → too short; increase so each action produces a meaningful episode. |

### Episode length vs finish signals

`in_game_episode_s = 13.0` with `par_time_s = 60.0` means the finish line is unreachable in a normal episode. `finish_bonus` and `finish_time_weight` have no effect until `in_game_episode_s` is increased to cover the full track. This is intentional during early training (focus on the track start), but should be revisited once the policy is competent for the first section.
