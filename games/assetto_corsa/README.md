# Assetto Corsa

Assetto Corsa integration for the tmnf-ai reinforcement learning framework. Uses the `assetto-corsa-rl` gym wrapper, which connects to the AC process via shared memory and an in-game Python plugin.

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Python dependencies](#python-dependencies)
- [Running Assetto Corsa](#running-assetto-corsa)
- [Configuration](#configuration)
- [Observation space](#observation-space)
- [Action space](#action-space)
- [Reward](#reward)
- [Example commands](#example-commands)
  - [Single experiment](#single-experiment)
  - [Grid search](#grid-search)
- [Supported policies](#supported-policies)

---

## Installation

### Prerequisites

- **Assetto Corsa** (commercial game, available on Steam)
- Python 3.11+, Poetry

### Python dependencies

```bash
poetry install --with assetto_corsa
```

This installs the `assetto-corsa-rl` package and all other project dependencies.

---

## Running Assetto Corsa

The `assetto-corsa-rl` package connects to a running AC session via shared memory. You must have AC open and a session active before starting training.

1. **Launch Assetto Corsa.**
2. **Enable the Python app plugin** that comes with `assetto-corsa-rl`. See that package's README for plugin installation instructions (typically copying files into the AC `apps/python/` folder and enabling the app in-game).
3. **Start a practice session** on the track you want to train on. The plugin will begin exposing telemetry data.
4. **Run the training command** — `AssettoCorsoEnv` connects to the running session automatically.

---

## Configuration

| File | Purpose |
|---|---|
| `games/assetto_corsa/config/training_params.yaml` | Episode settings, policy type, hyperparams, `n_vision` |
| `games/assetto_corsa/config/reward_config.yaml` | Reward weights |

To enable vision features, set `n_vision: N` in `training_params.yaml`. This appends `N` vision-distance features to the base observation vector.

---

## Observation space

16 base features (all normalised by their scale values before being fed to the policy):

| Feature | Scale | Description |
|---|---|---|
| `speed_ms` | 50.0 | Vehicle speed in m/s |
| `lateral_offset_m` | 5.0 | Metres from track centre |
| `yaw_error_rad` | π | Track heading minus car heading |
| `pitch_rad` | 0.3 | Nose pitch angle |
| `roll_rad` | 0.3 | Body roll angle |
| `track_progress` | 1.0 | Fraction of lap completed [0, 1] |
| `steering_angle` | 1.0 | Current steering input [−1, 1] |
| `engine_rpm` | 8000.0 | Engine RPM |
| `gear` | 6.0 | Current gear |
| `wheel_0_slip`–`wheel_3_slip` | 1.0 | Tyre slip per wheel |
| `angular_vel_x` | 5.0 | Roll rate |
| `angular_vel_y` | 5.0 | Yaw rate |
| `angular_vel_z` | 5.0 | Pitch rate |

Optional: set `n_vision: N` in `training_params.yaml` to append `N` forward-facing vision-distance features.

---

## Action space

Continuous: `Box([-1, 0, 0], [1, 1, 1], shape=(3,))`

| Output | Range | Effect |
|---|---|---|
| `steer` | [−1, 1] | Full left to full right |
| `accel` | [0, 1] | Throttle |
| `brake` | [0, 1] | Braking force |

Discrete policies use a 9-cell grid: {brake, coast, accel} × {left, straight, right}.

---

## Reward

Configured in `games/assetto_corsa/config/reward_config.yaml`:

| Parameter | Value | Effect |
|---|---|---|
| `progress_weight` | 1000.0 | Reward per unit of lap progress |
| `centerline_weight` | −0.5 | Penalty for lateral deviation |
| `centerline_exp` | 2.0 | Exponent for centerline penalty (quadratic) |
| `speed_weight` | 0.05 | Small bonus for higher speed |
| `step_penalty` | −0.05 | Per-step time cost |
| `finish_bonus` | 500.0 | One-time reward for completing the lap |
| `finish_time_weight` | −1.0 | Penalty proportional to lap time above par |
| `par_time_s` | 150.0 | Par lap time in seconds |
| `accel_bonus` | 0.5 | Bonus for applying throttle |
| `crash_threshold_m` | 25.0 | Lateral offset (m) that terminates the episode |

---

## Example commands

### Single experiment

```bash
python main.py my_ac_run --game assetto
```

Results are saved to `experiments/assetto/my_ac_run/results/`.

### Grid search

Create a YAML file with `game: assetto` and list-valued parameters, then run:

```bash
python grid_search.py my_ac_grid.yaml --game assetto
```

Model the YAML structure on `games/torcs/config/grid_search_template.yaml`.

---

## Supported policies

All policies in the framework work with Assetto Corsa. Set `policy_type` in `games/assetto_corsa/config/training_params.yaml`.

| `policy_type` | Algorithm | Notes |
|---|---|---|
| `hill_climbing` | Mutate-and-keep linear policy (WeightedLinearPolicy) | Good starting point; includes probe + cold-start phases |
| `neural_net` | MLP mutate-and-keep | Non-linear behaviour; configure `hidden_sizes` |
| `epsilon_greedy` | Tabular Q-learning, ε-greedy | Classical RL baseline |
| `mcts` | UCT-style Q-learning (UCB1 exploration) | More systematic exploration than ε-greedy |
| `genetic` | Population of WeightedLinearPolicy, evolutionary crossover+mutation | Good for escaping local optima |
| `cmaes` | (μ/μ_w, λ)-CMA-ES over flat weight vector | Best general-purpose choice for linear policies |
| `neural_dqn` | Deep Q-network, experience replay, target network | Gradient-based neural training |
| `reinforce` | Monte Carlo policy gradient | Stochastic policy, simpler than DQN |
| `lstm` | LSTM + isotropic Gaussian ES | Useful when temporal memory matters |

Policy-specific hyperparameters go under `policy_params:` in `training_params.yaml`. See the root `README.md` or `games/tmnf/README.md` for full param reference.
