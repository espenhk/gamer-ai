# BeamNG

BeamNG.drive integration for the tmnf-ai reinforcement learning framework. Uses the `beamng_gym` package, which connects to the BeamNG.drive process via its TCP API.

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Python dependencies](#python-dependencies)
- [Running BeamNG](#running-beamng)
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

- **BeamNG.drive** (commercial game): https://www.beamng.com/
- Python 3.11+, Poetry

### Python dependencies

Install `beamng-gym` separately (it is not included in the Poetry dependency groups):

```bash
pip install beamng-gym
```

Then install the rest of the project dependencies:

```bash
poetry install
```

---

## Running BeamNG

BeamNG.drive must be running with a scenario loaded before training starts. The `BeamNGEnv` wrapper connects to it via `beamng_gym.make()`.

1. **Launch BeamNG.drive.**
2. **Enable the BeamNG Python bridge** — see the `beamng-gym` documentation for setup steps.
3. **Load a scenario** in BeamNG (e.g. the default tech demo or a custom scenario).
4. **Run the training command** — the environment connects automatically.

---

## Configuration

| File | Purpose |
|---|---|
| `games/beamng/config/training_params.yaml` | Episode settings, policy type, hyperparams |
| `games/beamng/config/reward_config.yaml` | Reward weights |

---

## Observation space

13 features (all normalised by their scale values before being fed to the policy):

| Feature | Scale | Description |
|---|---|---|
| `speed_ms` | 50.0 | Vehicle speed in m/s |
| `lateral_offset_m` | 5.0 | Metres from track centre |
| `yaw_error_rad` | π | Track heading minus car heading |
| `track_progress` | 1.0 | Fraction of lap completed [0, 1] |
| `pitch_rad` | 0.3 | Nose pitch angle |
| `roll_rad` | 0.3 | Body roll angle |
| `wheel_0_contact`–`wheel_3_contact` | 1.0 | Ground contact per wheel (0 or 1) |
| `angular_vel_x` | 5.0 | Roll rate |
| `angular_vel_y` | 5.0 | Yaw rate |
| `angular_vel_z` | 5.0 | Pitch rate |

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

Configured in `games/beamng/config/reward_config.yaml`:

| Parameter | Value | Effect |
|---|---|---|
| `progress_weight` | 10000.0 | Reward per unit of lap progress |
| `centerline_weight` | −0.1 | Penalty for lateral deviation |
| `centerline_exp` | 2.0 | Exponent for centerline penalty (quadratic) |
| `speed_weight` | 0.05 | Small bonus for higher speed |
| `step_penalty` | −0.05 | Per-step time cost |
| `finish_bonus` | 5000.0 | One-time reward for completing the lap |
| `finish_time_weight` | −5.0 | Penalty proportional to lap time above par |
| `par_time_s` | 120.0 | Par lap time in seconds |
| `accel_bonus` | 0.5 | Bonus for applying throttle |
| `crash_threshold_m` | 25.0 | Lateral offset (m) that terminates the episode |

---

## Example commands

### Single experiment

```bash
python main.py my_beamng_run --game beamng
```

Results are saved to `experiments/beamng/my_beamng_run/results/`.

### Grid search

Create a YAML file with `game: beamng` and list-valued parameters, then run:

```bash
python grid_search.py my_beamng_grid.yaml --game beamng
```

Model the YAML structure on `games/torcs/config/grid_search_template.yaml`.

---

## Supported policies

All policies in the framework work with BeamNG. Set `policy_type` in `games/beamng/config/training_params.yaml`.

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
