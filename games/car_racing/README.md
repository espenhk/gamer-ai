# CarRacing

Gymnasium `CarRacing-v2` integration for the tmnf-ai reinforcement learning framework. No separate game binary is needed — the environment runs entirely inside Python.

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Python dependencies](#python-dependencies)
- [Running](#running)
- [Configuration](#configuration)
- [Observation space](#observation-space)
- [Action space](#action-space)
- [Rewards](#rewards)
- [Example commands](#example-commands)
  - [Single experiment](#single-experiment)
  - [Grid search](#grid-search)
- [Supported policies](#supported-policies)

---

## Installation

### Prerequisites

- Python 3.11+, Poetry

### Python dependencies

```bash
poetry install --with car_racing
```

This pulls in the `car_racing` optional group (`gymnasium[box2d]` — `box2d`,
`swig`, and `pygame`). Add `deep_rl` too if you want the SAC/PPO/etc.
policies (`poetry install --with car_racing,deep_rl`).

---

## Running

No external process is required. Gymnasium manages the environment lifecycle internally. Just run the training command and everything starts automatically.

---

## Configuration

| File | Purpose |
|---|---|
| `games/car_racing/config/training_params.yaml` | Episode settings, policy type, hyperparams |
| `games/car_racing/config/reward_config.yaml` | Reward weights |

---

## Observation space

The underlying `CarRacing-v3` environment uses 96×96×3 pixel observations. This integration extracts a compact feature vector instead — car-physics features plus track-relative perception features mirroring TMNF's `obs_spec.py` (see `games/tmnf/README.md`), so the agent can anticipate upcoming curves instead of only reacting to its own physics state. The perception features are derived at runtime from `env.unwrapped.track`, the list of `(alpha, beta, x, y)` centreline checkpoints `CarRacing-v3` already builds internally.

Defined in `games/car_racing/obs_spec.py`. 12 base features + 2×lookahead-points; 18 total by default (3 lookahead waypoints).

| Feature | Scale | Description |
|---|---|---|
| `speed` | 100.0 | Vehicle speed |
| `angular_vel` | 10.0 | Rotational velocity |
| `wheel_0_ang`–`wheel_3_ang` | 300.0 | Wheel angular velocities |
| `steering` | 1.0 | Current steering input [−1, 1] |
| `gas` | 1.0 | Current throttle input [0, 1] |
| `brake` | 1.0 | Current brake input [0, 1] |
| `lateral_offset_m` | 5.0 | Signed distance from track centreline (neg=left, pos=right) |
| `yaw_error_rad` | π | Track heading minus car heading, [−π, π] |
| `track_progress` | 1.0 | Fraction of the lap's centreline checkpoints passed, [0, 1] |
| `lookahead_{step}_lat` / `lookahead_{step}_yaw` | 5.0 / π | Lateral offset / heading change at each lookahead checkpoint. Default: 3 points at checkpoint-index steps `[5, 15, 30]`. Configurable via `training_params.yaml`'s `n_lookahead_points` / `lookahead_step_spacing` (mirrors TMNF's issue #493 pattern) — see `games/car_racing/obs_spec.py::build_lookahead_steps()`. |

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

## Rewards

Configured in `games/car_racing/config/reward_config.yaml`.

| Parameter | Default | Description |
|---|---|---|
| `native_reward_scale` | 1.0 | Multiplier applied to the raw per-step reward from Gymnasium's `CarRacing-v2`. The native signal is positive for track tiles driven over and turns negative if the episode ends without completing the track. |
| `step_penalty` | −0.1 | Flat per-step time cost added on top of the scaled native reward. Encourages faster completion. |
| `finish_bonus` | 100.0 | One-time reward when all track tiles have been visited. |
| `crash_threshold_m` | 25.0 | Reserved for a future centerline-based crash penalty; not currently applied. CarRacing-v3 has no exposed lateral-offset signal, so off-track termination is instead detected via the env's own out-of-bounds check (see `termination_reason="crash"` in the analytics report). |

---

## Example commands

### Single experiment

```bash
python main.py my_car_run --game car_racing
```

Results are saved to `experiments/car_racing/my_car_run/results/`. Alongside the
generic reward/timing plots, `results.md` includes a **Reward Moving Average**
section — the mean episode reward over a trailing 100-episode window, checked
against CarRacing-v2's published "solved" benchmark (average reward >= 900 over
100 consecutive episodes; see `reward_moving_average.png`). This is the metric
to check after a long SAC/PPO run (e.g. `gs_sac.yaml`) instead of eyeballing the
raw per-episode scatter.

### Grid search

Create a YAML file with `game: car_racing` and list-valued parameters, then run:

```bash
python grid_search.py my_car_grid.yaml --game car_racing
```

Model the YAML structure on `games/torcs/config/grid_search_template.yaml`.

A checked-in `sac` config is available for a quick training-loop sanity check
against CarRacing's published "solved" benchmark (average reward >= 900 over
100 consecutive episodes):

```bash
python grid_search.py games/car_racing/config/gs_sac.yaml --game car_racing
```

---

## Supported policies

All policies in the framework work with CarRacing. Set `policy_type` in `games/car_racing/config/training_params.yaml`.

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
| `ppo` | On-policy actor-critic, clipped surrogate + GAE (pure numpy) | On-policy gradient baseline; tune `clip_range`, `n_epochs`, `gae_lambda` |
| `sac` | Stable-Baselines3 Soft Actor-Critic | Off-policy, native continuous `Box` control; see `games/car_racing/config/gs_sac.yaml` for a reproducible starting config, not yet empirically validated against the solved benchmark (issue #482) |

Policy-specific hyperparameters go under `policy_params:` in `training_params.yaml`. See the root `README.md` or `games/tmnf/README.md` for full param reference.
