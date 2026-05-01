# CarRacing

Gymnasium `CarRacing-v2` integration for the tmnf-ai reinforcement learning framework. No separate game binary is needed — the environment runs entirely inside Python.

---

## Installation

### Prerequisites

- Python 3.11+, Poetry

### Python dependencies

Install the `gymnasium[box2d]` extras (provides `box2d-py` and `pygame`):

```bash
pip install "gymnasium[box2d]"
# or, if adding to the project:
poetry add "gymnasium[box2d]"
```

Then install the rest of the project dependencies:

```bash
poetry install
```

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

The underlying `CarRacing-v2` environment uses 96×96×3 pixel observations. This integration extracts a compact 9-feature vector instead:

| Feature | Scale | Description |
|---|---|---|
| `speed` | 100.0 | Vehicle speed |
| `angular_vel` | 10.0 | Rotational velocity |
| `wheel_0_ang`–`wheel_3_ang` | 100.0 | Wheel angular velocities |
| `steering` | 1.0 | Current steering input [−1, 1] |
| `gas` | 1.0 | Current throttle input [0, 1] |
| `brake` | 1.0 | Current brake input [0, 1] |

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

Configured in `games/car_racing/config/reward_config.yaml`:

| Parameter | Value | Effect |
|---|---|---|
| `native_reward_scale` | 1.0 | Scales the raw gymnasium reward signal |
| `step_penalty` | −0.1 | Per-step time cost |
| `finish_bonus` | 100.0 | One-time reward for completing the track |
| `crash_threshold_m` | 25.0 | Lateral offset (m) that terminates the episode |

---

## Example commands

### Single experiment

```bash
python main.py my_car_run --game car_racing
```

Results are saved to `experiments/car_racing/my_car_run/results/`.

### Grid search

Create a YAML file with `game: car_racing` and list-valued parameters, then run:

```bash
python grid_search.py my_car_grid.yaml --game car_racing
```

Model the YAML structure on `games/torcs/config/grid_search_template.yaml`.

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

Policy-specific hyperparameters go under `policy_params:` in `training_params.yaml`. See the root `README.md` or `games/tmnf/README.md` for full param reference.
