# TORCS

TORCS (The Open Racing Car Simulator) integration for the tmnf-ai reinforcement learning framework.

---

## Installation

### Prerequisites

- Linux (preferred) or Windows with MSYS2
- Python 3.11+, Poetry

### TORCS binary

Install TORCS from your distro's package manager or build from source:

```bash
# Ubuntu / Debian
sudo apt install torcs
```

Windows users: follow the MSYS2 build guide in the TORCS documentation.

### Python dependencies

Install `gym_torcs` from source (not on PyPI):

```bash
git clone https://github.com/ugo-nama-kun/gym_torcs
cd gym_torcs
pip install -e .
```

Then install the rest of the project dependencies from the repo root:

```bash
poetry install
```

No extra dependency group is needed — TORCS deps are all standard.

---

## Running TORCS

TORCS must be running as a separate process before training starts. The `gym_torcs` client connects automatically when the environment is reset.

1. Start the TORCS server (SCR server mode, default port 3001):

   ```bash
   torcs -nofuel -nodamage -nolaptime &
   ```

2. In the TORCS menu, start a race in **Practice** mode with the SCR server robot.

3. Leave TORCS running and proceed to the training commands below.

---

## Configuration

| File | Purpose |
|---|---|
| `games/torcs/config/training_params.yaml` | Episode length, policy type, hyperparams |
| `games/torcs/config/reward_config.yaml` | Reward weights |
| `games/torcs/config/grid_search_template.yaml` | Grid search starting point |

---

## Observation space

19 features (all normalised by the scale values listed below before being fed to the policy):

| Feature | Scale | Description |
|---|---|---|
| `speed_ms` | 50.0 | Vehicle speed in m/s (longitudinal) |
| `lateral_offset_m` | 5.0 | Metres from track centre |
| `yaw_error_rad` | π | Track heading minus car heading |
| `track_progress` | 1.0 | Fraction of lap completed [0, 1] |
| `rpm` | 10000.0 | Engine RPM |
| `wheel_0_spin`–`wheel_3_spin` | 200.0 | Wheel spin velocities (rad/s) |
| `track_edge_0`–`track_edge_8` | 200.0 | Track edge distances at angles −90°..+90° |
| `track_position` | 1.0 | Normalised track position [−1, 1] |

---

## Action space

Continuous: `Box([-1, 0, 0], [1, 1, 1], shape=(3,))`

| Output | Range | Effect |
|---|---|---|
| `steer` | [−1, 1] | Full left to full right |
| `accel` | [0, 1] | Throttle |
| `brake` | [0, 1] | Braking force |

Discrete policies use a 25-cell grid: {full brake, half brake, coast, half accel, full accel} × {full left, half left, straight, half right, full right}.

---

## Reward

Configured in `games/torcs/config/reward_config.yaml`:

| Parameter | Value | Effect |
|---|---|---|
| `progress_weight` | 10.0 | Reward per unit of lap progress |
| `centerline_weight` | −0.5 | Penalty for lateral deviation |
| `centerline_exp` | 2.0 | Exponent for centerline penalty (quadratic) |
| `speed_weight` | 0.05 | Small bonus for higher speed |
| `step_penalty` | −0.01 | Per-step time cost |
| `finish_bonus` | 100.0 | One-time reward for completing the lap |
| `finish_time_weight` | −0.1 | Penalty proportional to lap time above par |
| `par_time_s` | 120.0 | Par lap time in seconds |
| `accel_bonus` | 0.5 | Bonus for applying throttle |
| `crash_threshold_m` | 8.0 | Lateral offset (m) that terminates the episode |

---

## Example commands

### Single experiment

```bash
python main.py my_torcs_run --game torcs
```

Results are saved to `experiments/torcs/my_torcs_run/results/`.

### Grid search

A ready-made grid search template is included:

```bash
python grid_search.py games/torcs/config/grid_search_template.yaml --game torcs
```

You can also copy and modify one of the pre-built grid configs in `games/torcs/config/` (e.g. `gs_cmaes.yaml`, `gs_genetic.yaml`, `gs_hill_climbing.yaml`).

---

## Supported policies

All policies in the framework work with TORCS. Set `policy_type` in `games/torcs/config/training_params.yaml`.

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
