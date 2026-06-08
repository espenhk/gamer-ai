# gym_classic

Gymnasium [classic-control](https://gymnasium.farama.org/environments/classic_control/) integration. Runs five standard benchmark environments with no external binary — pure Python.

Supported environments (select via `map_name` in `training_params.yaml`):

| `map_name` | Description | Action space |
|---|---|---|
| `CartPole-v1` | Balance a pole on a cart | Discrete(2) |
| `MountainCar-v0` | Drive a car up a hill | Discrete(3) |
| `Acrobot-v1` | Swing a double pendulum | Discrete(3) |
| `Pendulum-v1` | Keep a pendulum upright | Continuous Box([-2], [2]) |
| `LunarLander-v2` | Land a spacecraft | Discrete(4) |

- [Installation](#installation)
- [Running](#running)
- [Configuration](#configuration)
- [Observation space](#observation-space)
- [Action space](#action-space)
- [Rewards](#rewards)
- [Example commands](#example-commands)
- [Supported policies](#supported-policies)

---

## Installation

```bash
pip install "gymnasium[classic_control]"
# LunarLander-v2 also needs:
pip install "gymnasium[box2d]"
```

---

## Running

No external process is required.

```bash
python main.py my_cartpole --game gym_classic
python main.py my_lunar --game gym_classic --track LunarLander-v2
```

---

## Configuration

| File | Purpose |
|---|---|
| `games/gym_classic/config/training_params.yaml` | Episode settings, env selection, policy type |
| `games/gym_classic/config/reward_config.yaml` | Reward shaping weights |

---

## Observation space

### CartPole-v1 (4 features)

| Feature | Scale | Description |
|---|---|---|
| `cart_pos` | 4.8 | Cart position on the track |
| `cart_vel` | 5.0 | Cart velocity |
| `pole_angle` | 0.418 | Pole angle in radians |
| `pole_ang_vel` | 5.0 | Pole angular velocity |

### MountainCar-v0 (2 features)

| Feature | Scale | Description |
|---|---|---|
| `position` | 1.2 | Car position [-1.2, 0.6] |
| `velocity` | 0.07 | Car velocity [-0.07, 0.07] |

### Acrobot-v1 (6 features)

| Feature | Scale | Description |
|---|---|---|
| `cos_theta1` | 1.0 | Cosine of joint 1 angle |
| `sin_theta1` | 1.0 | Sine of joint 1 angle |
| `cos_theta2` | 1.0 | Cosine of joint 2 angle |
| `sin_theta2` | 1.0 | Sine of joint 2 angle |
| `thetadot1` | 12.566 | Angular velocity of joint 1 |
| `thetadot2` | 28.274 | Angular velocity of joint 2 |

### Pendulum-v1 (3 features)

| Feature | Scale | Description |
|---|---|---|
| `cos_theta` | 1.0 | Cosine of pendulum angle |
| `sin_theta` | 1.0 | Sine of pendulum angle |
| `ang_vel` | 8.0 | Angular velocity [-8, 8] |

### LunarLander-v2 (8 features)

| Feature | Scale | Description |
|---|---|---|
| `x_pos` | 1.5 | Horizontal position |
| `y_pos` | 1.5 | Vertical position |
| `x_vel` | 2.0 | Horizontal velocity |
| `y_vel` | 2.0 | Vertical velocity |
| `angle` | 1.5 | Lander angle |
| `ang_vel` | 3.0 | Angular velocity |
| `left_contact` | 1.0 | Left leg ground contact (0 or 1) |
| `right_contact` | 1.0 | Right leg ground contact (0 or 1) |

---

## Action space

The framework exposes a 1-D continuous `Box(-1, 1)` to all policies.

For **discrete** environments (CartPole, MountainCar, Acrobot, LunarLander):
- Continuous values in [-1, 1] (evolutionary/gradient policies) are linearly
  mapped to the integer action range [0, n_actions − 1].
- Integer-valued floats from tabular/DQN policies are used directly as action
  indices.

For **Pendulum-v1** (continuous):
- The [-1, 1] output is scaled by 2 to produce a torque in [-2, 2].
- Tabular policies use an 11-point torque grid (provided as `DISCRETE_ACTIONS`).

---

## Rewards

Configured in `games/gym_classic/config/reward_config.yaml`.

| Parameter | Default | Description |
|---|---|---|
| `native_reward_scale` | 1.0 | Multiplier applied to the raw per-step reward from Gymnasium. |
| `step_penalty` | 0.0 | Optional flat per-step penalty (negative = time pressure). |

---

## Example commands

### Single experiment

```bash
# CartPole (default)
python main.py my_cartpole --game gym_classic

# Different env via --track
python main.py my_lunar --game gym_classic --track LunarLander-v2

# Grid search
python grid_search.py my_grid.yaml --game gym_classic
```

---

## Supported policies

All framework flat-observation policies work with gym_classic.  Set
`policy_type` in `games/gym_classic/config/training_params.yaml`.

| `policy_type` | Algorithm | Notes |
|---|---|---|
| `hill_climbing` | Mutate-and-keep linear | Good starting point |
| `genetic` | Population evolutionary | Default; handles noisy rewards well |
| `cmaes` | (μ/μ_w, λ)-CMA-ES | Best general-purpose evolutionary baseline |
| `neural_net` | MLP mutate-and-keep | Non-linear; configure `hidden_sizes` |
| `epsilon_greedy` | Tabular Q-learning | Discrete envs only; keep `n_bins` small |
| `ucb_q` | Tabular UCB1 Q-learning | Discrete envs only |
| `neural_dqn` | Deep Q-network | Discrete envs; sample-efficient |
| `reinforce` | Monte Carlo policy gradient | Simple gradient baseline |
| `lstm` | LSTM + evolutionary search | Useful when temporal memory matters |
| `ppo` | PPO (SB3) | Strong on-policy gradient baseline |
| `sac` | SAC (SB3) | Continuous control (Pendulum only) |
| `td3` | TD3 (SB3) | Continuous control (Pendulum only) |
