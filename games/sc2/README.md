# StarCraft II

StarCraft II (PySC2) integration for the tmnf-ai reinforcement learning framework.

- [Installation](#installation)
  - [SC2 binary](#sc2-binary)
  - [Maps](#maps)
  - [Python dependencies](#python-dependencies)
- [Running SC2](#running-sc2)
- [Configuration](#configuration)
- [Available maps](#available-maps)
  - [Minigames (13-dim observation)](#minigames-13-dim-observation)
  - [Ladder maps (21-dim observation)](#ladder-maps-21-dim-observation)
- [Observation space](#observation-space)
  - [Minigames — 13 features](#minigames--13-features)
  - [Ladder maps — 21 features (above + 8 more)](#ladder-maps--21-features-above--8-more)
- [Action space](#action-space)
- [Reward](#reward)
- [Example commands](#example-commands)
  - [Single experiment](#single-experiment)
  - [Grid search](#grid-search)
- [Supported policies](#supported-policies)

---

## Installation

### SC2 binary

- **Linux (headless):** Download the Linux headless build from https://github.com/Blizzard/s2client-proto#linux-packages and set the `SC2PATH` environment variable to the install root.
- **Windows / macOS:** Install the regular StarCraft II client from Battle.net; PySC2 will find it at the default path (`~/StarCraftII/`).

### Maps

Download the PySC2 minigame maps from https://github.com/Blizzard/s2client-proto#downloads and unzip them into the `Maps/` folder under the SC2 install root.

### Python dependencies

```bash
poetry install --with sc2
```

---

## Running SC2

No manual game startup is required. `SC2Env` launches and stops the StarCraft II process automatically via PySC2 each time a training session starts and ends. Just run the training command directly.

---

## Configuration

| File | Purpose |
|---|---|
| `games/sc2/config/training_params.yaml` | Map name, episode settings, policy type, hyperparams |
| `games/sc2/config/reward_config.yaml` | Reward weights |

Key config parameters:

| Parameter | Default | Description |
|---|---|---|
| `map_name` | `MoveToBeacon` | SC2 map / minigame to play |
| `agent_race` | `random` | Agent race (`terran`, `zerg`, `protoss`, `random`) |
| `bot_difficulty` | `very_easy` | Bot difficulty for 1v1 ladder maps |
| `step_mul` | `8` | Game steps per agent action |
| `screen_size` | `64` | Screen resolution in pixels |
| `minimap_size` | `64` | Minimap resolution in pixels |

---

## Available maps

### Minigames (13-dim observation)

| Map name | Task |
|---|---|
| `MoveToBeacon` *(default)* | Move marine to a moving beacon |
| `CollectMineralShards` | Collect mineral shards with two marines |
| `FindAndDefeatZerglings` | Find and kill Zerglings on a larger map |
| `DefeatRoaches` | Defeat Roaches with marines |
| `DefeatZerglingsAndBanelings` | Defeat mixed Zerg forces |
| `CollectMineralsAndGas` | Economy management minigame |
| `BuildMarines` | Production-chain minigame |

### Ladder maps (21-dim observation)

Any standard 1v1 map (e.g. `Simple64`) runs the agent against a bot and uses an extended observation with 8 additional economy/map features.

---

## Observation space

### Minigames — 13 features

| Feature | Description |
|---|---|
| `minerals` | Current mineral count |
| `vespene` | Current vespene count |
| `food_used` | Supply used |
| `food_cap` | Supply cap |
| `army_count` | Total army units |
| `selected_count` | Number of currently selected units |
| `selected_avg_hp` | Average HP of selected units |
| `screen_self_count` | Friendly units visible on screen |
| `screen_enemy_count` | Enemy units visible on screen |
| `screen_self_cx` | Centroid X of friendly units on screen |
| `screen_self_cy` | Centroid Y of friendly units on screen |
| `screen_enemy_cx` | Centroid X of enemy units on screen |
| `screen_enemy_cy` | Centroid Y of enemy units on screen |

### Ladder maps — 21 features (above + 8 more)

| Feature | Description |
|---|---|
| `idle_worker_count` | Workers with no assigned task |
| `warp_gate_count` | Active warp gates |
| `larva_count` | Available larvae (Zerg) |
| `minimap_self_count` | Friendly units visible on minimap |
| `minimap_enemy_count` | Enemy units visible on minimap |
| `minimap_visible_frac` | Fraction of minimap currently visible |
| `minimap_explored_frac` | Fraction of minimap ever explored |
| `game_loop` | Current game loop tick |

---

## Action space

Continuous: `Box([0, 0, 0, 0], [5, 1, 1, 1], shape=(4,))`

| Output | Range | Description |
|---|---|---|
| `fn_idx` | [0, 5] | Integer selecting the SC2 function to call |
| `x` | [0, 1] | Normalised screen X coordinate |
| `y` | [0, 1] | Normalised screen Y coordinate |
| `queue` | [0, 1] | Whether to queue the action (0 or 1) |

Discrete policies use a 9-cell grid: the centre cell selects the whole army (`select_army`); the 8 surrounding cells issue `Move_screen` commands to the corresponding screen region.

---

## Reward

Configured in `games/sc2/config/reward_config.yaml`:

| Parameter | Value | Effect |
|---|---|---|
| `score_weight` | 1.0 | PySC2 score delta per step (primary signal for minigames) |
| `win_bonus` | 100.0 | One-time reward for winning the episode |
| `loss_penalty` | −100.0 | One-time penalty for losing |
| `step_penalty` | −0.001 | Per-step time cost |
| `idle_penalty` | 0.0 | Penalty for idle units (disabled by default) |
| `economy_weight` | 0.0 | Economy score component (disabled by default) |

---

## Example commands

### Single experiment

```bash
# MoveToBeacon (default map)
python main.py my_sc2_run --game sc2
```

To use a different map, edit `map_name` in `games/sc2/config/training_params.yaml` before running.

Results are saved to `experiments/sc2/my_sc2_run/results/`.

### Grid search

Create a YAML file modelled on `games/torcs/config/grid_search_template.yaml` with `game: sc2` and list-valued parameters, then run:

```bash
python grid_search.py my_sc2_grid.yaml --game sc2
```

---

## Supported policies

All policies in the framework work with SC2. Set `policy_type` in `games/sc2/config/training_params.yaml`.

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
