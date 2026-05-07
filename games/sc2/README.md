# StarCraft II

StarCraft II (PySC2) integration for the tmnf-ai reinforcement learning framework.

- [Installation](#installation)
  - [SC2 binary](#sc2-binary)
  - [Maps](#maps)
  - [Python dependencies](#python-dependencies)
- [Running SC2](#running-sc2)
- [Configuration](#configuration)
- [Available maps](#available-maps)
  - [Minigames](#minigames)
  - [Ladder maps](#ladder-maps)
- [Observation space](#observation-space)
  - [How a step's observation is built](#how-a-steps-observation-is-built)
  - [Choosing a preset](#choosing-a-preset)
  - [Preset: `minigame` (13 dims)](#preset-minigame-13-dims)
  - [Preset: `ladder` (43 dims)](#preset-ladder-43-dims)
  - [Preset: `rich` (80 dims)](#preset-rich-80-dims)
  - [Backward compatibility / weight migration](#backward-compatibility--weight-migration)
- [Action space](#action-space)
- [Reward](#reward)
- [Example commands](#example-commands)
  - [Single experiment](#single-experiment)
  - [Grid search](#grid-search)
- [Supported policies](#supported-policies)
  - [`sc2_genetic` — SC2GeneticPolicy](#sc2_genetic--sc2geneticpolicy)
  - [`sc2_cnn` — SC2CNNEvolutionPolicy](#sc2_cnn--sc2cnnevolutionpolicy)
- [Analytics output](#analytics-output)
  - [Files written per experiment](#files-written-per-experiment)
  - [Plot reference](#plot-reference)
  - [Markdown report sections](#markdown-report-sections)
  - [Grid-search summary](#grid-search-summary)

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

### Minigames

Default observation preset: `minigame` (13 dims).

| Map name | Task |
|---|---|
| `MoveToBeacon` *(default)* | Move marine to a moving beacon |
| `CollectMineralShards` | Collect mineral shards with two marines |
| `FindAndDefeatZerglings` | Find and kill Zerglings on a larger map |
| `DefeatRoaches` | Defeat Roaches with marines |
| `DefeatZerglingsAndBanelings` | Defeat mixed Zerg forces |
| `CollectMineralsAndGas` | Economy management minigame |
| `BuildMarines` | Production-chain minigame |

### Ladder maps

Default observation preset: `ladder` (43 dims). Any standard 1v1 map (e.g. `Simple64`, `AbyssalReef`) runs the agent against a built-in bot and uses the extended observation that adds economy, score-cumulative and minimap-visibility features.

---

## Observation space

The flat observation vector is assembled per step from PySC2's structured fields (`obs.player`, `obs.score_cumulative`, `obs.feature_screen`, `obs.feature_minimap`, `obs.feature_units`, `obs.available_actions`). Three named **presets** decide which feature blocks land in the vector — `minigame` (13 dims), `ladder` (43 dims, default for non-minigame maps), and `rich` (80 dims, opt-in superset).

### How a step's observation is built

1. `SC2Client._timestep_to_obs_info()` calls one extractor per feature group: `_player_features`, `_selected_features`, `_screen_summary_features`, `_minimap_summary_features`, `_score_features`, `_screen_hp_features`, `_topk_enemy_features`, `_per_unit_type_features`, `_quadrant_features`, `_available_actions_features`, `_last_action_features`. Each returns a `{name: float}` dict; missing PySC2 fields default to `0.0` so unit tests with mocked timesteps don't crash.
2. The extractor outputs are merged into a single name-indexed dict.
3. The flat ndarray is built by **projecting** that dict onto the active preset's `spec.names` (`np.array([feats.get(n, 0.0) for n in spec.names], dtype=np.float32)`). Unused features are computed but discarded; missing keys silently default to zero.
4. Each feature has a per-`ObsDim` **scale** (see tables below). Policies divide raw values by `spec.scales` to normalise inputs into roughly the same magnitude before the linear / MLP / LSTM layer.

This name-driven assembly is why **switching presets is just a config change** — no extractor logic depends on which preset is active, and existing weight files migrate cleanly via "missing key → 0.0".

### Choosing a preset

Set `obs_spec_preset` in `games/sc2/config/training_params.yaml` (or the per-experiment copy under `experiments/`):

```yaml
# obs_spec_preset: minigame   # 13 dims — default for the 7 minigames
# obs_spec_preset: ladder     # 43 dims — default for ladder maps
obs_spec_preset: rich         # 80 dims — research-tier superset
```

Leave it unset for the historic map-based default. Setting `rich` explicitly upgrades any map (including minigames) to the full superset.

### Preset: `minigame` (13 dims)

Compact spec for the 7 standard PySC2 minigames. Unchanged since before issue #126 so existing minigame champion weight files keep loading.

| # | Name | Scale | Description |
|---|---|---:|---|
| 0 | `minerals` | 1000 | Current mineral count |
| 1 | `vespene` | 1000 | Current vespene count |
| 2 | `food_used` | 200 | Supply used |
| 3 | `food_cap` | 200 | Supply cap |
| 4 | `army_count` | 100 | Total army units |
| 5 | `selected_count` | 50 | Number of units currently selected |
| 6 | `selected_avg_hp` | 100 | Mean HP of selected units |
| 7 | `screen_self_count` | 200 | Friendly unit pixel count on screen |
| 8 | `screen_enemy_count` | 200 | Enemy unit pixel count on screen |
| 9 | `screen_self_cx` | 64 | Friendly unit centroid x (screen) |
| 10 | `screen_self_cy` | 64 | Friendly unit centroid y (screen) |
| 11 | `screen_enemy_cx` | 64 | Enemy unit centroid x (screen) |
| 12 | `screen_enemy_cy` | 64 | Enemy unit centroid y (screen) |

**Block view:** player base (5) + selected summary (2) + screen summary (6) = 13.

### Preset: `ladder` (43 dims)

Default for non-minigame maps. Adds the supply/economy split, minimap stats, the full `score_cumulative` breakdown, screen HP summaries and top-K enemy counts.

| # | Name | Scale | Description |
|---|---|---:|---|
| 0–12 | *(13 minigame features above)* | — | — |
| 13 | `idle_worker_count` | 50 | Idle worker count |
| 14 | `warp_gate_count` | 20 | Warp gate count |
| 15 | `larva_count` | 20 | Larva count |
| 16 | `food_workers` | 200 | Supply tied up in workers |
| 17 | `food_army` | 200 | Supply tied up in army units |
| 18 | `minimap_self_count` | 200 | Friendly pixel count on minimap |
| 19 | `minimap_enemy_count` | 200 | Enemy pixel count on minimap (visible only) |
| 20 | `minimap_visible_frac` | 1 | Fraction of minimap currently visible |
| 21 | `minimap_explored_frac` | 1 | Fraction of minimap ever explored |
| 22 | `minimap_camera_x` | 64 | Camera centroid x on minimap |
| 23 | `minimap_camera_y` | 64 | Camera centroid y on minimap |
| 24 | `game_loop` | 20000 | Current game loop tick |
| 25 | `score_total` | 10000 | Cumulative environment score |
| 26 | `idle_production_time` | 10000 | Time structures spent idle (sum) |
| 27 | `idle_worker_time` | 10000 | Time workers spent idle (sum) |
| 28 | `total_value_units` | 10000 | Mineral+vespene value of all units built |
| 29 | `total_value_structures` | 10000 | Mineral+vespene value of all structures built |
| 30 | `killed_value_units` | 10000 | Mineral+vespene value of enemy units killed |
| 31 | `killed_value_structures` | 10000 | Mineral+vespene value of enemy structures killed |
| 32 | `collected_minerals` | 10000 | Cumulative minerals collected |
| 33 | `collected_vespene` | 10000 | Cumulative vespene collected |
| 34 | `collection_rate_minerals` | 2000 | Mineral collection rate (per minute) |
| 35 | `collection_rate_vespene` | 2000 | Vespene collection rate (per minute) |
| 36 | `spent_minerals` | 10000 | Cumulative minerals spent |
| 37 | `spent_vespene` | 10000 | Cumulative vespene spent |
| 38 | `screen_unit_density_mean` | 16 | Mean unit density across screen |
| 39 | `screen_self_hp_mean` | 100 | Mean friendly unit HP on screen |
| 40 | `screen_enemy_hp_mean` | 100 | Mean enemy unit HP on screen |
| 41 | `topk_enemy_within_8` | 50 | Enemy units within 8 px of friendly centroid |
| 42 | `topk_enemy_within_24` | 50 | Enemy units within 24 px of friendly centroid |

**Block view:** minigame (13) + player extras (5) + minimap summary (7) + score-cumulative (13) + screen HP (3) + top-K counts (2) = 43.

### Preset: `rich` (80 dims)

Full superset for research / ablation / CNN-conditioning experiments.

| # | Name | Scale | Description |
|---|---|---:|---|
| 0–42 | *(43 ladder features above)* | — | — |
| 43 | `unit_count_Marine` | 50 | Friendly count of unit type Marine |
| 44 | `unit_count_SCV` | 50 | Friendly count of unit type SCV |
| 45 | `unit_count_Zergling` | 50 | Friendly count of unit type Zergling |
| 46 | `unit_count_Drone` | 50 | Friendly count of unit type Drone |
| 47 | `unit_count_Probe` | 50 | Friendly count of unit type Probe |
| 48 | `unit_count_Stalker` | 50 | Friendly count of unit type Stalker |
| 49 | `unit_count_Roach` | 50 | Friendly count of unit type Roach |
| 50 | `unit_count_Mutalisk` | 50 | Friendly count of unit type Mutalisk |
| 51 | `screen_self_NE_count` | 100 | Friendly count, NE screen quadrant |
| 52 | `screen_self_NW_count` | 100 | Friendly count, NW screen quadrant |
| 53 | `screen_self_SE_count` | 100 | Friendly count, SE screen quadrant |
| 54 | `screen_self_SW_count` | 100 | Friendly count, SW screen quadrant |
| 55 | `screen_enemy_NE_count` | 100 | Enemy count, NE screen quadrant |
| 56 | `screen_enemy_NW_count` | 100 | Enemy count, NW screen quadrant |
| 57 | `screen_enemy_SE_count` | 100 | Enemy count, SE screen quadrant |
| 58 | `screen_enemy_SW_count` | 100 | Enemy count, SW screen quadrant |
| 59 | `topk_enemy_0_rel_x` | 64 | Top-1 closest enemy: rel x to friendly centroid |
| 60 | `topk_enemy_1_rel_x` | 64 | Top-2 closest enemy: rel x to friendly centroid |
| 61 | `topk_enemy_2_rel_x` | 64 | Top-3 closest enemy: rel x to friendly centroid |
| 62 | `topk_enemy_0_rel_y` | 64 | Top-1 closest enemy: rel y to friendly centroid |
| 63 | `topk_enemy_1_rel_y` | 64 | Top-2 closest enemy: rel y to friendly centroid |
| 64 | `topk_enemy_2_rel_y` | 64 | Top-3 closest enemy: rel y to friendly centroid |
| 65 | `topk_enemy_0_hp_ratio` | 1 | Top-1 closest enemy: HP / max HP |
| 66 | `topk_enemy_1_hp_ratio` | 1 | Top-2 closest enemy: HP / max HP |
| 67 | `topk_enemy_2_hp_ratio` | 1 | Top-3 closest enemy: HP / max HP |
| 68–73 | `available_fn_{0..5}` | 1 | Binary mask of currently-available SC2 function ids |
| 74–79 | `last_fn_{0..5}` | 1 | One-hot of the last *executed* fn_idx (post-fallback) |

**Block view:** ladder (43) + per-unit-type counts (8) + screen quadrants (8) + top-3 closest enemies × {rel_x, rel_y, hp_ratio} (9) + available-actions mask (6) + last-action one-hot (6) = 80.

> The unit-type lookup uses `pysc2.lib.units` lazily — when PySC2 isn't installed (e.g. CI), the eight `unit_count_*` features stay at zero and the rest of the preset still works.

### Backward compatibility / weight migration

- Existing minigame champions (13-dim) keep loading under any preset because the minigame block is the prefix of all three presets and `WeightedLinearPolicy.from_cfg` / `SC2MultiHeadLinearPolicy.from_cfg` default missing weights to zero.
- Existing pre-#126 ladder champions (21-dim) load against the new 43-dim ladder spec the same way: weights for the new feature names initialise to zero, weights for the original 21 keys load unchanged.
- Pre-#122 weight files with `spatial_{0..8}_weights` keys are silently ignored; the new `x_weights` / `y_weights` initialise to zero (sigmoid head emits the centre of the screen until training drifts the weights).

---

## Action space

Continuous: `Box([0, 0, 0, 0], [5, 1, 1, 1], shape=(4,))`

| Output | Range | Description |
|---|---|---|
| `fn_idx` | [0, 5] | Integer selecting the SC2 function to call |
| `x` | [0, 1] | Normalised screen X coordinate |
| `y` | [0, 1] | Normalised screen Y coordinate |
| `queue` | [0, 1] | Whether to queue the action (0 or 1) |

`SC2MultiHeadLinearPolicy` (the `sc2_genetic` default) emits **continuous** `(x, y)` via a sigmoid head, so it can target any pixel on the screen rather than collapsing onto a fixed grid (issue #122).

Tabular / discrete-output policies (`epsilon_greedy`, `mcts`, `neural_dqn`, `reinforce`, the discrete LSTM head) read from `DISCRETE_ACTIONS` in `games/sc2/actions.py`, which has `2 + N×N` rows (default `N = SCREEN_GRID_RESOLUTION = 8`, so 66 rows):

- Row 0: `no_op` (issue #127 — required for "stand still and shoot" tactics)
- Row 1: `select_army` (also the warmup action; auto-issued when a Move_screen is blocked, see below)
- Rows 2…65: `Move_screen` to each cell centre of an 8×8 grid covering the screen at one-cell-per-8-pixels granularity

When the policy emits a unit-targeted action (`Move_screen` / `Attack_screen` / `Harvest_Gather_screen`) but no army is selected, `SC2Client._action_to_call` substitutes `select_army` instead of silently no-op'ing (issues #121, #124). The next step then has units selected and the move actually executes.

---

## Reward

Configured in `games/sc2/config/reward_config.yaml`:

| Parameter | Default | Effect |
|---|---:|---|
| `score_weight` | 1.0 | PySC2 score delta per step (primary signal for minigames) |
| `win_bonus` | 100.0 | One-time reward for winning the episode (ladder maps) |
| `loss_penalty` | −100.0 | One-time penalty for losing (ladder maps) |
| `step_penalty` | −0.001 | Per-step time cost |
| `idle_penalty` | 0.0 | Penalty when `army_count == 0 and food_used < food_cap` (BuildMarines / economy maps) |
| `idle_bonus` | 0.0 | Per-step bonus when the agent issues `no_op` AND friendly units are within combat range of an enemy on screen (issue #127). Useful for combat minigames |
| `economy_weight` | 0.0 | Coefficient on (minerals + vespene) delta — recommended `0.001` for ladder maps |

For ladder maps (`Simple64` etc.) the recommended preset is:

```yaml
score_weight: 0.0
win_bonus: 100.0
loss_penalty: -100.0
step_penalty: -0.001
economy_weight: 0.001
```

`win_bonus` and `loss_penalty` always fire on terminal `player_outcome` regardless of `score_weight`.

The reward calculator exposes a per-component breakdown via `compute_with_components()` (issue #128/2b). `SC2Env` accumulates per-episode totals into `info["episode_reward_components"]`, which the analytics layer plots as `reward_components.png` so you can attribute episode reward to `score` / `economy` / `idle_penalty` / `idle_bonus` / `step_penalty` / `terminal` separately.

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
| `sc2_genetic` | Population of SC2MultiHeadLinearPolicy, evolutionary crossover+mutation | SC2-native multi-head individuals; separate fn_idx and spatial heads |
| `cmaes` | (μ/μ_w, λ)-CMA-ES over flat weight vector | Best general-purpose choice for linear policies |
| `neural_dqn` | Deep Q-network, experience replay, target network | Gradient-based neural training |
| `reinforce` | Monte Carlo policy gradient | Stochastic policy, simpler than DQN |
| `lstm` | LSTM + isotropic Gaussian ES | Useful when temporal memory matters |
| `sc2_cnn` | CNN (two conv layers + FC) + isotropic Gaussian ES | **Requires `screen_layers` to be non-empty**; processes raw feature-layer pixels; fundamentally different observation pipeline from every other policy |

Policy-specific hyperparameters go under `policy_params:` in `training_params.yaml`. See the root `README.md` or `games/tmnf/README.md` for full param reference.

### `sc2_genetic` — SC2GeneticPolicy

An SC2-optimised variant of the genetic algorithm. Unlike the generic `genetic` policy (which uses a single linear head per output), `sc2_genetic` maintains a population of `SC2MultiHeadLinearPolicy` individuals, each with:

- **fn_idx head** — 6×obs_dim weight matrix; `argmax` selects the SC2 function ID.
- **spatial head** — 9×obs_dim weight matrix; `argmax` selects the target grid cell.

This gives each individual **15 × obs_dim** parameters (195 for minigames, 315 for ladder maps), compared to the 4 × obs_dim from the generic `genetic` policy.

```yaml
policy_type: sc2_genetic
policy_params:
  population_size: 30      # larger search space benefits from a bigger population
  elite_k: 5
  eval_episodes: 2         # average fitness over 2 noisy episodes per individual
  mutation_scale: 0.1
  mutation_share: 0.3      # sparse mutation — mutate 30% of weights per step
```

With `population_size: 30` and `eval_episodes: 2`, total episodes per generation = `population_size × eval_episodes = 60`. At `n_sims: 50` generations that is 3,000 episodes.

Champion weights are saved in `SC2MultiHeadLinearPolicy` YAML format.

---

### `sc2_cnn` — SC2CNNEvolutionPolicy

The CNN policy is **the only SC2 policy that consumes spatial (pixel-level) observations**. All other policies receive a flat `np.ndarray`; `sc2_cnn` receives a `dict` with two keys:

- `obs["flat"]` — the standard flat obs vector (13/43/97 dims, selected by `obs_spec_preset`)
- `obs["spatial"]` — a `(C, 64, 64)` float32 array of normalised feature-layer values, where `C = len(screen_layers) + len(minimap_layers)`

This dual-stream input is the **foundational difference** relative to every other policy in the framework. It is what makes the CNN uniquely capable of detecting spatial structure — enemy formations, unit clusters, HP gradients across the map — that flat scalar summaries (enemy centroid, screen pixel counts) necessarily discard.

#### Architecture

```
spatial input (C, 64, 64)
    │
Conv2d(C → 32, 3×3, ReLU)       valid padding; output (32, 62, 62)
Conv2d(32 → 64, 3×3, ReLU)      valid padding; output (64, 60, 60)
AdaptiveAvgPool2d(4×4)           output (64, 4, 4) → flatten → (1024,)
    │
Concat with normalised flat obs (obs_dim,)
    │                            fused vector: (1024 + obs_dim,)
FC(1024 + obs_dim → 256, ReLU)   single shared trunk
    │
  ┌───┴────────┐
fn_head        spatial_head
Linear(256→6)  Linear(256→9)
argmax→fn_idx  argmax→grid cell → (x, y)
```

The network is **evolved, not gradient-trained**. Weights are updated by an isotropic Gaussian ES with the 1/5 success rule (same algorithm used by `sc2_lstm`); no backpropagation occurs at any stage. This makes the implementation purely numpy and avoids the need for a deep learning framework.

#### Parameter count

With `C` spatial channels and `obs_dim`-dimensional flat obs:

```
CONV1 (weights + biases) : 32 × C × 9 + 32
CONV2 (weights + biases) : 64 × 32 × 9 + 64    = 18,496  (fixed)
FC   (weights + biases)  : 256 × (1024 + obs_dim) + 256
fn_head                  : 6 × 256 + 6           = 1,542   (fixed)
spatial_head             : 9 × 256 + 9           = 2,313   (fixed)
```

Example totals:

| C | obs_dim (preset) | Total params |
|---|---|---|
| 2 | 13 (minigame) | ~289 K |
| 2 | 97 (rich) | ~310 K |
| 4 | 13 (minigame) | ~290 K |

The FC layer `256 × (1024 + obs_dim)` dominates (~265 K of ~289 K). This is **400× the CMA-ES linear space** (~760 params, rich obs) and **17× the LSTM space** (h=32, rich obs: ~17 K params).

#### What is fundamentally different from every other SC2 policy

| Dimension | All other SC2 policies | `sc2_cnn` |
|---|---|---|
| **Observation type** | Flat `np.ndarray` (scalar summaries) | `dict` with `"flat"` + `"spatial"` keys |
| **`screen_layers` required** | Ignored (silently set to `[]`) | **Mandatory** — must be non-empty |
| **Spatial information** | Pre-aggregated scalars (centroid, pixel count) | Raw feature-layer pixels; CNN learns its own aggregation |
| **Learning mechanism** | Gradient descent **or** ES | Isotropic ES only (no backprop) |
| **Parameter space** | 640–42 K params | ~289 K params |
| **Champion save format** | YAML (`*_weights` keys) | NumPy `.npz` (`flat`, `n_channels`, `obs_dim`, `flat_dim`) |
| **Weight loading** | `policy.load_champion(weights_file)` on a `.yaml` path | `policy.load_champion(weights_file.replace(".yaml", ".npz"))` |

Because the observations are structurally incompatible, **you cannot warm-start a `sc2_cnn` run from a champion saved by any other policy type** (and vice versa).

#### Configuration

Add `screen_layers` to your `training_params.yaml`:

```yaml
policy_type: sc2_cnn
screen_layers:
  - player_relative    # friend / foe / neutral (values 0–4) — most informative single channel
  - unit_hit_points    # per-cell HP; lets the CNN learn health-aware spatial decisions
minimap_layers: []     # optional; adds minimap channels as extra input planes
obs_spec_preset: minigame   # flat part of the obs; CNN handles all spatial information
policy_params:
  population_size: 20
  initial_sigma: 0.01
  eval_episodes: 1
```

**Choosing `screen_layers`**: any subset of the PySC2 feature-layer names is valid (e.g. `player_relative`, `selected`, `unit_hit_points`, `unit_density`, `unit_type`, `visibility_map`). Two channels is a good starting point for minigames. Adding more channels increases `C`, which slightly increases `flat_dim` (the CONV1 term grows by `32 × 9 = 288` params per extra channel) and does not change the rest of the network.

**`minimap_layers`**: minimap channels are concatenated with screen channels along the channel axis, so `n_channels = len(screen_layers) + len(minimap_layers)`. Both must have the same spatial resolution (`screen_size == minimap_size`) because the conv stack treats them identically.

#### Hyperparameter tuning

| Param | Default | Range to explore | Notes |
|---|---|---|---|
| `population_size` (λ) | 20 | 10–30 | Larger λ reduces fitness-estimate noise per generation; 20 is a reasonable default for ~289 K dims |
| `initial_sigma` | 0.01 | 0.005–0.02 | **Start small.** The 1/5 success rule adapts σ automatically, but σ > 0.05 typically collapses weights before the distribution converges. If σ decays to < 1e-5 after a few generations, try a warm restart with `--re-initialize` |
| `eval_episodes` | 1 | 1–3 | Averaging over multiple episodes reduces stochastic noise in fitness; doubles/triples episode budget per generation |
| `n_sims` | 40–60 | — | One sim = one ES generation. With λ=20 and `eval_episodes=1`, 40 sims = 800 episodes |
| `screen_layers` | `[player_relative]` | 1–4 layers | More channels → marginally larger search space; diminishing returns beyond 3–4 layers |

**Episode budget**: `n_sims × population_size × eval_episodes`. At λ=20, `n_sims=40`, `eval_episodes=1` this is 800 episodes — roughly comparable to a short `sc2_cmaes` or `sc2_lstm` run.

#### Grid search

```bash
python grid_search.py games/sc2/config/gs_sc2_cnn_template.yaml --game sc2
```

The provided template sweeps `population_size ∈ {10, 20}`, `initial_sigma ∈ {0.005, 0.01}`, and `idle_bonus ∈ {0.0, 0.5}` on DefeatRoaches with the minigame flat-obs preset and two screen channels (`player_relative` + `unit_hit_points`). That is 2 × 2 × 2 = 8 combinations, each running 40 generations.

Champion weights are saved as `<experiment_dir>/weights.npz`. Trainer state (ES mean vector and current σ) is saved as `<experiment_dir>/trainer_state.npz` and is reloaded automatically on restart.

---

## Analytics output

Every training run automatically calls `games/sc2/analytics.py::save_experiment_results` at the end of the experiment. The module exposes `SUPPORTS_THROTTLE = False` and `SUPPORTS_PATH = False` flags confirming that racing-specific plots (throttle/brake timelines, bird's-eye path traces, weight heatmaps) are intentionally excluded for SC2.

### Files written per experiment

Output directory: `experiments/sc2_<map>/<name>/results/`.

| File | Produced when | Tool |
|---|---|---|
| `probe_rewards.png` | `data.probe_results` is non-empty | `framework.analytics.plot_probe_rewards` |
| `cold_start_best_rewards.png` | `data.cold_start_restarts` is non-empty | `plot_cold_start_rewards` |
| `greedy_rewards.png` | `data.greedy_sims` is non-empty | `plot_greedy_rewards` |
| `reward_components.png` | At least one greedy sim has non-zero `reward_components` (i.e. the reward calculator returned a per-term breakdown) | `plot_reward_components` |
| `action_frequency.png` | At least one greedy sim has non-empty `action_counts` | `plot_action_frequency` |
| `obs_averages.png` | At least one greedy sim has non-empty `obs_averages` AND at least one feature value is non-zero | `plot_obs_averages` |
| `spatial_heatmap.png` | At least one greedy sim has a non-zero `xy_hist` | `plot_spatial_heatmap` |
| `outcome_breakdown.png` | At least one greedy sim has a non-None `termination_reason` | `plot_outcome_breakdown` |
| `supply_capped.png` | At least one greedy sim has a non-None `supply_capped_fraction` | `plot_supply_capped` |
| `resource_series.png` | The best greedy sim has a non-empty `resource_series` | `plot_resource_series` |
| `army_count.png` | The best greedy sim has a non-empty `army_count_series` | `plot_army_count` |
| `build_order.png` | The best greedy sim has a non-empty `build_order` | `plot_build_order` |
| `reward_trajectory.png` | Always | `plot_reward_trajectory` |
| `results.md` | Always | Markdown report stitching the plots above with summary tables |
| `experiment_data.json` | Always (by `framework.analytics.save_experiment_data_json`) | JSON dump of `ExperimentData` for cross-experiment analysis |

Probe / cold-start / greedy phases are only run by certain policy types (only `hill_climbing` runs probe + cold-start; everything else jumps straight to greedy). Their plot files are skipped when the corresponding phase is empty.

### Plot reference

#### `probe_rewards.png` — Probe phase
Bar chart showing total episode reward for each fixed-action probe (e.g. `no_op`, `select_army`, `move_centre`, `move_top_left`, `move_bottom_right`). The best-scoring probe bar is highlighted yellow; a horizontal red dashed line marks the `probe_floor` used as the cold-start gate. Establishes a reward floor before random-restart hill-climbing kicks in.

#### `cold_start_best_rewards.png` — Cold-start phase
Bar chart, one bar per random restart, showing the best episode reward in that restart's hill-climbing run. Bars are coloured **green** (beat the probe floor) or **red** (below it). The probe floor is drawn as a horizontal dashed line. Cold-start search stops as soon as one restart beats the floor.

#### `greedy_rewards.png` — Greedy / generation phase
Per-simulation reward over the greedy (or per-generation, for evolutionary policies) phase. For genetic / CMA-ES / LSTM-ES this is the fitness of the best individual in each generation. Improvements (sims that updated the champion) are marked, and a running-best curve is overlaid so you can see the staircase of progress.

#### `reward_components.png` — Per-term reward attribution (issue #128/2b)
One line per active reward component (e.g. `score`, `economy`, `idle_penalty`, `idle_bonus`, `step_penalty`, `terminal`). Each line is the per-episode sum across the greedy sim. Components that are zero in *every* sim are omitted, so this plot is silently skipped on minigame runs where only `score` and `step_penalty` contribute. The horizontal `0` line is dashed for reference. Highest-value diagnostic: tells you whether the agent is winning by collecting score, by economy growth, by surviving longer, or by hitting the `idle_bonus` shaping reward.

#### `action_frequency.png` — Action-type breakdown (issue #128/2a)
Three-panel figure:
1. **Top** — stacked bar per greedy sim showing the fraction of steps spent on each function ID (`no_op`, `select_army`, `Move_screen`, …). Reveals whether the policy is stuck preferring one action.
2. **Middle** — aggregate total-step bar chart across all greedy sims. Quick read on how diverse the policy's repertoire is.
3. **Bottom** — per-sim action entropy *H = −Σ pᵢ log₂ pᵢ*. Should increase early as the policy diversifies; collapse indicates the policy has converged to a single action.

#### `obs_averages.png` — Game-state feature averages (issue #128/2c)
Multi-panel line chart, one panel per tracked observation feature (`army_count`, `food_used`, `food_cap`, `minerals`, `vespene`, `screen_self_count`, `screen_enemy_count`). The x-axis is the greedy-sim index. Reveals things like "army was wiped out in sim 40 and never recovered" or "economy plateaued early". Improvement sims are marked with a triangle.

#### `spatial_heatmap.png` — Action-target spatial distribution (issue #128/2d)
Aggregate 8×8 heatmap of normalised `(x, y)` screen coordinates targeted by the policy across all greedy-sim steps. Displayed log-scaled (log1p) so infrequent cells remain visible. For `MoveToBeacon`, a good policy should spread coverage across the beacon area rather than targeting one fixed corner.

#### `outcome_breakdown.png` — Episode termination reasons (issue #128/2e)
Stacked-bar chart, one bar per greedy sim, showing the termination category: **win** (green), **finish** (light green), **timeout** (orange), **loss** (red), **other** (grey). Useful for ladder maps where win/loss outcomes are meaningful. On pure minigames all bars will be `finish` or `timeout`, confirming the plot is non-noisy in that setting.

#### `supply_capped.png` — Time supply-capped per sim (2f)
Bar chart showing the fraction of each greedy sim's steps where `food_used >= food_cap`. Bars are colour-coded: **green** (≤25%), **orange** (25–50%), **red** (>50%). Supply-cap time is a classic SC2 macro-management metric: a high fraction indicates the agent should be training more units rather than idling production. Improvement sims are marked with a dark triangle.

#### `resource_series.png` — Resources available over time (2g)
Line chart of `minerals + vespene` over game time (seconds) for the **best** greedy sim (last improved, or last sim if none improved). A shaded region fills under the curve; a red dashed line marks the episode average. High average resources indicate the agent is collecting but not spending — an economy bottleneck rather than a production bottleneck.

#### `army_count.png` — Army count over time (2h)
Line chart of `army_count` (total friendly army units) over game time for the **best** greedy sim. A shaded region fills under the curve. Reveals when army grows (training units) or collapses (combat losses). Flat at zero indicates the agent never builds an army.

#### `build_order.png` — Unit-build timeline (2i)
Horizontal scatter plot of unit-count increase events for the **best** greedy sim. Each row on the y-axis is a distinct unit type (e.g. Marine, SCV, Zergling); each dot marks a game-time moment when that unit type's count increased above its previous observed value (i.e. a unit of that type was built or spawned). Multiple dots per unit type are possible if several units are built over the episode. Starting units present at episode reset are excluded from the baseline so they are not recorded as "built" events. A secondary x-axis above the chart shows `mm:ss` wall-clock labels. Only unit types recognised by the rich observation preset (`Marine`, `SCV`, `Zergling`, `Drone`, `Probe`, `Stalker`, `Roach`, `Mutalisk`) appear in the plot.

#### `reward_trajectory.png` — All phases on a single timeline
Scatter plot of per-episode reward across **all** phases on one cumulative-simulation x-axis: probe (blue), cold-start (purple), greedy (orange), with vertical dotted boundaries between phases. A black step-line traces the running best-so-far. Useful for spotting how much of the gain came from each phase and whether the greedy phase plateaued early.

### Markdown report sections

`results.md` contains, in order:

1. **Title + game label** — `# Experiment: <name>` / `**Game:** StarCraft 2`
2. **Timing breakdown** — wall-clock time spent in each phase (`_timings_md`)
3. **Summary** — single paragraph with policy type, total sims, best reward, etc. (`_summary_md`)
4. **Probe Phase** *(if probes were run)* — table of `(action, reward)` pairs + the probe-rewards plot
5. **Cold-start Phase** *(if cold-start was run)* — table of `(restart, best_reward, beat_floor)` + the cold-start plot
6. **Greedy Phase** *(if any greedy sims)* — table of `(sim, reward, improved, sigma, …)` + the greedy plot, and — when populated — the reward-components, action-frequency, obs-averages, spatial-heatmap, outcome-breakdown, supply-capped, resource-series, army-count, and build-order plots
7. **Reward trajectory** — the best-episode trajectory plot

### Grid-search summary

When `grid_search.py` runs many SC2 experiments, `games/sc2/analytics.py::save_grid_summary` delegates to `framework.analytics.save_grid_summary`. Output goes to the configured summary directory (typically `experiments/sc2_<map>/grid_search_<base_name>/`):

| File | Description |
|---|---|
| `comparison_rewards.png` | One curve per grid combination overlaying best-reward-so-far over the greedy-sim axis — direct visual comparison of hyperparameter settings, ranked by final best reward |
| `comparison_task_metrics.png` | Same axis, plotted against the task-progress metric (track progress / minigame progress when available). For SC2 minigames this metric is currently zero everywhere, so the plot exists but is informative only on TMNF/TORCS |
| `summary.md` | Two ranked tables: one by task progress, one by best reward, with the varied parameters on each row plus per-experiment best-reward, improvements, finish rate, and greedy runtime |

---

## Suggested follow-up issues (non-overlapping with current open SC2 issues)

These are candidate issues discovered while reviewing the current SC2 code/docs, excluding overlap with #146, #153, #155, #156, #159, #160 (and broad integration-testing issue #133):

1. **SC2 docs drift audit + sync pass**
   - `games/sc2/README.md` still describes older observation dimensions and policy compatibility in several places.
   - Proposed issue: align all SC2 docs to the current implementation (`minigame=15`, `ladder=45`, `rich=102`; and current policy compatibility constraints).

2. **Fail-fast validation for incompatible SC2 policy types**
   - The adapter currently accepts policy types that are known to be poor fits for SC2 output semantics.
   - Proposed issue: validate `policy_type` early and raise a clear config error with migration guidance instead of allowing silent misconfiguration.

3. **Enemy-unit feature ownership filtering**
   - `enemy_count_*` features currently count every non-self owner in `feature_units`, which can include neutral entities.
   - Proposed issue: count only true enemy ownership values and add tests covering neutral/non-enemy rows.

4. **Score-cumulative extraction robustness**
   - Score features are read by positional index from `score_cumulative`.
   - Proposed issue: prefer field-name-driven extraction with safe fallback for PySC2 schema/version differences.

5. **SC2 client hot-path action-mask caching**
   - Available-action feature extraction repeatedly resolves PySC2 function metadata on step hot paths.
   - Proposed issue: cache fn-id lookups in the client and add a small benchmark/regression test around per-step overhead.
