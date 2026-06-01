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
  - [Preset: `minigame` (15 dims)](#preset-minigame-15-dims)
  - [Preset: `ladder` (46 dims)](#preset-ladder-46-dims)
  - [Preset: `rich` (103 dims)](#preset-rich-103-dims)
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
| `games/sc2/config/belief_config.yaml` | Fog-of-war belief and scouting-drive parameters (`enable_belief: true` only) |

Key config parameters:

| Parameter | Default | Description |
|---|---|---|
| `map_name` | `MoveToBeacon` | SC2 map / minigame to play |
| `agent_race` | `random` | Agent race (`terran`, `zerg`, `protoss`, `random`) |
| `bot_difficulty` | `very_easy` | Bot difficulty for 1v1 ladder maps |
| `step_mul` | `8` | Game steps per agent action |
| `screen_size` | `64` | Screen resolution in pixels |
| `minimap_size` | `64` | Minimap resolution in pixels |
| `in_game_episode_s` | `120.0` | Wall-clock seconds before truncation (use `600.0` for ladder maps) |
| `obs_spec_preset` | *(map-based)* | Override observation preset: `"minigame"` (15 dims), `"ladder"` (46 dims), `"rich"` (103 dims). Unset = map-name default. |
| `screen_layers` | `[]` | Spatial feature-layer names for `sc2_cnn` (e.g. `[player_relative, unit_hit_points]`). Ignored by all other policies. |
| `minimap_layers` | `[]` | Minimap channels for `sc2_cnn`. Concatenated with screen channels; requires `screen_size == minimap_size`. |
| `adaptive_mutation` | `true` | Apply the 1/5 success rule to adapt mutation scale during the greedy phase. |
| `patience` | `0` | Stop early if no improvement for this many consecutive sims (`0` = run all). |
| `max_apm` | *(null)* | Actions-per-minute cap via rolling token-bucket. Unset = no limit. |
| `apm_burst_s` | `2.0` | Token-bucket burst window in seconds (default allows ~2 s of burst). |
| `enable_belief` | `false` | Activate fog-of-war belief + info-gain observation extension. Adds ~192 dims for default 8×8 grid. |

---

## Available maps

### Minigames

Default observation preset: `minigame` (15 dims).

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

Default observation preset: `ladder` (46 dims). Any standard 1v1 map (e.g. `Simple64`, `AbyssalReef`) runs the agent against a built-in bot and uses the extended observation that adds economy, score-cumulative and minimap-visibility features.

---

## Observation space

The flat observation vector is assembled per step from PySC2's structured fields (`obs.player`, `obs.score_cumulative`, `obs.feature_screen`, `obs.feature_minimap`, `obs.feature_units`, `obs.available_actions`). Three named **presets** decide which feature blocks land in the vector — `minigame` (15 dims), `ladder` (46 dims, default for non-minigame maps), and `rich` (103 dims, opt-in superset).

### How a step's observation is built

1. `SC2Client._timestep_to_obs_info()` calls one extractor per feature group: `_player_features`, `_selected_features`, `_screen_summary_features`, `_minimap_summary_features`, `_score_features`, `_screen_hp_features`, `_topk_enemy_features`, `_per_unit_type_features`, `_quadrant_features`, `_available_actions_features`, `_last_action_features`. Each returns a `{name: float}` dict; missing PySC2 fields default to `0.0` so unit tests with mocked timesteps don't crash.
2. The extractor outputs are merged into a single name-indexed dict.
3. The flat ndarray is built by **projecting** that dict onto the active preset's `spec.names` (`np.array([feats.get(n, 0.0) for n in spec.names], dtype=np.float32)`). Unused features are computed but discarded; missing keys silently default to zero.
4. Each feature has a per-`ObsDim` **scale** (see tables below). Policies divide raw values by `spec.scales` to normalise inputs into roughly the same magnitude before the linear / MLP / LSTM layer.

This name-driven assembly is why **switching presets is just a config change** — no extractor logic depends on which preset is active, and existing weight files migrate cleanly via "missing key → 0.0".

### Choosing a preset

Set `obs_spec_preset` in `games/sc2/config/training_params.yaml` (or the per-experiment copy under `experiments/`):

```yaml
# obs_spec_preset: minigame   # 15 dims — default for the 7 minigames
# obs_spec_preset: ladder     # 46 dims — default for ladder maps
obs_spec_preset: rich         # 103 dims — research-tier superset
```

Leave it unset for the historic map-based default. Setting `rich` explicitly upgrades any map (including minigames) to the full superset.

### Preset: `minigame` (15 dims)

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
| 13 | `minimap_enemy_cx` | 64 | Enemy (beacon) centroid x on minimap |
| 14 | `minimap_enemy_cy` | 64 | Enemy (beacon) centroid y on minimap |

**Block view:** player base (5) + selected summary (2) + screen summary (6) + minimap beacon (2) = 15.

### Preset: `ladder` (46 dims)

Default for non-minigame maps. Adds the supply/economy split, minimap stats, the full `score_cumulative` breakdown, screen HP summaries, top-K enemy counts, and an alert scalar.

| # | Name | Scale | Description |
|---|---|---:|---|
| 0–14 | *(15 minigame features above)* | — | — |
| 15 | `idle_worker_count` | 50 | Idle worker count |
| 16 | `warp_gate_count` | 20 | Warp gate count |
| 17 | `larva_count` | 20 | Larva count |
| 18 | `food_workers` | 200 | Supply tied up in workers |
| 19 | `food_army` | 200 | Supply tied up in army units |
| 20 | `minimap_self_count` | 200 | Friendly pixel count on minimap |
| 21 | `minimap_enemy_count` | 200 | Enemy pixel count on minimap (visible only) |
| 22 | `minimap_visible_frac` | 1 | Fraction of minimap currently visible |
| 23 | `minimap_explored_frac` | 1 | Fraction of minimap ever explored |
| 24 | `minimap_camera_x` | 64 | Camera centroid x on minimap |
| 25 | `minimap_camera_y` | 64 | Camera centroid y on minimap |
| 26 | `game_loop` | 20000 | Current game loop tick |
| 27 | `score_total` | 10000 | Cumulative environment score |
| 28 | `idle_production_time` | 10000 | Time structures spent idle (sum) |
| 29 | `idle_worker_time` | 10000 | Time workers spent idle (sum) |
| 30 | `total_value_units` | 10000 | Mineral+vespene value of all units built |
| 31 | `total_value_structures` | 10000 | Mineral+vespene value of all structures built |
| 32 | `killed_value_units` | 10000 | Mineral+vespene value of enemy units killed |
| 33 | `killed_value_structures` | 10000 | Mineral+vespene value of enemy structures killed |
| 34 | `collected_minerals` | 10000 | Cumulative minerals collected |
| 35 | `collected_vespene` | 10000 | Cumulative vespene collected |
| 36 | `collection_rate_minerals` | 2000 | Mineral collection rate (per minute) |
| 37 | `collection_rate_vespene` | 2000 | Vespene collection rate (per minute) |
| 38 | `spent_minerals` | 10000 | Cumulative minerals spent |
| 39 | `spent_vespene` | 10000 | Cumulative vespene spent |
| 40 | `screen_unit_density_mean` | 16 | Mean unit density across screen |
| 41 | `screen_self_hp_mean` | 100 | Mean friendly unit HP on screen |
| 42 | `screen_enemy_hp_mean` | 100 | Mean enemy unit HP on screen |
| 43 | `topk_enemy_within_8` | 50 | Enemy units within 8 px of friendly centroid |
| 44 | `topk_enemy_within_24` | 50 | Enemy units within 24 px of friendly centroid |
| 45 | `alert_count` | 2 | Number of active alerts (0–2); >0 means under major attack |

**Block view:** minigame (15) + player extras (5) + minimap summary (7) + score-cumulative (13) + screen HP (3) + top-K counts (2) + alerts (1) = 46.

### Preset: `rich` (103 dims)

Full superset for research / ablation / CNN-conditioning experiments.

| # | Name | Scale | Description |
|---|---|---:|---|
| 0–45 | *(46 ladder features above)* | — | — |
| 46 | `unit_count_Marine` | 50 | Friendly count of unit type Marine |
| 47 | `unit_count_SCV` | 50 | Friendly count of unit type SCV |
| 48 | `unit_count_Zergling` | 50 | Friendly count of unit type Zergling |
| 49 | `unit_count_Drone` | 50 | Friendly count of unit type Drone |
| 50 | `unit_count_Probe` | 50 | Friendly count of unit type Probe |
| 51 | `unit_count_Stalker` | 50 | Friendly count of unit type Stalker |
| 52 | `unit_count_Roach` | 50 | Friendly count of unit type Roach |
| 53 | `unit_count_Mutalisk` | 50 | Friendly count of unit type Mutalisk |
| 54 | `screen_self_NE_count` | 100 | Friendly count, NE screen quadrant |
| 55 | `screen_self_NW_count` | 100 | Friendly count, NW screen quadrant |
| 56 | `screen_self_SE_count` | 100 | Friendly count, SE screen quadrant |
| 57 | `screen_self_SW_count` | 100 | Friendly count, SW screen quadrant |
| 58 | `screen_enemy_NE_count` | 100 | Enemy count, NE screen quadrant |
| 59 | `screen_enemy_NW_count` | 100 | Enemy count, NW screen quadrant |
| 60 | `screen_enemy_SE_count` | 100 | Enemy count, SE screen quadrant |
| 61 | `screen_enemy_SW_count` | 100 | Enemy count, SW screen quadrant |
| 62 | `topk_enemy_0_rel_x` | 64 | Top-1 closest enemy: rel x to friendly centroid |
| 63 | `topk_enemy_1_rel_x` | 64 | Top-2 closest enemy: rel x to friendly centroid |
| 64 | `topk_enemy_2_rel_x` | 64 | Top-3 closest enemy: rel x to friendly centroid |
| 65 | `topk_enemy_0_rel_y` | 64 | Top-1 closest enemy: rel y to friendly centroid |
| 66 | `topk_enemy_1_rel_y` | 64 | Top-2 closest enemy: rel y to friendly centroid |
| 67 | `topk_enemy_2_rel_y` | 64 | Top-3 closest enemy: rel y to friendly centroid |
| 68 | `topk_enemy_0_hp_ratio` | 1 | Top-1 closest enemy: HP / max HP |
| 69 | `topk_enemy_1_hp_ratio` | 1 | Top-2 closest enemy: HP / max HP |
| 70 | `topk_enemy_2_hp_ratio` | 1 | Top-3 closest enemy: HP / max HP |
| 71–76 | `available_fn_{0..5}` | 1 | Binary mask of currently-available SC2 function ids |
| 77–82 | `last_fn_{0..5}` | 1 | One-hot of the last *executed* fn_idx (post-fallback) |
| 83 | `enemy_count_Marine` | 50 | Enemy count of unit type Marine |
| 84 | `enemy_count_SCV` | 50 | Enemy count of unit type SCV |
| 85 | `enemy_count_Zergling` | 50 | Enemy count of unit type Zergling |
| 86 | `enemy_count_Drone` | 50 | Enemy count of unit type Drone |
| 87 | `enemy_count_Probe` | 50 | Enemy count of unit type Probe |
| 88 | `enemy_count_Stalker` | 50 | Enemy count of unit type Stalker |
| 89 | `enemy_count_Roach` | 50 | Enemy count of unit type Roach |
| 90 | `enemy_count_Mutalisk` | 50 | Enemy count of unit type Mutalisk |
| 91 | `screen_self_shield_mean` | 100 | Mean shield of friendly units on screen |
| 92 | `screen_enemy_shield_mean` | 100 | Mean shield of enemy units on screen |
| 93 | `screen_self_energy_mean` | 200 | Mean energy of friendly units on screen |
| 94 | `minimap_creep_frac` | 1 | Fraction of minimap covered by Zerg creep |
| 95 | `upgrade_count` | 30 | Number of completed upgrades |
| 96 | `build_queue_size` | 20 | Units/structures currently under construction |
| 97 | `cargo_count` | 10 | Units currently in transports |
| 98 | `selected_avg_shields` | 100 | Mean shield of currently selected units |
| 99 | `selected_avg_energy` | 200 | Mean energy of currently selected units |
| 100 | `screen_visibility_frac` | 1 | Fraction of screen tiles currently visible |
| 101 | `screen_unit_density_aa_mean` | 16 | Mean anti-air unit density across screen |
| 102 | `self_weapon_cooldown_mean` | 50 | Mean weapon cooldown for friendly units (0 = all ready) |

**Block view:** ladder (46) + per-unit-type counts (8) + screen quadrants (8) + top-3 closest enemies × {rel_x, rel_y, hp_ratio} (9) + available-actions mask (6) + last-action one-hot (6) + enemy unit-type counts (8) + shield/energy means (3) + creep (1) + economy pipeline (3) + selected extras (2) + screen visibility (1) + anti-air density (1) + weapon cooldown (1) = 103.

> The unit-type lookup uses `pysc2.lib.units` lazily — when PySC2 isn't installed (e.g. CI), the eight `unit_count_*` and `enemy_count_*` features stay at zero and the rest of the preset still works.

### Backward compatibility / weight migration

- Existing minigame champions (15-dim) keep loading under any preset because the minigame block is the prefix of all three presets and `WeightedLinearPolicy.from_cfg` / `SC2MultiHeadLinearPolicy.from_cfg` default missing weights to zero.
- Existing pre-#126 ladder champions (21-dim) load against the new 46-dim ladder spec the same way: weights for the new feature names initialise to zero, weights for the original 21 keys load unchanged.
- Pre-#122 weight files with `spatial_{0..8}_weights` keys are silently ignored; the new `x_weights` / `y_weights` initialise to zero (sigmoid head emits the centre of the screen until training drifts the weights).

---

## APM limiting

Set `max_apm` in `training_params.yaml` to constrain the agent to at most that many real actions per minute, using a rolling token-bucket:

```yaml
max_apm: 300      # e.g. 300 APM ≈ human intermediate level
apm_burst_s: 2.0  # allow bursts up to 2 seconds of budget
```

`no_op` actions (fn_idx 0) are always free and never consume a token — they are not counted in real SC2 APM either. When the budget is exhausted, the intended action is replaced by `no_op`. Leave `max_apm` unset (null) for no limit.

---

## Fog-of-war belief system

Set `enable_belief: true` in `training_params.yaml` to activate the belief + info-gain observation extension:

```yaml
enable_belief: true
```

This adds **~192 extra dims** to the chosen preset (value + confidence per 8×8 minimap cell + staleness per cell, by default). The belief module tracks last-known enemy supply per region and decays confidence exponentially over time. A small intrinsic scouting reward (`scout_drive_weight` in `belief_config.yaml`) incentivises the agent to visit stale regions.

Belief parameters live in `games/sc2/config/belief_config.yaml`:

| Key | Default | Description |
|---|---|---|
| `region_grid` | `[8, 8]` | Minimap partition (rows × cols); determines belief vector length |
| `decay_tau` | `30.0` | Confidence half-life in seconds |
| `scout_drive_weight` | `0.1` | Coefficient on intrinsic per-step scouting reward |
| `scout_horizon_s` | `60.0` | Seconds until a region reaches full staleness |
| `stale_threshold` | `0.5` | Confidence below which re-discovery is rewarded |
| `never_seen_bonus` | `2.0` | Multiplier for first-time region discovery |

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
- Row 1: `select_army` (also the warmup action; the deferred-action queue described below auto-emits it when a chosen action requires any unit selected and none is)
- Rows 2…65: `Move_screen` to each cell centre of an 8×8 grid covering the screen at one-cell-per-8-pixels granularity

**Tech-tree-gated mask + deferred-action queue (issue #346).** Every fn_idx has a `Preconditions` record in [`games/sc2/tech_tree.py`](tech_tree.py) with (a) required buildings (OR-sets in DNF form, e.g. `Spire` needs `Lair OR Hive`), (b) required upgrades/research, (c) required selection type (none, any unit, or one of a named set of unit-types). `info["available_fn_ids"]` is now race ∩ PySC2 `available_actions` ∩ tech-tree ∩ selection-filtered, so an action like `Build_FusionCore_screen` is masked off until a Starport actually exists on the map. PySC2's `available_actions` is documented to check unit/ability requirements but **not** building prerequisites or arguments (PySC2 issues [#163](https://github.com/google-deepmind/pysc2/issues/163), [#291](https://github.com/google-deepmind/pysc2/issues/291)) — this hardcoded table closes that gap. Previous race-only filtering (PR #311) and reactive `select_army` substitution (PRs #307, #322) have been removed; their job is now done by the mask plus the queue.

When the policy emits an action whose selection requirement is unmet, `SC2Client._resolve_action()` emits the right `select_*` *this* tick and stores the original action in a 1-slot deferred-action FIFO so it replays *next* tick. The selector prefers `select_idle_worker` for Build actions, but falls back to `select_point` on a visible worker when no worker is idle (every SCV is mining/building) — issue #346 specifically required workers that aren't idle to still be selectable. For Train_* and Morph_* the resolver `select_point`s the producing building (Barracks, Stargate, Lair, …) cached from `feature_units` each step. A 1-step `select_army` warmup is still issued at the start of every episode (via `SC2Adapter.build_warmup`).

When DEBUG logging is enabled, `SC2Client` also dumps a readable game-state snapshot every ~10 s of wall-clock time: currently-owned friendly units (counts), currently-owned buildings, completed upgrades/research, currently-selected unit type, and the available action set with the unit/building each action needs selected. This makes it easy to verify the mask is doing the right thing without tailing 22.4 obs/s of raw step logs.

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
| `idle_bonus` | 0.0 | Per-step bonus when the agent issues `no_op` AND friendly units are within effective attack range of an enemy (unit-aware when `feature_units` is available). The gate uses a 5% inside-range margin to avoid edge-of-range stalling (issue #127). |
| `move_exploration_bonus` | 0.01 | Bonus for `Move_screen` targets that are at least `_MOVE_MIN_MEANINGFUL_FRAC` (6/64 ≈ 9% of screen) away from the previous move target. Sub-threshold moves receive no bonus, preventing stutter-stepping. |
| `move_repeat_penalty` | −0.02 | Penalty when a `Move_screen` command is less than `_MOVE_MIN_MEANINGFUL_FRAC` from the previous move target (covers both exact repeats and tiny stutter steps) |
| `move_self_penalty` | −0.01 | Penalty for issuing `Move_screen` to the friendly-unit centroid (discourages "move where we already are") |
| `attack_move_bonus` | 0.0 | Per-step bonus when the agent issues `Attack_screen` with the target on empty ground while enemies are visible (A-move). Opt-in. |
| `click_attack_bonus` | 0.0 | Per-step bonus when the agent issues `Attack_screen` directly on a visible enemy unit. Subject to `click_attack_cooldown_steps`. Opt-in. |
| `click_attack_cooldown_steps` | 8 | Minimum env steps between rewarded target switches for `click_attack_bonus`. |
| `economy_weight` | 0.0 | Coefficient on (minerals + vespene) delta — recommended `0.001` for ladder maps |
| `resource_banking_penalty` | 0.0 | Per-step penalty proportional to excess minerals above `mineral_banking_threshold` or vespene above `gas_banking_threshold`. Nudges the agent to spend banked resources. Recommended range: `-0.0001` to `-0.001`. Opt-in. |
| `mineral_banking_threshold` | 300.0 | Minerals above this level count as "banked" for `resource_banking_penalty`. |
| `gas_banking_threshold` | 200.0 | Vespene above this level counts as "banked" for `resource_banking_penalty`. |
| `small_selection_bonus` | 0.0 | Per-step bonus for unit-targeted commands when the active selection is one unit or under 50% of visible friendly units. Opt-in. |

`idle_bonus` uses PySC2 unit IDs plus a curated unit-range table in `games/sc2/client.py`.
PySC2 does not expose weapon ranges directly in `pysc2.lib.units`; to update ranges, use
Blizzard `s2client-proto` unit weapon data (`Weapon.range`) and/or Liquipedia unit stats:
`s2clientprotocol/data.proto` and `Unit_Statistics_(Legacy_of_the_Void)`.

For ladder maps (`Simple64` etc.) the recommended preset is:

```yaml
score_weight: 0.0
win_bonus: 100.0
loss_penalty: -100.0
step_penalty: -0.001
economy_weight: 0.001
```

`win_bonus` and `loss_penalty` always fire on terminal `player_outcome` regardless of `score_weight`.

The reward calculator exposes a per-component breakdown via `compute_with_components()` (issue #128/2b). `SC2Env` accumulates per-episode totals into `info["episode_reward_components"]`, which the analytics layer plots as `reward_components.png` so you can attribute episode reward to `score` / `economy` / `idle_penalty` / `idle_bonus` / `move_exploration` / `move_repeat_penalty` / `move_self_penalty` / `small_selection` / `step_penalty` / `terminal` separately.

---

## Example commands

### Single experiment

```bash
# MoveToBeacon (default map)
python main.py my_sc2_run --game sc2
```

To use a different map, edit `map_name` in `games/sc2/config/training_params.yaml` before running.

Results are saved to `experiments/sc2_<map>/my_sc2_run/results/`.

### Interactive play (human vs. trained agent)

```bash
python main.py my_sc2_run --game sc2 --play
```

Loads the champion policy from a completed experiment and launches a two-player PySC2 session. You control one side via the normal SC2 UI; the trained agent drives the other. No weight updates occur. An episode summary (score, game loop, outcome) is printed at game end.

### Grid search

Create a YAML file modelled on `games/torcs/config/grid_search_template.yaml` with `game: sc2` and list-valued parameters, then run:

```bash
python grid_search.py my_sc2_grid.yaml --game sc2
```

To fan out a grid search across multiple SC2 instances on the same machine,
use distributed mode plus local worker auto-spawn:

```bash
python grid_search.py my_sc2_grid.yaml --game sc2 --distribute --local-workers 4
```

This starts one coordinator and four local `distributed.worker` subprocesses.
Each worker runs one experiment at a time, so up to four combinations run in
parallel (CPU/RAM permitting).

---

## Supported policies

SC2-compatible policies are listed below. The framework's generic `hill_climbing`, `neural_net`, and base `genetic` policies are **not** compatible with SC2: their output encoding clips `fn_idx` to `[−1, 1]` and thresholds `x`/`y` to binary, which is unsuitable for the SC2 action space — use SC2-specific equivalents such as `sc2_neural_net`, `sc2_genetic`, or `sc2_cmaes`.

| `policy_type` | Algorithm | Notes |
|---|---|---|
| `sc2_genetic` | Population of `SC2MultiHeadLinearPolicy`, evolutionary crossover+mutation | **Recommended default.** SC2-native multi-head individuals; separate fn_idx (6×obs_dim) and sigmoid spatial (2×obs_dim) heads |
| `sc2_neural_net` | TMNF-style hill-climbing MLP | Uses `hidden_sizes` list (e.g. `[16, 64, 64, 16]`); outputs SC2-native `[fn_idx, x, y, queue]` |
| `sc2_reinforce` | Two-head REINFORCE MLP (softmax fn + sigmoid spatial) | Gradient-trained; recommended over legacy `reinforce` for SC2 |
| `sc2_cmaes` | (μ/μ_w, λ)-CMA-ES over `SC2MultiHeadLinearPolicy` flat weights | Recommended over legacy `cmaes` for SC2 |
| `sc2_lstm` | LSTM with SC2-native action encoding, trained by isotropic ES | Recommended over legacy `lstm` for SC2 |
| `sc2_cnn` | CNN (two conv layers + FC) + isotropic ES | **Requires `screen_layers` to be non-empty**; processes raw feature-layer pixels; fundamentally different observation pipeline from every other policy |
| `epsilon_greedy` | Tabular Q-learning, ε-greedy | Classical RL baseline; selects from `DISCRETE_ACTIONS` |
| `mcts` | UCT-style Q-learning (UCB1 exploration) | More systematic exploration than ε-greedy |
| `cmaes` | (μ/μ_w, λ)-CMA-ES over `SC2LinearPolicy` weights *(legacy)* | Prefer `sc2_cmaes` |
| `neural_dqn` | Deep Q-network, experience replay *(legacy)* | Selects from `DISCRETE_ACTIONS`; prefer `sc2_reinforce` |
| `reinforce` | Monte Carlo policy gradient *(legacy)* | Selects from `DISCRETE_ACTIONS`; prefer `sc2_reinforce` |
| `lstm` | LSTM + isotropic ES *(legacy)* | Prefer `sc2_lstm` |

Policy-specific hyperparameters go under `policy_params:` in `training_params.yaml`. See the root `README.md` or `games/tmnf/README.md` for full param reference.

### `sc2_genetic` — SC2GeneticPolicy

An SC2-optimised variant of the genetic algorithm. Unlike the generic `genetic` policy (which uses a single linear head per output), `sc2_genetic` maintains a population of `SC2MultiHeadLinearPolicy` individuals, each with:

- **fn_idx head** — 6×obs_dim weight matrix; `argmax` selects the SC2 function ID.
- **spatial head** — 2×obs_dim weight matrix; sigmoid produces continuous `(x, y) ∈ [0, 1]²` screen coordinates (issue #122).

This gives each individual **(6+2) × obs_dim = 8 × obs_dim** parameters (120 for minigames, 368 for ladder maps), compared to the 4 × obs_dim from the generic `genetic` policy.

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

### `sc2_reinforce` — SC2REINFORCEPolicy

Two-head REINFORCE (Monte Carlo policy gradient) for SC2. A **shared MLP trunk** (default `[128, 64]`) feeds two independent output heads:

- **fn_head** — 6 logits, softmax → `fn_idx ∈ {0…5}`; unavailable function IDs are masked to `−∞` before sampling.
- **spatial_head** — 2 logits, sigmoid → continuous `(x, y) ∈ [0, 1]²` screen coordinates.

Both heads are trained jointly by REINFORCE with discounted returns and an optional running-mean baseline.

```yaml
policy_type: sc2_reinforce
policy_params:
  hidden_sizes: [128, 64]   # shared trunk widths
  learning_rate: 0.0003
  gamma: 0.995
  entropy_coeff: 0.05       # entropy regularisation for both heads
  baseline: running_mean    # "running_mean" or "none"
```

Trainer state (network weights + Adam moments) is saved to `trainer_state.npz` and reloaded on restart.

---

### `sc2_neural_net` — SC2NeuralNetPolicy

TMNF-style MLP trained with the same mutation-and-keep hill-climbing loop as `neural_net`, but with SC2 action encoding:

- `fn_idx`: sigmoid-scaled to `[0, N_FUNCTION_IDS-1]` and snapped to an available function ID.
- `x`, `y`: sigmoid outputs in `[0, 1]`.
- `queue`: thresholded to `{0, 1}`.

```yaml
policy_type: sc2_neural_net
policy_params:
  hidden_sizes: [16, 64, 64, 16]
```

Grid-search template for very large networks:

```bash
python grid_search.py games/sc2/config/gs_sc2_neural_net_template.yaml --game sc2
```

---

### `sc2_cmaes` — SC2CMAESPolicy

(μ/μ_w, λ)-CMA-ES over the concatenated flat weight vector of `SC2MultiHeadLinearPolicy` — recommended over the legacy `cmaes` type for SC2.

```yaml
policy_type: sc2_cmaes
policy_params:
  population_size: 30   # λ
  initial_sigma: 0.5    # starting step size (adapts automatically)
  eval_episodes: 2
```

The distribution parameters (mean, σ, covariance) are saved to `trainer_state.npz`.

---

### `sc2_lstm` — SC2LSTMEvolutionPolicy

LSTM recurrent policy with SC2-native action encoding, trained by isotropic Gaussian ES (1/5 success rule). The LSTM hidden state is optionally reset between episodes (`reset_on_episode: true`, default).

```yaml
policy_type: sc2_lstm
policy_params:
  hidden_size: 64        # LSTM hidden / cell state dim
  population_size: 20    # λ
  initial_sigma: 0.03    # smaller than CMAESPolicy because the LSTM weight space is larger
  reset_on_episode: true
```

Saved champion weights are incompatible across different `hidden_size` or obs_spec values — use `--re-initialize` when changing either.

---

### `sc2_cnn` — SC2CNNEvolutionPolicy

The CNN policy is **the only SC2 policy that consumes spatial (pixel-level) observations**. All other policies receive a flat `np.ndarray`; `sc2_cnn` receives a `dict` with two keys:

- `obs["flat"]` — the standard flat obs vector (15/46/103 dims, selected by `obs_spec_preset`)
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
| 2 | 15 (minigame) | ~289 K |
| 2 | 103 (rich) | ~310 K |
| 4 | 15 (minigame) | ~290 K |

The FC layer `256 × (1024 + obs_dim)` dominates (~265 K of ~289 K). This is **~350× the CMA-ES linear space** (~824 params, rich obs) and **17× the LSTM space** (h=32, rich obs: ~17 K params).

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
| `skipped_frames.png` | At least one greedy sim has non-None `skipped_frames` telemetry | `plot_skipped_frames` |
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
One line per active reward component (e.g. `score`, `economy`, `idle_penalty`, `idle_bonus`, `move_exploration`, `move_repeat_penalty`, `move_self_penalty`, `step_penalty`, `terminal`). Each line is the per-episode sum across the greedy sim. Components that are zero in *every* sim are omitted, so this plot is silently skipped on minigame runs where only `score` and `step_penalty` contribute. The horizontal `0` line is dashed for reference. Highest-value diagnostic: tells you whether the agent is winning by collecting score, by economy growth, by surviving longer, or by shaping terms such as `idle_bonus` / movement exploration.

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

#### `skipped_frames.png` — Skipped frames per sim (2f)
Bar chart of per-sim skipped-frame count. A skipped frame means the SC2 `game_loop` advanced by more than `step_mul` since the previous action (the action arrived too late for those extra loops). Bars are green at `0` and red above `0`. Improvement sims are marked with a dark triangle so you can compare whether larger models trade quality for responsiveness.

#### `supply_capped.png` — Time supply-capped per sim (2g)
Bar chart showing the fraction of each greedy sim's steps where `food_used >= food_cap`. Bars are colour-coded: **green** (≤25%), **orange** (25–50%), **red** (>50%). Supply-cap time is a classic SC2 macro-management metric: a high fraction indicates the agent should be training more units rather than idling production. Improvement sims are marked with a dark triangle.

#### `resource_series.png` — Resources available over time (2h)
Line chart of `minerals + vespene` over game time (seconds) for the **best** greedy sim (last improved, or last sim if none improved). A shaded region fills under the curve; a red dashed line marks the episode average. High average resources indicate the agent is collecting but not spending — an economy bottleneck rather than a production bottleneck.

#### `army_count.png` — Army count over time (2i)
Line chart of `army_count` (total friendly army units) over game time for the **best** greedy sim. A shaded region fills under the curve. Reveals when army grows (training units) or collapses (combat losses). Flat at zero indicates the agent never builds an army.

#### `build_order.png` — Unit-build timeline (2j)
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
6. **Greedy Phase** *(if any greedy sims)* — table of `(sim, reward, improved, sigma, …)` + the greedy plot, and — when populated — the reward-components, action-frequency, obs-averages, spatial-heatmap, outcome-breakdown, skipped-frames, supply-capped, resource-series, army-count, and build-order plots
7. **Reward trajectory** — the best-episode trajectory plot

### Grid-search summary

When `grid_search.py` runs many SC2 experiments, `games/sc2/analytics.py::save_grid_summary` delegates to `framework.analytics.save_grid_summary` and then appends SC2-specific cross-run plots. Output goes to the configured summary directory (typically `experiments/sc2_<map>/grid_search_<base_name>/`):

| File | Description |
|---|---|
| `comparison_rewards.png` | One curve per grid combination overlaying best-reward-so-far over the greedy-sim axis — direct visual comparison of hyperparameter settings, ranked by final best reward |
| `comparison_task_metrics.png` | Same axis, plotted against the task-progress metric (track progress / minigame progress when available). For SC2 minigames this metric is currently zero everywhere, so the plot exists but is informative only on TMNF/TORCS |
| `comparison_action_entropy.png` | Cross-run bar chart of mean per-sim action entropy (bits), summarising how diverse each run's action mix was |
| `comparison_outcomes.png` | Stacked cross-run outcome fractions (win / finish / timeout / loss / other) over greedy sims per experiment |
| `comparison_skipped_frames.png` | Cross-run bar chart of mean skipped frames per greedy sim (lower is better responsiveness) |
| `comparison_supply_capped.png` | Cross-run bar chart of average fraction of time each experiment spent supply-capped |
| `comparison_spatial_heatmap.png` | Aggregate spatial-target heatmap built from all greedy sims across all runs (SC2 screen-target concentration overview) |
| `summary.md` | Framework tables plus an appended **SC2-specific cross-run charts** section embedding the SC2 summary plots above |
