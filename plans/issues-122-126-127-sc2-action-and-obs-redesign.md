# Plan: SC2 action/observation redesign (issues #122, #126, #127)

Branch: `claude/plan-issues-122-126-127-zbhRr`

The three issues all live in `games/sc2/` and touch overlapping files
(`actions.py`, `obs_spec.py`, `policies.py`, `sc2_policies.py`, `client.py`,
`env.py`).  Solving them in one coordinated series avoids two churn cycles
through the same code paths and makes weight-file migration a single event
rather than two.

---

## 1. Issue #122 — units only move to the outer (coarse) box

### Root cause

`games/sc2/actions.py` defines `DISCRETE_ACTIONS` as a 3×3 grid hard-coded at
normalised coords `{0.20, 0.50, 0.80}`:

```python
_GRID = [
    (0.20, 0.20), (0.50, 0.20), (0.80, 0.20),
    (0.20, 0.50), (0.50, 0.50), (0.80, 0.50),
    (0.20, 0.80), (0.50, 0.80), (0.80, 0.80),
]
```

With `screen_size=64`, `int(0.20*63)=12`, `int(0.50*63)=31`, `int(0.80*63)=50`
— exactly the `(12|31|50, 12|31|50)` coordinates seen in the debug log.

The active genetic policy (`SC2MultiHeadLinearPolicy` in
`games/sc2/sc2_policies.py`, registered as `sc2_genetic` via
`adapter.py:110`) picks the spatial cell by `argmax` over its 9-output
spatial head, so its target can only ever land on one of these 9 fixed
points.  Cell 4 (centre) is overloaded as `select_army`, which is why the
log shows 8 outer cells plus periodic `select_army` interjections.

The framework discrete-action policies (`NeuralDQNPolicy`, `REINFORCEPolicy`,
the discrete branch of `LSTMEvolutionPolicy`) in `games/sc2/policies.py`
likewise select rows from `DISCRETE_ACTIONS` and inherit the same coarseness.

`SC2LinearPolicy` (sigmoid-encoded continuous head, `policies.py:68`) is the
only existing policy that can emit arbitrary screen coordinates — but it is
only used by the CMA-ES path, not by `sc2_genetic` (the default).

### Fix

Two parts, both required:

1. **Replace the 9-cell argmax spatial head in `SC2MultiHeadLinearPolicy`
   with a continuous (x, y) head.**  Use the same sigmoid encoding as
   `SC2LinearPolicy`:

   ```text
   x = σ(w_x · norm_obs)      ∈ [0, 1]
   y = σ(w_y · norm_obs)      ∈ [0, 1]
   ```

   This collapses the spatial head from 9 rows × obs_dim down to 2 rows ×
   obs_dim, but lets the policy target any pixel.  Linear-in-obs sigmoid
   does NOT mean the output is constant — `norm_obs` includes friendly /
   enemy centroids and counts, so the policy can learn e.g. "drift x toward
   the enemy centroid x" naturally.  Keep the fn_idx head argmax-over-N as
   today (it is a discrete choice).

2. **Expand `DISCRETE_ACTIONS` for the discrete-action policies** to a finer
   grid.  Move from 3×3 to a configurable resolution (default 8×8 = 64
   cells, matching the 64-pixel screen at one-cell-per-8-pixels granularity)
   plus an explicit `no_op` row (see issue #127).  Layout:

   - row 0: `no_op` (see #127)
   - row 1: `select_army`
   - rows 2..N-1: `Move_screen` at each cell of an N×N grid

   Coordinates use cell centres (e.g. for 8×8: `0.0625, 0.1875, …, 0.9375`)
   so no cell collapses to a screen edge.

   Make the grid resolution a module-level constant
   (`SCREEN_GRID_RESOLUTION = 8`) so it can be tuned without touching the
   action-construction code.

### Files changed

| File | Change |
|---|---|
| `games/sc2/actions.py` | Add `SCREEN_GRID_RESOLUTION`, rebuild `DISCRETE_ACTIONS`, prepend `no_op` and `select_army` rows. |
| `games/sc2/sc2_policies.py` | Swap spatial head from 9-row argmax to 2-row sigmoid (x, y).  Update `to_cfg`/`from_cfg`/`to_flat`/`with_flat`/`mutated`/`_make_member` for new shape. |
| `games/sc2/policies.py` | `_N_DISCRETE_ACTIONS` derives from `len(DISCRETE_ACTIONS)`; verify `NeuralDQNPolicy`/`REINFORCEPolicy` net sizing, `SC2LinearPolicy` unaffected. |
| `tests/test_sc2_actions.py` | Update grid-cell assertions for new resolution; add tests for `no_op` row, `select_army` row, and that `Move_screen` rows span the unit square (min ≤ 0.1, max ≥ 0.9). |
| `tests/test_sc2_genetic_policy.py` | Update spatial-head dimensionality assertions. |

### Migration / breaking changes

- `SC2MultiHeadLinearPolicy` weight files written before this change will not
  round-trip: the spatial head goes from 9 rows × N to 2 rows × N.  Apply the
  same `from_cfg(... default 0.0)` migration that `WeightedLinearPolicy` uses
  (drop the obsolete `spatial_{0..8}_weights` keys, initialise `x_weights` /
  `y_weights` to zero).  Champions become essentially random, which is
  acceptable since they were stuck in the 9-cell trap anyway.
- `NeuralDQNPolicy` / `REINFORCEPolicy` saved trainer state has shape
  `(_, 9)` for the output layer.  Detect this on load (existing
  `n_layers`/shape checks) and raise the existing
  "Use --re-initialize to restart from scratch" error message.

### Validation

- Unit test: `SC2MultiHeadLinearPolicy.__call__` produces (x, y) varying
  continuously across a sweep of mocked observations.
- Smoke test on `MoveToBeacon`: confirm pixel coordinates in the debug log
  no longer cluster at `{12, 31, 50}` — they should span the full 0–63
  range over a few episodes.

---

## 2. Issue #127 — ensure "don't act" is in the action space

### Root cause

`DISCRETE_ACTIONS` contains 9 rows: 1 `select_army` + 8 `Move_screen`.  No
row maps to `no_op`.  Discrete-action policies (`NeuralDQNPolicy`,
`REINFORCEPolicy`, the discrete LSTM dispatch) cannot select "do nothing"
even in principle.

For `SC2MultiHeadLinearPolicy` the fn_idx head can pick `fn_idx=0` (no_op)
because the FUNCTION_IDS table includes it, but in practice the spatial head
is always on regardless of fn_idx, and the warmup / reward shaping push the
policy toward emitting Move_screen.

### Fix

1. **Add `no_op` as DISCRETE_ACTIONS row 0** (covered by the issue #122
   restructuring above).
2. **Stop overloading `select_army` as the warmup.**  Today
   `WARMUP_ACTION = [1, 0.5, 0.5, 0]` (select_army) is forced for the first
   N steps of an SC2 episode.  Keep that behaviour — `select_army` is a
   sensible precondition — but ensure `no_op` is reachable post-warmup.
3. **Reward shaping nudge.**  Add an optional `idle_bonus` to
   `SC2RewardConfig` (default 0.0 — i.e. opt-in, no behaviour change for
   existing experiments).  When the agent emits `no_op` AND has units in
   combat range of an enemy (detectable via the new screen summary
   features added under #126), award `idle_bonus`.  This makes the
   "don't move so units can shoot" lesson learnable rather than only
   discoverable through luck.
4. **Probe coverage.**  Add a `no_op` probe to `PROBE_ACTIONS` (it is
   already there as the first probe — verify and keep).  Its reward floor
   tells the cold-start search whether idling is competitive on the chosen
   map.

### Files changed

| File | Change |
|---|---|
| `games/sc2/actions.py` | `no_op` row in `DISCRETE_ACTIONS` (covered by #122). |
| `games/sc2/reward.py` | Add `idle_bonus` field to `SC2RewardConfig`; logic in `compute()`. |
| `games/sc2/config/reward_config.yaml` | Document the new param; default 0.0. |
| `tests/test_sc2_reward.py` | Test idle bonus fires only on no_op + combat-range condition. |
| `CLAUDE.md` | Document `idle_bonus` in the reward config table. |

### Validation

- Unit test: `NeuralDQNPolicy(obs).fn_idx == 0` is reachable when the
  network outputs argmax on row 0.
- Inspection: a freshly initialised genetic population produces a non-zero
  fraction of `no_op` actions in the first generation log.

---

## 3. Issue #126 — expand observation space

### What PySC2 currently offers

The observation dict returned per-step by PySC2 (via `feature_screen` /
`feature_minimap` / `player` / `feature_units` / etc.) includes:

#### Player vector (`obs.player`, 11 scalars)
Currently we use 8 of these.  Already mapped: `minerals`, `vespene`,
`food_used`, `food_cap`, `army_count`, `idle_worker_count`,
`warp_gate_count`, `larva_count`.  Not yet mapped:
- `food_army` — supply tied up in army units (vs workers).
- `food_workers` — supply tied up in workers.
- `player_id` — fixed per game; useful for multi-agent only.

#### Score cumulative (`obs.score_cumulative`, 13 scalars)
Currently we read only `score_cumulative[0]` (total score).  Each entry is a
named feature that exposes a richer breakdown:
`score, idle_production_time, idle_worker_time, total_value_units,
total_value_structures, killed_value_units, killed_value_structures,
collected_minerals, collected_vespene, collection_rate_minerals,
collection_rate_vespene, spent_minerals, spent_vespene`.

These are the standard PySC2 economic / military telemetry — the same set
the Blizzard tournament leaderboards report.  Adding them lets reward
shaping and policies condition on "I am ahead in killed value" or
"I am bleeding workers", both of which are key strategic signals.

#### Score by category / vital (`obs.score_by_category`, `obs.score_by_vital`)
Per-category breakdowns of the score (food, army, economy, etc.) and per
target type (units killed by category, damage taken, etc.).  Less urgent;
treat as Phase 2 of #126.

#### Game loop / available actions
- `game_loop` — already mapped on ladder spec only; promote to all specs.
- `available_actions` — currently only used by the client to mask invalid
  actions; also useful as a binary feature vector for the policy
  (cardinality = `len(FUNCTION_IDS)` = small).

#### Screen feature layers (`obs.feature_screen`, 17 channels of 64×64)
Today we read only `player_relative` and project it to 6 scalars (counts +
centroids of self / enemy).  PySC2 exposes:

| Layer | Description | Notes |
|---|---|---|
| `height_map` | terrain elevation | static; minimal value |
| `visibility_map` | hidden / fogged / visible | screen vis (0/1/2) |
| `creep` | zerg creep tile | meta-game; defer |
| `power` | protoss pylon power | meta-game; defer |
| `player_id` | which player owns each tile | duplicates player_relative for 1v1 |
| `player_relative` | 0=none, 1=self, 2=ally, 3=neutral, 4=enemy | ALREADY USED |
| `unit_type` | per-tile unit type id | huge cardinality (≥2000); use only as channel for CNN |
| `selected` | mask of currently-selected units | useful summary scalar |
| `unit_hit_points` / `_ratio` | HP per tile | aggregate to mean / min |
| `unit_shields` / `_ratio` | shields | protoss-specific |
| `unit_energy` / `_ratio` | caster energy | terran/protoss caster-specific |
| `unit_density` | units per tile | density of friendly cluster |
| `unit_density_aa` | anti-air density | air engagement signal |
| `effects` | spell effects (storms, etc.) | sparse |
| `pathable` | walkable tile | static |
| `buildable` | tile a building can be placed on | static |

#### Minimap feature layers (`obs.feature_minimap`, 7 channels of 64×64)
Today we read `player_relative` and `visibility_map` for ladder spec.
Other channels: `height_map`, `creep`, `camera`, `player_id`, `selected`.
Minimap visibility / explored already projected to scalars on ladder.
Not yet on minigame.

#### Single / multi select (`obs.single_select`, `obs.multi_select`)
Currently we project `selected` to `selected_count` and
`selected_avg_hp`.  PySC2 exposes per-row [unit_type, player_relative,
health, shields, energy, transport_slots_taken, build_progress].  We can
add `selected_avg_shields`, `selected_avg_energy`,
`selected_dominant_unit_type`.

#### Feature units (`obs.feature_units`, variable-length list)
Per-unit: position, type, owner, HP, shields, energy, build progress,
weapon cooldown, order id, target tag, is_selected, is_in_cargo, …
~30 fields.  This is the richest source.  Cannot fit into a fixed flat
vector directly — needs aggregation:

- For each of {self, enemy, neutral}: count, mean HP, mean shields, total
  food value, mean weapon cooldown, fraction with target acquired.
- Top-K closest enemies to friendly centroid: relative position, HP, type.
- Worker/army split for self.

#### Build queue / production / cargo / last_actions
Production-side telemetry; mostly relevant for ladder/build minigames.
Defer to Phase 2.

### Design

Rather than one giant flat vector, define **modular obs presets** so users
can opt into the level of richness their hardware budget allows.  Three
canonical specs:

1. **`SC2_MINIGAME_OBS_SPEC`** (currently 13 dims) — kept exactly as today
   for backward compatibility with existing minigame champions.
2. **`SC2_LADDER_OBS_SPEC`** (currently 21 dims) — extended to ~40 dims
   adding: economy breakdown (`collected_minerals_rate`,
   `collected_vespene_rate`, `total_value_units`, `total_value_structures`,
   `killed_value_units`, `killed_value_structures`,
   `idle_production_time_norm`, `idle_worker_time_norm`,
   `spent_minerals`, `spent_vespene`), army vs worker split (`food_army`,
   `food_workers`), screen unit-density / mean-HP summaries, top-K enemy
   features (counts only), minimap-camera position.
3. **`SC2_RICH_OBS_SPEC`** (new) — full superset:
   - Everything in `SC2_LADDER_OBS_SPEC`, plus
   - 8 per-unit-type counts (top-8 most common SC2 unit types per race,
     resolved at module load),
   - Per-direction (NE/NW/SE/SW) screen quadrant counts of self / enemy,
   - Top-3 closest enemies: (rel_x, rel_y, hp_ratio) → 9 floats,
   - Available-actions binary mask,
   - Last action issued (one-hot over `FUNCTION_IDS`).
   Total: ~70-90 dims; document precisely once code lands.

The map-to-spec mapping in `obs_spec.get_spec(map_name)` becomes a registry
keyed on the experiment's `obs_spec_preset` config field rather than purely
on map name.  Default: minigame → minigame, ladder → ladder.  Setting
`obs_spec_preset: rich` opts into the new spec.

#### Spatial channels (separate from flat spec)

`screen_layers` / `minimap_layers` already supports stacking arbitrary
named feature layers as `(C, H, W)` channels in a `Dict` observation
space.  Only `sc2_cnn` consumes them today.  Wire them through to other
policies opt-in (out of scope for #126 itself, but call out in the plan
since it's the natural follow-up — note in the README under the
`screen_layers` config knob).

#### Backward compatibility

- Existing minigame champions must keep loading.  Achieved trivially by
  leaving `SC2_MINIGAME_OBS_SPEC` unchanged and gating the new features
  behind explicit preset selection.
- Existing ladder champions (21 dims) load against the new ~40-dim ladder
  spec via the standard "missing key → 0.0" migration in
  `WeightedLinearPolicy.from_cfg` and the equivalent path in
  `SC2MultiHeadLinearPolicy.from_cfg`.  Document this in the migration
  notes.

### Files changed

| File | Change |
|---|---|
| `games/sc2/obs_spec.py` | Define `_PLAYER_DIMS`, `_SCORE_DIMS`, `_SCREEN_SUMMARY_DIMS`, `_MINIMAP_SUMMARY_DIMS`, `_PER_UNIT_DIMS`, `_AVAILABLE_ACTIONS_DIMS` building blocks; assemble three presets from them; update `get_spec()` to consult an `obs_spec_preset` config field.  Document each `ObsDim` with accurate scale + description. |
| `games/sc2/client.py` | Extend `_timestep_to_obs_info` with helpers `_score_features`, `_screen_summary_features`, `_minimap_summary_features`, `_top_k_enemy_features`, `_per_unit_type_counts`.  Each emits the slice expected by its corresponding `_*_DIMS` block.  Branch on `self._spec` to determine which slices to compute. |
| `games/sc2/env.py` | Read `obs_spec_preset` from training params and pass through to `SC2Client`. |
| `games/sc2/config/training_params.yaml` | Add `obs_spec_preset: minigame` (or `ladder` / `rich`) with comment. |
| `games/sc2/sc2_policies.py`, `games/sc2/policies.py` | No structural change; obs_dim is read from the supplied `obs_spec` so new presets work without code changes.  Verify `SC2LinearPolicy` migration covers new feature names. |
| `tests/test_sc2_obs_spec.py` | Tests for each preset's dim, names, scales, per-feature documentation completeness. |
| `tests/test_sc2_client.py` | Mock PySC2 timesteps to verify each new feature group is populated correctly and that `_safe_*` helpers tolerate missing fields. |
| `CLAUDE.md` | Update the SC2 observation-space section with all three presets and a feature-name table. |

### Validation

- Unit tests on each preset: `len(spec.dims) == expected_n_dims`,
  `spec.scales` finite, `spec.names` unique.
- Mocked-PySC2 client test: feed in synthetic timesteps with known
  `score_cumulative` / `feature_units` / `feature_screen` arrays; assert
  the produced flat vector equals the expected per-feature output.
- Migration test: load a 13-dim minigame champion under the rich preset
  and confirm it produces a finite, non-NaN action.

---

## 4. Cross-cutting design decisions

| Decision | Rationale |
|---|---|
| Single PR / branch covering all three issues. | Action / obs surface change once; one weight-file migration cycle. |
| Continuous (x, y) sigmoid head replaces 9-cell argmax for `SC2MultiHeadLinearPolicy`. | Roots out #122 at the policy level; finer DISCRETE_ACTIONS for tabular policies is the parallel fix. |
| `no_op` is row 0 of `DISCRETE_ACTIONS`. | Issue #127 requires it be selectable.  Row 0 keeps it visually obvious in logs. |
| Three named obs presets, opt-in via config key. | Avoids breaking existing experiments; explicit upgrade path. |
| `idle_bonus` reward param defaults to 0.0. | Reward-shaping change must not silently alter existing run outcomes. |

## 5. Implementation phasing

Order matters because some test assertions depend on the new shapes.

1. **Phase A — actions (issues #122, #127).**
   1. Edit `games/sc2/actions.py`: new `SCREEN_GRID_RESOLUTION`,
      regenerate `DISCRETE_ACTIONS`, add `no_op` row, update
      `WARMUP_ACTION` documentation to call out it's still
      `select_army`.
   2. Edit `games/sc2/sc2_policies.py`:
      `SC2MultiHeadLinearPolicy` spatial head → 2 rows (x, y) with
      sigmoid encoding; update serialisation, flat-vector layout,
      mutation, crossover.
   3. Update `games/sc2/policies.py` derived constants
      (`_N_DISCRETE_ACTIONS`).
   4. Update `tests/test_sc2_actions.py`,
      `tests/test_sc2_genetic_policy.py`,
      `tests/test_sc2_simple64_training.py` to match new shapes.
   5. Run `pytest tests/test_sc2_actions.py
      tests/test_sc2_genetic_policy.py tests/test_sc2_client.py
      tests/test_sc2_simple64_training.py`; iterate to green.

2. **Phase B — reward `idle_bonus` (issue #127, opt-in).**
   1. Add `idle_bonus` field + logic in `games/sc2/reward.py`.
   2. Document in `games/sc2/config/reward_config.yaml` and `CLAUDE.md`.
   3. Add test.

3. **Phase C — observations (issue #126).**
   1. Refactor `games/sc2/obs_spec.py` into modular dim blocks +
      three presets + `obs_spec_preset` registry.
   2. Extend `games/sc2/client.py::_timestep_to_obs_info` with the
      new feature extractors.  Each is a small static method so unit
      tests can target them directly.
   3. Wire `obs_spec_preset` config knob through `env.py` and
      `adapter.py`.
   4. Update `tests/test_sc2_obs_spec.py`,
      `tests/test_sc2_client.py`.
   5. Update `CLAUDE.md` SC2 section.

4. **Phase D — verification.**
   1. Full `pytest tests/` to catch unrelated regressions.
   2. Mocked-environment smoke test of `SC2GeneticPolicy` end-to-end
      using a stub PySC2 env (one already exists in
      `tests/test_sc2_genetic_policy.py`); confirm action coordinates
      span > 5 distinct (x, y) values across an episode.
   3. (Manual, on a Linux+SC2 box if available; not blocking) run a
      few `MoveToBeacon` episodes and inspect the debug log to
      confirm the (12, 31, 50) coordinate cluster is gone.

5. **Phase E — commit + push.**
   1. Single commit per phase, descriptive messages referencing the
      issue numbers.
   2. `git push -u origin claude/plan-issues-122-126-127-zbhRr`.
   3. Do NOT open a PR until the user asks.

## 6. Risks and open questions

- **Genetic policy expressivity.**  Replacing argmax-over-9-cells with a
  sigmoid (x, y) head changes the search landscape.  The 9-cell head can
  switch sharply; sigmoid is smooth.  For `MoveToBeacon` (a single beacon
  to walk to) sigmoid should help; for `CollectMineralShards` (multiple
  scattered targets) it may regress.  Mitigation: keep an experimental
  `policy_type: sc2_genetic_grid` registered as a fallback that uses the
  old 9-cell encoding.  Decide based on Phase D smoke results — only
  worth carrying both if the regression is real.

- **Rich obs preset cardinality.**  ~70-90 dims is enough that linear
  policies start to overfit observation noise.  Keep `mutation_share:
  0.3` default for the rich preset (sparse mutation).  Document this in
  the README.

- **Per-unit-type counts depend on race.**  The "top-8 most common unit
  types per race" list is hard-coded today.  For now, ship with a
  reasonable union (Marine, SCV, Zergling, Drone, Probe, Stalker,
  Roach, Mutalisk).  Future work: derive this dynamically from the
  current map's `feature_units` distribution at episode 0.

- **`feature_units` may be unavailable** depending on PySC2 version /
  AgentInterfaceFormat flags.  We already pass
  `use_feature_units=True` in `client._make_sc2_env`, so this is fine,
  but the `_safe_array` fallback should still cover it.

- **CNN policy compatibility.**  Adding more `feature_screen` channels
  is gated by `screen_layers` config and unaffected by the obs_spec
  refactor.  No action needed.
