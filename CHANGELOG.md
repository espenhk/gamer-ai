# Changelog

All notable user- and developer-visible changes to `gamer-ai` are recorded
here. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project does
not cut numbered releases yet, so entries are grouped by date.

**Updating this file is part of every PR.** See the
[PR template](.github/PULL_REQUEST_TEMPLATE.md) checklist and the
"Changelog" section in [CLAUDE.md](CLAUDE.md). Add an `## [Unreleased]`
bullet whenever a change is user- or developer-meaningful (new feature,
new config key, breaking change, bug fix, dependency change, doc-only
change to a public-facing file). Trivial commits — experiment dumps,
formatting, internal refactors with no behaviour change — can be skipped.

---

## [Unreleased]


---

## [0.5.6] - 2026-06-07

### Added
- SC2: `build_repeat_penalty` reward parameter. Per-step penalty when the
  agent issues the same build fn_idx on two consecutive env steps. Breaks
  "Can't find placement location" spam loops where a failed placement earns
  no consequence from the reward function. Setting `build_repeat_penalty:
  -build_train_bonus` (e.g. `-0.5`) makes repeated build spam net-zero
  reward while leaving the first issuance fully rewarded. Default `0.0`
  (opt-in); shipped ladder config enables it at `-0.5`. Appears as
  `build_repeat_penalty` in the reward components dict.

---

## [0.5.3] - 2026-06-06

### Fixed
- SC2 training loop no longer slows down progressively over long runs (issue #378).
  Three root causes removed:
  - Army and resource time-series were appended every env step; they now sample
    every 10 steps (`_SERIES_SAMPLE_RATE`), bounding each episode's series to
    ~1 340 points at `step_mul=1` for a 10-minute game instead of ~13 400.
  - Six expensive per-step info fields (`episode_action_counts`,
    `episode_action_name_map`, `episode_xy_hist`, `episode_obs_averages`,
    `episode_army_series`, `episode_resource_series`) were rebuilt from scratch
    on every step; they are now computed only at episode end.
  - `greedy_sims` in the training loop held full series data for every
    simulation; non-improving sims now store `None` for series fields, so
    memory use no longer grows with the number of simulations.
- Action frequency bar chart in the live GUI now groups SC2 `Move_screen` /
  `Attack_screen` actions by action type instead of showing every distinct
  `(x, y)` coordinate as a separate bar, giving a readable frequency summary
  across the 64-cell grid.

---

## [0.5.2] - 2026-06-06

### Added
- SC2: `build_train_bonus` reward parameter (issue #416). Per-step bonus
  when the agent issues any "build" or "train" category action (supply
  depots, barracks, workers, marines, etc.). Encourages production over
  idle movement in the early game. Default `0.0` (opt-in); shipped ladder
  preset enables it at `0.5`. Normalised in analytics via the
  `build_train` component key.
- SC2: `new_action_unlock_bonus` reward parameter (issue #360). One-shot
  bonus the first time a tech-tree-gated fn_idx becomes fully executable
  in an episode. Paired with `new_action_usage_bonus` which rewards
  actually issuing those newly-available actions.
- SC2: `damage_taken_penalty` reward parameter (issue #401). Per-step
  penalty proportional to HP+shield lost across visible friendly units.
  Default `0.0` (opt-in); shipped ladder preset enables it at `-0.01`.

---

## [0.5.1] - 2026-06-05

### Changed
- Live GUI (`--live_gui`) now updates every 50 steps instead of every env
  step, eliminating the per-step Tkinter redraw that was blocking the SC2
  training loop and making the game chug (issue #378). The interval is
  configurable via `live_gui_update_interval` in `training_params.yaml`
  (default `50`). Rolling reward statistics are still accumulated every
  step so the displayed means remain accurate.
- The "Last 10 actions" list in the live GUI has been replaced with an
  action-frequency bar chart showing the count and percentage of each
  distinct action taken in the most recent batch of steps. No-op actions
  are still hidden.

---

## [0.5.0] - 2026-06-04

### Added
- Framework-level behaviour-cloning seam (issue #393, parent #392). New
  modules `framework/bc.py` (`BCAdapter` Protocol + `run()` orchestrator)
  and `framework/bc_io.py` (`load_dataset`, `save_summary`) lift the
  game-agnostic parts of the SC2 BC pipeline into the framework. The
  `demos.npz` schema is byte-compatible with what `games/sc2/replay_bc.py`
  has been writing since #351, so existing datasets load unchanged. The
  `GameAdapter` Protocol gains an optional `bc: BCAdapter | None`
  attribute.
- SC2 BC ported onto the framework seam (issue #394, parent #392). New
  `games/sc2/bc_adapter.py` implements `BCAdapter` for SC2 and is wired
  on `SC2Adapter.bc`. `--bc` is now game-agnostic at the CLI level
  (`_run_bc(adapter, args)` in `main.py`); games without a wired
  `BCAdapter` exit with a clear error. `games/sc2/replay_bc.py` still
  owns the replay parser and per-target fitters, and its `run()` remains
  as a backward-compat shim with the legacy summary shape — external
  scripts and `tests/test_sc2_replay_bc.py` keep working unchanged.
- TMNF BC adapter (issue #395, parent #392). New
  `games/tmnf/bc_adapter.py` implements `BCAdapter` for TMNF using the
  in-game `SimplePolicy` as the demonstration source — drives N laps
  (default 3, override via `bc_n_demo_laps`) and least-squares-fits a
  `WeightedLinearPolicy` on the resulting (obs, action) pairs.  Wired
  on `TMNFAdapter.bc`, so `python main.py <experiment> --game tmnf
  --bc` is the supported way to reproduce the legacy `do_pretrain`
  warm-start.  `.Replay.Gbx` ingest is tracked separately in #396.
- BC refactor — Phase 4: docs, polish, and dead-code cleanup (issue #397,
  parent #392).  New `docs/framework/bc_adapter.md` documents the
  `BCAdapter` Protocol, `demos.npz` dataset schema, `bc_summary.json`
  schema, and a worked example of adding BC to a new game.
  `docs/framework/README.md` updated to list the new page.  `CLAUDE.md`
  gains a game-agnostic `--bc` section under **Running** (alongside
  `--play` / `--eval`); the SC2 `--bc` paragraph now cross-links to the
  framework doc.  `games/tmnf/README.md` updated: `.Replay.Gbx` note
  changed from "tracked in #396" to "not currently supported".
  `games/sc2/README.md` BC intro cross-links framework doc.  Stale
  `#396` / phase-3 references removed from `games/tmnf/bc_adapter.py`
  and `main.py`.

### Fixed
- BC review polish (#413): `main.py --bc` help text no longer claims
  TMNF is unimplemented; `_run_bc` now catches `ImportError` and emits
  the same install-hint message style as `--play` / `--eval`;
  `framework/bc_io.load_dataset` opens the NPZ via a context manager so
  the file descriptor is released before its `TemporaryDirectory` is
  cleaned up (fixes a Windows file-lock edge case);
  `framework/bc.py` module docstring corrected to point at
  `games/<game>/bc_adapter.py` (matches actual convention).

### Breaking
- **Migration: `do_pretrain: true` → `--bc`.**  The `do_pretrain: true`
  training-params key and its `rl/pretrain.py` landing pad were removed
  in issue #395 (parent #392).  To reproduce the old warm-start
  behaviour, run a one-off BC step before the regular training run:
  ```bash
  python main.py <experiment> --game tmnf --bc   # produces policy_weights.yaml
  python main.py <experiment> --game tmnf        # fine-tunes from BC weights
  ```
  The BC output is byte-compatible with what `do_pretrain` produced.
  Stale `do_pretrain: true` keys in existing `training_params.yaml`
  files are silently ignored by `RunConfig.from_training_params`, so
  legacy configs continue to load without error.

---

## [0.4.6] - 2026-06-04

---

## [0.4.5] - 2026-06-04

### Changed
- **SC2 ladder default reward enables `damage_taken_penalty`** (issue #401).
  The bundled `games/sc2/config/reward_config.yaml` now sets
  `damage_taken_penalty: -0.01` (was `0.0`), so the agent now pays a small
  per-HP cost when friendly units take damage on screen.  A 40-HP Marine
  loss costs ~0.4 — comparable to a single `unit_loss_penalty` hit — while
  staying small enough to tolerate the feature_units off-screen noise.
  No code changes; set the key back to `0.0` to recover the previous behaviour.

### Added
- **SC2 Refinery target snap** (issue #402).  `SC2Client` now tracks visible
  neutral vespene geysers from `feature_units` each step and rewrites the
  target coordinates of `Build_Refinery_screen`, `Build_Assimilator_screen`,
  and `Build_Extractor_screen` actions onto the nearest cached geyser
  before issuing the PySC2 call.  Geysers occupied by an existing
  Refinery/Assimilator/Extractor drop out of the neutral list naturally
  (the friendly building takes their place), so the snap won't pick an
  already-occupied geyser.  When no geyser is visible the action passes
  through unchanged (PySC2 then no-ops it, identical to the previous
  behaviour).  New `GEYSER_NAMES` frozenset in `games/sc2/tech_tree.py`.

---

## [0.4.4] - 2026-06-04

---

## [0.4.3] - 2026-06-04

### Fixed
- **Atari: `neural_dqn` (and `reinforce`, `lstm`) policy registration** (issue #399).
  `games/atari/adapter.py` never imported `games.atari.policies`, so Atari-specific
  policy types were never registered in `POLICY_REGISTRY`, causing
  `ValueError: Unknown policy_type: 'neural_dqn'` when running any Atari grid search.
  Fixed by adding `import games.atari.policies` as a side-effect import in
  `build_game_spec()`, matching the pattern already used by the TMNF and SC2 adapters.

### Added
- **Atari: `reinforce` and `lstm` policy thin wrappers** (issue #399).
  `games/atari/policies.py` now also registers `REINFORCEPolicy` (Monte Carlo
  policy gradient, softmax over 18 actions) and `LSTMEvolutionPolicy` (LSTM +
  isotropic ES), both gated with the same duplicate-registration guards as
  `NeuralDQNPolicy`.
- **Atari grid search templates** (issue #399): `gs_genetic_template.yaml` and
  `gs_reinforce_template.yaml` under `games/atari/config/`, joining the existing
  `gs_neural_dqn_template.yaml`.  All three templates include budget notes,
  sweep-axis rationale, and second-pass ablation suggestions.

---

## [0.4.2] - 2026-06-03

### Added
- **SC2 `new_action_usage_bonus` reward component** (issue #400).  New opt-in
  reward that fires each time the agent *issues* a tech-gated fn_idx that has
  already been unlocked this episode, up to a configurable cap
  (`new_action_usage_max_uses`, default 50 uses per fn_idx per episode).
  Complements the existing `new_action_unlock_bonus` (issue #360): unlocking
  a tech earns one large reward; using it consistently earns additional shaping.
  Both bonuses are independent and can be enabled together or separately.
  New `reward_config.yaml` keys: `new_action_usage_bonus` (default `0.0`) and
  `new_action_usage_max_uses` (default `50`).  Emitted as
  `components["new_action_usage"]` in the reward breakdown.

---

## [0.3.23] - 2026-06-02

### Fixed
- `_fit_bc_tabular()` Q-normalization bug (issue #354): Q-values are now
  divided by the total state-visit count `_n_s[state]` to produce a proper
  action-frequency distribution summing to 1.0 per state, rather than dividing
  each `q[s,a]` by its own `_n_sa[s,a]` (which always yielded 1.0).
- `_fit_bc_dqn()` terminal transition bug (issue #354): transitions at episode
  boundaries now store a zero-vector as `next_obs` instead of leaking the first
  observation of the following episode, which would corrupt the Bellman target.
- `fit_bc(target="sc2_lstm")` now raises a descriptive `ValueError` listing the
  missing keys when `episode_starts` / `episode_lengths` are absent from the
  dataset, rather than propagating a bare `KeyError` from `_iter_episodes_from_dataset`.
- `_copy_bc_weights()` in `grid_search.py` now also copies `policy_weights.npz`
  (the `sc2_cnn` weight format) alongside `policy_weights.yaml`, fixing the
  CNN warm-start path that previously silently left the experiment without weights.
- `_fit_bc_cnn()` spatial-head OOM (issue #354): replaced the dense
  `(N × _N_SPATIAL_CELLS)` distance and one-hot matrices with batched distance
  computation (4096-row chunks) and scatter-add normal equations
  `(H^T H) W = H^T Y`, reducing peak memory from O(N × S) to O(N × FC_DIM)
  plus O(FC_DIM × S).

### Added
- SC2 replay BC documentation pass (issue #355, [6/6]): `games/sc2/README.md`
  — removed stub text, added full `bc_*` config-key table, per-policy warm-start
  support table (`sc2_reinforce` / `sc2_genetic` / `sc2_cmaes` / `sc2_neural_net`
  / `sc2_neural_dqn` / `sc2_lstm` / `sc2_cnn` / tabular families), and
  coordinate/resolution caveats; `CLAUDE.md` — noted `--bc` mode in the Running
  section and all `bc_*` keys in the Config knobs table.
- BC warm-start integration with `grid_search.py` (issue #354, [6/6]): two ways
  to warm-start every combo in a grid search from a behaviour-cloning checkpoint.
  (1) **Post-hoc warm-start** — pass `--bc-warmstart-dir <path>` pointing to an
  existing BC experiment directory (one that contains `policy_weights.yaml` +
  `bc_summary.json`).  Before any combo runs, a policy-compatibility check reads
  `bc_target` from `bc_summary.json` and validates it against every combo's
  `policy_type` via `_BC_COMPATIBLE_POLICY_TYPES` (cross-compatibility:
  `sc2_genetic` ↔ `sc2_cmaes`; all others self-only).  Weight files
  (`policy_weights.yaml`, `trainer_state.npz`, `policy_weights_qtable.pkl`) are
  copied into each combo's experiment directory before `train_rl` is called.
  (2) **Inline BC** — add a `bc:` section to the grid config YAML with at
  minimum `replay_dir` and optionally `bc_target`, `player_id`, `race`,
  `bc_epochs`, `bc_learning_rate`, and all other BC knobs.  `grid_search.py`
  runs BC once into a shared `<base_name>__bc_warmstart/` directory, skips re-run
  if `policy_weights.yaml` + `bc_summary.json` already exist, then validates
  compatibility and copies weights into every combo directory.  If both
  `--bc-warmstart-dir` and an inline `bc:` section are present,
  `--bc-warmstart-dir` takes precedence with a warning.
  `_load_grid_config` now returns a 7-tuple including `bc_cfg`.
  New abbreviated entries added to `_ABBREV` for all `bc_*` config keys.
- SC2 BC warm-start for all policy families (issue #354, [5/6]): `fit_bc` now
  accepts seven additional targets beyond the original `sc2_reinforce` /
  `sc2_genetic` pair.  `sc2_cmaes` — linear least-squares fit into
  `SC2MultiHeadLinearPolicy`, then seeds `SC2CMAESPolicy.initialize_from_champion`
  so the CMA-ES distribution mean starts at the fitted weights.  `sc2_neural_net`
  — mini-batch MSE regression (logit-transformed fn_idx / x / y targets) into
  `SC2NeuralNetPolicy`.  `sc2_neural_dqn` — pre-fills the `MaskedReplayBuffer`
  with demo transitions matched to the nearest `DISCRETE_ACTIONS` row by L1
  distance; BC "loss" is reported as fill fraction.  `sc2_lstm` — collects LSTM
  hidden states via a full episode-sequence forward pass (with proper h/c resets
  at episode boundaries), then trains only the output head (`W_out`/`b_out`) via
  cross-entropy SGD, and seeds `SC2LSTMEvolutionPolicy.initialize_from_champion`.
  `sc2_cnn` — zeroes the two conv layers and fits a random projection from
  obs → FC_DIM, then solves for the fn and spatial output heads via closed-form
  least squares, seeding `SC2CNNEvolutionPolicy._champion` and `_mean`.
  `epsilon_greedy` / `ucb_q` — seeds `_q_table`, `_n_sa`, and `_n_s` from
  binned demo (state, action_idx) visits; Q-values are normalised by visit count.
  SB3 targets raise `ValueError` with an explicit "SB3" message.  New `fit_bc`
  params: `n_channels` (CNN), `n_bins` (tabular), `bc_lstm_hidden_size` (LSTM).
  `run()` forwards all new params through to `fit_bc`.
- SC2 behaviour-cloning core fit + `--bc` entry point (issue #353, [4/6]):
  `fit_bc(dataset, obs_spec, *, target, ...)` in `games/sc2/replay_bc.py`
  pre-trains a policy from a demonstration NPZ: `target="sc2_reinforce"`
  trains a two-head REINFORCE MLP via mini-batch gradient descent (cross-entropy
  on fn_idx, MSE on spatial coords); `target="sc2_genetic"` fits a
  `SC2MultiHeadLinearPolicy` via closed-form least squares.  `run(replay_dir,
  experiment_dir, obs_spec, **opts)` wires the full pipeline
  (replay → dataset → fit → save `policy_weights.yaml` + trainer state +
  `bc_summary.json`).  New `main.py --bc` mode (SC2-only, mutually exclusive
  with `--play`/`--eval`) and matching config keys in
  `games/sc2/config/training_params.yaml` (`bc_player_id`, `bc_race`,
  `bc_target`, `bc_epochs`, `bc_learning_rate`, `bc_batch_size`,
  `bc_ignore_noop`, `bc_step_mul`, `bc_max_replays`).
- SC2 replay BC dataset builder at `games/sc2/replay_bc.py` (issue #351,
  [2/6]): reads `.SC2Replay` files via the PySC2 replay API and produces
  sequence-aware NPZ demonstration datasets (`obs`, `actions`,
  `episode_starts`, `episode_lengths`, `episode_id`, `meta`).  Public API:
  `iter_replays(folder)`, `replay_observations(path, ...)` (generator),
  `build_dataset(folder, save_path, ...)`, and `load_dataset(path, ...)`.
  Supports race filtering, winner/explicit player selection, configurable
  `step_mul`, and a `multi_action_strategy` knob (`"first"` /
  `"first_non_noop"`) for steps with multiple simultaneous actions.  All
  PySC2 imports are lazy (inside function bodies) so the module is
  importable without PySC2 installed.
- SC2 behaviour-cloning primitives (issue #350, part 1 of #349): a new
  `function_call_to_action()` in `games/sc2/actions.py` inverts
  `action_to_function_call`, converting a PySC2 `FunctionCall` back into the
  framework's `[fn_idx, x, y, queue]` vector (spatial coords renormalised to
  `[0, 1]`, non-spatial actions report centre coords `0.5, 0.5`, unknown
  function ids return `None` as a skip sentinel). A module-level
  `extract_flat_obs()` in `games/sc2/client.py` is now the single source of
  truth for PySC2-TimeStep → flat-observation projection, so the live client
  and the upcoming offline replay reader share one code path with no drift.

### Changed
- `SC2Client._timestep_to_obs_info` and its per-block feature extractors are
  refactored into thin wrappers over the new shared module-level functions.
  No runtime behaviour change — flat observations and info dicts are
  identical (existing SC2 tests unchanged and passing).

---

## [0.3.22] - 2026-06-02

### Fixed
- **SC2 analytics: no_op excluded from action-frequency plot** (issue #382).
  `plot_action_frequency()` now strips `fn_idx=0` (no_op) from all three panels — per-sim
  stacked bars, aggregate counts, and entropy — so the remaining actions are legible.  When
  every recorded action is no_op the function returns early and writes no file.
- **SC2 select_idle_worker spam suppressed** (issue #383).
  `SC2Client._compute_available_fn_ids()` now removes `select_idle_worker` (fn_idx 4) from the
  available set when a worker (SCV / Probe / Drone) is already in the current selection,
  preventing the policy from issuing a redundant re-selection.  `SC2Env` also switches
  `action_counts` tracking from the executed fn_idx to the policy-requested fn_idx so the
  analytics chart reflects policy intent rather than auto-injected intermediate selects.

---

## [0.3.21] - 2026-06-02

### Added
- **SC2 hierarchical action space** (issue #388): actions are now grouped into
  five meta-categories (`move`, `attack`, `build`, `train`, `upgrade`).
  `ACTION_CATEGORIES`, `CATEGORY_NAMES`, `N_CATEGORIES`, and
  `FN_IDX_TO_CATEGORY` exported from `games.sc2.actions`.
- `SC2HierarchicalLinearPolicy`: two-stage linear policy that first selects a
  meta-category via one weight head, then selects a specific action within that
  category via a second head.  Also adds a learned `queue` head so orders can
  be queued or issued immediately.
- `SC2HierarchicalGeneticPolicy` (`policy_type: sc2_hierarchical`): evolutionary
  search over `SC2HierarchicalLinearPolicy` individuals; drop-in replacement for
  `sc2_genetic` with the narrowed action-selection structure.

---

## [0.3.20] - 2026-06-01

### Added
- **Atari: DQN grid-search template** (`games/atari/config/gs_neural_dqn_template.yaml`, issue #385).
  Sweeps `learning_rate × epsilon_decay_steps` (6 combos) on `Pong-v5` using DQN-paper
  conventions: reward clipping (`clip_sign: true`), Double-DQN targets, Huber loss,
  `replay_buffer_size: 100 000`, and `hidden_sizes: [256, 256]` for the 128-dim RAM
  observation.  `games/atari/README.md` updated to reference the template.

---

## [0.3.19] - 2026-06-01

### Changed
- **SC2 training log: action shares now show function names instead of raw indices** (issue #375).
  `>> NEW BEST` and `[stats @ sim N]` blocks now display e.g. `Attack_screen=3.0%` instead
  of `3=3.0%`.  The mapping is supplied by `SC2Env` via the new `episode_action_name_map`
  info key; other games without a name map continue to show raw keys unchanged.
- **Per-episode non-improvement log lines demoted to DEBUG** (issue #377).
  `ep end`, `>> no improvement`, and `[stats @ sim N]` blocks are now `logger.debug` so they
  are hidden at the default `INFO` log level.  `>> NEW BEST` and its reward breakdown remain
  at `INFO`.
- **SC2 feature extraction skips unused preset groups** (issue #379).
  `SC2Client._timestep_to_obs_info` now gates expensive ladder- and rich-only feature extractors
  behind `_use_ladder_obs` / `_use_rich_obs` flags computed once at init.  Minigame runs skip
  score, screen-HP, top-K enemy, and alert extractors; ladder runs additionally skip all
  rich-only extractors (quadrant counts, per-unit-type enemies, shield/energy, creep, weapon
  cooldown, etc.), reducing per-step Python overhead for the two most common presets.

---

## [0.3.18] - 2026-06-01

---

## [0.3.17] - 2026-06-01

### Added
- SC2: new `resource_banking_penalty`, `mineral_banking_threshold`, and
  `gas_banking_threshold` reward config keys (issue #372).  Per-step penalty
  proportional to minerals above 300 or vespene above 200, nudging the agent
  to invest banked resources rather than hoard them.  Opt-in (default `0.0`);
  recommended range for ladder maps: `-0.0001` to `-0.001`.

---

## [0.3.16] - 2026-06-01

### Added
- SC2: `new_action_unlock_bonus` reward config key (issue #360). Fires a
  one-shot bonus per episode the first time a tech-tree-gated action appears
  in `available_fn_ids` (i.e. becomes fully executable — prerequisite building
  exists, correct unit selected, and affordable). Only actions with at least
  one `required_buildings` precondition are eligible; selection-only and
  always-available actions are excluded. Default `0.0` — opt-in. Set in
  `reward_config.yaml`.

---

## [0.3.15] - 2026-05-31

### Added
- SC2: new `idle_worker_penalty` reward config key (issue #358).  Per-step
  penalty scaled by `idle_worker_count` from PySC2 — each idle worker
  subtracts `idle_worker_penalty` per step per idle worker.  Default `0.0`
  (opt-in); recommended range for economy/ladder maps: `-0.05` to `-0.5`.
  The count is now forwarded in the env `info` dict so downstream reward
  shaping can always read it.

---

## [0.3.14] - 2026-05-31

### Changed
- **SC2: removed the Patrol commands from the action space (issue #359).**
  `Patrol_screen` and `Patrol_minimap` (the rarely-useful follow/patrol-unit
  orders) are dropped from `FUNCTION_IDS`, the race fn-id sets, and the
  tech-tree `PRECONDITIONS` / `RESOURCE_COSTS` tables.  All following fn_ids
  shift down by 2 to keep the table contiguous (`FUNCTION_IDS` now has 116
  entries, `0–115`).  Existing SC2 champion weight files migrate via the
  standard "missing key → 0.0" path; the fn_idx head is now 2 rows smaller.

---

## [0.3.13] - 2026-05-31

### Fixed
- SC2 agent no longer freezes on ladder maps (issue #356).  Three bugs
  introduced by PR #348 (tech-tree preconditions + deferred-action queue)
  are fixed:
  - **H1 — infinite select_army oscillation**: `step()` now bypasses
    `_resolve_action()` when consuming a deferred action.  Previously,
    re-running the resolver on the replayed action would emit another
    `select_army` (always "available" in PySC2 even with an empty army)
    and re-defer indefinitely, stalling the agent for the entire episode.
  - **H2 — buildings disappear on camera move**: `_owned_buildings` is now
    accumulated across steps via `_owned_buildings_seen` so structures that
    scroll off-screen no longer vanish from the tech-tree mask and block
    build/train actions mid-episode.
  - **H3 — per-step tech-tree CPU overhead**: `_compute_available_fn_ids()`
    caches its result and skips the 118-call `fn_idx_satisfied()` loop when
    `owned_buildings`, `completed_upgrades`, `selected_unit_types`, and the
    PySC2 candidate set are all unchanged since the previous step.

---

## [0.3.12] - 2026-05-31

### Fixed
- SC2: build and train actions are now excluded from `available_fn_ids` when
  the agent cannot afford them (issue #357). The action mask now filters by
  mineral and vespene cost in addition to the existing tech-tree, building
  prerequisite, and selection checks. `fn_idx_satisfied()` in
  `games/sc2/tech_tree.py` accepts optional `minerals` and `vespene`
  arguments (defaulting to `inf` for backwards compatibility); the client
  tracks current resource counts each step and passes them to the filter.
  Costs for every build/train fn_idx with a non-zero mineral or vespene
  cost across Terran, Protoss, and Zerg are recorded in the new
  `RESOURCE_COSTS` table in `tech_tree.py`; zero-cost actions (movement,
  selection, energy abilities, mode-change morphs) have no entry.

---

## [0.3.11] - 2026-05-30

### Added
- SC2 self-play now supports three opponent-selection modes (issue #345).
  Set `self_play_mode` in `training_params.yaml` (default `"exact"`):
  - `"exact"` — opponent is a fresh snapshot of the current champion,
    refreshed every generation (previously the opponent was set only once
    at run start and never updated).
  - `"mutated"` — opponent is a slightly mutated copy of the champion;
    mutation strength controlled by `self_play_mutation_scale` (default
    inherits `mutation_scale`).
  - `"top_n"` — opponent is drawn uniformly at random from a pool of the
    top-N champions seen so far (pool capacity set by `self_play_top_n`,
    default 5); weakest pool entry is replaced when a stronger champion
    arrives.
  Implemented in `framework/self_play.py` (`SelfPlayManager`); the
  opponent is refreshed at the end of each generation in all four greedy
  loops (`hill_climbing`, `q_learning`, `cmaes`, `genetic`).

---

## [0.3.10] - 2026-05-30

---

## [0.3.9] - 2026-05-29

### Added
- SC2 hardcoded tech tree at `games/sc2/tech_tree.py` and a
  deferred-action queue in `games/sc2/client.py` (issue #346). Every
  action in the new `PRECONDITIONS` table records its building/upgrade
  prerequisites and required selection type; `info["available_fn_ids"]`
  is now race ∩ PySC2 ∩ tech-tree ∩ selection-filtered, so
  `Build_FusionCore_screen` is no longer reachable in the extreme-random
  phase before a Starport exists. When the policy emits an action whose
  selection requirement is unmet, the client emits the right `select_*`
  this tick (preferring `select_idle_worker`, falling back to
  `select_point` on a mining/busy worker — issue #346 explicitly
  required workers that aren't idle to still be selectable) and queues
  the original action for the next tick. All unit and building morphs
  (Baneling, Ravager, Lurker, BroodLord, Archon, Lair, Hive, Overseer,
  GreaterSpire) route through `UNIT_PRODUCERS` via the same `_train()`
  helper as regular units; Archon morph now accepts HighTemplar *or*
  DarkTemplar selection (previously HT-only).
- `SC2Client._compute_selected_unit_types` (renamed from
  `_compute_selected_unit_type`) now returns a ``frozenset[str]`` of
  every distinct unit type in the current selection rather than collapsing
  multi-type selections to ``None``. ``fn_idx_satisfied`` consumes the
  set so that ``ANY_UNIT`` actions (Move/Attack/Stop/Patrol/HoldPosition)
  stay satisfied after ``select_army`` on a mixed army, and ``OF_TYPE``
  actions match whenever any selected type is in the target set (PySC2
  applies the command to compatible units in the selection).
- `SC2Client._compute_owned_buildings` now filters scanned
  ``feature_units`` rows against the new `games.sc2.tech_tree.STRUCTURE_NAMES`
  set, so SCVs / Marines / Probes no longer leak into the "buildings"
  side of the tech-tree mask or the periodic state dump.
- DEBUG-level periodic state dump in `SC2Client` (issue #346 follow-up).
  Every ~10 s of wall-clock time, the client logs a readable snapshot of
  current units, owned buildings, completed upgrades, currently-selected
  unit type, and the available action set with the unit/building each
  action would need selected.
- Grid-search abbreviations for `initial_extreme_random_fraction` (`ierf`)
  and `initial_extreme_random_runs` (`ierr`) so PR #339's config keys
  appear in generated experiment directory names.

### Changed
- **Breaking (internals):** `info["available_fn_ids"]` is now always a
  set (never `None`). Policies that already mask on this field are
  unaffected.
- **Breaking (internals):** removed the proactive `select_army` /
  `select_point` substitution guard from `SC2Client.step()` and the
  reactive substitution branch from `SC2Client._action_to_call()` (both
  added in PR #322). Selection injection is now handled by the
  deferred-action queue in `_resolve_action`. The
  `_blocked_unit_targeted_steps` counter and
  `_SELECT_ARMY_RETRY_BLOCKED_STEPS` constant were dropped along with
  this.
- `SC2Client._build_unit_type_lookup()` now covers the full PySC2 unit
  enum (not just the rich-preset combat-unit subset) so the tech-tree
  filter can recognise structures and morph parents. The
  `_RICH_UNIT_TYPES`-driven per-unit-type observation features are
  unaffected (they filter at the feature-dict layer).

---

## [0.3.8] - 2026-05-28

---

## [0.3.7] - 2026-05-28

---

## [0.3.6] - 2026-05-28

### Added
- Atari 2600 integration via `ale-py` + Gymnasium (`--game atari`,
  issue #217). New `games/atari/` package with adapter, env, obs spec,
  reward, and analytics; 128-byte RAM observation that is compatible with
  every flat-observation framework policy. Configurable `map_name` (any
  ALE-registered title; default `Pong-v5`), `clip_sign` DQN-style reward
  clipping, and a `--track` override. New optional Poetry group
  `atari` (`poetry install --with atari`).

---

## [0.3.5] - 2026-05-28

### Changed
- Assetto Corsa now conforms to the `GameAdapter` protocol and is registered
  in `GAME_ADAPTERS` like the other seven games (#326). The bespoke
  `_run_assetto()` entry point in `main.py` and the old `games/assetto_corsa/entry.py`
  runner are removed; `--game assetto` now flows through the same unified
  `_run_one()` path as every other game. The experiment directory layout
  changes from `experiments/assetto_corsa/<track>/<name>/` to
  `experiments/assetto_corsa/<policy>/<track>/<name>/` (matching the
  convention used by all other games).

---

## [0.3.4] - 2026-05-28

### Documentation

- Expanded `CLAUDE.md` with a policy-selection guide, run-sizing formulas, and
  audited per-policy hyperparameter defaults / tuning notes. Added a regression
  test so the documented `policy_params` tables stay aligned with the registered
  policy implementations.

---

## [0.3.3] - 2026-05-28

---

## [0.3.2] - 2026-05-28

---

## [0.3.1] - 2026-05-28

### Added

- **Gradient deep-RL policies (Stable-Baselines3-backed):** `a2c`, `sac`, `td3`,
  `qr_dqn` (distributional value, SB3-Contrib), and `recurrent_ppo`
  (gradient-trained LSTM, SB3-Contrib), plus an SB3-backed `ppo` that **replaces
  the pure-numpy `ppo` introduced in 0.3.0** (same `policy_type`, now backed by
  SB3). They share a new `LOOP_TYPE = "sb3"` driven by `framework/sb3_support.py`
  (SB3 owns its own training loop); budget is set via
  `policy_params.total_timesteps` (default `n_sims × steps_per_sim`). Models are
  saved next to `policy_weights.yaml` as `*_sb3_model.zip`. `sac`/`td3` are
  continuous-only; `qr_dqn` wraps the env's `Box` into a `Discrete` index over
  `discrete_actions`. All are gated off SC2's multi-head action encoding.
- **New optional Poetry group `deep_rl`** (`stable-baselines3`, `sb3-contrib`;
  pulls `torch`): `poetry install --with deep_rl`. Cross-platform.
- **`alphazero_mcts`** — a *real* model-based Monte-Carlo Tree Search policy
  (pure numpy): PUCT search expanded by cloning + stepping the env, guided by a
  policy/value network trained from self-play (`LOOP_TYPE = "alphazero"`,
  `framework/alphazero.py`). Requires a deterministic cloneable simulator, so
  it is gated off all current games (their envs bind to live processes) until a
  game's env gains a `clone()` / `deepcopy` capability.

### Changed

- **`neural_dqn` / `sc2_neural_dqn` DQN upgrades extended.** On top of 0.3.0's
  `double_dqn` + `dueling` options, added Huber (smooth-L1) loss and global-norm
  gradient clipping. The SB3-aligned knobs are now **on by default**: new
  `policy_params` `huber_loss` (`true`), `huber_kappa` (`1.0`),
  `max_grad_norm` (`10.0`), and `double_dqn` now defaults `true` (was `false` in
  0.3.0); `dueling` stays opt-in (`false`). Set `double_dqn: false`,
  `huber_loss: false`, `max_grad_norm: null` to recover vanilla behaviour.
  **Behaviour change** for existing DQN configs.
- **BREAKING: `mcts` policy_type renamed to `ucb_q`** (class
  `MCTSPolicy` → `UCBQPolicy`). The policy was never tree search — it is a
  tabular UCB1 online Q-learner — so the name was misleading. Update any config
  with `policy_type: mcts` to `policy_type: ucb_q`; the bare name `mcts` is no
  longer registered (the accurate, model-based MCTS is `alphazero_mcts`).

### Documentation

- Added an "implementation fidelity vs the field" subsection to
  `docs/research/competing-projects.md` comparing our shared-name algorithms
  (CMA-ES, DQN, REINFORCE, LSTM, MCTS) against their canonical/field versions.
- Updated `CLAUDE.md`, `README.md`, `framework/README.md`, and
  `tests/README.md` for the new policies, the DQN knobs, and the rename.

---

## [0.3.0] - 2026-05-27

### Added
- New `ppo` policy: a pure-numpy on-policy actor-critic (`framework/ppo.py`,
  `PPOPolicy`) with a clipped surrogate objective, Generalised Advantage
  Estimation, an entropy bonus, and multi-epoch minibatch updates. It reuses the
  existing `q_learning` greedy loop (buffers transitions in `update`, learns in
  `on_episode_end`) and is registered framework-wide, so it is available on every
  game with a discrete action set (TMNF, CarRacing, TORCS, BeamNG, …). It declares
  itself incompatible with SC2 (use `sc2_reinforce`). New `policy_params`:
  `hidden_sizes`, `learning_rate`, `gamma`, `gae_lambda`, `clip_range`,
  `n_epochs`, `entropy_coeff`, `value_coeff`, `minibatch_size` (#328).
- `neural_dqn` (and `sc2_neural_dqn`) gain two opt-in upgrades over vanilla DQN:
  `double_dqn` (online-net action selection, target-net evaluation — curbs
  Q-value overestimation) and `dueling` (separate value + advantage streams,
  aggregated as `Q = V + (A − mean A)`). Both default `false`; existing weight
  files load unchanged (#328).

### Removed
- Deleted the orphaned Stable-Baselines3 PPO script `rl/train.py`. PPO is now a
  first-class, registry-integrated, pure-numpy policy (see above); SB3/torch were
  never core dependencies (#328).

---

## [0.2.19] - 2026-05-27

### Changed (framework genericity — issue #325)

- **`framework/policies.py`**: Replaced the hardcoded `SC2_GAME_NAME = "sc2"` gate in `WeightedLinearPolicy`, `NeuralNetPolicy`, and `GeneticPolicy` with a capability registry (`register_continuous_action_incompatible`) that any game adapter can call at import time to declare itself incompatible with steer/accel/brake policies. `_sc2_incompatible()` removed; new public helpers `register_continuous_action_incompatible()` and `check_continuous_action_compatible()` added.
- **`games/sc2/adapter.py`**: Registers `"sc2"` via `register_continuous_action_incompatible` at module level, providing per-policy-type replacement hints (`cmaes`→`sc2_cmaes`, etc.).
- **`games/tmnf/policies.py`**: Added `compatible_with()` to `NeuralDQNPolicy`, `CMAESPolicy`, `REINFORCEPolicy`, and `LSTMEvolutionPolicy` (the four TMNF-specific policy wrappers) using the same registry, so they are rejected on SC2 via `_assert_policy_compatible` without the old `_SC2_REMOVED_BARE_POLICY_TYPES` gate.
- **`framework/training.py`**: Removed `_SC2_REMOVED_BARE_POLICY_TYPES` constant and the `if game_name == "sc2"` gate in `_make_policy`. Replay numbering now counts files matching `{experiment}_best-*` (any extension) instead of `*.SC2Replay`; `_finalize_candidate_replay` derives the extension from the candidate file path. Removed direct `from games.sc2.actions import FUNCTION_IDS` import; action counts are now logged by raw key. Updated section comments to be game-agnostic. `_log_new_best_details` section 5 now iterates all keys in `episode_obs_averages` instead of a hardcoded SC2 list.
- **`framework/obs_spec.py`**: Added generic `ObsSpec.with_extra_dims(dims)` extension method. `with_lidar()` now delegates to `with_extra_dims()` (kept for backward compatibility).
- **`framework/live_monitor.py`**: `_REWARD_ORDER` reduced to `["total_reward"]`; all other reward keys are now sorted alphabetically. New games no longer need to edit this file to get sensible live-monitor display ordering.
- **`framework/analytics.py`**: `save_grid_summary` default `task_metric_label` changed from `"Best Track Progress"` to `"Best Task Metric"`. Callers that want TMNF-style labelling should pass `task_metric_label="Best Track Progress"` explicitly.

---

## [0.2.18] - 2026-05-26

---

## [0.2.17] - 2026-05-26

### Documentation

- Add `docs/research/competing-projects.md` — a verified survey of competing
  open-source RL-for-games projects, prioritising racing (Trackmania: `tmrl`,
  Linesight, `TrackMania_AI`, TMAI, ShubhamGajjar's IQN agent, MOSEAC,
  TMInterface; Gran Turismo: GT Sophy) and SC2 (AlphaStar, PySC2/SC2LE,
  `reaver`, `pysc2-examples`, plus an OpenAI Five / Dota 2 contrast), plus a
  lighter pass over general libraries
  (Stable-Baselines3, CleanRL, RLlib, Gymnasium, ALE, MineRL/BASALT, RLGym).
  Includes a master comparison table and a "takeaways for gamer-ai" section
  cross-referenced to #327/#328 (#329).

---

## [0.2.16] - 2026-05-25

### Documentation
- Top-level docs reframed from "TMNF-only" to the multi-game framework they
  describe, and the game roster corrected from six to **eight** (#323):
  - `README.md` retitled to *gamer-ai — multi-game RL agent framework*; intro
    now lists all eight games and states per-game platform support; the false
    "this project only runs on Windows" prerequisite is scoped to TMNF; the
    `--game` quick-start examples gain `sc2`, `rocket_league`, and `iracing`.
  - `CLAUDE.md` "six games" → "eight games", with `rocket_league` and
    `iracing` added to the runtime table, the repository-structure tree, and
    the grid-search `--game` choices.
  - `CONTRIBUTING.md` roster and project-layout tree updated to all eight
    games.
  - New `test_game_adapter.py` check asserts every `GAME_ADAPTERS` key (plus
    the special-cased `assetto` choice) appears in `CLAUDE.md`, so the roster
    can't silently drift again.
- Removed stale/duplicate docs (#324): deleted the obsolete `CLAUDE.original.md`
  and the empty `scratch.txt`; replaced the duplicate `docs/README_TORCS.md`
  and `docs/README_ASSETTO.md` with one-line pointers to the canonical
  `games/<name>/README.md`. `CONTRIBUTING.md` now states the convention
  explicitly: per-game docs live in `games/<name>/README.md`; `docs/framework/`
  holds cross-game protocol docs.
- `CLAUDE.md` and `CONTRIBUTING.md` now record a hard labelling rule:
  `documentation` issues must never carry the `good first issue` label.

---

## [0.2.15] - 2026-05-22

---

## [0.2.14] - 2026-05-22

---

## [0.2.13] - 2026-05-22

---

## [0.2.12] - 2026-05-22

---

## [0.2.11] - 2026-05-22

---

## [0.2.10] - 2026-05-21

---

## [0.2.9] - 2026-05-21

---

## [0.2.8] - 2026-05-21

---

## [0.2.7] - 2026-05-21

---

## [0.2.6] - 2026-05-21

---

## [0.2.5] - 2026-05-21

---

## [0.2.4] - 2026-05-21

---

## [0.2.3] - 2026-05-21

### Added
- **Rocket League integration** (`--game rocket_league`): single-agent RL via
  [RLGym](https://rlgym.org/).  142-dim observation (car/ball + 2 teammate
  cars + 3 opponent cars + relative features + boost pad availability),
  8-dim continuous action space
  (throttle/steer/pitch/yaw/roll/jump/boost/handbrake), and a dense+sparse
  reward calculator (velocity-to-ball, ball touch, goal scored/conceded).
  Requires Rocket League (commercial, Windows) + Bakkesmod + `pip install rlgym`.
  See `games/rocket_league/README.md` for install instructions.
  New reward config keys: `vel_to_ball_weight`, `boost_weight`, `touch_bonus`,
  `goal_weight`, `concede_penalty`.  New training param: `tick_skip`.
- **Rocket League grid-search templates**: added one tuned template per currently
  supported policy type (`hill_climbing`, `neural_net`, `epsilon_greedy`,
  `mcts`, `genetic`) under `games/rocket_league/config/`.

### Fixed
- **Rocket League tick-skip wiring**: `tick_skip` now propagates from
  `training_params.yaml` through `RocketLeagueAdapter` into
  `rlgym.make(tick_skip=...)`, so grid-search/template values affect simulator
  stepping as intended.
- **Rocket League reward scaling**: `vel_towards_ball` no longer re-scales
  observation entries in `RocketLeagueEnv._compute_vel_towards_ball()`, avoiding
  inflated dense reward from double-scaling.
- **Rocket League team support**: env now uses `team_size=3` and observation
  slots for 3 opponents + 2 friendlies, enabling 3v3-state training inputs.
- **iRacing live action injection** (`games/iracing/controller.py`):
  Phase 2 action injection via vJoy virtual joystick.  New
  `action_mode` training param (`"telemetry_only"` default,
  `"live"` for vJoy injection).  `VJoyController` maps
  steer/throttle/brake to vJoy axes; `NullController` preserves
  existing telemetry-only behaviour.  `pyvjoy` is an optional
  dependency (only required for `action_mode: live`).

---

## [0.2.2] - 2026-05-21

- **TMNF bug fix:** Finish-detection threshold raised from 0.95 → 0.98 so that
  near-end track sections (e.g. the drop at the end of A03) no longer trigger a
  false finish before the physical finish line is reached.

---

## [0.2.1] - 2026-05-21

### Added
- **iRacing telemetry integration** (`games/iracing/`): Phase 1 read-only
  telemetry via `pyirsdk`.  21-dim observation (speed, RPM, gear, fuel,
  tire loads/temps, brake bias, lap times, …), standard steer/accel/brake
  action space, progress + centerline + off-track reward.  Registered as
  `--game iracing` in `main.py` and `grid_search.py`.

### Fixed
- **SC2 action names**: Renamed `select_point_screen` → `select_point`
  and `select_rect_screen` → `select_rect` in `FUNCTION_IDS` to match
  actual PySC2 function names.  Fixes `KeyError` in integration tests
  when epsilon-greedy or other policies select these actions.

---

## [0.2.0] - 2026-05-21

---

## [0.1.9] - 2026-05-21

### Fixed
- **TMNF accel bonus**: `accel_bonus` is now only awarded when the throttle
  is pressed AND the car's speed actually increased that step (`curr speed >
  prev speed`).  Previously the bonus was given whenever the gas pedal was
  pressed, allowing a stuck or wall-spinning car to accumulate unlimited reward.

---

## [0.1.8] - 2026-05-21

---

## [0.1.7] - 2026-05-21

### Added
- Distributed coordinator: new mobile-friendly `/monitor` web app with
  username/password login, a run selector, and per-run queued / active /
  completed state so users can watch multi-machine runs from a phone. By
  default the monitor username is `monitor` and the password reuses the
  coordinator token; both can be overridden with
  `distribute.monitor_username` / `distribute.monitor_password` or the
  matching CLI flags.

---

## [0.1.6] - 2026-05-21

### Fixed
- **Live GUI** (`framework/live_monitor.py`): the "Last 10 actions" panel now renders 3-axis driving actions in a legible control-oriented format, treating tiny pedal values (`<= 0.01`) as effectively zero so it cleanly shows `accel % | steer ...` or `brake % | steer ...` instead of raw vectors.

---

## [0.1.5] - 2026-05-21

### Changed
- **Live GUI** (`framework/live_monitor.py`): window shrunk from 1200×850 to 960×720; reward and observation panels are now independently scrollable (mousewheel supported); reward components are displayed in a fixed logical order (no more jumping); scalar observations are shown in stable obs-spec order rather than sorted by magnitude; subtle vertical grid lines added to all bar charts; a new "Last 10 actions" panel shows the most recent actions in a fixed sidebar.
- `framework/training.py`: `live_monitor.on_step()` now receives the current `action` so the actions panel can display it.

---

## [0.1.4] - 2026-05-21

### Documentation

- Add `docs/framework/` — one Markdown page per framework-side protocol
  (`GameAdapter`, the `GameSpec`/`RunConfig`/`ProbeSpec`/`WarmupSpec`/
  `PolicyExtras` config bundles, `BaseGameEnv`, `RewardCalculatorBase`,
  `BasePolicy`, `ObsSpec`), each with method contracts and a worked
  example. Linked bidirectionally from `CONTRIBUTING.md` ("Adding a new
  game") and `CLAUDE.md` (#220).

---

## [0.1.3] - 2026-05-20

### Fixed
- SC2: agents on BuildMarines, Simple64, and every other non-movement map were
  unable to train units, construct buildings, or issue any race-specific command
  because `FUNCTION_IDS` only contained 6 entries (movement + harvest).
  `FUNCTION_IDS` now covers all 118 Terran, Protoss, and Zerg build / train /
  morph / ability commands used in standard PySC2 play (fn_idx 0–117).
  `SPATIAL_FN_IDS` is auto-derived as the frozenset of fn_ids whose names end in
  `_screen` or `_minimap` (55 entries); every spatial function now gets a full
  `N×N` (`SCREEN_GRID_RESOLUTION²`, default 8×8 = 64 rows) block in
  `DISCRETE_ACTIONS`, giving a uniform `[command × location]` layout (3 583 rows
  total).  Race gating (`RACE_FUNCTION_IDS`, `fn_ids_for_race()`, private
  `_TERRAN_FN_IDS` / `_PROTOSS_FN_IDS` / `_ZERG_FN_IDS` sets) ensures that
  agents only ever see the actions valid for their race — this permanent mask is
  applied in every SC2 policy's `__call__` before the per-step
  `available_fn_ids` mask.  All four multi-head SC2 policies
  (`SC2GeneticPolicy`, `SC2CMAESPolicy`, `SC2LSTMEvolutionPolicy`, and the
  `SC2MultiHeadLinearPolicy` base) accept and propagate a `race` parameter.
  `N_FUNCTION_IDS` grows from 6 to 118 automatically; existing weight files
  migrate cleanly via the zero-default path.  (Closes #276)

---

## [0.1.2] - 2026-05-20

---

## [0.1.1] - 2026-05-20

### Documentation
- `README.md` now links directly to the
  `good first issue` filter, `CONTRIBUTING.md` documents the canonical
  issue-label taxonomy, and the shared issue template now applies the
  default `triage` label on newly opened issues.
- PR template (`.github/PULL_REQUEST_TEMPLATE.md`) now carries a
  `Closes #<issue>` line near the top so PRs auto-close their issue on
  merge.  `CLAUDE.md` gains a **Pull requests** section requiring every
  PR description to be filled in from the template with that
  `Closes #<issue>` link.
- `CLAUDE.md` brought back in sync with the codebase:
  - Documents all six supported games (adds CarRacing, BeamNG, Assetto
    Corsa alongside TMNF / TORCS / SC2) in the intro, repository-structure
    tree, and run examples.
  - Corrects the master-config location — configs are per-game under
    `games/<game>/config/`, not a top-level `config/` directory.
  - Refreshes the **Dependencies** section for the current Poetry group
    layout (core vs `tmnf` / `tmnf-test` / `torcs` / optional `sc2` /
    `assetto_corsa`, plus CarRacing/BeamNG out-of-group deps).
  - Adds the `sc2_neural_net` policy, the `log_stats_every_n_sims`
    training param, the SC2 `--eval` mode, the `attack_friendly_penalty`
    and `small_selection_bonus` SC2 reward keys, the
    `grid_search --local-workers` / `--local-worker-stagger` flags, the
    SC2 map-access-gate env vars, and the `main.py` `--track` / `--workers`
    / `--log-level` override flags.
  - Updates the `move_exploration_bonus` / `move_repeat_penalty`
    descriptions to match the issue #253 unit-position tracking fix.

### Added
- `POLICY_REGISTRY` and `register_policy` decorator in `framework/policies.py`; the five built-in policies (`hill_climbing`, `neural_net`, `epsilon_greedy`, `mcts`, `genetic`) are now self-describing with `POLICY_TYPE`, `LOOP_TYPE`, `VALID_POLICY_PARAMS`, and `_construct_or_resume`. `framework/training.py:_make_policy` is now a single `POLICY_REGISTRY` lookup plus a game-compatibility check (Phases B–D of #224).
- Phase C of #224: all game-specific policies migrated to thin registered subclasses. `games/tmnf/policies.py` now registers `neural_dqn`, `cmaes`, `reinforce`, and `lstm` as `@register_policy` subclasses of the framework algorithm classes; `games/sc2/sc2_policies.py` registers `sc2_genetic`, `sc2_reinforce`, `sc2_cmaes`, and `sc2_lstm`. Factory closures and loop-dispatch entries in both adapters removed for the migrated types; `build_extras` in TMNF adapter now returns `None`. Net: ~2 500 lines of duplicated algorithm code deleted.
- `BasePolicy.compatible_with(game_name)` class hook returning `(ok, migration_hint)`; the continuous-action framework policies (`hill_climbing`, `neural_net`, `genetic`) override it to reject SC2, replacing the SC2 adapter's free-function check. `GameSpec` gains a `game_name` field so the check has the game identity (Phase D of #224, #231).
- `_GradEntry` namedtuple exported from `framework/reinforce.py` and used by `TwoHeadREINFORCEPolicy` for per-step trajectory storage; re-exported from `games/sc2/sc2_policies.py` for backward compatibility.
- `framework/README.md`: developer guide documenting the policy registry, all five algorithm modules, per-algorithm adaptation hooks, and a worked example showing how to create a new game's policies from scratch.
- New post-merge workflow `.github/workflows/auto-version-bump.yml` that
  automatically runs after a PR is merged into `main`, infers release bump
  type from PR-template checkboxes (`Patch` default, `Minor`, `Major`),
  computes the next SemVer, and runs `scripts/release.py --no-tag` to bump
  `pyproject.toml` + `framework/version.py` and roll `## [Unreleased]` into
  a dated version section.
- Analytics reports now surface code version tags more prominently:
  single-run `results.md` includes a dedicated **Code Version** block, and
  grid-search `summary.md` includes a **Code Versions** section plus a
  per-experiment `Code version` stat row.
- Optional live training GUI (`--live-gui`) for both `main.py` and
  `grid_search.py`. The window updates during training (not post-run only):
  - reward-component bar chart per step with a 5-step rolling average, plus
    total step reward;
  - live observation visualizations using feature-aware layouts (scalar bars,
    x/y pair vectors, indexed strips, and quadrant grids when detected).
- **SC2 `attack_bonus` reward component** (issue #251).  New opt-in reward
  config key `attack_bonus` (default `0.0`) awards a flat bonus whenever the
  agent issues `Attack_screen` (fn_idx 3), regardless of whether the target
  is a visible enemy unit (click-to-attack) or on open ground (A-move).  Acts as
  a simpler alternative to enabling both `attack_move_bonus` and
  `click_attack_bonus` separately; all three can be active simultaneously.
  The contribution is tracked as a separate `"attack_bonus"` entry in
  `reward_components` and is normalised in cross-experiment grid-search
  summaries alongside the existing attack bonus components.
- **Analytics: reward component breakdown charts** (issue #252).
  - `framework.analytics.plot_reward_component_breakdown` — diverging stacked
    bar chart (one bar per greedy sim, positive components above zero, negative
    below) written to `reward_component_breakdown.png` alongside the existing
    per-component line chart.
  - `games.sc2.analytics.plot_gs_reward_component_breakdown` — cross-experiment
    diverging horizontal bar chart (one row per experiment, showing mean per-sim
    component contributions) written to `comparison_reward_breakdown.png` in the
    grid-search summary directory and linked from `summary.md`.
- **SC2 periodic stats logging** (issue #240).  Training now logs reward
  component totals and action-frequency ratios every `log_stats_every_n_sims`
  sims (default `10`, set to `0` to disable).  Covers all four SC2 greedy
  loops (`_greedy_loop`, `_greedy_loop_cmaes`, `_greedy_loop_genetic`,
  `_greedy_loop_q_learning`).  New training param: `log_stats_every_n_sims`
  (integer, default `10`, stored in `training_params.yaml`).
- **SC2 intra-run parallel evaluation** (issue #229).
  Population-based SC2 policies (`sc2_genetic`, `sc2_cmaes`, `sc2_lstm`,
  `sc2_cnn`) can now evaluate individuals concurrently across multiple
  local SC2 binaries.  Set `n_workers > 1` in `training_params.yaml`
  to spawn a persistent worker pool (one SC2 env per worker, spawn
  start method) — each generation's offspring are scored in parallel
  while the distribution update remains generation-synchronous (genetic
  and cmaes loop dispatch).
  New config keys: `n_workers` (default `1`),
  `worker_start_stagger_s` (default `5.0`),
  `worker_warmup_timeout_s` (default `90.0`),
  `worker_base_seed` (default `0`).  See the *Intra-run parallel
  evaluation* subsection in `CLAUDE.md` for sizing guidance.
- `framework.parallel_eval.ParallelEvaluator` — game-agnostic worker
  pool used internally by `train_rl` when `n_workers > 1`.

### Changed
- Phase D of #224 (#231): removed the `PolicyExtras` infrastructure. `framework/training.py:_make_policy` is now a single `POLICY_REGISTRY` lookup plus a compatibility check; `train_rl` no longer takes an `extras=` parameter and reads the greedy-loop type directly from `policy.LOOP_TYPE`. `GameAdapter.build_extras` and all per-game implementations are deleted; each adapter now registers its policy types via a side-effect import inside `build_game_spec`. The remaining SC2 policies (`sc2_cnn`, `sc2_neural_net`, `sc2_neural_dqn`) are now `@register_policy` classes. Misconfigured/unknown `policy_type` values fail fast before the game is launched.
- `SC2REINFORCEPolicy` is now a thin subclass of `framework.reinforce.TwoHeadREINFORCEPolicy`; all gradient math is inherited from the framework class, eliminating ~430 lines of duplicated REINFORCE code from `games/sc2/sc2_policies.py`. YAML champion format and test compatibility are preserved.
- `SC2LSTMEvolutionPolicy` is now a thin subclass of `framework.lstm.LSTMEvolutionPolicy` (which gained an optional `_template` keyword argument); all isotropic-ES mechanics are inherited, eliminating ~200 lines of duplicated ES code. The inner `SC2LSTMPolicy` individual is injected via `_template`.
- `framework.lstm.LSTMEvolutionPolicy.__init__` accepts a new keyword-only `_template` parameter: when supplied, it is used as the inner individual instead of constructing a `LSTMCore`. Existing call sites are unaffected (default `None`).
- Fixed `TwoHeadREINFORCEPolicy` handling of `hidden_sizes=[]`: the constructor now uses `list(hidden_sizes) if hidden_sizes is not None else [128, 64]` instead of `list(hidden_sizes or [128, 64])`, which incorrectly treated an empty list as falsy.
- Experiment output directories now use nested folders instead of encoding
  policy/grid params into one long experiment folder name:
  `experiments/<game>/<policy>/<map>/<experiment_name>/<param_1>__<param_2>...`.
  Single runs now live at `experiments/<game>/<policy>/<map>/<experiment_name>/`,
  and grid-search runs place the varied-parameter suffix in the final folder.
- Distributed grid-search coordinator now supports LAN-focused multi-machine
  home setups out of the box:
  - New `--bind-host` / `distribute.bind_host` to select the interface/IP the
    coordinator listens on.
  - New LAN-only default request filter (loopback/private/link-local source
    IPs only); override with `--allow-non-lan` /
    `distribute.allow_non_lan` when explicitly required.
  - Distributed runs now default to `local_workers=1`, so the driver/coordinator
    machine contributes one local worker by default while remote workers can
    join over the LAN.
- **SC2 `move_exploration_bonus` now decays explored cells** (issue #262).
  The grid-cell visit tracking added in #253 marked a cell explored *once per
  episode*, which (a) paid the agent to blanket-roam the whole screen to
  collect every per-cell bonus and (b) went permanently silent once the screen
  was covered, making *freezing in place* optimal — observed in training as
  units spamming moves everywhere and then hyperfixating in a small area. A
  cell now **expires** `move_exploration_decay_steps` env steps after the
  friendly-unit centroid last left it, so returning to a stale area is
  rewarded again and the bonus never goes silent. A stationary centroid keeps
  refreshing its own cell every step, so the anti-command-spam guarantee from
  #253 is preserved. Two new reward-config keys (both with sensible defaults,
  so existing configs keep working):
  - `move_exploration_grid_size` (int, default `8`) — cells per axis of the
    screen grid, replacing the previously hard-coded 8×8.
  - `move_exploration_decay_steps` (int, default `50`) — env steps before an
    explored cell may be rewarded again; `0` restores the previous permanent
    once-per-episode behaviour. Because the default is non-zero, the bonus can
    now pay more than `grid_size²` times per episode, increasing its effective
    magnitude versus pre-#262 runs — retune `move_exploration_bonus` if needed.

  The bundled `games/sc2/config/reward_config.yaml` is retuned to match: the
  `move_exploration_bonus` is lowered (`1.0` → `0.15`) and
  `move_exploration_decay_steps` raised (`50` → `120`) so the term re-rewards
  only genuine relocation and stays a minority contributor, and `score_weight`
  is raised (`10.0` → `100.0`) so task score dominates the shaping terms.

### Removed
- **Breaking (SC2 only):** the legacy bare-name SC2 policy types `cmaes`, `reinforce`, `lstm`, and `neural_dqn` are no longer constructible on the `sc2` game — they were only reachable through the now-deleted `build_extras` factories and cannot share the registry names used by their TMNF counterparts. Selecting one now fails with the standard "Unknown policy_type" error; use the `sc2_`-prefixed equivalents (`sc2_cmaes`, `sc2_reinforce`, `sc2_lstm`, `sc2_neural_dqn`). This pulls forward the breaking change originally scheduled for Phase E of #224.

### Fixed
- SC2 `.SC2Map` file race when multiple PySC2 binaries boot on the same
  host (issue #254). `games.sc2.client.SC2Client._make_sc2_env` now
  routes every `SC2Env` construction through a cross-process
  *map-access gate* (`games.sc2.map_access_gate.acquire_map_access_slot`)
  that enforces a minimum 5 s gap between consecutive grants. This
  covers not only the initial worker launches but every subsequent
  SC2 reboot — distributed local workers picking up successive
  experiments, intra-run parallel-eval workers (`n_workers > 1`), and
  any future SC2 multi-instance scenarios. The gate uses an
  `fcntl.flock`-serialised timestamp file under the system temp dir
  and is tunable via two env vars:
  - `GAMER_AI_SC2_MAP_GAP_S` — gap in seconds (default `5.0`; set to
    `0` to disable, e.g. for single-process runs).
  - `GAMER_AI_SC2_MAP_LOCK_PATH` — custom timestamp-file path (mainly
    useful for tests).

  As a complementary defence-in-depth, `grid_search.py --distribute
  --local-workers N` also launches the local worker subprocesses with a
  cascading 5 s delay (first immediate, second waits 5 s, third waits
  another 5 s, …). Tunable via the new `--local-worker-stagger` CLI
  flag or `distribute.local_worker_stagger` config key (default `5.0`;
  set to `0` to disable).
- `move_exploration_bonus` exploit: bonus now tracks actual unit centroid
  positions on an 8×8 screen grid rather than move command targets, so
  spamming `Move_screen` to many locations without moving units yields no
  repeated reward. Grid cells are marked visited whenever friendly units are
  visible, and the bonus fires at most once per grid cell per episode.
- Versioning + release system. `framework/version.py` resolves a
  runtime `code_version` string of the form
  `<PACKAGE_VERSION>+g<sha7>[.dirty]`; the value is persisted in every
  run's `experiment_data.json`, surfaced in the analytics summary
  table, and logged at startup by `main.py` and `grid_search.py`.
- `python main.py --version` prints the current code version without
  starting a run.
- `scripts/release.py` cuts a release: bumps `pyproject.toml` +
  `framework/version.py`, promotes `## [Unreleased]` in `CHANGELOG.md`
  to a dated `## [X.Y.Z]` section, commits, and tags `vX.Y.Z`.

---

## 2026-05-18

### Added
- Contribution guide (`CONTRIBUTING.md`), PR template, and issue
  templates (bug report / feature request / new game integration)
  (#223).
- SC2 reward shaping: `unit_loss_penalty`, `damage_taken_penalty`, and
  `passive_under_fire_penalty` — penalise army loss, friendly HP/shield
  damage, and standing idle while under fire (#230).
- SC2 reward shaping for small-unit selection micro (#243).

### Changed
- SC2 attack reward shaping persists across `no_op` chains until the
  agent issues a different action (#242).
- `CLAUDE.md` now documents the wall-clock interaction between
  `step_mul` and `max_apm` (#213).

---

## 2026-05-17

### Added
- SC2 `--eval` mode with configurable playback speed and per-step action
  logging (#211).
- Planning spec for the SC2 win-rate chart that will replace
  track-progress in analytics (#212).

---

## 2026-05-08

### Added
- `batch_run.sh` helper and matching grid-search configs for running
  multiple experiments back-to-back.
- Cross-run reward trajectory chart and per-experiment skipped-frame
  tracking in SC2 analytics (#193, #199).
- `sc2_neural_net` policy (TMNF-style MLP) and a massive grid-search
  template covering it (#203).
- Recursive cross-grid analytics report comparing whole grid-search
  families (#205).
- Full cross-game parameter abbreviation coverage in grid-search
  experiment naming (#204).

### Changed
- `idle_bonus` (SC2) is now unit-range aware — only granted when a
  friendly unit is inside combat range of a visible enemy (#202).
- `redo_analytics.py` now uses the shared analytics summary strategy and
  is robust to differing per-experiment reward configs (#200).
- Reward normalisation in analytics no longer amplifies small reward
  differences across runs with different reward configs (#193).

### Fixed
- Post-win SC2 rounds no longer stall: blocked-action streaks now
  trigger an army re-selection retry (#207).

---

## 2026-05-07

### Added
- `attack_move_bonus` and `click_attack_bonus` rewards for SC2
  `Attack_screen` actions (#163).
- `alert_count` observation; player and score-cumulative field names are
  now sourced from PySC2 directly rather than hard-coded by position
  (#158, #177).
- Hard-coded burst APM budget protection on top of the rolling
  token-bucket limiter (#162).
- Fail-fast validation for incompatible SC2 `policy_type` values and
  unknown `policy_params` keys (#179).
- SC2-specific cross-run charts in the grid-search summary (#181).
- CarRacing + SC2 integration / end-to-end tests, gated as a
  post-approval merge gate (#154) and scoped to relevant changed paths
  only (#174).
- `NEW BEST` SC2 log line now includes the full reward breakdown plus
  scalar outcome / reward / score (#189).
- SC2 reward shaping that discourages move-target hyperfixation; SC2
  grid-search templates re-aligned (#164).

### Changed
- Full SC2 documentation audit — `CLAUDE.md` and `games/sc2/README.md`
  synced to the current implementation (#176).
- SC2 analytics: reward-based run comparisons are now normalised across
  differing reward configs (#175).
- Movement exploration reward only fires when the target is at least
  the minimum-meaningful distance from the previous move target (#185).
- SC2 client caches action-mask `fn_id` lookups on the hot path (#178).
- `tests/README.md` no longer lists per-file test counts (they go stale
  immediately) (#191).

### Fixed
- Four root causes of poor SC2 genetic policy improvement (#157).
- SC2 genetic policy idle / `select_army` spam via available-actions
  masking (#161).
- `enemy_count_*` in SC2 observations now excludes neutral and ally
  units (#187).
- `Attack_screen` commands targeting friendly units are penalised
  (#186).

---

## 2026-05-06

### Added
- SC2-specific analytics plots: build order, supply cap, resource and
  army time-series; non-racing games now skip racing-only plots (#141,
  #147).
- `SC2CMAESPolicy` (`sc2_cmaes`, issue #108) and `SC2LSTMPolicy`
  (`sc2_lstm`, issue #109) (#143).
- `redo_analytics.py` — regenerate experiment analytics from saved data
  without re-running the training loop (#144).
- SC2 grid-search template and README for the CNN policy (#150).
- Rolling token-bucket APM limiting for SC2 (#148), measured in
  in-game seconds so caps are training-speed independent.
- Info-log split: compact per-episode lines, expanded reward breakdown
  on `NEW BEST` (#149).
- Rich SC2 observation preset filled out: selected-unit shields/energy,
  screen visibility fraction, anti-air density, mean weapon cooldown
  (#151).
- Azure infrastructure: VM cloud-init installs the selected game on
  boot; coordinator/worker scripts gained a `-Game` flag and route
  workers by game (#152).

### Fixed
- Post-beacon `select_army` spam in SC2 minigames; minimap beacon
  locator added to the minigame observation (#140).

---

## 2026-05-05

### Added
- Fog-of-war belief system wired into `SC2Env` (issue #111, #136).
- Rich SC2 observation preset extended with 15 missing PySC2 features
  (#135, #137).
- Two-head REINFORCE policy with available-actions masking (#131).
- `tests/README.md` with per-test rundown and runtime analysis (#134).

### Changed
- SC2 spatial action head: the 9-cell argmax is replaced with a
  continuous `(x, y) ∈ [0, 1]²` sigmoid head; `DISCRETE_ACTIONS` reshaped
  around an 8×8 grid; `no_op` is now a first-class action (issues
  #122, #126, #127; #132).

---

## 2026-05-03 – 2026-05-04

### Added
- `SC2NeuralDQNPolicy` (`sc2_neural_dqn`) with available-actions
  masking (#120, #130).
- Human-vs-AI interactive `--play` mode for SC2 (#117).
- Planning spec for the SC2 action and observation redesign (#129,
  feeds #122 / #126 / #127).

### Fixed
- Episode-length curriculum scaling is now also applied to
  `NeuralNetPolicy` in the greedy loop (#119).

---

## 2026-05-02

### Added
- `SC2GeneticPolicy` (`sc2_genetic`) with a multi-head individual
  representation — separate fn_idx and spatial heads (#113).
- Full 1v1 RL training loop against the built-in SC2 bot (#112).
- SC2 CNN policy (`sc2_cnn`) trained by isotropic evolutionary strategy
  on feature-layer pixel observations (#116).

### Changed
- `main.py` / `grid_search.py` are now game-agnostic via an adapter
  pattern; the same scripts drive every game under `games/<name>/`
  (#115).

---

## 2026-05-01

### Added
- Framework support for partial observability — belief decay and a
  scouting urge signal (#98).
- Per-game `README.md` files under `games/<name>/` documenting
  installation, setup, and policy reference, plus root README tables of
  contents (#102).

### Fixed
- Test dependency wiring and a batch of failing tests (#100).

---
