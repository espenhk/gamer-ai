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
  is a visible enemy unit (click-to-attack) or open ground (A-move).  Acts as
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

### Changed
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

### Added
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
