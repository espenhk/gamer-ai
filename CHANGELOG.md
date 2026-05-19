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

### Added
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

### Fixed
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
