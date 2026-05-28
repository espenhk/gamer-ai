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
