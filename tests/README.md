# Tests

- [Coverage at a glance](#coverage-at-a-glance)
- [Framework / shared](#framework--shared)
  - [test\_analytics\_no\_matplotlib.py — analytics importable when matplotlib missing](#test_analytics_no_matplotlibpy--analytics-importable-when-matplotlib-missing)
  - [test\_analytics\_task\_metrics.py — `TaskMetrics` dataclass + summary table formatting](#test_analytics_task_metricspy--taskmetrics-dataclass--summary-table-formatting)
  - [test\_belief.py — fog-of-war belief encoder](#test_beliefpy--fog-of-war-belief-encoder)
  - [test\_build\_centerline.py — `tracks/registry.json` builder](#test_build_centerlinepy--tracksregistryjson-builder)
  - [test\_consolidate.py — grid-search result consolidation](#test_consolidatepy--grid-search-result-consolidation)
  - [test\_cross\_grid\_report.py — recursive cross-grid summary collation](#test_cross_grid_reportpy--recursive-cross-grid-summary-collation)
  - [test\_curiosity.py — ICM/RND curiosity bonuses](#test_curiositypy--icmrnd-curiosity-bonuses)
  - [test\_discretize\_obs.py — continuous→discrete obs binning](#test_discretize_obspy--continuousdiscrete-obs-binning)
  - [test\_distributed.py — coordinator/worker protocol + HTTP server](#test_distributedpy--coordinatorworker-protocol--http-server)
  - [test\_early\_stopping.py — early-stop logic in greedy + Q loops](#test_early_stoppingpy--early-stop-logic-in-greedy--q-loops)
  - [test\_env\_termination.py — `_classify_termination()`](#test_env_terminationpy--_classify_termination)
  - [test\_game\_adapter.py — TMNF/TORCS/SC2/BeamNG adapter abstractions](#test_game_adapterpy--tmnftorcssc2beamng-adapter-abstractions)
  - [test\_grid\_search.py — Cartesian-product expansion + naming](#test_grid_searchpy--cartesian-product-expansion--naming)
  - [test\_info\_gain.py — staleness-based intrinsic reward](#test_info_gainpy--staleness-based-intrinsic-reward)
  - [test\_live\_monitor.py — live GUI monitor helpers](#test_live_monitorpy--live-gui-monitor-helpers)
  - [test\_obs\_memory.py — frame-stacking observation wrapper](#test_obs_memorypy--frame-stacking-observation-wrapper)
  - [test\_parallel\_eval.py — intra-run parallel evaluator (issue #229)](#test_parallel_evalpy--intra-run-parallel-evaluator-issue-229)
  - [test\_reward.py — TMNF reward calculator + curiosity glue](#test_rewardpy--tmnf-reward-calculator--curiosity-glue)
  - [test\_train\_rl\_signature.py — public `train_rl()` API](#test_train_rl_signaturepy--public-train_rl-api)
  - [test\_new\_best\_logging.py — `_log_new_best_details` + `_print_episode_summary`](#test_new_best_loggingpy--_log_new_best_details--_print_episode_summary)
  - [test\_utils.py — math/state-extraction utils](#test_utilspy--mathstate-extraction-utils)
  - [test\_track.py — centreline geometry helpers](#test_trackpy--centreline-geometry-helpers)
  - [test\_version.py — `code_version()` + git revision reporting](#test_versionpy--code_version--git-revision-reporting)
- [TMNF policies](#tmnf-policies)
  - [test\_weighted\_linear\_policy.py — linear `WeightedLinearPolicy`](#test_weighted_linear_policypy--linear-weightedlinearpolicy)
  - [test\_neural\_net\_policy.py — pure-numpy MLP policy](#test_neural_net_policypy--pure-numpy-mlp-policy)
  - [test\_genetic\_policy.py — population evolutionary loop](#test_genetic_policypy--population-evolutionary-loop)
  - [test\_cmaes\_policy.py — CMA-ES on linear weights](#test_cmaes_policypy--cma-es-on-linear-weights)
  - [test\_epsilon\_greedy\_policy.py — tabular Q-learning](#test_epsilon_greedy_policypy--tabular-q-learning)
  - [test\_mcts\_policy.py — UCT-style online Q](#test_mcts_policypy--uct-style-online-q)
  - [test\_neural\_dqn\_policy.py — DQN (replay + target net)](#test_neural_dqn_policypy--dqn-replay--target-net)
  - [test\_reinforce\_policy.py — Monte-Carlo policy gradient](#test_reinforce_policypy--monte-carlo-policy-gradient)
  - [test\_lstm\_policy.py — LSTM evolution policy](#test_lstm_policypy--lstm-evolution-policy)
- [TMNF I/O](#tmnf-io)
  - [test\_rl\_client.py — `RLClient` (game-thread bridge, action windowing)](#test_rl_clientpy--rlclient-game-thread-bridge-action-windowing)
- [TORCS](#torcs)
  - [test\_torcs\_obs\_spec.py — TORCS observation spec (19-dim)](#test_torcs_obs_specpy--torcs-observation-spec-19-dim)
  - [test\_torcs\_client.py — TORCS state→obs and action mapping](#test_torcs_clientpy--torcs-stateobs-and-action-mapping)
  - [test\_torcs\_env.py — Gym env wrapper](#test_torcs_envpy--gym-env-wrapper)
  - [test\_torcs\_reward.py — TORCS reward calc](#test_torcs_rewardpy--torcs-reward-calc)
  - [test\_torcs\_analytics.py — TORCS plots/report](#test_torcs_analyticspy--torcs-plotsreport)
- [SC2](#sc2)
  - [test\_sc2\_obs\_spec.py — SC2 obs spec](#test_sc2_obs_specpy--sc2-obs-spec)
  - [test\_sc2\_actions.py — discrete action grid + race gating](#test_sc2_actionspy--discrete-action-grid--race-gating)
  - [test\_sc2\_reward.py — SC2 reward calc](#test_sc2_rewardpy--sc2-reward-calc)
  - [test\_sc2\_client.py — PySC2 client wrapper](#test_sc2_clientpy--pysc2-client-wrapper)
  - [test\_sc2\_env.py — SC2 env wrapper](#test_sc2_envpy--sc2-env-wrapper)
  - [test\_sc2\_replay.py — SC2 replay saving on new-best events (issue #210)](#test_sc2_replaypy--sc2-replay-saving-on-new-best-events-issue-210)
  - [test\_sc2\_apm\_limiter.py — token-bucket APM limiter + SC2Env integration](#test_sc2_apm_limiterpy--token-bucket-apm-limiter--sc2env-integration)
  - [test\_sc2\_belief\_integration.py — fog-of-war belief system wired into SC2Env (issue #111)](#test_sc2_belief_integrationpy--fog-of-war-belief-system-wired-into-sc2env-issue-111)
  - [test\_sc2\_cmaes\_policy.py — `SC2CMAESPolicy` (CMA-ES over multi-head linear policy)](#test_sc2_cmaes_policypy--sc2cmaespolicy-cma-es-over-multi-head-linear-policy)
  - [test\_sc2\_lstm\_policy.py — `SC2LSTMPolicy` + `SC2LSTMEvolutionPolicy`](#test_sc2_lstm_policypy--sc2lstmpolicy--sc2lstmevolutionpolicy)
  - [test\_sc2\_genetic\_policy.py — `SC2MultiHeadLinearPolicy` + genetic trainer](#test_sc2_genetic_policypy--sc2multiheadlinearpolicy--genetic-trainer)
  - [test\_sc2\_neural\_net\_policy.py — hill-climbing MLP policy for SC2](#test_sc2_neural_net_policypy--hill-climbing-mlp-policy-for-sc2)
  - [test\_sc2\_neural\_dqn\_policy.py — masked DQN for SC2](#test_sc2_neural_dqn_policypy--masked-dqn-for-sc2)
  - [test\_sc2\_cnn\_policy.py — CNN feature extractor + CMA-ES variant](#test_sc2_cnn_policypy--cnn-feature-extractor--cma-es-variant)
  - [test\_sc2\_reinforce\_policy.py — REINFORCE policy for SC2](#test_sc2_reinforce_policypy--reinforce-policy-for-sc2)
  - [test\_sc2\_eval.py — `--eval` evaluation mode](#test_sc2_evalpy----eval-evaluation-mode)
  - [test\_sc2\_play.py — `play_sc2.py` script](#test_sc2_playpy--play_sc2py-script)
  - [test\_sc2\_simple64\_training.py — Simple64 ladder integration](#test_sc2_simple64_trainingpy--simple64-ladder-integration)
  - [test\_sc2\_analytics.py — SC2-specific analytics plots and flags](#test_sc2_analyticspy--sc2-specific-analytics-plots-and-flags)
- [CLI / misc](#cli--misc)
  - [cli/test\_game\_flag.py — `--game` CLI flag in `main.py`](#clitest_game_flagpy----game-cli-flag-in-mainpy)
  - [assetto\_corsa/test\_smoke.py — Assetto Corsa smoke tests (against fake client)](#assetto_corsatest_smokepy--assetto-corsa-smoke-tests-against-fake-client)
- [Integration tests (`tests/integration/`)](#integration-tests-testsintegration)
  - [integration/test\_car\_racing.py — CarRacing real-env end-to-end tests](#integrationtest_car_racingpy--caracing-real-env-end-to-end-tests)
  - [integration/test\_sc2.py — SC2 real-binary end-to-end tests](#integrationtest_sc2py--sc2-real-binary-end-to-end-tests)

```bash
# CarRacing only (requires gymnasium[box2d])
pip install gymnasium[box2d]
python -m pytest tests/integration/test_car_racing.py -m integration -v

# SC2 only (requires pysc2 + Blizzard SC2 binary + maps)
export SC2PATH=~/StarCraftII
python -m pytest tests/integration/test_sc2.py -m integration -v
```

## Coverage at a glance

The suite is exhaustive on **pure logic** — config parsing, reward math, policy
math, save/load round-trips, CLI flag dispatch — and silent on **anything that
needs a running game or display**. Every game client is replaced by a fake or
a `MagicMock`; no Trackmania, TORCS, SC2, BeamNG or Assetto Corsa binary is
ever launched, no game window is grabbed, no matplotlib window is rendered, no
keyboard/joystick output is sent. End-to-end integration is covered separately
in `tests/integration/` for CarRacing (Box2D, no external binary) and SC2
(headless Blizzard binary + PySC2 minigame maps).

Per-area summary below; per-file details follow.

## Framework / shared

Game-agnostic plumbing under `framework/`, `distributed/`, `config/`,
`analytics.py`, `grid_search.py`, the train_rl entry point, and shared utility
modules.

**Tested.** Reward calculator math (linear components, n_ticks scaling,
finish-bonus / progress invariants, curiosity glue); curiosity modules (ICM
and RND, factory dispatch); fog-of-war belief encoder; staleness-based
info-gain; `TaskMetrics` aggregation and summary-table formatting;
discretisation, frame-stacking and obs-memory wrappers; centreline geometry
and the `tracks/registry.json` builder; grid-search Cartesian expansion +
naming + nested `policy_params` promotion + local-worker process orchestration;
the early-stop streak logic in
both greedy and Q loops; the distributed coordinator/worker JSON protocol and
the in-process HTTP server (work queue, heartbeat re-queue, auth);
`train_rl()`'s public signature; the new-best log helpers
(`_log_new_best_details` per-component / action-frequency / TMNF / SC2 kill
groups, `_print_episode_summary` compact format); that `framework.analytics`
imports cleanly on a machine with no matplotlib; and the `redo_analytics.py`
script (game auto-detection, single-experiment regeneration,
multi-experiment summaries, `--no-individual`, inferred summary dir,
graceful skip of missing experiments), plus `cross_grid_report.py`
(recursive `<policy>/vX/` discovery, copied-summary link rewriting, and
cross-grid family comparison tables).

The intra-run parallel evaluator (`framework/parallel_eval.py`, issue #229)
is unit-tested end-to-end against a picklable dummy env / policy: worker
pool spawn (spawn context), per-individual ordering, crash isolation,
sentinel shutdown, determinism under fixed seed, per-candidate
episode-time-limit dispatch, and the `train_rl` guard
(`_maybe_build_evaluator`).  The underlying PySC2 binary spawn path is
opt-in via `RUN_SC2_TESTS=1`.

**Not tested.** Real distributed training across multiple machines (only the
in-process HTTP loopback is exercised); actual matplotlib rendering or PNG
diff'ing (analytics tests assert files appear, not their contents); the Azure
Terraform stack under `infrastructure/`; the Windows bootstrap script
`setup_and_run.ps1` (PowerShell-only; cannot run on Linux CI); long convergence
behaviour of the actual `train_rl()` loop end-to-end on a real env;
the actual PySC2 binary spawn path under `ParallelEvaluator` — only the
worker mechanics are unit-tested with a dummy env.

### test_analytics_no_matplotlib.py — analytics importable when matplotlib missing
- framework analytics import works without matplotlib
- TMNF analytics import works without matplotlib
- SC2 analytics import works without matplotlib

### test_analytics_task_metrics.py — `TaskMetrics` dataclass + summary table formatting
- new fields default to `None`; finish_time / lateral_offset / reward_components stored
- finish-rate aggregation: empty / none / all / partial
- summary string: empty / contains finish rate / best finish time / no time when none / lateral offset
- summary table: progress / finish-time / dash-on-no-finish / lateral-offset columns
- `plot_gs_reward_trajectories`: chart written by `save_grid_summary` / referenced in summary.md / no crash with empty sims
- `save_grid_summary` task-metric plugin: default uses track progress with `.4f` format; custom fn replaces label+value; custom fn drives ranking; explicit `task_metric_fmt` overrides format independently of fn
- `plot_reward_component_breakdown`: renders to file / skips when no component data / skips when no sims / skips when all-zero / positive-only / negative-only / partial-None sims use zero for missing keys

### test_belief.py — fog-of-war belief encoder
- initial encode all zero; update sets value+confidence; project decays confidence
- scout-then-lose-sight then decay; reset clears; per-slot τ; encode shape

### test_build_centerline.py — `tracks/registry.json` builder
- creates registry when absent; entry has expected fields; upsert overwrites; upsert preserves siblings; multi-track sorted keys

### test_consolidate.py — grid-search result consolidation
- experiment-data round-trip / load missing raises / valid JSON
- creates results dir; produces summary; infers summary dir; skips missing; detects varied keys

### test_cross_grid_report.py — recursive cross-grid summary collation
- discovery: only `*__summary/summary.md` under `<policy>/vX/` is treated as a cross-grid candidate
- report generation: copies summary bundles, rewrites copied summary links back to original run assets, ranks discovered grid-search families, pushes missing-reward families to the bottom, ignores prior output trees on rerun, remaps moved reward-config/weights paths, and surfaces average/best reward + generation-to-best + population-size / hidden-layer / best-run param choices

### test_redo_analytics.py — `redo_analytics.py` re-generation script
- game detection: honors explicit `training_params.game` (incl. `assetto_corsa` alias), SC2 inferred from `map_name` / `agent_race`, TMNF default, empty params → TMNF
- analytics loader: tmnf returns callables; unknown game falls back gracefully; `assetto` alias resolves to `games.assetto_corsa.analytics`
- single experiment: regenerates results.md / greedy_rewards.png; no summary without `--summary-name`; summary produced when `--summary-name` given; missing dir skipped without crash; `--no-individual` without summary raises ValueError
- multiple experiments: summary.md written; contains experiment names; individual results regenerated; `--no-individual` skips results.md but writes summary; summary dir inferred from common parent; default summary name is 'combined'; missing experiment skipped; SC2 auto-detected and run without crash; summary varied-params include reward-config differences; duplicate experiment names no longer collide in varied-key detection; malformed/unreadable reward configs are tolerated

### test_curiosity.py — ICM/RND curiosity bonuses
- ICM: reward decreases on repeat / non-negative / dim mismatch raises / β raises / η scales
- RND: reward decreases / target frozen / non-negative
- factory: `none` returns `None`; ICM / RND factories; unknown kind raises

### test_discretize_obs.py — continuous→discrete obs binning
- zero→middle bin; clipped high→max; clipped low→min; symmetry; tuple-of-int return; length matches obs_dim

### test_distributed.py — coordinator/worker protocol + HTTP server
- ComboSpec round-trip + JSON serialisable; ResultPayload round-trip + valid JSON
- payload preserves greedy_sims / throttle counts / trace / none-trace / metadata / task-metric fields / SC2 analytics fields
- numpy arrays serialised
- HTTP: serves all combos / status endpoint / result accepted + done event
- mobile monitor: `/monitor` login page is public, `/monitor/api/status` requires session login, authenticated status includes per-run queued/in-progress/done state + selected-run details
- unknown combo rejected; duplicate result ignored; stale worker re-queued; heartbeat prevents requeue
- empty queue returns immediately; unauthorized rejected
- LAN source-IP filtering: private/loopback/link-local clients allowed by default; public IPs rejected with 403 unless `allow_non_lan=True`
- game-filter: X-Worker-Game header returns matching combo; 204 when no match; preserves queue order for non-matching items
- skip endpoint: POST /skip returns in-progress item to queue; unknown item is no-op
- ComboSpec.game defaults to 'tmnf'; explicit game values round-trip correctly

### test_early_stopping.py — early-stop logic in greedy + Q loops
- stops after patience-no-improvement; patience=0 runs all; streak resets on improvement; early-stop sim recorded
- Q loop: stops on patience / patience=0 runs all

### test_env_termination.py — `_classify_termination()`
- finish / crash / hard-crash / timeout / still-running; finish > crash priority; reason key always present

### test_game_adapter.py — TMNF/TORCS/SC2/BeamNG adapter abstractions
- registry: all games registered; adapter instantiable
- TMNF: experiment_dir includes game/policy/track hierarchy, track override, track_label default+override, build_probe/build_warmup/build_extras, decorate_reward_cfg
- TORCS: experiment_dir root/dir includes game/policy/map hierarchy, track_label default+override, build_probe/warmup/extras = None
- SC2: experiment_dir includes game/policy/map hierarchy, track override, track_label, build_probe/warmup = None
- SC2 policy validation: hill_climbing/genetic/neural_net rejected with ValueError; error contains migration hint (sc2_genetic); unknown policy_params keys rejected per type (sc2_genetic, cmaes, sc2_cmaes, sc2_lstm, sc2_reinforce); valid params accepted without error; empty policy_params never raises
- BeamNG: experiment_dir / build_probe = None
- AssettoCorsa: experiment_dir / build_probe = None

### test_sc2_map_access_gate.py — Cross-process SC2 map-access serialiser (issue #254)
- single-process: first call returns 0 wait / second call within gap waits remainder / second call after gap waits 0 / `gap_s=0` short-circuits with no I/O / negative `gap_s` treated as disabled / corrupt or empty timestamp file treated as "no prior access" / future timestamp clamped to now (clock-skew robustness)
- env-var configuration: `GAMER_AI_SC2_MAP_LOCK_PATH` overrides lock path / `GAMER_AI_SC2_MAP_GAP_S` overrides gap / invalid value falls back to default with warning / negative value falls back to default / `0` disables the gate
- multi-process: three concurrent fork()-spawned workers are serialised so each consecutive grant is ≥ `gap_s` apart (POSIX-only)
- real-clock integration: a single end-to-end call against the real `time.sleep` / `time.time` confirms the test-seam wiring matches actual behaviour

### test_grid_search.py — Cartesian-product expansion + naming
- expansion: no variation / single training axis / single reward axis / cartesian product / fixed params preserved
- flat dict: contains varied / no-flat-key when not varied
- naming: no varied / single varied / negative-float `n` prefix / multiple varied / unknown key passthrough / split `<base>__<params>` into nested folder components
- local distributed helpers: launching expected `distributed.worker` subprocess commands (including custom coordinator host wiring) / cascading start-stagger between consecutive worker spawns (issue #254; first immediate, subsequent wait `start_stagger_s` each) / no stagger sleep when `start_stagger_s=0` / launch-failure cleanup for already-started workers / best-effort worker shutdown (graceful terminate + timeout kill) / non-negative integer parsing used for `--local-workers` and `distribute.local_workers`
- abbreviation coverage: every default game `training_params.yaml` + `reward_config.yaml` key has a short folder-name abbreviation; all promoted top-level policy params do too
- nested policy_params: passthrough / top-level promoted / top-level overrides nested / all keys mapped / correct names
- promoted-keys: no params returns empty / lstm hidden_size / reinforce baseline / genetic mutation_scale + mutation_share / keys in map with correct names
- format helpers: int / float strips zeros / negative float / string
- `--game` flag: default tmnf / honoured / track field / track none / unaffected by game field

### test_info_gain.py — staleness-based intrinsic reward
- initial staleness all 1; never-observed = max; just-observed near zero; grows linearly
- reward fires stale→fresh; zero when weight=0; reset restores

### test_live_monitor.py — live GUI monitor helpers
- reward-component extraction prefers per-step components and falls back to differencing cumulative episode totals
- rolling-average values are computed from the latest 5 steps
- observation grouping detects x/y pairs, indexed vectors, quadrant grids, and scalar fallbacks

### test_obs_memory.py — frame-stacking observation wrapper
- shape; reset fills initial; step shifts frames; k=1 passthrough; invalid k raises; most-recent zero-padded; clear

### test_reward.py — TMNF reward calculator + curiosity glue
- Config: defaults / custom / unknown-key raises / valid keys / partial keys
- Components: progress / no-progress / centerline quadratic / on-center zero / finish bonus / over-par penalty / no-finish-no-bonus / accel bonus / step penalty / airborne / airborne above centreline
- n_ticks scaling: centerline / speed / airborne / accel-bonus scale; finish-bonus + progress do *not* scale
- Track fields: default name / centerline path / custom / from yaml / backward-compat / default config back-compat
- Curiosity: ICM adds positive intrinsic / scales w/ n_ticks / skipped when obs missing / reset propagates / yaml accepts new keys
- `compute_with_components`: scalar matches / sum=total / keys present / progress / centerline / finish-bonus / finish-time over-par / no-finish / step-penalty / accel-bonus / curiosity zero w/o module

### test_parallel_eval.py — intra-run parallel evaluator (issue #229)
- `ParallelEvaluator` returns results sorted by `individual_idx` regardless of submission order
- matches a serial-reference evaluation byte-for-byte on dummy env + dummy policy
- `eval_episodes>1` reports the per-episode mean reward and summed total_steps
- worker crash on one individual returns `-inf` for that idx; other workers keep serving
- `close()` joins all workers (no zombies); idempotent across repeat calls; rejects post-close `evaluate()`
- empty candidate list → empty result; `n_workers=0` raises; more candidates than workers all evaluated
- determinism: identical `base_seed` produces identical results
- `episode_time_limit_s` reaches workers without crashing the dispatch
- `_maybe_build_evaluator` returns None for `n_workers=1`, raises `ValueError` for non-population policies and `q_learning` loop dispatch, caps `n_workers` at `population_size` with a warning, and accepts `cmaes` loop dispatch for `sc2_cmaes` / `sc2_lstm` / `sc2_cnn`
- SC2 binary spawn smoke test is opt-in (skipped unless `RUN_SC2_TESTS=1`); the worker mechanics are exercised entirely against the dummy env in the unit tests

### test_train_rl_signature.py — public `train_rl()` API
- accepts game+config params; accepts optional specs; accepts control flags; no legacy flat params

### test_new_best_logging.py — `_log_new_best_details` + `_print_episode_summary`
- `_print_episode_summary`: terminated/finished/truncated one-liner; `r=` and `steps=` present; laps and progress omitted
- SC2 summary formatting: scalar `outcome` (`win`/`loss`/`draw`) plus scalar `reward=` and `score=` values
- `_log_new_best_details` — empty info emits nothing
- reward components: logs all non-zero components, always includes `score` (even when 0), and explicitly logs `win_bonus`/`loss_penalty` split from terminal reward with previous-best comparison
- action frequency: one log line per action with SC2 function names; prev comparison shown
- TMNF metrics: progress; lateral offset; finish time only when `finished=True`; prev comparison for progress + offset (all on one combined line)
- SC2 kills: units + structures on one line; prev comparison; absent when key not in info; suppressed when both values zero
- SC2 game-state averages: one log line per non-zero metric; zero values omitted; prev comparison
- all five groups together emit nine lines (2 components + win/loss + 2 actions + 1 progress + 1 kills + 1 game-state)

### test_utils.py — math/state-extraction utils
- vector magnitude: zero / unit / 3D / compute_speed alias
- yaw/pitch/roll identity = 0; 90° yaw correct
- state extraction: velocity / wheels / centerline-progress / 3-entry lookahead with centerline / zero lookahead without

### test_track.py — centreline geometry helpers
- start / end / midpoint progress; nonzero lateral; on-centreline zero lateral; forward unit vector
- lookahead: returns two floats / straight-track zero heading change / finite lateral / clamps at end / opposite sign across centreline

### test_version.py — `code_version()` + git revision reporting
- `PACKAGE_VERSION` is SemVer-shaped
- `code_version()` starts with `PACKAGE_VERSION`
- when run inside a git repo: matches `<version>+g<sha7>[.dirty]` shape
- `code_version()` is cached (identical return on repeated calls)
- when `git` is unavailable: `git_revision()` returns `(None, False)` and `code_version()` falls back to bare `PACKAGE_VERSION`

## TMNF policies

Trackmania-Nations-Forever-specific code under `games/tmnf/`. Policies live
in `games/tmnf/policies.py`; the bridge to the live game is in
`games/tmnf/clients/`.

**Tested.** Every policy listed in CLAUDE.md (`WeightedLinearPolicy`,
`NeuralNetPolicy`, `EpsilonGreedyPolicy`, `MCTSPolicy`, `GeneticPolicy`,
`CMAESPolicy`, `NeuralDQNPolicy`, `REINFORCEPolicy`, `LSTMEvolutionPolicy`)
is exercised in isolation: action shape and range, deterministic forward
pass, mutation produces different weights, save/load YAML round-trips
losslessly (including replay buffer + Adam moments for DQN, σ + covariance
for CMA-ES, hidden-state reset for LSTM), `from_cfg` rejects shape
mismatches, and the optimisation loop converges on a tiny stand-in
problem (2-arm bandit, quadratic max). `RLClient`'s threading model — the
tick-window state machine, decision_idx clamping, and the finish/respawn /
hard-crash forced-commit paths — is fully covered against a `MagicMock`
TMInterface.

**Not tested.** The actual TMInterface bind to a running Trackmania process;
the `mss` window-grab + OpenCV LIDAR pipeline (only the *configuration* of
LIDAR is reached via reward-config tests, the raycast loop is not unit
tested); pywin32 keyboard injection; `.Gbx` replay parsing via `pygbx`; any
real driving on the `a03_centerline` track.

### test_weighted_linear_policy.py — linear `WeightedLinearPolicy`
- action in range; deterministic; accel/brake weight dominance; coast within threshold; left/right steer
- from_cfg roundtrip; mutated weights differ; obs_scales length matches names; action is int

### test_neural_net_policy.py — pure-numpy MLP policy
- action in range; deterministic; from_cfg roundtrip; hidden_sizes preserved; output 9 actions; mutated differs; weight matrix shapes

### test_genetic_policy.py — population evolutionary loop
- init random pop size + champion set; init from champion seeds pop
- init from champion: champion as first member / rest are distinct mutants / mutation_scale property getter+setter
- evaluate-and-evolve: champion reward / returns true on improve / false otherwise
- crossover from both parents; pop replaced after evolution
- eval_episodes: default=1 / stored / in to_cfg / cfg roundtrip / cfg default / single reward / 3-episode average / reset count
- adaptive mutation: mutation_scale logged in GreedySimResult / scale decreases on zero-improvement run / disabled leaves scale unchanged

### test_cmaes_policy.py — CMA-ES on linear weights
- defaults: pop size / μ=λ/2 / weights sum=1 / C=I at init
- init: random zero mean / from champion seeds mean / sets champion
- sample: returns count / WeightedLinearPolicy instances / fills pop_xs/ys
- update: sets champion first / true on improve / false otherwise / tracks best / generation increments / wrong reward count raises / no sample raises / mean moves
- call: raises before update / valid action after
- cfg: required keys / policy_type / save writes WL yaml
- eval_episodes: default 1 / stored / in to_cfg / clamped ≥ 1 / single reward / 3-ep average / reset count
- convergence: quadratic max / save+load roundtrip all arrays / wrong dim raises

### test_epsilon_greedy_policy.py — tabular Q-learning
- action in range; greedy picks best; update +/- reward; Bellman backup; ε decays per episode; ε floored

### test_mcts_policy.py — UCT-style online Q
- action in range; unseen state random; visit count increments; Q changes; exploitation prefers high Q; visits accumulate

### test_neural_dqn_policy.py — DQN (replay + target net)
- ReplayBuffer: push+len / circular eviction / sample shapes / w/o replacement / with replacement when small
- Policy: action shape+range / greedy discrete / random when ε=1 / buffer fills on update / ε decays / floored / target sync / weight shapes
- Cfg: roundtrip / policy_type key / on_episode_end no-op / missing keys raise / shape mismatch raises
- Bandit convergence; save/load replay buffer + Adam moments; wrong obs_dim raises

### test_reinforce_policy.py — Monte-Carlo policy gradient
- action shape; steer range; accel/brake binary; discrete
- buffers: fill / clear on episode end / empty on_episode_end no-op
- weights match hidden_sizes; buffer lengths match; weights change after update; gradient direction
- entropy_coeff = 0 vs nonzero; cfg required keys / policy_type / restore weights+hyperparams; save+reload; lidar; baseline roundtrip; wrong obs dim raises

### test_lstm_policy.py — LSTM evolution policy
- Forward: action shape / steer range / accel+brake binary / hidden updates / episode reset zeros / update no-op / different history → different action
- Flat encoding: dim correct / to_flat shape / roundtrip / zeros hidden / preserves weights / wrong size raises / mutated differs / same hidden_size / lidar roundtrip
- Cfg: required keys / policy_type / from_cfg roundtrip / save+reload
- Trainer: pop size / σ property / champion = -inf / flat dim matches template / μ=λ/2 / recomb sum=1
- Sample: count / LSTM type / fills buffer / distinct individuals
- Update: true first time / sets champion / tracks best / false on no improve / mean shifts / wrong count raises / no sample raises / σ adapts
- Call: raises before update / valid after; on_episode_end resets champion hidden state
- to_cfg keys / policy_type / save yaml / init from champion / mean→target / save+load roundtrip / wrong flat dim raises

## TMNF I/O

Split out from the TMNF policy section because it covers the
client/threading boundary rather than learning algorithms.

**Tested.** The `RLClient` bridge: that `set_input_state` is called with the
right values on normal ticks, that the action-window state machine commits
once per window and emits states only during the observation phase, that
finish/respawn and hard-crash both force an immediate commit, and that
decision-offset and window parameters are clamped and validated.

**Not tested.** That those calls actually move the car — TMInterface and the
game itself are mocked.

### test_rl_client.py — `RLClient` (game-thread bridge, action windowing)
- set_input_state called on tick / steer+accel+brake values / brake / full-left / clamped
- not called when finish-respawn pending; finish-respawn resets running flag
- shape+dtype; coast straight; decision_idx computed
- action window: first tick commits / later ticks don't / observation phase emits states / transit phase drains+puts
- pending action locked after decision tick; finish forces immediate commit; hard crash forces commit
- decision_idx clamped ≥1 / ≤ window-1 / legacy window=1 commits every tick
- window must be positive; decision_offset_pct bounds

## TORCS

`games/torcs/` — the TORCS racing-simulator integration via `gym_torcs`.

**Tested.** The 19-dim observation spec (names, scales, ordering, lidar
extension); the client's mapping from a synthetic `gym_torcs` state dict to
the observation vector and from a continuous action back to throttle / brake
/ steer (including clipping); the `TorcsEnv` Gym wrapper end-to-end against
a fake client (reset, 5-tuple step, crash termination, info keys,
close-propagation, probe / warmup specs); the TORCS reward calculator
(progress, centerline, speed, finish, accel, step penalty, n_ticks scaling,
lap wraparound); analytics smoke tests for greedy action distribution,
progress, termination reasons, weight heatmap and grid summary.

**Not tested.** The actual `gym_torcs` package or the TORCS binary;
real driving against the simulator; rendered analytics output (only that
files are written without crashing).

### test_torcs_obs_spec.py — TORCS observation spec (19-dim)
- ObsSpec instance; dim matches base; dim=19; names length; scales shape+positive; obs_spec_list match; names unique; first=speed; last=track_pos; track_edges present
- with_zero_lidar same; with_lidar extends dim+names

### test_torcs_client.py — TORCS state→obs and action mapping
- output shape; speed conversion; lateral offset; angle; progress; rpm; wheel spin; track position
- full throttle straight; full brake; steer clipped; combined accel+brake

### test_torcs_env.py — Gym env wrapper
- obs/action space shape+bounds; episode time-limit get/set; reset returns obs+info; step 5-tuple; crash terminates; info keys; close calls client.close
- discrete actions shape; probe count; warmup shape

### test_torcs_reward.py — TORCS reward calc
- defaults / custom / from_yaml / unknown raises / loads torcs config
- progress / centerline / speed / finish / accel / step penalty; n_ticks scaling; lap wraparound

### test_torcs_analytics.py — TORCS plots/report
- plot greedy action dist / progress / termination reasons / cold-start dist
- weight heatmap no-file safe; weight evolution no-weights; save plots no crash; save full report; empty experiment safe; grid summary; gs comparison progress

## SC2

`games/sc2/` — the StarCraft 2 integration via PySC2. Largest per-game
section because both the minigame and Simple64 ladder paths,
plus six policy variants and three obs encodings (flat / dict / spatial),
have to be covered.

**Tested.** All three observation specs (15-dim minigame, 46-dim ladder,
103-dim rich) and the get-spec dispatch; the minimap enemy centroid
(`minimap_enemy_cx/cy`) that enables the policy to locate the beacon even
when it is off the current camera view (beacon-idling fix); the `_action_to_call`
no-spam fix (blocked Move_screen issues select_army once, then no_op on
consecutive blocked steps); the 118-entry `FUNCTION_IDS` table and the
`DISCRETE_ACTIONS` uniform `[command × location]` grid (3 583 rows: every
spatial fn_id gets a full 8×8 = 64-cell block, every non-spatial fn_id gets 1
row); `SPATIAL_FN_IDS` derivation; race gating (`RACE_FUNCTION_IDS`,
`fn_ids_for_race()`, pairwise-disjoint race-specific sets); the SC2 reward calculator
(score delta, win bonus, loss penalty, economy weight, idle penalty, step
penalty); the SC2 client wrapper (flat-obs construction, score-delta
threading, player_relative centroid, terminal-outcome handling, ladder
visibility tracking with fog distinction, the rich-preset extractors added
in issue #135: enemy unit-type counts, shield/energy screen summaries, creep
coverage fraction, and economy-pipeline features, and `save_replay` delegation
to PySC2 including None-guard, makedirs-inside-try, and exception safety); the
SC2 env wrapper (reset / step / done / info / custom reward config /
`save_replay` passthrough); replay saving on new-best events (`_try_save_replay`
for single-episode loops, candidate-pattern helpers `_save_candidate_replay` /
`_finalize_candidate_replay` / `_discard_candidate_replay` for multi-episode
loops: sequential naming, candidate-file exclusion from counts, exception
safety throughout); the APM limiter (`ApmLimiter`
token-bucket: construction, no-op exemption, burst cap, rolling refill,
env integration including throttled action substitution and per-episode
counters); the full lifecycle of
`SC2LinearPolicy` and the multi-head `sc2_genetic` trainer (init, crossover
from both parents, evolution, champion YAML round-trip, `from_cfg` defaulting
missing features to zero); the masked DQN with action-availability masking,
including a regression test that an illegal action is never bootstrapped from;
the CNN encoder + CMA-ES variant with spatial obs; the `play_sc2.py` script's
policy loading, episode loop, lifecycle hooks and outcome handling; a
Simple64-specific integration suite that runs every supported policy through one
training-loop iteration against a mocked env, plus trainer-state save/load
round-trips for cmaes and neural_dqn; and the SC2-specific analytics module
(`SUPPORTS_THROTTLE`/`SUPPORTS_PATH` flags, action-frequency breakdown, obs
feature averages, spatial heatmap, outcome breakdown, and the full
`save_experiment_results` integration including that no racing plots appear).

**Not tested.** PySC2 against the actual Blizzard SC2 binary; real
1v1 games against the built-in bot; minimap rendering; the deferred
fog-of-war belief machinery beyond the standalone `test_belief.py`
encoder; long-horizon RL convergence on Simple64 (loops are run for a
handful of iterations only).

### test_sc2_obs_spec.py — SC2 obs spec
- minigame dim (15); ladder dim (46); rich dim (103 = exact breakdown); ladder extends minigame; default = minigame; get_spec for minigame / ladder; minigame count; obs_names match dims
- minimap_enemy_cx/cy present in all presets (minigame, ladder, rich)
- rich spec contains new rich-only feature names (selected_avg_shields/energy, screen_visibility_frac, screen_unit_density_aa_mean, self_weapon_cooldown_mean); not present in ladder
- alert_count present in ladder and rich; absent from minigame

### test_sc2_actions.py — discrete action grid + race gating
- `TestSC2Actions`: shape (`_N_SPATIAL × N² + _N_NON_SPATIAL` rows, 4 cols) / dtype float32 / x+y in unit square
- row 0 = no_op; row 1 = select_army; fn_idx_row_layout — each fn_id has N² rows if spatial, 1 row otherwise, in ascending fn_idx order
- spatial actions span unit square (x and y both reach near 0 and near 1) for every spatial fn_id
- Move_screen cells unique (scoped to the Move_screen N² block, not all of DISCRETE_ACTIONS)
- probe actions count=5 / shape (4,) / include no_op; warmup shape (4,) / is select_army; function_ids table complete
- SPATIAL_FN_IDS contains exactly the fn_ids whose names end in `_screen` or `_minimap`
- `TestRaceGating` (9 tests): all four race keys exist in RACE_FUNCTION_IDS; each race's ids are a subset of FUNCTION_IDS; random race = all fn_ids; race-specific sets (_TERRAN / _PROTOSS / _ZERG) are pairwise disjoint; unknown race falls back to all fn_ids; Terran has Build_Barracks_screen (8) not Build_Nexus_screen (50); Protoss has Build_Nexus_screen (50) not Build_Barracks_screen (8); Zerg has Build_Hatchery_screen (82) not Build_Barracks_screen (8); all three named races include Move_screen (2) and no_op (0)
- `TestActionToFunctionCall`: fake PySC2 module validates encoding branches — `_quick` emits queue-only args, `select_point_screen` and `select_rect_screen` emit screen coords, `_minimap` actions scale with `minimap_size` (not `screen_size`), and `_screen` actions keep using `screen_size`

### test_sc2_reward.py — SC2 reward calc
- defaults; from_yaml; unknown raises; loads bundled config
- score delta; step penalty only; step penalty n_ticks scaling; win bonus; loss penalty; no-outcome no bonus; economy weight; idle penalty when idle / not when busy
- idle bonus: uses unit-aware max attack range when available and only fires at 95% inside that range (e.g. 19/20 fires, 20/20 skips); skipped for non-no_op / far enemy / no enemy / no self; disabled by default; n_ticks scaling
- attack_move_bonus: fires when Attack_screen target is on empty ground with enemies visible; skipped for Move_screen / no enemy; disabled by default; n_ticks scaling; persists across consecutive no_op steps until another non-no_op action is issued
- click_attack_bonus: fires when Attack_screen target is on/near enemy centroid; skipped when target far from enemy / no enemy; disabled by default; n_ticks scaling; persists across consecutive no_op steps until another non-no_op action is issued
- cooldown: default=8; same target always fires; rapid switch withheld; fires again after cooldown elapsed; reset() clears state; both bonuses mutually exclusive
- movement shaping: exploration bonus fires on first `Move_screen` command when the unit centroid is in a new grid cell (actual-visit tracking, default 8×8); visited cells are updated on any step with visible friendlies (not only `Move_screen`), so later move commands do not re-earn bonus for already-occupied cells; bonus does not fire a second time when centroid stays in the same cell; bonus fires again when centroid moves to a new cell; exploit regression — spamming move commands to far targets while units stay put yields at most one bonus; no bonus when no friendly units are visible; repeat penalty still fires for tiny command moves below the distance threshold; command move at threshold does not trigger the repeat penalty; penalty for moving to friendly centroid; self-penalty skipped when no friendly units are visible; `TestSC2IdleBonus._make_calc` disables move-shaping terms so idle-bonus isolation tests are not affected
- exploration decay (#262): a cell vacated for longer than `move_exploration_decay_steps` is rewarded again on return; a centroid that never leaves its cell is rewarded once even past the decay window (refreshes its own timestamp each step → no command-spam farming); `decay_steps == 0` keeps once-per-episode behaviour (no re-reward); `move_exploration_grid_size` controls cell granularity (a 2×2 grid merges centroids an 8×8 grid separates)
- attack_friendly_penalty: fires when Attack_screen targets near friendly centroid; skipped for target far from friendly / no friendly on screen / Move_screen; disabled when zero; n_ticks scaling; appears in components dict; default is strongly negative
- unit_loss_penalty: fires per unit lost (army_count drop); zero when no loss / army grows; disabled by default; appears in components dict
- damage_taken_penalty: fires per HP+shield point lost across visible friendlies; zero when unchanged / healing; safe default when info keys absent; disabled by default; appears in components dict
- passive_under_fire_penalty: fires on no_op or Move_screen when enemies within attack range; suppressed by Attack_screen; skipped when enemy out of range / no enemy / no self; respects explicit self_attack_range_px; n_ticks scaling; disabled by default; appears in components dict
- small_selection_bonus: fires on unit-targeted commands when selection is a single unit or less than half of visible friendly units; skipped for non-unit-targeted actions and exactly-half selections; appears in components dict
- attack_bonus: fires on any Attack_screen (fn_idx 3) regardless of target type; persists across carried attack no_op steps; skipped for pure no_op (no prior attack); disabled by default; appears in components dict
- components sum: new terms included in total (extended sum test)

### test_sc2_client.py — PySC2 client wrapper
- minigame flat obs shape; score-delta threading; player_relative centroid; terminal outcome recorded
- ladder flat obs shape; visibility tracking; fogged ≠ visible; ladder terminal outcome; non-terminal = None
- info dict includes unit-aware `self_attack_range_px` derived from visible friendly `feature_units` (uses max friendly range from curated PySC2 unit-range table)
- `total_self_hp`: info dict sums visible friendly health+shield from `feature_units`; also exposes `visible_self_unit_count`; helper returns 0 for no-self / missing `feature_units` / short rows
- rich extractors (#135): enemy unit-type counts (owner==4 only; neutral owner==3 excluded; ally owner==2 excluded; missing field; unknown type); shield/energy (self shield mean, no units, None screen); creep (half coverage, no creep, None minimap); economy pipeline (upgrade count, build queue, cargo, all missing); rich spec contains new names; ladder spec unchanged
- selected-unit extras: `selected_count` and shields + energy from cols 3/4 of single/multi_select; empty selection → zeros; short rows don't crash
- screen_visibility_frac: all visible → 1.0; half visible → 0.25; fogged not counted; None screen / missing layer → 0
- screen_unit_density_aa_mean: mean of unit_density_aa layer; zero layer; None screen / missing layer → 0
- self_weapon_cooldown_mean: mean for alliance==1 from col 25; all ready → 0; no self units → 0; missing feature_units → 0; too few cols → 0
- alerts: empty array → 0; one alert → 1; two alerts → 2; missing key → 0; None value → 0; alert_count present in ladder names
- minimap enemy centroid: minimap_enemy_cx/cy computed from player_relative==4 layer; correct when beacon present; zero when no beacon on minimap (edge case)
- action fallback (#124, beacon-idling fix): blocked Move_screen → select_army immediately, then no_op on short blocked streaks with periodic select_army retries on long streaks; legal Move_screen resets the blocked-streak tracker; no_op action passes through unchanged
- action-mask caching overhead (#140): _available_actions_features uses module-level cache (no per-call pysc2 import); correctness test with injected cache; deterministic regression asserting _get_pysc2_id_to_fn_idx is called exactly once per _available_actions_features invocation (not once per FUNCTION_IDS entry)
- score_features field-name access: named NamedNumpyArray access takes priority over positional; score→score_total rename applied; missing score array → all zeros
- last_fn_idx property: select_army substitution → idx=1; consecutive block → no_op idx=0; passthrough → requested idx
- realtime param: default False stored; True stored; forwarded as kwarg to SC2Env constructor

### test_sc2_env.py — SC2 env wrapper
- minigame obs space; action space shape+bounds; ladder obs space; episode time-limit get/set
- reset returns obs+info; step 5-tuple; score-delta reward; done terminates; loss outcome
- close calls client.close; info keys; prev_score threaded; prev_army_count + prev_total_self_hp seeded from reset and updated each step; executed `last_fn_idx` (not merely requested fn_idx) drives reward-shaping metadata; skipped-frame counters (default 0, per-step accumulation from game_loop deltas); custom reward config
- end-screen analytics: series absent on mid-episode step; present on terminal step; supply_capped_fraction correct; army series value; resource series sums minerals+vespene; starting units excluded from build order; new units produce events; empty build order when no unit_counts

### test_sc2_replay.py — SC2 replay saving on new-best events (issue #210)
- `SC2Client.save_replay`: returns None when SC2 env not running; delegates to pysc2 save_replay with correct dir and prefix keyword; `os.makedirs` inside try block so directory creation failures are swallowed; swallows SC2 exceptions and returns None
- `SC2Env.save_replay`: thin delegation to client
- `_try_save_replay` (single-episode loops): no-op for envs without save_replay (non-SC2 games); first new-best → `_best-01` prefix; second new-best → `_best-02` when one confirmed .SC2Replay exists; candidate (`_`-prefixed) files excluded from sequential count; replay_dir always `<experiment_dir>/replays/`; only `.SC2Replay` files counted; exception swallowed
- `_save_candidate_replay`: no-op for non-SC2; calls save_replay with `_candidate` prefix; exception swallowed, returns None
- `_finalize_candidate_replay`: no-op when path is None or file missing; renames to next sequential `_best-N.SC2Replay`; candidate files excluded from confirmed-best count; two confirmed → `_best-02`
- `_discard_candidate_replay`: no-op on None or missing path; deletes existing file

### test_sc2_apm_limiter.py — token-bucket APM limiter + SC2Env integration
- Construction: valid / zero/negative max_apm raises / zero/negative burst_s raises / max_tokens formula / starts full
- Basic behaviour: no-op always allowed; no-op free (no token consumed); first action passes; second blocked when empty; refills over time; tokens capped at max; reset refills; burst capacity; high-APM burst; default fn_idx consumes token; burst-budget protection mode caps non-dangerous bursts at steady one-second capacity
- Rolling budget: 300 APM over 60 s allows ~300 total; first second capped at burst window
- Env integration (disabled): no limiter attribute; action passed unchanged; apm_throttled=False; episode count stays zero
- Env integration (enabled): limiter created; first action passes; second throttled to no_op; no_op never throttled; throttled-steps counter accumulates; counter resets on new episode; action passes after refill; burst budget remains protected

### test_sc2_belief_integration.py — fog-of-war belief system wired into SC2Env (issue #111)
- obs shape = base + 192 dims with `enable_belief=True` for both minigame and ladder maps
- `enable_belief=False` leaves obs shape unchanged
- reset obs: belief enc all-zero, staleness all-ones; second reset clears prior state
- step appends dims even when `minimap_vis` absent; enc non-zero + staleness drops after visible step
- scout reward > 0 on first visit with fully-visible minimap; zero when no visible regions
- `episode_reward_components` always contains `scout` key; accumulates across steps

### test_sc2_cmaes_policy.py — `SC2CMAESPolicy` (CMA-ES over multi-head linear policy)
- Dimension: θ = (N_FUNCTION_IDS + N_SPATIAL_ROWS) × obs_dim for minigame and ladder obs specs
- Sample population: count matches λ / all individuals are SC2MultiHeadLinearPolicy
- Mechanics: update_distribution before sample raises / σ adapts across generations / champion improves monotonically / wrong reward count raises
- Call: raises before first generation / returns valid 4-vec after one generation
- Masking: restricts fn_idx to available set / fallback to no_op when set empty / updated via update() kwargs / no masking when None
- Serialisation: champion YAML round-trip lossless / trainer-state npz round-trip / dim mismatch raises / initialize_from_champion sets mean / initialize_random zeros mean

### test_sc2_lstm_policy.py — `SC2LSTMPolicy` + `SC2LSTMEvolutionPolicy`
- Structure: hidden state zero at init / W_out shape = (N_FUNCTION_IDS+N_LSTM_SPATIAL_CELLS, hidden_size) / flat_dim formula matches both minigame and ladder / to_flat length correct / with_flat round-trip / wrong size raises
- Action: shape (4,) / fn_idx in [0, N_FUNCTION_IDS) / x,y in [0,1] / hidden state advances after step
- Masking: never selects unavailable fn / set via on_episode_start info / updated via update kwargs / fallback to no_op when all masked
- Hidden state reset: reset_on_episode=True zeros state on episode start + end / reset_on_episode=False carries state across resets
- Serialisation: to_cfg / from_cfg round-trip lossless / save / load round-trip / policy_type = "sc2_lstm" / explicit `race=` argument to `from_cfg` takes precedence over any stored cfg race
- Evolution: population size / individuals are SC2LSTMPolicy / call raises before generation / champion set after one generation / σ adapts / wrong reward count raises / flat_dim mismatch raises / initialize_from_champion sets mean / on_episode_start forwarded to champion / save writes yaml / trainer-state round-trip / dim mismatch raises

### test_sc2_genetic_policy.py — `SC2MultiHeadLinearPolicy` + genetic trainer
- Weight shapes (fn / spatial × minigame / ladder); flat dim (mini/ladder); explicit weights stored
- Call: returns 4-vec / fn_idx range / spatial unit range / queue=0 / max-fn / max-spatial
- Cfg: keys / *_weights suffix / values are obs-name dicts / from_cfg roundtrip / yaml lossless / missing default zero
- Flat: length / roundtrip / mutated differs / preserves spec / weights differ / share=0 unchanged
- Genetic: pop / elite_k / eval_episodes / head names cover rows
- Init random: pop size / SC2 policies / champion set / pop θ dim
- Init from champion: pop size / champion / first member is champion / rest are mutants / all SC2 policies
- Crossover: valid policy / from both parents / θ dim / weights only from parents
- Evolve: champion reward / true on improve / false otherwise / elite count preserved / new members are SC2
- Save: yaml / champion lossless / cfg policy_type=sc2_genetic / from_cfg roundtrip / restores champion / no champion key OK
- Call: 4-vec after init / raises before init
- Available-actions masking: no-mask selects highest fn / None by default / masking blocks unavailable fn / selects best available / on_episode_start caches ids / no key clears mask / None info clears mask / update caches ids / no key in update leaves unchanged / mask applied after on_episode_start / mask applied after update / empty set falls back to no_op

### test_sc2_neural_net_policy.py — hill-climbing MLP policy for SC2
- Action: shape (4,) / fn_idx range / x+y unit square / queue binary
- Determinism and cfg roundtrip: deterministic calls / from_cfg action-equivalent
- Params and weights: hidden_sizes preserved / layer shapes match architecture
- Mutation and masking: mutated weights differ / unavailable fn_idx masked / update caches available_fn_ids

### test_sc2_neural_dqn_policy.py — masked DQN for SC2
- fn_idx_for_cell: centre=select_army / others=move_screen / consistent / int
- Available mask: None=all true / empty=all false / select_army only / move_screen only / both / dtype bool
- Masked replay buffer: push+len / default mask all true / 6-tuple sample / mask shape / preserves mask / circular eviction
- Policy: illegal never picked greedy / picks best / random respects mask / all-masked fallback / mask cached from update info / from None info / on_episode_start resets / buffer is masked / mask stored / all-true when no info / fills on update / gradient step runs / illegal Q not bootstrapped
- Cfg: policy_type / from_cfg roundtrip / trainer-state roundtrip / masks preserved / backward-compat no masks / shape mismatch raises
- available_fn_ids: None when missing / key always present

### test_sc2_cnn_policy.py — CNN feature extractor + CMA-ES variant
- Conv layer: output shape / ReLU zeros negatives
- Pool: output shape / uniform input preserved
- CNN: flat_dim formula / varies w/ channels / forward shapes / callable returns 4-vec / with_flat roundtrip / wrong size raises / non-dict obs raises / flat-concat dim
- CMA-ES: pop size / sample correct count / individuals callable / update returns bool / champion improves / champion callable / wrong rewards raises / no sample raises / σ adapts / trainer-state roundtrip / save+load champion
- Spatial obs: flat space when no layers / dict space when layers / spatial shape matches channels / reset dict obs / fills zeros when none / step dict obs / spatial in info / normalised / no spatial when no layers

### test_sc2_reinforce_policy.py — REINFORCE policy for SC2
- Action: 4-vec shape / fn_idx in range / x+y in unit square / queue=0
- Buffers: episode buffer fills / clears on end / empty end no-op
- Gradient: weights change after update / direction improves expected action
- Available-actions masking: illegal fn_idx masked out / mask updates from info kwarg
- Serialisation: cfg keys / policy_type / from_cfg roundtrip / save+reload / wrong obs dim raises

### test_sc2_eval.py — `--eval` evaluation mode
- CLI validation (real main.py parser): num_episodes 0/negative rejected; 1 accepted; eval_speed 0 rejected; positive accepted; --play/--eval mutually exclusive; cheater_easy/elite rejected; valid names accepted
- _print_action_breakdown: fn names present; 70/30% correct; substitution line shown when nonzero / hidden when zero; zero total_steps no divide-by-zero
- _print_aggregate_summary: 66.7% win rate for 2/3 wins; 100% all wins; single episode no crash
- _run_episode: policy called each step; step count matches env steps; on_episode_start(info=<dict>) — info kwarg present with available_fn_ids; update(prev_obs, action, reward, next_obs, done, info=info) — shapes, available_fn_ids, done=True on last step; outcome from terminal info; substitution counted when executed≠requested; no substitution when match; cumulative reward summed

### test_sc2_play.py — `play_sc2.py` script
- Missing weights raises; loads sc2_multi_head for sc2_genetic / correct weights / neural_dqn / reinforce / lstm; cmaes no policy_type → SC2Linear; unknown → SC2Linear
- Outcome handling: win / loss / draw / none
- Episode loop: calls policy each step / client until done / on_episode_start(info=<dict>)+end / works without lifecycle hooks
- Play-mode flag: stored / default false; `make_sc2_env` lazy on reset
- Game loop: present in info / value extracted from obs

### test_sc2_simple64_training.py — Simple64 ladder integration
- Obs: 21 dim; reset shape; step 5-tuple
- Outcomes: win+bonus / loss+penalty; economy reward flows through
- Action: fn_idx valid / xy continuous / queue binary / shape
- Pop usage: sc2_genetic uses SC2Linear / cmaes offspring use SC2Linear; SC2Linear and sc2_genetic action shape
- Per-policy on Simple64: epsilon_greedy / mcts / neural_dqn shape+update / cmaes sample+update / reinforce shape+episode / lstm shape / lstm-evolution sample+update
- Training loops (mocked env): sc2_genetic / cmaes / neural_dqn / reinforce / lstm
- Trainer state roundtrips: cmaes / neural_dqn

### test_sc2_analytics.py — SC2-specific analytics plots and flags
- `SUPPORTS_THROTTLE=False` / `SUPPORTS_PATH=False` flags
- `GreedySimResult` new fields: `action_counts` / `obs_averages` / `xy_hist` / `skipped_frames` — default None; stored correctly
- `GreedySimResult` end-screen fields: `supply_capped_fraction` / `build_order` / `army_count_series` / `resource_series` — default None; stored correctly
- `plot_action_frequency`: renders to file / skips when no data / skips when no sims / single fn_idx
- `plot_obs_averages`: renders to file / skips when no data / skips when all-zero / unknown feature key safe
- `plot_spatial_heatmap`: renders to file / skips when no data / skips all-zero hist / partial None sims ignored
- `plot_outcome_breakdown`: renders to file / skips when all None / skips when no sims / win+loss ladder
- `plot_skipped_frames`: renders to file / skips when all None / zero-only still renders
- `plot_supply_capped`: renders to file / skips when all None / skips when no sims / zero fraction renders
- `plot_resource_series`: renders to file / uses best sim / skips when no series / skips when no sims / falls back to last sim when none improved
- `plot_army_count`: renders to file / skips when no series / skips when no sims
- `plot_build_order`: renders to file / skips when no build order / skips when no sims / single unit type / multiple unit types
- `save_experiment_results`: writes results.md / writes SC2 plots (incl. skipped_frames) / mentions game / no crash empty sims / no racing plots written
- `plot_reward_component_breakdown` (framework): diverging stacked bar per sim — renders to file / skips when no component data / positive stack above zero / negative stack below zero / zero-everywhere components omitted (tested in `test_analytics_task_metrics.py`)
- `plot_gs_reward_component_breakdown`: cross-experiment diverging horizontal bar — renders to file / skips when no component data / skips empty runs / skips all-zero components / single experiment / written by `save_grid_summary` / linked in summary.md
- `save_experiment_results` now also writes `reward_component_breakdown.png` (regression-guarded in `test_writes_sc2_plots`)
- `save_grid_summary`: forwards config-normalized rewards **and per-component contributions** using `v / max(abs(weight), 1.0)` — weights ≥ 1.0 are divided (making large-weight components comparable across grid-search runs), weights < 1.0 use the raw value (which already encodes the weight, so dividing would amplify by ×1000); wires SC2 extra-plot hook into framework summary generation; covers no-components fallback, multi-sim normalization, malformed YAML fallback, non-mapping YAML fallback, non-numeric weight fallback, allow-listed `scout` component (no unmapped-key warning), step_penalty with sub-1.0 weight passes through raw (-0.5 not -500), idle_penalty with sub-1.0 weight passes through raw, any sub-1.0 weight (0.0001 or 0.001) both use scale=1.0, the new `unit_loss` / `damage_taken` / `passive_under_fire` components normalize through their config weights without unmapped-key warnings, realistic positive reward stays positive after normalization (313.5 ✓), and emits SC2 cross-run charts + summary links; passes `task_metric_fn`, `task_metric_fmt` (percentage formatter) to framework; `attack_bonus` mapped to `attack_bonus` config key in normalisation
- `_sc2_task_metric`: empty sims → 0.0; win+finish counted as success; loss/timeout/None/other not counted; all-wins → 1.0; `_GS_SUCCESS_REASONS` constant contains win+finish, excludes loss+timeout

## CLI / misc

The entry point in `main.py` and the Assetto Corsa adapter, which is recent
enough to only have a smoke test.

**Tested.** That `main.py --game <name>` accepts every supported choice,
rejects unknown ones, exposes the option in `--help`, accepts `--track`,
and dispatches to the right runner per game (`run_one` for tmnf / beamng /
car_racing / torcs / sc2, `run_assetto` for assetto, with a clear error
when the optional dependency is missing); that the Assetto Corsa adapter's
obs spec, env wrapper, reward calc and a 5-episode training loop all run
against a stubbed client.

**Not tested.** The interactive CLI itself (no terminal harness); the real
Assetto Corsa shared-memory client.

### cli/test_game_flag.py — `--game` CLI flag in `main.py`
- default tmnf; all valid choices accepted; invalid → SystemExit; help text mentions flag; `--track` accepted; main parser has all choices
- dispatch: tmnf / beamng / car_racing / torcs / sc2 → run_one; assetto → run_assetto; assetto missing dep → ValueError; adapter experiment_dir contains game name

### assetto_corsa/test_smoke.py — Assetto Corsa smoke tests (against fake client)
- obs spec dimensions match base obs_dim; env reset obs shape; step 5-tuple finite reward; info reflects current step; env terminates on finish; vision features; reward calc finite; 5-episode training loop with linear policy

---

## Integration tests (`tests/integration/`)

End-to-end tests that spin up a real gymnasium environment (no mocking) and
exercise actual game physics.  Marked `integration` so they are excluded from
the fast unit-test suite.  Run by the `integration-tests` workflow after PR
approval (final gate before merge to `main`); also available on demand via::

    pytest tests/integration/ -m integration -v

The ``integration-tests`` workflow only runs the suites relevant to the files
that changed in the PR:

- Changes confined to ``games/sc2/`` (or ``tests/integration/test_sc2.py``)
  trigger only the **sc2** job.
- Changes confined to ``games/car_racing/`` (or
  ``tests/integration/test_car_racing.py``) trigger only the **car-racing** job.
- Changes to shared code (``framework/``, ``main.py``, ``grid_search.py``,
  ``policies.py``, ``analytics.py``, ``config/``, ``pyproject.toml``) trigger
  **both** jobs because they could affect either game.
- A manual ``workflow_dispatch`` run always executes **all** suites regardless
  of path changes.

**Requires** `gymnasium[box2d]` (`pip install gymnasium[box2d]`).  The tests
are skipped gracefully with `pytest.mark.skipif` when the extra is absent.

CarRacing is the only game in this repository that can run headless on a CPU-only
GitHub runner without an external binary or display server: it uses the
`Box2D` physics engine (pure Python/C) and `pygame-ce` for its renderer,
which is never called in headless mode.

### integration/test_car_racing.py — CarRacing real-env end-to-end tests

**Tested.** That `CarRacingEnv.reset()` returns a float32 array of the right
shape; that reset is repeatable under the same seed; that `step()` returns a
valid 5-tuple with finite rewards; that `info['native_reward']` and
`info['termination_reason']` are populated correctly; that episodes truncate
within the step limit; that total accumulated reward is finite; that multiple
reset/step cycles leave no leaked state.  Three training-loop tests run 1
hill-climbing sim, 1 genetic generation (population 2), and 1 ε-greedy episode
against the real CarRacing environment to verify the full stack end-to-end.

- basics: reset obs shape; seed repeatability; step 5-tuple; reward finite; obs shape consistent; termination reason set; native_reward present; close idempotent
- full episode: terminates within step limit; total reward finite; multiple resets safe
- training loop: hill_climbing 1 sim; genetic 1 generation (pop=2); epsilon_greedy 1 episode

### integration/test_sc2.py — SC2 real-binary end-to-end tests

**Requires** the Blizzard SC2 headless Linux binary (4.10), PySC2 mini-game
maps in `$SC2PATH/Maps/mini_games/`, and the `pysc2` Python package.  Skipped
gracefully when any of these are absent.  The CI workflow downloads the binary
and maps automatically; locally set `SC2PATH` and run::

    pytest tests/integration/test_sc2.py -m integration -v

**Tested.** SC2Client low-level: `reset()` returns a numpy obs + info dict;
`step()` returns a valid 4-tuple (obs, score, done, info); `select_army` →
`Move_screen` sequence works.  SC2Env lifecycle: `reset()` returns correct obs
dimension; `step()` returns gymnasium 5-tuple with finite rewards; episode
terminates within time limit; `close()` is idempotent; multiple resets mid-episode
are safe.  Full episode: varied `select_army` + `Move_screen` actions run to
completion with finite reward; `info["score"]` present.  Training loop: 1
generation of `SC2GeneticPolicy` (pop=2) and 1 `EpsilonGreedyPolicy` episode
both execute end-to-end against the real SC2 binary.

- SC2Client basics: reset returns obs+info; step returns 4-tuple; select_army then Move_screen
- SC2Env lifecycle: reset obs shape; step 5-tuple; reward finite; obs shape consistent; episode terminates; close idempotent; multiple resets
- full episode: varied actions; info contains score
- training loop: genetic 1 generation (pop=2); epsilon_greedy 1 episode

## Why tests run fast

These tests look heavy because of the names ("training loop", "env reset", "DQN convergence") but operationally they're almost all pure-Python unit tests with zero external I/O:

1. **No game binaries are launched.** TMInterface, the SC2 binary, and TORCS are never started. `RLClient`, `SC2Client`, the SC2 env, and the TMNF env are all driven through fakes and `MagicMock` patches (e.g. `test_rl_client.py`, `test_env_termination.py`, `test_sc2_play.py`). The single "five-episode training loop" smoke test (`assetto_corsa/test_smoke.py`) runs against a stubbed client.
2. **All "policies" are pure numpy.** No PyTorch, no TensorFlow, no GPU. The DQN, REINFORCE, LSTM, CMA-ES, CNN, and SC2 multi-head policies are hand-rolled numpy with hidden sizes like `[8, 8]` or `hidden_size=4` in tests. Forward+backward passes are sub-millisecond.
3. **Tiny tensors.** Where convergence is asserted (`test_neural_dqn_policy.test_bandit_convergence`, `test_cmaes_policy.test_converges_toward_quadratic_maximum`, `test_reinforce_policy.test_gradient_direction`), the problem is a 2-arm bandit or a quadratic — a few hundred steps on tiny vectors.
4. **Whole files are config / dataclass tests.** `test_grid_search.py`, `test_reward.py`, `test_sc2_genetic_policy.py`, `test_torcs_obs_spec.py`, `test_analytics_task_metrics.py`, `test_game_adapter.py` are mostly "from_yaml round-trip / shape / default-value / cartesian product" — microseconds each.
5. **No matplotlib rendering.** TORCS analytics tests use `Agg` (non-interactive) and dump to `tmp_path`; `test_analytics_no_matplotlib.py` explicitly checks the import path that *avoids* it.
6. **Filesystem work uses `tmp_path`** (RAM-backed `/tmp`), and the only network is `test_distributed.py` binding `localhost` for HTTP coordinator tests — which is why that's the one file with `time.sleep` and is still milliseconds because it talks to itself.
7. **Heavy collection work is amortised.** `pytest`'s collection phase is a small share of wall clock; once collected, the assertions run in seconds.

The suite contains no tests that wait on a game tick, a network packet, or a GPU.

The integration tests in `tests/integration/` are excluded from the fast unit-test run. CarRacing tests run real Box2D physics and take ~2 s; SC2 tests launch the Blizzard headless binary and take ~1–3 minutes for test execution (the CI workflow additionally downloads the ~2 GB binary once during the setup step).
