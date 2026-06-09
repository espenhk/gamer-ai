# Tests

- [Coverage at a glance](#coverage-at-a-glance)
- [Framework / shared](#framework--shared)
  - [test\_cmaes\_distribution.py — `CMAESDistribution` pure-math unit tests](#test_cmaes_distributionpy--cmaesdistribution-pure-math-unit-tests)
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
  - [test\_game\_adapter.py — TMNF/TORCS/SC2/BeamNG/iRacing adapter abstractions](#test_game_adapterpy--tmnftorcssc2beamngiracing-adapter-abstractions)
  - [test\_grid\_search.py — Cartesian-product expansion + naming](#test_grid_searchpy--cartesian-product-expansion--naming)
  - [test\_iracing\_controller.py — iRacing action injection controller](#test_iracing_controllerpy--iracing-action-injection-controller)
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
  - [test\_policy\_registry.py — `POLICY_REGISTRY` + `BasePolicy` registry machinery](#test_policy_registrypy--policy_registry--basepolicy-registry-machinery)
  - [test\_dqn\_upgrades.py — Double-DQN / Huber / gradient clipping](#test_dqn_upgradespy--double-dqn--huber--gradient-clipping)
  - [test\_sb3\_policies.py — Stable-Baselines3-backed deep-RL policies](#test_sb3_policiespy--stable-baselines3-backed-deep-rl-policies)
  - [test\_alphazero\_mcts.py — AlphaZero-style model-based MCTS](#test_alphazero_mctspy--alphazero-style-model-based-mcts)
  - [test\_sc2\_legacy\_names\_rejected.py — SC2 bare legacy policy names rejected](#test_sc2_legacy_names_rejectedpy--sc2-bare-legacy-policy-names-rejected)
- [TMNF policies](#tmnf-policies)
  - [test\_weighted\_linear\_policy.py — linear `WeightedLinearPolicy`](#test_weighted_linear_policypy--linear-weightedlinearpolicy)
  - [test\_neural\_net\_policy.py — pure-numpy MLP policy](#test_neural_net_policypy--pure-numpy-mlp-policy)
  - [test\_genetic\_policy.py — population evolutionary loop](#test_genetic_policypy--population-evolutionary-loop)
  - [test\_cmaes\_policy.py — `CMAESPolicy` (framework) via TMNF factory](#test_cmaes_policypy--cmaespolicy-framework-via-tmnf-factory)
  - [test\_epsilon\_greedy\_policy.py — tabular Q-learning](#test_epsilon_greedy_policypy--tabular-q-learning)
  - [test\_ucb\_q\_policy.py — UCB1 online Q (formerly mcts)](#test_ucb_q_policypy--ucb1-online-q-formerly-mcts)
  - [test\_neural\_dqn\_policy.py — `DQNPolicy` (framework) + `ReplayBuffer`](#test_neural_dqn_policypy--dqnpolicy-framework--replaybuffer)
  - [test\_reinforce\_policy.py — `REINFORCEPolicy` (framework) Monte-Carlo PG](#test_reinforce_policypy--reinforcepolicy-framework-monte-carlo-pg)
  - [test\_lstm\_policy.py — `LSTMCore` + `LSTMEvolutionPolicy` (framework)](#test_lstm_policypy--lstmcore--lstmevolutionpolicy-framework)
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
  - [test\_sc2\_tech\_tree.py — hardcoded tech-tree preconditions (issue #346)](#test_sc2_tech_treepy--hardcoded-tech-tree-preconditions-issue-346)
  - [test\_sc2\_reward.py — SC2 reward calc](#test_sc2_rewardpy--sc2-reward-calc)
  - [test\_sc2\_client.py — PySC2 client wrapper](#test_sc2_clientpy--pysc2-client-wrapper)
  - [test\_sc2\_env.py — SC2 env wrapper](#test_sc2_envpy--sc2-env-wrapper)
  - [test\_sc2\_replay.py — SC2 replay saving on new-best events (issue #210)](#test_sc2_replaypy--sc2-replay-saving-on-new-best-events-issue-210)
  - [test\_sc2\_apm\_limiter.py — token-bucket APM limiter + SC2Env integration](#test_sc2_apm_limiterpy--token-bucket-apm-limiter--sc2env-integration)
  - [test\_sc2\_belief\_integration.py — fog-of-war belief system wired into SC2Env (issue #111)](#test_sc2_belief_integrationpy--fog-of-war-belief-system-wired-into-sc2env-issue-111)
  - [test\_sc2\_cmaes\_policy.py — `SC2CMAESPolicy` (CMA-ES over multi-head linear policy)](#test_sc2_cmaes_policypy--sc2cmaespolicy-cma-es-over-multi-head-linear-policy)
  - [test\_sc2\_lstm\_policy.py — `SC2LSTMPolicy` + `SC2LSTMEvolutionPolicy` (thin framework subclass)](#test_sc2_lstm_policypy--sc2lstmpolicy--sc2lstmevolutionpolicy-thin-framework-subclass)
  - [test\_sc2\_genetic\_policy.py — `SC2MultiHeadLinearPolicy` + genetic trainer](#test_sc2_genetic_policypy--sc2multiheadlinearpolicy--genetic-trainer)
  - [test\_sc2\_hierarchical\_policy.py — `SC2HierarchicalLinearPolicy` + `SC2HierarchicalGeneticPolicy` (issue #388)](#test_sc2_hierarchical_policypy--sc2hierarchicallinearpolicy--sc2hierarchicalgeneticpolicy-issue-388)
  - [test\_sc2\_neural\_net\_policy.py — hill-climbing MLP policy for SC2](#test_sc2_neural_net_policypy--hill-climbing-mlp-policy-for-sc2)
  - [test\_sc2\_neural\_dqn\_policy.py — masked DQN for SC2](#test_sc2_neural_dqn_policypy--masked-dqn-for-sc2)
  - [test\_sc2\_cnn\_policy.py — CNN feature extractor + CMA-ES variant](#test_sc2_cnn_policypy--cnn-feature-extractor--cma-es-variant)
  - [test\_sc2\_reinforce\_policy.py — `SC2REINFORCEPolicy` (thin subclass of `TwoHeadREINFORCEPolicy`)](#test_sc2_reinforce_policypy--sc2reinforcepolicy-thin-subclass-of-twoheadreinforce)
  - [test\_sc2\_eval.py — `--eval` evaluation mode](#test_sc2_evalpy----eval-evaluation-mode)
  - [test\_sc2\_play.py — `play_sc2.py` script](#test_sc2_playpy--play_sc2py-script)
  - [test\_sc2\_simple64\_training.py — Simple64 ladder integration](#test_sc2_simple64_trainingpy--simple64-ladder-integration)
  - [test\_sc2\_self\_play.py — `SelfPlayManager` (three self-play opponent modes)](#test_sc2_self_playpy--selfplaymanager-three-self-play-opponent-modes)
  - [test\_sc2\_analytics.py — SC2-specific analytics plots and flags](#test_sc2_analyticspy--sc2-specific-analytics-plots-and-flags)
  - [test\_sc2\_replay\_bc.py — SC2 replay BC: dataset, fit, run, and `--bc` CLI (issues #351–#354)](#test_sc2_replay_bcpy--sc2-replay-bc-dataset-fit-run-and---bc-cli-issues-351354)
  - [test\_sc2\_bc\_adapter.py — SC2 BC adapter ↔ framework integration (issue #394)](#test_sc2_bc_adapterpy--sc2-bc-adapter--framework-integration-issue-394)
- [Framework BC seam](#framework-bc-seam)
  - [test\_framework\_bc.py — `BCAdapter` Protocol + `framework.bc.run` orchestrator (issue #393)](#test_framework_bcpy--bcadapter-protocol--frameworkbcrun-orchestrator-issue-393)
- [TMNF BC](#tmnf-bc)
  - [test\_tmnf\_bc.py — TMNF BC adapter (SimplePolicy source) + `do_pretrain` removal (issue #395)](#test_tmnf_bcpy--tmnf-bc-adapter-simplepolicy-source--do_pretrain-removal-issue-395)
- [MineRL](#minerl)
  - [test\_minerl\_obs\_spec.py — MineRL observation spec (3-dim vector)](#test_minerl_obs_specpy--minerl-observation-spec-3-dim-vector)
  - [test\_minerl\_adapter.py — MineRL game adapter](#test_minerl_adapterpy--minerl-game-adapter)
  - [test\_minerl\_reward.py — MineRL reward config and calculator](#test_minerl_rewardpy--minerl-reward-config-and-calculator)
- [Rocket League](#rocket-league)
  - [test\_rocket\_league\_obs\_spec.py — Rocket League observation spec (142-dim)](#test_rocket_league_obs_specpy--rocket-league-observation-spec-142-dim)
  - [test\_rocket\_league\_reward.py — Rocket League reward calc](#test_rocket_league_rewardpy--rocket-league-reward-calc)
  - [test\_rocket\_league\_env.py — Rocket League env wrapper (mocked rlgym)](#test_rocket_league_envpy--rocket-league-env-wrapper-mocked-rlgym)
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

**Tested.** `CMAESDistribution` pure-math (initialisation invariants,
sample/update mechanics, convergence on a quadratic, save/load .npz
round-trips, dimension mismatch guard); framework ↔ TMNF byte-identical
forward-pass verification for `DQNPolicy`, `REINFORCEPolicy`, and `LSTMCore`
(Q-values, softmax logits, hidden-state trajectory, seed-matched initial
weights). Reward calculator math (linear components, n_ticks scaling,
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

### test_cmaes_distribution.py — `CMAESDistribution` pure-math unit tests
- Init: n / λ / μ=λ/2 / recombination weights sum=1 and decreasing / σ stored / C=I / ps+pc=0 / gen=0 / μ_eff > 1
- initialize_random: mean set to zero
- sample: returns λ vectors of shape (n,) / distinct / fills pop_xs/pop_ys / reproducible with same seed
- update: returns (best_r, best_idx) tuple / correct best values / gen increments / mean shifts / σ positive / C symmetric / C positive-definite / multiple generations move mean / wrong reward count raises / no-sample raises
- Convergence: mean converges toward quadratic minimum after 30 generations
- Save/load: preserves mean / σ / covariance / generation / dim mismatch raises / evolution continues correctly from loaded state

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
- `save_grid_summary` task-metric plugin: default label is "Best Task Metric" with `.4f` format; custom fn replaces label+value; custom fn drives ranking; explicit `task_metric_fmt` overrides format independently of fn
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
- HTTP: serves all combos / lightweight status endpoint (counts + workers) / result accepted + done event
- mobile monitor: `/monitor` login page is public, does not prefill custom monitor usernames, `/monitor/api/status` requires session login, authenticated status includes per-run queued/in-progress/done state + selected-run details
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

### test_game_adapter.py — TMNF/TORCS/SC2/BeamNG/AssettoCorsa/iRacing adapter abstractions
- registry: all games registered (including assetto); adapter instantiable
- TMNF: experiment_dir includes game/policy/track hierarchy, track override, track_label default+override, build_probe/build_warmup, decorate_reward_cfg
- TORCS: experiment_dir root/dir includes game/policy/map hierarchy, track_label default+override, build_probe/warmup = None
- SC2: experiment_dir includes game/policy/map hierarchy, track override, track_label, build_probe/warmup = None
- BeamNG: experiment_dir / build_probe = None
- AssettoCorsa: experiment_dir/root includes game/policy/track hierarchy; track_label default+override+from-params; build_probe returns ProbeSpec; build_warmup returns WarmupSpec(steps=5); decorate_reward_cfg is a no-op; name="assetto"
- iRacing: experiment_dir, track_label default (laguna_seca) + override, build_probe/warmup = None
- docs roster sync (issue #323): every `GAME_ADAPTERS` key appears in `CLAUDE.md`, so the top-level roster can't silently drift from the registry

### test_atari_obs_spec.py — Atari RAM observation spec
- 128-dim flat float32 spec; one feature per RAM byte
- names are unique, ordered as `ram_000`..`ram_127`, each scaled by 255.0
- every `ObsDim` carries a description

### test_atari_env.py — `AtariEnv` wrapper (mocked gymnasium / ale-py)
- reset returns a 128-dim float32 RAM vector plus an info dict
- step returns the 5-tuple shape and seeds `info["native_reward"]` / `info["action_index"]`
- episode terminates when the underlying fake env signals `terminated`
- `n_legal_actions` mirrors the underlying env's `Discrete(N).n`
- action mapping: continuous `[-1, +1]` → linear scale over `[0, N_legal-1]`; discrete index passthrough in range; out-of-range discrete index clamps to NOOP
- env-id resolution: bare `Pong-v5` gets the `ALE/` prefix; already-qualified ids pass through

### test_atari_reward.py — Atari reward calculator
- defaults: `native_reward_scale=1.0`, `clip_sign=False`, `step_penalty=0.0`
- `from_yaml` loads known fields and ignores unknown ones
- `compute` passes the native reward through, applies scale and step penalty, and (with `clip_sign`) clips to `{-1, 0, 1}` (zero stays zero)
- missing `native_reward` in `info` defaults to `0.0`

### test_atari_adapter.py — Atari game adapter
- adapter registered under the `atari` key with the expected `name` / `config_dir`
- `experiment_dir` embeds game / policy / map / experiment name; `--track` override replaces `map_name`
- `track_label` defaults to `Pong-v5`, sanitises slashes (e.g. `ALE/Pong-v5` → `ALE_Pong-v5`)
- `build_probe` / `build_warmup` / `decorate_reward_cfg` are no-ops
- `build_game_spec` wires in the 128-dim obs spec, the 18-row `DISCRETE_ACTIONS`, and a callable `make_env_fn` / `save_results_fn`
- `build_game_spec` threads `render_mode` from `training_params` into `make_env()` (patched to verify the kwarg)
- `build_game_spec` side-effect: `neural_dqn`, `reinforce`, and `lstm` are registered in `POLICY_REGISTRY` after the call

(SC2 policy/param validation moved to test_policy_registry.py with the
`compatible_with` hook in Phase D — `build_extras` was deleted.)

### test_iracing_controller.py — iRacing action injection controller
- NullController: implements BaseController, send/reset/close are safe no-ops
- axis conversion: bipolar [-1,1]→[1,32768] boundary values (−1→min, 0→mid, 1→max), clamping, half values; unipolar [0,1]→[1,32768] boundary values, clamping
- make_controller factory: `"telemetry_only"` → NullController, unknown mode raises ValueError, `"live"` without pyvjoy raises ImportError
- VJoyController (mock-based): send sets all three axes correctly, full-left/full-brake, out-of-range clamping, reset centres steer + zeros pedals, close resets axes

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
- BC config section: no `bc:` key returns empty dict / `bc:` contents returned verbatim / 7-tuple unpacking still valid
- BC compatible policy types: `sc2_genetic`↔`sc2_cmaes` cross-compatible / `sc2_reinforce` self-only / tabular policies self-only / all nine targets are keys
- BC warmstart validation (`_validate_bc_warmstart_combos`): passing compatible target returns `bc_target` string / `sc2_genetic` warmstart accepted by `sc2_cmaes` combo / incompatible target raises `ValueError` mentioning "incompatible" / error lists all failing combos but not passing ones / missing `bc_summary.json` raises / missing `bc_target` field in summary raises
- BC weight copy (`_copy_bc_weights`): copies `policy_weights.yaml` / copies `policy_weights.npz` (sc2_cnn) / copies `trainer_state.npz` when present / copies `policy_weights_qtable.pkl` when present / silently skips absent optional files / raises `FileNotFoundError` when no weight files at all / never copies `bc_summary.json`
- multi-map axis (`_extract_map_axis`): `map_name` list returns key+values / `track` list returns key+values / scalar `map_name` returns `(None, None)` / no map key returns `(None, None)` / `_MAP_AXIS_KEYS` contains expected keys / after extraction, inner `_expand_grid` does not include the map key in `varied_keys` / two maps × two inner values → 2 independent combo sets of 2 each / both map-axis keys list-valued raises `ValueError` / empty map-axis list raises `ValueError`

### test_info_gain.py — staleness-based intrinsic reward
- initial staleness all 1; never-observed = max; just-observed near zero; grows linearly
- reward fires stale→fresh; zero when weight=0; reset restores

### test_live_monitor.py — live GUI monitor helpers
- reward-component extraction prefers per-step components and falls back to differencing cumulative episode totals
- rolling-average values are computed from the latest 5 steps
- observation grouping detects x/y pairs, indexed vectors (including mid-index names like `wheel_0_contact`), quadrant grids, and scalar fallbacks
- reward ordering puts `total_reward` first; all other keys (including former TMNF/SC2-specific names) are sorted alphabetically
- layout helpers split rows into display columns while preserving order and switch observation panel column count from 3 to 4 on wide canvases
- action formatting renders 3-value TMNF controls with steer direction/percent, treats tiny pedal values (`<= 0.01`) as effectively zero when choosing accel-only vs brake-only display, and still truncates long vectors after six entries
- action panel (bar chart): all-no-op buffer shows "(only no-ops)" status and draws no bars; real actions produce bar entries with percentage annotations

### test_obs_memory.py — frame-stacking observation wrapper
- shape; reset fills initial; step shifts frames; k=1 passthrough; invalid k raises; most-recent zero-padded; clear

### test_reward.py — TMNF reward calculator + curiosity glue
- Config: defaults / custom / unknown-key raises / valid keys / partial keys
- Components: progress / no-progress / centerline quadratic / on-center zero / finish bonus / over-par penalty / no-finish-no-bonus / accel bonus (requires actual speed gain) / accel bonus suppressed when stuck (speed unchanged) / step penalty / airborne / airborne above centreline
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
- accepts game+config params; accepts optional specs (probe/warmup); `extras` param removed (Phase D); accepts control flags; no legacy flat params; `GameSpec` requires explicit `game_name` (no empty-default bypass)

### test_new_best_logging.py — `_log_new_best_details` + `_print_episode_summary`
- `_print_episode_summary`: terminated/finished/truncated one-liner; `r=` and `steps=` present; laps and progress omitted
- SC2 summary formatting: scalar `outcome` (`win`/`loss`/`draw`) plus scalar `reward=` and `score=` values
- `_log_new_best_details` — empty info emits nothing
- reward components: logs all non-zero components, always includes `score` (even when 0); `win_bonus` shown only when terminal > 0, `loss_penalty` only when terminal < 0 (neither shown on non-terminal episodes); prev-best comparison included when available
- action frequency: one log line per action; uses `episode_action_name_map` when present (e.g. `Attack_screen =`), falls back to raw key stringified when absent; count + percentage shown with labels padded to a common width so the `=` columns align; prev comparison shown
- action frequency grouped: when `episode_action_category_map` is present, actions are grouped under `[category]` headings emitted in the canonical `move`/`attack`/`build`/`train`/`upgrade` order (same layout as the periodic stats log); aligned `=` columns across all groups
- task metrics: generic `episode_task_metrics` dict (pre-formatted strings); progress, lateral offset, finish time only when present; prev comparison for each key (all on one combined line); adapters are responsible for populating and formatting values
- SC2 kills: units + structures on one line; prev comparison; absent when key not in info; suppressed when both values zero
- SC2 game-state averages: one log line per non-zero metric; zero values omitted; prev comparison
- all five groups together emit seven lines when no terminal event (2 components + 2 actions + 1 task metric + 1 kills + 1 game-state)

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

### test_policy_registry.py — `POLICY_REGISTRY` + `BasePolicy` registry machinery
- `register_policy` raises on duplicate `POLICY_TYPE`
- `register_policy` raises when `POLICY_TYPE == ""`
- All five built-ins (`hill_climbing`, `neural_net`, `epsilon_greedy`, `ucb_q`, `genetic`) are registered after importing `framework.policies`
- Each registered class maps to the correct concrete class
- Each registered class has `LOOP_TYPE` in `{"hill_climbing", "q_learning", "cmaes", "genetic", "sb3", "alphazero"}`
- `_validate_params` raises on unknown keys when `VALID_POLICY_PARAMS` is non-empty
- `_validate_params` is a no-op when `VALID_POLICY_PARAMS` is empty
- `_validate_params` accepts all valid keys without raising
- `_make_policy("hill_climbing", ...)` returns a `WeightedLinearPolicy` via the registry path
- `_make_policy` raises on an unknown `policy_type`
- compatibility hook: `hill_climbing`/`genetic`/`neural_net` rejected on the `sc2` game via `_make_policy(game_name="sc2")` with a ValueError naming the bad type and the `sc2_`-prefixed migration hint; the same policies are accepted on non-SC2 games; `BasePolicy.compatible_with` defaults to allow-all; SC2 adapter registers the game via `register_continuous_action_incompatible("sc2", ...)`
- SC2-native registry policies (`sc2_genetic`/`sc2_neural_net`/`sc2_neural_dqn`/`sc2_cnn`) require `game_name=="sc2"` and reject non-SC2 game names with an explicit hint
- every registered policy with a non-empty `VALID_POLICY_PARAMS` rejects a bogus key
- SC2 policies (after importing every game's policy module): the three Phase-D-migrated types (`sc2_cnn`, `sc2_neural_net`, `sc2_neural_dqn`) are registered with the expected `LOOP_TYPE`; per-type `VALID_POLICY_PARAMS` rejects unknown keys (sc2_genetic/sc2_neural_net/sc2_cmaes/sc2_lstm/sc2_reinforce/sc2_neural_dqn/cmaes) and accepts valid + empty params — replaces the SC2 `build_extras` validation cases removed from test_game_adapter.py

### test_sc2_legacy_names_rejected.py — SC2 bare legacy policy names rejected
- `_make_policy(..., game_name="sc2")` rejects TMNF bare-name `cmaes`/`reinforce`/`lstm`/`neural_dqn` with a "not compatible" ValueError
- Error message includes the expected `sc2_`-prefixed alternative for each rejected type

### test_dqn_upgrades.py — Double-DQN / Huber / gradient clipping
- `DQNPolicy` defaults are upgraded (`double_dqn`/`huber_loss` on, `max_grad_norm=10.0`)
- `to_cfg`/`from_cfg` round-trip the new knobs; a legacy cfg without them loads the upgraded defaults
- `_loss_grad` clamps the residual to ±`huber_kappa` under Huber and is `2·residual` under MSE
- gradient clipping keeps weights finite under extreme TD error; Double-DQN path runs and stays finite

### test_sb3_policies.py — Stable-Baselines3-backed deep-RL policies
- Skipped unless `stable-baselines3` / `sb3-contrib` are installed (`poetry install --with deep_rl`)
- `ppo`/`a2c`/`sac`/`td3`/`qr_dqn`/`recurrent_ppo` are registered with `LOOP_TYPE == "sb3"`; all gated off SC2, allowed on racing games
- unknown `policy_params` rejected; `total_timesteps` resolution (explicit vs `n_sims × steps_per_sim`)
- end-to-end training on a dummy Gym env (continuous `a2c`/`ppo` and discrete `qr_dqn`): episodes recorded, `*_sb3_model.zip` saved, resume + predict

### test_alphazero_mcts.py — AlphaZero-style model-based MCTS
- `alphazero_mcts` registered with `LOOP_TYPE == "alphazero"`; gated off non-cloneable games, allowed on a cloneable game name
- `__call__` returns a valid action from the policy head; the loop raises on a non-cloneable env
- end-to-end self-play on a toy cloneable corridor MDP reaches the goal (positive best reward), saves weights, and resumes

## TMNF policies

Trackmania-Nations-Forever-specific code under `games/tmnf/`. Policies live
in `games/tmnf/policies.py`; the bridge to the live game is in
`games/tmnf/clients/`.

**Tested.** Every policy listed in CLAUDE.md (`WeightedLinearPolicy`,
`NeuralNetPolicy`, `EpsilonGreedyPolicy`, `UCBQPolicy`, `GeneticPolicy`,
`CMAESPolicy`, `NeuralDQNPolicy` (incl. Double / Dueling / Huber),
`REINFORCEPolicy`, `LSTMEvolutionPolicy`)
is exercised in isolation: action shape and range, deterministic forward
pass, mutation produces different weights, save/load YAML round-trips
losslessly (including replay buffer + Adam moments for DQN, σ + covariance
for CMA-ES, hidden-state reset for LSTM), `from_cfg` rejects shape
mismatches, and the optimisation loop converges on a tiny stand-in
problem (2-arm bandit, quadratic max). `RLClient`'s threading model — the
tick-window state machine, decision_idx clamping, and the finish/respawn /
hard-crash forced-commit paths — is fully covered against a `MagicMock`
TMInterface.

Note: `test_cmaes_policy.py`, `test_neural_dqn_policy.py`,
`test_reinforce_policy.py`, and `test_lstm_policy.py` now import from
`framework.cmaes`, `framework.dqn` / `framework.replay`, `framework.reinforce`,
and `framework.lstm` respectively (Phase A extraction).  They exercise the
framework classes through TMNF-flavoured instantiation
(`obs_spec=TMNF_OBS_SPEC`, TMNF action decoder / `DISCRETE_ACTIONS`), so all
TMNF-specific behaviour is still covered.

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

### test_cmaes_policy.py — `CMAESPolicy` (framework) via TMNF factory
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

### test_ucb_q_policy.py — UCB1 online Q (formerly mcts)
- action in range; unseen state random; visit count increments; Q changes; exploitation prefers high Q; visits accumulate

### test_neural_dqn_policy.py — `DQNPolicy` (framework) + `ReplayBuffer`
- ReplayBuffer (`framework.replay`): push+len / circular eviction / sample shapes / w/o replacement / with replacement when small
- DQNPolicy (`framework.dqn`): action shape+range / greedy discrete / random when ε=1 / buffer fills on update / ε decays / floored / target sync / weight shapes
- Cfg: roundtrip (`policy_type="dqn"`) / on_episode_end no-op / missing keys raise / shape mismatch raises (via `TMNF_OBS_SPEC.with_lidar`)
- Bandit convergence; save/load replay buffer + Adam moments; wrong obs_dim raises
- Double DQN: flags default off / `_next_state_q` = target-Q at online-argmax / `<=` vanilla max & differs when nets disagree / cfg roundtrip preserves `double_dqn` / bandit convergence
- Dueling: value head built (`value_w`/`value_b`) / `Q == V + (A − mean A)` / gradient step updates value head / cfg roundtrip preserves `dueling` + value-head weights / bandit convergence

### test_reinforce_policy.py — `REINFORCEPolicy` (framework) Monte-Carlo PG
- action shape; steer range; accel/brake discrete; action from DISCRETE_ACTIONS
- buffers: fill / clear on episode end / empty on_episode_end no-op
- weights match hidden_sizes; buffer lengths match; weights change after update; gradient direction
- entropy_coeff = 0 vs nonzero; cfg required keys (`output_dim` not `n_lidar_rays`) / policy_type / restore weights+hyperparams; save+reload; lidar via `obs_spec.with_lidar(4)`; baseline roundtrip; wrong obs dim raises

### test_lstm_policy.py — `LSTMCore` + `LSTMEvolutionPolicy` (framework)
- LSTMCore (`framework.lstm`): action shape / steer range / accel+brake binary / hidden updates / episode reset zeros / update no-op / different history → different action
- Flat encoding: dim correct / to_flat shape / roundtrip / zeros hidden / preserves weights / wrong size raises / mutated differs / same hidden_size / lidar roundtrip via `obs_spec.with_lidar(3)`
- Cfg: required keys (`obs_dim` not `n_lidar_rays`) / policy_type / from_cfg roundtrip / save+reload
- LSTMEvolutionPolicy: pop size / σ property / champion = -inf / flat dim matches template / μ=λ/2 / recomb sum=1
- Sample: count / LSTMCore type / fills buffer / distinct individuals
- Update: true first time / sets champion / tracks best / false on no improve / mean shifts / wrong count raises / no sample raises / σ adapts
- Call: raises before update / valid after; on_episode_end resets champion hidden state
- to_cfg keys (`sigma` + `champion_reward`; no `n_lidar_rays`) / policy_type / save yaml / init from champion / mean→target / save+load roundtrip / wrong flat dim raises

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
consecutive blocked steps); the 116-entry `FUNCTION_IDS` table and the
`DISCRETE_ACTIONS` uniform `[command × location]` grid (3 455 rows: every
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
`save_experiment_results` integration including that no racing plots appear);
and `SelfPlayManager` — all three opponent modes (`exact`, `mutated`, `top_n`),
pool growth/eviction semantics, and wiring through `train_rl`.

The offline replay-reader and BC fit modules (`games/sc2/replay_bc.py`, issues
#351–#354) are covered in `test_sc2_replay_bc.py`: the full PySC2 replay API
(run_config, controller, features) is replaced by lightweight fakes injected
into `sys.modules` before import, so the suite runs with no SC2 binary or
PySC2 package installed. `validate_replay_dir` (issue #352) is tested against
real temp-dir fixtures — no fakes needed since it only touches the filesystem.
`fit_bc` (MLP and linear paths), `run` (full pipeline), and the `main.py --bc`
entry point are all unit-tested with synthetic numpy datasets and mocked
pipeline helpers (issue #353).  The policy-agnostic warm-start extension
(issue #354) adds round-trip tests for all SC2-compatible policy families:
`sc2_cmaes`, `sc2_neural_net`, `sc2_neural_dqn`, `sc2_lstm`, `sc2_cnn`,
`epsilon_greedy`, and `ucb_q`, plus error-path tests for SB3 targets and
unknown target names.

**Not tested.** PySC2 against the actual Blizzard SC2 binary; real
1v1 games against the built-in bot; minimap rendering; the deferred
fog-of-war belief machinery beyond the standalone `test_belief.py`
encoder; long-horizon RL convergence on Simple64 (loops are run for a
handful of iterations only); reading real `.SC2Replay` files end-to-end
(no integration test exists yet for `replay_bc`).

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
- `TestActionToFunctionCall`: fake PySC2 module validates encoding branches — `_quick` emits queue-only args, `select_point` and `select_rect` emit screen coords, `_minimap` actions scale with `minimap_size` (not `screen_size`), and `_screen` actions keep using `screen_size`
- `TestFunctionCallToAction` (#350): `function_call_to_action` (the inverse of `action_to_function_call`) round-trips `[fn_idx, x, y, queue]` for non-spatial (`no_op` / `select_army` / `select_idle_worker` → centre coords 0.5,0.5), `_quick` (queue preserved), `select_point` / `select_rect`, `_screen` (queue preserved), and `_minimap` (coords scale with `minimap_size`) actions; grid-aligned coords round-trip exactly (incl. odd `screen_size` 0.5 midpoint); an explicit mid-screen `FunctionCall` normalises coords into the unit square; unknown PySC2 function ids return `None` (skip sentinel, no exception). The id→fn_idx cache is reset inside the faked-PySC2 context per round-trip.

### test_sc2_tech_tree.py — hardcoded tech-tree preconditions (issue #346)
- `TestBuildingPrereqsMet`: no-prereq buildings always buildable; Terran chain (Barracks needs SupplyDepot; FusionCore needs Starport); Zerg OR-set semantics (Spire requires Lair OR Hive)
- `TestFnIdxSatisfiedTerran`: Build_FusionCore_screen blocked when only CC + SCV visible; unlocked once Starport exists (regression for the issue body); Build_SupplyDepot only needs a worker selected; Train_Marine needs Barracks selected (SCV selection rejected); Train_Marauder needs BarracksTechLab; Train_Battlecruiser chain (Starport + StarportTechLab + FusionCore); Effect_Stim needs Stim upgrade and Marine/Marauder selected (SCV rejected)
- `TestFnIdxSatisfiedProtoss`: Carrier needs FleetBeacon; Stalker accepts Gateway OR WarpGate selection
- `TestFnIdxSatisfiedZerg`: Morph_Hive needs Lair selected + InfestationPit exists; Train_Lurker needs Hydralisk selected + LurkerDenMP; Train_Baneling needs Zergling selected + BanelingNest (Roach rejected); Morph_Overseer needs Overlord selected + Lair OR Hive (Zergling rejected); Morph_Lair needs Hatchery selected + SpawningPool; Train_BroodLord needs Corruptor selected + GreaterSpire
- `TestMorphsFullyIntegrated`: morph fn_ids route through UNIT_PRODUCERS via `_train()`; Morph_Archon accepts HighTemplar OR DarkTemplar (Zealot rejected); Morph_SiegeMode needs SiegeTank + FactoryTechLab; Morph_Unsiege needs SiegeTankSieged
- `TestUniversalActions`: no_op / select_army / select_point always satisfied; Move_screen requires any unit selected
- `TestResourceCostFilter` (issue #357): insufficient minerals blocks build/train actions (0 minerals blocks SupplyDepot); boundary condition — exactly at cost is allowed, one below is not; vespene cost gates actions independently; battlecruiser double-cost (400 minerals + 300 vespene) requires both; universal no-cost actions (no_op, select_army, Move_screen) pass with zero resources; morph-only mode-change actions (SiegeMode, Unsiege) have no resource cost and pass with zero minerals; backwards-compatible call without minerals/vespene args defaults to infinity (all existing callers unaffected); every fn_idx in RESOURCE_COSTS is a known PRECONDITIONS key; Zerg morph cost (Morph_Lair: 150 minerals + 100 vespene)
- `TestPreconditionsTableShape`: every fn_idx in FUNCTION_IDS has a Preconditions entry; universal actions never require OF_TYPE selection; every Build_*_screen action (except `Build_CreepTumor_screen`, which is a Queen ability) requires worker selection

### test_sc2_reward.py — SC2 reward calc
- defaults; from_yaml; unknown raises; loads bundled config
- score delta; step penalty only; step penalty n_ticks scaling; win bonus; loss penalty; no-outcome no bonus; economy weight; idle penalty when idle / not when busy
- idle_worker_penalty (issue #358): fires per idle worker (scaled by count); zero when no idle workers; zero when key absent; disabled by default; n_ticks scaling; appears in components dict
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
- new_action_unlock_bonus (issue #360): `_TECH_GATED_FN_IDS` class attribute is non-empty and includes fn_idx 8 (`Build_Barracks_screen`); fn_idx 7 (`Train_Marine_quick`) included via selection_target gate (Barracks has non-empty BUILDING_PREREQS); fn_idx 35 (`Train_Hellion_quick`) included via Factory prereq; fn_idx 64 (`Train_Zealot_quick`) included via Gateway prereq; fn_idx 10 (`Train_SCV_quick`) excluded (CommandCenter has no prereqs); fn_idx 95 (`Train_Drone_quick`) excluded (Larva not in BUILDING_PREREQS); fn_idx 63 (`Train_Probe_quick`) excluded (Nexus has no prereqs); end-to-end: bonus silent for fn_idx 63; non-gated fn_ids (0, 1, 2) excluded; bonus fires on first appearance of a tech-gated fn_idx in `available_fn_ids`; end-to-end: bonus fires when fn_idx 7 in available_fn_ids; silent for fn_idx 10; does not fire again same episode; fires again after `reset()`; scales with count of newly unlocked fn_ids; non-gated actions do not trigger; disabled when bonus is 0.0; missing `available_fn_ids` key does not crash; components sum equals total
- new_action_usage_bonus (issue #400): disabled when bonus is 0.0; default max_uses is 50; no bonus before action appears in `available_fn_ids`; bonus fires when tech-gated action is issued after unlock; bonus fires on the same step as unlock (unlock updates set first); fires repeatedly up to max_uses then stops; per-fn_idx counts are independent; usage counts reset after `reset()`; no_op (fn_idx 0) does not trigger; non-tech-gated fn_ids do not trigger; works independently when unlock_bonus is 0.0; both bonuses fire independently when both enabled; `components["new_action_usage"]` present even when disabled; components sum equals total
- build_bonus and train_bonus (issue #416): both disabled by default; build_bonus fires only on build fn_idx (8, Build_Barracks_screen) not train; train_bonus fires only on train fn_idx (7, Train_Marine_quick) not build; neither fires on move (2) or no_op (0); zero when disabled; train_bonus scales independently of build_bonus; canonical 4× relationship verified; n_ticks scaling for both; both present in components dict when disabled; components sum equals total
- build_repeat_penalty: disabled by default; first build issuance is never penalised; second consecutive same-fn_idx build fires penalty; third consecutive also fires; different build fn_idx does not fire; intervening move action resets counter; intervening train action resets counter; zero when weight is 0.0; n_ticks scaling; present in components dict when disabled; net zero when penalty equals -build_bonus; components sum equals total; reset() clears last build fn_idx
- train_repeat_penalty: disabled by default; first train issuance is never penalised; second consecutive same-fn_idx train fires penalty; third consecutive also fires; different train fn_idx does not fire; intervening move action resets counter; intervening build action resets counter; zero when weight is 0.0; n_ticks scaling; present in components dict when disabled; net zero when penalty equals -train_bonus; components sum equals total; reset() clears last train fn_idx
- resource_banking_penalty (issue #372): disabled by default; default thresholds mineral=300/gas=200; zero when both resources below threshold; zero at exact thresholds; fires proportionally for excess minerals only; fires proportionally for excess gas only; accumulates both resources; n_ticks scaling; zero when info keys absent; appears in components dict; zero in components when disabled; components sum equals total; custom thresholds honoured
- macro-progression rewards: supply_block fires when capped (food_used≥food_cap, cap<200) and is skipped at the 200 hard cap / when supply slack exists; supply/worker/army growth reward only increases (delta-based, first step is baseline, decreases ignored); tech_building one-shot per newly-seen structure type; expansion fires on a new running-max town-hall count (starting base not rewarded, not paid twice for the same count, temporary drops ignored); `scout_explore` rewards `minimap_explored_frac` increases and ignores decreases (named distinctly from the belief system's `scout` component to avoid collision); all new components present and zero when disabled; components sum equals total with the full progression block enabled
- components sum: new terms included in total (extended sum test)

### test_sc2_client.py — PySC2 client wrapper
- minigame flat obs shape; score-delta threading; player_relative centroid; terminal outcome recorded
- shared obs extraction (#350): the per-block extractors (`_player_features`, `_score_features`, …) and `_timestep_to_obs_info`'s flat-obs half are now thin wrappers over the module-level `extract_flat_obs` / free extractor functions in `games/sc2/client.py` (single code path shared with the offline replay reader). These tests exercise the instance-method API unchanged, so they also cover the module-level functions by delegation; flat-obs values and info-dict contents are asserted identical to before the refactor.
- ladder flat obs shape; visibility tracking; fogged ≠ visible; ladder terminal outcome; non-terminal = None
- info dict includes unit-aware `self_attack_range_px` derived from visible friendly `feature_units` (uses max friendly range from curated PySC2 unit-range table)
- `total_self_hp`: info dict sums visible friendly health+shield from `feature_units`; also exposes `visible_self_unit_count`; helper returns 0 for no-self / missing `feature_units` / short rows
- progression info keys: info dict surfaces `food_workers`, `food_army`, `minimap_explored_frac`, `owned_building_names`, and `townhall_count` for the macro-progression reward terms; `townhall_count` counts friendly town-hall `feature_units` (CommandCenter/Nexus/Hatchery + morphs) and returns 0 for no-self / missing `feature_units` / short rows
- rich extractors (#135): enemy unit-type counts (owner==4 only; neutral owner==3 excluded; ally owner==2 excluded; missing field; unknown type); shield/energy (self shield mean, no units, None screen); creep (half coverage, no creep, None minimap); economy pipeline (upgrade count, build queue, cargo, all missing); rich spec contains new names; ladder spec unchanged
- selected-unit extras: `selected_count` and shields + energy from cols 3/4 of single/multi_select; empty selection → zeros; short rows don't crash
- screen_visibility_frac: all visible → 1.0; half visible → 0.25; fogged not counted; None screen / missing layer → 0
- screen_unit_density_aa_mean: mean of unit_density_aa layer; zero layer; None screen / missing layer → 0
- self_weapon_cooldown_mean: mean for alliance==1 from col 25; all ready → 0; no self units → 0; missing feature_units → 0; too few cols → 0
- alerts: empty array → 0; one alert → 1; two alerts → 2; missing key → 0; None value → 0; alert_count present in ladder names
- minimap enemy centroid: minimap_enemy_cx/cy computed from player_relative==4 layer; correct when beacon present; zero when no beacon on minimap (edge case)
- action translation (#346): `_action_to_call` is now a thin translator — legal actions pass through, PySC2-unavailable actions become `no_op` (no implicit `select_army` / `select_point` substitution; that logic moved to `_resolve_action`)
- resolve + defer (#346): `step()` runs `_resolve_action` on the policy's chosen action; if the action requires a different unit-type selection than the current one, the right `select_*` is emitted *this* tick and the original action goes into a 1-slot deferred FIFO that replays on the next `step()`. Covers: no_op / `select_army` / legal action passes through; `Move_screen` with empty selection → `select_army` then deferred replay; `Build_Barracks` with non-worker selection → `select_idle_worker` (when available) then deferred replay; `Build_Barracks` with all workers busy and no `select_idle_worker` → `select_point` on a cached worker position (issue #346 mining/building worker requirement); `Train_Marine` with no Barracks anywhere → passes through and PySC2 no-ops it (mask should have prevented this upstream); `extreme_random` samples from `_available_fn_ids` only (tech-tree-filtered), not raw PySC2 `available_actions`; phase disabled after `_extreme_random_run_count` episodes
- state dump logging (#346 follow-up): periodic readable game-state debug log throttled to one emission per `_STATE_LOG_INTERVAL_S` (10 s); skipped when the logger isn't at DEBUG; rendered units / buildings / upgrades / selection / available-action list with selection-requirement hints; empty state renders dashes
- action-mask caching overhead (#140): _available_actions_features uses module-level cache (no per-call pysc2 import); correctness test with injected cache; deterministic regression asserting _get_pysc2_id_to_fn_idx is called exactly once per _available_actions_features invocation (not once per FUNCTION_IDS entry)
- score_features field-name access: named NamedNumpyArray access takes priority over positional; score→score_total rename applied; missing score array → all zeros
- last_fn_idx property: `_resolve_action` routing → idx=1 when select_army resolved; no_op fallback on blocked PySC2 mask → idx=0; passthrough → requested idx
- realtime param: default False stored; True stored; forwarded as kwarg to SC2Env constructor
- deferred-action oscillation fix (#356 H1): deferred `Move_screen` replays directly on step 2 even when army is still empty; `select_army` appears exactly once across both steps (not re-emitted, not re-deferred)
- owned-buildings accumulation fix (#356 H2): drives real `_timestep_to_obs_info()` with patched `_compute_owned_buildings()` — buildings seen in prior steps remain in `_owned_buildings` after the camera pans away; new buildings are unioned in each step; episode `reset()` clears both `_owned_buildings` and `_owned_buildings_seen`
- fn_idx_satisfied cache fix (#356 H3): patches `games.sc2.client.fn_idx_satisfied` to count calls — zero new calls on cache hit (identical state); fresh pass triggered when `owned_buildings`, `selected_unit_types`, or the PySC2 candidate set changes
- select_idle_worker suppression (issue #383): fn_idx 4 removed from available set when SCV is selected; present when no worker selected; suppression covers all three races (SCV / Probe / Drone)
- Refinery target snap (issue #402): `_update_unit_screen_positions` collects neutral (alliance=3) vespene-geyser positions from `feature_units` (excludes non-geyser neutrals like mineral fields, excludes friendly Refineries on top of geysers, resets cache each step); `_snap_refinery_to_geyser` rewrites Build_Refinery/Assimilator/Extractor (fn_idx 22/51/82) target coordinates to the nearest cached geyser, leaves other actions untouched, passes the snap input unchanged when no geyser is visible, and uses a +0.5-px offset so downstream int-truncation lands on the geyser pixel exactly; `_action_to_call` invokes the snap before delegating to PySC2.

### test_sc2_env.py — SC2 env wrapper
- minigame obs space; action space shape+bounds; ladder obs space; episode time-limit get/set
- reset returns obs+info; step 5-tuple; score-delta reward; done terminates; loss outcome
- close calls client.close; info keys; prev_score threaded; prev_army_count + prev_total_self_hp seeded from reset and updated each step; executed `last_fn_idx` (not merely requested fn_idx) drives reward-shaping metadata; skipped-frame counters (default 0, per-step accumulation from game_loop deltas); custom reward config
- end-screen analytics: series absent on mid-episode step; present on terminal step; supply_capped_fraction correct; army series value; resource series sums minerals+vespene; starting units excluded from build order; new units produce events; empty build order when no unit_counts

### test_sc2_replay.py — SC2 replay saving on new-best events (issue #210)
- `SC2Client.save_replay`: returns None when SC2 env not running; delegates to pysc2 save_replay with correct dir and prefix keyword; `os.makedirs` inside try block so directory creation failures are swallowed; swallows SC2 exceptions and returns None
- `SC2Env.save_replay`: thin delegation to client
- `_try_save_replay` (single-episode loops): no-op for envs without save_replay; first new-best → `_best-01` prefix; second new-best → `_best-02` when one confirmed best exists; candidate (`_`-prefixed) files excluded from sequential count; replay_dir always `<experiment_dir>/replays/`; files not matching `{experiment}_best-\d+` regex pattern ignored for numbering (including same-prefix non-numeric files like `{exp}_best-notes.txt`); exception swallowed
- `_save_candidate_replay`: no-op for envs without save_replay; calls save_replay with `_candidate` prefix; exception swallowed, returns None
- `_finalize_candidate_replay`: no-op when path is None or file missing; renames to next sequential `_best-N{ext}` preserving the candidate's extension; candidate files excluded from confirmed-best count; two confirmed → `_best-02`
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

### test_sc2_hierarchical_policy.py — `SC2HierarchicalLinearPolicy` + `SC2HierarchicalGeneticPolicy` (issue #388)
- Categories: all fn_ids covered / no fn_id in multiple categories / names match keys / N_CATEGORIES correct / inverse map complete / inverse map consistent / sentinel members in correct categories
- Init: meta/fn/spatial/queue weight shapes (minigame + ladder) / default race=random
- Call: output shape (4,) / dtype float32 / fn_idx in valid range / spatial in unit square / queue is 0 or 1 / positive logit → queue=1 / negative logit → queue=0 / respects availability mask / meta scores route to correct category / empty category falls back / selected fn_idx always in chosen category
- Serialisation: to_cfg has meta + queue keys / cfg round-trip / save+load round-trip / missing keys default zero
- Flat: length = (N_CATEGORIES + N_FUNCTION_IDS + N_SPATIAL_ROWS + N_QUEUE_ROWS) × obs_dim / round-trip / mutated differs / same shape after mutation
- Genetic: policy_type=sc2_hierarchical / compatible with sc2 / incompatible with tmnf / initialize_random + call / save champion + reload / from_cfg round-trip

**Not tested:** actual two-player SC2 binary execution, multi-generation evolution convergence.

### test_sc2_neural_net_policy.py — hill-climbing MLP policy for SC2
- Action: shape (4,) / fn_idx range / x+y unit square / queue binary
- Determinism and cfg roundtrip: deterministic calls / from_cfg action-equivalent
- Params and weights: hidden_sizes preserved / layer shapes match architecture
- Mutation and masking: mutated weights differ / unavailable fn_idx masked / update caches available_fn_ids

### test_sc2_neural_dqn_policy.py — masked DQN for SC2
- fn_idx_for_cell: centre=select_army / others=move_screen / consistent / int
- Available mask: all fn_ids / empty / select_army only / move_screen only / both / dtype bool
- Policy: illegal never picked greedy / random respects mask / all-true mask allows all fn_idx values / mask cached from update info / missing info resets to all-true / on_episode_start primes from reset info
- Cfg + trainer state: policy_type / from_cfg roundtrip / trainer-state roundtrip / shape mismatch raises
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
- Missing weights raises; loads sc2_multi_head for sc2_genetic / correct weights / sc2_neural_dqn / sc2_reinforce / sc2_lstm; legacy bare-name reinforce/lstm weight formats fail fast; cmaes no policy_type → SC2Linear; unknown → SC2Linear
- Outcome handling: win / loss / draw / none
- Episode loop: calls policy each step / client until done / on_episode_start(info=<dict>)+end / works without lifecycle hooks
- Play-mode flag: stored / default false; `make_sc2_env` lazy on reset
- Game loop: present in info / value extracted from obs

### test_sc2_simple64_training.py — Simple64 ladder integration
- Obs: 21 dim; reset shape; step 5-tuple
- Outcomes: win+bonus / loss+penalty; economy reward flows through
- Action: fn_idx valid / xy continuous / queue binary / shape
- Pop usage: sc2_genetic uses SC2Linear / cmaes offspring use SC2Linear; SC2Linear and sc2_genetic action shape
- Per-policy on Simple64: epsilon_greedy / ucb_q / sc2_neural_dqn shape+update / sc2_cmaes sample+update / sc2_reinforce shape+episode / sc2_lstm shape / sc2_lstm-evolution sample+update
- Training loops (mocked env): sc2_genetic / sc2_cmaes / sc2_neural_dqn / sc2_reinforce / sc2_lstm
- Trainer state roundtrips: sc2_cmaes / sc2_neural_dqn

### test_sc2_self_play.py — `SelfPlayManager` (three self-play opponent modes)
- `TestSelfPlayManagerExactMode`: invalid mode raises / `build_initial_opponent` returns a callable / `step` returns a fresh callable every generation regardless of `improved`
- `TestSelfPlayManagerMutatedMode`: `step` calls `mutated()` on the champion when available / falls back to deepcopy when policy has no `mutated` method
- `TestSelfPlayManagerTopNMode`: pool grows on improvement / capped at `top_n` / weakest entry replaced by a stronger champion / no pool update when not improved and pool is non-empty / callable returned from pool / weak champion does not displace a stronger pool entry
- `TestGreedyLoopCmaesWithSelfPlay`: `_greedy_loop_cmaes` calls `manager.step()` once per generation and forwards the returned opponent to `env.set_opponent_policy()`; without a manager `set_opponent_policy` is never called
- `TestTrainRLSelfPlayModes`: verifies that all three modes (`exact`, `mutated`, `top_n`) wire a `SelfPlayManager` through `train_rl` and that the genetic greedy loop receives the correct `self_play_manager` kwarg

### test_sc2_analytics.py — SC2-specific analytics plots and flags
- `SUPPORTS_THROTTLE=False` / `SUPPORTS_PATH=False` flags
- `GreedySimResult` new fields: `action_counts` / `obs_averages` / `xy_hist` / `skipped_frames` — default None; stored correctly
- `GreedySimResult` end-screen fields: `supply_capped_fraction` / `build_order` / `army_count_series` / `resource_series` — default None; stored correctly
- `plot_action_frequency`: renders to file / skips when no data / skips when no sims / single fn_idx / no_op (fn_idx 0) excluded from all panels (issue #382) / all-no_op input → no file written (early return)
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
- `save_grid_summary`: forwards config-normalized rewards **and per-component contributions** using `v / max(abs(weight), 1.0)` — weights ≥ 1.0 are divided (making large-weight components comparable across grid-search runs), weights < 1.0 use the raw value (which already encodes the weight, so dividing would amplify by ×1000); wires SC2 extra-plot hook into framework summary generation; covers no-components fallback, multi-sim normalization, malformed YAML fallback, non-mapping YAML fallback, non-numeric weight fallback, allow-listed `scout` component (belief intrinsic reward, no unmapped-key warning) and the reward calculator's normally-scaled `scout_explore` map-exploration component (also no unmapped-key warning), step_penalty with sub-1.0 weight passes through raw (-0.5 not -500), idle_penalty with sub-1.0 weight passes through raw, any sub-1.0 weight (0.0001 or 0.001) both use scale=1.0, the new `unit_loss` / `damage_taken` / `passive_under_fire` components normalize through their config weights without unmapped-key warnings, realistic positive reward stays positive after normalization (313.5 ✓), and emits SC2 cross-run charts + summary links; passes `task_metric_fn`, `task_metric_fmt` (percentage formatter) to framework; `attack_bonus` mapped to `attack_bonus` config key in normalisation
- `_sc2_task_metric`: empty sims → 0.0; win+finish counted as success; loss/timeout/None/other not counted; all-wins → 1.0; `_GS_SUCCESS_REASONS` constant contains win+finish, excludes loss+timeout

### test_sc2_replay_bc.py — SC2 replay BC: dataset, fit, run, and `--bc` CLI (issues #351–#354)
- `TestValidateReplayDir`: nonexistent folder raises `ValueError`; file path raises `ValueError`; empty
  folder raises with "No .SC2Replay files found"; returns only `.SC2Replay` files sorted; race filter warning
  emitted when race is non-null/non-"any"; no warning for `race="any"` or `race=None`; version mismatch
  warning emitted when filenames don't contain the version string; no warning when all filenames match;
  partial mismatch warning includes count ratio (e.g. "1/2")
- `TestIterReplays`: finds only `.SC2Replay` files; sorted order; empty folder returns `[]`
- `TestParseReplayInfo`: winner + races parsed correctly; winner is player 2; undecided returns `(0, {})`;
  race integer mapping (1=terran, 2=zerg, 3=protoss, 4=random); unknown race falls back to `"random"`
- `TestResolvePlayerId`: `"winner"` resolves to winner pid; fallback to 1 when winner=0; explicit int id;
  explicit int id with zero winner
- `TestPickBestAction`: empty returns None; `"first"` returns first even if no_op; `"first_non_noop"` skips
  no_op; fallback to first when all no_op; single action returned regardless of strategy
- `TestReplayObservations`: obs shape matches spec (float32, `[D]`); action shape+dtype `([4], float32)`;
  fn_idx preserved in action vector; temporal order preserved via monotonically increasing x-coord;
  steps with no actions skipped; `[no_op, Move_screen]` → `Move_screen` selected; unknown PySC2 fn_id
  skipped; `winner_id=0` falls back to player 1; explicit `player_id=2` overrides winner; `controller.step()`
  called once per step including no-action steps; unknown fn_id still calls `step()`
- `TestReadOneReplay`: race match returns `(True, player_race, pairs)`; race mismatch returns `(False, ..., [])`;
  no race filter processes all; race filter drops replay → `start_replay` not called; `player_race` always
  returned even when filter drops replay
- `TestBuildDataset` (mocks `_read_one_replay`): single replay writes correct NPZ with `n_episodes=1`;
  `obs`/`actions` shapes correct; episode boundaries for two replays (`episode_starts`, `episode_lengths`,
  `episode_id`); temporal order preserved within episode; meta JSON round-trip (`player_id`, `step_mul`,
  `screen_size`, `source_filenames`); all replays dropped by race filter raises `ValueError` with race name
  in message; no replay files raises `ValueError`; `race="any"` keeps all; `source_filenames` recorded
  in meta; `episode_id` covers all rows
- `TestLoadDataset`: flat load returns all expected keys; correct shapes; `as_episodes=True` yields correct
  per-episode shapes; episode `episode_starts` partitions correctly; temporal order preserved per episode;
  `meta` parsed as dict in `as_episodes` mode; round-trip save+load preserves obs+action values
- `TestFitBCMLP` (issue #353): loss decreases over more epochs on a separable dataset; saved weights
  load back as `SC2REINFORCEPolicy` via `from_cfg`; all-noop dataset raises `ValueError` when
  `bc_ignore_noop=True`; `bc_ignore_noop=False` keeps all steps; pairwise accuracy > 80%
  (logit of correct fn_idx class exceeds logit of the alternative class) after 20 epochs with a
  linear model (`hidden_sizes=[]`) on a linearly-separable two-class dataset; accepts a `Path`
  argument in place of a dict dataset
- `TestFitBCLinear` (issue #353): returns a loadable `SC2MultiHeadLinearPolicy`; saved weights load back
  via `sc2_genetic` path; `fn_weights` shape is `(N_FUNCTION_IDS, obs_dim)`; `sp_weights` shape is
  `(2, obs_dim)` for spatial fn_idx actions; when no spatial steps are present `sp_weights` are all zero
- `TestRunBC` (issue #353, mocks `validate_replay_dir`/`build_dataset`/`load_dataset`): writes
  `policy_weights.yaml`; writes `bc_summary.json` with all required keys (`n_replays_kept`,
  `n_replays_skipped_race`, `n_episodes`, `n_pairs`, `fn_idx_histogram`, `bc_player_id`, `bc_race`,
  `bc_target`, `final_bc_loss`); `fn_idx_histogram` covers all actions from the synthetic dataset;
  writes `trainer_state.npz` for the MLP target; linear target writes `policy_weights.yaml` without
  trainer state; `winner` player id and passed race filter appear in summary
- `TestBCCLIMain` (issue #353): `--bc` flag is parsed; default player is `None` (config fallback);
  `winner`/`1`/`2` are valid `--bc-player` choices; all four race choices accepted for `--bc-race`;
  `sc2_reinforce`/`sc2_genetic` accepted for `--bc-target`; `--bc` with `--game != sc2` raises
  `SystemExit`; `--bc` and `--play` are mutually exclusive; `--bc` and `--eval` are mutually exclusive
- `TestFitBCCMAES` (issue #354): returns `SC2CMAESPolicy`; `_champion` is set after fit; distribution
  mean equals `champion.to_flat()`; champion callable → `(4,)` action
- `TestFitBCNeuralNet` (issue #354): returns `SC2NeuralNetPolicy`; layer weight shapes match
  `[obs_dim, hidden, 4]`; callable → `(4,)` action after fit; loss decreases over more epochs; round-trip
  save+reload via `SC2NeuralNetPolicy.from_cfg`
- `TestFitBCDQN` (issue #354): returns `SC2NeuralDQNPolicy`; replay buffer has transitions after fill;
  `bc_loss` equals fill fraction `len(replay) / capacity`; episode boundary steps marked done
- `TestFitBCLSTM` (issue #354): returns `SC2LSTMEvolutionPolicy`; `_champion` is set; champion callable;
  loss is finite and non-negative; `_mean` equals `champion.to_flat()`; round-trip save+reload via
  `SC2LSTMPolicy.from_cfg`
- `TestFitBCCNN` (issue #354): returns `SC2CNNEvolutionPolicy`; `_champion` is set; `_mean` equals
  `champion.to_flat()`; `W1`/`W2` (conv layers) are zeroed; obs-portion of `W3` is non-zero
- `TestFitBCTabular` (issue #354): `epsilon_greedy` returns `EpsilonGreedyPolicy`; `ucb_q` returns
  `UCBQPolicy`; Q-table populated after seeding; `_n_sa` populated; Q-values normalised by visit count
  in `[0, 1]`; `epsilon_greedy` callable → `(4,)` action
- `TestFitBCUnknownTarget` (issue #354): SB3 targets (`ppo`, `a2c`, `sac`, `td3`, `qr_dqn`,
  `recurrent_ppo`) each raise `ValueError` mentioning "SB3"; completely unknown targets raise
  `ValueError` with the target name in the message; error message lists supported targets
- `TestBugFixes354`: tabular Q-values sum to exactly 1.0 per state (not per-action count-divided);
  DQN terminal transitions store a zero-vector `next_obs` (not the next episode's first obs);
  `sc2_lstm` raises `ValueError` with "episode_starts" in the message when those keys are absent

### test_sc2_bc_adapter.py — SC2 BC adapter ↔ framework integration (issue #394)
- Protocol surface: `name="sc2"`, `default_target="sc2_reinforce"`, all nine
  `sc2_*` / tabular targets in `supported_targets`, all four protocol methods
  callable
- `validate_replay_dir(None)` raises with a pointer to `--replay-dir` /
  `bc_replay_dir`
- End-to-end through `framework.bc.run` with mocked SC2 pipeline: emits the
  canonical framework summary (`game`, `bc_target`, `n_episodes`, `n_pairs`,
  `bc_race`, `extras`); SC2 stats (`fn_idx_histogram`, `bc_player_id`,
  `n_replays_kept`, `n_replays_skipped_race`) land under `extras`;
  `policy_weights.yaml` written; on-disk JSON matches return value
- `supported_targets` membership is asserted explicitly so a new
  per-target fitter added to `games.sc2.replay_bc.fit_bc` without the
  corresponding adapter entry fails this test rather than silently being
  unreachable from `--bc`

## Framework BC seam

`framework/bc.py` and `framework/bc_io.py` — game-agnostic BC orchestrator
(issue #393, parent #392).  Two game-specific implementations are wired
on top: `games/sc2/bc_adapter.py` (issue #394) and
`games/tmnf/bc_adapter.py` (issue #395).  See
`docs/framework/bc_adapter.md` for the full protocol contract.

**Tested.** `BCAdapter` Protocol conformance via a fake adapter, the
generic `run()` orchestrator's validate → build → fit → save flow, NPZ
dataset round-trips (flat and per-episode), and `bc_summary.json` JSON
shape.  These tests do not touch any game.

### test_framework_bc.py — `BCAdapter` Protocol + `framework.bc.run` orchestrator (issue #393)
- `bc_io.load_dataset` round-trip (flat): all six keys present; obs / actions
  / `episode_starts` / `episode_lengths` / `episode_id` shapes and contents
  match what was written; `meta` JSON parses to original dict
- `bc_io.load_dataset` (as_episodes=True): returns one `(obs_seq, act_seq)`
  per episode in temporal order; episode boundaries preserved
- `bc_io.load_dataset` rejects an NPZ missing required keys with `KeyError`
- `bc_io.save_summary` writes indented JSON; creates missing
  `experiment_dir` ancestors
- `bc.run` orchestrator drives `validate_replay_dir` → `build_dataset` →
  `fit_bc` in order; passes the orchestrator's `training_params` through
  to `fit_bc` unchanged; canonical summary shape (`game`, `bc_target`,
  `n_episodes`, `n_pairs`, `bc_race`, `final_bc_loss`, `extras`); on-disk
  JSON matches return value
- `bc.run` saves `policy_weights.yaml` via the policy's `save()` method
- Race normalisation: `"ANY"` / `""` / `None` → no filter in the summary
  (`bc_race == "any"`); `"Terran"` → lowercase passed to the adapter
- Unsupported `target` raises `ValueError` before any adapter method is
  called
- `bc.run` creates the experiment directory if it doesn't exist
- Protocol attribute surface: a fake adapter satisfies the `BCAdapter`
  shape; the `GameAdapter` Protocol declares the optional `bc` attribute

## TMNF BC

`games/tmnf/bc_adapter.py` — TMNF implementation of the framework BC seam
introduced in issue #395.  Uses the in-game `SimplePolicy` as the
demonstration source (`.Replay.Gbx` replay-file ingest not currently supported).

**Tested.** Protocol surface, lstsq fit recovers a synthetic target
policy, full pipeline through `framework.bc.run` with a stubbed env, and
the removal of the legacy `do_pretrain: true` / `rl/pretrain.py`
landing pad.  The real TMNF env (TMInterface + Trackmania) is not
exercised — `_make_tmnf_env` is patched out and a fake env emits the
episodes.

### test_tmnf_bc.py — TMNF BC adapter (SimplePolicy source) + `do_pretrain` removal (issue #395)
- Protocol surface: `name="tmnf"`, `supported_targets=("hill_climbing",)`,
  `default_target="hill_climbing"`, all four methods callable
- `validate_replay_dir(None)` returns a synthetic
  `"<simple-policy-live-demos>"` marker
- `validate_replay_dir(<dir with .Replay.Gbx>)` raises `ValueError`
  with "not currently supported"; `build_dataset` with a non-None
  `replay_dir` raises the same way
- `fit_weighted_linear` recovers a synthetic target weight vector via
  lstsq (tolerance `1e-4`) when the training signal is noiseless
- `fit_bc` returns `(policy, loss)` where `loss` matches
  `_bc_residual_loss` on the same inputs; non-`hill_climbing` targets
  raise `ValueError`
- `collect_demos` produces the requested number of laps (`n_laps × ep_steps`
  rows, float32)
- `build_dataset` writes a framework-loadable NPZ: `meta["source"] ==
  "simple_policy"`, single contiguous episode, dimensions match the
  obs_spec
- `build_dataset` default `bc_n_demo_laps` matches the historical
  `N_DEMO_LAPS = 3` constant
- `bc_n_demo_laps <= 0` raises `ValueError` before any env is built
- End-to-end through `framework.bc.run` with a stubbed env: canonical
  summary shape; `policy_weights.yaml` and `bc_summary.json` written
- `TMNFAdapter.bc` is a `TMNFBCAdapter` instance
- `do_pretrain` removal: `do_pretrain` no longer a `RunConfig` field;
  `rl/pretrain.py` deleted; legacy `do_pretrain: true` in
  `training_params.yaml` is silently dropped by
  `RunConfig.from_training_params` rather than raising

## MineRL

`games/minerl/` — Phase 1 scaffold for Minecraft RL via MineRL (issue #215).

**Tested.** The 3-dim vector observation spec (compass angle, dirt inventory,
log inventory); the reward calculator (native reward scaling, step penalty,
finish bonus, reset clears episode accumulator); and the game adapter
(registered in `GAME_ADAPTERS`, experiment dir layout, track label with slash
sanitisation, build_game_spec wires the correct obs_spec / head_names /
discrete_actions).

**Not tested.** The actual `MineRLEnv` wrapper (requires `minerl` + Java 8
to be installed); real episode rollouts; MineRL-specific obs dict parsing
against a live environment.  Full env tests are deferred to the env
implementation sub-issue (#438).

### test_minerl_obs_spec.py — MineRL observation spec (3-dim vector)
- dim=3; names length=3; names unique; `compass_angle`, `inventory_dirt`, `inventory_log` present; compass scale=180.0; inventory scales=64.0; all dims carry descriptions

### test_minerl_adapter.py — MineRL game adapter
- registered under `minerl` key; `name`/`config_dir` correct; `experiment_dir` embeds game/policy/map/run; track override replaces map_name; `track_label` defaults to `MineRLNavigateDense-v0`; sanitises slashes; `build_probe`/`build_warmup` return None; `decorate_reward_cfg` is a no-op; `build_game_spec` returns correct `game_name`/`head_names`/`obs_spec`/`discrete_actions` shape (9,1)

### test_minerl_reward.py — MineRL reward config and calculator
- config defaults; `from_yaml` loads fields and ignores unknown keys; `compute` scales native reward, applies step penalty, adds finish bonus on termination only; missing `native_reward` defaults to 0.0; `reset()` clears `_total_native_reward` accumulator

## Rocket League

`games/rocket_league/` — single-agent RL for Rocket League via RLGym.

**Tested.** The 142-dim observation spec (self car, ball, 2 teammates,
3 opponents, relative features, boost pads); the reward calculator (velocity-to-ball dense
shaping, one-shot touch bonus that resets each episode, goal-scored /
goal-conceded sparse rewards, step penalty, combined additive total); and the
env wrapper (obs/action spaces, episode time-limit get/set, step 5-tuple, info
dict keys, boost detection from action[6], timeout truncation, close delegation,
tick_skip forwarding incl. `team_size=3`)
— all exercised against a mocked `rlgym` so no Rocket League install is required.

**Not tested.** The actual RLGym + Bakkesmod + Rocket League binary plumbing;
real episode rollouts and goal-detection signals from the game process; the
`vel_towards_ball` computation accuracy against live game data.

### test_rocket_league_obs_spec.py — Rocket League observation spec (142-dim)
- ObsSpec instance; dim matches base; dim=142; names length; scales shape+positive; obs_spec_list match; names unique; first=car_pos_x; boost_amount present; ball/friendly/opponent features present; relative features present; boost pad features present (10)
- car features are indices 0–17; ball features are 18–26; teammates are 27–62; opponents are 63–116; boost pads are 132–141
- with_zero_lidar same; with_lidar extends dim

### test_rocket_league_reward.py — Rocket League reward calc
- defaults; custom values; from_yaml; unknown key raises; loads bundled config
- step penalty; vel_to_ball; touch bonus fires once per episode; touch bonus resets after episode reset; goal scored; goal conceded; boost weight; combined additive total

### test_rocket_league_env.py — Rocket League env wrapper (mocked rlgym)
- obs/action space shape+bounds; episode time-limit get/set; reset obs+info; step 5-tuple; info keys; boost flag from action[6]; timeout truncation; close delegates to rlgym env
- tick_skip is forwarded from env/factory into `rlgym.make(..., team_size=3, self_play=False)`
- discrete actions shape (≥9, 8-dim); probe count=6, shape; warmup shape; action bounds respected

## CLI / misc

The entry point in `main.py` and the Assetto Corsa adapter, which is recent
enough to only have a smoke test.

**Tested.** That `main.py --game <name>` accepts every supported choice,
rejects unknown ones, exposes the option in `--help`, accepts `--track`,
and dispatches to `_run_one` for all games including assetto (now unified
via the `GameAdapter` protocol); that the Assetto Corsa adapter is
registered in `GAME_ADAPTERS`; that the Assetto Corsa adapter's obs spec,
env wrapper, reward calc and a 5-episode training loop all run against a
stubbed client.

**Not tested.** The interactive CLI itself (no terminal harness); the real
Assetto Corsa shared-memory client.

### cli/test_game_flag.py — `--game` CLI flag in `main.py`
- default tmnf; all valid choices accepted; invalid → SystemExit; help text mentions flag; `--track` accepted; main parser has all choices
- dispatch: all games (tmnf / beamng / car_racing / torcs / sc2 / assetto / rocket_league / iracing) → run_one; assetto registered in GAME_ADAPTERS; adapter experiment_dir contains game name

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
