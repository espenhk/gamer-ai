# Tests

- [Coverage at a glance](#coverage-at-a-glance)
- [Framework / shared](#framework--shared)
  - [test\_analytics\_no\_matplotlib.py (3) — analytics importable when matplotlib missing](#test_analytics_no_matplotlibpy-3--analytics-importable-when-matplotlib-missing)
  - [test\_analytics\_task\_metrics.py (17) — `TaskMetrics` dataclass + summary table formatting](#test_analytics_task_metricspy-17--taskmetrics-dataclass--summary-table-formatting)
  - [test\_belief.py (7) — fog-of-war belief encoder](#test_beliefpy-7--fog-of-war-belief-encoder)
  - [test\_build\_centerline.py (5) — `tracks/registry.json` builder](#test_build_centerlinepy-5--tracksregistryjson-builder)
  - [test\_consolidate.py (8) — grid-search result consolidation](#test_consolidatepy-8--grid-search-result-consolidation)
  - [test\_curiosity.py (12) — ICM/RND curiosity bonuses](#test_curiositypy-12--icmrnd-curiosity-bonuses)
  - [test\_discretize\_obs.py (6) — continuous→discrete obs binning](#test_discretize_obspy-6--continuousdiscrete-obs-binning)
  - [test\_distributed.py (29) — coordinator/worker protocol + HTTP server](#test_distributedpy-29--coordinatorworker-protocol--http-server)
  - [test\_early\_stopping.py (6) — early-stop logic in greedy + Q loops](#test_early_stoppingpy-6--early-stop-logic-in-greedy--q-loops)
  - [test\_env\_termination.py (7) — `_classify_termination()`](#test_env_terminationpy-7--_classify_termination)
  - [test\_game\_adapter.py (30) — TMNF/TORCS/SC2/BeamNG adapter abstractions](#test_game_adapterpy-30--tmnftorcssc2beamng-adapter-abstractions)
  - [test\_grid\_search.py (29) — Cartesian-product expansion + naming](#test_grid_searchpy-29--cartesian-product-expansion--naming)
  - [test\_info\_gain.py (7) — staleness-based intrinsic reward](#test_info_gainpy-7--staleness-based-intrinsic-reward)
  - [test\_obs\_memory.py (7) — frame-stacking observation wrapper](#test_obs_memorypy-7--frame-stacking-observation-wrapper)
  - [test\_reward.py (44) — TMNF reward calculator + curiosity glue](#test_rewardpy-44--tmnf-reward-calculator--curiosity-glue)
  - [test\_train\_rl\_signature.py (4) — public `train_rl()` API](#test_train_rl_signaturepy-4--public-train_rl-api)
  - [test\_new\_best\_logging.py (24) — `_log_new_best_details` + `_print_episode_summary`](#test_new_best_loggingpy-24--_log_new_best_details--_print_episode_summary)
  - [test\_utils.py (13) — math/state-extraction utils](#test_utilspy-13--mathstate-extraction-utils)
  - [test\_track.py (11) — centreline geometry helpers](#test_trackpy-11--centreline-geometry-helpers)
- [TMNF policies](#tmnf-policies)
  - [test\_weighted\_linear\_policy.py (11) — linear `WeightedLinearPolicy`](#test_weighted_linear_policypy-11--linear-weightedlinearpolicy)
  - [test\_neural\_net\_policy.py (7) — pure-numpy MLP policy](#test_neural_net_policypy-7--pure-numpy-mlp-policy)
  - [test\_genetic\_policy.py (16) — population evolutionary loop](#test_genetic_policypy-16--population-evolutionary-loop)
  - [test\_cmaes\_policy.py (33) — CMA-ES on linear weights](#test_cmaes_policypy-33--cma-es-on-linear-weights)
  - [test\_epsilon\_greedy\_policy.py (7) — tabular Q-learning](#test_epsilon_greedy_policypy-7--tabular-q-learning)
  - [test\_mcts\_policy.py (6) — UCT-style online Q](#test_mcts_policypy-6--uct-style-online-q)
  - [test\_neural\_dqn\_policy.py (22) — DQN (replay + target net)](#test_neural_dqn_policypy-22--dqn-replay--target-net)
  - [test\_reinforce\_policy.py (22) — Monte-Carlo policy gradient](#test_reinforce_policypy-22--monte-carlo-policy-gradient)
  - [test\_lstm\_policy.py (49) — LSTM evolution policy](#test_lstm_policypy-49--lstm-evolution-policy)
- [TMNF I/O](#tmnf-io)
  - [test\_rl\_client.py (22) — `RLClient` (game-thread bridge, action windowing)](#test_rl_clientpy-22--rlclient-game-thread-bridge-action-windowing)
- [TORCS](#torcs)
  - [test\_torcs\_obs\_spec.py (14) — TORCS observation spec (19-dim)](#test_torcs_obs_specpy-14--torcs-observation-spec-19-dim)
  - [test\_torcs\_client.py (12) — TORCS state→obs and action mapping](#test_torcs_clientpy-12--torcs-stateobs-and-action-mapping)
  - [test\_torcs\_env.py (13) — Gym env wrapper](#test_torcs_envpy-13--gym-env-wrapper)
  - [test\_torcs\_reward.py (13) — TORCS reward calc](#test_torcs_rewardpy-13--torcs-reward-calc)
  - [test\_torcs\_analytics.py (11) — TORCS plots/report](#test_torcs_analyticspy-11--torcs-plotsreport)
- [SC2](#sc2)
  - [test\_sc2\_obs\_spec.py (17) — SC2 obs spec](#test_sc2_obs_specpy-17--sc2-obs-spec)
  - [test\_sc2\_actions.py (14) — discrete action grid](#test_sc2_actionspy-14--discrete-action-grid)
  - [test\_sc2\_reward.py (23) — SC2 reward calc](#test_sc2_rewardpy-23--sc2-reward-calc)
  - [test\_sc2\_client.py (64) — PySC2 client wrapper](#test_sc2_clientpy-64--pysc2-client-wrapper)
  - [test\_sc2\_env.py (27) — SC2 env wrapper](#test_sc2_envpy-27--sc2-env-wrapper)
  - [test\_sc2\_apm\_limiter.py (29) — token-bucket APM limiter + SC2Env integration](#test_sc2_apm_limiterpy-29--token-bucket-apm-limiter--sc2env-integration)
  - [test\_sc2\_belief\_integration.py (15) — fog-of-war belief system wired into SC2Env (issue #111)](#test_sc2_belief_integrationpy-15--fog-of-war-belief-system-wired-into-sc2env-issue-111)
  - [test\_sc2\_cmaes\_policy.py (21) — `SC2CMAESPolicy` (CMA-ES over multi-head linear policy)](#test_sc2_cmaes_policypy-21--sc2cmaespolicy-cma-es-over-multi-head-linear-policy)
  - [test\_sc2\_lstm\_policy.py (35) — `SC2LSTMPolicy` + `SC2LSTMEvolutionPolicy`](#test_sc2_lstm_policypy-35--sc2lstmpolicy--sc2lstmevolutionpolicy)
  - [test\_sc2\_genetic\_policy.py (68) — `SC2MultiHeadLinearPolicy` + genetic trainer](#test_sc2_genetic_policypy-68--sc2multiheadlinearpolicy--genetic-trainer)
  - [test\_sc2\_neural\_dqn\_policy.py (17) — masked DQN for SC2](#test_sc2_neural_dqn_policypy-17--masked-dqn-for-sc2)
  - [test\_sc2\_cnn\_policy.py (32) — CNN feature extractor + CMA-ES variant](#test_sc2_cnn_policypy-32--cnn-feature-extractor--cma-es-variant)
  - [test\_sc2\_reinforce\_policy.py (39) — REINFORCE policy for SC2](#test_sc2_reinforce_policypy-39--reinforce-policy-for-sc2)
  - [test\_sc2\_play.py (21) — `play_sc2.py` script](#test_sc2_playpy-21--play_sc2py-script)
  - [test\_sc2\_simple64\_training.py (31) — Simple64 ladder integration](#test_sc2_simple64_trainingpy-31--simple64-ladder-integration)
  - [test\_sc2\_analytics.py (51) — SC2-specific analytics plots and flags](#test_sc2_analyticspy-51--sc2-specific-analytics-plots-and-flags)
- [CLI / misc](#cli--misc)
  - [cli/test\_game\_flag.py (14) — `--game` CLI flag in `main.py`](#clitest_game_flagpy-14----game-cli-flag-in-mainpy)
  - [assetto\_corsa/test\_smoke.py (8) — Assetto Corsa smoke tests (against fake client)](#assetto_corsatest_smokepy-8--assetto-corsa-smoke-tests-against-fake-client)
- [Why 868 tests run in ~50 s](#why-868-tests-run-in-50-s)

895 tests across 54 files. Runs in ~50 seconds via `python -m pytest tests/` (excluding tests that require tminterface, pysc2 live env, gym_torcs, or the SC2 binary). The full suite including those files has 1030 tests.

## Coverage at a glance

The suite is exhaustive on **pure logic** — config parsing, reward math, policy
math, save/load round-trips, CLI flag dispatch — and silent on **anything that
needs a running game or display**. Every game client is replaced by a fake or
a `MagicMock`; no Trackmania, TORCS, SC2, BeamNG or Assetto Corsa binary is
ever launched, no game window is grabbed, no matplotlib window is rendered, no
keyboard/joystick output is sent. End-to-end "does the agent actually drive
faster after training" is not covered — that belongs to manual experiment
runs, not the unit suite.

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
naming + nested `policy_params` promotion; the early-stop streak logic in
both greedy and Q loops; the distributed coordinator/worker JSON protocol and
the in-process HTTP server (work queue, heartbeat re-queue, auth);
`train_rl()`'s public signature; the new-best log helpers
(`_log_new_best_details` per-component / action-frequency / TMNF / SC2 kill
groups, `_print_episode_summary` compact format); that `framework.analytics`
imports cleanly on a machine with no matplotlib; and the `redo_analytics.py`
script (game auto-detection, single-experiment regeneration,
multi-experiment summaries, `--no-individual`, inferred summary dir,
graceful skip of missing experiments).

**Not tested.** Real distributed training across multiple machines (only the
in-process HTTP loopback is exercised); actual matplotlib rendering or PNG
diff'ing (analytics tests assert files appear, not their contents); the Azure
Terraform stack under `infrastructure/`; the Windows bootstrap script
`setup_and_run.ps1` (PowerShell-only; cannot run on Linux CI); long convergence
behaviour of the actual `train_rl()` loop end-to-end on a real env.

### test_analytics_no_matplotlib.py (3) — analytics importable when matplotlib missing
- framework analytics import works without matplotlib
- TMNF analytics import works without matplotlib
- SC2 analytics import works without matplotlib

### test_analytics_task_metrics.py (17) — `TaskMetrics` dataclass + summary table formatting
- new fields default to `None`; finish_time / lateral_offset / reward_components stored
- finish-rate aggregation: empty / none / all / partial
- summary string: empty / contains finish rate / best finish time / no time when none / lateral offset
- summary table: progress / finish-time / dash-on-no-finish / lateral-offset columns

### test_belief.py (7) — fog-of-war belief encoder
- initial encode all zero; update sets value+confidence; project decays confidence
- scout-then-lose-sight then decay; reset clears; per-slot τ; encode shape

### test_build_centerline.py (5) — `tracks/registry.json` builder
- creates registry when absent; entry has expected fields; upsert overwrites; upsert preserves siblings; multi-track sorted keys

### test_consolidate.py (8) — grid-search result consolidation
- experiment-data round-trip / load missing raises / valid JSON
- creates results dir; produces summary; infers summary dir; skips missing; detects varied keys

### test_redo_analytics.py (20) — `redo_analytics.py` re-generation script
- game detection: SC2 inferred from `map_name` / `agent_race`; TMNF default; empty params → TMNF
- analytics loader: tmnf returns callables; unknown game falls back gracefully
- single experiment: regenerates results.md / greedy_rewards.png; no summary without `--summary-name`; summary produced when `--summary-name` given; missing dir skipped without crash; `--no-individual` without summary raises ValueError
- multiple experiments: summary.md written; contains experiment names; individual results regenerated; `--no-individual` skips results.md but writes summary; summary dir inferred from common parent; default summary name is 'combined'; missing experiment skipped; SC2 auto-detected and run without crash

### test_curiosity.py (12) — ICM/RND curiosity bonuses
- ICM: reward decreases on repeat / non-negative / dim mismatch raises / β raises / η scales
- RND: reward decreases / target frozen / non-negative
- factory: `none` returns `None`; ICM / RND factories; unknown kind raises

### test_discretize_obs.py (6) — continuous→discrete obs binning
- zero→middle bin; clipped high→max; clipped low→min; symmetry; tuple-of-int return; length matches obs_dim

### test_distributed.py (29) — coordinator/worker protocol + HTTP server
- ComboSpec round-trip + JSON serialisable; ResultPayload round-trip + valid JSON
- payload preserves greedy_sims / throttle counts / trace / none-trace / metadata / task-metric fields / SC2 analytics fields
- numpy arrays serialised
- HTTP: serves all combos / status endpoint / result accepted + done event
- unknown combo rejected; duplicate result ignored; stale worker re-queued; heartbeat prevents requeue
- empty queue returns immediately; unauthorized rejected
- game-filter: X-Worker-Game header returns matching combo; 204 when no match; preserves queue order for non-matching items
- skip endpoint: POST /skip returns in-progress item to queue; unknown item is no-op
- ComboSpec.game defaults to 'tmnf'; explicit game values round-trip correctly

### test_early_stopping.py (6) — early-stop logic in greedy + Q loops
- stops after patience-no-improvement; patience=0 runs all; streak resets on improvement; early-stop sim recorded
- Q loop: stops on patience / patience=0 runs all

### test_env_termination.py (7) — `_classify_termination()`
- finish / crash / hard-crash / timeout / still-running; finish > crash priority; reason key always present

### test_game_adapter.py (30) — TMNF/TORCS/SC2/BeamNG adapter abstractions
- registry: all games registered; adapter instantiable
- TMNF: experiment_dir includes track / track override / track_label default+override / build_probe / build_warmup / build_extras / decorate_reward_cfg
- TORCS: experiment_dir root / dir / track_label default+override / build_probe/warmup/extras = None
- SC2: experiment_dir includes map_name / track override / track_label / build_probe/warmup = None
- BeamNG: experiment_dir / build_probe = None
- AssettoCorsa: experiment_dir / build_probe = None

### test_grid_search.py (29) — Cartesian-product expansion + naming
- expansion: no variation / single training axis / single reward axis / cartesian product / fixed params preserved
- flat dict: contains varied / no-flat-key when not varied
- naming: no varied / single varied / negative-float `n` prefix / multiple varied / unknown key passthrough
- nested policy_params: passthrough / top-level promoted / top-level overrides nested / all keys mapped / correct names
- promoted-keys: no params returns empty / lstm hidden_size / reinforce baseline
- format helpers: int / float strips zeros / negative float / string
- `--game` flag: default tmnf / honoured / track field / track none / unaffected by game field

### test_info_gain.py (7) — staleness-based intrinsic reward
- initial staleness all 1; never-observed = max; just-observed near zero; grows linearly
- reward fires stale→fresh; zero when weight=0; reset restores

### test_obs_memory.py (7) — frame-stacking observation wrapper
- shape; reset fills initial; step shifts frames; k=1 passthrough; invalid k raises; most-recent zero-padded; clear

### test_reward.py (44) — TMNF reward calculator + curiosity glue
- Config: defaults / custom / unknown-key raises / valid keys / partial keys
- Components: progress / no-progress / centerline quadratic / on-center zero / finish bonus / over-par penalty / no-finish-no-bonus / accel bonus / step penalty / airborne / airborne above centreline
- n_ticks scaling: centerline / speed / airborne / accel-bonus scale; finish-bonus + progress do *not* scale
- Track fields: default name / centerline path / custom / from yaml / backward-compat / default config back-compat
- Curiosity: ICM adds positive intrinsic / scales w/ n_ticks / skipped when obs missing / reset propagates / yaml accepts new keys
- `compute_with_components`: scalar matches / sum=total / keys present / progress / centerline / finish-bonus / finish-time over-par / no-finish / step-penalty / accel-bonus / curiosity zero w/o module

### test_train_rl_signature.py (4) — public `train_rl()` API
- accepts game+config params; accepts optional specs; accepts control flags; no legacy flat params

### test_new_best_logging.py (24) — `_log_new_best_details` + `_print_episode_summary`
- `_print_episode_summary`: terminated/finished/truncated one-liner; `r=` and `steps=` present; laps and progress omitted
- `_log_new_best_details` — empty info emits nothing
- reward components: one log line per non-zero component; zero values omitted; prev comparison shown; no-prev no comparison
- action frequency: one log line per action with SC2 function names; prev comparison shown
- TMNF metrics: progress; lateral offset; finish time only when `finished=True`; prev comparison for progress + offset (all on one combined line)
- SC2 kills: units + structures on one line; prev comparison; absent when key not in info; suppressed when both values zero
- SC2 game-state averages: one log line per non-zero metric; zero values omitted; prev comparison
- all five groups together emit seven lines (2 components + 2 actions + 1 progress + 1 kills + 1 game-state)

### test_utils.py (13) — math/state-extraction utils
- vector magnitude: zero / unit / 3D / compute_speed alias
- yaw/pitch/roll identity = 0; 90° yaw correct
- state extraction: velocity / wheels / centerline-progress / 3-entry lookahead with centerline / zero lookahead without

### test_track.py (11) — centreline geometry helpers
- start / end / midpoint progress; nonzero lateral; on-centreline zero lateral; forward unit vector
- lookahead: returns two floats / straight-track zero heading change / finite lateral / clamps at end / opposite sign across centreline

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

### test_weighted_linear_policy.py (11) — linear `WeightedLinearPolicy`
- action in range; deterministic; accel/brake weight dominance; coast within threshold; left/right steer
- from_cfg roundtrip; mutated weights differ; obs_scales length matches names; action is int

### test_neural_net_policy.py (7) — pure-numpy MLP policy
- action in range; deterministic; from_cfg roundtrip; hidden_sizes preserved; output 9 actions; mutated differs; weight matrix shapes

### test_genetic_policy.py (16) — population evolutionary loop
- init random pop size + champion set; init from champion seeds pop
- evaluate-and-evolve: champion reward / returns true on improve / false otherwise
- crossover from both parents; pop replaced after evolution
- eval_episodes: default=1 / stored / in to_cfg / cfg roundtrip / cfg default / single reward / 3-episode average / reset count

### test_cmaes_policy.py (33) — CMA-ES on linear weights
- defaults: pop size / μ=λ/2 / weights sum=1 / C=I at init
- init: random zero mean / from champion seeds mean / sets champion
- sample: returns count / WeightedLinearPolicy instances / fills pop_xs/ys
- update: sets champion first / true on improve / false otherwise / tracks best / generation increments / wrong reward count raises / no sample raises / mean moves
- call: raises before update / valid action after
- cfg: required keys / policy_type / save writes WL yaml
- eval_episodes: default 1 / stored / in to_cfg / clamped ≥ 1 / single reward / 3-ep average / reset count
- convergence: quadratic max / save+load roundtrip all arrays / wrong dim raises

### test_epsilon_greedy_policy.py (7) — tabular Q-learning
- action in range; greedy picks best; update +/- reward; Bellman backup; ε decays per episode; ε floored

### test_mcts_policy.py (6) — UCT-style online Q
- action in range; unseen state random; visit count increments; Q changes; exploitation prefers high Q; visits accumulate

### test_neural_dqn_policy.py (22) — DQN (replay + target net)
- ReplayBuffer: push+len / circular eviction / sample shapes / w/o replacement / with replacement when small
- Policy: action shape+range / greedy discrete / random when ε=1 / buffer fills on update / ε decays / floored / target sync / weight shapes
- Cfg: roundtrip / policy_type key / on_episode_end no-op / missing keys raise / shape mismatch raises
- Bandit convergence; save/load replay buffer + Adam moments; wrong obs_dim raises

### test_reinforce_policy.py (22) — Monte-Carlo policy gradient
- action shape; steer range; accel/brake binary; discrete
- buffers: fill / clear on episode end / empty on_episode_end no-op
- weights match hidden_sizes; buffer lengths match; weights change after update; gradient direction
- entropy_coeff = 0 vs nonzero; cfg required keys / policy_type / restore weights+hyperparams; save+reload; lidar; baseline roundtrip; wrong obs dim raises

### test_lstm_policy.py (49) — LSTM evolution policy
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

### test_rl_client.py (22) — `RLClient` (game-thread bridge, action windowing)
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

### test_torcs_obs_spec.py (14) — TORCS observation spec (19-dim)
- ObsSpec instance; dim matches base; dim=19; names length; scales shape+positive; obs_spec_list match; names unique; first=speed; last=track_pos; track_edges present
- with_zero_lidar same; with_lidar extends dim+names

### test_torcs_client.py (12) — TORCS state→obs and action mapping
- output shape; speed conversion; lateral offset; angle; progress; rpm; wheel spin; track position
- full throttle straight; full brake; steer clipped; combined accel+brake

### test_torcs_env.py (13) — Gym env wrapper
- obs/action space shape+bounds; episode time-limit get/set; reset returns obs+info; step 5-tuple; crash terminates; info keys; close calls client.close
- discrete actions shape; probe count; warmup shape

### test_torcs_reward.py (13) — TORCS reward calc
- defaults / custom / from_yaml / unknown raises / loads torcs config
- progress / centerline / speed / finish / accel / step penalty; n_ticks scaling; lap wraparound

### test_torcs_analytics.py (11) — TORCS plots/report
- plot greedy action dist / progress / termination reasons / cold-start dist
- weight heatmap no-file safe; weight evolution no-weights; save plots no crash; save full report; empty experiment safe; grid summary; gs comparison progress

## SC2

`games/sc2/` — the StarCraft 2 integration via PySC2. Largest per-game
section (≈250 tests) because both the minigame and Simple64 ladder paths,
plus six policy variants and three obs encodings (flat / dict / spatial),
have to be covered.

**Tested.** All three observation specs (15-dim minigame, 45-dim ladder,
97-dim rich) and the get-spec dispatch; the minimap enemy centroid
(`minimap_enemy_cx/cy`) that enables the policy to locate the beacon even
when it is off the current camera view (beacon-idling fix); the `_action_to_call`
no-spam fix (blocked Move_screen issues select_army once, then no_op on
consecutive blocked steps); the 9-cell discrete action grid
(centre = `select_army`, others = `Move_screen`); the SC2 reward calculator
(score delta, win bonus, loss penalty, economy weight, idle penalty, step
penalty); the SC2 client wrapper (flat-obs construction, score-delta
threading, player_relative centroid, terminal-outcome handling, ladder
visibility tracking with fog distinction, and the rich-preset extractors added
in issue #135: enemy unit-type counts, shield/energy screen summaries, creep
coverage fraction, and economy-pipeline features); the SC2 env wrapper (reset /
step / done / info / custom reward config); the APM limiter (`ApmLimiter`
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

### test_sc2_obs_spec.py (17) — SC2 obs spec
- minigame dim (15); ladder dim (45); rich dim (102 = exact breakdown); ladder extends minigame; default = minigame; get_spec for minigame / ladder; minigame count; obs_names match dims
- minimap_enemy_cx/cy present in all presets (minigame, ladder, rich)
- rich spec contains new rich-only feature names (selected_avg_shields/energy, screen_visibility_frac, screen_unit_density_aa_mean, self_weapon_cooldown_mean); not present in ladder

### test_sc2_actions.py (14) — discrete action grid
- shape / dtype / xy in unit square; centre = select_army; others = move_screen
- probe actions count / shape; warmup shape / select_army; function_ids table complete

### test_sc2_reward.py (23) — SC2 reward calc
- defaults; from_yaml; unknown raises; loads bundled config
- score delta; step penalty only; step penalty n_ticks scaling; win bonus; loss penalty; no-outcome no bonus; economy weight; idle penalty when idle / not when busy

### test_sc2_client.py (64) — PySC2 client wrapper
- minigame flat obs shape; score-delta threading; player_relative centroid; terminal outcome recorded
- ladder flat obs shape; visibility tracking; fogged ≠ visible; ladder terminal outcome; non-terminal = None
- rich extractors (#135): enemy unit-type counts (owner filter, missing field, unknown type); shield/energy (self shield mean, no units, None screen); creep (half coverage, no creep, None minimap); economy pipeline (upgrade count, build queue, cargo, all missing); rich spec contains new names; ladder spec unchanged
- selected-unit extras: shields + energy from cols 3/4 of single/multi_select; empty selection → zeros; short rows don't crash
- screen_visibility_frac: all visible → 1.0; half visible → 0.25; fogged not counted; None screen / missing layer → 0
- screen_unit_density_aa_mean: mean of unit_density_aa layer; zero layer; None screen / missing layer → 0
- self_weapon_cooldown_mean: mean for alliance==1 from col 25; all ready → 0; no self units → 0; missing feature_units → 0; too few cols → 0
- minimap enemy centroid: minimap_enemy_cx/cy computed from player_relative==4 layer; correct when beacon present; zero when no beacon on minimap (edge case)
- action fallback (#124, beacon-idling fix): blocked Move_screen → select_army once, then no_op on consecutive blocked steps; pending flag cleared when Move_screen available; no_op action passes through unchanged

### test_sc2_env.py (27) — SC2 env wrapper
- minigame obs space; action space shape+bounds; ladder obs space; episode time-limit get/set
- reset returns obs+info; step 5-tuple; score-delta reward; done terminates; loss outcome
- close calls client.close; info keys; prev_score threaded; custom reward config
- end-screen analytics: series absent on mid-episode step; present on terminal step; supply_capped_fraction correct; army series value; resource series sums minerals+vespene; starting units excluded from build order; new units produce events; empty build order when no unit_counts

### test_sc2_apm_limiter.py (29) — token-bucket APM limiter + SC2Env integration
- Construction: valid / zero/negative max_apm raises / zero/negative burst_s raises / max_tokens formula / starts full
- Basic behaviour: no-op always allowed; no-op free (no token consumed); first action passes; second blocked when empty; refills over time; tokens capped at max; reset refills; burst capacity; high-APM burst; default fn_idx consumes token
- Rolling budget: 300 APM over 60 s allows ~300 total; first second capped at burst window
- Env integration (disabled): no limiter attribute; action passed unchanged; apm_throttled=False; episode count stays zero
- Env integration (enabled): limiter created; first action passes; second throttled to no_op; no_op never throttled; throttled-steps counter accumulates; counter resets on new episode; action passes after refill

### test_sc2_belief_integration.py (15) — fog-of-war belief system wired into SC2Env (issue #111)
- obs shape = base + 192 dims with `enable_belief=True` for both minigame and ladder maps
- `enable_belief=False` leaves obs shape unchanged
- reset obs: belief enc all-zero, staleness all-ones; second reset clears prior state
- step appends dims even when `minimap_vis` absent; enc non-zero + staleness drops after visible step
- scout reward > 0 on first visit with fully-visible minimap; zero when no visible regions
- `episode_reward_components` always contains `scout` key; accumulates across steps

### test_sc2_cmaes_policy.py (21) — `SC2CMAESPolicy` (CMA-ES over multi-head linear policy)
- Dimension: θ = (N_FUNCTION_IDS + N_SPATIAL_ROWS) × obs_dim for minigame and ladder obs specs
- Sample population: count matches λ / all individuals are SC2MultiHeadLinearPolicy
- Mechanics: update_distribution before sample raises / σ adapts across generations / champion improves monotonically / wrong reward count raises
- Call: raises before first generation / returns valid 4-vec after one generation
- Masking: restricts fn_idx to available set / fallback to no_op when set empty / updated via update() kwargs / no masking when None
- Serialisation: champion YAML round-trip lossless / trainer-state npz round-trip / dim mismatch raises / initialize_from_champion sets mean / initialize_random zeros mean

### test_sc2_lstm_policy.py (35) — `SC2LSTMPolicy` + `SC2LSTMEvolutionPolicy`
- Structure: hidden state zero at init / W_out shape = (N_FUNCTION_IDS+N_LSTM_SPATIAL_CELLS, hidden_size) / flat_dim formula matches both minigame and ladder / to_flat length correct / with_flat round-trip / wrong size raises
- Action: shape (4,) / fn_idx in [0, N_FUNCTION_IDS) / x,y in [0,1] / hidden state advances after step
- Masking: never selects unavailable fn / set via on_episode_start info / updated via update kwargs / fallback to no_op when all masked
- Hidden state reset: reset_on_episode=True zeros state on episode start + end / reset_on_episode=False carries state across resets
- Serialisation: to_cfg / from_cfg round-trip lossless / save / load round-trip / policy_type = "sc2_lstm"
- Evolution: population size / individuals are SC2LSTMPolicy / call raises before generation / champion set after one generation / σ adapts / wrong reward count raises / flat_dim mismatch raises / initialize_from_champion sets mean / on_episode_start forwarded to champion / save writes yaml / trainer-state round-trip / dim mismatch raises

### test_sc2_genetic_policy.py (68) — `SC2MultiHeadLinearPolicy` + genetic trainer
- Weight shapes (fn / spatial × minigame / ladder); flat dim (mini/ladder); explicit weights stored
- Call: returns 4-vec / fn_idx range / spatial unit range / queue=0 / max-fn / max-spatial
- Cfg: keys / *_weights suffix / values are obs-name dicts / from_cfg roundtrip / yaml lossless / missing default zero
- Flat: length / roundtrip / mutated differs / preserves spec / weights differ / share=0 unchanged
- Genetic: pop / elite_k / eval_episodes / head names cover rows
- Init random: pop size / SC2 policies / champion set / pop θ dim
- Init from champion: pop size / champion / SC2 policies
- Crossover: valid policy / from both parents / θ dim / weights only from parents
- Evolve: champion reward / true on improve / false otherwise / elite count preserved / new members are SC2
- Save: yaml / champion lossless / cfg policy_type=sc2_genetic / from_cfg roundtrip / restores champion / no champion key OK
- Call: 4-vec after init / raises before init
- Available-actions masking: no-mask selects highest fn / None by default / masking blocks unavailable fn / selects best available / on_episode_start caches ids / no key clears mask / None info clears mask / update caches ids / no key in update leaves unchanged / mask applied after on_episode_start / mask applied after update / empty set falls back to no_op

### test_sc2_neural_dqn_policy.py (17) — masked DQN for SC2
- fn_idx_for_cell: centre=select_army / others=move_screen / consistent / int
- Available mask: None=all true / empty=all false / select_army only / move_screen only / both / dtype bool
- Masked replay buffer: push+len / default mask all true / 6-tuple sample / mask shape / preserves mask / circular eviction
- Policy: illegal never picked greedy / picks best / random respects mask / all-masked fallback / mask cached from update info / from None info / on_episode_start resets / buffer is masked / mask stored / all-true when no info / fills on update / gradient step runs / illegal Q not bootstrapped
- Cfg: policy_type / from_cfg roundtrip / trainer-state roundtrip / masks preserved / backward-compat no masks / shape mismatch raises
- available_fn_ids: None when missing / key always present

### test_sc2_cnn_policy.py (32) — CNN feature extractor + CMA-ES variant
- Conv layer: output shape / ReLU zeros negatives
- Pool: output shape / uniform input preserved
- CNN: flat_dim formula / varies w/ channels / forward shapes / callable returns 4-vec / with_flat roundtrip / wrong size raises / non-dict obs raises / flat-concat dim
- CMA-ES: pop size / sample correct count / individuals callable / update returns bool / champion improves / champion callable / wrong rewards raises / no sample raises / σ adapts / trainer-state roundtrip / save+load champion
- Spatial obs: flat space when no layers / dict space when layers / spatial shape matches channels / reset dict obs / fills zeros when none / step dict obs / spatial in info / normalised / no spatial when no layers

### test_sc2_reinforce_policy.py (39) — REINFORCE policy for SC2
- Action: 4-vec shape / fn_idx in range / x+y in unit square / queue=0
- Buffers: episode buffer fills / clears on end / empty end no-op
- Gradient: weights change after update / direction improves expected action
- Available-actions masking: illegal fn_idx masked out / mask updates from info kwarg
- Serialisation: cfg keys / policy_type / from_cfg roundtrip / save+reload / wrong obs dim raises

### test_sc2_play.py (21) — `play_sc2.py` script
- Missing weights raises; loads sc2_multi_head for sc2_genetic / correct weights / neural_dqn / reinforce / lstm; cmaes no policy_type → SC2Linear; unknown → SC2Linear
- Outcome handling: win / loss / draw / none
- Episode loop: calls policy each step / client until done / on_episode_start+end / works without lifecycle hooks
- Play-mode flag: stored / default false; `make_sc2_env` lazy on reset
- Game loop: present in info / value extracted from obs

### test_sc2_simple64_training.py (31) — Simple64 ladder integration
- Obs: 21 dim; reset shape; step 5-tuple
- Outcomes: win+bonus / loss+penalty; economy reward flows through
- Action: fn_idx valid / xy continuous / queue binary / shape
- Pop usage: sc2_genetic uses SC2Linear / cmaes offspring use SC2Linear; SC2Linear and sc2_genetic action shape
- Per-policy on Simple64: epsilon_greedy / mcts / neural_dqn shape+update / cmaes sample+update / reinforce shape+episode / lstm shape / lstm-evolution sample+update
- Training loops (mocked env): sc2_genetic / cmaes / neural_dqn / reinforce / lstm
- Trainer state roundtrips: cmaes / neural_dqn

### test_sc2_analytics.py (51) — SC2-specific analytics plots and flags
- `SUPPORTS_THROTTLE=False` / `SUPPORTS_PATH=False` flags
- `GreedySimResult` new fields: `action_counts` / `obs_averages` / `xy_hist` — default None; stored correctly
- `GreedySimResult` end-screen fields: `supply_capped_fraction` / `build_order` / `army_count_series` / `resource_series` — default None; stored correctly
- `plot_action_frequency`: renders to file / skips when no data / skips when no sims / single fn_idx
- `plot_obs_averages`: renders to file / skips when no data / skips when all-zero / unknown feature key safe
- `plot_spatial_heatmap`: renders to file / skips when no data / skips all-zero hist / partial None sims ignored
- `plot_outcome_breakdown`: renders to file / skips when all None / skips when no sims / win+loss ladder
- `plot_supply_capped`: renders to file / skips when all None / skips when no sims / zero fraction renders
- `plot_resource_series`: renders to file / uses best sim / skips when no series / skips when no sims / falls back to last sim when none improved
- `plot_army_count`: renders to file / skips when no series / skips when no sims
- `plot_build_order`: renders to file / skips when no build order / skips when no sims / single unit type / multiple unit types
- `save_experiment_results`: writes results.md / writes SC2 plots (incl. new 4) / mentions game / no crash empty sims / no racing plots written

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

### cli/test_game_flag.py (14) — `--game` CLI flag in `main.py`
- default tmnf; all valid choices accepted; invalid → SystemExit; help text mentions flag; `--track` accepted; main parser has all choices
- dispatch: tmnf / beamng / car_racing / torcs / sc2 → run_one; assetto → run_assetto; assetto missing dep → ValueError; adapter experiment_dir contains game name

### assetto_corsa/test_smoke.py (8) — Assetto Corsa smoke tests (against fake client)
- obs spec dimensions match base obs_dim; env reset obs shape; step 5-tuple finite reward; info reflects current step; env terminates on finish; vision features; reward calc finite; 5-episode training loop with linear policy

---

## Why 854 tests run in ~50 s

These tests look heavy because of the names ("training loop", "env reset", "DQN convergence") but operationally they're almost all pure-Python unit tests with zero external I/O:

1. **No game binaries are launched.** TMInterface, the SC2 binary, and TORCS are never started. `RLClient`, `SC2Client`, the SC2 env, and the TMNF env are all driven through fakes and `MagicMock` patches (e.g. `test_rl_client.py`, `test_env_termination.py`, `test_sc2_play.py`). The single "five-episode training loop" smoke test (`assetto_corsa/test_smoke.py`) runs against a stubbed client.
2. **All "policies" are pure numpy.** No PyTorch, no TensorFlow, no GPU. The DQN, REINFORCE, LSTM, CMA-ES, CNN, and SC2 multi-head policies are hand-rolled numpy with hidden sizes like `[8, 8]` or `hidden_size=4` in tests. Forward+backward passes are sub-millisecond.
3. **Tiny tensors.** Where convergence is asserted (`test_neural_dqn_policy.test_bandit_convergence`, `test_cmaes_policy.test_converges_toward_quadratic_maximum`, `test_reinforce_policy.test_gradient_direction`), the problem is a 2-arm bandit or a quadratic — a few hundred steps on tiny vectors.
4. **Whole files are config / dataclass tests.** `test_grid_search.py` (29), `test_reward.py` (44), `test_sc2_genetic_policy.py` (68), `test_torcs_obs_spec.py` (14), `test_analytics_task_metrics.py` (17), `test_game_adapter.py` (26) are mostly "from_yaml round-trip / shape / default-value / cartesian product" — microseconds each.
5. **No matplotlib rendering.** TORCS analytics tests use `Agg` (non-interactive) and dump to `tmp_path`; `test_analytics_no_matplotlib.py` explicitly checks the import path that *avoids* it.
6. **Filesystem work uses `tmp_path`** (RAM-backed `/tmp`), and the only network is `test_distributed.py` binding `localhost` for HTTP coordinator tests — which is why that's the one file with `time.sleep` and is still milliseconds because it talks to itself.
7. **Heavy collection work is amortised.** `pytest`'s ~1 second startup + 53 collection modules is a small share of the wall clock; once collected, 839 mostly-arithmetic asserts run in the remaining ~49 seconds.

Roughly: 854 tests × ~58 ms average = ~49 s of work + ~1 s of import/collection ≈ 50 s total — nothing in the suite waits on a game tick, a network packet, or a GPU.
