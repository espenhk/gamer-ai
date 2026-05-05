# Tests

740 tests across 45 files. Runs in ~25 seconds via `python -m pytest tests/`.

## Framework / shared

### test_analytics_no_matplotlib.py (2) — analytics importable when matplotlib missing
- framework analytics import works without matplotlib
- TMNF analytics import works without matplotlib

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

### test_curiosity.py (12) — ICM/RND curiosity bonuses
- ICM: reward decreases on repeat / non-negative / dim mismatch raises / β raises / η scales
- RND: reward decreases / target frozen / non-negative
- factory: `none` returns `None`; ICM / RND factories; unknown kind raises

### test_discretize_obs.py (6) — continuous→discrete obs binning
- zero→middle bin; clipped high→max; clipped low→min; symmetry; tuple-of-int return; length matches obs_dim

### test_distributed.py (20) — coordinator/worker protocol + HTTP server
- ComboSpec round-trip + JSON serialisable; ResultPayload round-trip + valid JSON
- payload preserves greedy_sims / throttle counts / trace / none-trace / metadata / task-metric fields
- numpy arrays serialised
- HTTP: serves all combos / status endpoint / result accepted + done event
- unknown combo rejected; duplicate result ignored; stale worker re-queued; heartbeat prevents requeue
- empty queue returns immediately; unauthorized rejected

### test_early_stopping.py (6) — early-stop logic in greedy + Q loops
- stops after patience-no-improvement; patience=0 runs all; streak resets on improvement; early-stop sim recorded
- Q loop: stops on patience / patience=0 runs all

### test_env_termination.py (7) — `_classify_termination()`
- finish / crash / hard-crash / timeout / still-running; finish > crash priority; reason key always present

### test_game_adapter.py (26) — TMNF/TORCS/SC2/BeamNG adapter abstractions
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

### test_utils.py (13) — math/state-extraction utils
- vector magnitude: zero / unit / 3D / compute_speed alias
- yaw/pitch/roll identity = 0; 90° yaw correct
- state extraction: velocity / wheels / centerline-progress / 3-entry lookahead with centerline / zero lookahead without

### test_track.py (11) — centreline geometry helpers
- start / end / midpoint progress; nonzero lateral; on-centreline zero lateral; forward unit vector
- lookahead: returns two floats / straight-track zero heading change / finite lateral / clamps at end / opposite sign across centreline

## TMNF policies

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

### test_rl_client.py (22) — `RLClient` (game-thread bridge, action windowing)
- set_input_state called on tick / steer+accel+brake values / brake / full-left / clamped
- not called when finish-respawn pending; finish-respawn resets running flag
- shape+dtype; coast straight; decision_idx computed
- action window: first tick commits / later ticks don't / observation phase emits states / transit phase drains+puts
- pending action locked after decision tick; finish forces immediate commit; hard crash forces commit
- decision_idx clamped ≥1 / ≤ window-1 / legacy window=1 commits every tick
- window must be positive; decision_offset_pct bounds

## TORCS

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

### test_sc2_obs_spec.py (8) — SC2 obs spec
- minigame dim; ladder dim; ladder extends minigame; default = minigame; get_spec for minigame / ladder; minigame count; obs_names match dims

### test_sc2_actions.py (10) — discrete action grid
- shape / dtype / xy in unit square; centre = select_army; others = move_screen
- probe actions count / shape; warmup shape / select_army; function_ids table complete

### test_sc2_reward.py (13) — SC2 reward calc
- defaults; from_yaml; unknown raises; loads bundled config
- score delta; step penalty only; step penalty n_ticks scaling; win bonus; loss penalty; no-outcome no bonus; economy weight; idle penalty when idle / not when busy

### test_sc2_client.py (9) — PySC2 client wrapper
- minigame flat obs shape; score-delta threading; player_relative centroid; terminal outcome recorded
- ladder flat obs shape; visibility tracking; fogged ≠ visible; ladder terminal outcome; non-terminal = None

### test_sc2_env.py (15) — SC2 env wrapper
- minigame obs space; action space shape+bounds; ladder obs space; episode time-limit get/set
- reset returns obs+info; step 5-tuple; score-delta reward; done terminates; loss outcome
- close calls client.close; info keys; prev_score threaded; custom reward config

### test_sc2_genetic_policy.py (53) — `SC2LinearPolicy` + genetic trainer
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

### test_sc2_neural_dqn_policy.py (37) — masked DQN for SC2
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

## CLI / misc

### cli/test_game_flag.py (14) — `--game` CLI flag in `main.py`
- default tmnf; all valid choices accepted; invalid → SystemExit; help text mentions flag; `--track` accepted; main parser has all choices
- dispatch: tmnf / beamng / car_racing / torcs / sc2 → run_one; assetto → run_assetto; assetto missing dep → ValueError; adapter experiment_dir contains game name

### assetto_corsa/test_smoke.py (8) — Assetto Corsa smoke tests (against fake client)
- obs spec dimensions match base obs_dim; env reset obs shape; step 5-tuple finite reward; info reflects current step; env terminates on finish; vision features; reward calc finite; 5-episode training loop with linear policy

---

## Why 740 tests run in ~25 s

These tests look heavy because of the names ("training loop", "env reset", "DQN convergence") but operationally they're almost all pure-Python unit tests with zero external I/O:

1. **No game binaries are launched.** TMInterface, the SC2 binary, and TORCS are never started. `RLClient`, `SC2Client`, the SC2 env, and the TMNF env are all driven through fakes and `MagicMock` patches (e.g. `test_rl_client.py`, `test_env_termination.py`, `test_sc2_play.py`). The single "five-episode training loop" smoke test (`assetto_corsa/test_smoke.py`) runs against a stubbed client.
2. **All "policies" are pure numpy.** No PyTorch, no TensorFlow, no GPU. The DQN, REINFORCE, LSTM, CMA-ES, CNN, and SC2 multi-head policies are hand-rolled numpy with hidden sizes like `[8, 8]` or `hidden_size=4` in tests. Forward+backward passes are sub-millisecond.
3. **Tiny tensors.** Where convergence is asserted (`test_neural_dqn_policy.test_bandit_convergence`, `test_cmaes_policy.test_converges_toward_quadratic_maximum`, `test_reinforce_policy.test_gradient_direction`), the problem is a 2-arm bandit or a quadratic — a few hundred steps on tiny vectors.
4. **Whole files are config / dataclass tests.** `test_grid_search.py` (29), `test_reward.py` (44), `test_sc2_genetic_policy.py` (53), `test_torcs_obs_spec.py` (14), `test_analytics_task_metrics.py` (17), `test_game_adapter.py` (26) are mostly "from_yaml round-trip / shape / default-value / cartesian product" — microseconds each.
5. **No matplotlib rendering.** TORCS analytics tests use `Agg` (non-interactive) and dump to `tmp_path`; `test_analytics_no_matplotlib.py` explicitly checks the import path that *avoids* it.
6. **Filesystem work uses `tmp_path`** (RAM-backed `/tmp`), and the only network is `test_distributed.py` binding `localhost` for HTTP coordinator tests — which is why that's the one file with `time.sleep` and is still milliseconds because it talks to itself.
7. **Heavy collection work is amortised.** `pytest`'s ~half-second startup + 41 collection modules is a big share of the wall clock; once collected, 740 mostly-arithmetic asserts run in about 30 µs each.

Roughly: 740 tests × ~25 ms average = ~18 s of work + ~7 s of import/collection — fits the 25 s budget exactly because nothing in the suite waits on a game tick, a network packet, or a GPU.
