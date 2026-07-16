# Plan: TMNF SAC-run prep (issues #489, #491, #492, #493, #494)

Branch: `claude/issues-489-491-494-gmjqaq`

All five issues trace back to #485's close read of `tmrl` and Linesight and
exist to get #489 (the TMNF long-horizon SAC run) ready to launch on solid
footing rather than on guessed defaults. #492/#493 are code changes that
land regardless of when #489 actually runs; #494 is a config decision that
belongs in #489's run config; #491 is a comparison that should inform #489's
config before it's finalized. Solving them together avoids two churn cycles
through `games/tmnf/env.py` / `obs_spec.py` and lets #489's config template
carry #494 (and, once run, #491's result) from the start.

## What this change can and can't do

This repo's automation has no Windows box, no TMNF binary, and no Azure
credentials. That means:

- **#492 and #493 are fully implemented, tested, and documented** — pure
  code changes with unit-test coverage, no game process required.
- **#494 is fully implemented as config** — folded into a new experiment
  template (`games/tmnf/config/gs_sac_long_horizon.yaml`) rather than the
  shared master config, since it's a decision specific to #489's run, not a
  change to every TMNF experiment's default.
- **#489 and #491 cannot be executed here.** #489 needs real Azure spend and
  a multi-day unattended run against a live TMNF process; #491 needs two
  real training sessions against a live TMNF process. Both are prepped —
  ready-to-run config, and for #489 a crash-restart supervisor script — but
  *not run*. Nothing in this change provisions Azure resources or launches
  training. See each section below for exactly what "prepped" means and
  what a human still needs to do.

---

## 1. Issue #492 — grace-period ("no progress for N steps") crash termination

### Design

`RewardConfig` (`games/tmnf/reward.py`) gains two opt-in fields:

- `no_progress_patience_ticks: int = 0` — end the episode once
  `track_progress` hasn't advanced for this many consecutive **game ticks**
  (not RL steps — this keeps the patience window's real-time meaning
  independent of `action_window_ticks`, per #494's control-rate change).
  `0` disables the check, matching pre-#492 behaviour bit-for-bit.
- `no_progress_min_ticks: int = 0` — the check cannot fire until this many
  game ticks have elapsed in the episode, mirroring tmrl's `MIN_STEPS`.

Both are termination parameters living in `RewardConfig`, not reward
weights — the same precedent `crash_threshold_m` already sets (the env
reads it directly for termination, not for the reward sum).

`TMNFEnv.step()` (`games/tmnf/env.py`) tracks `self._ticks_since_progress`,
reset on any positive `track_progress` delta and on episode reset,
incremented by the step's tick count otherwise. The whole block is guarded
by `no_progress_patience_ticks > 0` so it's a complete no-op — including
never touching `prev_state.track_progress` — when disabled, preserving
existing behaviour and existing test mocks exactly. `no_progress` is OR'd
into `terminated` alongside `finished`/`crashed`, with its own
`termination_reason = "no_progress"`; `crash` still takes priority when both
conditions are true on the same step (matches the existing
finish-over-crash priority pattern).

### Files changed

| File | Change |
|---|---|
| `games/tmnf/reward.py` | Add `no_progress_patience_ticks` / `no_progress_min_ticks` fields + docstring. |
| `games/tmnf/env.py` | Track `_ticks_since_progress`; opt-in no-progress check in `step()`; new `termination_reason = "no_progress"`; docstring updates. |
| `games/tmnf/config/reward_config.yaml` | Document both keys, disabled (`0`) by default. |
| `games/tmnf/README.md` | New Rewards-table rows. |
| `CLAUDE.md` | New `reward_config.yaml` table rows + Termination section update. |
| `tests/test_env_termination.py` | New `TestNoProgressTermination` class: disabled-by-default, fires after patience, gated by min_ticks, resets on progress, crash-takes-priority. Also fixed a pre-existing harness gap (`_prev_obs`, `_ep_lateral_sum`/`_ep_lateral_count`/`_ep_reward_components` were never set on the bypassed-`__init__` test `TMNFEnv`, so every test in the file raised `AttributeError` the moment `compute_with_components`/lateral-offset accumulation were added in an earlier change — the exclusion of this file from CI, https://github.com/espenhk/gamer-ai actions `test-tmnf` job, meant nobody noticed). Verified locally by stubbing `win32gui`/`win32`/`pywintypes` so `games.tmnf.env` imports on Linux; all 12 tests (7 pre-existing + 5 new) pass. |
| `tests/README.md` | New coverage bullets for the no-progress path. |

### Validation

Ran locally with `win32gui`/`win32`/`pywintypes` stubbed into `sys.modules`
(the only way to exercise `games.tmnf.env` off Windows — see CI's own
comment in `.github/workflows/tests.yml` explaining why `test_env_termination.py`
is excluded there). All 12 tests pass. `tests/test_reward.py` (YAML
round-trip, unaffected by termination logic) also passes unmodified.

---

## 2. Issue #493 — configurable lookahead observation

### Design

`games/tmnf/obs_spec.py` keeps `N_LOOKAHEAD` / `LOOKAHEAD_STEPS` as the
legacy-default constants (`3` / `[10, 25, 50]`) but adds:

- `build_lookahead_steps(n_lookahead_points=None, lookahead_step_spacing=None)`
  — both `None` reproduces the legacy list exactly; either set switches to
  an evenly-spaced schedule (`spacing, 2*spacing, ..., n*spacing`),
  defaulting the other argument to its legacy-equivalent value.
- `build_tmnf_obs_spec_from_steps(lookahead_steps)` / `build_tmnf_obs_spec(...)`
  — build an `ObsSpec` (no LIDAR) from a resolved step list or from the
  raw config knobs respectively. `TMNF_OBS_SPEC` (the module-level constant
  everything already imports) is now `build_tmnf_obs_spec()` — identical
  output, just derived instead of hand-written, so nothing importing it
  needs to change.

The resolved step list threads through the whole construction path so
obs_dim, the `ObsSpec` used for policy weight naming, and the actual
per-step observation values all agree:

```
adapter.py (reads training_params.n_lookahead_points / lookahead_step_spacing)
  → build_lookahead_steps() → lookahead_steps: list[int]
  → build_tmnf_obs_spec_from_steps(lookahead_steps)   [for GameSpec.obs_spec]
  → make_env(..., lookahead_steps=lookahead_steps)
      → TMNFEnv(..., lookahead_steps=...)              [obs_dim via the same builder]
          → RLClient(..., lookahead_steps=...)
              → StateData(..., lookahead_steps=...)     [actual project_ahead() calls]
```

`TMNFEnv._build_obs()` needed **no change** — its lookahead-flattening list
comprehension (`[v for lat, yaw in d.lookahead for v in (lat, yaw)]`) was
already length-agnostic.

### A note on scope: which policies actually pick this up

Investigated whether TMNF's local policy subclasses
(`games/tmnf/policies.py`'s `WeightedLinearPolicy`, `GeneticPolicy`,
`NeuralNetPolicy`, `QTablePolicy`, `EpsilonGreedyPolicy`, `UCBQPolicy`) also
needed threading. They don't: none of them carry `@register_policy`, so
`POLICY_REGISTRY["hill_climbing"|"genetic"|"neural_net"|"epsilon_greedy"|"ucb_q"]`
resolve to the game-agnostic `framework/policies.py` classes instead, which
already construct from the `obs_spec` passed through `GameSpec` — i.e. from
`adapter.py`'s `build_game_spec()`, which this change updates. The TMNF-local
classes are legacy/unregistered, exercised only by their own direct unit
tests and by `bc_adapter.py` — confirmed by grepping every call site. The
policies that *are* registered under `games/tmnf/policies.py`
(`cmaes`, `reinforce`, `lstm`, `neural_dqn`) all implement
`_construct_or_resume(*, obs_spec, ...)` using the passed-in `obs_spec`
directly, so they're already correctly wired. Net effect: every policy
reachable through the normal `--game tmnf` training path respects
`n_lookahead_points` / `lookahead_step_spacing`; BC pre-training
(`main.py --bc`) and the standalone `games/tmnf/policies.py` classes remain
on the legacy 3-point schedule — not a regression (that was already the only
schedule available to them), just not extended in this change. Flagged here
rather than silently left as a gap.

### Files changed

| File | Change |
|---|---|
| `games/tmnf/obs_spec.py` | `build_lookahead_steps()`, `build_tmnf_obs_spec_from_steps()`, `build_tmnf_obs_spec()`; `TMNF_OBS_SPEC` now derived. |
| `games/tmnf/state.py` | `StateData.__init__` takes `lookahead_steps` (default `None` → legacy `LOOKAHEAD_STEPS`). |
| `games/tmnf/clients/rl_client.py` | `RLClient.__init__` takes and stores `lookahead_steps`; passed to both `StateData(...)` construction sites it owns. |
| `games/tmnf/env.py` | `TMNFEnv.__init__` / `make_env()` take `lookahead_steps`; obs_dim computed via `build_tmnf_obs_spec_from_steps()` instead of the fixed `BASE_OBS_DIM` constant. |
| `games/tmnf/adapter.py` | `build_game_spec()` resolves `n_lookahead_points`/`lookahead_step_spacing` from `training_params`, builds `obs_spec` and `lookahead_steps` once, passes both through. |
| `games/tmnf/config/training_params.yaml` | Document the two new (commented-out, opt-in) keys. |
| `games/tmnf/README.md`, `CLAUDE.md` | Observation-table updates (also fixed pre-existing drift: `CLAUDE.md`'s obs table was missing the lookahead block entirely). |
| `tests/test_tmnf_obs_spec.py` (new) | `build_lookahead_steps()` / `build_tmnf_obs_spec()` unit tests; `StateData` lookahead-wiring tests via a mocked centerline. |
| `tests/test_game_adapter.py` | Two new TMNF `build_game_spec()` tests: default obs_spec.dim (21), custom lookahead+LIDAR resizing. |
| `tests/README.md` | New file/section entries. |

### Validation

`build_game_spec()` doesn't import anything Windows-only (the `_make_env`
closure defers `games.tmnf.env` import until actually called), so its tests
run directly in this sandbox. Full relevant suite green: 410 tests across
`test_reward.py`, `test_rl_client.py`, `test_track.py`,
`test_build_centerline.py`, `test_tmnf_obs_spec.py`, `test_game_adapter.py`,
and every TMNF policy test file. `test_env_termination.py` re-verified
green under the win32 stub after these changes too (obs_dim/lookahead
changes touch the same `TMNFEnv.__init__` the termination tests construct).

---

## 3. Issue #494 — `action_window_ticks: 5` for #489's run

Pure config decision, folded directly into `games/tmnf/config/gs_sac_long_horizon.yaml`
(built for #489 — see below) rather than the shared master
`training_params.yaml`, since this is specific to the SAC run, not a
framework-wide default change. `decision_offset_pct` is left at its
default per the issue's proposed solution.

---

## 4. Issue #489 — SAC run config, Azure VM provisioning, unattended execution

Three parts, each handled differently given what can and can't run here:

1. **Config** — `games/tmnf/config/gs_sac_long_horizon.yaml` (new): a
   single-combo grid-search template (no swept axes — `grid_search.py` with
   one combo per axis just creates one experiment, which is the established
   pattern for template-driven experiment setup in this repo). Sets
   `n_lidar_rays: 16`, `policy_type: sac` with a starting `policy_params`
   budget (`total_timesteps: 3_000_000`, `buffer_size: 750_000`,
   `batch_size: 256`, `learning_starts: 10_000`, `train_freq: 1`,
   `tau: 0.005`, `gamma: 0.99`), `checkpoint_freq: 20_000` (a multi-day-run
   cadence — ~150 checkpoints over the 3M-step budget), and #494's
   `action_window_ticks: 5`. `centerline_weight` is left at the current
   default pending #491 (see the file's header comment for the exact
   pointer to update once that ablation runs). Verified the file parses and
   expands to exactly one combo named `gs_sac_long_horizon` via
   `grid_search.py`'s own `_load_grid_config()` / `_expand_grid()` /
   `_make_experiment_name()` — without ever calling `train_rl()` or touching
   a real TMInterface connection.
2. **Infra** — `infrastructure/environment/run_supervised.ps1` (new): a
   crash-restart supervisor that loops
   `main.py <experiment> --game tmnf --no-interrupt`, restarting with a
   backoff delay on any non-zero exit and relying on #488's checkpoint/
   resume mechanism to pick up where it left off rather than retraining
   from scratch. Documented in a new "Single-VM unattended long-horizon
   run" section of `infrastructure/README.md`: `worker_vm_count = 1` (one
   experiment, not a grid search — deliberately bypassing the distributed
   coordinator/worker protocol), a one-time RDP session to materialize the
   experiment directory via `grid_search.py`, then handing off to the
   supervisor via `worker_command`.
3. **Execution** — explicitly not done. No Terraform command was run, no
   VM was created, no training was started. Renting real Azure VM hours is
   a human, budget-owning decision per the issue's own text ("needs a human
   decision on cloud spend rather than something to do unattended"), and a
   multi-day run against a live TMNF process can't be exercised in this
   environment regardless. #489 stays open after this change — what's
   landed is everything that *doesn't* require spending money or a live
   game process; a human still needs to run `terraform apply` and RDP in to
   kick off training and its supervisor.

### Files changed

| File | Change |
|---|---|
| `games/tmnf/config/gs_sac_long_horizon.yaml` (new) | SAC run config template. |
| `infrastructure/environment/run_supervised.ps1` (new) | Crash-restart supervisor. |
| `infrastructure/README.md` | New "Single-VM unattended long-horizon run" section. |

---

## 5. Issue #491 — centerline_weight ablation

`games/tmnf/config/gs_ablation_centerline.yaml` (new): a two-combo
grid-search config sweeping `reward_params.centerline_weight` over
`[-0.083, 0.0]` with everything else fixed (same `policy_type: genetic`,
`n_sims`, `n_lidar_rays` as this game's other baseline ablation template,
`gs_genetic_v1.yaml`, so results are comparable). Verified it expands to
exactly two combos with clean experiment names
(`gs_ablation_centerline__cwn0.083`, `gs_ablation_centerline__cw0`) via the
same dry validation as #489's config.

**Not run**, for the same reason as #489: no TMNF-capable machine in this
environment. The file's header comment makes this explicit and points back
to this doc and to #489's config note so the deferred status isn't lost.
Whoever runs it should compare the two resulting experiments' analytics (or
#481's canonical score once it lands) and, per the issue's validation
section, either update `games/tmnf/config/reward_config.yaml`'s default (if
progress-only matches or beats the current shaping) or record the negative
result in `games/tmnf/README.md`'s Rewards section so it isn't
re-litigated blind later — either way, update
`games/tmnf/config/gs_sac_long_horizon.yaml`'s `centerline_weight` to match
before #489's real run starts.

### Files changed

| File | Change |
|---|---|
| `games/tmnf/config/gs_ablation_centerline.yaml` (new) | A/B ablation config. |

---

## 6. What's still open after this change

- **#489**: config and infra are ready; a human still needs to
  `terraform apply` the single-VM environment, RDP in, and kick off
  `grid_search.py` → the supervisor loop. Requires #481 and #482 per the
  issue's stated prerequisites (both referenced, not re-litigated here).
- **#491**: ablation config is ready; needs to actually run on a
  TMNF-capable machine, and its result needs to feed back into
  `gs_sac_long_horizon.yaml` before #489's real run.
- **BC lookahead**: `main.py --bc`'s TMNF obs_spec (`_build_bc_obs_spec` in
  `main.py`) and `games/tmnf/policies.py`'s local classes stay on the
  legacy 3-point lookahead — see the #493 section above. Not a regression,
  but a natural follow-up if BC pre-training should also use a custom
  lookahead schedule.
