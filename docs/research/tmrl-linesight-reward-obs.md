# tmrl & Linesight: reward, observation, and step-timing design — a close read

> Tracking issue: [#485](https://github.com/espenhk/gamer-ai/issues/485).
> Scope: before or in parallel with [#489](https://github.com/espenhk/gamer-ai/issues/489)
> (TMNF long-horizon SAC run). This is a written comparison, not code — concrete
> deltas are filed as follow-up issues, listed at the end.

## Methodology & caveats

Unlike [`competing-projects.md`](competing-projects.md) (issue #329), which
surveys ~20 projects at README depth, this doc goes one level deeper on
exactly two: `trackmania-rl/tmrl` and Linesight
(`pb4git/trackmania_rl_public`, mirrored at `Linesight-RL/linesight`) — the
two projects issue #485 named. Facts below are read directly from source
files (config constants, quoted verbatim) and maintainer statements
(a GitHub Discussion), not blog summaries. Verified July 2026.

- Every constant quoted below is copy-pasted from the projects' own config
  files (linked in [Sources](#sources)); nothing is inferred or rounded.
- Where a project's own naming/units are ambiguous (e.g. whether a distance
  constant is metres), that's flagged rather than guessed.
- Neither project publishes a "why we chose 7 seconds" writeup — the closest
  either gets is the maintainer's Discussion comment on tmrl's LIDAR (quoted
  below). Design rationale beyond the numbers is this doc's own inference,
  labelled as such.
- Licenses: tmrl is MIT. Linesight/`trackmania_rl_public` ships no `LICENSE`
  file — read for ideas, do not copy code (same caveat as #329).

---

## Our baseline (what these findings get compared against)

- **Reward** — `games/tmnf/config/reward_config.yaml` / `games/tmnf/reward.py`:
  `progress_weight: 10000.0` (× `track_progress` delta, dominant term) +
  `centerline_weight: -0.083` (quadratic lateral-offset penalty, every tick)
  + `speed_weight: 0.042` + `step_penalty: -0.05` + `accel_bonus: 0.5` +
  `airborne_penalty: -0.83` + `lidar_wall_weight: -5.0` (quadratic nearest-wall
  penalty) + one-time `finish_bonus`/`finish_time_weight`.
- **Observation** — `games/tmnf/obs_spec.py`: 21 base floats (speed, lateral/
  vertical offset, yaw/pitch/roll, `track_progress`, steer, 4× wheel contact,
  3× angular velocity) + a 3-point lookahead (`LOOKAHEAD_STEPS = [10, 25, 50]`
  centerline-point indices, i.e. **20 m / 50 m / 100 m ahead** at the
  default 2.0 m point spacing set by
  `games/tmnf/tools/build_centerline.py --spacing`) + optional
  `n_lidar_rays` (default 8).
- **LIDAR** — `games/tmnf/lidar.py`: `n_rays` (default 16 at the sensor level,
  8 wired in by default via `n_lidar_rays`) cast from `0` to `π` radians
  (front half-plane, left-to-right) off an MSS screenshot processed through
  greyscale → threshold → Canny → dilate → blur, angle-dependent perspective
  correction applied per ray.
- **Step timing** — `games/tmnf/config/training_params.yaml`:
  `action_window_ticks: 1` (act every game tick, i.e. the finest possible
  grain) with an optional observe/commit split via `decision_offset_pct`
  (issue #65). `speed: 10.0`, documented as "TMInterface max 10×".

---

## tmrl (`trackmania-rl/tmrl`)

Source: `TmrlData/config/config.json` (default `ENV_CONFIG` for `FULL` and
`LIDAR` environments) plus a maintainer reply in
[Discussion #29](https://github.com/trackmania-rl/tmrl/discussions/29).

### Reward
Progress along a **pre-recorded human/AI demonstration trajectory**, split
into equally-spaced waypoints; "the reward is... the number of such points
that the car has passed since the previous time-step" (README). Supporting
constants from `config.json`:

| Constant | Value | Reading |
|---|---|---|
| `END_OF_TRACK` | `100.0` | One-time finish bonus |
| `CONSTANT_PENALTY` | `0.0` | No per-tick time cost by default |
| `CHECK_FORWARD` | `500` | Max waypoints searched ahead when matching car position to the trajectory |
| `CHECK_BACKWARD` | `10` | Max waypoints searched behind — tolerates brief backtracking without instantly failing progress-matching |
| `FAILURE_COUNTDOWN` | `10` | Steps of *no* progress tolerated before the episode is failed |
| `MIN_STEPS` | `70` | Episode must run at least this many steps before `FAILURE_COUNTDOWN` can end it |
| `MAX_STRAY` | `100.0` | Distance from the trajectory that triggers failure |

There is **no separate centerline/lateral-offset penalty term** — deviation
is punished only indirectly (you stop banking progress if you're not near
the trajectory, and `MAX_STRAY`/`FAILURE_COUNTDOWN` end the episode if that
persists). This is a single reward mechanism, not a weighted sum of several
shaping terms like ours.

### Observation / LIDAR
Two modes:
- **LIDAR mode**: **19 beams**, extracted from a front-camera screenshot with
  the car model hidden, "works only on plain road with black borders" — the
  maintainer's own words in Discussion #29: *"The way we compute the LIDAR in
  the TM LIDAR environment is very specific to the plain road in TrackMania...
  We just look for black pixels on the trajectory of each ray."* Frame history
  `IMG_HIST_LEN = 4` (stacked LIDAR frames feed an MLP; the maintainer notes
  it's set to `1` for RNN variants). Speed is appended.
- **Full mode**: raw screenshots (+ speed/gear/rpm) into a CNN, also
  frame-stacked. The maintainer, in the same discussion, frames LIDAR as a
  **simplification that "probably won't work" outside this exact plain-road
  case**, and says modern tmrl training has moved to **full unprocessed
  images with CNNs** as the primary approach — LIDAR is a training
  accelerant, not the direction of travel.

### Step timing / action-repeat
Not a fixed engine-tick multiplier — tmrl runs **real-time** via `rtgym`'s
elastic-timestep model:

| Constant | Value | Reading |
|---|---|---|
| `time_step_duration` | `0.05` s | **20 Hz** control rate |
| `start_obs_capture` | `0.04` s | Observation captured near the end of the window, not the start |
| `time_step_timeout_factor` | `1.0` | Elasticity: a slow step can borrow up to one full window before being flagged |
| `act_buf_len` | `2` | Two most-recent actions appended to the observation |
| `ep_max_length` | `1000` | Steps per episode |

`act_buf_len` exists because the environment is **not paused** while the
agent computes its action — the action taken now was decided one step ago,
so the "in-flight" action(s) must be part of the state for the process to be
Markov (this is the RTRL/RDMDP lineage already covered in #329's takeaway
#12). This doesn't apply to us: TMInterface is paused between our RL steps
(sped up, not real-time), so we don't need `act_buf_len`'s specific fix —
but see Linesight's `n_prev_actions_in_inputs` below, which is a *different*
motivation for the same-shaped feature.

---

## Linesight (`pb4git/trackmania_rl_public`, mirrored as `Linesight-RL/linesight`)

Source: `config_files/config.py`, `config_files/inputs_list.py`. This is the
project that reached human-level driving (~May 2023) and beat official
world records (May 2024) — the strongest evidence in this space.

### Reward
Two components dominate; four "engineered" bonuses ship **disabled by
default** on a training-step *schedule*:

| Constant | Value | Reading |
|---|---|---|
| `constant_reward_per_ms` | `-6/5000` (= `-0.0012`/ms, **-1.2/s**) | Small, constant time penalty |
| `reward_per_m_advanced_along_centerline` | `5/500` (= `0.01`/m) | Primary signal — linear in metres advanced, **not** a fraction-of-track like our `track_progress` |
| `engineered_speedslide_reward_schedule` | `[(0, 0)]` | Off by default; schedule format `[(step, value), ...]` — a ramp-in mechanism, not a flat constant |
| `engineered_neoslide_reward_schedule` | `[(0, 0)]` | Off by default |
| `engineered_kamikaze_reward_schedule` | `[(0, 0)]` | Off by default |
| `engineered_close_to_vcp_reward_schedule` | `[(0, 0)]` | Off by default |

Like tmrl, **there is no explicit centerline-deviation penalty term.**
Progress-along-line plus a time cost is the whole base reward; the
`_schedule` mechanism (ramp a shaping term in/out over training rather than
holding it constant for the whole run) is itself a technique we don't use
anywhere in `reward.py` — every weight in our `RewardConfig` is a flat
scalar for the run's duration.

Episodes are **not full laps**. `temporal_mini_race_duration_ms = 7000` —
training runs on repeated **7-second "mini-races"**, matching the public
description ("distance travelled along a reference trajectory over roughly
the next 7 seconds"). This is a fundamentally different sampling scheme from
our single continuous `in_game_episode_s` run per episode: each mini-race
plausibly starts from a different point along the reference line (the repo
supports this via its checkpoint/zone-center structure), giving far more
varied initial states per wall-clock hour than always starting from the
same line.

### Observation
- **CNN input**: `W_downsized = 160`, `H_downsized = 120` — greyscale, matches
  the public description exactly.
- **Lookahead geometry, fed alongside the CNN** (not pixels-only!):
  `n_zone_centers_in_inputs = 40`, `one_every_n_zone_centers_in_inputs = 20`,
  `distance_between_checkpoints = 0.5` (base zone spacing). Read together:
  the base zone-center array is spaced `0.5` (units, presumed game metres —
  unconfirmed) apart; the network sees every 20th one, i.e. **input points
  ~10 units apart**, **40 of them** → a **~400-unit lookahead window**, far
  longer and denser than our 3-point, 100 m-max lookahead
  (`LOOKAHEAD_STEPS = [10, 25, 50]`).
- **Action history**: `n_prev_actions_in_inputs = 5` — the 5 most recent
  discrete actions are appended to the input, structurally the same idea as
  tmrl's `act_buf_len` but motivated differently here (Linesight is
  sped-up/pausable like us, so this isn't solving a real-time-delay problem —
  it's giving the value network short-term action-repeat/momentum context).

**Net read:** the SOTA TMNF agent is CNN-primary but still hand-feeds
engineered lookahead-geometry points — the same idea as our
`lookahead_{10,25,50}_{lat,yaw}` features and GT Sophy's "course-ahead
points" (#329) — just far more of them, further out. Pure pixels were not
enough on their own.

### Action space
**Discrete, 12 actions** — every combination of {accelerate, brake, nothing}
× {left, right, straight}, e.g. `forward`, `forward_left`, `brake_right`,
`brake_and_accelerate` (simultaneous brake+accel, a genuine TMNF technique).
This is a structural difference from both tmrl (continuous, `vgamepad`) and
our own `Box([-1,0,0],[1,1,1])` — Linesight's value-based IQN algorithm
needs a discrete action set by construction, so the action space follows
from the algorithm choice rather than the other way around.

### Step timing / action-repeat
| Constant | Value | Reading |
|---|---|---|
| `ms_per_tm_engine_step` | `10` ms | One TMNF engine tick |
| `tm_engine_step_per_action` | `5` | Engine ticks held per RL action |
| `ms_per_action` | `= 10 × 5 = 50` ms | **20 Hz** control rate |
| `running_speed` | `80` | TMInterface game-speed multiplier during training |

**`ms_per_action = 50` ms matches tmrl's `time_step_duration = 0.05` s
exactly** — two independent TMNF/TM2020 projects, one real-time and one
sped-up, converged on the same 20 Hz control rate. Our
`action_window_ticks: 1` acts on **every single 10 ms engine tick** — 5×
finer-grained than either external project (issue #65 built the
`action_window_ticks` knob precisely to allow coarser windows like this, but
the shipped default is still `1`).

`running_speed = 80` is also notable against our own documented ceiling:
CLAUDE.md's `training_params.yaml` table calls `speed: 10.0` "TMInterface
max 10×". Nothing found in TMInterface's own docs during this pass states a
hard multiplier cap — the practical constraint is desync/dropped-input risk
at high speeds, not an enforced ceiling — and Linesight training at 80×
directly contradicts "max 10×" as a hard limit (though it doesn't tell us
whether 80× is safe for *our* client/threading model, which differs from
Linesight's). Flagged as a doc-accuracy question, not a proven "raise
`speed`" recommendation — see follow-ups.

### Algorithm (context for the reward/obs choices, not itself the issue's scope)
Distributional value-based: **IQN** (`iqn_embedding_dimension = 64`,
`iqn_n = 8`, `iqn_k = 32`, `iqn_kappa = 5e-3`), **3-step returns**
(`n_steps = 3`), and a **discount schedule that reaches 1.0**
(`gamma_schedule = [(0, 0.999), (1_500_000, 0.999), (2_500_000, 1)]`) — only
well-defined because each episode is a bounded 7-second mini-race, so an
undiscounted sum stays finite. Exploration is ε-greedy with its own schedule
(`epsilon_schedule`, decaying 1.0 → 0.03) plus an alternative Boltzmann
schedule. This confirms #329's IQN/distributional-value finding at the
config level rather than changing it.

---

## Side-by-side: our config keys vs. the field

| Design axis | `gamer-ai` (TMNF) | tmrl | Linesight |
|---|---|---|---|
| Reward shape | Weighted sum of 7+ terms (progress, centerline, speed, step, accel, airborne, lidar-wall) | Single term: waypoints passed since last step, + finish bonus | Two terms: m advanced (linear) + constant time cost; optional scheduled bonuses |
| Explicit centerline/lateral penalty? | **Yes** — `centerline_weight`, quadratic, every tick | **No** | **No** |
| Episode structure | One continuous run of `in_game_episode_s` (default 30 s) from track start | Up to `ep_max_length=1000` steps (50s @ 20Hz) along the full trajectory | **7-second mini-races**, resettable to arbitrary points along the reference line |
| Crash/failure handling | Instant terminate at `|lateral_offset| > crash_threshold_m` | Grace period: `FAILURE_COUNTDOWN=10` steps of no progress, `MAX_STRAY=100` | Bounded episode length itself limits damage; no separate crash rule found |
| Lookahead geometry in obs | 3 points, out to 100 m | n/a (LIDAR/pixels only) | 40 points, ~10-unit spacing, ~400-unit horizon |
| LIDAR rays | 8 (default), 0–π half-plane | 19, front-camera black-pixel raycast | n/a (full CNN instead) |
| Primary obs modality | Telemetry vector (+ optional LIDAR) | LIDAR-MLP *or* pixel-CNN | Pixel-CNN (+ lookahead points) |
| Action history in obs | No | 2 previous actions (`act_buf_len`) | 5 previous actions (`n_prev_actions_in_inputs`) |
| Control rate | Every engine tick (`action_window_ticks: 1`) | 20 Hz (`time_step_duration=0.05s`) | 20 Hz (`ms_per_action=50ms`, `tm_engine_step_per_action=5`) |
| Training speed multiplier | 10× (documented as "max") | 1× (real-time by design) | **80×** |
| Action space | Continuous `Box` | Continuous (`vgamepad`) | Discrete, 12 actions |

---

## Concrete deltas worth trying

Each maps to a specific file/key and is filed as a separate follow-up issue
so it can be scoped and prioritized independently of this doc.

1. **Ablate `centerline_weight` (test progress-only shaping).** Both tmrl and
   Linesight — the two projects with published TMNF results — run with *no*
   explicit lateral-offset penalty. Our `centerline_weight: -0.083` may be
   redundant with (or fighting) `progress_weight`: cutting a corner
   off-centerline can be the *fastest* line, and a quadratic centerline
   penalty discourages exactly that. Worth an A/B run with
   `centerline_weight: 0.0` before finalizing #489's config. →
   [#491](https://github.com/espenhk/gamer-ai/issues/491)
2. **Grace-period crash termination instead of an instant threshold.** tmrl
   tolerates `FAILURE_COUNTDOWN=10` steps of no progress and `MAX_STRAY=100`
   before failing, rather than ending the episode the instant
   `|lateral_offset| > crash_threshold_m`. A "no progress for N steps" rule
   (independent of, or alongside, the existing hard offset threshold) would
   be more forgiving of brief off-line excursions that still recover. →
   [#492](https://github.com/espenhk/gamer-ai/issues/492)
3. **Extend the lookahead observation (more points, longer horizon,
   configurable).** Linesight feeds 40 lookahead points spanning roughly
   400 m (subsampled from a 0.5-unit-spaced zone-center array) alongside its
   CNN; GT Sophy's ablation (#329) independently found lookahead points beat
   wall-LIDAR. Our `N_LOOKAHEAD=3` / `LOOKAHEAD_STEPS=[10,25,50]` (indices,
   ~20–100 m at 2 m point spacing) is far shorter and sparser. Worth
   prototyping a longer/denser lookahead as a `training_params.yaml`-tunable
   knob rather than the current hardcoded 3-point list in `obs_spec.py`. →
   [#493](https://github.com/espenhk/gamer-ai/issues/493)
4. **Try `action_window_ticks: 5` (20 Hz) as the SAC-run default for #489.**
   tmrl and Linesight independently converged on the same 20 Hz control rate
   (`time_step_duration=0.05s` / `ms_per_action=50ms`); our default
   (`action_window_ticks: 1`) acts on every 10 ms tick, 5× finer. Finer
   control isn't free — it multiplies RL-steps-per-episode and therefore
   replay-buffer fill rate and gradient-step count for a fixed episode-time
   budget. Set `action_window_ticks: 5` for #489's SAC run instead of the
   default. → [#494](https://github.com/espenhk/gamer-ai/issues/494)

**Not filed as an issue (needs more evidence first):** `running_speed=80` in
Linesight directly contradicts CLAUDE.md's "TMInterface max 10×" framing as
a hard technical ceiling, but nothing found here confirms 80× is safe under
*our* client/threading model (`clients/rl_client.py`'s action/state-queue
handshake, not Linesight's). Before changing our documented default or
`training_params.yaml`'s `speed`, someone should test whether our client
desyncs above 10× — this is a narrower "verify, don't yet act on" item.

---

## Sources

- tmrl — https://github.com/trackmania-rl/tmrl
- tmrl config (`config.json` defaults, read via `TmrlData/config/config.json` per the README's config-file docs) and README's reward description
- tmrl LIDAR discussion (maintainer comment) — https://github.com/trackmania-rl/tmrl/discussions/29
- Linesight (repo) — https://github.com/Linesight-RL/linesight
- Linesight source — `pb4git/trackmania_rl_public` — https://github.com/pb4git/trackmania_rl_public
- Linesight config — https://github.com/pb4git/trackmania_rl_public/blob/main/config_files/config.py
- Linesight action set — https://github.com/pb4git/trackmania_rl_public/blob/main/config_files/inputs_list.py
- Linesight docs site (landing page reachable; deep pages returned HTTP 403
  during this pass and could not be read) — https://linesight-rl.github.io/linesight/build/html/
- The History of Machine Learning in Trackmania (background, cited in #329) — https://hallofdreams.org/posts/trackmania-1/
- TMInterface — https://donadigo.com/tminterface/ (speed-limit question: no
  documented hard cap found; community consensus is that very high speeds
  risk dropped/desynced inputs rather than being blocked outright)

See also: [`competing-projects.md`](competing-projects.md) (#329) for the
broader field survey these two projects sit within.
