# Competing & open-source RL-for-games projects — a survey

> Tracking issue: [#329](https://github.com/espenhk/gamer-ai/issues/329).
> This is a written survey, not code.

`gamer-ai` was built largely in isolation. Before investing further — new
policies ([#327](https://github.com/espenhk/gamer-ai/issues/327),
[#328](https://github.com/espenhk/gamer-ai/issues/328)), new games, reward
design — this document surveys existing open-source RL-for-games projects to
learn what observation / action / reward designs and algorithms others
converged on, what we could borrow, and what to benchmark against. It
prioritises our two flagship games, **Trackmania Nations Forever (TMNF)** and
**StarCraft II (SC2)**, then takes a lighter pass over other-game and general
game-RL libraries.

## Methodology & caveats

- **Verified as of May 2026.** Every project below was opened directly (repo
  README, docs site, LICENSE file, or the canonical paper) and only confirmed
  facts are stated. Star counts / release dates are point-in-time.
- **Verify, label, or drop.** Claims I could not confirm from a primary source
  are marked `(unverified)`; projects I could not confirm exist at all are
  omitted. Where two secondary sources disagreed (e.g. Linesight's algorithm),
  the repo/README wins.
- **Licenses are noted before any "borrow" suggestion.** Several hobby repos
  ship **no license file**, which legally means all rights reserved — we can
  read them for ideas but must not copy code. This is called out per project.
- Borrowing means *ideas*, not copied code. Respect every license.

---

## How `gamer-ai` works today (the baseline we compare against)

A compact summary of our own design, so the comparisons have an anchor. See
[`games/tmnf/README.md`](../../games/tmnf/README.md),
[`games/sc2/README.md`](../../games/sc2/README.md), and the framework protocol
docs in [`docs/framework/`](../framework/README.md) for detail.

**TMNF (flagship racing).**
- *Observation:* a **telemetry feature vector** (15 floats: speed, lateral /
  vertical offset from centreline, yaw error, pitch, roll, track progress,
  steer value, 4 wheel-contact flags, 3 angular velocities) plus an **optional
  virtual-LIDAR** block (`n_lidar_rays`) raycast from an MSS screen-grab +
  OpenCV edge image. Not raw pixels into a CNN.
- *Action:* continuous `Box` — steer `[-1,1]`, accel `[0,1]`, brake `[0,1]`;
  discrete policies use a 25-action abstraction.
- *Reward:* progress-dominant (`progress_weight` 10000 × centreline-progress
  delta) plus centreline-offset penalty, speed bonus, step penalty, finish
  bonus / finish-time term, LIDAR wall-proximity penalty, airborne penalty.
- *Infra:* binds to the live TMNF process via **TMInterface**, **sped up to
  ≤10×** real time; Windows-only; single process.

**SC2 (flagship RTS).**
- *Observation:* PySC2 feature-layer summaries collapsed into fixed vectors —
  presets of **15 (minigame) / 46 (ladder) / 103 (rich)** dims; an optional
  CNN path consumes spatial feature layers directly (`sc2_cnn`).
- *Action:* continuous `Box([fn_idx, x, y, queue])`; tabular policies use a
  discrete `[no_op, select_army, Move_screen × N×N]` grid. Optional `max_apm`
  token-bucket throttle in game-time.
- *Reward:* PySC2 score delta + win/loss + economy delta, plus a large set of
  opt-in micro shaping terms (attack/move/idle bonuses, ally-fire penalties…).
- *Infra:* **headless PySC2** on Linux; intra-run parallel evaluation across
  several SC2 binaries (`n_workers`).

**Policy zoo:** mutate-and-keep linear / MLP, tabular Q, UCT, genetic, CMA-ES,
vanilla DQN, REINFORCE, LSTM-ES, SC2 CNN-ES. **All numpy / evolutionary /
simple-gradient — there is no PyTorch deep-RL policy (no PPO/SAC/TD3) wired
into the registry yet.** (An orphaned Stable-Baselines3 PPO script exists at
`rl/train.py` but is not registered — see #328.)

The single biggest structural difference from the field below: **we lean
evolutionary + sped-up/headless**, whereas the strongest comparable racing
projects lean **gradient deep-RL (SAC / value-based) and frequently real-time.**

---

## Master comparison table

Racing (TMNF unless noted), then SC2, then general libraries. "Obs" =
observation design; "Act" = action space. URLs are in [Sources](#sources).

| Project | Game(s) | Algorithm(s) | Obs | Act | Reward | License | Activity |
|---|---|---|---|---|---|---|---|
| **tmrl** | TM2020 (TMNF legacy) | SAC, REDQ-SAC | 19-beam LIDAR *or* full screenshots (CNN); +speed/gear/rpm; frame-stack | continuous gas/brake/steer (vgamepad) | progress along recorded trajectory | MIT | active (v0.7.1, May 2025) |
| **Linesight** | TMNF | value-based, discrete-action (DQN-family) | CNN on 160×120 greyscale frames | discrete | distance along reference line over ~7 s | none found `(unverified)` | active (v3.0.0, Jun 2024) |
| **MOSEAC** (Wang & Beltrame) | Trackmania (on tmrl) | SAC + **elastic time steps** (off-policy AC) | 4× stacked 64×64 RGB → CNN (128-d) + speed/gear/RPM/step/prev-actions (143-d) | continuous gas/brake/steer **+ control rate 5–30 Hz** | tmrl-style path-progress × time term | MIT (code) | paper Aug 2024; code low |
| **GT Sophy** (Sony AI) | Gran Turismo Sport | **QR-SAC** (distributional SAC) | state vector: kinematics + per-tyre + 3×60 course-ahead points (~6 s) + opponent points | continuous throttle/brake + steer (10 Hz) | progress (off-course-masked) + off-course/wall/tyre/collision + passing + unsporting penalties | not public (paper) | *Nature* 2022 |
| **AndrejGobeX/TrackMania_AI** | TM | SAC, PPO, TD3; (old branch) NEAT + supervised | 13 wall-distances + velocity + wall-contact + 2 prev actions; frame-stack | continuous steer + throttle/brake | adapted Gran-Turismo-Sport reward (Fuchs et al.) | GPL-3.0 | active |
| **LouisDeOliveira/TMAI** | TMNF | targets DQN/DDPG/PPO | MSS+OpenCV edge → raycast LIDAR + TMInterface telemetry | discrete keys + continuous gamepad | telemetry-based | none found `(unverified)` | low |
| **Anca-Mt/TrackmaniaRL-AI** | TM (on tmrl) | DDPG, PPO, SAC (single+dual critic) | pixel-CNN *and* LIDAR variants | continuous (via tmrl) | custom per-track | none found `(unverified)` | coursework, inactive |
| **Pheoxis/AITrackmania** | TM2020 | SAC, REDQ (it is a **tmrl fork**) | as tmrl | as tmrl | as tmrl | MIT | fork |
| **TheoBoyer/TMForge** | TM2020 | DQN (reference) | CNN on frames; Gym-like API | discrete | not specified | none found `(unverified)` | beta, low |
| **Urela/DriveForever** | TMNF | not specified | 4× stacked 84×84 greyscale (CV, no API) | gym sample | car speed | none found `(unverified)` | toy, low |
| **ShubhamGajjar/TrackMania-RL** | TMNF | **IQN** (distributional) | dxcam frames + telemetry + virtual checkpoints (~10 m) | (n/s) | checkpoint progress − collisions/off-track | MIT | WIP (v1.4, Sep 2024) |
| **terafear/trackmania_rl_public** | TMNF | (n/s; earlier public "Linesight") | (n/s) | discrete keyboard | (n/s) | none found `(unverified)` | superseded by Linesight-RL |
| **TMInterface** (donadigo) | TMNF | n/a — *tooling* (TAS / programmatic control) | n/a | n/a | n/a | closed freeware; public repo unlicensed | maintained |
| **AlphaStar** (DeepMind) | SC2 (full game) | supervised init + multi-agent RL (PFSP league) | raw entity/unit list + spatial + scalars | autoregressive function+args, APM-limited | win/loss + pseudo-rewards | Apache-2.0 (offline-RL code only) | archived |
| **OpenAI Five** (OpenAI) | Dota 2 (5v5) | **PPO** + 4096-LSTM, **pure self-play** | ~16k curated API features/hero (not pixels) | factored discrete (~8k–80k/step) | hand-shaped (gold/XP/kills/towers) + team spirit | not public (paper) | beat world champs 2019 |
| **PySC2 / SC2LE** (DeepMind) | SC2 | env + scripted baselines | feature layers | function-id + args | game score / minigame-specific | Apache-2.0 | active (v4.0, 2022) |
| **reaver** (inoryy) | SC2 minigames | A2C, PPO | feature layers | PySC2 action spec | minigame score | MIT | unmaintained (2018) |
| **chris-chris/pysc2-examples** | SC2 minigames | DQN (+PER/dueling), A2C/A3C | feature layers | PySC2 action spec | minigame score | Apache-2.0 | dated |
| **Stable-Baselines3** | any (Gymnasium) | PPO, A2C, DQN, SAC, TD3, DDPG (+contrib QR-DQN/TQC/TRPO/ARS/CrossQ) | env-defined | env-defined | env-defined | MIT | active (v2.8.0, Apr 2026) |
| **CleanRL** | any (Gymnasium) | single-file PPO, DQN, C51, SAC, TD3, DDPG, PPG, RND | env-defined | env-defined | env-defined | MIT | active |
| **Ray RLlib** | any | PPO, IMPALA, APPO, SAC, DQN… (scalable) | env-defined | env-defined | env-defined | Apache-2.0 | active (Ray 2.55, Apr 2026) |
| **Gymnasium** | any | env API (no algos) | standard | standard | standard | MIT | active (v1.3.0, Apr 2026) |
| **ALE** (Atari) | Atari 2600 | env (benchmark) | pixels / RAM | 18 discrete | game score | GPL-2.0 | active (v0.11.2, 2025) |
| **MineRL / BASALT** | Minecraft | env + human-demo datasets | 640×360 pixels | near-human GUI/mouse | task / learned | see repo `(unverified)` | semi-active (v1.0.2, 2023) |
| **RLGym** | Rocket League | env API | game state | continuous controller | user-defined | Apache-2.0 | active |

---

## Trackmania (TMNF) & sim-racing (highest priority)

### tmrl — the closest comparable
`trackmania-rl/tmrl` is a real-time distributed RL framework whose flagship
application is Trackmania. It primarily targets **TM2020** with **TMNF legacy
support**. Default algorithm is **Soft Actor-Critic (SAC)**, with a
**REDQ-SAC** variant (an ensemble of value networks, sampled as a subset) for
sample efficiency. It ships **two observation modes**: a **19-beam LIDAR**
vector (+ speed) for an MLP, and **full screenshots** (+ speed/gear/rpm) for a
CNN, both with configurable frame-stacking; a LIDAR-plus-track-progress variant
also exists. Actions are **continuous** gas/brake/steer emulated through an
Xbox360 controller (`vgamepad`). The **reward** is progress along a recorded
demonstration trajectory split into equally spaced points — "the number of
points passed since the previous timestep." Infra is built on **`rtgym`**
(elastically-constrained real-time timesteps) in a **single-server /
multiple-clients** distributed layout; crucially it is **real-time, not sped
up**. **License: MIT** — the most permissive flagship here. Active (v0.7.1, May
2025; ~707★).

**Real-time-RL lineage.** `rtgym` and tmrl's SAC descend from two papers by
overlapping authors (Bouteiller, tmrl's lead, co-authors the second):
**Ramstedt & Pal, "Real-Time Reinforcement Learning," NeurIPS 2019**
(arXiv:1911.04448) — the **RTMDP** formulation + **RTAC** — and **Ramstedt,
Bouteiller et al., "Reinforcement Learning with Random Delays," ICLR 2021**
(arXiv:2010.02966) — the **RDMDP** + **DCAC**, from which `rtgym` derives. Their
core result: when an environment **cannot be paused**, the action you compute
now is applied a step late, so the bare observation is **not Markov** — you must
augment the state with the **in-flight action(s)** (and, under random delays,
the delay values) and prefer a **state-value critic**; naive SAC on the raw
delayed observation collapses to near-random. The reduction `SAC ⊂ RTAC ⊂ DCAC`
makes this a clean lineage.

> Why it matters most: same game family, same LIDAR-vs-pixels question we face,
> but it answers the algorithm question with **SAC/REDQ** and is MIT-licensed.

### Linesight — the state of the art on TMNF
`Linesight-RL/linesight` is by a wide margin the most capable Trackmania AI:
**first to demonstrate human-level driving (~May 2023)** and **first to beat
world records on official campaign tracks (May 2024)**. It targets **TMNF via
TMInterface** (same tool we depend on). On algorithm, the README itself frames
it as a **value-based, discrete-action approach** ("discrete input algorithms
like DQN can be applied"); the **observation** is a **CNN over a 160×120
greyscale image** of the screen, and the **reward** is the **distance travelled
along a reference racing line over roughly the next 7 seconds**. **Sped-up via
TMInterface.** **License: no `LICENSE` file found in the repo root**
`(unverified)` — treat as all-rights-reserved; read for ideas, do not copy
code. Active (v3.0.0, Jun 2024). The older `terafear/trackmania_rl_public`
repo (still online) is an earlier public "Linesight" codebase, superseded by
the `Linesight-RL` org.

> The standout result in the entire racing field uses **pixels + CNN +
> value-based RL + a reference-line progress reward** — a different bet than our
> telemetry-vector + evolutionary stack.

### MOSEAC / "Elastic Time Steps" — academic SAC variant on Trackmania
An academic line of work from the MIST Lab (Polytechnique Montréal): **MOSEAC
(Multi-Objective Soft Elastic Actor-Critic)**, published as "Reinforcement
Learning with Elastic Time Steps" (arXiv:2402.14961, v4 Aug 2024; earlier
workshop version "MOSEAC: Streamlined Variable Time Step RL", arXiv:2406.01521).
MOSEAC is an **off-policy actor-critic that extends SAC** with **elastic /
variable time steps**: the **control frequency is itself part of the action**,
so the agent learns to act at the *lowest viable rate* (5–30 Hz here) to save
compute. It is validated on **Trackmania**, deliberately reusing the **tmrl
framework, map, and progress-reward methodology** for a head-to-head against
tmrl's SAC.
- *Observation (143-dim):* **4 stacked 64×64 RGB frames → CNN → 128-d
  embedding**, concatenated with car speed, gear, wheel RPM, the episode step
  index, the inter-frame time interval, and the previous two actions — i.e.
  **pixels + telemetry**, like tmrl's full mode.
- *Action (4-dim, continuous):* gas, brake, yaw (steer), **plus the control
  rate (5–30 Hz)** — the elastic-timestep term.
- *Reward:* tmrl-style — progress over evenly spaced path points toward the
  shortest route (÷100), combined *multiplicatively* with a time term; realistic
  physics but no crash detection.
- *Infra / results:* **real-time** (1320+ hours on an i5-13600K + RTX 4070);
  best lap **43.202 s**, with lower energy and time cost than CTCO, SEAC, and
  SAC (20 Hz); convergence backed by a Lyapunov analysis.
- *Code / license:* `alpaficia/MOSEAC` (**MIT**, the core algorithm — tested
  there in a Newton gym env). The Trackmania-specific `TMRL_MOSEAC` repo cited
  in the paper currently 404s.

> The standout idea: **fold the control rate into the action** so the policy
> minimises its own compute. Directly relevant to our fixed step-rate /
> episode-budget story (#327) and a candidate algorithm for #328.

### AndrejGobeX/TrackMania_AI — the algorithm buffet
A notably broad project: the current branch implements **SAC, PPO, and TD3**;
an older branch used **neuroevolution/NEAT and supervised learning**. Its
**observation** is a compact engineered vector — **13 wall-distances**, vehicle
velocity, a wall-contact flag, and the **2 previous actions**, frame-stacked —
i.e. a hand-rolled LIDAR very close in spirit to our `n_lidar_rays`. Continuous
steer + throttle/brake. The **reward** is an adaptation of the **Gran Turismo
Sport** deep-RL reward (Fuchs, Song, Kaufmann, Scaramuzza, Dürr). Game
interface via an **OpenPlanet** data-grab plugin + `vgamepad`. **License:
GPL-3.0** (copyleft — ideas only, do not vendor code). ~117★, active.

> Concrete evidence that **NEAT works on Trackmania** (de-risks the NEAT
> candidate in #328) and that a **13-ray wall-distance vector** is a viable obs.

### LouisDeOliveira/TMAI
A TMNF Gym environment targeting **DQN/DDPG/PPO** (implementation maturity
unclear). Observation is **multi-modal**: MSS screen-capture → OpenCV
(greyscale → threshold → Canny → dilate → blur) → **raycast LIDAR**, *plus*
TMInterface telemetry (speed/yaw/pitch/roll) — essentially the same pipeline as
our `lidar.py`. Discrete arrow-keys or continuous gamepad. **No license file**
`(unverified)`. Small (~13★), low activity.

### Anca-Mt/TrackmaniaRL-AI
A course project (MGAIA Assignment 3) **built on top of tmrl**, comparing
**DDPG, PPO, and two SAC variants** (single- vs dual-critic) across **pixel-CNN
and LIDAR** observation modes with a custom per-track reward. Useful as a small
head-to-head of off-policy vs on-policy on the same task. **No license file**
`(unverified)`; inactive.

### Pheoxis/AITrackmania
Confirmed to be a **fork/mirror of tmrl** (same SAC/REDQ, LIDAR/CNN obs, MIT
"Bouteiller and Geze" notice). No independent contribution to catalogue; listed
only to disambiguate it from original work.

### TheoBoyer/TMForge
A **TM2020** experimentation toolkit exposing a Gym-like API, with a **DQN**
reference agent over a **CNN on frames** and discrete actions. Self-described
beta ("a lot of bugs", "default hyperparameters are still unstable"). **No
license file** `(unverified)`; ~1★, low activity.

### Urela/DriveForever
A minimal TMNF agent that, lacking an official API, uses **computer vision** to
read the screen — **4 stacked 84×84 greyscale frames** — with the **reward
simply equal to car speed**; Linux-only via `pynput`. Algorithm not specified.
Credits Sentdex's GTA-V RL work and tmrl as inspiration. Toy-scale, **no
license** `(unverified)`.

### ShubhamGajjar/TrackMania-ReinforcementLearning
A TMNF agent using **Implicit Quantile Networks (IQN)** — a **distributional**
value method. Observation combines **`dxcam` frames + telemetry + virtual
checkpoints spaced ~10 m** along the track; the **reward** rewards checkpoint
progress and penalises collisions / off-track. **TMInterface ≤1.4.3.**
**License: MIT.** Explicitly **work-in-progress** (v1.4, Sep 2024; ~8★).

> Second TMNF project pointing at **distributional value-based RL** (with
> Linesight), and its **virtual-checkpoint** progress signal mirrors our
> `track_progress`.

### TMInterface (donadigo) — shared infrastructure
`donadigo/TMInterfacePublic` is the public-resources repo for **TMInterface**,
the state-of-the-art **TAS / programmatic-control tool for TMNF** that
`gamer-ai`, Linesight, TMAI and ShubhamGajjar all build on (programmatic
inputs, car-state telemetry, screenshots, savestates). The **core tool is
closed-source freeware**; the public repo (C++, ~25★) ships only non-sensitive
resources and **carries no visible license**. Listed because it is the
de-facto integration layer for the whole TMNF-RL ecosystem, ours included.

### Gran Turismo (GT Sophy, Sony AI) — the sim-racing reference point
A different game (**Gran Turismo Sport** on PS4), but the **academic state of the
art for racing RL** and the reward-design lineage some Trackmania projects borrow
(AndrejGobeX adapts the related Fuchs et al. work, a co-author here). Wurman et
al., *Nature* 2022, "Outracing champion Gran Turismo drivers with deep
reinforcement learning": **GT Sophy beat top human GT drivers**, winning a
4-vs-4 event **104–52** (Oct 2021) and time-trials against world champions across
three car/track combinations. Figures below are from the paper.
- **Algorithm: QR-SAC** — a **distributional (quantile-regression) Soft
  Actor-Critic**: off-policy continuous control with **7-step returns**, clipped
  double-Q, and 32 quantiles; 4×2048 ReLU MLPs. This is exactly the **SAC ×
  distributional-RL intersection** our #328 candidates point at; an ablation
  shows plain SAC is markedly worse (≈117.1 s vs 114.5 s on Maggiore).
- **Observation — engineered state vector, not pixels:** car kinematics (3D
  velocity / angular velocity / acceleration), per-tyre load & slip, course
  progress, surface incline, heading vs the centre line, **3 × 60 "course-ahead"
  points (left/centre/right edges) spanning ~6 s of travel**, contact/off-course
  flags, and last steering/throttle/brake. Opponents are encoded as
  relative position/velocity points (separate front & behind lists) plus a
  slipstream flag. Notably, an **ablation found these course-ahead points beat a
  wall-LIDAR + curvature encoding**.
- **Action — continuous, 10 Hz:** combined throttle/brake ∈ [−1,1] and steering
  ∈ [−1,1] (squashed-normal); they found **no gain from acting faster than
  10 Hz** despite the 60 Hz sim.
- **Reward — hand-tuned multi-term:** course-progress (primary, **masked when
  off-course** so corner-cutting isn't rewarded), off-course / wall / tyre-slip
  penalties, a symmetric **passing bonus** (within 20 m behind / 40 m ahead), and
  collision penalties (any-collision, a squared-speed rear-end term, and an
  "unsporting-collision" term on Sarthe). Racing **etiquette is hard-coded as
  collision penalties** — fault-attribution rewards were tried and abandoned for
  producing "much too aggressive" agents.
- **Infra — real-time, distributed:** an actor-learner setup over **10–21 PS4s**
  (each PS4 hosting up to 20 cars), Reverb replay with **multi-table stratified
  sampling**; ~8 days (time-trial) to 7–12 days (racing); **>45,000 driving
  hours** of experience for one track. Crucially, **pure self-play was
  inadequate** — racing has an "exposure problem" (some skills need cooperative
  opponents) and an asymmetric-penalty structure absent from zero-sum games — so
  training used **mixed-population opponents** (past checkpoints + built-in AI +
  scripted PID controllers).
- **Caveats:** **one policy per car/track** (not a single generalist); **weak
  strategy** (passes too early, over-aggressive near penalties); opponents seen
  only as points; and a heavy **human-in-the-loop policy-selection** pipeline. No
  weights/code released (pseudocode only; game API access is restricted).

> The highest-value reference for our racing games: it says **distributional SAC +
> a lookahead track-geometry observation + a masked-progress reward** is the
> winning recipe — reinforcing #328 (SAC *and* distributional RL) and suggesting a
> "course-ahead points" observation worth trying beside our `n_lidar_rays`.

---

## StarCraft II (high priority)

### AlphaStar (DeepMind) — the reference point
The landmark SC2 agent (Vinyals et al., *Nature* 2019, "Grandmaster level in
StarCraft II using multi-agent reinforcement learning"). Figures below are from
the paper.

- **Observation:** the **raw interface** — a structured list of **up to 512
  units** with attributes, a **minimap**, and scalar stats — *not* rendered
  pixels — but **constrained to a camera-like view** (info on off-camera enemy
  units is hidden; some actions can only target inside the camera).
- **Action space:** a structured, **typed function-with-arguments** action
  (action type, selected units, target unit *or* a point on a 256×256 grid,
  queued, repeat, delay) — **≈10²⁶ choices per step**.
- **Architecture (≈139M params training / 55M at inference):** scalar stats →
  MLP, the **entity list → Transformer (self-attention)**, the minimap →
  **ResNet**, fused by novel **scatter connections**, into a **deep LSTM core**
  for partial observability, then an **auto-regressive action head** (type →
  delay → queued → selected-units → target) using a **pointer network** to pick
  units. Ablation (supervised win-rate vs the Elite bot): the **Transformer**
  (+35 pts) and **pointer network** (+29 pts) are the biggest contributors.
- **Constraints (human-comparable, pro-player-approved):** capped at **22
  non-duplicate actions per 5-second window**; ~**110 ms** action delay and
  ~**370 ms** average between observations. Notably, **raising the APM cap
  *hurt* Elo** (the agent over-invests in micro at strategy's expense).
- **Supervised bootstrap:** every agent is initialised by **imitation learning
  on 971,000 human replays (MMR > 3500, top ~22%)**, then fine-tuned on **16,000
  winning replays at MMR > 6200**. The policy is conditioned on a statistic
  **`z`** = build order (first 20 buildings/units) + cumulative stats; `z` is
  zeroed 10% of the time so an unconditional mode is also learned.
- **RL:** reward is terminal **{−1, 0, +1}** (win/draw/loss) plus **pseudo-
  rewards** for matching the sampled `z` (build-order edit distance,
  cumulative-stats Hamming distance), each active 25% of the time with **its own
  value head**. The update is **actor-critic** with **V-trace** (policy) +
  **TD(λ)** (value) + **UPGO** (a self-imitation term that bootstraps toward
  better-than-average actions), plus a **KL penalty toward the frozen supervised
  policy** to retain human-like diversity. The critic is **opponent-aware
  during training only** (a privileged baseline — ablation: 22% → 82% win-rate).
- **League training (PFSP):** **3 main agents** (one per race, never reset) +
  **3 main exploiters** + **6 league exploiters**, ~**900 distinct players**
  total. Opponents are sampled by **prioritised fictitious self-play**
  (weighting `f_hard(x)=(1−x)ᵖ`, focusing on the hardest unbeaten opponents);
  exploiters periodically reset to the supervised weights to keep finding
  weaknesses. The league Nash does not cycle or regress.
- **Compute:** each of the 12 concurrent agents trained on **32 third-gen TPUs
  for 44 days**, running ~16,000 concurrent matches; the learner consumes
  ~50,000 agent-steps/s.
- **Result:** evaluated as **three separate per-race agents** (not one
  multi-race agent) on Battle.net under blind/anonymous conditions. **AlphaStar
  Final**: MMR **6,275 (Protoss) / 6,048 (Terran) / 5,835 (Zerg)** —
  **Grandmaster, > 99.8%** of ranked players (top ~0.15%). **AlphaStar
  Supervised** (imitation only, no RL) already reached **3,699 MMR / top 16%**;
  AlphaStar Mid (27 days of league) ≈ top 0.5%. Uses **no tree search / MCTS**
  (a deliberate contrast to AlphaGo/AlphaZero).
- **Open source = AlphaStar Unplugged, not the Nature system.** The repo
  `google-deepmind/alphastar` (**Apache-2.0**, ~567★, archived) implements the
  **AlphaStar Unplugged** *offline-RL* benchmark (Mathieu et al., 2023,
  arXiv:2308.03526): a fixed ~**1.4M-game** human-replay dataset (MMR > 3500,
  all 3 races + 10 maps, **no environment interaction during training**) with six
  reference agents — behaviour cloning, fine-tuned BC, offline actor-critic
  (OAC), emphatic OAC, and MuZero-Supervised ± inference-time MCTS. The released
  artefact is the **BC backbone**; the **online league-training system was *not*
  released**. Result: the best offline agent (MZS-MCTS) wins **~90% vs the
  all-races BC baseline** but sits at **Elo 1578 vs online AlphaStar Final's
  2968** — offline RL narrows the gap but stays far below online play. The
  winning recipe was **one-step offline RL** (improve off a *fixed* behaviour
  value); bootstrapping a target-policy value diverged, and DQN/CRR/BCQ/AWR/
  PPO/RCBC all failed to beat plain BC.

> Out of our league computationally, but it is the canonical design for **any
> future SC2 self-play / 1v1 ladder ambition**: imitation bootstrap → PFSP
> league self-play, entity-transformer obs, autoregressive actions. Our
> `max_apm` already echoes its action-rate cap; everything else (transformer
> encoder, league, V-trace/UPGO) is far beyond our current numpy/ES stack.

### OpenAI Five (OpenAI) — the large-scale-self-play contrast (Dota 2)
A different game (**Dota 2**, a 5v5 MOBA), included as the natural **contrast to
AlphaStar**: both are landmark large-scale RL agents for real-time team games
that made opposite bets. Berner et al., 2019 (arXiv:1912.06680): OpenAI Five
**beat the human world champions (Team OG) 2-0 in April 2019** — the first AI to
win an esports world title — then went **99.4% across 7,257 public games**.
- **Algorithm:** **PPO** (actor-critic + GAE) with a single **4096-unit LSTM**
  core (~159M params, 84% in the LSTM) and **pure self-play, no imitation** (the
  lone use of human data is item-build randomisation) — the sharp opposite of
  AlphaStar's replay bootstrap.
- **Self-play:** 80% vs the latest policy, 20% vs past versions sampled by a
  quality score — **no league/exploiter structure**. A **"team spirit"** scalar
  (annealed 0→1) interpolates each hero's reward between selfish and team-shared.
- **Observation:** **not pixels** — ~16,000 curated API features/hero, unit-
  centric over 189 units; **five identical-parameter networks** (one per hero),
  each with its own LSTM state + a hero-id embedding.
- **Action:** factored discrete (primary × delay × unit-selection × offset),
  ~8k–80k effective choices/step; acts every 4th frame (7.5 Hz, ~217 ms).
- **Reward:** hand-shaped (gold, XP, kills, towers, win +5, …) with zero-sum
  symmetrisation and time-decay; a win/loss-only ablation still learned, slower.
- **Scale (the headline):** ~770 PFlops/s-day, **~180 training-days over 10
  calendar months**, batch up to ~3M timesteps, **tens of thousands of CPU cores
  + ~512–1,536 GPUs**; the **"surgery"** technique let one parameter set survive
  ~20 env/architecture changes without a restart.
- **Restrictions:** **17 of 117 heroes**; some multi-unit-control items banned;
  item-buying / ability-builds / courier scripted.

> The **AlphaStar vs OpenAI Five split** is the key lesson for us: **imitation +
> structured league** (AlphaStar) vs **pure self-play + shaped reward at brute
> scale** (OpenAI Five). Both reached superhuman team play, but OpenAI Five's
> route cost months on tens of thousands of cores — underlining that the
> self-play-at-scale regime is far outside our budget, and that any SC2 ambition
> of ours should lean on AlphaStar-style **imitation**, not OpenAI-Five-style
> **scale**.

### PySC2 / SC2LE (DeepMind) — the environment we build on
`google-deepmind/pysc2` (**Apache-2.0**, ~8.3k★) is DeepMind's Python wrapper
around Blizzard's SC2 ML API — the foundation of our SC2 integration. It
defines the **feature-layer observation API** (minimap + screen, used at 64×64),
the **function-id + args action API**, ships the **7 minigames** and scripted
baseline agents, and accompanies the SC2LE paper ("StarCraft II: A New Challenge
for RL", arXiv:1708.04782). v4.0 (2022) added the C++ converters used by
AlphaStar; actively maintained.

Design details confirmed from the SC2LE paper, several of which our SC2
integration mirrors:
- **Action space** is an atomic *compound function* `(function-id, args)` — e.g.
  `move_screen(queued, screen)`, `select_rect(...)` — over ~300 function ids
  with up to 13 argument types; **available-action masking** filters illegal
  actions each step. Policies factor **auto-regressively**
  (π(a|s) = ∏ π(aˡ|a^<l, s)); the baselines simplify to independent heads. This
  is the lineage of both AlphaStar's action head and our `[fn_idx, x, y, queue]`.
- **Two reward structures**: ternary win/0/loss, and the denser player-centric
  **"Blizzard score"** (running sum of resources + upgrades + live/built units) —
  what our `score_weight` tracks.
- **Baselines**: trained with **A3C**, 64 async threads, 100 hyperparameter runs
  each, **600M steps**, acting every 8 game frames (≈180 APM). Three
  architectures — **Atari-net**, **FullyConv** (resolution-preserving conv for
  spatial actions), **FullyConv-LSTM** — plus random-policy / random-search.
- **Full 1v1 game** (Abyssal Reef LE, TvT, 30-min cap): **no agent won a single
  game against the easiest built-in AI**, under either reward — Blizzard-score
  agents collapsed to trivial "keep mining" behaviour. The full game is
  intractable for off-the-shelf RL without extra signal (human replays).
- **A `raw` API** (full list of visible units with attributes, no camera) also
  exists; the paper flags it as **"cheating"** versus humans and does not use it
  for the baseline agents. (AlphaStar later used the raw interface.)

The **published baseline minigame scores** are the natural benchmark for our SC2
minigame runs. Reproduced from SC2LE Table 1 (per-episode score; human/random
rows are MEAN, agent columns are **best mean** across 100 hyperparameter runs):

| Minigame | Random | Rand-search | DeepMind human | GrandMaster | Atari-net | FullyConv | FullyConv-LSTM |
|---|---|---|---|---|---|---|---|
| MoveToBeacon | 1 | 25 | 26 | 28 | 25 | 26 | 26 |
| CollectMineralShards | 17 | 32 | 133 | 177 | 96 | 103 | 104 |
| FindAndDefeatZerglings | 4 | 21 | 46 | 61 | 49 | 45 | 44 |
| DefeatRoaches | 1 | 51 | 41 | 215 | 101 | 100 | 98 |
| DefeatZerglingsAndBanelings | 23 | 55 | 729 | 727 | 81 | 62 | 96 |
| CollectMineralsAndGas | 12 | 2318 | 6880 | 7566 | 3356 | 3978 | 3351 |
| BuildMarines | <1 | 8 | 138 | 133 | <1 | 3 | 6 |

> Reward shapes worth noting for our own configs: DefeatRoaches gives **+10 per
> roach killed, −1 per marine lost**; DefeatZerglingsAndBanelings **+5 per kill**.
> Agents reach near-human play on MoveToBeacon/CollectMineralShards but lag the
> GrandMaster badly on combat and economy micro — a realistic bar for our runs.

### reaver (inoryy)
`inoryy/reaver` is a modular deep-RL framework focused on SC2 minigames,
implementing **A2C and PPO** (with GAE, reward/grad clipping, advantage
normalisation) over PySC2's obs/action specs, with multi-env parallelism.
**License: MIT**, ~561★, but **explicitly unmaintained** (last release Nov
2018). Its README independently **reproduces three of the SC2LE baselines**
(MoveToBeacon 26.3, CollectMineralShards 102.8, DefeatRoaches 72.5 with A2C) —
a useful sanity check that the table above is reproducible.

### chris-chris/pysc2-examples
`chris-chris/pysc2-examples` (**Apache-2.0**, ~757★) is an early tutorial-grade
collection of SC2 minigame agents — **DQN** (with optional prioritized replay
and dueling) and **A2C/A3C** — mainly on CollectMineralShards and
DefeatZerglings. Dated but useful as a minimal worked example of wiring DQN to
PySC2.

### python-sc2 / BurnySc2 — *not* RL (disambiguation)
The popular `python-sc2` / `BurnySc2` library is a **scripted-bot API** for SC2,
not a reinforcement-learning project. Mentioned only so it isn't mistaken for an
RL baseline; excluded from the RL comparison.

---

## General / other-game libraries (lighter pass)

These are reference implementations and environment standards rather than
game-specific agents — relevant as algorithm sources (#328) and as a
"which-policy-when" reference (#327).

- **Stable-Baselines3** (`DLR-RM/stable-baselines3`, **MIT**, ~13.3k★, v2.8.0
  Apr 2026): the de-facto PyTorch baseline library — **PPO, A2C, DQN, SAC, TD3,
  DDPG** (+ SB3-Contrib **QR-DQN, TQC, TRPO, ARS, CrossQ, RecurrentPPO,
  MaskablePPO**), all on the **Gymnasium** API. This is exactly the menu #328
  proposes; our orphaned `rl/train.py` already uses SB3.
- **CleanRL** (`vwxyzjn/cleanrl`, **MIT**, ~9.8k★): **single-file**, readable
  implementations of **PPO, DQN, C51, SAC, TD3, DDPG, PPG, RND** (PyTorch, some
  JAX). Best source for *understanding* an algorithm before porting it to our
  numpy/BasePolicy contract.
- **Ray RLlib** (`ray-project/ray`, **Apache-2.0**, ~42.6k★): **scalable /
  distributed** RL (PPO, IMPALA, APPO, SAC, DQN…). Relevant to our
  `distributed/` and `n_workers` story if we ever outgrow the in-house setup.
- **Gymnasium** (`Farama-Foundation/Gymnasium`, **MIT**, ~11.9k★): the
  maintained successor to OpenAI Gym and the env-API standard our
  `BaseGameEnv` mirrors.
- **Arcade Learning Environment** (`Farama-Foundation/Arcade-Learning-Environment`,
  **GPL-2.0**, ~2.4k★): the Atari 2600 benchmark — the classic discrete-action
  pixels testbed, relevant to the Gym classic-control / Atari game-support ideas
  in #215/#216/#217.
- **MineRL / BASALT** (`minerllabs/minerl`, license **`(unverified)`**, ~948★):
  Minecraft environments + **human-demonstration datasets**; the BASALT
  competition centres on **learning from human demos** — the clearest
  open-source precedent for an imitation-bootstrap path (cf. AlphaStar).
- **RLGym** (`RLGym/rlgym`, **Apache-2.0**, ~239★): a Gym-style API for **Rocket
  League** (RocketSim backend) — directly relevant since we integrate Rocket
  League; a model for state-based (non-pixel) continuous control.

---

## Takeaways for `gamer-ai`

Concrete, evidence-backed ideas, cross-referenced to open issues. Licenses
noted above — these are *ideas to try*, not code to copy.

1. **SAC is the field's default for continuous-control racing → prioritise it
   in #328.** tmrl (SAC + **REDQ**), Anca-Mt (SAC variants), AndrejGobeX
   (SAC/TD3), MOSEAC (SAC + elastic time steps), and — most tellingly — **GT
   Sophy (QR-SAC), which beat champion human Gran Turismo drivers** all use
   off-policy actor-critics on the exact `Box` steer/accel/brake shape our
   racing games expose. **REDQ-SAC** specifically
   targets sample efficiency — attractive because, unlike sped-up TMNF, several
   of our racing targets (BeamNG/Assetto/Rocket League) are effectively
   real-time. *Action:* land SAC (and consider REDQ) for the continuous racing
   games in #328, alongside the already-flagged PPO.

2. **PPO really is the universal baseline → land it first (as #328 already
   suggests).** Every general library (SB3/CleanRL/RLlib) leads with PPO, the SC2
   minigame agent `reaver` uses A2C/PPO, and AndrejGobeX uses PPO for racing
   (`pysc2-examples` covers the A2C/DQN side). This survey corroborates #328's
   claim that PPO is "the most-cited baseline competing projects report";
   promoting the orphaned SB3 PPO in `rl/train.py` into a registered `ppo`
   policy is the highest-value first step.

3. **Distributional value methods are proven racing winners → a concrete target
   beyond vanilla DQN (#328's "Rainbow line").** Linesight (value-based,
   DQN-family), ShubhamGajjar (**IQN**), and above all **GT Sophy's QR-SAC** —
   the champion-beating Gran Turismo agent fuses *distributional* value
   estimation with SAC — show this is the racing SOTA, not a niche. IQN/QR-DQN is
   a well-scoped upgrade to `framework/dqn.py` (SB3-Contrib's **QR-DQN** is a
   ready reference), and GT Sophy is the proof that **combining #328's SAC and
   distributional candidates (i.e. QR-SAC) is itself worth scoping**. GT Sophy
   also found **multi-step (7-step) returns** a large, cheap win over 1-step — an
   easy lever for any of our value-based/actor-critic additions.

4. **NEAT is validated on Trackmania → de-risks the NEAT candidate (#328).**
   AndrejGobeX's neuroevolution branch shows topology-evolving ES works on this
   problem, complementing our fixed-architecture ES policies.

5. **Our LIDAR/telemetry observation is mainstream — but the SOTA uses pixels.**
   tmrl (19-beam LIDAR), AndrejGobeX (13 wall-distances), and TMAI (raycast
   LIDAR from edge images) validate our `n_lidar_rays` design and the
   MSS+OpenCV pipeline in `lidar.py`. **However**, the single best result in the
   field — Linesight beating world records — uses a **CNN on a 160×120 greyscale
   frame**, as do tmrl's full mode and DriveForever. **But GT Sophy, the
   champion-beating agent, deliberately uses a telemetry/feature vector and an
   ablation found its `3×60 course-ahead points` beat a wall-LIDAR encoding** —
   so engineered features are not a dead end. *Takeaway for #327/#328:* document
   the **pixels-vs-telemetry tradeoff** explicitly; consider adding **lookahead
   "track-ahead" geometry points** beside our `n_lidar_rays`, and a pixel-CNN
   racing policy as a stretch goal if features plateau.

6. **Progress-based rewards are the norm → compare against our `progress_weight`;
   borrow GT Sophy's refinements.** tmrl rewards **distance advanced along a
   recorded demo trajectory**; Linesight rewards **distance along a reference
   line over the next ~7 s**; ShubhamGajjar uses **~10 m virtual checkpoints**;
   GT Sophy uses **centre-line arc-length progress, masked to zero when
   off-course** so corner-cutting earns nothing. Our centreline-fraction
   `track_progress` is the same idea, but our reward lacks GT Sophy's
   **off-course mask** and its companion penalty palette (wall ∝ speed², tyre
   slip, and — for any multi-car racing like Rocket League — a symmetric passing
   bonus plus collision/"unsporting" penalties, since fault-attribution rewards
   made agents too aggressive). A **lookahead** ("progress over the next N
   seconds/metres") variant is also worth prototyping — it gave Linesight a
   denser, smoother signal.

7. **For SC2 ambitions beyond minigames, AlphaStar is the blueprint.** If we
   ever pursue 1v1 ladder play seriously, the proven recipe is **imitation
   bootstrap from human replays → PFSP league self-play**, with an
   **entity-transformer + deep-LSTM** obs encoder and an **autoregressive
   function+args** action head. Our current evolutionary/tabular SC2 policies
   won't reach that ceiling; this is a multi-issue research arc, not a quick
   win. The **APM-limiting** idea is already reflected in our `max_apm`. The
   *alternative* — OpenAI Five's **pure self-play at brute scale** (no
   imitation, but months on tens of thousands of cores) — is even further out of
   reach, which is exactly why the **imitation-bootstrap path is the tractable
   one for us** (see #8), not raw self-play scale.

8. **Imitation / offline RL is a gap we have no answer for — and it pays off
   hugely.** AlphaStar (replays), AlphaStar Unplugged (offline RL / behaviour
   cloning), SC2LE (replay-supervised policies beat its RL agents on
   BuildMarines), and MineRL/BASALT (human-demo datasets) all bootstrap from
   demonstrations. The quantified payoff is striking: **AlphaStar's
   imitation-only "Supervised" agent already reached the top 16% of ranked
   players (3,699 MMR) before any RL**, and every league agent was *initialised*
   from it. We have no imitation path. For games where we *can* record human or
   scripted demos (TMNF replays exist in `replays/`; SC2 has replay files), a
   behaviour-cloning warm start could dramatically cut cold-start cost — a
   candidate future issue.

9. **Benchmark SC2 minigames against the published baselines.** Use the
   DeepMind SC2LE scores (MoveToBeacon 26, CollectMineralShards 103,
   DefeatRoaches 100) as explicit targets/regression markers for our SC2
   minigame runs, rather than judging runs in isolation.

10. **Borrow algorithm *ideas* from SB3/CleanRL, not code, and mind licenses.**
    CleanRL's single-file style is the cleanest reference for porting PPO/SAC to
    our `BasePolicy` contract (#328). Note **GPL-3.0** (AndrejGobeX) and
    **unlicensed** repos (Linesight, TMAI, TMForge, DriveForever, terafear):
    study for design, do not vendor.

11. **Elastic / variable time steps — let the policy choose its control rate
    (MOSEAC).** Our step rate is fixed per game (TMNF `speed`, SC2 `step_mul`).
    MOSEAC folds the **control frequency into the action** (5–30 Hz) and
    optimises a multi-objective reward that favours the *lowest viable* rate,
    cutting compute without hurting task performance. Worth scoping both as an
    obs/action-design experiment and as a compute-budget lever for #327's
    "sizing a run", and as a candidate algorithm for #328.

12. **Our sped-up / headless setup *sidesteps* the real-time-RL problem — state
    that as an advantage, and know the cost if we go live.** The strongest
    racing agents train against a **non-pausable live game** (tmrl, GT Sophy on
    a PS4) and therefore pay for it: per the RTRL/RDMDP papers (arXiv:1911.04448,
    arXiv:2010.02966), when you can't pause, the action computed now lands a step
    late, the bare observation isn't Markov, and you must augment the state with
    the in-flight action(s) + delays and use a state-value critic (naive SAC
    collapses to near-random). **Our TMInterface speed-up (≤10×) and pausable,
    headless PySC2 make interaction effectively turn-based, so none of this
    machinery is needed** — a real, if unglamorous, advantage of our
    architecture. The flip side: if we ever target a genuinely real-time,
    non-pausable game (BeamNG / Assetto / Rocket League at native speed), we'd
    need exactly this — RTMDP/RDMDP state augmentation (cf. our existing
    prev-action features) and a delay-aware critic.

---

## Sources

All links verified to resolve, May 2026.

**TMNF / Trackmania**
- tmrl — https://github.com/trackmania-rl/tmrl
- Linesight (repo) — https://github.com/Linesight-RL/linesight
- Linesight (docs) — https://linesight-rl.github.io/linesight/build/html/
- AndrejGobeX/TrackMania_AI — https://github.com/AndrejGobeX/TrackMania_AI
- LouisDeOliveira/TMAI — https://github.com/LouisDeOliveira/TMAI
- Anca-Mt/TrackmaniaRL-AI — https://github.com/Anca-Mt/TrackmaniaRL-AI
- Pheoxis/AITrackmania — https://github.com/Pheoxis/AITrackmania
- TheoBoyer/TMForge — https://github.com/TheoBoyer/TMForge
- Urela/DriveForever — https://github.com/Urela/DriveForever
- ShubhamGajjar/TrackMania-ReinforcementLearning — https://github.com/ShubhamGajjar/TrackMania-ReinforcementLearning
- terafear/trackmania_rl_public — https://github.com/terafear/trackmania_rl_public
- TMInterface (public resources) — https://github.com/donadigo/TMInterfacePublic
- MOSEAC — "RL with Elastic Time Steps" (arXiv) — https://arxiv.org/abs/2402.14961
- MOSEAC — workshop version (arXiv) — https://arxiv.org/abs/2406.01521
- MOSEAC code — https://github.com/alpaficia/MOSEAC
- GT Sophy — Wurman et al., *Nature* 2022 — https://www.nature.com/articles/s41586-021-04357-7
- GT Sophy — race videos — https://sonyai.github.io/gt_sophy_public
- Real-Time Reinforcement Learning (RTAC), NeurIPS 2019 — https://arxiv.org/abs/1911.04448
- Reinforcement Learning with Random Delays (DCAC / `rtgym`), 2021 — https://arxiv.org/abs/2010.02966
- The History of Machine Learning in Trackmania — https://hallofdreams.org/posts/trackmania-1/

**StarCraft II**
- AlphaStar (open source) — https://github.com/google-deepmind/alphastar
- AlphaStar blog — https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/
- AlphaStar (Nature 2019) — https://www.nature.com/articles/s41586-019-1724-z
- AlphaStar Unplugged (offline RL) — https://arxiv.org/abs/2308.03526
- PySC2 — https://github.com/google-deepmind/pysc2
- SC2LE paper — https://arxiv.org/abs/1708.04782
- reaver — https://github.com/inoryy/reaver
- chris-chris/pysc2-examples — https://github.com/chris-chris/pysc2-examples
- OpenAI Five — "Dota 2 with Large Scale Deep RL" — https://arxiv.org/abs/1912.06680

**General / other-game libraries**
- Stable-Baselines3 — https://github.com/DLR-RM/stable-baselines3
- CleanRL — https://github.com/vwxyzjn/cleanrl
- Ray RLlib — https://github.com/ray-project/ray
- Gymnasium — https://github.com/Farama-Foundation/Gymnasium
- Arcade Learning Environment — https://github.com/Farama-Foundation/Arcade-Learning-Environment
- MineRL — https://github.com/minerllabs/minerl
- RLGym — https://github.com/RLGym/rlgym
