# CLAUDE.md

## Knowledge graph

A knowledge graph of this codebase lives at `graphify-out/graph.json` (7,025 nodes,
16,446 edges, 371 communities). It is kept current by CI on every merge to `main`.

**Use `/graphify query "<question>"` instead of reading individual files when exploring
architecture, tracing a dependency, or understanding how a concept is used across the
codebase.** The graph traversal is faster and gives cross-file context that file-by-file
reading misses.

Key god nodes (highest betweenness centrality — touch everything):

| Node | File | Role |
|---|---|---|
| `ObsSpec` | `framework/obs_spec.py` | Input contract — single source of truth for observation feature names, scales, and normalization across all games |
| `ExperimentData` | `framework/analytics.py` | Output contract — every run's result flows through this dataclass |
| `BasePolicy` | `framework/policies.py` | Policy protocol — all algorithms implement this interface |
| `GreedySimResult` | `framework/analytics.py` | Per-episode result — bridges training loop and analytics |
| `SC2Client` | `games/sc2/client.py` | SC2-specific god node by accident — has leaked beyond the adapter boundary (see issue tracker) |

When a task involves any of these nodes, query the graph first to understand the blast
radius before touching code.

To regenerate the Obsidian vault locally: `/graphify --obsidian`

---

## Memory management

Grid searches run many experiments back-to-back in the same process, so unbounded
in-memory accumulation is a hard constraint, not a style preference.  Before adding
any data structure that grows over time, check that it has a clear upper bound or an
explicit eviction/trim path.

Rules that apply to every code addition:

- **Replay buffers and ring buffers** must be bounded by a configured `maxlen` /
  `capacity` and must evict old entries when full.  Never append to an unbounded list
  across episodes or simulations.
- **Per-episode or per-step bookkeeping** (reward components, action histories,
  belief grids, logging buffers) must be cleared at episode reset.  Storing these on
  `self` is fine; letting them grow across episodes is not.
- **Caches and dicts keyed on episode or step indices** must be pruned or recreated
  at episode / experiment boundaries so memory does not grow with run length.
- **Population arrays** (genetic, CMA-ES, LSTM-ES) are naturally bounded by
  `population_size`; don't accumulate the full history of all generations.
- **History lists used for analytics** (reward curves, champion logs) belong in
  `ExperimentData`, not in policy or env objects that persist across experiments.
- **Logging and debug buffers** must be off by default or bounded (e.g. a circular
  buffer of the last N entries).  Never log every step unconditionally.

When reviewing a diff, flag any `list.append`, `dict[key] =`, or `deque` that lacks
a matching size cap or reset call — those are the usual sources of slow memory leaks
across a long grid search.

---

Multi-game RL agent framework. A game-agnostic training loop and policy
set (`framework/`) drive per-game integrations (`games/<game>/`) through a
common adapter interface (`framework/game_adapter.py`), training autonomous
agents via hill-climbing / evolutionary / CMA-ES / Q-learning /
policy-gradient methods. The project began as a Trackmania Nations Forever
(TMNF) agent and now spans nine games.

**Runtime per game** (select with `--game`; default `tmnf`):
- TMNF: Windows-only (`pywin32`, `mss` window grab, `tminterface` bind to live game process).
- TORCS: Linux/Windows via `gym_torcs`.
- SC2: Linux/Windows/Mac via PySC2; runs headless on Linux against the Blizzard SC2 binary.
- CarRacing: cross-platform; Gymnasium `CarRacing-v2`, pure-Python (needs `gymnasium[box2d]`, no separate game binary).
- BeamNG: BeamNG.drive (commercial) via the `beamng_gym` TCP API.
- Assetto Corsa: Assetto Corsa (commercial) via the `assetto-corsa-rl` shared-memory wrapper; launched through its own entry point (`--game assetto`).
- Rocket League: Rocket League (commercial, Windows) via the `rlgym` API + Bakkesmod (`--game rocket_league`).
- iRacing: iRacing (commercial, Windows) telemetry via `pyirsdk`, with optional vJoy live action injection (`--game iracing`).
- Atari: cross-platform, headless; Atari 2600 via `ale-py` (MIT-licensed ROMs bundled). 128-byte RAM observation, `Discrete(N)` action space. `poetry install --with atari` (`--game atari`).

Each game ships a `games/<game>/README.md` documenting its install,
config, observation/action space, reward, and supported policies. This file
is the architecture overview — defer to those per-game READMEs for
game-specific specifics.

**Per-game reward documentation convention.** Every game README must contain
a `## Rewards` section (plural) with a Markdown table — columns
`Parameter | Default | Description` — covering every key in the game's
`reward_config.yaml`. This is where formulas, interaction effects, tuning
notes, and preset recommendations live. The YAML file itself should be a
clean list of `key: value` pairs with a single one-line header comment
pointing to the README (`# Reward configuration for <GAME>. See
games/<name>/README.md for full documentation.`). Avoid paragraph-style
explanations or multi-line comment blocks inside the YAML. Short inline
hints (e.g. `# "none" | "icm" | "rnd"` for an enum value) are acceptable
when the valid options are not obvious from context. The
`games/_template/` files show the expected pattern for new games.

The framework-side protocols a contributor implements to ship a game or a
policy are documented one file per protocol under
[`docs/framework/`](docs/framework/README.md): `GameAdapter`, the
`GameSpec` / `RunConfig` / `ProbeSpec` / `WarmupSpec` / `PolicyExtras`
config bundles, `BaseGameEnv`, `RewardCalculatorBase`, `BasePolicy`,
`ObsSpec`, and `BCAdapter`. The sections below describe how these are
configured and combined; the `docs/framework/` pages give the
per-protocol method contracts and worked examples.

---

## Repository Structure

```
gamer-ai/
├── main.py                 # Entry point — python main.py <experiment_name> [--game <game>]
├── grid_search.py          # Grid search over param combos (supports --distribute / --local-workers)
├── analytics.py            # Backward-compat shim → framework.analytics
├── cross_grid_report.py    # Recursive cross-grid-search comparison report
├── redo_analytics.py       # Regenerate analytics from saved experiment data (no re-training)
├── param_explorer.py       # Interactive weight/param exploration tool
├── policies.py             # Backward-compat shim → framework + games/tmnf policies
├── batch_run.sh            # Run multiple experiments back-to-back
├── setup_and_run.ps1       # Windows bootstrap script
├── scripts/                # Dev scripts (e.g. release.py)
├── CHANGELOG.md            # Running log of user/developer-visible changes
├── CONTRIBUTING.md         # Contribution guide
├── pyproject.toml
├── framework/              # Game-agnostic training loop, obs spec, analytics, base policies,
│                           #   game_adapter registry, parallel_eval, version
├── games/                  # Per-game integrations; each holds its own config/ master files
│   ├── tmnf/               # TMNF-specific env, reward, lidar, steering, policies, clients, tools
│   ├── torcs/              # TORCS racing simulator integration
│   ├── sc2/                # StarCraft 2 integration via PySC2 (Linux-friendly, headless)
│   ├── car_racing/         # Gymnasium CarRacing-v2 (pure-Python, no game binary)
│   ├── beamng/             # BeamNG.drive via beamng_gym TCP API
│   ├── assetto_corsa/      # Assetto Corsa via assetto-corsa-rl shared memory
│   ├── rocket_league/      # Rocket League via rlgym + Bakkesmod
│   ├── iracing/            # iRacing telemetry via pyirsdk (+ optional vJoy injection)
│   └── atari/              # Atari 2600 via ale-py (RAM observation, Discrete actions)
├── clients/                # Backward-compat shim → games/tmnf/clients
├── rl/                     # Backward-compat shim (legacy; BC now lives in framework/bc.py)
├── distributed/            # Coordinator, worker, protocol for distributed grid search
├── infrastructure/         # Terraform: auth, remote_state, environment (Azure VMs)
├── experiments/            # Per-experiment results (git-ignored)
├── tests/                  # Unit tests
├── tracks/, replays/       # Centerline .npy files and TMNF replay .Gbx files
└── runs/, plans/           # Saved run metadata and planning notes
```

Master configs and grid-search templates live **per game** under
`games/<game>/config/` (resolved at runtime via the adapter's
`config_dir`); there is no top-level `config/` directory.

---

## Running

```bash
# Single experiment (TMNF — default)
python main.py <experiment_name> [--no-interrupt] [--re-initialize]

# Run on a different game (--game: tmnf | torcs | sc2 | car_racing | beamng | assetto)
python main.py <experiment_name> --game torcs
python main.py <experiment_name> --game sc2
python main.py <experiment_name> --game car_racing

# Grid search over param combinations (config lives under games/<game>/config/)
python grid_search.py games/sc2/config/grid_search_template.yaml [--no-interrupt]

# Tests
python -m pytest tests/
```

First run with new name: `experiments/<track>/<name>/` created, both
master configs copied in from `games/<game>/config/`. Edit experiment
copies to tune without affecting others. `--re-initialize` ignores
existing weights file, re-runs probe + cold-start.

Other useful flags: `--track <name>` overrides the config's track/map
(e.g. `aalborg`, `CollectMineralShards`); `--workers N` overrides
`n_workers` for SC2 intra-run parallel evaluation; `--log-level` sets
verbosity (`DEBUG` / `INFO` / `WARNING` / `ERROR`, default `INFO`).
SC2-only modes `--play` and `--eval` are covered in the StarCraft 2
section.

**Behaviour cloning (`--bc`).** Any game whose `GameAdapter` exposes a
`BCAdapter` can pre-train a policy from demonstration data before the RL
loop:

```bash
# SC2 — clone from replay files, then fine-tune
python main.py myrun --game sc2 --bc --replay-dir /path/to/replays --bc-race terran
python main.py myrun --game sc2

# TMNF — drive SimplePolicy live for 3 laps, then fine-tune
python main.py myrun --game tmnf --bc
python main.py myrun --game tmnf
```

`--bc` writes `policy_weights.yaml` (+ optional `trainer_state.npz`) and
`bc_summary.json` into the experiment directory; the normal training run
picks them up automatically.  Mutually exclusive with `--play` and `--eval`.
See [`docs/framework/bc_adapter.md`](docs/framework/bc_adapter.md) for the
per-game extension contract, and the per-game READMEs for game-specific
workflow details.

---

## Test suite README

`tests/README.md` documents every test (one line each, grouped by file and
area) plus a per-area "what is and isn't tested" summary and a runtime
explanation. **Whenever tests are added, removed or substantially changed,
update `tests/README.md` in the same change.** At minimum: add/remove the
file's section, and revise the relevant area's tested/not-tested paragraph if the boundary between
unit-tested logic and mocked integration shifts.

---

## Pre-commit checks

The repository uses [pre-commit](https://pre-commit.com/) to enforce YAML
validity, merge-conflict markers, end-of-file newlines, trailing whitespace,
and Ruff lint + format. **All pre-commit checks must pass before committing.**

```bash
# Install once
pip install pre-commit
pre-commit install   # installs the git hook so it runs automatically

# Run manually on all files
pre-commit run --all-files
```

Ruff's `--fix` flag auto-corrects most lint issues in-place; run
`pre-commit run --all-files` a second time after it modifies files to confirm
everything is clean before staging. **Never use `--no-verify` to skip the
hook.** Fix the underlying issue instead.

---

## Changelog

`CHANGELOG.md` is the project's running log of user- and
developer-visible changes (new features, new config keys, breaking
changes, bug fixes, dependency changes, doc-only changes to public
files). **Every PR that introduces such a change must add an entry
under `## [Unreleased]`.** Trivial commits — experiment dumps,
formatting, internal refactors with no behaviour change — can be
skipped. The PR template's Documentation checklist enforces this.

---

## Pull requests

Every PR description **must** be filled in using the repo's PR template
(`.github/PULL_REQUEST_TEMPLATE.md`) — keep the section(s) that apply to
the change and delete the rest.

- **Link the issue at the top.** Put `Closes #<issue>` near the top of
  the description (above the `## Summary` section) so the PR
  auto-closes its issue on merge. Use `Refs #<issue>` for related
  issues the PR should *not* close.
- Fill in `## Summary` with 1–3 bullets on what changed and why, plus
  the bug-fix / feature / refactor detail section that matches the
  change.
- Record validation commands and complete the template's checklist —
  including the `CHANGELOG.md`, `README.md`, `CLAUDE.md`, and
  `tests/README.md` items where they apply.

---

## Issue labels

The canonical label set is documented in `CONTRIBUTING.md`. One hard
rule on top of it:

- **Documentation issues must never carry the `good first issue` label.**
  A `documentation` issue is not a `good first issue` — onboarding
  newcomers via docs work means trusting them to set the project's
  authoritative description of itself, which demands more context than a
  first contribution should. Keep the two labels mutually exclusive: if
  an issue is labelled `documentation`, do not add `good first issue`,
  and when triaging an existing issue that has both, drop
  `good first issue`.

---

## Versioning & releases

`gamer-ai` does not (yet) ship to PyPI, but it has a version number so
runs can be traced back to a known code state.

**Source of truth.** The current version lives in two places that
`scripts/release.py` keeps in lockstep:

- `pyproject.toml` — `[tool.poetry] version`.
- `framework/version.py` — `PACKAGE_VERSION`.

**Runtime resolution.** `framework.version.code_version()` returns:

```
<PACKAGE_VERSION>+g<sha7>            # clean working tree
<PACKAGE_VERSION>+g<sha7>.dirty      # uncommitted changes
<PACKAGE_VERSION>                    # not in a git repo
```

`main.py` and `grid_search.py` log this at startup, and the value is
persisted in every run's `experiment_data.json` (`code_version` field)
and surfaced in the analytics summary table. Print the current value
without running anything via:

```bash
python main.py --version
```

Use the recorded `code_version` to check out the exact code that
produced a given run:

```bash
git checkout <sha7>
```

**Cutting a release.** From a clean working tree on `main`:

```bash
python scripts/release.py 0.2.0
git push origin main --tags
```

The script bumps `pyproject.toml` + `framework/version.py`, promotes
`## [Unreleased]` in `CHANGELOG.md` to `## [0.2.0] - YYYY-MM-DD`, commits
as `Release v0.2.0`, and creates an annotated `v0.2.0` tag. Until a
tag is pushed, all runs report a development version
(`0.2.0+g<sha>...`) — that's intentional: the SHA already pins the
state, the tag just makes it human-quotable.

Version numbers follow [SemVer](https://semver.org/): bump **patch**
for bug fixes, **minor** for backward-compatible features, **major**
for breaking changes to configs, weight-file format, or experiment
data on disk.

---

## Policies

Framework policies live in `framework/policies.py`, with additional game-specific policies under `games/<name>/`. All inherit `BasePolicy`. Active policy set via `policy_type` in `training_params.yaml`. The `BasePolicy` method contract (`__call__` / `update` / `save` / trainer-state, the weight-file format, and the "missing key → 0.0" migration rule) is documented in [`docs/framework/policies.md`](docs/framework/policies.md).

| `policy_type` | Class | Algorithm |
|---|---|---|
| `hill_climbing` | `WeightedLinearPolicy` | Mutate-and-keep. Includes probe + cold-start phases. |
| `neural_net` | `NeuralNetPolicy` | MLP (pure numpy). Mutate-and-keep on network weights. |
| `epsilon_greedy` | `EpsilonGreedyPolicy` | Tabular Q-learning, ε-greedy exploration, ε decays per episode. |
| `ucb_q` | `UCBQPolicy` | Tabular UCB1 online Q-learner (renamed from `mcts`; **not** tree search — no env cloning, builds Q/count tables over real episodes). |
| `genetic` | `GeneticPolicy` | Population of `WeightedLinearPolicy` instances. Evolutionary selection + crossover + mutation. |
| `cmaes` | `CMAESPolicy` | `(μ/μ_w, λ)-CMA-ES` (Hansen 2016) over flat `WeightedLinearPolicy` weights. Automatic step-size + covariance adaptation. |
| `neural_dqn` | `NeuralDQNPolicy` | Deep Q-learning with replay buffer + target network. Double-DQN target, Huber loss, and gradient clipping on by default; optional Dueling architecture (see DQN knobs below). |
| `reinforce` | `REINFORCEPolicy` | Monte Carlo policy gradient (optional running-mean baseline). |
| `lstm` | `LSTMEvolutionPolicy` | Recurrent (LSTM) policy, trained by evolutionary search over network weights. |
| `alphazero_mcts` | `AlphaZeroMCTSPolicy` | **Real** model-based MCTS (pure numpy): PUCT tree search expanded by cloning + stepping the env, guided by a policy/value net trained from self-play. Needs a cloneable simulator — see below. |

**Gradient deep-RL policies (Stable-Baselines3-backed).** These wrap SB3 /
SB3-Contrib algorithms behind `BasePolicy` and share `LOOP_TYPE = "sb3"` (SB3
owns its own training loop, driven by `framework/sb3_support.py`).  They live in
`framework/sb3_policies.py`, register at framework load, and import the heavy
SB3 / torch stack lazily.  Install with `poetry install --with deep_rl`.

| `policy_type` | Class | Algorithm |
|---|---|---|
| `ppo` | `PPOPolicy` | Proximal Policy Optimization (on-policy, clipped surrogate + GAE). |
| `a2c` | `A2CPolicy` | Advantage Actor-Critic (synchronous on-policy). |
| `sac` | `SACPolicy` | Soft Actor-Critic (off-policy, max-entropy; **continuous action only**). |
| `td3` | `TD3Policy` | Twin Delayed DDPG (off-policy deterministic; **continuous action only**). |
| `qr_dqn` | `QRDQNPolicy` | Quantile-Regression DQN (distributional value; SB3-Contrib). Wraps the env's `Box` into a `Discrete` index over `discrete_actions`. |
| `recurrent_ppo` | `RecurrentPPOPolicy` | PPO with an LSTM policy (gradient-trained recurrence; SB3-Contrib). The gradient counterpart to the ES-trained `lstm`. |

SB3 policy budget is governed by `policy_params.total_timesteps` (default
`n_sims × steps_per_sim`, `steps_per_sim` default `1000`); one
`GreedySimResult` is recorded per completed episode, and the trained model is
saved next to `policy_weights.yaml` as `*_sb3_model.zip`.  All SB3 policies are
gated off SC2 (its multi-head `[fn_idx, x, y, queue]` action encoding is not a
plain `Box`/`Discrete`) — use the `sc2_`-prefixed policies there.

`alphazero_mcts` is gated off every current game (`compatible_with` returns
`False` for TMNF/SC2/TORCS/CarRacing/BeamNG/Assetto/Rocket League/iRacing)
because their envs bind to live processes/sockets and cannot be cloned for tree
expansion.  To enable it on a game, give that game's env a `clone()` method (or
make it `copy.deepcopy`-able) and drop it from
`framework.alphazero._NON_CLONEABLE_GAMES`.

`SimplePolicy` = non-trainable hand-coded PD baseline (see `steering.py`).

### Choosing a policy

| Situation | Start with | Why / constraints |
|---|---|---|
| Fastest sanity check on a continuous `[steer, accel, brake]` game | `hill_climbing` | No `policy_params`, tiny weight files, and one real episode per greedy sim. Great for validating observations and rewards before spending more compute. |
| Linear control underfits but you still want a cheap evolutionary baseline | `neural_net` | Same mutate-and-keep loop as `hill_climbing`, but with a small MLP for non-linear feature interactions. |
| Cheap simulator and noisy / non-smooth rewards | `genetic` or `cmaes` | Population methods parallelise well and tolerate noisy returns, but budget scales with `population_size` (and `eval_episodes` when present). |
| Best sample-efficiency on a discrete action library | `neural_dqn` or `qr_dqn` | Replay + target networks reuse data far better than evolutionary search; size these runs in env steps, not just episodes. |
| Simple episodic policy-gradient baseline on a discrete action set | `reinforce` | Fewer moving parts than DQN; useful when you want a small, easy-to-reason-about gradient baseline. |
| Partial observability / memory matters | `lstm` or `recurrent_ppo` | `lstm` if simulations are cheap and ES search is acceptable; `recurrent_ppo` if sample efficiency matters more. |
| Native continuous `Box` control with gradient RL | `sac` or `td3` | SB3-backed continuous-control baselines; `sac` is usually the safer first try, `td3` the deterministic alternative. |
| Discrete control with a mature on-policy SB3 baseline | `ppo` | Strong first choice when you want a standard library implementation instead of the pure-numpy policies. |
| Tiny discrete state space after coarse binning | `epsilon_greedy` or `ucb_q` | Tabular methods are only practical when `n_bins` stays small enough that the table is still visitable. |
| Cloneable deterministic simulator and you explicitly want planning | `alphazero_mcts` | Only appropriate when the env can be cloned cheaply; all currently supported games are gated off today. |
| Any SC2 run | `sc2_genetic`, `sc2_hierarchical`, `sc2_cmaes`, `sc2_reinforce`, `sc2_neural_dqn`, `sc2_lstm`, `sc2_cnn`, or `sc2_neural_net` | The generic continuous-action framework policies (`hill_climbing`, `neural_net`, `genetic`, `cmaes`, `neural_dqn`, `reinforce`, `lstm`) and all SB3 policies are structurally incompatible with SC2's `[fn_idx, x, y, queue]` action encoding. Only the SC2-native wrappers (plus tabular `epsilon_greedy` / `ucb_q`) should be considered there. |

### Sizing a run

- **Mutate-and-keep (`hill_climbing`, `neural_net`)**: greedy cost is `n_sims`
  full episodes. A fresh run can additionally spend up to
  `6 + cold_restarts × cold_sims` episodes in probe + cold-start.
- **Tabular + episodic gradient (`epsilon_greedy`, `ucb_q`, `reinforce`)**:
  roughly `n_sims` full episodes. The tuning knobs change how much signal you
  extract from each episode, not the raw episode count.
- **Population methods (`genetic`, `cmaes`)**:
  `total_episodes = n_sims × population_size × eval_episodes`.
- **LSTM evolution (`lstm`)**:
  `total_episodes = n_sims × population_size`.
- **DQN-family (`neural_dqn`, `qr_dqn`)**: real episode count is still about
  `n_sims`, but the important budget is **environment steps**. Keep
  `epsilon_decay_steps`, `min_replay_size`, and `target_update_freq` in the same
  ballpark as `total_env_steps ≈ n_sims × average_episode_steps`.
- **SB3 policies (`ppo`, `a2c`, `sac`, `td3`, `qr_dqn`, `recurrent_ppo`)**:
  training budget is `total_timesteps` if set, else
  `n_sims × steps_per_sim` (`steps_per_sim` defaults to `1000`).
- **AlphaZero MCTS**: you play `n_sims` real self-play games, but each action
  also runs `n_simulations` clone-and-search rollouts, so wall-clock cost scales
  much more like `episode_steps × n_simulations`.

### WeightedLinearPolicy

Three independent linear heads (steer, accel, brake), each
`dot(weights, normalised_obs)`. Weights stored in YAML. `mutated(scale, share)`
adds Gaussian noise to random `share` fraction of weights; features are
pre-normalised so all contribute equally per mutation step. Auto-migrates
existing weight files when new observation features are added.

**Hyperparams** (in `policy_params`): none.

**Reasonable choices / tune first**: the useful knobs live in
`training_params.yaml`, not `policy_params`. Start with `mutation_scale` around
`0.02–0.1`, `mutation_share` around `0.3–1.0`, and `n_sims` around `50–200`.
If this cannot beat a hand-coded baseline, fix the reward or observation design
before moving to a larger policy.

### NeuralNetPolicy

Small MLP policy trained via the same mutate-and-keep loop as
`WeightedLinearPolicy`.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `hidden_sizes` | `[16, 16]` | Hidden layer widths of the policy MLP |

**Reasonable choices / tune first**: start with `[16, 16]`, move to `[32, 32]`
or `[64, 64]` when the linear policy is clearly underfitting, and avoid going
much larger unless you also raise the episode budget. This is still a
one-episode-per-mutation policy.

### EpsilonGreedyPolicy

Tabular Q-learning with ε-greedy exploration over the game's discrete action
set. Exploration decays **per episode**.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `epsilon` | `1.0` | Initial exploration rate |
| `n_bins` | `3` | Observation bins per dimension for state discretisation |
| `epsilon_decay` | `0.995` | Multiplicative ε decay applied at episode end |
| `epsilon_min` | `0.05` | Floor on ε after decay |
| `alpha` | `0.1` | Q-learning step size |
| `gamma` | `0.99` | Discount factor |

**Reasonable choices / tune first**: tune `n_bins` first because it controls
whether the table is even usable; `3–7` is the practical range before the state
space explodes. Then tune exploration: `epsilon: 0.8–1.0`,
`epsilon_decay: 0.99–0.999`, `epsilon_min: 0.02–0.1`. Keep `alpha` in
`0.05–0.2` and `gamma` in `0.95–0.995`.

### UCBQPolicy

Tabular online Q-learning with UCB1 action selection instead of ε-greedy.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `c` | `1.41` | UCB1 exploration constant |
| `alpha` | `0.1` | Q-learning step size |
| `gamma` | `0.99` | Discount factor |
| `n_bins` | `3` | Observation bins per dimension for state discretisation |

**Reasonable choices / tune first**: again, make sure `n_bins` (`3–7`) keeps
the table size manageable. Then tune `c`: `0.5–2.0` is the useful range, with
smaller values exploiting earlier and larger values forcing more coverage.
`alpha` and `gamma` usually stay in the same `0.05–0.2` / `0.95–0.995` ranges
as `epsilon_greedy`.

### GeneticPolicy

Maintains a population of `WeightedLinearPolicy` instances. Each generation:
evaluate all individuals (`eval_episodes` episodes each, averaged), keep top
`elite_k` unchanged, breed the rest via uniform crossover between two random
elites + mutation. Best individual ever seen = champion, saved to YAML for
inference.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `population_size` | `10` | Individuals evaluated per generation |
| `elite_k` | `3` | Top individuals copied unchanged into the next generation |
| `mutation_scale` | `0.1` | Std-dev of Gaussian mutation noise |
| `mutation_share` | `1.0` | Fraction of weights perturbed per child |
| `eval_episodes` | `1` | Episodes per individual per generation (averaged for fitness) |

**Reasonable choices / tune first**: tune `population_size` and
`mutation_scale` first. `population_size: 10–40` is typical; raise it when the
reward is noisy or crossover quality matters more than wall clock. Keep
`elite_k` small — about `10–20%` of the population, and always less than
`population_size` — so you preserve good individuals without collapsing
diversity. `mutation_scale: 0.02–0.15`, `mutation_share: 0.25–1.0`, and
`eval_episodes: 1–3` cover most use cases.

### CMAESPolicy

Implements `(μ/μ_w, λ)-CMA-ES` (Hansen 2016) over the concatenated
`[steer | accel | brake]` weight vector of `WeightedLinearPolicy`
(~63 dimensions for the base observation space).

**Training loop** (called from `_greedy_loop_cmaes`):
1. `sample_population()` — draw λ offspring from `N(mean, σ²·C)` using cached eigen-factorization `C = B D² Bᵀ`
2. Evaluate each offspring for `eval_episodes` episodes → average reward → reward vector
3. `update_distribution(rewards)` — weighted mean recombination (top μ = λ//2 elites), cumulative step-size adaptation (CSA) for σ, and rank-1 + rank-μ covariance updates

**Key properties**: `population_size` (λ), `sigma` (current σ), `champion_reward`.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `population_size` | `20` | λ — offspring sampled per generation |
| `initial_sigma` | `0.3` | Starting step size (adapts via CSA each generation) |
| `eval_episodes` | `1` | Episodes per individual per generation (averaged for fitness) |

`n_sims` controls generations; total episodes = `n_sims × population_size × eval_episodes`. No `mutation_scale` tuning is needed — σ adapts automatically.

> **Budget note**: `eval_episodes > 1` multiplies total episode count by that factor. For `GeneticPolicy` the same formula applies: `n_sims × population_size × eval_episodes`. Keep `eval_episodes: 1` in grid-search templates unless you are explicitly studying variance reduction vs. episode budget.

`save()` writes the champion in `WeightedLinearPolicy` YAML format so analytics,
weight heatmaps, and inference work unchanged.

**Reasonable choices / tune first**: `initial_sigma` is the first knob to move.
Use `0.1–0.2` when resuming from a decent champion or when rewards are already
smooth, and `0.3–0.5` when you need broader early exploration. Keep
`population_size` around `12–40`; larger populations help on noisy tasks but
scale cost linearly. Prefer `eval_episodes: 1` unless one lucky episode often
mis-ranks offspring.

### NeuralDQNPolicy

MLP Q-network (pure numpy) with experience replay and a periodically-synced
target network. Action space is the 9-element discrete set
(`DISCRETE_ACTIONS`). ε-greedy exploration decays linearly from
`epsilon_start` to `epsilon_end` over `epsilon_decay_steps` environment steps.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `hidden_sizes` | `[64, 64]` | Hidden layer widths of the Q-network MLP |
| `replay_buffer_size` | `10000` | Max transitions stored in the replay buffer |
| `batch_size` | `64` | Transitions sampled per gradient update |
| `min_replay_size` | `500` | Buffer must reach this size before training starts |
| `target_update_freq` | `200` | Steps between copying online weights → target network |
| `learning_rate` | `0.001` | Adam-style gradient step size |
| `epsilon_start` | `1.0` | Initial exploration rate |
| `epsilon_end` | `0.05` | Final exploration rate |
| `epsilon_decay_steps` | `5000` | Steps over which ε decays linearly |
| `gamma` | `0.99` | Discount factor |
| `double_dqn` | `true` | Use a Double-DQN target (online net selects the bootstrap action, target net evaluates it). `false` = vanilla `max_a Q_target`. |
| `dueling` | `false` | Dueling architecture: split the head into a value stream + advantage stream, aggregated as `Q = V + (A − mean A)`. Opt-in (changes the weight-file shape). |
| `huber_loss` | `true` | Huber (smooth-L1) loss: clamp the TD residual to ±`huber_kappa` for a bounded gradient. `false` = MSE. |
| `huber_kappa` | `1.0` | Huber band half-width. |
| `max_grad_norm` | `10.0` | Global-norm gradient clip applied before the Adam step. `null` disables clipping. |

> `double_dqn`, `huber_loss`, and `max_grad_norm` default to the upgraded
> (SB3-aligned) behaviour; `dueling` stays opt-in. Set `double_dqn: false`,
> `huber_loss: false`, `max_grad_norm: null` to recover the old vanilla DQN.
> All apply to `neural_dqn` (TMNF) and `sc2_neural_dqn` too.

**Reasonable choices / tune first**: size this run in **env steps**, not just
episodes. A good first pass is to keep `epsilon_decay_steps` at roughly
`25–75%` of the total env-step budget, `min_replay_size` at about `1–10%` of
`replay_buffer_size`, and `target_update_freq` in the same order of magnitude
as one to a few episode lengths (`200–2000` steps for short racing episodes).
Start with `hidden_sizes` around `[64, 64]` or `[128, 128]`,
`batch_size: 32–128`, `replay_buffer_size: 10_000–100_000`,
`learning_rate: 1e-4–3e-3`, and `epsilon_end: 0.02–0.1`. Only enable
`dueling` after the vanilla upgraded defaults are stable.

### REINFORCEPolicy

Monte Carlo policy-gradient over the discrete action set. Collect full
episodes, compute discounted returns, optionally subtract a running-mean
baseline, then update the softmax policy network via gradient ascent.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `hidden_sizes` | `[64, 64]` | Hidden layer widths of the policy MLP |
| `learning_rate` | `0.001` | Gradient step size |
| `gamma` | `0.99` | Discount factor for return computation |
| `entropy_coeff` | `0.01` | Entropy regularisation weight (encourages exploration) |
| `baseline` | `"running_mean"` | Return baseline: `"running_mean"` or `"none"` |

**Reasonable choices / tune first**: tune `learning_rate` first
(`3e-4–3e-3`), then `entropy_coeff` (`0.0–0.05`) to keep exploration alive
without washing out the reward signal. `[64, 64]` is a good default hidden
stack; go to `[128, 128]` only when you have enough episodes to support it.
`baseline: "running_mean"` is the sensible default unless you are debugging the
raw Monte-Carlo estimator.

### LSTMEvolutionPolicy

LSTM recurrent policy trained by evolutionary search over flattened network
weights. The hidden state resets each episode; at each step the LSTM receives
the current observation and emits logits over the discrete action set.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `hidden_size` | `32` | LSTM hidden/cell state dimensionality |
| `population_size` | `20` | λ — offspring evaluated per generation |
| `initial_sigma` | `0.05` | Starting perturbation scale (smaller than CMAESPolicy because the LSTM weight space is larger) |

**Reasonable choices / tune first**: only pay for this policy when memory is
actually useful. `hidden_size: 16–64` covers most cases; larger hidden states
make the search space much bigger. Keep `population_size` around `10–30` and
`initial_sigma` around `0.02–0.1`; if the search explodes, lower
`initial_sigma` before shrinking the hidden state.

### AlphaZeroMCTSPolicy

AlphaZero-style policy/value MCTS over a **cloneable** simulator. Every action
selection expands a PUCT tree using cloned env states, then trains a shared
policy/value network from self-play targets.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `n_simulations` | `32` | Tree-search expansions per real env decision |
| `c_puct` | `1.5` | PUCT exploration constant |
| `gamma` | `0.99` | Discount factor used for value targets |
| `hidden_sizes` | `[64, 64]` | Hidden layer widths of the shared policy/value MLP |
| `learning_rate` | `0.001` | Gradient step size for the policy/value net |
| `temperature` | `1.0` | Visit-count temperature when sampling self-play actions |
| `dirichlet_alpha` | `0.3` | Dirichlet prior concentration for root exploration noise |
| `dirichlet_frac` | `0.25` | Fraction of root prior replaced by Dirichlet noise |
| `value_loss_coef` | `1.0` | Weight on the value-loss term during training |
| `train_batch_size` | `32` | Minibatch size for self-play updates |
| `seed` | `null` | RNG seed for reproducibility |

**Reasonable choices / tune first**: tune `n_simulations` first because it is
the dominant wall-clock multiplier; `16–128` is the practical range. Then tune
`c_puct` (`1.0–2.5`) and `temperature` (`0.3–1.0`) depending on how sharp you
want self-play exploration to be. Keep the net modest (`[64, 64]` or
`[128, 128]`) unless the cloneable simulator is cheap enough to feed it.

### PPOPolicy

Stable-Baselines3-backed PPO. The framework now hands training over to SB3's
own loop (`LOOP_TYPE = "sb3"`), so the budget is **env steps**, not one PPO
update per episode. Works on the framework's non-SC2 games with native
SB3-compatible action spaces (for example, TMNF's `Box` action space).

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `total_timesteps` | `n_sims × steps_per_sim` | Total SB3 env-step budget; overrides `steps_per_sim` when set explicitly |
| `steps_per_sim` | `1000` | Fallback env-step budget per `n_sims` unit |
| `learning_rate` | `SB3 default` | Passed through to SB3 only when set |
| `gamma` | `SB3 default` | Discount factor override |
| `hidden_sizes` | `SB3 default` | Hidden layer widths override for `policy_kwargs.net_arch` |
| `seed` | `null` | RNG seed override |
| `verbose` | `0` | SB3 logging verbosity |
| `n_steps` | `SB3 default` | Rollout length per update |
| `batch_size` | `SB3 default` | Minibatch size |
| `n_epochs` | `SB3 default` | Optimisation passes per rollout |
| `gae_lambda` | `SB3 default` | GAE smoothing |
| `clip_range` | `SB3 default` | PPO clipping epsilon |
| `ent_coef` | `SB3 default` | Entropy-bonus coefficient |
| `vf_coef` | `SB3 default` | Critic-loss coefficient |

**Reasonable choices / tune first**: `ppo` is the safest first SB3 baseline.
Tune `total_timesteps` first, then `n_steps` and
`batch_size`. Useful override ranges are `n_steps: 256–2048`,
`batch_size: 64–256`, `n_epochs: 4–10`, `gae_lambda: 0.95–0.99`, and
`clip_range: 0.1–0.3`. Leave `learning_rate` on the SB3 default unless you have
a reason to try `1e-4–3e-4`.

### A2CPolicy

Stable-Baselines3 A2C. Lighter-weight than PPO, but usually a weaker default.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `total_timesteps` | `n_sims × steps_per_sim` | Total SB3 env-step budget; overrides `steps_per_sim` when set explicitly |
| `steps_per_sim` | `1000` | Fallback env-step budget per `n_sims` unit |
| `learning_rate` | `SB3 default` | Passed through to SB3 only when set |
| `gamma` | `SB3 default` | Discount factor override |
| `hidden_sizes` | `SB3 default` | Hidden layer widths override for `policy_kwargs.net_arch` |
| `seed` | `null` | RNG seed override |
| `verbose` | `0` | SB3 logging verbosity |
| `n_steps` | `SB3 default` | Rollout length before each synchronous update |
| `gae_lambda` | `SB3 default` | GAE smoothing |
| `ent_coef` | `SB3 default` | Entropy-bonus coefficient |
| `vf_coef` | `SB3 default` | Critic-loss coefficient |

**Reasonable choices / tune first**: prefer `ppo` unless you want a smaller,
faster on-policy baseline. Keep `n_steps` fairly short (`5–32`) and use
`gae_lambda: 0.9–0.99`, `ent_coef: 0.0–0.01`, and `vf_coef: 0.5–1.0`. If A2C is
unstable, increasing the total step budget usually helps more than widening the
network.

### SACPolicy

Stable-Baselines3 Soft Actor-Critic for native continuous `Box` action spaces.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `total_timesteps` | `n_sims × steps_per_sim` | Total SB3 env-step budget; overrides `steps_per_sim` when set explicitly |
| `steps_per_sim` | `1000` | Fallback env-step budget per `n_sims` unit |
| `learning_rate` | `SB3 default` | Passed through to SB3 only when set |
| `gamma` | `SB3 default` | Discount factor override |
| `hidden_sizes` | `SB3 default` | Hidden layer widths override for `policy_kwargs.net_arch` |
| `seed` | `null` | RNG seed override |
| `verbose` | `0` | SB3 logging verbosity |
| `buffer_size` | `SB3 default` | Replay-buffer capacity |
| `batch_size` | `SB3 default` | Minibatch size |
| `tau` | `SB3 default` | Target-network Polyak averaging factor |
| `train_freq` | `SB3 default` | Environment steps between optimisation phases |
| `learning_starts` | `SB3 default` | Warmup steps before updates start |
| `ent_coef` | `SB3 default` | Entropy coefficient (`"auto"` in SB3 by default) |

**Reasonable choices / tune first**: `sac` is the first gradient-RL policy to
try on continuous control. Spend your budget on `total_timesteps` and
`buffer_size` before changing architecture. Good override ranges are
`buffer_size: 50_000–1_000_000`, `batch_size: 128–512`, `tau: 0.005–0.02`,
`train_freq: 1–16`, and `learning_starts: 1_000–10_000`. Leave `ent_coef` on
SB3's automatic default unless you have a strong reason not to.

### TD3Policy

Stable-Baselines3 Twin Delayed DDPG for native continuous `Box` action spaces.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `total_timesteps` | `n_sims × steps_per_sim` | Total SB3 env-step budget; overrides `steps_per_sim` when set explicitly |
| `steps_per_sim` | `1000` | Fallback env-step budget per `n_sims` unit |
| `learning_rate` | `SB3 default` | Passed through to SB3 only when set |
| `gamma` | `SB3 default` | Discount factor override |
| `hidden_sizes` | `SB3 default` | Hidden layer widths override for `policy_kwargs.net_arch` |
| `seed` | `null` | RNG seed override |
| `verbose` | `0` | SB3 logging verbosity |
| `buffer_size` | `SB3 default` | Replay-buffer capacity |
| `batch_size` | `SB3 default` | Minibatch size |
| `tau` | `SB3 default` | Target-network Polyak averaging factor |
| `train_freq` | `SB3 default` | Environment steps between optimisation phases |
| `learning_starts` | `SB3 default` | Warmup steps before updates start |
| `policy_delay` | `SB3 default` | Critic updates per actor update |

**Reasonable choices / tune first**: choose `td3` when you want a deterministic
continuous-control baseline after trying `sac`. The same `buffer_size`,
`batch_size`, `tau`, and `learning_starts` ranges apply; `policy_delay: 2–4` is
the main extra knob. If TD3 is brittle, increase total steps before increasing
network size.

### QRDQNPolicy

SB3-Contrib QR-DQN. The framework wraps the env's continuous action `Box` into a
discrete index over `discrete_actions`, then learns a distributional value
function over that discrete set.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `total_timesteps` | `n_sims × steps_per_sim` | Total SB3 env-step budget; overrides `steps_per_sim` when set explicitly |
| `steps_per_sim` | `1000` | Fallback env-step budget per `n_sims` unit |
| `learning_rate` | `SB3 default` | Passed through to SB3 only when set |
| `gamma` | `SB3 default` | Discount factor override |
| `hidden_sizes` | `SB3 default` | Hidden layer widths override for `policy_kwargs.net_arch` |
| `seed` | `null` | RNG seed override |
| `verbose` | `0` | SB3 logging verbosity |
| `buffer_size` | `SB3 default` | Replay-buffer capacity |
| `batch_size` | `SB3 default` | Minibatch size |
| `learning_starts` | `SB3 default` | Warmup steps before updates start |
| `target_update_interval` | `SB3 default` | Steps between target-network syncs |
| `train_freq` | `SB3 default` | Environment steps between optimisation phases |
| `exploration_fraction` | `SB3 default` | Fraction of total training spent decaying ε |
| `exploration_final_eps` | `SB3 default` | Final ε after decay |
| `n_quantiles` | `SB3 default` | Number of value-distribution quantiles |

**Reasonable choices / tune first**: use `qr_dqn` when you want an SB3-managed
value learner over a discrete action library. The most important knobs are
`total_timesteps`, `exploration_fraction`, and `target_update_interval`.
Reasonable override ranges are `buffer_size: 50_000–500_000`,
`batch_size: 32–256`, `learning_starts: 1_000–10_000`,
`target_update_interval: 500–5_000`, `exploration_fraction: 0.1–0.5`,
`exploration_final_eps: 0.01–0.1`, and `n_quantiles: 25–200`.

### RecurrentPPOPolicy

SB3-Contrib Recurrent PPO with an LSTM policy network. This is the
gradient-trained counterpart to the ES-trained `lstm`.

**Hyperparams** (in `policy_params`):

| Param | Default | Description |
|---|---|---|
| `total_timesteps` | `n_sims × steps_per_sim` | Total SB3 env-step budget; overrides `steps_per_sim` when set explicitly |
| `steps_per_sim` | `1000` | Fallback env-step budget per `n_sims` unit |
| `learning_rate` | `SB3 default` | Passed through to SB3 only when set |
| `gamma` | `SB3 default` | Discount factor override |
| `hidden_sizes` | `SB3 default` | Hidden layer widths override for `policy_kwargs.net_arch` |
| `seed` | `null` | RNG seed override |
| `verbose` | `0` | SB3 logging verbosity |
| `n_steps` | `SB3 default` | Rollout length per update |
| `batch_size` | `SB3 default` | Minibatch size |
| `n_epochs` | `SB3 default` | Optimisation passes per rollout |
| `gae_lambda` | `SB3 default` | GAE smoothing |
| `clip_range` | `SB3 default` | PPO clipping epsilon |
| `ent_coef` | `SB3 default` | Entropy-bonus coefficient |

**Reasonable choices / tune first**: pick this when partial observability
matters and you still want gradient training. Start with the same PPO-style
budget logic, but keep `n_steps` a bit shorter (`128–1024`) so the recurrent
state updates more often. `batch_size: 64–256`, `n_epochs: 4–10`,
`gae_lambda: 0.95–0.99`, and `clip_range: 0.1–0.3` are sensible override
ranges.

---

## Training Phases

Only `hill_climbing` runs probe and cold-start. All others go straight to greedy.

**1. Probe** (no weights file, or `--re-initialize`)
Runs 6 fixed-action episodes (brake/accel × left/straight/right, `probe_s` seconds each). Establishes reward floor for cold-start comparison.

**2. Cold-start search**
Up to `cold_restarts` rounds of random-init hill-climbing, `cold_sims` simulations each. Stops early if any restart beats probe floor. Best policy saved, used as greedy starting point.

**3. Greedy optimisation**
`n_sims` iterations (or generations for `genetic`). Best weights saved after each improvement.

---

## Configuration

Each game keeps its own master configs under `games/<game>/config/`. The
tables below describe the **TMNF** masters
(`games/tmnf/config/training_params.yaml` and `reward_config.yaml`); other
games define their own keys — see the per-game `README.md`. The SC2 keys
are documented in the StarCraft 2 section below.

### `games/tmnf/config/training_params.yaml`

| Parameter | Default | Description |
|---|---|---|
| `track` | `a03_centerline` | Stem of `.npy` file in `tracks/` |
| `speed` | `10.0` | Game speed multiplier (TMInterface max 10×) |
| `in_game_episode_s` | `30.0` | In-game seconds per episode |
| `n_sims` | `100` | Greedy simulations / generations |
| `mutation_scale` | `0.05` | Std-dev of Gaussian noise per mutation |
| `mutation_share` | `1.0` | Fraction of weights perturbed per mutation (1.0 = all) |
| `probe_s` | `15.0` | In-game seconds per probe action run |
| `cold_restarts` | `20` | Max random restarts in cold-start search |
| `cold_sims` | `5` | Hill-climb sims per cold-start restart |
| `n_lidar_rays` | `8` | LIDAR rays appended to observation (0 = disabled) |
| `policy_type` | `genetic` | Algorithm (see Policies table above) |
| `policy_params` | `{}` | Type-specific hyperparams |

### `games/tmnf/config/reward_config.yaml`

| Parameter | Default | Description |
|---|---|---|
| `progress_weight` | `10000.0` | Primary signal — proportional to track progress delta |
| `centerline_weight` | `-0.1` | Lateral offset penalty coefficient |
| `centerline_exp` | `2.0` | Exponent for centerline penalty (2 = quadratic) |
| `speed_weight` | `0.05` | Bonus per m/s (tie-breaker) |
| `step_penalty` | `-0.05` | Per-tick time cost |
| `finish_bonus` | `5000.0` | One-time bonus at `track_progress >= 1.0` |
| `finish_time_weight` | `-5.0` | Penalty/bonus relative to `par_time_s` |
| `par_time_s` | `60.0` | Reference lap time in seconds |
| `accel_bonus` | `0.5` | Flat reward per step when throttle pressed |
| `airborne_penalty` | `-1.0` | Applied when ≤1 wheel contact AND `vertical_offset <= 0` |
| `lidar_wall_weight` | `-5.0` | Wall proximity: `weight * (1 - min_ray)^2` |
| `crash_threshold_m` | `25.0` | Terminates episode when `|lateral_offset| > threshold` |

---

## RL Environment (`rl/env.py`)

The game-agnostic `BaseGameEnv` contract (`reset`/`step`, `terminated` vs
`truncated`, the `info` dict, the optional episode-time-limit hooks, and
threading expectations) is documented in
[`docs/framework/base_env.md`](docs/framework/base_env.md); the
reward-side contract is in
[`docs/framework/reward.md`](docs/framework/reward.md). The TMNF-specific
spaces below are one concrete instance of that contract.

### Observation (15 + n_lidar_rays floats, float32)

Defined in `obs_spec.py` — single source of truth for feature names, scales, descriptions. The generic `ObsSpec` / `ObsDim` contract (naming, normalisation, auto-migration) is documented in [`docs/framework/obs_spec.md`](docs/framework/obs_spec.md).

| Index | Name | Scale | Description |
|-------|------|-------|-------------|
| 0 | `speed_ms` | 50.0 | Vehicle speed in m/s |
| 1 | `lateral_offset_m` | 5.0 | Metres from centreline (neg=left, pos=right) |
| 2 | `vertical_offset_m` | 2.0 | Metres above (+) / below (-) centreline |
| 3 | `yaw_error_rad` | π | Track heading minus car heading, [−π, π] |
| 4 | `pitch_rad` | 0.3 | Nose-up/down rotation |
| 5 | `roll_rad` | 0.3 | Tilt left/right |
| 6 | `track_progress` | 1.0 | Fraction of track completed, [0, 1] |
| 7 | `turning_rate` | 65536.0 | Raw TMInterface steer value, ±65536 |
| 8–11 | `wheel_N_contact` | 1.0 | Ground contact per wheel (0 or 1) |
| 12–14 | `angular_vel_N` | 5.0 | Angular velocity x/y/z (rad/s) |
| 15+ | `lidar_i` | 1.0 | Wall distance rays ~[0, 1] (if `n_lidar_rays > 0`) |

### Action Space

`Box([-1, 0, 0], [1, 1, 1], shape=(3,), dtype=float32)`

| Index | Name | Range | Notes |
|-------|------|-------|-------|
| 0 | steer | [−1, 1] | Maps to [−65536, 65536] in-game |
| 1 | accel | [0, 1] | Thresholded at 0.5 → bool |
| 2 | brake | [0, 1] | Thresholded at 0.5 → bool; can fire simultaneously with accel |

Policies using the Discrete(25) abstraction ({full brake, half brake, coast, half accel, full accel} × {full left, half left, straight, half right, full right}) convert internally via `ACTIONS` in `clients/rl_client.py`.

### Termination

- **Finished:** `track_progress >= 1.0`
- **Crashed:** `|lateral_offset| > crash_threshold_m`
- **Truncated:** elapsed time exceeded

### Episode Warmup

First 100 steps force full-throttle straight (`accel + straight, no brake`) regardless of policy. Covers braking-start phase so weights/Q-tables not updated during forced behaviour.

---

## LIDAR (`lidar.py`)

Set `n_lidar_rays > 0` to append wall-distance observations. `LidarSensor`:
1. Captures game window via MSS
2. Converts to 128×32 binary edge image (grayscale → threshold → Canny → dilate → blur)
3. Raycasts `n_lidar_rays` evenly spaced angles from 0 to π, returning normalised distances ~[0, 1]

LIDAR rays appended to observation. All policies handle variable-length observations; `WeightedLinearPolicy` auto-migrates weight files to add new keys (initialised to 0.0).

Requires `mss`, `opencv-python`, `pywin32`.

---

## Grid Search (`grid_search.py`)

```bash
python grid_search.py games/<game>/config/my_grid.yaml
```

Ready-made templates ship per game (e.g.
`games/sc2/config/grid_search_template.yaml`,
`games/torcs/config/gs_genetic.yaml`). Pick the target game with `--game`
(choices: `tmnf`, `beamng`, `car_racing`, `torcs`, `sc2`, `rocket_league`,
`iracing`).

Set any param to list to sweep it:

```yaml
base_name: "gs_v1"
training_params:
  mutation_scale: [0.05, 0.1, 0.2]   # 3-way sweep
  n_sims: 50
reward_params:
  centerline_weight: [-0.1, -0.5]    # 2-way sweep
```

Creates one experiment per Cartesian-product combination (3 × 2 = 6 here). Names encode only varied params: `gs_v1__ms0.05__cw_n0.1`.

---

## Distributed training (`distributed/`)

Scale grid search across multiple Windows VMs by splitting combinations over coordinator + worker pool.

- `distributed/coordinator.py` — HTTP work-queue server. Bearer-token auth; heartbeat-based re-queue of stalled jobs.
- `distributed/worker.py` — polls `/work`, runs `train_rl()` locally against its TMInterface session, posts `ExperimentData` back to `/result`.
- `distributed/protocol.py` — `ComboSpec` / `ResultPayload` dataclasses + JSON (de)serialization shared by both sides.

Entry point: `python grid_search.py <config> --distribute` (coordinator mode). Workers launched independently on each VM.

**Local multi-instance.** `python grid_search.py <config> --distribute
--local-workers N` runs the coordinator and spawns `N` worker
subprocesses on the same host (one experiment per worker — inter-run
parallelism, distinct from SC2's intra-run `n_workers`). Worker launches
are staggered to avoid startup races; tune with `--local-worker-stagger`
(seconds, default `5.0`; `0` to disable) or the
`distribute.local_worker_stagger` config key.

**LAN safety / multi-machine home networks.** Distributed mode now defaults to
LAN-only request filtering: the coordinator accepts only loopback/private/
link-local source IPs. Configure `--bind-host` (or `distribute.bind_host`) to
choose the listening interface, and pass `--allow-non-lan` only when you
explicitly want public/non-LAN clients. The coordinator also defaults to
`--local-workers 1`, so the driver node contributes one local worker while
remote workers run on other machines.

**SC2 map-access gate.** When several PySC2 binaries boot on one host,
`SC2Env` construction is serialised through a cross-process gate enforcing
a minimum gap between map loads. Tunable via env vars:
`GAMER_AI_SC2_MAP_GAP_S` (gap seconds, default `5.0`; `0` disables) and
`GAMER_AI_SC2_MAP_LOCK_PATH` (custom timestamp-file path, mainly for tests).

---

## Infrastructure (Azure)

Three-stage Terraform stack under `infrastructure/` provisions distributed training fleet.

- `auth/` — service principal + role assignments.
- `remote_state/` — storage account for shared Terraform state.
- `environment/` — Windows 11 Pro VMs (1 coordinator + N workers), Key Vault for admin passwords, NSG allows RDP only from single configured IP.

See `infrastructure/README.md` for operational commands (plan/apply, start/stop/deallocate, worker scaling).

---

## Analytics (`analytics.py`)

Called automatically at end of each experiment/grid-search run. Writes plots and summary JSON to `experiments/<track>/<name>/results/`. Skipped phases produce no output files.

---

## Threading Model

TMInterface callback-driven (`on_run_step`); RL loop step-driven (`env.step()`). `RLClient` bridges with:
- `_action` queue (RL thread → game thread)
- `_state_queue` (game thread → RL thread, maxsize=1; drain before put)
- `_episode_ready` event (signals env reset complete)

Daemon keepalive thread keeps `iface.running` alive. `on_registered` sets event that `TMNFEnv.__init__` waits on before returning.

---

## StarCraft 2 (`games/sc2/`)

**Scope**: 7 standard PySC2 minigames and full 1v1 RL training against a built-in bot on any ladder map (e.g. `Simple64`). Fog-of-war belief machinery is deferred to a future issue.

### Setup

1. Install the Blizzard StarCraft 2 binary. On Linux use the [headless build](https://github.com/Blizzard/s2client-proto#linux-packages); on Windows/Mac the regular client works.
2. Set `SC2PATH` to the install root (Linux), or place the install in `~/StarCraftII/` (the PySC2 default).
3. Download the [PySC2 maps](https://github.com/Blizzard/s2client-proto#downloads) (mini-games + ladder maps) and unzip them into `Maps/` under the install root.
4. Install the optional sc2 dependency group:
   ```bash
   poetry install --with sc2
   ```

### Running

```bash
# Default minigame (MoveToBeacon)
python main.py myrun --game sc2

# Choose a different minigame: edit games/sc2/config/training_params.yaml
# (or the per-experiment copy under experiments/) and set map_name.

# 1v1 ladder training against a very_easy bot on Simple64:
#   1. Edit experiments/sc2_Simple64/<name>/training_params.yaml:
#        map_name: Simple64
#        in_game_episode_s: 600.0
#        policy_type: sc2_genetic    # or sc2_cmaes, sc2_reinforce, sc2_lstm
#   2. Edit experiments/sc2_Simple64/<name>/reward_config.yaml:
#        score_weight: 0.0
#        economy_weight: 0.001
#   3. Run:
python main.py myrun --game sc2 --no-interrupt

# Interactive human-vs-AI play (loads champion from a completed experiment):
python main.py myrun --game sc2 --play

# Evaluation: replay the champion against the bot for N episodes and report
# aggregate win rate / avg score / avg game length (no weight updates):
python main.py myrun --game sc2 --eval --num-episodes 10

# Behaviour cloning from replays (pre-train weights, then fine-tune):
python main.py myrun --game sc2 --bc --replay-dir /path/to/replays --bc-race terran
python main.py myrun --game sc2   # fine-tune from BC weights
```

The first run creates `experiments/sc2_<map>/<name>/` and copies both master configs in. Edit the experiment-local copies to tune without affecting other runs.

`--play` launches a two-player PySC2 session; you control one side via the SC2 UI while the trained champion policy drives the other. No weight updates occur.

`--eval` (SC2 only) loads the champion and runs it against the AI bot for `--num-episodes` episodes, reporting aggregate statistics. `--bot-difficulty` overrides the opponent difficulty and `--eval-speed` overrides `step_mul` for the eval run.

`--bc` runs SC2 offline behaviour cloning into the experiment directory.
SC2 reads `.SC2Replay` files (with `--replay-dir`, `--bc-race`,
`--bc-player`).  The framework BC contract and TMNF details are in the
game-agnostic `--bc` section under **Running** above and in
[`docs/framework/bc_adapter.md`](docs/framework/bc_adapter.md).

### Config knobs

| Key | Default | Notes |
|---|---|---|
| `map_name` | `MoveToBeacon` | Any of the 7 minigame names or a ladder map (e.g. `Simple64`). |
| `agent_race` | `random` | `protoss` / `terran` / `zerg` / `random`. |
| `bot_difficulty` | `very_easy` | Only for ladder maps; ignored for minigames. |
| `step_mul` | `1` | Game ticks per env step. `1` = finest granularity; the policy can act every engine tick (22.4 ticks/s). Higher values coarsen control and raise the effective APM ceiling (e.g. `step_mul=8` ≈ 168 game-time APM max). With `step_mul=1`, `max_apm` is the sole throttle on action rate. |
| `screen_size`, `minimap_size` | `64` | Square feature-layer resolutions. |
| `in_game_episode_s` | `120.0` | Seconds before truncation; use `600.0` for ladder maps. |
| `obs_spec_preset` | *(map-based)* | Override observation preset: `"minigame"` (15 dims), `"ladder"` (46 dims), `"rich"` (103 dims). Unset = minigame names → minigame, others → ladder. |
| `screen_layers` | `[]` | Spatial feature-layer names for `sc2_cnn` (e.g. `[player_relative, unit_hit_points]`). Ignored by all other policies. |
| `minimap_layers` | `[]` | Minimap channel names to concatenate with screen channels for `sc2_cnn`. |
| `adaptive_mutation` | `true` | Apply the 1/5 success rule to adapt `mutation_scale` during the greedy phase. |
| `patience` | `0` | Stop the greedy loop early if no improvement for this many consecutive sims (0 = run all). |
| `log_stats_every_n_sims` | `10` | Log reward-component totals and action-frequency ratios every N sims/generations across all SC2 greedy loops (`0` = disable). |
| `max_apm` | *(null)* | APM cap using a rolling token-bucket limiter measured in **game time** (`game_loop / 22.4`). `max_apm=300` means exactly 300 in-game actions per in-game minute regardless of training speed. Leave unset for no limit. |
| `apm_burst_s` | `2.0` | Token-bucket burst window in seconds. |
| `enable_belief` | `false` | Activate the fog-of-war belief + info-gain observation extension (adds ~192 dims for an 8×8 grid). |
| `bc_replay_dir` | *(null)* | Directory of `.SC2Replay` files for `--bc` mode. Also settable via `--replay-dir`. |
| `bc_player_id` | `winner` | Which player to clone: `winner`, `1`, or `2`. |
| `bc_race` | `any` | Race filter: `terran`, `zerg`, `protoss`, or `any`. |
| `bc_target` | `sc2_reinforce` | Policy architecture to pre-train. Supported: `sc2_reinforce`, `sc2_genetic`, `sc2_cmaes`, `sc2_neural_net`, `sc2_neural_dqn`, `sc2_lstm`, `sc2_cnn`, `epsilon_greedy`, `ucb_q`. |
| `bc_max_replays` | *(null)* | Cap on how many replays to process; `null` = all. |
| `bc_step_mul` | `step_mul` | Game ticks per replay step (should match the training `step_mul`). |
| `bc_epochs` | `10` | Gradient-descent passes (MLP/LSTM BC targets only). |
| `bc_learning_rate` | `0.001` | Learning rate for gradient BC targets. |
| `bc_batch_size` | `256` | Mini-batch size for gradient BC targets. |
| `bc_ignore_noop` | `true` | Drop `fn_idx=0` (no-op) steps before fitting. |
| `bc_lstm_hidden_size` | `64` | LSTM hidden-state size (`sc2_lstm` BC target only). |

### Supported policies

SC2-specific policies and tabular framework policies work on both the 15-dim
minigame and 46-dim ladder observation spaces.  The framework's generic linear
policies (`hill_climbing`, `neural_net`, and the base `genetic`) are **not**
compatible with SC2 because their output encoding clips `fn_idx` to `[-1, 1]`
and thresholds `x`/`y` to binary — use `sc2_genetic` instead.

| `policy_type` | Algorithm | Notes |
|---|---|---|
| `sc2_genetic` | Evolutionary (population of `SC2MultiHeadLinearPolicy`) | Default; recommended. Separate fn_idx (6×obs_dim) and sigmoid spatial (2×obs_dim) heads. |
| `sc2_hierarchical` | Evolutionary (population of `SC2HierarchicalLinearPolicy`) | Hierarchical action space (issue #388): first selects meta-category (move/attack/build/train/upgrade), then fn_idx within it, then spatial + queue. |
| `sc2_reinforce` | Two-head REINFORCE MLP (softmax fn + sigmoid spatial) | `games/sc2/sc2_policies.py`; gradient-trained per episode |
| `sc2_cmaes` | (μ/μ_w, λ)-CMA-ES over `SC2MultiHeadLinearPolicy` flat weights | `games/sc2/sc2_policies.py` |
| `sc2_lstm` | LSTM with SC2-native action encoding, trained by isotropic ES | `games/sc2/sc2_policies.py` |
| `sc2_cnn` | CNN (two conv layers + FC) + isotropic ES; spatial pixel obs | `games/sc2/cnn_policy.py`; requires non-empty `screen_layers` |
| `sc2_neural_net` | TMNF-style MLP (pure numpy) with SC2-native multi-head action encoding; mutate-and-keep evolutionary search | `games/sc2/sc2_policies.py`; hyperparam `hidden_sizes` |
| `epsilon_greedy` | Tabular Q-learning over `DISCRETE_ACTIONS` | Framework tabular policy |
| `ucb_q` | UCB1 online Q-learning over `DISCRETE_ACTIONS` | Framework tabular policy (renamed from `mcts`) |

> **Breaking change vs. earlier versions:** The bare-name `cmaes`, `reinforce`, `lstm`, and `neural_dqn` policy_types are no longer valid for SC2. Migrate configs to `sc2_cmaes`, `sc2_reinforce`, `sc2_lstm`, and `sc2_neural_dqn` respectively.

### Reward tuning (`reward_config.yaml`)

| Reward param | Default | Description |
|---|---|---|
| `score_weight` | `1.0` | Coefficient on the PySC2 cumulative `score` delta. |
| `win_bonus` / `loss_penalty` | `100` / `-100` | Terminal reward on `player_outcome > 0` / `< 0` (ladder maps only). |
| `step_penalty` | `-0.001` | Per-tick time cost. |
| `idle_penalty` | `0.0` | Per-step penalty when `army_count == 0 and food_used < food_cap` (BuildMarines / economy maps). |
| `idle_worker_penalty` | `0.0` | Per-step penalty scaled by `idle_worker_count` from PySC2 (issue #358).  Each idle worker subtracts this amount per step — workers should always be mining or building.  Opt-in default `0.0`; recommended range for economy/ladder maps: `-0.05` to `-0.5`. |
| `idle_bonus` | `0.0` | Per-step bonus when the agent issues `no_op` AND friendly units are within combat range of an enemy on screen.  Issue #127 — opt-in, default `0.0`.  Useful for combat minigames (DefeatRoaches, DefeatZerglingsAndBanelings) where standing still lets units shoot. |
| `move_exploration_bonus` | `0.01` | Bonus when a `Move_screen` is issued and the friendly-unit centroid is in a currently-unexplored cell of a `move_exploration_grid_size`² screen grid (default 8×8). A cell stays explored while the centroid keeps refreshing it and expires `move_exploration_decay_steps` env steps after the centroid leaves. A stationary centroid never re-triggers (blocks command spam); the decay stops the bonus going permanently silent once the screen is covered (which otherwise makes freezing optimal). |
| `move_exploration_grid_size` | `8` | Cells per axis of the move-exploration screen grid (higher = finer cells / smaller meaningful relocation). |
| `move_exploration_decay_steps` | `50` | Env steps before an explored cell may be rewarded again on a return visit. `0` = never expire (once per cell per episode); larger keeps an area "explored" longer. |
| `move_repeat_penalty` | `-0.02` | Penalty when a `Move_screen` command is less than `_MOVE_MIN_MEANINGFUL_FRAC` from the previous move target (covers both exact repeats and tiny stutter steps). |
| `move_self_penalty` | `-0.01` | Penalty for issuing `Move_screen` to the centroid of currently-visible friendly units. |
| `attack_move_bonus` | `0.0` | Per-step bonus when the agent issues `Attack_screen` (fn_idx 3) with the target on empty ground while enemies are visible (A-move). Opt-in. |
| `click_attack_bonus` | `0.0` | Per-step bonus when the agent issues `Attack_screen` with the target directly on a visible enemy unit. Subject to `click_attack_cooldown_steps`. Opt-in. |
| `click_attack_cooldown_steps` | `8` | Minimum env steps between rewarded target switches for `click_attack_bonus`. |
| `attack_bonus` | `0.0` | Per-step bonus whenever the agent issues `Attack_screen` (fn_idx 3), regardless of target type (A-move or click-to-attack). Simpler alternative to enabling both `attack_move_bonus` and `click_attack_bonus`. Opt-in. |
| `attack_friendly_penalty` | `-10.0` | Per-step penalty when an `Attack_screen` target lands on/near the centroid of visible friendly units (ally fire). Penalised heavily; set `0.0` to disable. |
| `economy_weight` | `0.0` | Coefficient on (minerals + vespene) delta — recommended `0.001` for ladder maps. |
| `resource_banking_penalty` | `0.0` | Per-step penalty proportional to excess minerals/gas above the thresholds (issue #372). Fires when `minerals > mineral_banking_threshold` or `vespene > gas_banking_threshold`. Nudges agent to invest banked resources rather than hoard them. Recommended range for ladder maps: `-0.0001` to `-0.001`. Opt-in. |
| `mineral_banking_threshold` | `300.0` | Minerals above this level are considered "banked" for `resource_banking_penalty`. |
| `gas_banking_threshold` | `200.0` | Vespene above this level is considered "banked" for `resource_banking_penalty`. |
| `unit_loss_penalty` | `0.0` | Penalty per army unit lost this step (army_count drop). Opt-in. |
| `damage_taken_penalty` | `0.0` | Penalty per raw HP+shield point lost across visible friendly units. Only on-screen units counted — keep weight small. Opt-in. |
| `passive_under_fire_penalty` | `0.0` | Per-step penalty when enemies are within attack range of friendlies and the agent did not issue `Attack_screen`. Opt-in. |
| `small_selection_bonus` | `0.0` | Per-step bonus for unit-targeted commands (`Move_screen` / `Attack_screen` / `Harvest_Gather_screen`) when the active selection is a single unit or under 50% of visible friendlies. Encourages micro over full-army commands. Opt-in. |
| `new_action_unlock_bonus` | `0.0` | One-shot bonus per tech-gated fn_idx that appears in `available_fn_ids` for the first time in an episode (issue #360). Tech-gated = has a required-building precondition **or** selection target is a production building that itself requires construction (e.g. `Train_Marine_quick` via Barracks, `Train_Zealot_quick` via Gateway). Basic-worker trains and always-available actions excluded. Recommended range: `1.0–10.0`. Opt-in. |
| `new_action_usage_bonus` | `0.0` | Per-step bonus when the agent issues a tech-gated fn_idx that has already been unlocked this episode (issue #400). Fires up to `new_action_usage_max_uses` times per fn_idx per episode. Complements `new_action_unlock_bonus`. Recommended range: `0.1–2.0`. Opt-in. |
| `new_action_usage_max_uses` | `50` | Cap on how many times per fn_idx per episode `new_action_usage_bonus` fires. After this many uses the bonus is silenced for that fn_idx for the rest of the episode. |
| `build_bonus` | `0.0` | Per-step bonus when the agent issues a "build" (structure) category action (issue #416). Covers all construction fn_idxs for all three races. Encourages building infrastructure over idle movement. Recommended range: `0.2–1.0`. Opt-in. |
| `build_repeat_penalty` | `0.0` | Per-step penalty when the same build fn_idx is issued on two consecutive steps. Set to `−build_bonus` to break placement-spam loops. Opt-in. |
| `train_bonus` | `0.0` | Per-step bonus when the agent issues a "train" (unit production) category action. Intended to be `4 × build_bonus` — unit production has more immediate combat impact than placing a structure. Opt-in. |
| `train_repeat_penalty` | `0.0` | Per-step penalty when the same train fn_idx is issued on two consecutive steps. Set to `−train_bonus` to break production-spam loops. Opt-in. |
| `supply_block_penalty` | `0.0` | Per-step penalty while supply-blocked (`food_used >= food_cap` and `food_cap < 200`). Production halts when capped — the most common macro failure. Range `-0.05` to `-0.5`. Opt-in. |
| `supply_growth_bonus` | `0.0` | Bonus per point of `food_cap` increase (build supply structures / expand). Only increases rewarded. Range `0.5–3.0`. Opt-in. |
| `worker_growth_bonus` | `0.0` | Bonus per point of `food_workers` increase (train workers). Pairs with `idle_worker_penalty`. Range `0.5–3.0`. Opt-in. |
| `army_growth_bonus` | `0.0` | Bonus per point of `food_army` increase (produce combat units). Only increases rewarded (losses → `unit_loss_penalty`). Range `0.5–3.0`. Opt-in. |
| `tech_building_bonus` | `0.0` | One-shot bonus the first time each friendly structure *type* is seen this episode (climbing the build tree). Range `2.0–10.0`. Opt-in. |
| `expansion_bonus` | `0.0` | One-shot bonus each time the friendly town-hall count reaches a new episode max (an expansion). Counted from visible town halls; running max keeps it monotonic and never rewards the starting base. Range `5.0–25.0`. Opt-in. |
| `scout_bonus` | `0.0` | Bonus proportional to the increase in `minimap_explored_frac` (revealing new map). Captures scouting beyond the screen. Per-step delta is tiny, so the weight is large. Range `5.0–50.0`. Opt-in. |

The bundled `reward_config.yaml` ships **tuned for 1v1 ladder play**:
`score_weight: 0.0` (PySC2's spiky cumulative score would swamp the win/loss
outcome over a full game), outcome-driven `win_bonus`/`loss_penalty`, and the
macro-progression block (supply / worker / army growth, supply-block, tech
unlock + usage, tech-building, expansion, scouting) enabled. Combat/movement
shaping is dialled down from the combat-minigame values. The minimal ladder
essentials are:

```yaml
score_weight: 0.0
win_bonus: 100.0
loss_penalty: -100.0
step_penalty: -0.002
economy_weight: 0.01
supply_block_penalty: -0.1
supply_growth_bonus: 1.0
worker_growth_bonus: 1.0
army_growth_bonus: 1.0
tech_building_bonus: 5.0
expansion_bonus: 10.0
scout_bonus: 20.0
new_action_usage_bonus: 0.1
```

`win_bonus` and `loss_penalty` are always active for ladder maps regardless of `score_weight`.

### Action space (issues #122, #127)

Continuous `Box([fn_idx, x, y, queue], shape=(4,))`.  `DISCRETE_ACTIONS` (used by tabular policies) is now ``[no_op, select_army, Move_screen × N×N]`` where ``N = SCREEN_GRID_RESOLUTION`` (default 8 → 64-cell grid, 66 rows total).

- Row 0 is `no_op` so tabular policies can elect to do nothing (issue #127) — necessary for "stand still and shoot" tactics.
- Row 1 is `select_army` (also the warmup action).
- Rows 2..N-1 are `Move_screen` calls at cell centres of an 8×8 grid covering the screen at one-cell-per-8-pixels granularity.

`SC2MultiHeadLinearPolicy` (the `sc2_genetic` default) now emits **continuous** `(x, y) ∈ [0, 1]²` coordinates via a sigmoid head (issue #122).  Pre-#122 weight files (with `spatial_{0..8}_weights` keys) load with the new `x_weights` / `y_weights` defaulting to zero.

**Tech-tree mask + deferred-action queue (issue #346).** `info["available_fn_ids"]` is now race ∩ PySC2 `available_actions` ∩ tech-tree ∩ selection-filtered. Preconditions live in `games/sc2/tech_tree.py` (`Preconditions` records with required buildings as DNF OR-sets, required upgrades/research, and a `SelectionReq` — `NONE` / `ANY_UNIT` / `OF_TYPE` with a `selection_target` set). Closes the gap that prior PRs (#307, #311, #315, #317, #322, #339) didn't: PySC2 reports `Build_FusionCore_screen` as available with an SCV selected regardless of whether a Starport exists; the hardcoded tech tree now drops it from the mask, and the extreme-random sampler reads the filtered set rather than raw PySC2 IDs. When the policy's chosen action needs a different selection, `SC2Client._resolve_action()` emits the right `select_*` this tick (preferring `select_idle_worker`, falling back to `select_point` on a mining/busy worker — issue #346 explicitly required workers that aren't idle to still be selectable) and queues the original action in a 1-slot FIFO so it replays on the next tick. The reactive `select_army` substitution from `_action_to_call` and the proactive guard at `step()` were removed. With DEBUG logging enabled, `SC2Client` also dumps a readable game-state snapshot (current units / buildings / upgrades / available actions with per-action selection hints) every ~10 s.

### Observation space (issue #126)

Three preset specs, opt-in via the `obs_spec_preset` training param:

| Preset | Dim | Default for | Notes |
|---|---|---|---|
| `minigame` | 15 | All minigame names | Player totals, selected-unit summary, screen `player_relative` summary, and minimap beacon centroid (`minimap_enemy_cx`/`minimap_enemy_cy`). |
| `ladder` | 46 | All non-minigame maps | Adds `food_workers`/`food_army`, idle worker / warp gate / larva counts, minimap stats (`minimap_self_count`, `minimap_enemy_count`, `minimap_visible_frac`, `minimap_explored_frac`, `minimap_camera_x/y`, `game_loop`), the 13 PySC2 score-cumulative entries, screen unit-density / mean-HP, top-K enemy counts (`topk_enemy_within_8`, `topk_enemy_within_24`), and `alert_count` (non-zero when under major attack). |
| `rich`  | 103 | Opt-in only | Adds 8 friendly unit-type counts (Marine…Mutalisk), 8 screen-quadrant counts (NE/NW/SE/SW × self/enemy), top-3 closest enemies (rel_x, rel_y, hp_ratio), available-actions binary mask (6), last-action one-hot (6), 8 enemy unit-type counts, screen shield/energy means (self_shield, enemy_shield, self_energy), minimap creep fraction, economy pipeline (upgrade_count, build_queue_size, cargo_count), selected-unit shield/energy averages, screen visibility fraction, anti-air density, and mean weapon cooldown. |

Set `obs_spec_preset: rich` in `training_params.yaml` to opt into the rich preset on any map.  Existing weight files migrate via the standard "missing key → 0.0" path — old champions can be loaded under any preset, and the new feature weights default to zero.

### Intra-run parallel evaluation (issue #229)

Population-based policies (`sc2_genetic`, `sc2_cmaes`, `sc2_lstm`,
`sc2_cnn`) can evaluate individuals concurrently across multiple local
SC2 binaries.  Each worker holds one persistent `SC2Env` so the
~10–20 s binary startup is paid once per run, not once per generation.

Set `n_workers > 1` in `training_params.yaml`:

| Key | Default | Notes |
|---|---|---|
| `n_workers` | `1` | Number of parallel SC2 binaries.  `1` keeps the existing serial behaviour, zero overhead.  Capped automatically at `population_size` (extra workers would just idle). |
| `worker_start_stagger_s` | `5.0` | Sleep between spawning child workers so PySC2's port-picker / `absl.flags` init don't race. |
| `worker_warmup_timeout_s` | `90.0` | Per-worker startup budget folded into the evaluator's per-job deadline. |
| `worker_base_seed` | `0` | Worker `i` is seeded with `base_seed + i`. |

**Semantics.** Evaluation is generation-synchronous: every individual in
the generation is scored before the distribution update runs, so
`sc2_cmaes`/`sc2_lstm`/`sc2_cnn` and `sc2_genetic` produce the same
champion sequence as the serial loop (just faster wallclock).  True
asynchronous ES is deferred — see `plans/issue-229-...md` §9.

**Sizing.** One headless SC2 binary on Linux uses ≈1 CPU + ≈1.5 GB RSS;
a 16-core / 32 GB box sustains roughly 8 binaries with overhead for the
trainer process.  Total binaries = `n_workers + 1`: each worker holds
its own SC2 env, plus the main process keeps one extra env so the
greedy loop can call `env.get_episode_time_limit()` / scale episode
length per generation.  Budget your `n_workers` accordingly.

**Scope.**  Only the four population-based SC2 policies are eligible.
Setting `n_workers > 1` for tabular / gradient-based policies
(`epsilon_greedy`, `ucb_q`, `sc2_reinforce`, `neural_dqn`, `reinforce`)
fails fast in `train_rl` before any binary spawns.  This is intra-run
parallelism; for inter-run parallelism (one experiment per worker)
keep using `python grid_search.py --local-workers N` (PR #244).

---

## Dependencies

Managed by Poetry. Run `poetry install` from repo root.

Core deps (always installed): `numpy`, `gymnasium`, `pyyaml`, `matplotlib`.
Game- and tooling-specific deps live in Poetry groups:

| Group | Installs | For |
|---|---|---|
| `tmnf` | `pandas`, `scipy`, `stable-baselines3`, `tensorboard`, `opencv-python`, `mss`, `pywin32`, `tminterface`, `pygbx` | TMNF (Windows) + LIDAR |
| `tmnf-test` | `pytest`, `scipy` | Running the unit suite |
| `torcs` | *(empty group)* | TORCS — `gym_torcs` installed from source (not on PyPI) |
| `sc2` *(optional)* | `pysc2`, `protobuf` | StarCraft 2 — `poetry install --with sc2` |
| `assetto_corsa` *(optional)* | `assetto-corsa-rl` | Assetto Corsa — `poetry install --with assetto_corsa` |
| `atari` *(optional)* | `ale-py` (MIT-licensed Atari 2600 ROMs bundled) | Atari — `poetry install --with atari` |
| `deep_rl` *(optional)* | `stable-baselines3`, `sb3-contrib` (pulls `torch`) | Gradient deep-RL policies (`ppo`, `a2c`, `sac`, `td3`, `qr_dqn`, `recurrent_ppo`) — `poetry install --with deep_rl`. Cross-platform. |

CarRacing needs `gymnasium[box2d]` (install separately, e.g.
`poetry add "gymnasium[box2d]"`); BeamNG needs `beamng-gym` (`pip install
beamng-gym`). Both are pulled in outside the Poetry groups — see the
respective `games/<game>/README.md`.

`tminterface`, `pygbx`, and `gym_torcs` are not on PyPI — install from
source before `poetry install`. Optional groups (`sc2`, `assetto_corsa`,
`atari`) are skipped by default so a Windows-only TMNF setup doesn't pull
in their stacks.
