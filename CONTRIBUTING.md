# Contributing to gamer-ai

Thanks for thinking about contributing! `gamer-ai` started as a Trackmania
Nations Forever experiment and has grown into a small multi-game RL
framework — TMNF, TORCS, StarCraft 2, BeamNG, Assetto Corsa, and
Gymnasium's CarRacing all share the same training loop today, and the
explicit goal is to keep adding games.

This document covers:

- [Code of conduct](#code-of-conduct)
- [Ways to contribute](#ways-to-contribute)
- [Getting set up](#getting-set-up)
- [Running tests](#running-tests)
- [Project layout (the short version)](#project-layout-the-short-version)
- [Adding a new game](#adding-a-new-game)
- [Adding a new policy / algorithm](#adding-a-new-policy--algorithm)
- [Coding conventions](#coding-conventions)
- [Documentation conventions](#documentation-conventions)
- [Submitting a pull request](#submitting-a-pull-request)
- [Reporting bugs and proposing features](#reporting-bugs-and-proposing-features)
- [Issue labels](#issue-labels)
- [License](#license)

---

## Code of conduct

Be kind, be specific, assume good faith. We're a small project — disagreements
should be technical, not personal. Harassment, personal attacks, or
discriminatory language are not welcome and will get you removed from the
project.

---

## Ways to contribute

You don't have to write code to help:

- **Try a game integration** that already exists, file bugs about the
  rough edges, and share your training plots.
- **Improve documentation** — every `README.md` (root + `games/<name>/`)
  is fair game, and "this section confused me" is a real bug.
- **Propose a new game** in a GitHub issue using the shared issue
  template. Even if you can't implement it, a well-scoped proposal is
  often the hardest part.
- **Pick up a "good first issue"** — these are tagged in the
  [issue tracker](https://github.com/espenhk/gamer-ai/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
- **Tighten the framework** — e.g. add a new policy, a new analytics view,
  or improve the distributed coordinator.
- **Write tests** for areas the `tests/README.md` "what isn't tested"
  paragraphs admit are thin.

---

## Getting set up

### Prerequisites

- **Python 3.11+** and **Poetry** (≥ 1.7).
- A working `git`.
- Platform notes:
  - **TMNF** training is Windows-only (TMInterface attaches to the live
    game process). You can still work on the framework or other games
    from Linux / macOS.
  - **TORCS / SC2 / CarRacing** all work on Linux. SC2 has a headless
    Linux binary; CarRacing needs no external binary at all.
  - **BeamNG / Assetto Corsa** are Windows-only games but their Python
    wrappers can be imported (and unit-tested) anywhere.

### Clone and install

```bash
git clone https://github.com/espenhk/gamer-ai.git
cd gamer-ai

# Cross-platform dev install — enough to run the unit tests on any OS:
poetry install --with tmnf-test

# Add the game-specific groups you'll actually use:
poetry install --with tmnf-test,sc2          # Linux/macOS SC2 work
poetry install --with tmnf-test,torcs        # TORCS work
poetry install --with tmnf,tmnf-test         # full TMNF on Windows
```

If `poetry install --with tmnf` fails on Linux with `pywin32` / `mss` /
`tminterface` errors, that's expected — install `tmnf-test` instead. The
`tmnf-test` group pulls only the cross-platform dependencies the unit
tests need.

### Smoke test

Run the same command CI runs so you don't get blocked by tests that need
optional binaries (TMInterface, SC2, Box2D):

```bash
PYTHONPATH=. poetry run python -m pytest tests/ \
    --ignore=tests/test_env_termination.py \
    --ignore=tests/test_grid_search.py \
    --ignore=tests/integration/
```

You should get a green run in well under a minute. The ignored top-level
files import `tminterface` (Windows-only); `tests/integration/` needs
either `gymnasium[box2d]` or the SC2 binary. On a fully-provisioned
Windows dev box you can drop the `--ignore` flags to run the full suite.

---

## Running tests

The CI pipelines under `.github/workflows/` are the source of truth.
Practically:

| Command | What it runs |
|---|---|
| `PYTHONPATH=. poetry run python -m pytest tests/ --ignore=tests/test_env_termination.py --ignore=tests/test_grid_search.py --ignore=tests/integration/` | Cross-platform unit-test suite — same set of files the `Tests / test` CI job runs. The two ignored top-level files import `tminterface` (Windows-only); `tests/integration/` needs optional binaries. |
| `poetry run python -m pytest tests/` | Full suite. Only green if you've installed the `tmnf` group on Windows **and** the integration extras (`gymnasium[box2d]`, SC2 binary). Skip this unless you're on a fully-provisioned dev box. |
| `poetry run python -m pytest tests/integration/ -m integration` | Integration tests on their own (need `gymnasium[box2d]` for CarRacing, the SC2 binary for SC2). |
| `poetry run python -m pytest tests/test_<area>.py -v` | Focus a single area. |
| `python main.py smoke --game car_racing --no-interrupt` | End-to-end smoke run that doesn't need any external binary. Great for spot-checking framework changes. |

### Test-suite contract

`tests/README.md` documents every test (one line each, grouped by area)
plus per-area "what is and isn't tested" paragraphs. **When you add,
remove, or substantially change tests, update `tests/README.md` in the
same PR.** This is checked in review.

---

## Project layout (the short version)

```
gamer-ai/
├── main.py                 # Entry point — python main.py <experiment> [--game <name>]
├── framework/              # Game-agnostic training loop, base env, base policies, analytics
├── games/
│   ├── tmnf/               # Trackmania Nations Forever (primary)
│   ├── torcs/              # TORCS (open-source racing)
│   ├── sc2/                # StarCraft 2 (PySC2)
│   ├── beamng/             # BeamNG.drive
│   ├── assetto_corsa/      # Assetto Corsa
│   └── car_racing/         # Gymnasium CarRacing-v2 (no binary)
├── distributed/            # HTTP coordinator + worker for multi-machine grid search
├── infrastructure/         # Terraform stack for Azure worker VMs
├── config/                 # Master training_params.yaml / reward_config.yaml templates
├── tests/                  # Pytest suite — see tests/README.md
├── README.md               # User-facing setup + run guide
├── CLAUDE.md               # Architecture-level docs (the source of truth for design)
└── CONTRIBUTING.md         # This file
```

The deep dive lives in `CLAUDE.md` (architecture, observation specs,
policy taxonomy, reward knobs, threading model). When user-visible
behaviour changes, update `README.md`; when architecture changes,
update `CLAUDE.md`.

---

## Adding a new game

A new game lives entirely under `games/<name>/` and plugs into the
framework through a single `GameAdapter`. The pattern is the same for
every game in the repo — `games/car_racing/` is the smallest reference
implementation; copy it as a starting point.

> **Read the protocol docs first.** Every framework-side seam you'll touch
> is documented one file per protocol under
> [`docs/framework/`](docs/framework/README.md) — `GameAdapter`,
> `GameSpec` / `RunConfig` / `ProbeSpec` / `WarmupSpec` / `PolicyExtras`,
> `BaseGameEnv`, `RewardCalculatorBase`, `BasePolicy`, and `ObsSpec`, each
> with a worked example. You should be able to read those end-to-end and
> write a `_template` adapter without opening any `games/<name>/` code.

### What you need to implement

| File | Role | Protocol doc |
|---|---|---|
| `games/<name>/adapter.py` | Implements `framework.game_adapter.GameAdapter`. Wires the game-specific objects below into the framework. Must expose `make_adapter()`. | [`game_adapter.md`](docs/framework/game_adapter.md), [`run_config.md`](docs/framework/run_config.md) |
| `games/<name>/env.py` | `BaseGameEnv` subclass (a `gymnasium.Env`). Resets the game, steps it, returns `(obs, reward, terminated, truncated, info)`. | [`base_env.md`](docs/framework/base_env.md) |
| `games/<name>/obs_spec.py` | Module-level `ObsSpec` describing the flat observation vector (names + scales). Used for normalisation, analytics labels, and weight-file migration. | [`obs_spec.md`](docs/framework/obs_spec.md) |
| `games/<name>/actions.py` | `DISCRETE_ACTIONS` — the list of discrete action tuples that tabular policies can pick from. | — |
| `games/<name>/reward.py` | `RewardCalculator` + `RewardConfig` for the game. | [`reward.md`](docs/framework/reward.md) |
| `games/<name>/analytics.py` | `save_experiment_results(...)` — at minimum produce a `results.md` and a reward-over-time plot. | — |
| `games/<name>/config/training_params.yaml` | Master training params copied into each new experiment. | — |
| `games/<name>/config/reward_config.yaml` | Master reward weights copied into each new experiment. | — |
| `games/<name>/README.md` | Per-game user-facing README — install, run, obs/action/reward tables. | — |

Then register the adapter in `framework/game_adapter.GAME_ADAPTERS` and
add `<name>` to the `--game` choices in `main.py`.

### Step-by-step

1. **Open a new-game issue first** using the shared
   [issue template](.github/ISSUE_TEMPLATE/issue_template.md), so we can agree on scope
   before code lands. Keep the "New game proposal details" section and
   remove the rest.
2. **Copy `games/car_racing/` to `games/<name>/`** and rename the
   `CarRacingAdapter` class. Strip out anything that doesn't apply.
3. **Stand up `env.py`** so `env.reset()` / `env.step(action)` work
   against the real game. Keep imports of the game's SDK *inside*
   functions / methods, not at module top — Linux CI imports the
   module to run lazy unit tests, and a hard import of a Windows-only
   SDK breaks that. (See `games/assetto_corsa/clients/ac_client.py`
   for the pattern.)
4. **Pin the observation spec** in `obs_spec.py`. Every feature gets a
   name and a scale; the framework divides the raw value by the scale
   before handing it to a policy. Aim for normalised values roughly in
   `[-1, 1]`.
5. **Write `reward.py`** with a `RewardCalculator(reward_cfg, ...)` that
   exposes a `compute(state, prev_state)` (or similar) method. Default
   knobs go in `reward_config.yaml`.
6. **Hook up `adapter.py`** — fill in `experiment_dir`, `build_game_spec`,
   `build_probe`, `build_warmup`, `build_extras` as needed.
7. **Register** in `framework/game_adapter.py` and `main.py`.
8. **Add tests** under `tests/test_<name>_<area>.py`. The minimum set is:
   - `test_<name>_obs_spec.py` — round-trip a state through your obs
     extractor and check shapes / scales.
   - `test_<name>_reward.py` — feed canned state deltas through the
     reward calc and assert the expected signs.
   - `test_<name>_env.py` — instantiate `make_env()` with a mock client
     and step it a few times.
   - Update `tests/README.md` to describe each new test.
9. **Add a section** to your new `games/<name>/README.md` and link it
   from the root `README.md`'s `--game` table.
10. **Optionally** add an integration job to
    `.github/workflows/integration-tests.yml` if the game has a free,
    headless way to run end-to-end.

### What "done" looks like

- `python main.py smoke --game <name> --no-interrupt` finishes a short
  run without crashing.
- `poetry run python -m pytest tests/test_<name>_*.py -v` is green.
- Reward-over-time plot under
  `experiments/<name>/smoke/results/` shows non-trivial learning (even a
  flat curve is fine for the smoke run — we're checking the pipe, not
  the algorithm).

---

## Adding a new policy / algorithm

Policies live in `framework/policies.py` (game-agnostic) or under
`games/<name>/` (game-specific, e.g. `sc2_genetic`). Every policy
inherits from `BasePolicy` —
[`docs/framework/policies.md`](docs/framework/policies.md) documents the
full interface with a worked example. In short:

- `__call__(obs)` — pick an action given an observation vector (the
  "`act`" step).
- `update(...)` — for trainable online policies; no-op for hand-coded
  baselines and evolutionary policies.
- `save(path)` writes; loading happens through construction
  (`from_cfg` / the weights-file constructor argument) — there is no
  `load()` method.

Add the new `policy_type` string to `CLAUDE.md`'s policy table, update
the `README.md` policy table if it's game-agnostic, and ship at least
one test under `tests/test_<policy>_policy.py`.

---

## Coding conventions

- **Python 3.11+**, `from __future__ import annotations`, type hints on
  public functions.
- **Numpy-first**: pure numpy is preferred for new tabular / linear /
  small-network policies — keeps installs lean and CI fast.
- **No `print` for diagnostics** — use `logging.getLogger(__name__)`.
- **No dead code, no commented-out blocks, no TODO without an issue
  number**.
- **Comments**: write the *why*, not the *what*. Don't restate the code
  in prose. Only add a comment when a future reader would otherwise be
  surprised.
- **Imports**: heavy / OS-specific imports go inside the function that
  needs them, so cross-platform CI can import the module without
  pulling in `pywin32` / `tminterface` / `pysc2`.

---

## Documentation conventions

When you change code, ask which of these need to follow:

| Change | Update |
|---|---|
| Any user- or developer-visible change (new feature, new config key, breaking change, bug fix, dependency change) | `CHANGELOG.md` — add a bullet under `## [Unreleased]` |
| New / changed CLI flag, config key, or install step | `README.md` + relevant `games/<name>/README.md` |
| New training-loop semantics, observation features, or policy | `CLAUDE.md` |
| New / removed / substantially-changed tests | `tests/README.md` |
| New game | New `games/<name>/README.md` + entry in root `README.md` `--game` table |
| New infrastructure / terraform change | `infrastructure/README.md` |

PR review *will* push back if any of these are out of date.

---

## Submitting a pull request

1. **Branch off `main`** with a descriptive name —
   `add-rocket-league-game`, `fix-sc2-replay-overwrite`, etc.
2. **Make small, focused commits**. A bug fix and an unrelated refactor
   should be two PRs.
3. **Run the unit tests locally** before opening the PR, using the same
   command CI runs so you get the same result:
   ```bash
   PYTHONPATH=. poetry run python -m pytest tests/ \
       --ignore=tests/test_env_termination.py \
       --ignore=tests/test_grid_search.py \
       --ignore=tests/integration/
   ```
   If you're on a fully-provisioned Windows dev box with `tminterface`
   installed, drop the `--ignore` flags to run the full suite.
4. **Open the PR against `main`**. Use the
   [PR template](.github/PULL_REQUEST_TEMPLATE.md), keep the section(s)
   that apply to your change, and delete the rest.
5. **CI gates**:
   - `Tests / test` (the unit-test workflow) runs on every PR.
   - `Integration Tests / car-racing` / `sc2` run after an approving
     review, and only if files in their watched paths changed.
6. **AI review is optional**. If you want an automated reviewer
   (e.g. `/ultrareview`) to run on your PR, request it in the PR
   conversation.
7. **Address review comments** in new commits (don't force-push over an
   in-progress review unless asked — it makes re-review harder).
8. **Squash on merge** — keep `main` linear.

### Cutting a release

`gamer-ai` carries a `pyproject.toml` / `framework/version.py` version
number so every run can be traced back to a known code state via the
`code_version` field recorded in `experiment_data.json`. Cut a release
when a logical chunk of work lands and you want a quotable tag to refer
to it from issues, plots, or experiment notes.

From a clean `main`:

```bash
python scripts/release.py 0.2.0
git push origin main --tags
```

The script bumps both version locations, promotes `## [Unreleased]` in
`CHANGELOG.md` to `## [0.2.0] - YYYY-MM-DD`, commits as `Release v0.2.0`,
and creates an annotated `v0.2.0` tag. Follow
[SemVer](https://semver.org/): patch for bug fixes, minor for
backward-compatible features, major for breaking changes to configs,
weight-file format, or experiment data on disk.

### What the reviewer is looking for

- The change does what the issue / PR description says, and nothing
  more.
- New code has tests, and `tests/README.md` reflects them.
- Docs match the new behaviour.
- No accidental commits of secrets, large binaries, or experiment
  outputs.

---

## Reporting bugs and proposing features

Use the issue template. It has separate bug / feature / new-game
sections — keep what applies and delete the rest.

For open-ended "should we do X?" threads, prefer
[GitHub Discussions](https://github.com/espenhk/gamer-ai/discussions)
over an issue.

---

## Issue labels

Issues are triaged with a small canonical label set:

| Label | Meaning |
|---|---|
| `bug` / `enhancement` / `documentation` | Issue type |
| `good first issue` / `help wanted` | Contribution-friendly backlog items |
| `game-support` | New `games/<name>/` proposals and game-integration requests |
| `framework` / `analytics` / `infrastructure` / `tooling` | Area ownership |
| `triage` | Default label for newly opened issues; remove once assessed |

If you're new here, start with issues labeled `good first issue`.

---

## License

By contributing you agree your contributions will be licensed under the
MIT License, the same license that covers the rest of the project (see
[LICENSE](LICENSE)).
