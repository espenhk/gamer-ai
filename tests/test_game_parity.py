"""Game-parity completeness check (issue #452).

Asserts that every game registered in ``framework.game_adapter.GAME_ADAPTERS``
ships the expected set of per-game artifacts, so gaps stop appearing silently:

1. ``games/<dir>/README.md`` with a ``## Rewards`` section containing a
   ``Parameter | Default | Description`` table (the CLAUDE.md convention).
2. Both config masters: ``config/training_params.yaml`` and
   ``config/reward_config.yaml``.
3. At least one grid-search template (``config/gs_*.yaml``).
4. A non-stub ``analytics.py`` — mechanical criterion: the module *defines*
   (not just re-exports) at least one ``plot_*`` / ``_plot*`` function.  The
   ~62-line stubs only re-export framework plots and define none of their own.
5. The minimum per-game unit-test set from CONTRIBUTING.md: obs-spec, reward,
   and env tests (default paths ``tests/test_<game>_{obs_spec,reward,env}.py``,
   with documented per-game aliases for the games that predate the naming
   convention).

Known gaps are tracked in explicit per-check exemption sets below, each with
the issue / PR where the gap is being closed.  Fixing a gap should shrink the
matching exemption set in the same PR — the test fails if an exemption becomes
stale (artifact exists but the game is still listed), so the sets cannot rot.
"""

from __future__ import annotations

import ast
import glob
import os

import pytest

from framework.game_adapter import GAME_ADAPTERS

# ---------------------------------------------------------------------------
# Game roster
# ---------------------------------------------------------------------------

#: Registry key → directory under games/ (only spelled out where they differ).
_GAME_DIR_OVERRIDES = {"assetto": "assetto_corsa"}

GAMES = sorted(GAME_ADAPTERS.keys())


def _game_dir(game: str) -> str:
    return os.path.join("games", _GAME_DIR_OVERRIDES.get(game, game))


# ---------------------------------------------------------------------------
# Documented exemptions (issue #452 explicitly allows per-game opt-outs).
# Shrink these sets as the linked issues/PRs land.
# ---------------------------------------------------------------------------

#: Games with no grid-search template yet.
#: (beamng and iracing left this set when PR #500 / issue #446 merged.)
KNOWN_MISSING_GRID_TEMPLATES = {
    "minerl",  # newest integration; no template/tracking issue yet
}

#: Games whose analytics.py is still the framework-reexport stub.
#: (car_racing, assetto, beamng and rocket_league left this set as
#: PRs #497 / #501 / #502 / #503 merged.)
KNOWN_STUB_ANALYTICS = {
    "atari",  # issue #465 — real analytics in PR #471
    "minerl",  # issue #449 (umbrella); no per-game issue yet
}

#: (game, kind) pairs where the minimum unit-test file is missing.
KNOWN_MISSING_TESTS = {
    ("beamng", "obs_spec"),  # no beamng obs-spec/reward unit tests yet
    ("beamng", "reward"),
    ("car_racing", "obs_spec"),  # covered by tests/integration/test_car_racing.py only
    ("car_racing", "reward"),
    ("car_racing", "env"),
    ("iracing", "reward"),  # telemetry/env/obs covered (PR #479); reward calc is not
}

#: Per-game aliases for the minimum test files (games that predate the
#: tests/test_<game>_<kind>.py convention).  Values are repo-relative paths.
_TEST_FILE_ALIASES: dict[tuple[str, str], str] = {
    ("tmnf", "reward"): "tests/test_reward.py",
    ("tmnf", "env"): "tests/test_env_termination.py",
    # The Assetto Corsa suite lives in a per-game directory; test_smoke.py
    # covers obs-spec dims, reward calc, and env reset/step in one file.
    ("assetto", "obs_spec"): "tests/assetto_corsa/test_smoke.py",
    ("assetto", "reward"): "tests/assetto_corsa/test_smoke.py",
    ("assetto", "env"): "tests/assetto_corsa/test_smoke.py",
}


def _test_file_for(game: str, kind: str) -> str:
    return _TEST_FILE_ALIASES.get((game, kind), f"tests/test_{game}_{kind}.py")


# ---------------------------------------------------------------------------
# 1. README with a ## Rewards table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("game", GAMES)
def test_readme_has_rewards_table(game):
    path = os.path.join(_game_dir(game), "README.md")
    assert os.path.exists(path), f"{game}: missing {path}"
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "## Rewards" in text, f"{game}: README.md has no '## Rewards' section"
    rewards = text.split("## Rewards", 1)[1]
    header = next(
        (line for line in rewards.splitlines() if line.lstrip().startswith("|") and "Parameter" in line),
        None,
    )
    assert header is not None, f"{game}: '## Rewards' section has no 'Parameter | ...' table"
    assert "Default" in header and "Description" in header, (
        f"{game}: Rewards table header must include Parameter | Default | Description, got: {header!r}"
    )


# ---------------------------------------------------------------------------
# 2. Config masters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("game", GAMES)
@pytest.mark.parametrize("master", ["training_params.yaml", "reward_config.yaml"])
def test_config_masters_exist(game, master):
    path = os.path.join(_game_dir(game), "config", master)
    assert os.path.exists(path), f"{game}: missing config master {path}"


# ---------------------------------------------------------------------------
# 3. Grid-search templates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("game", GAMES)
def test_has_grid_search_template(game):
    templates = glob.glob(os.path.join(_game_dir(game), "config", "gs_*.yaml"))
    if game in KNOWN_MISSING_GRID_TEMPLATES:
        assert not templates, f"{game}: has grid templates now — remove it from KNOWN_MISSING_GRID_TEMPLATES"
        pytest.xfail(f"{game}: no grid template yet (see KNOWN_MISSING_GRID_TEMPLATES)")
    assert templates, f"{game}: no gs_*.yaml grid-search template under config/"


# ---------------------------------------------------------------------------
# 4. Non-stub analytics
# ---------------------------------------------------------------------------


def _own_plot_functions(game: str) -> list[str]:
    """Top-level plot_* / _plot* functions *defined* in the game's analytics.py.

    Parses the source with ast instead of importing the module, so the check
    has no side effects (matplotlib backend init, optional game deps) and
    re-exported framework plots don't count.
    """
    path = os.path.join(_game_dir(game), "analytics.py")
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (node.name.startswith("plot_") or node.name.startswith("_plot"))
    ]


@pytest.mark.parametrize("game", GAMES)
def test_analytics_is_not_a_stub(game):
    own_plots = _own_plot_functions(game)
    if game in KNOWN_STUB_ANALYTICS:
        assert not own_plots, f"{game}: analytics defines its own plots now — remove it from KNOWN_STUB_ANALYTICS"
        pytest.xfail(f"{game}: stub analytics (see KNOWN_STUB_ANALYTICS)")
    assert own_plots, (
        f"{game}: games/.../analytics.py defines no plot_* functions of its own — "
        "it only re-exports framework plots (stub)"
    )


# ---------------------------------------------------------------------------
# 5. Minimum per-game test files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("game", GAMES)
@pytest.mark.parametrize("kind", ["obs_spec", "reward", "env"])
def test_minimum_test_files_exist(game, kind):
    path = _test_file_for(game, kind)
    if (game, kind) in KNOWN_MISSING_TESTS:
        assert not os.path.exists(path), f"{game}/{kind}: {path} exists now — remove it from KNOWN_MISSING_TESTS"
        pytest.xfail(f"{game}: no {kind} test yet (see KNOWN_MISSING_TESTS)")
    assert os.path.exists(path), f"{game}: missing minimum test file {path}"
