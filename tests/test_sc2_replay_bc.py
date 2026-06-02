"""Tests for games.sc2.replay_bc — replay BC dataset builder (issue #351).

The SC2 binary and PySC2 are never imported; all controller / features
interactions are replaced by lightweight fakes injected into sys.modules
before the module under test is imported, so lazy imports inside
replay_bc.py see the fakes.  The suite runs in milliseconds with no
external dependencies.
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Fake PySC2 + s2clientprotocol modules — injected before any import of
# games.sc2.replay_bc so lazy imports inside that module see the fakes.
#
# The module-level run_configs_mod and features_mod references let tests
# configure return values without needing patch() on module attributes.
# ---------------------------------------------------------------------------


def _make_fake_pysc2():
    """Build a minimal fake pysc2 hierarchy and inject it into sys.modules.

    Also injects ``pysc2.lib.actions`` with an identity PySC2-id mapping
    (PySC2 id == internal fn_idx) so ``function_call_to_action`` can resolve
    ``_FakeFunctionCall(fn_id=N)`` → internal fn_idx N.  This lets test
    helpers use internal fn_idx values directly as the function call id.
    """
    pysc2 = types.ModuleType("pysc2")
    run_configs_mod = types.ModuleType("pysc2.run_configs")
    lib_mod = types.ModuleType("pysc2.lib")
    features_mod = types.ModuleType("pysc2.lib.features")
    actions_mod = types.ModuleType("pysc2.lib.actions")

    class _Dimensions:
        def __init__(self, *, screen, minimap):
            self.screen = screen
            self.minimap = minimap

    features_mod.Dimensions = _Dimensions
    features_mod.features_from_game_info = MagicMock(return_value=MagicMock())

    # Build FUNCTIONS with identity mapping so PySC2 id == internal fn_idx.
    # function_call_to_action does getattr(FUNCTIONS, name) → .id to get the
    # PySC2 id for the call, and _get_pysc2_id_to_fn_idx builds the reverse
    # mapping.  Identity keeps both consistent without needing real PySC2.
    from games.sc2.actions import FUNCTION_IDS

    _functions_ns = types.SimpleNamespace()
    for fn_idx, fn_name in FUNCTION_IDS.items():
        setattr(_functions_ns, fn_name, types.SimpleNamespace(id=fn_idx))

    class _FunctionCall:
        def __init__(self, function, arguments):
            self.function = function
            self.arguments = arguments

    actions_mod.FUNCTIONS = _functions_ns
    actions_mod.FunctionCall = _FunctionCall

    lib_mod.features = features_mod
    lib_mod.actions = actions_mod

    run_configs_mod.get = MagicMock(return_value=MagicMock())

    pysc2.run_configs = run_configs_mod
    pysc2.lib = lib_mod

    sys.modules["pysc2"] = pysc2
    sys.modules["pysc2.run_configs"] = run_configs_mod
    sys.modules["pysc2.lib"] = lib_mod
    sys.modules["pysc2.lib.features"] = features_mod
    sys.modules["pysc2.lib.actions"] = actions_mod
    return pysc2, run_configs_mod, features_mod


def _make_fake_s2api():
    """Build minimal s2clientprotocol proto stubs."""
    s2_mod = types.ModuleType("s2clientprotocol")
    api_mod = types.ModuleType("s2clientprotocol.sc2api_pb2")

    class _ProtoStub:
        """Accept any kwargs and store them as attrs."""

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    api_mod.InterfaceOptions = _ProtoStub
    api_mod.SpatialCameraSetup = _ProtoStub
    api_mod.Size2DI = _ProtoStub
    api_mod.RequestStartReplay = _ProtoStub

    s2_mod.sc2api_pb2 = api_mod
    sys.modules["s2clientprotocol"] = s2_mod
    sys.modules["s2clientprotocol.sc2api_pb2"] = api_mod
    return s2_mod, api_mod


# Inject fakes before importing the module under test.
_pysc2_pkg, _run_configs_mod, _features_mod = _make_fake_pysc2()
_s2_pkg, _sc2_api_mod = _make_fake_s2api()

from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC  # noqa: E402
from games.sc2.replay_bc import (  # noqa: E402
    _parse_replay_info,
    _pick_best_action,
    _read_one_replay,
    _resolve_player_id,
    build_dataset,
    fit_bc,
    iter_replays,
    load_dataset,
    replay_observations,
    validate_replay_dir,
)
from games.sc2.replay_bc import (
    run as bc_run,
)

_OBS_DIM = len(SC2_MINIGAME_OBS_SPEC.names)
_OBS_NAMES = SC2_MINIGAME_OBS_SPEC.names


# ---------------------------------------------------------------------------
# Helpers — mock replay infrastructure
# ---------------------------------------------------------------------------


class _FakePlayerInfo:
    """Simulates a single entry in info.player_info."""

    def __init__(self, player_id: int, race_int: int, result: int):
        class _PI:
            pass

        pi = _PI()
        pi.player_id = player_id
        pi.race_actual = race_int
        self.player_info = pi

        class _PR:
            pass

        pr = _PR()
        pr.result = result
        self.player_result = pr


def _fake_replay_info(*, winner_pid: int, races: dict[int, int]):
    """Return a fake replay_info proto with given winner + race map.

    *races* maps player_id → race_int (1=terran, 2=zerg, 3=protoss, 4=random).
    """
    info = MagicMock()
    info.player_info = [
        _FakePlayerInfo(pid, race_int, 1 if pid == winner_pid else 2) for pid, race_int in races.items()
    ]
    return info


class _FakeFunctionCall:
    """Minimal FunctionCall stub: .function (int, the PySC2 fn_id) and .arguments."""

    def __init__(self, fn_id: int, arguments: list | None = None):
        self.function = fn_id
        self.arguments = arguments or []


def _build_mock_controller(
    replay_info: object,
    observe_sequence: list[dict],
    actions_per_step: list[list[_FakeFunctionCall]],
) -> MagicMock:
    """Build a mock controller for a single replay.

    *observe_sequence* is a list of obs dicts (one per step).  After all
    steps a terminal ``player_result`` response is appended automatically.
    *actions_per_step* is a parallel list of per-step action lists.
    """
    assert len(observe_sequence) == len(actions_per_step)

    class _FakeObsProto:
        def __init__(self, obs_dict, raw_actions, done=False):
            self.observation = obs_dict
            self.actions = raw_actions
            self.player_result = [object()] if done else []

    responses = [_FakeObsProto(obs, acts) for obs, acts in zip(observe_sequence, actions_per_step)]
    responses.append(_FakeObsProto({}, [], done=True))
    observe_iter = iter(responses)

    ctrl = MagicMock()
    ctrl.replay_info.return_value = replay_info
    ctrl.observe.side_effect = lambda: next(observe_iter)
    ctrl.step.return_value = None
    ctrl.start_replay.return_value = None
    ctrl.game_info.return_value = MagicMock()
    return ctrl


def _build_mock_features() -> MagicMock:
    """Build a mock Features object.

    ``transform_obs`` returns its argument unchanged (the obs dict is already
    in the right format for ``extract_flat_obs``).
    ``reverse_action`` returns its argument unchanged (FakeFunctionCall
    already has the right interface).
    """
    feat = MagicMock()
    feat.transform_obs.side_effect = lambda obs_dict: obs_dict
    feat.reverse_action.side_effect = lambda a: a
    return feat


def _patch_run_config(controller: MagicMock) -> MagicMock:
    """Return a mock run_config whose ``start()`` yields *controller*."""

    @contextmanager
    def _start(**_kwargs):
        yield controller

    rc = MagicMock()
    rc.start.side_effect = _start
    rc.replay_data.return_value = b"fake_replay_bytes"
    return rc


def _configure_fakes(rc: MagicMock, feat: MagicMock) -> None:
    """Point the globally-injected pysc2 fakes at *rc* and *feat*.

    Since replay_bc.py uses lazy imports (``from pysc2 import run_configs``
    inside functions), patching module-level attributes is not possible.
    Instead we configure the already-injected fakes in sys.modules directly.
    """
    _run_configs_mod.get.return_value = rc
    _features_mod.features_from_game_info.return_value = feat


def _reset_fakes() -> None:
    """Reset the injected fakes to avoid test cross-contamination."""
    _run_configs_mod.get.reset_mock(return_value=True, side_effect=True)
    _features_mod.features_from_game_info.reset_mock(return_value=True, side_effect=True)


# ---------------------------------------------------------------------------
# Tests: validate_replay_dir (issue #352)
# ---------------------------------------------------------------------------


class TestValidateReplayDir(unittest.TestCase):
    def test_nonexistent_folder_raises(self):
        with self.assertRaises(ValueError) as ctx:
            validate_replay_dir("/nonexistent/path/that/does/not/exist")
        self.assertIn("does not exist", str(ctx.exception))

    def test_file_path_raises(self):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".SC2Replay") as f:
            with self.assertRaises(ValueError) as ctx:
                validate_replay_dir(f.name)
        self.assertIn("not a directory", str(ctx.exception))

    def test_empty_folder_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError) as ctx:
                validate_replay_dir(d)
        self.assertIn("No .SC2Replay files found", str(ctx.exception))

    def test_returns_sc2replay_files_only(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "game1.SC2Replay").touch()
            (d / "game2.SC2Replay").touch()
            (d / "notes.txt").touch()
            result = validate_replay_dir(d)
            self.assertEqual([p.name for p in result], ["game1.SC2Replay", "game2.SC2Replay"])

    def test_returns_sorted_paths(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "b.SC2Replay").touch()
            (d / "a.SC2Replay").touch()
            result = validate_replay_dir(d)
            self.assertEqual(result[0].name, "a.SC2Replay")
            self.assertEqual(result[1].name, "b.SC2Replay")

    def test_race_filter_warning_emitted(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "game.SC2Replay").touch()
            with self.assertLogs("games.sc2.replay_bc", level="WARNING") as cm:
                validate_replay_dir(d, race="terran")
            self.assertTrue(any("terran" in msg for msg in cm.output))

    def test_race_any_no_warning(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "game.SC2Replay").touch()
            with self.assertLogs("games.sc2.replay_bc", level="INFO") as cm:
                validate_replay_dir(d, race="any")
            # No WARNING lines about race should appear
            self.assertFalse(any("WARNING" in msg and "Race filter" in msg for msg in cm.output))

    def test_race_none_no_warning(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "game.SC2Replay").touch()
            with self.assertLogs("games.sc2.replay_bc", level="INFO") as cm:
                validate_replay_dir(d, race=None)
            self.assertFalse(any("Race filter" in msg for msg in cm.output))

    def test_version_mismatch_warning_emitted(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "replay_3.0.0.12345.SC2Replay").touch()
            with self.assertLogs("games.sc2.replay_bc", level="WARNING") as cm:
                validate_replay_dir(d, version="4.9.3")
            self.assertTrue(any("4.9.3" in msg for msg in cm.output))

    def test_version_match_no_warning(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "replay_4.9.3.77379.SC2Replay").touch()
            with self.assertLogs("games.sc2.replay_bc", level="INFO") as cm:
                validate_replay_dir(d, version="4.9.3")
            self.assertFalse(any("WARNING" in msg and "version" in msg.lower() for msg in cm.output))

    def test_version_partial_mismatch_warns(self):
        """Warns when SOME but not all replays match the version string."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "replay_4.9.3.77379.SC2Replay").touch()
            (d / "replay_4.8.2.70000.SC2Replay").touch()
            with self.assertLogs("games.sc2.replay_bc", level="WARNING") as cm:
                validate_replay_dir(d, version="4.9.3")
            warning_msgs = [m for m in cm.output if "WARNING" in m and "version" in m.lower()]
            self.assertEqual(len(warning_msgs), 1)
            self.assertIn("1/2", warning_msgs[0])


# ---------------------------------------------------------------------------
# Tests: iter_replays
# ---------------------------------------------------------------------------


class TestIterReplays(unittest.TestCase):
    def test_finds_sc2replay_files(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "game1.SC2Replay").touch()
            (d / "game2.SC2Replay").touch()
            (d / "not_a_replay.txt").touch()
            result = iter_replays(d)
            self.assertEqual([p.name for p in result], ["game1.SC2Replay", "game2.SC2Replay"])

    def test_sorted_order(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "b.SC2Replay").touch()
            (d / "a.SC2Replay").touch()
            result = iter_replays(d)
            self.assertEqual(result[0].name, "a.SC2Replay")

    def test_empty_folder(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(iter_replays(d), [])


# ---------------------------------------------------------------------------
# Tests: _parse_replay_info
# ---------------------------------------------------------------------------


class TestParseReplayInfo(unittest.TestCase):
    def test_winner_and_races(self):
        info = _fake_replay_info(winner_pid=1, races={1: 1, 2: 3})
        winner_id, player_races = _parse_replay_info(info)
        self.assertEqual(winner_id, 1)
        self.assertEqual(player_races, {1: "terran", 2: "protoss"})

    def test_winner_is_player2(self):
        info = _fake_replay_info(winner_pid=2, races={1: 2, 2: 1})
        winner_id, _ = _parse_replay_info(info)
        self.assertEqual(winner_id, 2)

    def test_undecided_game(self):
        info = MagicMock()
        info.player_info = []
        winner_id, player_races = _parse_replay_info(info)
        self.assertEqual(winner_id, 0)
        self.assertEqual(player_races, {})

    def test_race_int_mapping(self):
        info = _fake_replay_info(winner_pid=1, races={1: 2, 2: 4})
        _, player_races = _parse_replay_info(info)
        self.assertEqual(player_races[1], "zerg")
        self.assertEqual(player_races[2], "random")

    def test_unknown_race_falls_back_to_random(self):
        info = _fake_replay_info(winner_pid=1, races={1: 99})
        _, player_races = _parse_replay_info(info)
        self.assertEqual(player_races[1], "random")


# ---------------------------------------------------------------------------
# Tests: _resolve_player_id
# ---------------------------------------------------------------------------


class TestResolvePlayerId(unittest.TestCase):
    def test_winner_with_known_winner(self):
        self.assertEqual(_resolve_player_id("winner", winner_id=2), 2)

    def test_winner_falls_back_to_1_when_no_winner(self):
        self.assertEqual(_resolve_player_id("winner", winner_id=0), 1)

    def test_explicit_player_id(self):
        self.assertEqual(_resolve_player_id(2, winner_id=1), 2)

    def test_explicit_player_id_string_cast(self):
        self.assertEqual(_resolve_player_id(1, winner_id=0), 1)


# ---------------------------------------------------------------------------
# Tests: _pick_best_action
# ---------------------------------------------------------------------------


class TestPickBestAction(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(_pick_best_action([], "first_non_noop"))

    def test_first_strategy_returns_first_even_if_noop(self):
        calls = [_FakeFunctionCall(0), _FakeFunctionCall(2)]
        result = _pick_best_action(calls, "first")
        self.assertEqual(result.function, 0)

    def test_first_non_noop_skips_noop(self):
        calls = [_FakeFunctionCall(0), _FakeFunctionCall(2), _FakeFunctionCall(3)]
        result = _pick_best_action(calls, "first_non_noop")
        self.assertEqual(result.function, 2)

    def test_first_non_noop_falls_back_to_first_when_all_noop(self):
        calls = [_FakeFunctionCall(0), _FakeFunctionCall(0)]
        result = _pick_best_action(calls, "first_non_noop")
        self.assertEqual(result.function, 0)

    def test_single_action_returned_regardless_of_strategy(self):
        calls = [_FakeFunctionCall(5)]
        self.assertEqual(_pick_best_action(calls, "first_non_noop").function, 5)
        self.assertEqual(_pick_best_action(calls, "first").function, 5)


# ---------------------------------------------------------------------------
# Tests: replay_observations
# ---------------------------------------------------------------------------


class TestReplayObservations(unittest.TestCase):
    """Public generator — controller fully mocked via sys.modules fakes."""

    def setUp(self):
        _reset_fakes()

    def tearDown(self):
        _reset_fakes()

    def _run(self, obs_dicts, actions_per_step, *, winner_pid=1, races=None, player_id="winner", screen_size=64):
        if races is None:
            races = {1: 1, 2: 3}
        spec = SC2_MINIGAME_OBS_SPEC
        info = _fake_replay_info(winner_pid=winner_pid, races=races)
        ctrl = _build_mock_controller(info, obs_dicts, actions_per_step)
        rc = _patch_run_config(ctrl)
        feat = _build_mock_features()
        _configure_fakes(rc, feat)
        pairs = list(
            replay_observations(
                "fake.SC2Replay",
                player_id=player_id,
                obs_spec=spec,
                screen_size=screen_size,
            )
        )
        return pairs, ctrl

    def test_obs_shape_matches_spec(self):
        obs_dicts = [{} for _ in range(3)]
        actions = [[_FakeFunctionCall(2, [[0], [10, 10]])] for _ in range(3)]
        pairs, _ = self._run(obs_dicts, actions)
        self.assertEqual(len(pairs), 3)
        for obs_vec, act_vec in pairs:
            self.assertEqual(obs_vec.shape, (_OBS_DIM,))
            self.assertEqual(obs_vec.dtype, np.float32)

    def test_action_shape_and_dtype(self):
        obs_dicts = [{}]
        actions = [[_FakeFunctionCall(2, [[0], [32, 32]])]]
        pairs, _ = self._run(obs_dicts, actions)
        self.assertEqual(len(pairs), 1)
        _, act_vec = pairs[0]
        self.assertEqual(act_vec.shape, (4,))
        self.assertEqual(act_vec.dtype, np.float32)

    def test_action_fn_idx_preserved(self):
        """The fn_idx in the action vector must match the issued action."""
        # select_army (internal fn_idx=1) → PySC2 id for select_army
        from games.sc2.actions import FUNCTION_IDS

        pysc2_id_for_select_army = next(i for i, name in FUNCTION_IDS.items() if name == "select_army")
        obs_dicts = [{}]
        actions = [[_FakeFunctionCall(pysc2_id_for_select_army)]]
        pairs, _ = self._run(obs_dicts, actions)
        self.assertEqual(len(pairs), 1)
        _, act_vec = pairs[0]
        self.assertEqual(int(act_vec[0]), 1)  # internal fn_idx for select_army

    def test_temporal_order_preserved_via_action_x_coords(self):
        """Action vectors must appear in the same order as steps (checked via x-coord)."""
        screen_size = 64
        obs_dicts = [{} for _ in range(5)]
        # Each step issues Move_screen with a strictly increasing x pixel coordinate.
        # PySC2 fn_id for Move_screen is 2 in FUNCTION_IDS (internal fn_idx 2).
        actions = [[_FakeFunctionCall(2, [[0], [i * 10, 5]])] for i in range(5)]
        pairs, _ = self._run(obs_dicts, actions, screen_size=screen_size)
        self.assertEqual(len(pairs), 5)
        # x_norm = pixel_x / (screen_size - 1); must be monotonically increasing.
        x_coords = [float(act_vec[1]) for _, act_vec in pairs]
        for a, b in zip(x_coords, x_coords[1:]):
            self.assertLess(a, b)

    def test_step_with_no_actions_skipped(self):
        obs_dicts = [{}, {}]
        actions = [[], [_FakeFunctionCall(2, [[0], [32, 32]])]]
        pairs, _ = self._run(obs_dicts, actions)
        self.assertEqual(len(pairs), 1)

    def test_multiple_actions_first_non_noop_selected(self):
        """[no_op, Move_screen] → Move_screen (fn_idx 2) selected."""
        obs_dicts = [{}]
        actions = [[_FakeFunctionCall(0), _FakeFunctionCall(2, [[0], [10, 10]])]]
        pairs, _ = self._run(obs_dicts, actions)
        self.assertEqual(len(pairs), 1)
        _, act_vec = pairs[0]
        self.assertEqual(int(act_vec[0]), 2)

    def test_unknown_fn_id_step_skipped(self):
        """Unknown PySC2 fn_id (not in FUNCTION_IDS) must be skipped."""
        obs_dicts = [{}]
        actions = [[_FakeFunctionCall(9999)]]
        pairs, _ = self._run(obs_dicts, actions)
        self.assertEqual(len(pairs), 0)

    def test_winner_resolution_fallback_to_player1(self):
        """winner_id=0 (undecided) must fall back to player 1."""
        obs_dicts = [{}]
        actions = [[_FakeFunctionCall(1)]]
        pairs, _ = self._run(obs_dicts, actions, winner_pid=0)
        self.assertEqual(len(pairs), 1)

    def test_explicit_player_id_overrides_winner(self):
        """player_id=2 uses player 2 regardless of who won."""
        obs_dicts = [{}]
        actions = [[_FakeFunctionCall(1)]]
        pairs, _ = self._run(obs_dicts, actions, winner_pid=1, player_id=2)
        self.assertEqual(len(pairs), 1)

    def test_controller_step_called_for_each_step(self):
        """controller.step() must be called once per step (including no-action steps)."""
        obs_dicts = [{} for _ in range(4)]
        actions = [
            [],
            [_FakeFunctionCall(2, [[0], [10, 10]])],
            [],
            [_FakeFunctionCall(2, [[0], [20, 20]])],
        ]
        _, ctrl = self._run(obs_dicts, actions)
        self.assertEqual(ctrl.step.call_count, 4)

    def test_unknown_fn_id_still_calls_step(self):
        """Unknown fn_id steps are skipped but controller.step() is still called."""
        obs_dicts = [{}]
        actions = [[_FakeFunctionCall(9999)]]
        _, ctrl = self._run(obs_dicts, actions)
        self.assertEqual(ctrl.step.call_count, 1)


# ---------------------------------------------------------------------------
# Tests: _read_one_replay — single-launch race filtering
# ---------------------------------------------------------------------------


class TestReadOneReplay(unittest.TestCase):
    """Single-launch helper: race check + playback in one SC2 process."""

    def setUp(self):
        _reset_fakes()

    def tearDown(self):
        _reset_fakes()

    def _run(self, obs_dicts, actions_per_step, *, race_filter, winner_pid=1, races=None):
        if races is None:
            races = {1: 1, 2: 3}  # player1=terran, player2=protoss
        spec = SC2_MINIGAME_OBS_SPEC
        info = _fake_replay_info(winner_pid=winner_pid, races=races)
        ctrl = _build_mock_controller(info, obs_dicts, actions_per_step)
        rc = _patch_run_config(ctrl)
        feat = _build_mock_features()
        _configure_fakes(rc, feat)
        return (
            _read_one_replay(
                Path("fake.SC2Replay"),
                player_id="winner",
                race_filter=race_filter,
                obs_spec=spec,
                step_mul=1,
                screen_size=64,
                minimap_size=64,
                multi_action_strategy="first_non_noop",
            ),
            ctrl,
        )

    def test_race_match_returns_true_and_pairs(self):
        obs_dicts = [{}]
        actions = [[_FakeFunctionCall(2, [[0], [10, 10]])]]
        (race_ok, player_race, pairs), _ = self._run(obs_dicts, actions, race_filter="terran")
        self.assertTrue(race_ok)
        self.assertEqual(player_race, "terran")
        self.assertEqual(len(pairs), 1)

    def test_race_mismatch_returns_false_no_pairs(self):
        obs_dicts = [{}]
        actions = [[_FakeFunctionCall(2, [[0], [10, 10]])]]
        (race_ok, player_race, pairs), _ = self._run(obs_dicts, actions, race_filter="zerg")
        self.assertFalse(race_ok)
        self.assertEqual(player_race, "terran")
        self.assertEqual(pairs, [])

    def test_no_race_filter_processes_all(self):
        obs_dicts = [{}]
        actions = [[_FakeFunctionCall(2, [[0], [10, 10]])]]
        (race_ok, _, pairs), _ = self._run(obs_dicts, actions, race_filter=None)
        self.assertTrue(race_ok)
        self.assertEqual(len(pairs), 1)

    def test_race_filter_skips_start_replay(self):
        """start_replay must NOT be called when the race filter drops the replay."""
        spec = SC2_MINIGAME_OBS_SPEC
        # player1=protoss, race_filter="terran" → should skip
        info = _fake_replay_info(winner_pid=1, races={1: 3, 2: 1})
        ctrl = _build_mock_controller(info, [], [])
        rc = _patch_run_config(ctrl)
        feat = _build_mock_features()
        _configure_fakes(rc, feat)
        _read_one_replay(
            Path("fake.SC2Replay"),
            player_id="winner",
            race_filter="terran",
            obs_spec=spec,
            step_mul=1,
            screen_size=64,
            minimap_size=64,
            multi_action_strategy="first_non_noop",
        )
        ctrl.start_replay.assert_not_called()

    def test_player_race_always_returned(self):
        """player_race is populated even when the race filter drops the replay."""
        obs_dicts = [{}]
        actions = [[_FakeFunctionCall(2, [[0], [10, 10]])]]
        (race_ok, player_race, _), _ = self._run(obs_dicts, actions, race_filter="zerg", races={1: 1, 2: 3})
        self.assertFalse(race_ok)
        self.assertEqual(player_race, "terran")


# ---------------------------------------------------------------------------
# Tests: build_dataset
# ---------------------------------------------------------------------------


class TestBuildDataset(unittest.TestCase):
    """Full dataset builder — filesystem + _read_one_replay mocked."""

    def _setup_folder(self, tmp_path: Path, names: list[str]) -> Path:
        folder = tmp_path / "replays"
        folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            (folder / name).touch()
        return folder

    def _mock_read_one(self, return_values: list):
        """Patch _read_one_replay at module level to return successive values."""
        from unittest.mock import patch

        side_effects = iter(return_values)
        return patch(
            "games.sc2.replay_bc._read_one_replay",
            side_effect=lambda *a, **kw: next(side_effects),
        )

    def test_single_replay_correct_npz(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            folder = self._setup_folder(tmp, ["g.SC2Replay"])
            save_path = tmp / "demos.npz"
            spec = SC2_MINIGAME_OBS_SPEC
            obs0 = np.zeros(_OBS_DIM, dtype=np.float32)
            act0 = np.array([2.0, 0.5, 0.5, 0.0], dtype=np.float32)
            pairs = [(obs0, act0), (obs0, act0)]
            with self._mock_read_one([(True, "terran", pairs)]):
                meta = build_dataset(folder, save_path, obs_spec=spec)
            self.assertTrue(save_path.exists())
            self.assertEqual(meta["n_episodes"], 1)
            self.assertEqual(meta["n_steps"], 2)
            self.assertEqual(meta["obs_dim"], _OBS_DIM)

    def test_obs_actions_shape(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            folder = self._setup_folder(tmp, ["g.SC2Replay"])
            save_path = tmp / "demos.npz"
            spec = SC2_MINIGAME_OBS_SPEC
            pairs = [(np.zeros(_OBS_DIM, dtype=np.float32), np.zeros(4, dtype=np.float32)) for _ in range(5)]
            with self._mock_read_one([(True, "terran", pairs)]):
                build_dataset(folder, save_path, obs_spec=spec)
            data = np.load(str(save_path), allow_pickle=False)
            self.assertEqual(data["obs"].shape, (5, _OBS_DIM))
            self.assertEqual(data["actions"].shape, (5, 4))

    def test_episode_boundaries_two_replays(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            folder = self._setup_folder(tmp, ["a.SC2Replay", "b.SC2Replay"])
            save_path = tmp / "demos.npz"
            spec = SC2_MINIGAME_OBS_SPEC
            pairs_a = [(np.zeros(_OBS_DIM, dtype=np.float32), np.zeros(4, dtype=np.float32))] * 3
            pairs_b = [(np.zeros(_OBS_DIM, dtype=np.float32), np.zeros(4, dtype=np.float32))] * 2
            with self._mock_read_one([(True, "terran", pairs_a), (True, "terran", pairs_b)]):
                build_dataset(folder, save_path, obs_spec=spec)
            data = np.load(str(save_path), allow_pickle=False)
            np.testing.assert_array_equal(data["episode_starts"], [0, 3])
            np.testing.assert_array_equal(data["episode_lengths"], [3, 2])
            np.testing.assert_array_equal(data["episode_id"], [0, 0, 0, 1, 1])

    def test_rows_stored_in_temporal_order(self):
        """obs values within an episode must preserve temporal order."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            folder = self._setup_folder(tmp, ["g.SC2Replay"])
            save_path = tmp / "demos.npz"
            spec = SC2_MINIGAME_OBS_SPEC
            pairs = [(np.full(_OBS_DIM, float(i), dtype=np.float32), np.zeros(4, dtype=np.float32)) for i in range(4)]
            with self._mock_read_one([(True, "terran", pairs)]):
                build_dataset(folder, save_path, obs_spec=spec)
            data = np.load(str(save_path), allow_pickle=False)
            for i in range(4):
                self.assertAlmostEqual(data["obs"][i, 0], float(i), places=5)

    def test_meta_json_round_trip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            folder = self._setup_folder(tmp, ["x.SC2Replay"])
            save_path = tmp / "demos.npz"
            spec = SC2_MINIGAME_OBS_SPEC
            pairs = [(np.zeros(_OBS_DIM, dtype=np.float32), np.zeros(4, dtype=np.float32))]
            with self._mock_read_one([(True, "terran", pairs)]):
                build_dataset(folder, save_path, obs_spec=spec, player_id="winner", step_mul=2, screen_size=32)
            data = np.load(str(save_path), allow_pickle=False)
            meta = json.loads(str(data["meta"]))
            self.assertEqual(meta["player_id"], "winner")
            self.assertEqual(meta["step_mul"], 2)
            self.assertEqual(meta["screen_size"], 32)
            self.assertIn("source_filenames", meta)

    def test_race_filter_drops_all_raises(self):
        """All replays dropped by race filter → ValueError with race in message."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            folder = self._setup_folder(tmp, ["a.SC2Replay", "b.SC2Replay"])
            save_path = tmp / "demos.npz"
            spec = SC2_MINIGAME_OBS_SPEC
            with self._mock_read_one([(False, "protoss", []), (False, "zerg", [])]):
                with self.assertRaises(ValueError) as ctx:
                    build_dataset(folder, save_path, obs_spec=spec, race="terran")
            self.assertIn("terran", str(ctx.exception))

    def test_no_replay_files_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            empty = tmp / "empty"
            empty.mkdir()
            with self.assertRaises(ValueError):
                build_dataset(empty, tmp / "demos.npz", obs_spec=SC2_MINIGAME_OBS_SPEC)

    def test_race_any_keeps_all(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            folder = self._setup_folder(tmp, ["a.SC2Replay"])
            save_path = tmp / "demos.npz"
            spec = SC2_MINIGAME_OBS_SPEC
            pairs = [(np.zeros(_OBS_DIM, dtype=np.float32), np.zeros(4, dtype=np.float32))]
            with self._mock_read_one([(True, "zerg", pairs)]):
                meta = build_dataset(folder, save_path, obs_spec=spec, race="any")
            self.assertEqual(meta["n_episodes"], 1)

    def test_source_filenames_recorded(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            folder = self._setup_folder(tmp, ["replay1.SC2Replay"])
            save_path = tmp / "demos.npz"
            spec = SC2_MINIGAME_OBS_SPEC
            pairs = [(np.zeros(_OBS_DIM, dtype=np.float32), np.zeros(4, dtype=np.float32))]
            with self._mock_read_one([(True, "terran", pairs)]):
                meta = build_dataset(folder, save_path, obs_spec=spec)
            self.assertIn("replay1.SC2Replay", meta["source_filenames"])

    def test_episode_id_partitions_all_rows(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            folder = self._setup_folder(tmp, ["a.SC2Replay", "b.SC2Replay"])
            save_path = tmp / "demos.npz"
            spec = SC2_MINIGAME_OBS_SPEC
            pairs2 = [(np.zeros(_OBS_DIM, dtype=np.float32), np.zeros(4, dtype=np.float32))] * 2
            with self._mock_read_one([(True, "t", pairs2), (True, "t", pairs2)]):
                build_dataset(folder, save_path, obs_spec=spec)
            data = np.load(str(save_path), allow_pickle=False)
            self.assertEqual(len(data["episode_id"]), 4)


# ---------------------------------------------------------------------------
# Tests: load_dataset
# ---------------------------------------------------------------------------


class TestLoadDataset(unittest.TestCase):
    """Loader helper — pure NumPy + JSON, no SC2 involved."""

    def _write(
        self,
        path: Path,
        obs: np.ndarray,
        actions: np.ndarray,
        episode_starts: list[int],
        episode_lengths: list[int],
        meta: dict,
    ) -> None:
        ep_starts = np.array(episode_starts, dtype=np.int64)
        ep_lengths = np.array(episode_lengths, dtype=np.int64)
        ep_id = np.concatenate([np.full(n, i, dtype=np.int64) for i, n in enumerate(episode_lengths)])
        np.savez_compressed(
            str(path),
            obs=obs,
            actions=actions,
            episode_starts=ep_starts,
            episode_lengths=ep_lengths,
            episode_id=ep_id,
            meta=np.array(json.dumps(meta)),
        )

    def test_flat_load_returns_correct_keys(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.npz"
            obs = np.zeros((5, _OBS_DIM), dtype=np.float32)
            actions = np.zeros((5, 4), dtype=np.float32)
            self._write(p, obs, actions, [0, 3], [3, 2], {"n_episodes": 2})
            data = load_dataset(p)
            for key in ("obs", "actions", "episode_starts", "episode_lengths", "episode_id", "meta"):
                self.assertIn(key, data)

    def test_flat_load_shapes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.npz"
            obs = np.random.rand(7, _OBS_DIM).astype(np.float32)
            actions = np.random.rand(7, 4).astype(np.float32)
            self._write(p, obs, actions, [0, 4], [4, 3], {})
            data = load_dataset(p)
            self.assertEqual(data["obs"].shape, (7, _OBS_DIM))
            self.assertEqual(data["actions"].shape, (7, 4))

    def test_as_episodes_shapes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.npz"
            obs = np.zeros((10, _OBS_DIM), dtype=np.float32)
            actions = np.zeros((10, 4), dtype=np.float32)
            self._write(p, obs, actions, [0, 6], [6, 4], {"n_episodes": 2})
            episodes = load_dataset(p, as_episodes=True)
            self.assertEqual(len(episodes), 2)
            obs_seq0, act_seq0 = episodes[0]
            obs_seq1, act_seq1 = episodes[1]
            self.assertEqual(obs_seq0.shape, (6, _OBS_DIM))
            self.assertEqual(obs_seq1.shape, (4, _OBS_DIM))
            self.assertEqual(act_seq0.shape, (6, 4))
            self.assertEqual(act_seq1.shape, (4, 4))

    def test_episode_temporal_order_preserved(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.npz"
            obs = np.tile(np.arange(8, dtype=np.float32).reshape(8, 1), _OBS_DIM)[:, :_OBS_DIM]
            actions = np.zeros((8, 4), dtype=np.float32)
            self._write(p, obs, actions, [0, 5], [5, 3], {})
            episodes = load_dataset(p, as_episodes=True)
            obs_seq0, _ = episodes[0]
            for i in range(5):
                self.assertAlmostEqual(float(obs_seq0[i, 0]), float(i), places=5)

    def test_meta_parsed_as_dict(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.npz"
            obs = np.zeros((2, _OBS_DIM), dtype=np.float32)
            actions = np.zeros((2, 4), dtype=np.float32)
            meta_in = {"n_episodes": 1, "step_mul": 4, "race_filter": "terran"}
            self._write(p, obs, actions, [0], [2], meta_in)
            data = load_dataset(p)
            self.assertIsInstance(data["meta"], dict)
            self.assertEqual(data["meta"]["step_mul"], 4)
            self.assertEqual(data["meta"]["race_filter"], "terran")

    def test_episode_starts_partitions_correctly(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.npz"
            obs = np.arange(_OBS_DIM * 9, dtype=np.float32).reshape(9, _OBS_DIM)
            actions = np.zeros((9, 4), dtype=np.float32)
            self._write(p, obs, actions, [0, 4, 7], [4, 3, 2], {})
            data = load_dataset(p)
            np.testing.assert_array_equal(data["episode_starts"], [0, 4, 7])
            np.testing.assert_array_equal(data["episode_lengths"], [4, 3, 2])

    def test_round_trip_save_load(self):
        """build_dataset output is readable by load_dataset without data loss."""
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            folder = tmp / "replays"
            folder.mkdir()
            (folder / "g.SC2Replay").touch()
            save_path = tmp / "demos.npz"
            spec = SC2_MINIGAME_OBS_SPEC

            obs_expected = np.arange(_OBS_DIM, dtype=np.float32)
            act_expected = np.array([2.0, 0.3, 0.7, 0.0], dtype=np.float32)
            pairs = [(obs_expected, act_expected)]

            with patch(
                "games.sc2.replay_bc._read_one_replay",
                return_value=(True, "terran", pairs),
            ):
                build_dataset(folder, save_path, obs_spec=spec)

            data = load_dataset(save_path)
            np.testing.assert_array_almost_equal(data["obs"][0], obs_expected)
            np.testing.assert_array_almost_equal(data["actions"][0], act_expected)

            episodes = load_dataset(save_path, as_episodes=True)
            self.assertEqual(len(episodes), 1)
            np.testing.assert_array_almost_equal(episodes[0][0][0], obs_expected)


# ---------------------------------------------------------------------------
# Tests: fit_bc — MLP target (issue #353)
# ---------------------------------------------------------------------------


class TestFitBCMLP(unittest.TestCase):
    """Tests for fit_bc with target='sc2_reinforce' (MLP gradient descent)."""

    def _make_separable_dataset(self, n: int = 200, fn_idx_label: int = 2, seed: int = 0) -> dict:
        """Create a synthetic dataset where obs uniquely predicts fn_idx."""
        rng = np.random.default_rng(seed)
        obs = rng.standard_normal((n, _OBS_DIM)).astype(np.float32)
        # One-hot-like: always the same fn_idx label, with consistent x/y
        fn_idx = np.full(n, fn_idx_label, dtype=np.float32)
        x_coord = np.full(n, 0.5, dtype=np.float32)
        y_coord = np.full(n, 0.5, dtype=np.float32)
        actions = np.stack([fn_idx, x_coord, y_coord, np.zeros(n, dtype=np.float32)], axis=1)
        return {"obs": obs, "actions": actions}

    def test_loss_decreases_over_epochs(self):
        """BC loss on a simple separable dataset must fall with more epochs."""
        dataset = self._make_separable_dataset(n=400, fn_idx_label=2)
        _, loss_few = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_reinforce",
            hidden_sizes=[16, 8],
            bc_epochs=1,
            bc_learning_rate=0.01,
            bc_batch_size=64,
            bc_ignore_noop=False,
            seed=42,
        )
        _, loss_more = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_reinforce",
            hidden_sizes=[16, 8],
            bc_epochs=20,
            bc_learning_rate=0.01,
            bc_batch_size=64,
            bc_ignore_noop=False,
            seed=42,
        )
        self.assertLess(loss_more, loss_few, "Loss after 20 epochs must be lower than after 1")

    def test_saved_weights_load_as_sc2_reinforce(self):
        """Policy saved after BC must be re-loadable via SC2REINFORCEPolicy.from_cfg."""
        import tempfile

        from games.sc2.sc2_policies import SC2REINFORCEPolicy

        dataset = self._make_separable_dataset(n=100, fn_idx_label=1)
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_reinforce",
            hidden_sizes=[8],
            bc_epochs=2,
            bc_learning_rate=0.01,
            bc_batch_size=50,
            bc_ignore_noop=False,
            seed=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy_weights.yaml"
            policy.save(str(path))
            loaded = SC2REINFORCEPolicy.from_cfg(
                __import__("yaml").safe_load(path.read_text()) or {},
                SC2_MINIGAME_OBS_SPEC,
            )
        # Should run without error
        obs = np.zeros(_OBS_DIM, dtype=np.float32)
        action = loaded(obs)
        self.assertEqual(action.shape, (4,))

    def test_bc_ignore_noop_all_noop_raises(self):
        """Dataset of pure no-ops with bc_ignore_noop=True must raise ValueError."""
        obs = np.zeros((10, _OBS_DIM), dtype=np.float32)
        actions = np.zeros((10, 4), dtype=np.float32)  # fn_idx=0 everywhere
        dataset = {"obs": obs, "actions": actions}
        with self.assertRaises(ValueError):
            fit_bc(dataset, SC2_MINIGAME_OBS_SPEC, target="sc2_reinforce", bc_ignore_noop=True, bc_epochs=1, seed=0)

    def test_bc_ignore_noop_false_keeps_all_steps(self):
        """bc_ignore_noop=False must not filter any steps."""
        obs = np.zeros((10, _OBS_DIM), dtype=np.float32)
        actions = np.zeros((10, 4), dtype=np.float32)  # fn_idx=0 everywhere
        dataset = {"obs": obs, "actions": actions}
        # Should not raise — 10 steps are kept
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_reinforce",
            bc_ignore_noop=False,
            bc_epochs=1,
            bc_learning_rate=0.001,
            bc_batch_size=10,
            seed=0,
        )
        self.assertIsNotNone(policy)

    def test_fn_idx_accuracy_improves(self):
        """After many epochs on a separable dataset, the network favours the correct fn_idx.

        Uses a linear model (hidden_sizes=[]) so the gradient signal reaches
        fn_w directly without vanishing through trunk layers.  Obs are scaled
        by the spec scales so that the normalised inputs have std ≈ 1 and the
        gradient is non-negligible.  Pairwise check: for each sample the logit
        of the correct class (1 or 2) should exceed the logit of the other
        class (2 or 1).
        """
        # Build a dataset where obs[:,0] / scale[0] sign predicts fn_idx.
        # Scale raw obs by spec scales so normalised values have std ~ 1.
        rng = np.random.default_rng(7)
        n = 400
        raw = rng.standard_normal((n, _OBS_DIM)).astype(np.float32)
        obs = (raw * SC2_MINIGAME_OBS_SPEC.scales).astype(np.float32)
        fn_labels = np.where(obs[:, 0] > 0, 2, 1).astype(int)
        actions = np.stack(
            [
                fn_labels.astype(np.float32),
                np.full(n, 0.5, dtype=np.float32),
                np.full(n, 0.5, dtype=np.float32),
                np.zeros(n, np.float32),
            ],
            axis=1,
        )
        dataset = {"obs": obs, "actions": actions}

        # Linear model (no trunk): gradient is direct, convergence is fast and reliable.
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_reinforce",
            hidden_sizes=[],
            bc_epochs=20,
            bc_learning_rate=0.05,
            bc_batch_size=64,
            bc_ignore_noop=False,
            seed=7,
        )
        # Pairwise accuracy: correct class logit > other class logit (1 vs 2 only).
        # h_last = obs_norm when there is no trunk.
        obs_norm = obs / SC2_MINIGAME_OBS_SPEC.scales
        fn_logits = obs_norm.astype(np.float64) @ policy._fn_w.T.astype(np.float64) + policy._fn_b.astype(np.float64)
        logit_correct = fn_logits[np.arange(n), fn_labels]
        wrong_labels = 3 - fn_labels  # label=1→wrong=2, label=2→wrong=1
        logit_wrong = fn_logits[np.arange(n), wrong_labels]
        pair_accuracy = float((logit_correct > logit_wrong).mean())
        self.assertGreater(pair_accuracy, 0.8, f"Expected pairwise accuracy > 0.8, got {pair_accuracy:.3f}")

    def test_dataset_from_path(self):
        """fit_bc accepts a file path (not just a dict)."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "demos.npz"
            obs = np.zeros((20, _OBS_DIM), dtype=np.float32)
            actions = np.column_stack(
                [
                    np.full(20, 2, dtype=np.float32),
                    np.full(20, 0.3, dtype=np.float32),
                    np.full(20, 0.7, dtype=np.float32),
                    np.zeros(20, np.float32),
                ]
            )
            ep_id = np.zeros(20, dtype=np.int64)
            np.savez_compressed(
                str(p),
                obs=obs,
                actions=actions,
                episode_starts=np.array([0], dtype=np.int64),
                episode_lengths=np.array([20], dtype=np.int64),
                episode_id=ep_id,
                meta=np.array(json.dumps({"n_episodes": 1})),
            )
            policy, loss = fit_bc(
                p,
                SC2_MINIGAME_OBS_SPEC,
                target="sc2_reinforce",
                bc_epochs=1,
                bc_learning_rate=0.001,
                bc_ignore_noop=False,
                seed=0,
            )
        self.assertIsNotNone(policy)
        self.assertIsInstance(loss, float)


# ---------------------------------------------------------------------------
# Tests: fit_bc — linear target (issue #353)
# ---------------------------------------------------------------------------


class TestFitBCLinear(unittest.TestCase):
    """Tests for fit_bc with target='sc2_genetic' (closed-form least squares)."""

    def _make_dataset(self, n: int = 100, fn_idx_label: int = 2) -> dict:
        rng = np.random.default_rng(0)
        obs = rng.standard_normal((n, _OBS_DIM)).astype(np.float32)
        actions = np.column_stack(
            [
                np.full(n, fn_idx_label, np.float32),
                np.full(n, 0.6, np.float32),
                np.full(n, 0.4, np.float32),
                np.zeros(n, np.float32),
            ]
        )
        return {"obs": obs, "actions": actions}

    def test_returns_loadable_sc2_multihead_policy(self):
        """fit_bc sc2_genetic must return an SC2MultiHeadLinearPolicy."""
        from games.sc2.sc2_policies import SC2MultiHeadLinearPolicy

        dataset = self._make_dataset(n=80, fn_idx_label=2)
        policy, loss = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_genetic",
            bc_ignore_noop=False,
        )
        self.assertIsInstance(policy, SC2MultiHeadLinearPolicy)
        self.assertIsInstance(loss, float)

    def test_saved_weights_load_as_sc2_genetic(self):
        """Saved linear BC policy loads cleanly as SC2MultiHeadLinearPolicy."""
        import tempfile

        from games.sc2.sc2_policies import SC2MultiHeadLinearPolicy

        dataset = self._make_dataset(n=50, fn_idx_label=1)
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_genetic",
            bc_ignore_noop=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy_weights.yaml"
            policy.save(str(path))
            loaded = SC2MultiHeadLinearPolicy.load(str(path), SC2_MINIGAME_OBS_SPEC)
        obs = np.zeros(_OBS_DIM, dtype=np.float32)
        action = loaded(obs)
        self.assertEqual(action.shape, (4,))

    def test_fn_weights_shape(self):
        """fn_weights must have shape (N_FUNCTION_IDS, obs_dim)."""
        from games.sc2.sc2_policies import N_FUNCTION_IDS

        dataset = self._make_dataset(n=60)
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_genetic",
            bc_ignore_noop=False,
        )
        self.assertEqual(policy._fn_weights.shape, (N_FUNCTION_IDS, _OBS_DIM))

    def test_spatial_weights_shape(self):
        """sp_weights must have shape (2, obs_dim)."""
        dataset = self._make_dataset(n=60, fn_idx_label=2)  # Move_screen is spatial
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_genetic",
            bc_ignore_noop=False,
        )
        self.assertEqual(policy._sp_weights.shape, (2, _OBS_DIM))

    def test_no_spatial_steps_sp_weights_zero(self):
        """When no spatial steps exist, sp_weights should remain all-zero."""
        from games.sc2.actions import SPATIAL_FN_IDS

        # Use a fn_idx not in SPATIAL_FN_IDS (e.g. select_army=1)
        non_spatial = min(k for k in range(118) if k not in SPATIAL_FN_IDS and k != 0)
        dataset = self._make_dataset(n=60, fn_idx_label=non_spatial)
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_genetic",
            bc_ignore_noop=False,
        )
        np.testing.assert_array_equal(policy._sp_weights, np.zeros((2, _OBS_DIM), dtype=np.float32))


# ---------------------------------------------------------------------------
# Tests: run (full pipeline) (issue #353)
# ---------------------------------------------------------------------------


class TestRunBC(unittest.TestCase):
    """Tests for the run() orchestration function."""

    def _synthetic_dataset_dict(self, n: int = 30) -> dict:
        obs = np.zeros((n, _OBS_DIM), dtype=np.float32)
        actions = np.column_stack(
            [
                np.full(n, 2, np.float32),
                np.full(n, 0.5, np.float32),
                np.full(n, 0.5, np.float32),
                np.zeros(n, np.float32),
            ]
        )
        ep_id = np.zeros(n, dtype=np.int64)
        return {
            "obs": obs,
            "actions": actions,
            "episode_starts": np.array([0], dtype=np.int64),
            "episode_lengths": np.array([n], dtype=np.int64),
            "episode_id": ep_id,
            "meta": {"n_episodes": 1, "n_steps": n},
        }

    def _patch_run(self, dataset_dict: dict, target: str = "sc2_reinforce"):
        """Context manager: patch build_dataset + load_dataset so run() never touches replays."""
        from unittest.mock import patch

        meta_return = {
            "n_episodes": 1,
            "n_steps": dataset_dict["obs"].shape[0],
            "obs_dim": _OBS_DIM,
            "n_replays_skipped_race": 0,
            "player_id": "winner",
            "race_filter": None,
        }
        return patch.multiple(
            "games.sc2.replay_bc",
            validate_replay_dir=lambda d, race=None, version=None: [Path(d) / "fake.SC2Replay"],
            build_dataset=lambda *a, **kw: meta_return,
            load_dataset=lambda p: dataset_dict,
        )

    def test_writes_policy_weights_yaml(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            replay_dir = Path(tmp) / "replays"
            replay_dir.mkdir()
            (replay_dir / "g.SC2Replay").touch()
            dataset = self._synthetic_dataset_dict()
            with self._patch_run(dataset, target="sc2_reinforce"):
                bc_run(
                    replay_dir,
                    experiment_dir,
                    SC2_MINIGAME_OBS_SPEC,
                    target="sc2_reinforce",
                    bc_epochs=1,
                    bc_learning_rate=0.001,
                    bc_batch_size=10,
                    bc_ignore_noop=False,
                    seed=0,
                )
            self.assertTrue((experiment_dir / "policy_weights.yaml").exists())

    def test_writes_bc_summary_json(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            replay_dir = Path(tmp) / "replays"
            replay_dir.mkdir()
            (replay_dir / "g.SC2Replay").touch()
            dataset = self._synthetic_dataset_dict()
            with self._patch_run(dataset):
                bc_run(
                    replay_dir,
                    experiment_dir,
                    SC2_MINIGAME_OBS_SPEC,
                    target="sc2_reinforce",
                    bc_epochs=1,
                    bc_learning_rate=0.001,
                    bc_batch_size=10,
                    bc_ignore_noop=False,
                    seed=0,
                )
            summary_file = experiment_dir / "bc_summary.json"
            self.assertTrue(summary_file.exists())
            loaded = json.loads(summary_file.read_text())
            for key in (
                "n_replays_kept",
                "n_episodes",
                "n_pairs",
                "fn_idx_histogram",
                "bc_player_id",
                "bc_race",
                "bc_target",
                "final_bc_loss",
            ):
                self.assertIn(key, loaded, f"Missing key '{key}' in bc_summary.json")

    def test_summary_fn_idx_histogram_covers_actions(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            replay_dir = Path(tmp) / "replays"
            replay_dir.mkdir()
            (replay_dir / "g.SC2Replay").touch()
            dataset = self._synthetic_dataset_dict(n=30)  # all fn_idx=2
            with self._patch_run(dataset):
                summary = bc_run(
                    replay_dir,
                    experiment_dir,
                    SC2_MINIGAME_OBS_SPEC,
                    target="sc2_reinforce",
                    bc_epochs=1,
                    bc_batch_size=10,
                    bc_ignore_noop=False,
                    seed=0,
                )
            histogram = summary["fn_idx_histogram"]
            self.assertIn(2, histogram)
            self.assertEqual(histogram[2], 30)

    def test_writes_trainer_state_for_mlp_target(self):
        """run() with sc2_reinforce must also write trainer_state.npz."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            replay_dir = Path(tmp) / "replays"
            replay_dir.mkdir()
            (replay_dir / "g.SC2Replay").touch()
            dataset = self._synthetic_dataset_dict()
            with self._patch_run(dataset, target="sc2_reinforce"):
                bc_run(
                    replay_dir,
                    experiment_dir,
                    SC2_MINIGAME_OBS_SPEC,
                    target="sc2_reinforce",
                    bc_epochs=1,
                    bc_batch_size=10,
                    bc_ignore_noop=False,
                    seed=0,
                )
            self.assertTrue((experiment_dir / "trainer_state.npz").exists())

    def test_linear_target_writes_policy_weights(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            replay_dir = Path(tmp) / "replays"
            replay_dir.mkdir()
            (replay_dir / "g.SC2Replay").touch()
            dataset = self._synthetic_dataset_dict()
            with self._patch_run(dataset, target="sc2_genetic"):
                bc_run(
                    replay_dir,
                    experiment_dir,
                    SC2_MINIGAME_OBS_SPEC,
                    target="sc2_genetic",
                    bc_ignore_noop=False,
                )
            self.assertTrue((experiment_dir / "policy_weights.yaml").exists())
            summary_file = experiment_dir / "bc_summary.json"
            self.assertTrue(summary_file.exists())
            summary = json.loads(summary_file.read_text())
            self.assertEqual(summary["bc_target"], "sc2_genetic")

    def test_winner_player_id_default_in_summary(self):
        """bc_player_id defaults to 'winner' in bc_summary.json."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            replay_dir = Path(tmp) / "replays"
            replay_dir.mkdir()
            (replay_dir / "g.SC2Replay").touch()
            dataset = self._synthetic_dataset_dict()
            with self._patch_run(dataset):
                summary = bc_run(
                    replay_dir,
                    experiment_dir,
                    SC2_MINIGAME_OBS_SPEC,
                    target="sc2_reinforce",
                    bc_epochs=1,
                    bc_batch_size=10,
                    bc_ignore_noop=False,
                    seed=0,
                    # Do NOT pass player_id — should default to "winner"
                )
            self.assertEqual(summary["bc_player_id"], "winner")

    def test_race_filter_passthrough_in_summary(self):
        """bc_race is recorded in bc_summary.json."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "exp"
            replay_dir = Path(tmp) / "replays"
            replay_dir.mkdir()
            (replay_dir / "g.SC2Replay").touch()
            dataset = self._synthetic_dataset_dict()
            with self._patch_run(dataset):
                summary = bc_run(
                    replay_dir,
                    experiment_dir,
                    SC2_MINIGAME_OBS_SPEC,
                    target="sc2_reinforce",
                    race="terran",
                    bc_epochs=1,
                    bc_batch_size=10,
                    bc_ignore_noop=False,
                    seed=0,
                )
            self.assertEqual(summary["bc_race"], "terran")


# ---------------------------------------------------------------------------
# Tests: --bc CLI guard (issue #353)
# ---------------------------------------------------------------------------


class TestBCCLIMain(unittest.TestCase):
    """Tests for --bc CLI flag behaviour in main.py."""

    def _parse(self, argv: list[str]) -> "argparse.Namespace":  # noqa: F821
        from main import _build_arg_parser

        return _build_arg_parser().parse_args(argv)

    def test_bc_flag_parsed(self):
        args = self._parse(["myexp", "--game", "sc2", "--bc", "--replay-dir", "/tmp/r"])
        self.assertTrue(args.bc)
        self.assertEqual(args.replay_dir, "/tmp/r")

    def test_bc_default_player_is_winner(self):
        """--bc-player defaults to None (resolved to 'winner' in _run_bc_sc2)."""
        args = self._parse(["myexp", "--game", "sc2", "--bc", "--replay-dir", "/r"])
        self.assertIsNone(args.bc_player)

    def test_bc_player_choices(self):
        for choice in ["winner", "1", "2"]:
            args = self._parse(["myexp", "--game", "sc2", "--bc", "--replay-dir", "/r", "--bc-player", choice])
            self.assertEqual(args.bc_player, choice)

    def test_bc_race_choices(self):
        for race in ["terran", "protoss", "zerg", "any"]:
            args = self._parse(["myexp", "--game", "sc2", "--bc", "--replay-dir", "/r", "--bc-race", race])
            self.assertEqual(args.bc_race, race)

    def test_bc_target_choices(self):
        for target in ["sc2_reinforce", "sc2_genetic"]:
            args = self._parse(["myexp", "--game", "sc2", "--bc", "--replay-dir", "/r", "--bc-target", target])
            self.assertEqual(args.bc_target, target)

    def test_bc_rejected_for_non_sc2(self):
        """--bc with --game != sc2 must raise SystemExit."""

        from main import main as main_fn

        with unittest.mock.patch(
            "sys.argv",
            ["main.py", "myexp", "--game", "tmnf", "--bc", "--replay-dir", "/r"],
        ):
            with self.assertRaises(SystemExit) as ctx:
                main_fn()
        self.assertNotEqual(ctx.exception.code, 0)

    def test_bc_mutually_exclusive_with_play(self):
        with self.assertRaises(SystemExit):
            self._parse(["myexp", "--game", "sc2", "--bc", "--play", "--replay-dir", "/r"])

    def test_bc_mutually_exclusive_with_eval(self):
        with self.assertRaises(SystemExit):
            self._parse(["myexp", "--game", "sc2", "--bc", "--eval", "--replay-dir", "/r"])


# ---------------------------------------------------------------------------
# Tests: fit_bc — new policy targets (issue #354)
# ---------------------------------------------------------------------------


def _make_flat_dataset(n: int = 80, fn_idx_label: int = 2) -> dict:
    """Build a minimal flat dataset dict for non-LSTM BC tests."""
    rng = np.random.default_rng(42)
    obs = rng.standard_normal((n, _OBS_DIM)).astype(np.float32)
    actions = np.column_stack(
        [
            np.full(n, fn_idx_label, np.float32),
            np.full(n, 0.5, np.float32),
            np.full(n, 0.5, np.float32),
            np.zeros(n, np.float32),
        ]
    )
    ep_starts = np.array([0, n // 2], dtype=np.int64)
    ep_lengths = np.array([n // 2, n // 2], dtype=np.int64)
    ep_id = np.concatenate([np.zeros(n // 2, np.int64), np.ones(n // 2, np.int64)])
    return {
        "obs": obs,
        "actions": actions,
        "episode_starts": ep_starts,
        "episode_lengths": ep_lengths,
        "episode_id": ep_id,
        "meta": {"n_episodes": 2, "n_steps": n},
    }


class TestFitBCCMAES(unittest.TestCase):
    """Tests for fit_bc with target='sc2_cmaes' (issue #354)."""

    def _make_dataset(self, n: int = 80) -> dict:
        return _make_flat_dataset(n=n, fn_idx_label=2)

    def test_returns_sc2cmaes_policy(self):
        """fit_bc sc2_cmaes must return an SC2CMAESPolicy."""
        from games.sc2.sc2_policies import SC2CMAESPolicy

        dataset = self._make_dataset()
        policy, loss = fit_bc(dataset, SC2_MINIGAME_OBS_SPEC, target="sc2_cmaes", bc_ignore_noop=False)
        self.assertIsInstance(policy, SC2CMAESPolicy)
        self.assertIsInstance(loss, float)

    def test_champion_is_set(self):
        """After BC fit, _champion should not be None."""
        dataset = self._make_dataset()
        policy, _ = fit_bc(dataset, SC2_MINIGAME_OBS_SPEC, target="sc2_cmaes", bc_ignore_noop=False)
        self.assertIsNotNone(policy._champion)

    def test_dist_mean_equals_champion_flat(self):
        """Distribution mean must equal champion.to_flat()."""

        dataset = self._make_dataset()
        policy, _ = fit_bc(dataset, SC2_MINIGAME_OBS_SPEC, target="sc2_cmaes", bc_ignore_noop=False)
        np.testing.assert_allclose(
            policy._dist._mean,
            policy._champion.to_flat().astype(np.float64),
            rtol=1e-5,
        )

    def test_champion_callable(self):
        """Champion policy should produce a valid 4-element action."""
        dataset = self._make_dataset()
        policy, _ = fit_bc(dataset, SC2_MINIGAME_OBS_SPEC, target="sc2_cmaes", bc_ignore_noop=False)
        obs = np.zeros(_OBS_DIM, dtype=np.float32)
        action = policy(obs)
        self.assertEqual(action.shape, (4,))


class TestFitBCNeuralNet(unittest.TestCase):
    """Tests for fit_bc with target='sc2_neural_net' (issue #354)."""

    def _make_dataset(self, n: int = 60) -> dict:
        return _make_flat_dataset(n=n, fn_idx_label=2)

    def test_returns_sc2neuralnet_policy(self):
        """fit_bc sc2_neural_net must return an SC2NeuralNetPolicy."""
        from games.sc2.sc2_policies import SC2NeuralNetPolicy

        dataset = self._make_dataset()
        policy, loss = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_neural_net",
            hidden_sizes=[8],
            bc_epochs=1,
            bc_ignore_noop=False,
            seed=0,
        )
        self.assertIsInstance(policy, SC2NeuralNetPolicy)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)

    def test_weight_shapes_match_architecture(self):
        """Layer weight shapes must match [obs_dim, hidden, 4]."""
        dataset = self._make_dataset()
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_neural_net",
            hidden_sizes=[8, 8],
            bc_epochs=1,
            bc_ignore_noop=False,
            seed=0,
        )
        self.assertEqual(policy._weights[0].shape, (8, _OBS_DIM))
        self.assertEqual(policy._weights[1].shape, (8, 8))
        self.assertEqual(policy._weights[2].shape, (4, 8))

    def test_callable_after_fit(self):
        """Policy must produce a (4,) action after BC fit."""
        dataset = self._make_dataset()
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_neural_net",
            hidden_sizes=[4],
            bc_epochs=1,
            bc_ignore_noop=False,
            seed=0,
        )
        obs = np.zeros(_OBS_DIM, dtype=np.float32)
        action = policy(obs)
        self.assertEqual(action.shape, (4,))

    def test_loss_decreases_over_epochs(self):
        """Loss should decrease (or stay flat) over multiple epochs on a small dataset."""
        rng = np.random.default_rng(10)
        n = 200
        raw_obs = rng.standard_normal((n, _OBS_DIM)).astype(np.float32)
        obs = (raw_obs * SC2_MINIGAME_OBS_SPEC.scales).astype(np.float32)
        # All same fn_idx: simple regression target
        actions = np.column_stack(
            [
                np.full(n, 2, np.float32),
                np.full(n, 0.7, np.float32),
                np.full(n, 0.3, np.float32),
                np.zeros(n, np.float32),
            ]
        )
        dataset = {"obs": obs, "actions": actions}
        _, loss_1 = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_neural_net",
            hidden_sizes=[],
            bc_epochs=1,
            bc_learning_rate=0.01,
            bc_ignore_noop=False,
            seed=0,
        )
        _, loss_20 = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_neural_net",
            hidden_sizes=[],
            bc_epochs=20,
            bc_learning_rate=0.01,
            bc_ignore_noop=False,
            seed=0,
        )
        self.assertLessEqual(loss_20, loss_1 + 1e-3, "Loss should not increase over more epochs")

    def test_saved_weights_reload(self):
        """SC2NeuralNetPolicy.to_cfg() / from_cfg() round-trip after BC."""
        import tempfile

        from games.sc2.sc2_policies import SC2NeuralNetPolicy

        dataset = self._make_dataset()
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_neural_net",
            hidden_sizes=[4],
            bc_epochs=1,
            bc_ignore_noop=False,
            seed=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path as _Path

            path = str(_Path(tmp) / "policy_weights.yaml")
            policy.save(path)
            loaded = SC2NeuralNetPolicy.from_cfg(__import__("yaml").safe_load(open(path)), SC2_MINIGAME_OBS_SPEC)
        obs = np.zeros(_OBS_DIM, dtype=np.float32)
        action = loaded(obs)
        self.assertEqual(action.shape, (4,))


class TestFitBCDQN(unittest.TestCase):
    """Tests for fit_bc with target='sc2_neural_dqn' (issue #354)."""

    def _make_dataset(self, n: int = 60, fn_idx_label: int = 2) -> dict:
        return _make_flat_dataset(n=n, fn_idx_label=fn_idx_label)

    def test_returns_sc2neuraldqn_policy(self):
        """fit_bc sc2_neural_dqn must return an SC2NeuralDQNPolicy."""
        from games.sc2.sc2_policies import SC2NeuralDQNPolicy

        dataset = self._make_dataset()
        policy, loss = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_neural_dqn",
            bc_ignore_noop=False,
            seed=0,
        )
        self.assertIsInstance(policy, SC2NeuralDQNPolicy)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)

    def test_replay_buffer_filled(self):
        """Replay buffer should have transitions after BC fill."""
        dataset = self._make_dataset(n=40)
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_neural_dqn",
            bc_ignore_noop=False,
            seed=0,
        )
        # Should have n transitions pushed
        self.assertGreater(len(policy._replay), 0)

    def test_fill_fraction_as_bc_loss(self):
        """bc_loss should be fill_fraction = transitions / buffer_capacity."""
        n = 50
        dataset = self._make_dataset(n=n)
        policy, bc_loss = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_neural_dqn",
            bc_ignore_noop=False,
            seed=0,
        )
        expected = len(policy._replay) / max(policy._buf_maxlen, 1)
        self.assertAlmostEqual(bc_loss, expected, places=6)

    def test_episode_starts_used_for_done_flags(self):
        """Transitions at episode boundaries should be marked done."""
        n = 10
        obs = np.zeros((n, _OBS_DIM), dtype=np.float32)
        actions = np.column_stack(
            [
                np.full(n, 2, np.float32),
                np.full(n, 0.5, np.float32),
                np.full(n, 0.5, np.float32),
                np.zeros(n, np.float32),
            ]
        )
        # Two episodes: [0..4] and [5..9]
        dataset = {
            "obs": obs,
            "actions": actions,
            "episode_starts": np.array([0, 5], dtype=np.int64),
            "episode_lengths": np.array([5, 5], dtype=np.int64),
            "episode_id": np.array([0] * 5 + [1] * 5, dtype=np.int64),
            "meta": {"n_episodes": 2, "n_steps": n},
        }
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_neural_dqn",
            bc_ignore_noop=False,
            seed=0,
        )
        # Check that step 4 (last of episode 0) is marked done in the buffer.
        buf = list(policy._replay._buf)
        done_flags = [entry[4] for entry in buf]
        self.assertGreater(sum(done_flags), 0)


class TestFitBCLSTM(unittest.TestCase):
    """Tests for fit_bc with target='sc2_lstm' (issue #354)."""

    def _make_dataset(self, n: int = 40) -> dict:
        return _make_flat_dataset(n=n, fn_idx_label=2)

    def test_returns_sc2lstmevolution_policy(self):
        """fit_bc sc2_lstm must return an SC2LSTMEvolutionPolicy."""
        from games.sc2.sc2_policies import SC2LSTMEvolutionPolicy

        dataset = self._make_dataset()
        policy, loss = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_lstm",
            bc_lstm_hidden_size=8,
            bc_epochs=1,
            bc_learning_rate=0.001,
            bc_ignore_noop=False,
            seed=0,
        )
        self.assertIsInstance(policy, SC2LSTMEvolutionPolicy)
        self.assertIsInstance(loss, float)

    def test_champion_is_set(self):
        """After BC fit, _champion must not be None."""
        dataset = self._make_dataset()
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_lstm",
            bc_lstm_hidden_size=8,
            bc_epochs=1,
            bc_ignore_noop=False,
            seed=0,
        )
        self.assertIsNotNone(policy._champion)

    def test_champion_callable(self):
        """Champion must produce a (4,) action."""
        dataset = self._make_dataset()
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_lstm",
            bc_lstm_hidden_size=8,
            bc_epochs=1,
            bc_ignore_noop=False,
            seed=0,
        )
        obs = np.zeros(_OBS_DIM, dtype=np.float32)
        action = policy(obs)
        self.assertEqual(action.shape, (4,))

    def test_loss_is_finite(self):
        """BC loss must be a finite non-negative float."""
        dataset = self._make_dataset()
        _, loss = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_lstm",
            bc_lstm_hidden_size=8,
            bc_epochs=2,
            bc_ignore_noop=False,
            seed=0,
        )
        self.assertFalse(np.isinf(loss))
        self.assertGreaterEqual(loss, 0.0)

    def test_mean_matches_champion_flat(self):
        """Evolution mean should equal champion.to_flat() after seeding."""
        dataset = self._make_dataset()
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_lstm",
            bc_lstm_hidden_size=8,
            bc_epochs=1,
            bc_ignore_noop=False,
            seed=0,
        )
        np.testing.assert_allclose(
            policy._mean,
            policy._champion.to_flat().astype(np.float64),
            rtol=1e-5,
        )

    def test_saved_weights_load_as_sc2_lstm(self):
        """Saved LSTM BC policy must reload cleanly via SC2LSTMPolicy.from_cfg."""
        import tempfile

        from games.sc2.sc2_policies import SC2LSTMPolicy

        dataset = self._make_dataset()
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_lstm",
            bc_lstm_hidden_size=8,
            bc_epochs=1,
            bc_ignore_noop=False,
            seed=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path as _Path

            path = str(_Path(tmp) / "policy_weights.yaml")
            policy.save(path)
            loaded = SC2LSTMPolicy.from_cfg(__import__("yaml").safe_load(open(path)), SC2_MINIGAME_OBS_SPEC)
        self.assertEqual(loaded._hidden_size, 8)
        obs = np.zeros(_OBS_DIM, dtype=np.float32)
        action = loaded(obs)
        self.assertEqual(action.shape, (4,))


class TestFitBCCNN(unittest.TestCase):
    """Tests for fit_bc with target='sc2_cnn' (issue #354)."""

    def _make_dataset(self, n: int = 60) -> dict:
        return _make_flat_dataset(n=n, fn_idx_label=2)

    def test_returns_sc2cnn_evolution_policy(self):
        """fit_bc sc2_cnn must return an SC2CNNEvolutionPolicy."""
        from games.sc2.cnn_policy import SC2CNNEvolutionPolicy

        dataset = self._make_dataset()
        policy, loss = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_cnn",
            n_channels=1,
            bc_ignore_noop=False,
            seed=0,
        )
        self.assertIsInstance(policy, SC2CNNEvolutionPolicy)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)

    def test_champion_is_set(self):
        """_champion must be set after CNN BC fit."""
        dataset = self._make_dataset()
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_cnn",
            n_channels=1,
            bc_ignore_noop=False,
            seed=0,
        )
        self.assertIsNotNone(policy._champion)

    def test_mean_matches_champion_flat(self):
        """_mean should equal champion.to_flat() after seeding."""
        dataset = self._make_dataset()
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_cnn",
            n_channels=1,
            bc_ignore_noop=False,
            seed=0,
        )
        np.testing.assert_allclose(
            policy._mean,
            policy._champion.to_flat().astype(np.float64),
            rtol=1e-5,
        )

    def test_conv_weights_zeroed(self):
        """W1 and W2 (conv layers) must be zeroed in the warm-start champion."""
        dataset = self._make_dataset()
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_cnn",
            n_channels=1,
            bc_ignore_noop=False,
            seed=0,
        )
        champ = policy._champion
        np.testing.assert_array_equal(champ.W1, np.zeros_like(champ.W1))
        np.testing.assert_array_equal(champ.W2, np.zeros_like(champ.W2))

    def test_w3_obs_portion_nonzero(self):
        """The obs-portion of W3 should be non-zero (random projection)."""
        dataset = self._make_dataset()
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_cnn",
            n_channels=1,
            bc_ignore_noop=False,
            seed=0,
        )
        champ = policy._champion
        pool_flat = champ._pool_flat
        obs_portion = champ.W3[:, pool_flat:]
        self.assertFalse(np.all(obs_portion == 0), "W3 obs portion should be non-zero")


class TestFitBCTabular(unittest.TestCase):
    """Tests for fit_bc with targets 'epsilon_greedy' / 'ucb_q' (issue #354)."""

    def _make_dataset(self, n: int = 60) -> dict:
        return _make_flat_dataset(n=n, fn_idx_label=2)

    def test_epsilon_greedy_returns_correct_type(self):
        """fit_bc epsilon_greedy must return an EpsilonGreedyPolicy."""
        from framework.policies import EpsilonGreedyPolicy

        dataset = self._make_dataset()
        policy, loss = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="epsilon_greedy",
            n_bins=2,
            bc_ignore_noop=False,
            seed=0,
        )
        self.assertIsInstance(policy, EpsilonGreedyPolicy)
        self.assertEqual(loss, 0.0)

    def test_ucb_q_returns_correct_type(self):
        """fit_bc ucb_q must return a UCBQPolicy."""
        from framework.policies import UCBQPolicy

        dataset = self._make_dataset()
        policy, loss = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="ucb_q",
            n_bins=2,
            bc_ignore_noop=False,
            seed=0,
        )
        self.assertIsInstance(policy, UCBQPolicy)
        self.assertEqual(loss, 0.0)

    def test_q_table_populated(self):
        """Q-table should have entries after seeding."""
        dataset = self._make_dataset(n=40)
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="epsilon_greedy",
            n_bins=2,
            bc_ignore_noop=False,
            seed=0,
        )
        self.assertGreater(len(policy._q_table), 0)

    def test_n_sa_populated(self):
        """Visit count table _n_sa should be populated."""
        dataset = self._make_dataset(n=40)
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="ucb_q",
            n_bins=2,
            bc_ignore_noop=False,
            seed=0,
        )
        self.assertGreater(len(policy._n_sa), 0)

    def test_q_values_normalised(self):
        """Q-values should be normalised by visit count (sum over actions = 1 for seeded states)."""
        dataset = self._make_dataset(n=60)
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="epsilon_greedy",
            n_bins=2,
            bc_ignore_noop=False,
            seed=0,
        )
        for state, q in policy._q_table.items():
            counts = policy._n_sa.get(state, np.zeros_like(q))
            if counts.sum() > 0:
                # Q-values for seeded actions should be in [0, 1]
                seeded_q = q[counts > 0]
                self.assertTrue(np.all(seeded_q >= 0.0))
                self.assertTrue(np.all(seeded_q <= 1.0 + 1e-6))

    def test_epsilon_greedy_callable(self):
        """Policy must return a (4,) action without raising."""
        dataset = self._make_dataset()
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="epsilon_greedy",
            n_bins=2,
            bc_ignore_noop=False,
            seed=0,
        )
        obs = np.zeros(_OBS_DIM, dtype=np.float32)
        action = policy(obs)
        self.assertEqual(action.shape, (4,))


class TestFitBCUnknownTarget(unittest.TestCase):
    """Tests for fit_bc error handling with invalid/unsupported targets (issue #354)."""

    def _make_dataset(self) -> dict:
        return _make_flat_dataset(n=20, fn_idx_label=1)

    def test_sb3_ppo_raises_value_error(self):
        """SB3 policy targets must raise ValueError."""
        dataset = self._make_dataset()
        with self.assertRaises(ValueError) as ctx:
            fit_bc(dataset, SC2_MINIGAME_OBS_SPEC, target="ppo", bc_ignore_noop=False)
        self.assertIn("SB3", str(ctx.exception))

    def test_sb3_a2c_raises_value_error(self):
        dataset = self._make_dataset()
        with self.assertRaises(ValueError):
            fit_bc(dataset, SC2_MINIGAME_OBS_SPEC, target="a2c", bc_ignore_noop=False)

    def test_sb3_sac_raises_value_error(self):
        dataset = self._make_dataset()
        with self.assertRaises(ValueError):
            fit_bc(dataset, SC2_MINIGAME_OBS_SPEC, target="sac", bc_ignore_noop=False)

    def test_sb3_td3_raises_value_error(self):
        dataset = self._make_dataset()
        with self.assertRaises(ValueError):
            fit_bc(dataset, SC2_MINIGAME_OBS_SPEC, target="td3", bc_ignore_noop=False)

    def test_sb3_qr_dqn_raises_value_error(self):
        dataset = self._make_dataset()
        with self.assertRaises(ValueError):
            fit_bc(dataset, SC2_MINIGAME_OBS_SPEC, target="qr_dqn", bc_ignore_noop=False)

    def test_sb3_recurrent_ppo_raises_value_error(self):
        dataset = self._make_dataset()
        with self.assertRaises(ValueError):
            fit_bc(dataset, SC2_MINIGAME_OBS_SPEC, target="recurrent_ppo", bc_ignore_noop=False)

    def test_unknown_target_raises_value_error(self):
        """Completely unknown target names must raise ValueError."""
        dataset = self._make_dataset()
        with self.assertRaises(ValueError) as ctx:
            fit_bc(dataset, SC2_MINIGAME_OBS_SPEC, target="banana", bc_ignore_noop=False)
        self.assertIn("banana", str(ctx.exception))

    def test_error_message_lists_supported_targets(self):
        """ValueError for unknown target should list the supported targets."""
        dataset = self._make_dataset()
        with self.assertRaises(ValueError) as ctx:
            fit_bc(dataset, SC2_MINIGAME_OBS_SPEC, target="xyz", bc_ignore_noop=False)
        self.assertIn("Supported", str(ctx.exception))


class TestBugFixes354(unittest.TestCase):
    """Regression tests for bugs fixed in issue #354 Copilot review."""

    # --- Fix 1: Q-normalization sums to 1 per state ---

    def test_tabular_q_sums_to_one_per_state(self):
        """After BC fit, Q-values across actions must sum to 1.0 for every state."""
        dataset = _make_flat_dataset(n=60, fn_idx_label=2)
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="epsilon_greedy",
            n_bins=2,
            bc_ignore_noop=False,
            seed=0,
        )
        for state, q in policy._q_table.items():
            self.assertAlmostEqual(float(q.sum()), 1.0, places=5, msg=f"Q sums to {q.sum():.6f} for state {state}")

    # --- Fix 2: DQN terminal transitions have zero next_obs ---

    def test_dqn_terminal_next_obs_is_zeros(self):
        """Transitions marked done=True must have a zero-vector next_obs in the replay buffer."""
        n = 10
        obs = np.ones((n, _OBS_DIM), dtype=np.float32)
        actions = np.column_stack(
            [
                np.full(n, 2, np.float32),
                np.full(n, 0.5, np.float32),
                np.full(n, 0.5, np.float32),
                np.zeros(n, np.float32),
            ]
        )
        dataset = {
            "obs": obs,
            "actions": actions,
            "episode_starts": np.array([0, 5], dtype=np.int64),
            "episode_lengths": np.array([5, 5], dtype=np.int64),
            "episode_id": np.array([0] * 5 + [1] * 5, dtype=np.int64),
            "meta": {"n_episodes": 2, "n_steps": n},
        }
        policy, _ = fit_bc(
            dataset,
            SC2_MINIGAME_OBS_SPEC,
            target="sc2_neural_dqn",
            bc_ignore_noop=False,
            seed=0,
        )
        buf = list(policy._replay._buf)
        for entry in buf:
            # Entry is (obs, action_idx, reward, next_obs, done[, mask])
            next_obs, done = entry[3], entry[4]
            if done:
                np.testing.assert_array_equal(
                    next_obs,
                    np.zeros_like(next_obs),
                    err_msg="next_obs should be zeros for terminal transitions",
                )

    # --- Fix 3: LSTM missing episode keys raises ValueError ---

    def test_lstm_missing_episode_keys_raises_value_error(self):
        """fit_bc sc2_lstm must raise ValueError when episode boundary keys are absent."""
        rng = np.random.default_rng(0)
        obs = rng.standard_normal((20, _OBS_DIM)).astype(np.float32)
        actions = np.column_stack(
            [
                np.full(20, 2, np.float32),
                np.full(20, 0.5, np.float32),
                np.full(20, 0.5, np.float32),
                np.zeros(20, np.float32),
            ]
        )
        dataset_no_boundaries = {"obs": obs, "actions": actions}
        with self.assertRaises(ValueError) as ctx:
            fit_bc(
                dataset_no_boundaries,
                SC2_MINIGAME_OBS_SPEC,
                target="sc2_lstm",
                bc_lstm_hidden_size=8,
                bc_epochs=1,
                bc_ignore_noop=False,
                seed=0,
            )
        self.assertIn("episode_starts", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
