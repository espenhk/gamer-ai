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
    iter_replays,
    load_dataset,
    replay_observations,
    validate_replay_dir,
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


if __name__ == "__main__":
    unittest.main()
