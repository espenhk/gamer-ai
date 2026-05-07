"""Tests for the SC2 client's PySC2 timestep flattening.

We don't want a hard PySC2 dependency in CI, so these tests construct fake
TimeStep-shaped objects (NamedTuples / dicts) and pass them through the
client's flattening path.
"""
import unittest
from collections import namedtuple

import numpy as np

from games.sc2.client import SC2Client
from games.sc2.obs_spec import BASE_OBS_DIM, LADDER_OBS_DIM


# Minimal stand-in for pysc2.lib.named_array.NamedNumpyArray indexed by name.
class _NamedArr:
    def __init__(self, mapping: dict[str, float]):
        self._mapping = mapping

    def __getitem__(self, key: str) -> float:
        return self._mapping[key]

    def get(self, key: str, default=None):
        return self._mapping.get(key, default)


# Stand-in for pysc2.env.environment.TimeStep.
_TimeStep = namedtuple("TimeStep", ["observation", "reward", "step_type"])


def _last_step_type():
    """Sentinel object whose .last() check uses the wrapper's logic."""
    return 2  # PySC2 uses StepType.LAST = 2


class _FakeTimeStep:
    def __init__(self, observation, reward=0.0, last=False):
        self.observation = observation
        self.reward = reward
        self._last = last

    def last(self) -> bool:
        return self._last


class TestSC2ClientMinigameFlatten(unittest.TestCase):

    def setUp(self):
        self.client = SC2Client(map_name="MoveToBeacon")

    def test_minigame_flat_obs_shape(self):
        observation = {
            "player": _NamedArr({
                "minerals": 50, "vespene": 0, "food_used": 1, "food_cap": 15,
                "army_count": 0, "idle_worker_count": 0,
                "warp_gate_count": 0, "larva_count": 0,
            }),
            "single_select": np.zeros((0, 7), dtype=np.int32),
            "multi_select": np.zeros((0, 7), dtype=np.int32),
            "feature_screen": np.zeros((17, 64, 64), dtype=np.int32),
            "score_cumulative": np.array([0]),
        }
        ts = _FakeTimeStep(observation)
        flat, info = self.client._timestep_to_obs_info(ts)
        self.assertEqual(flat.shape, (BASE_OBS_DIM,))
        self.assertEqual(flat.dtype, np.float32)

    def test_score_delta_threading(self):
        """score becomes prev_score on the second call."""
        ob = {
            "player": _NamedArr({
                "minerals": 0, "vespene": 0, "food_used": 0, "food_cap": 0,
                "army_count": 0, "idle_worker_count": 0,
                "warp_gate_count": 0, "larva_count": 0,
            }),
            "feature_screen": np.zeros((17, 64, 64), dtype=np.int32),
            "score_cumulative": np.array([7]),
        }
        ts = _FakeTimeStep(ob)
        _, info = self.client._timestep_to_obs_info(ts)
        self.assertEqual(info["score"], 7.0)
        self.assertEqual(info["prev_score"], 0.0)

        ob["score_cumulative"] = np.array([12])
        _, info2 = self.client._timestep_to_obs_info(_FakeTimeStep(ob))
        self.assertEqual(info2["score"], 12.0)
        self.assertEqual(info2["prev_score"], 7.0)

    def test_player_relative_centroid(self):
        """A non-empty player_relative layer should yield centroid coords."""
        screen = np.zeros((17, 64, 64), dtype=np.int32)
        # Place a single friendly pixel at (10, 20) — channel 5 = player_relative.
        screen[5, 20, 10] = 1  # row=20 → y, col=10 → x
        ob = {
            "player": _NamedArr({
                "minerals": 0, "vespene": 0, "food_used": 0, "food_cap": 0,
                "army_count": 0, "idle_worker_count": 0,
                "warp_gate_count": 0, "larva_count": 0,
            }),
            "feature_screen": screen,
            "score_cumulative": np.array([0]),
        }
        flat, _ = self.client._timestep_to_obs_info(_FakeTimeStep(ob))
        # Indices 7=screen_self_count, 9=screen_self_cx, 10=screen_self_cy.
        self.assertEqual(flat[7], 1.0)
        self.assertAlmostEqual(flat[9], 10.0)
        self.assertAlmostEqual(flat[10], 20.0)

    def test_terminal_outcome_recorded(self):
        """For minigames, player_outcome is always None (timestep.reward is
        the per-step score delta, not a terminal win/loss signal)."""
        ob = {
            "player": _NamedArr({
                "minerals": 0, "vespene": 0, "food_used": 0, "food_cap": 0,
                "army_count": 0, "idle_worker_count": 0,
                "warp_gate_count": 0, "larva_count": 0,
            }),
            "feature_screen": np.zeros((17, 64, 64), dtype=np.int32),
            "score_cumulative": np.array([0]),
        }
        ts = _FakeTimeStep(ob, reward=1.0, last=True)
        _, info = self.client._timestep_to_obs_info(ts)
        self.assertIsNone(info["player_outcome"])
        self.assertTrue(info["is_last"])


class TestSC2ClientLadderFlatten(unittest.TestCase):

    def setUp(self):
        self.client = SC2Client(map_name="Simple64")

    def test_ladder_flat_obs_shape(self):
        ob = {
            "player": _NamedArr({
                "minerals": 50, "vespene": 0, "food_used": 12, "food_cap": 15,
                "army_count": 1, "idle_worker_count": 2,
                "warp_gate_count": 0, "larva_count": 3,
            }),
            "feature_screen": np.zeros((17, 64, 64), dtype=np.int32),
            "feature_minimap": np.zeros((11, 64, 64), dtype=np.int32),
            "score_cumulative": np.array([0]),
            "game_loop": np.array([100]),
        }
        flat, _ = self.client._timestep_to_obs_info(_FakeTimeStep(ob))
        self.assertEqual(flat.shape, (LADDER_OBS_DIM,))

    def _name_idx(self, name: str) -> int:
        """Look up a feature index by name on the active spec."""
        return self.client._spec.names.index(name)

    def test_visibility_tracking(self):
        """Explored fraction should be monotonically non-decreasing."""
        mmap = np.zeros((11, 64, 64), dtype=np.int32)
        # Channel 1 = visibility_map; mark a quadrant visible (value 2).
        mmap[1, :32, :32] = 2
        ob = {
            "player": _NamedArr({
                "minerals": 0, "vespene": 0, "food_used": 0, "food_cap": 0,
                "army_count": 0, "idle_worker_count": 0,
                "warp_gate_count": 0, "larva_count": 0,
            }),
            "feature_screen": np.zeros((17, 64, 64), dtype=np.int32),
            "feature_minimap": mmap,
            "score_cumulative": np.array([0]),
            "game_loop": np.array([0]),
        }
        flat1, _ = self.client._timestep_to_obs_info(_FakeTimeStep(ob))
        # Now zero visibility but explored mask should persist.
        ob["feature_minimap"] = np.zeros((11, 64, 64), dtype=np.int32)
        flat2, _ = self.client._timestep_to_obs_info(_FakeTimeStep(ob))

        vis_i = self._name_idx("minimap_visible_frac")
        exp_i = self._name_idx("minimap_explored_frac")
        self.assertGreater(flat1[vis_i], 0.0)  # visible_frac > 0 first call
        self.assertEqual(flat2[vis_i], 0.0)    # visible_frac = 0 second call
        # Explored remains > 0 in both calls.
        self.assertGreater(flat1[exp_i], 0.0)
        self.assertGreater(flat2[exp_i], 0.0)
        self.assertGreaterEqual(flat2[exp_i], flat1[exp_i])

    def test_visibility_fogged_not_counted_as_visible(self):
        """visible_frac uses == 2; fogged tiles (value 1) must not be counted."""
        mmap = np.zeros((11, 64, 64), dtype=np.int32)
        # Mark top-left quadrant as fogged (1) and bottom-right as visible (2).
        mmap[1, :32, :32] = 1   # fogged — explored but not currently visible
        mmap[1, 32:, 32:] = 2   # fully visible
        ob = {
            "player": _NamedArr({
                "minerals": 0, "vespene": 0, "food_used": 0, "food_cap": 0,
                "army_count": 0, "idle_worker_count": 0,
                "warp_gate_count": 0, "larva_count": 0,
            }),
            "feature_screen": np.zeros((17, 64, 64), dtype=np.int32),
            "feature_minimap": mmap,
            "score_cumulative": np.array([0]),
            "game_loop": np.array([0]),
        }
        flat, _ = self.client._timestep_to_obs_info(_FakeTimeStep(ob))
        vis_i = self._name_idx("minimap_visible_frac")
        exp_i = self._name_idx("minimap_explored_frac")
        # visible_frac should only count the 32×32 visible quadrant.
        expected_visible = (32 * 32) / (64 * 64)
        self.assertAlmostEqual(float(flat[vis_i]), expected_visible, places=5)
        # explored_frac counts both fogged (1) and visible (2).
        expected_explored = (32 * 32 + 32 * 32) / (64 * 64)
        self.assertAlmostEqual(float(flat[exp_i]), expected_explored, places=5)

    def test_ladder_terminal_outcome_set(self):
        """For ladder maps, player_outcome is set from timestep.reward on last step."""
        ob = {
            "player": _NamedArr({
                "minerals": 0, "vespene": 0, "food_used": 0, "food_cap": 0,
                "army_count": 0, "idle_worker_count": 0,
                "warp_gate_count": 0, "larva_count": 0,
            }),
            "feature_screen": np.zeros((17, 64, 64), dtype=np.int32),
            "feature_minimap": np.zeros((11, 64, 64), dtype=np.int32),
            "score_cumulative": np.array([0]),
            "game_loop": np.array([0]),
        }
        # Simulate a win (reward=1) on the terminal step.
        ts = _FakeTimeStep(ob, reward=1.0, last=True)
        _, info = self.client._timestep_to_obs_info(ts)
        self.assertEqual(info["player_outcome"], 1.0)
        self.assertTrue(info["is_last"])

    def test_ladder_non_terminal_outcome_is_none(self):
        """player_outcome is None on non-terminal ladder steps."""
        ob = {
            "player": _NamedArr({
                "minerals": 0, "vespene": 0, "food_used": 0, "food_cap": 0,
                "army_count": 0, "idle_worker_count": 0,
                "warp_gate_count": 0, "larva_count": 0,
            }),
            "feature_screen": np.zeros((17, 64, 64), dtype=np.int32),
            "feature_minimap": np.zeros((11, 64, 64), dtype=np.int32),
            "score_cumulative": np.array([0]),
            "game_loop": np.array([0]),
        }
        ts = _FakeTimeStep(ob, reward=0.0, last=False)
        _, info = self.client._timestep_to_obs_info(ts)
        self.assertIsNone(info["player_outcome"])


# ---------------------------------------------------------------------------
# Issue #126: feature-block extractors and rich preset assembly
# ---------------------------------------------------------------------------

class TestSC2ClientFeatureExtractors(unittest.TestCase):
    """Targeted tests for the per-block extractors added in #126."""

    def setUp(self):
        from games.sc2.client import SC2Client
        self.client = SC2Client(map_name="Simple64", obs_spec_preset="rich")

    def _ladder_ob(self):
        return {
            "player": _NamedArr({
                "minerals": 100.0, "vespene": 50.0,
                "food_used": 10.0, "food_cap": 15.0,
                "army_count": 3.0, "idle_worker_count": 2.0,
                "warp_gate_count": 0.0, "larva_count": 1.0,
                "food_workers": 6.0, "food_army": 4.0,
            }),
            "feature_screen":  np.zeros((17, 64, 64), dtype=np.int32),
            "feature_minimap": np.zeros((11, 64, 64), dtype=np.int32),
            "score_cumulative": np.arange(13, dtype=np.int32) * 10,
            "game_loop":       np.array([100]),
            "available_actions": np.array([0, 1, 2], dtype=np.int32),
        }

    def test_player_features_includes_food_split(self):
        feats = self.client._player_features(self._ladder_ob())
        self.assertEqual(feats["food_workers"], 6.0)
        self.assertEqual(feats["food_army"],    4.0)

    def test_score_features_returns_all_13_named_entries(self):
        feats = self.client._score_features(self._ladder_ob())
        self.assertIn("score_total", feats)
        self.assertIn("collected_minerals", feats)
        self.assertIn("spent_vespene", feats)
        # score_cumulative was [0,10,20,...,120] — collected_minerals is index 7.
        self.assertEqual(feats["collected_minerals"], 70.0)

    def test_score_features_missing_score_array_returns_zeros(self):
        feats = self.client._score_features({})
        self.assertEqual(feats["score_total"], 0.0)
        self.assertEqual(feats["collected_minerals"], 0.0)

    def test_screen_summary_friendly_only(self):
        screen = np.zeros((17, 64, 64), dtype=np.int32)
        screen[5, 10, 20] = 1   # one friendly pixel
        feats = self.client._screen_summary_features(screen)
        self.assertEqual(feats["screen_self_count"],  1.0)
        self.assertEqual(feats["screen_enemy_count"], 0.0)
        self.assertAlmostEqual(feats["screen_self_cx"], 20.0)
        self.assertAlmostEqual(feats["screen_self_cy"], 10.0)

    def test_quadrant_features_split_correctly(self):
        screen = np.zeros((17, 64, 64), dtype=np.int32)
        # NW (top-left): row<32, col<32
        screen[5, 5, 5] = 1
        # SE (bottom-right): row>=32, col>=32
        screen[5, 50, 50] = 4
        feats = self.client._quadrant_features(screen)
        self.assertEqual(feats["screen_self_NW_count"],  1.0)
        self.assertEqual(feats["screen_self_NE_count"],  0.0)
        self.assertEqual(feats["screen_enemy_SE_count"], 1.0)
        self.assertEqual(feats["screen_enemy_NE_count"], 0.0)

    def test_topk_enemy_features_distance_buckets(self):
        screen = np.zeros((17, 64, 64), dtype=np.int32)
        # friendly centroid → (32, 32)
        screen[5, 32, 32] = 1
        # enemy at distance 5
        screen[5, 32, 37] = 4
        # enemy at distance 20
        screen[5, 32, 52] = 4
        # enemy at distance 30 (outside the 24-radius bucket)
        screen[5, 32, 62] = 4
        feats = self.client._topk_enemy_features(screen)
        self.assertEqual(feats["topk_enemy_within_8"],  1.0)
        self.assertEqual(feats["topk_enemy_within_24"], 2.0)

    def test_last_action_features_track_issued_fn(self):
        # Manually simulate _action_to_call having been called with fn_idx=2.
        self.client._last_fn_idx = 2
        feats = self.client._last_action_features()
        self.assertEqual(feats["last_fn_2"], 1.0)
        self.assertEqual(feats["last_fn_0"], 0.0)

    def test_rich_preset_flat_obs_dim(self):
        """Rich preset assembled flat obs has the expected length."""
        from games.sc2.obs_spec import RICH_OBS_DIM
        ob = self._ladder_ob()
        flat, _ = self.client._timestep_to_obs_info(_FakeTimeStep(ob))
        self.assertEqual(flat.shape, (RICH_OBS_DIM,))

    def test_minigame_preset_unaffected_by_new_features(self):
        """Minigame default still produces a BASE_OBS_DIM flat obs."""
        from games.sc2.client import SC2Client
        from games.sc2.obs_spec import BASE_OBS_DIM
        c = SC2Client(map_name="MoveToBeacon")
        ob = self._ladder_ob()  # extra fields are tolerated
        flat, _ = c._timestep_to_obs_info(_FakeTimeStep(ob))
        self.assertEqual(flat.shape, (BASE_OBS_DIM,))

    def test_minimap_enemy_centroid_populated(self):
        """minimap_enemy_cx/cy (beacon locator) are computed from the minimap
        player_relative layer and included in the minigame flat obs."""
        from games.sc2.client import SC2Client
        from games.sc2.obs_spec import BASE_OBS_DIM, SC2_MINIGAME_OBS_SPEC

        c = SC2Client(map_name="MoveToBeacon")
        minimap = np.zeros((11, 64, 64), dtype=np.int32)
        # Place a fake beacon (enemy, player_relative=4) at pixel (40, 20).
        minimap[5, 20, 40] = 4   # channel 5 = player_relative
        ob = {
            "player": _NamedArr({
                "minerals": 0, "vespene": 0, "food_used": 1, "food_cap": 15,
                "army_count": 1, "idle_worker_count": 0,
                "warp_gate_count": 0, "larva_count": 0,
            }),
            "feature_screen":  np.zeros((17, 64, 64), dtype=np.int32),
            "feature_minimap": minimap,
            "score_cumulative": np.array([0]),
        }
        flat, info = c._timestep_to_obs_info(_FakeTimeStep(ob))
        self.assertEqual(flat.shape, (BASE_OBS_DIM,))

        cx_idx = SC2_MINIGAME_OBS_SPEC.names.index("minimap_enemy_cx")
        cy_idx = SC2_MINIGAME_OBS_SPEC.names.index("minimap_enemy_cy")
        # Raw centroid values (flat obs is unnormalized): cx=40.0, cy=20.0.
        self.assertAlmostEqual(flat[cx_idx], 40.0, places=3)
        self.assertAlmostEqual(flat[cy_idx], 20.0, places=3)

    def test_minimap_enemy_centroid_zero_when_no_beacon(self):
        """When there are no enemy pixels on the minimap (e.g. during the brief
        beacon-scored / pre-respawn transition) the centroid defaults to 0.0."""
        from games.sc2.client import SC2Client
        from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC

        c = SC2Client(map_name="MoveToBeacon")
        # All-zero minimap: no enemies present.
        ob = {
            "player": _NamedArr({
                "minerals": 0, "vespene": 0, "food_used": 1, "food_cap": 15,
                "army_count": 1, "idle_worker_count": 0,
                "warp_gate_count": 0, "larva_count": 0,
            }),
            "feature_screen":  np.zeros((17, 64, 64), dtype=np.int32),
            "feature_minimap": np.zeros((11, 64, 64), dtype=np.int32),
            "score_cumulative": np.array([0]),
        }
        flat, _ = c._timestep_to_obs_info(_FakeTimeStep(ob))

        cx_idx = SC2_MINIGAME_OBS_SPEC.names.index("minimap_enemy_cx")
        cy_idx = SC2_MINIGAME_OBS_SPEC.names.index("minimap_enemy_cy")
        self.assertAlmostEqual(flat[cx_idx], 0.0, places=5)
        self.assertAlmostEqual(flat[cy_idx], 0.0, places=5)


# ---------------------------------------------------------------------------
# Issue #124 / #121: Move_screen blocked by selection precondition should
# fall back to select_army rather than silently no-op'ing.
# ---------------------------------------------------------------------------

class _FakeFn:
    def __init__(self, fn_id: int):
        self.id = fn_id


class _FakeFunctions:
    no_op       = _FakeFn(0)
    select_army = _FakeFn(7)
    Move_screen = _FakeFn(331)


class _FakeFunctionCall:
    def __init__(self, function: int, args: list):
        self.function = function
        self.args = args


class _FakeActionsModule:
    FUNCTIONS = _FakeFunctions
    FunctionCall = _FakeFunctionCall


class _FakeLib:
    actions = _FakeActionsModule


class _FakePySc2:
    lib = _FakeLib


class TestSC2ClientActionFallback(unittest.TestCase):
    """Verify the auto-select-army fallback (#121, #124)."""

    def setUp(self):
        from unittest.mock import patch
        # Patch pysc2 + the games.sc2.actions.action_to_function_call helper so
        # _action_to_call can be exercised without a real pysc2 install.
        patcher_pysc2 = patch.dict("sys.modules", {
            "pysc2":         _FakePySc2,
            "pysc2.lib":     _FakePySc2.lib,
            "pysc2.lib.actions": _FakeActionsModule,
        })
        patcher_pysc2.start()
        self.addCleanup(patcher_pysc2.stop)

        from games.sc2 import client as client_mod
        # action_to_function_call also imports pysc2 internally; replace it
        # with a stub that returns a FakeFunctionCall with the obvious mapping.
        def _fake_action_to_call(action, screen_size):
            fn_idx = int(action[0])
            fn_id = {
                0: _FakeFunctions.no_op.id,
                1: _FakeFunctions.select_army.id,
                2: _FakeFunctions.Move_screen.id,
            }.get(fn_idx, _FakeFunctions.no_op.id)
            return _FakeFunctionCall(fn_id, [])

        patcher_helper = patch.object(
            client_mod, "action_to_function_call", _fake_action_to_call,
        )
        patcher_helper.start()
        self.addCleanup(patcher_helper.stop)

        from games.sc2.client import SC2Client
        self.client = SC2Client(map_name="MoveToBeacon")

    def test_move_screen_with_no_army_selected_substitutes_select_army(self):
        """Issue #124: blocked Move_screen → auto-select-army (not no_op)."""
        # Available actions: no_op (0) + select_army (7).  Move_screen (331) is NOT.
        self.client._available_actions = {
            _FakeFunctions.no_op.id, _FakeFunctions.select_army.id,
        }
        action = np.array([2, 0.4, 0.6, 0], dtype=np.float32)
        call = self.client._action_to_call(action)
        self.assertEqual(call.function, _FakeFunctions.select_army.id)

    def test_blocked_no_op_when_select_army_also_unavailable(self):
        """If select_army is also blocked, fall back to no_op as before."""
        self.client._available_actions = {_FakeFunctions.no_op.id}
        action = np.array([2, 0.4, 0.6, 0], dtype=np.float32)
        call = self.client._action_to_call(action)
        self.assertEqual(call.function, _FakeFunctions.no_op.id)

    def test_legal_move_screen_passes_through(self):
        """When Move_screen IS available, the original action is preserved."""
        self.client._available_actions = {
            _FakeFunctions.no_op.id,
            _FakeFunctions.select_army.id,
            _FakeFunctions.Move_screen.id,
        }
        action = np.array([2, 0.4, 0.6, 0], dtype=np.float32)
        call = self.client._action_to_call(action)
        self.assertEqual(call.function, _FakeFunctions.Move_screen.id)

    def test_no_op_action_not_redirected(self):
        """A policy emitting no_op should pass through unchanged."""
        self.client._available_actions = {
            _FakeFunctions.no_op.id,
            _FakeFunctions.select_army.id,
        }
        action = np.array([0, 0.5, 0.5, 0], dtype=np.float32)
        call = self.client._action_to_call(action)
        self.assertEqual(call.function, _FakeFunctions.no_op.id)

    def test_select_army_substitution_not_repeated_on_consecutive_blocked_steps(self):
        """Issue (beacon idling): when Move_screen is blocked for multiple
        consecutive steps, select_army should be issued ONCE, then no_op on
        subsequent blocked steps — not select_army every time."""
        blocked_available = {
            _FakeFunctions.no_op.id, _FakeFunctions.select_army.id,
        }
        self.client._available_actions = blocked_available
        action = np.array([2, 0.4, 0.6, 0], dtype=np.float32)

        # Step 1: first blocked step → should substitute select_army.
        call1 = self.client._action_to_call(action)
        self.assertEqual(call1.function, _FakeFunctions.select_army.id)
        self.assertTrue(self.client._select_army_pending)

        # Step 2: still blocked → should NOT repeat select_army; use no_op.
        call2 = self.client._action_to_call(action)
        self.assertEqual(call2.function, _FakeFunctions.no_op.id)

        # Step 3: still blocked → no_op again (not select_army).
        call3 = self.client._action_to_call(action)
        self.assertEqual(call3.function, _FakeFunctions.no_op.id)

    def test_pending_flag_cleared_when_move_screen_becomes_available(self):
        """After a select_army substitution, when Move_screen finally becomes
        available again the pending flag is reset and a future blocked step
        will trigger a fresh select_army substitution."""
        blocked = {_FakeFunctions.no_op.id, _FakeFunctions.select_army.id}
        available = {
            _FakeFunctions.no_op.id,
            _FakeFunctions.select_army.id,
            _FakeFunctions.Move_screen.id,
        }
        action = np.array([2, 0.4, 0.6, 0], dtype=np.float32)

        # First blocked step: select_army substituted, pending=True.
        self.client._available_actions = blocked
        self.client._action_to_call(action)
        self.assertTrue(self.client._select_army_pending)

        # Move_screen becomes available: pending should reset.
        self.client._available_actions = available
        self.client._action_to_call(action)
        self.assertFalse(self.client._select_army_pending)

        # Now block again: should trigger select_army again (not no_op).
        self.client._available_actions = blocked
        call = self.client._action_to_call(action)
        self.assertEqual(call.function, _FakeFunctions.select_army.id)

class TestSC2ClientAvailableFnIds(unittest.TestCase):
    """Tests for the info["available_fn_ids"] field added by _timestep_to_obs_info."""

    def _minigame_ob(self, available_actions: np.ndarray | None = None) -> dict:
        ob = {
            "player": _NamedArr({
                "minerals": 0, "vespene": 0, "food_used": 0, "food_cap": 0,
                "army_count": 0, "idle_worker_count": 0,
                "warp_gate_count": 0, "larva_count": 0,
            }),
            "feature_screen": np.zeros((17, 64, 64), dtype=np.int32),
            "score_cumulative": np.array([0]),
        }
        if available_actions is not None:
            ob["available_actions"] = available_actions
        return ob

    def test_available_fn_ids_absent_when_no_available_actions(self):
        """When the observation has no available_actions key, available_fn_ids is None."""
        client = SC2Client(map_name="MoveToBeacon")
        ob = self._minigame_ob(available_actions=None)
        _, info = client._timestep_to_obs_info(_FakeTimeStep(ob))
        self.assertIn("available_fn_ids", info)
        self.assertIsNone(info["available_fn_ids"])

    def test_available_fn_ids_none_when_mapping_unavailable(self):
        """When the PySC2 ID→fn_idx mapping is empty (PySC2 not installed),
        available_fn_ids is None even if available_actions is present."""
        import games.sc2.client as sc2_client_mod
        old_cache = sc2_client_mod._pysc2_id_to_fn_idx
        try:
            # Simulate PySC2 not installed: the cache resolves to an empty dict.
            sc2_client_mod._pysc2_id_to_fn_idx = {}
            client = SC2Client(map_name="MoveToBeacon")
            ob = self._minigame_ob(available_actions=np.array([0, 1, 2]))
            _, info = client._timestep_to_obs_info(_FakeTimeStep(ob))
            self.assertIsNone(info["available_fn_ids"])
        finally:
            sc2_client_mod._pysc2_id_to_fn_idx = old_cache

    def test_available_fn_ids_mapped_correctly_with_known_id_table(self):
        """With an injected PySC2-ID→fn_idx table, available_fn_ids contains
        only the fn_idx values whose PySC2 IDs appear in available_actions."""
        import games.sc2.client as sc2_client_mod
        old_cache = sc2_client_mod._pysc2_id_to_fn_idx
        try:
            # Inject a synthetic mapping: PySC2 IDs 0→fn_idx 0, 7→fn_idx 1, 331→fn_idx 2.
            sc2_client_mod._pysc2_id_to_fn_idx = {0: 0, 7: 1, 331: 2}
            client = SC2Client(map_name="MoveToBeacon")
            # Observation exposes PySC2 IDs 0 (no_op) and 331 (Move_screen).
            ob = self._minigame_ob(available_actions=np.array([0, 331]))
            _, info = client._timestep_to_obs_info(_FakeTimeStep(ob))
            self.assertIsNotNone(info["available_fn_ids"])
            self.assertEqual(info["available_fn_ids"], {0, 2})
        finally:
            sc2_client_mod._pysc2_id_to_fn_idx = old_cache

    def test_available_fn_ids_excludes_unknown_pysc2_ids(self):
        """PySC2 IDs with no mapping entry must be silently dropped."""
        import games.sc2.client as sc2_client_mod
        old_cache = sc2_client_mod._pysc2_id_to_fn_idx
        try:
            sc2_client_mod._pysc2_id_to_fn_idx = {0: 0}  # only no_op mapped
            client = SC2Client(map_name="MoveToBeacon")
            # available_actions includes an unknown ID (999).
            ob = self._minigame_ob(available_actions=np.array([0, 999]))
            _, info = client._timestep_to_obs_info(_FakeTimeStep(ob))
            self.assertEqual(info["available_fn_ids"], {0})
        finally:
            sc2_client_mod._pysc2_id_to_fn_idx = old_cache

    def test_available_fn_ids_is_set_type(self):
        """available_fn_ids must be a set (not a list or dict) for O(1) lookup."""
        import games.sc2.client as sc2_client_mod
        old_cache = sc2_client_mod._pysc2_id_to_fn_idx
        try:
            sc2_client_mod._pysc2_id_to_fn_idx = {0: 0, 7: 1}
            client = SC2Client(map_name="MoveToBeacon")
            ob = self._minigame_ob(available_actions=np.array([0, 7]))
            _, info = client._timestep_to_obs_info(_FakeTimeStep(ob))
            self.assertIsInstance(info["available_fn_ids"], set)
        finally:
            sc2_client_mod._pysc2_id_to_fn_idx = old_cache


# ---------------------------------------------------------------------------
# Issue #135: new rich-preset feature extractors
# ---------------------------------------------------------------------------

class TestSC2ClientRichExtractors(unittest.TestCase):
    """Targeted tests for the per-block extractors added in #135."""

    def setUp(self):
        self.client = SC2Client(map_name="Simple64", obs_spec_preset="rich")

    def _screen_with_pr(self, friendly_yx=None, enemy_yx=None, n_layers=17):
        """Build a fake feature_screen with player_relative set at given pixels."""
        screen = np.zeros((n_layers, 64, 64), dtype=np.int32)
        if friendly_yx:
            for y, x in friendly_yx:
                screen[5, y, x] = 1
        if enemy_yx:
            for y, x in enemy_yx:
                screen[5, y, x] = 4
        return screen

    # --- enemy unit type counts ---

    def test_enemy_unit_type_features_counts_enemy_only(self):
        """Rows with owner != 1 are counted; owner == 1 rows are skipped."""
        # Inject a synthetic unit_type_id_to_name mapping.
        self.client._unit_type_id_to_name = {1: "Marine", 2: "Zergling"}
        # feature_units: columns [unit_type, owner]
        feat_units = np.array([
            [1, 1],  # friendly Marine → skip
            [2, 4],  # enemy Zergling → count
            [2, 4],  # enemy Zergling → count
        ], dtype=np.int32)
        ob = {"feature_units": feat_units}
        feats = self.client._enemy_unit_type_features(ob)
        self.assertEqual(feats["enemy_count_Zergling"], 2.0)
        self.assertEqual(feats["enemy_count_Marine"],   0.0)

    def test_enemy_unit_type_features_missing_feature_units(self):
        out = self.client._enemy_unit_type_features({})
        self.assertTrue(all(v == 0.0 for v in out.values()))

    def test_enemy_unit_type_features_unknown_unit_type_ignored(self):
        self.client._unit_type_id_to_name = {1: "Marine"}
        feat_units = np.array([[999, 4]], dtype=np.int32)  # unknown unit type
        ob = {"feature_units": feat_units}
        feats = self.client._enemy_unit_type_features(ob)
        self.assertTrue(all(v == 0.0 for v in feats.values()))

    # --- shield / energy summaries ---

    def test_shield_energy_features_self_shield(self):
        """shield layer pixels at friendly positions are averaged for self_shield."""
        screen = self._screen_with_pr(friendly_yx=[(10, 10), (10, 11)])
        # Channel index for unit_shields: look it up via named access — fake it
        # by building a _NamedArr-style screen stub.
        screen_named = np.zeros((17, 64, 64), dtype=np.int32)
        screen_named[5, 10, 10] = 1  # player_relative = friendly
        screen_named[5, 10, 11] = 1

        # Build a dict-accessible screen object so _extract_named_layer works.
        class _NamedScreen:
            def __init__(self, arr):
                self._arr = arr
                self._extra = {}

            def __getitem__(self, key):
                if key == "player_relative":
                    return self._arr[5]
                if key in self._extra:
                    return self._extra[key]
                raise KeyError(key)

        named = _NamedScreen(screen_named)
        shield_layer = np.zeros((64, 64), dtype=np.float32)
        shield_layer[10, 10] = 80.0
        shield_layer[10, 11] = 40.0
        named._extra["unit_shields"] = shield_layer
        named._extra["unit_energy"] = np.zeros((64, 64), dtype=np.float32)

        feats = self.client._shield_energy_features(named)
        self.assertAlmostEqual(feats["screen_self_shield_mean"], 60.0)  # (80+40)/2
        self.assertAlmostEqual(feats["screen_enemy_shield_mean"], 0.0)

    def test_shield_energy_features_no_units_returns_zeros(self):
        screen = np.zeros((17, 64, 64), dtype=np.int32)
        feats = self.client._shield_energy_features(screen)
        self.assertEqual(feats["screen_self_shield_mean"],  0.0)
        self.assertEqual(feats["screen_enemy_shield_mean"], 0.0)
        self.assertEqual(feats["screen_self_energy_mean"],  0.0)

    def test_shield_energy_features_none_screen(self):
        feats = self.client._shield_energy_features(None)
        self.assertTrue(all(v == 0.0 for v in feats.values()))

    # --- creep coverage ---

    def test_creep_features_half_coverage(self):
        """A minimap with half creep should yield frac ≈ 0.25 for a quadrant."""

        class _NamedMinimap:
            def __getitem__(self, key):
                if key == "creep":
                    arr = np.zeros((64, 64), dtype=np.int32)
                    arr[:32, :32] = 1  # top-left quadrant = creep
                    return arr
                raise KeyError(key)

        feats = self.client._creep_features(_NamedMinimap())
        expected = (32 * 32) / (64 * 64)
        self.assertAlmostEqual(feats["minimap_creep_frac"], expected, places=5)

    def test_creep_features_no_creep(self):
        mmap = np.zeros((11, 64, 64), dtype=np.int32)
        feats = self.client._creep_features(mmap)
        self.assertEqual(feats["minimap_creep_frac"], 0.0)

    def test_creep_features_none_minimap(self):
        feats = self.client._creep_features(None)
        self.assertEqual(feats["minimap_creep_frac"], 0.0)

    # --- economy pipeline ---

    def test_economy_pipeline_upgrade_count(self):
        ob = {"upgrades": np.array([3, 7, 12], dtype=np.int32)}
        feats = self.client._economy_pipeline_features(ob)
        self.assertEqual(feats["upgrade_count"], 3.0)

    def test_economy_pipeline_build_queue_size(self):
        # 2-row build queue (2 units under construction)
        ob = {"build_queue": np.zeros((2, 7), dtype=np.int32)}
        feats = self.client._economy_pipeline_features(ob)
        self.assertEqual(feats["build_queue_size"], 2.0)

    def test_economy_pipeline_cargo_count(self):
        ob = {"cargo": np.zeros((4, 7), dtype=np.int32)}
        feats = self.client._economy_pipeline_features(ob)
        self.assertEqual(feats["cargo_count"], 4.0)

    def test_economy_pipeline_all_missing_returns_zeros(self):
        feats = self.client._economy_pipeline_features({})
        self.assertEqual(feats["upgrade_count"],    0.0)
        self.assertEqual(feats["build_queue_size"], 0.0)
        self.assertEqual(feats["cargo_count"],      0.0)

    # --- integration: new dims appear in flat rich obs ---

    def test_rich_obs_includes_new_feature_names(self):
        """The rich spec should contain all 15 new feature names from issue #135."""
        from games.sc2.obs_spec import RICH_OBS_NAMES
        for name in ("enemy_count_Marine", "screen_self_shield_mean",
                     "minimap_creep_frac", "upgrade_count",
                     "build_queue_size", "cargo_count"):
            self.assertIn(name, RICH_OBS_NAMES, f"{name!r} missing from rich spec")

    def test_ladder_spec_unchanged_by_issue_135(self):
        """Ladder spec must NOT contain the new rich-only features."""
        from games.sc2.obs_spec import LADDER_OBS_NAMES
        for name in ("enemy_count_Marine", "screen_self_shield_mean",
                     "minimap_creep_frac", "upgrade_count"):
            self.assertNotIn(name, LADDER_OBS_NAMES,
                             f"{name!r} should not be in ladder spec")

    # --- selected-unit shield and energy averages ---

    def test_selected_features_shields_and_energy(self):
        """selected_avg_shields and selected_avg_energy are computed from cols 3/4."""
        # multi_select rows: [unit_type, player_relative, hp, shields, energy, ...]
        multi = np.array([
            [48, 1, 45, 80, 0],   # Marine: hp=45, shields=80, energy=0
            [48, 1, 55, 20, 0],   # Marine: hp=55, shields=20, energy=0
        ], dtype=np.int32)
        ob = {"multi_select": multi}
        feats = self.client._selected_features(ob)
        self.assertAlmostEqual(feats["selected_avg_shields"], 50.0)  # (80+20)/2
        self.assertAlmostEqual(feats["selected_avg_energy"],   0.0)

    def test_selected_features_energy_populated(self):
        """selected_avg_energy reflects col 4 of the select array."""
        single = np.array([[76, 1, 100, 0, 150]], dtype=np.int32)  # HT: energy=150
        ob = {"single_select": single}
        feats = self.client._selected_features(ob)
        self.assertAlmostEqual(feats["selected_avg_energy"], 150.0)
        self.assertAlmostEqual(feats["selected_avg_shields"],  0.0)

    def test_selected_features_empty_selection_returns_zeros(self):
        """Empty selection → all selected_* features are 0."""
        ob = {
            "single_select": np.zeros((0, 7), dtype=np.int32),
            "multi_select":  np.zeros((0, 7), dtype=np.int32),
        }
        feats = self.client._selected_features(ob)
        self.assertEqual(feats["selected_avg_shields"], 0.0)
        self.assertEqual(feats["selected_avg_energy"],  0.0)

    def test_selected_features_short_rows_dont_crash(self):
        """If the select array has fewer than 5 columns, defaults to 0."""
        ob = {"single_select": np.array([[48, 1, 55]], dtype=np.int32)}  # only 3 cols
        feats = self.client._selected_features(ob)
        self.assertEqual(feats["selected_avg_shields"], 0.0)
        self.assertEqual(feats["selected_avg_energy"],  0.0)

    # --- screen visibility fraction ---

    def test_screen_visibility_features_all_visible(self):
        """All-2 visibility_map → screen_visibility_frac == 1.0."""
        class _NamedScreen:
            def __getitem__(self, key):
                if key == "visibility_map":
                    return np.full((64, 64), 2, dtype=np.int32)
                raise KeyError(key)

        feats = self.client._screen_visibility_features(_NamedScreen())
        self.assertAlmostEqual(feats["screen_visibility_frac"], 1.0)

    def test_screen_visibility_features_half_visible(self):
        """Half of tiles visible → screen_visibility_frac ≈ 0.25 (quadrant)."""
        class _NamedScreen:
            def __getitem__(self, key):
                if key == "visibility_map":
                    arr = np.zeros((64, 64), dtype=np.int32)
                    arr[:32, :32] = 2  # top-left quadrant visible
                    return arr
                raise KeyError(key)

        feats = self.client._screen_visibility_features(_NamedScreen())
        expected = (32 * 32) / (64 * 64)
        self.assertAlmostEqual(feats["screen_visibility_frac"], expected, places=5)

    def test_screen_visibility_features_fogged_not_counted(self):
        """Fogged tiles (value == 1) must NOT count as visible."""
        class _NamedScreen:
            def __getitem__(self, key):
                if key == "visibility_map":
                    arr = np.full((64, 64), 1, dtype=np.int32)  # all fogged
                    return arr
                raise KeyError(key)

        feats = self.client._screen_visibility_features(_NamedScreen())
        self.assertAlmostEqual(feats["screen_visibility_frac"], 0.0)

    def test_screen_visibility_features_none_screen(self):
        feats = self.client._screen_visibility_features(None)
        self.assertEqual(feats["screen_visibility_frac"], 0.0)

    def test_screen_visibility_features_missing_layer_returns_zero(self):
        """If visibility_map is absent from the feature screen, returns 0."""
        feats = self.client._screen_visibility_features(np.zeros((17, 64, 64), dtype=np.int32))
        self.assertEqual(feats["screen_visibility_frac"], 0.0)

    # --- anti-air density ---

    def test_screen_antiair_features_mean(self):
        """Mean of unit_density_aa layer is returned."""
        class _NamedScreen:
            def __getitem__(self, key):
                if key == "unit_density_aa":
                    arr = np.zeros((64, 64), dtype=np.float32)
                    arr[:32, :32] = 4.0  # top-left quadrant has density 4
                    return arr
                raise KeyError(key)

        feats = self.client._screen_antiair_features(_NamedScreen())
        expected = 4.0 * (32 * 32) / (64 * 64)
        self.assertAlmostEqual(feats["screen_unit_density_aa_mean"], expected, places=4)

    def test_screen_antiair_features_zero_layer(self):
        class _NamedScreen:
            def __getitem__(self, key):
                if key == "unit_density_aa":
                    return np.zeros((64, 64), dtype=np.float32)
                raise KeyError(key)

        feats = self.client._screen_antiair_features(_NamedScreen())
        self.assertAlmostEqual(feats["screen_unit_density_aa_mean"], 0.0)

    def test_screen_antiair_features_none_screen(self):
        feats = self.client._screen_antiair_features(None)
        self.assertEqual(feats["screen_unit_density_aa_mean"], 0.0)

    def test_screen_antiair_features_missing_layer_returns_zero(self):
        feats = self.client._screen_antiair_features(np.zeros((17, 64, 64), dtype=np.int32))
        self.assertEqual(feats["screen_unit_density_aa_mean"], 0.0)

    # --- weapon cooldown ---

    def test_weapon_cooldown_features_mean(self):
        """Mean cooldown for friendly units is computed from col 25."""
        # 26-column feature_units: col 1 = alliance, col 25 = weapon_cooldown
        feat_units = np.zeros((3, 26), dtype=np.float32)
        feat_units[0, 1] = 1.0;  feat_units[0, 25] = 10.0   # self, cooldown=10
        feat_units[1, 1] = 1.0;  feat_units[1, 25] = 30.0   # self, cooldown=30
        feat_units[2, 1] = 4.0;  feat_units[2, 25] = 99.0   # enemy → excluded
        ob = {"feature_units": feat_units}
        feats = self.client._weapon_cooldown_features(ob)
        self.assertAlmostEqual(feats["self_weapon_cooldown_mean"], 20.0)  # (10+30)/2

    def test_weapon_cooldown_features_all_ready(self):
        """All friendly units with cooldown 0 → mean == 0."""
        feat_units = np.zeros((2, 26), dtype=np.float32)
        feat_units[:, 1] = 1.0  # all self; cooldown col stays 0
        ob = {"feature_units": feat_units}
        feats = self.client._weapon_cooldown_features(ob)
        self.assertAlmostEqual(feats["self_weapon_cooldown_mean"], 0.0)

    def test_weapon_cooldown_features_no_self_units_returns_zero(self):
        """If all units are enemy, returns 0."""
        feat_units = np.zeros((2, 26), dtype=np.float32)
        feat_units[:, 1] = 4.0  # all enemy
        feat_units[:, 25] = 50.0
        ob = {"feature_units": feat_units}
        feats = self.client._weapon_cooldown_features(ob)
        self.assertEqual(feats["self_weapon_cooldown_mean"], 0.0)

    def test_weapon_cooldown_features_missing_feature_units_returns_zero(self):
        feats = self.client._weapon_cooldown_features({})
        self.assertEqual(feats["self_weapon_cooldown_mean"], 0.0)

    def test_weapon_cooldown_features_too_few_columns_returns_zero(self):
        """feature_units with < 26 columns is tolerated gracefully."""
        feat_units = np.zeros((2, 10), dtype=np.float32)
        feat_units[:, 1] = 1.0
        ob = {"feature_units": feat_units}
        feats = self.client._weapon_cooldown_features(ob)
        self.assertEqual(feats["self_weapon_cooldown_mean"], 0.0)

    # --- alerts ---

    def test_alerts_features_empty_no_alerts(self):
        """Empty alerts array → alert_count == 0."""
        feats = self.client._alerts_features({"alerts": np.array([], dtype=np.int32)})
        self.assertEqual(feats["alert_count"], 0.0)

    def test_alerts_features_one_alert(self):
        """Single alert entry → alert_count == 1."""
        feats = self.client._alerts_features({"alerts": np.array([2], dtype=np.int32)})
        self.assertEqual(feats["alert_count"], 1.0)

    def test_alerts_features_two_alerts(self):
        """Two simultaneous alerts (PySC2 max) → alert_count == 2."""
        feats = self.client._alerts_features({"alerts": np.array([2, 2], dtype=np.int32)})
        self.assertEqual(feats["alert_count"], 2.0)

    def test_alerts_features_missing_key_returns_zero(self):
        """No 'alerts' key in observation → alert_count == 0."""
        feats = self.client._alerts_features({})
        self.assertEqual(feats["alert_count"], 0.0)

    def test_alerts_features_none_ob_returns_zero(self):
        """_safe_array with None result → alert_count == 0."""
        feats = self.client._alerts_features({"alerts": None})
        self.assertEqual(feats["alert_count"], 0.0)

    def test_alert_count_appears_in_ladder_flat_obs(self):
        """alert_count must appear in the ladder flat obs vector."""
        from games.sc2.obs_spec import LADDER_OBS_NAMES
        self.assertIn("alert_count", LADDER_OBS_NAMES)

    # --- integration: new dims appear in flat rich obs (TestSC2ClientRichExtractors) ---

    def test_rich_obs_includes_new_feature_names(self):
        """The rich spec should contain all new feature names."""
        from games.sc2.obs_spec import RICH_OBS_NAMES
        for name in ("enemy_count_Marine", "screen_self_shield_mean",
                     "minimap_creep_frac", "upgrade_count",
                     "build_queue_size", "cargo_count",
                     "selected_avg_shields", "selected_avg_energy",
                     "screen_visibility_frac", "screen_unit_density_aa_mean",
                     "self_weapon_cooldown_mean"):
            self.assertIn(name, RICH_OBS_NAMES, f"{name!r} missing from rich spec")


if __name__ == "__main__":
    unittest.main()
