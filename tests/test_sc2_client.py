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
        """Minigame default still produces a 13-dim flat obs (BC)."""
        from games.sc2.client import SC2Client
        from games.sc2.obs_spec import BASE_OBS_DIM
        c = SC2Client(map_name="MoveToBeacon")
        ob = self._ladder_ob()  # extra fields are tolerated
        flat, _ = c._timestep_to_obs_info(_FakeTimeStep(ob))
        self.assertEqual(flat.shape, (BASE_OBS_DIM,))


if __name__ == "__main__":
    unittest.main()
