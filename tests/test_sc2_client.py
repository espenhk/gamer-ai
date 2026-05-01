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
        self.assertEqual(info["player_outcome"], 1.0)
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

    def test_visibility_tracking(self):
        """Explored fraction should be monotonically non-decreasing."""
        mmap = np.zeros((11, 64, 64), dtype=np.int32)
        # Channel 1 = visibility_map; mark a quadrant visible.
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

        # Index 19 = minimap_visible_frac, 20 = minimap_explored_frac
        # (offset by 13 from minigame dims).
        self.assertGreater(flat1[18], 0.0)  # visible_frac > 0 first call
        self.assertEqual(flat2[18], 0.0)    # visible_frac = 0 second call
        # Explored remains > 0 in both calls.
        self.assertGreater(flat1[19], 0.0)
        self.assertGreater(flat2[19], 0.0)
        self.assertGreaterEqual(flat2[19], flat1[19])


if __name__ == "__main__":
    unittest.main()
