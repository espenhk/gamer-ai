"""Tests for WeightedLinearPolicy in tmnf/policies.py."""
import unittest

import numpy as np

from helpers import make_wlp
from games.tmnf.obs_spec import BASE_OBS_DIM
from games.tmnf.policies import WeightedLinearPolicy

_N = BASE_OBS_DIM


class TestWeightedLinearPolicy(unittest.TestCase):

    def assert_action_vector(self, action):
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (3,))
        self.assertGreaterEqual(float(action[0]), -1.0)
        self.assertLessEqual(float(action[0]), 1.0)
        self.assertIn(float(action[1]), {0.0, 1.0})
        self.assertIn(float(action[2]), {0.0, 1.0})

    def test_action_in_range(self):
        p = make_wlp()
        for _ in range(10):
            obs = np.random.randn(_N).astype(np.float32)
            self.assert_action_vector(p(obs))

    def test_deterministic(self):
        p = make_wlp()
        obs = np.ones(_N, dtype=np.float32)
        np.testing.assert_array_equal(p(obs), p(obs))

    def test_accel_weights_dominate(self):
        # Large positive weight on speed_ms (index 0, scale 50) → throttle score ≫ threshold
        tw = np.zeros(_N, dtype=np.float32)
        tw[0] = 1000.0
        obs = np.array([50.0] + [0.0] * (_N - 1), dtype=np.float32)
        p = make_wlp(throttle_weights=tw)
        action = p(obs)
        self.assertEqual(float(action[1]), 1.0)
        self.assertEqual(float(action[2]), 0.0)

    def test_brake_weights_dominate(self):
        tw = np.zeros(_N, dtype=np.float32)
        tw[0] = -1000.0
        obs = np.array([50.0] + [0.0] * (_N - 1), dtype=np.float32)
        p = make_wlp(throttle_weights=tw)
        action = p(obs)
        self.assertEqual(float(action[1]), 0.0)
        self.assertEqual(float(action[2]), 1.0)

    def test_coast_within_threshold(self):
        # All-zero weights → throttle score = 0 → within threshold → coast actions
        obs = np.ones(_N, dtype=np.float32)
        p = make_wlp()
        action = p(obs)
        self.assertEqual(float(action[1]), 0.0)
        self.assertEqual(float(action[2]), 0.0)

    def test_steer_left_action(self):
        # Large negative steer weight on lateral_offset (index 1) → steer score ≪ -threshold → left
        # Large positive throttle weight → accel; combined: action 6 (accel+left)
        sw = np.zeros(_N, dtype=np.float32)
        sw[1] = -1000.0
        tw = np.zeros(_N, dtype=np.float32)
        tw[0] = 1000.0
        obs = np.array([50.0, 1.0] + [0.0] * (_N - 2), dtype=np.float32)
        p = make_wlp(steer_weights=sw, throttle_weights=tw)
        action = p(obs)
        self.assertEqual(float(action[0]), -1.0)
        self.assertEqual(float(action[1]), 1.0)
        self.assertEqual(float(action[2]), 0.0)

    def test_steer_right_action(self):
        sw = np.zeros(_N, dtype=np.float32)
        sw[1] = 1000.0
        tw = np.zeros(_N, dtype=np.float32)
        tw[0] = 1000.0
        obs = np.array([50.0, 1.0] + [0.0] * (_N - 2), dtype=np.float32)
        p = make_wlp(steer_weights=sw, throttle_weights=tw)
        action = p(obs)
        self.assertEqual(float(action[0]), 1.0)
        self.assertEqual(float(action[1]), 1.0)
        self.assertEqual(float(action[2]), 0.0)

    def test_from_cfg_roundtrip(self):
        p = make_wlp()
        cfg = p.to_cfg()
        p2 = WeightedLinearPolicy.from_cfg(cfg)
        obs = np.random.randn(_N).astype(np.float32)
        np.testing.assert_array_equal(p(obs), p2(obs))

    def test_mutated_weights_differ(self):
        sw = np.ones(_N, dtype=np.float32)
        p = make_wlp(steer_weights=sw)
        mutated = p.mutated(scale=1.0)
        orig = list(p.to_cfg()["steer_weights"].values())
        mut  = list(mutated.to_cfg()["steer_weights"].values())
        self.assertFalse(np.allclose(orig, mut))

    def test_obs_scales_length_matches_obs_names(self):
        self.assertEqual(len(WeightedLinearPolicy.OBS_SCALES),
                         len(WeightedLinearPolicy.OBS_NAMES))

    def test_action_is_int(self):
        p = make_wlp()
        action = p(np.zeros(_N, dtype=np.float32))
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (3,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
