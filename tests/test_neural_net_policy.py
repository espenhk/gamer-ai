"""Tests for NeuralNetPolicy in tmnf/policies.py."""
import unittest

import numpy as np

from games.tmnf.obs_spec import BASE_OBS_DIM
from games.tmnf.policies import NeuralNetPolicy

_N = BASE_OBS_DIM


class TestNeuralNetPolicy(unittest.TestCase):

    def assert_action_vector(self, action):
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (3,))
        self.assertGreaterEqual(float(action[0]), -1.0)
        self.assertLessEqual(float(action[0]), 1.0)
        self.assertIn(float(action[1]), {0.0, 1.0})
        self.assertIn(float(action[2]), {0.0, 1.0})

    def test_action_in_range(self):
        p = NeuralNetPolicy(hidden_sizes=[8])
        obs = np.random.randn(_N).astype(np.float32)
        self.assert_action_vector(p(obs))

    def test_deterministic(self):
        p = NeuralNetPolicy(hidden_sizes=[8])
        obs = np.random.randn(_N).astype(np.float32)
        np.testing.assert_array_equal(p(obs), p(obs))

    def test_from_cfg_roundtrip(self):
        p = NeuralNetPolicy(hidden_sizes=[8, 8])
        obs = np.random.randn(_N).astype(np.float32)
        p2 = NeuralNetPolicy.from_cfg(p.to_cfg())
        np.testing.assert_allclose(p(obs), p2(obs))

    def test_hidden_sizes_preserved_in_cfg(self):
        p = NeuralNetPolicy(hidden_sizes=[32, 16])
        self.assertEqual(p.to_cfg()["hidden_sizes"], [32, 16])

    def test_output_always_9_actions(self):
        p = NeuralNetPolicy(hidden_sizes=[4])
        for _ in range(20):
            obs = np.random.randn(_N).astype(np.float32) * 100
            self.assert_action_vector(p(obs))

    def test_mutated_has_different_weights(self):
        p = NeuralNetPolicy(hidden_sizes=[8])
        m = p.mutated(scale=1.0)
        orig = p.to_cfg()["weights"][0]
        mutd = m.to_cfg()["weights"][0]
        self.assertFalse(np.allclose(orig, mutd))

    def test_weight_matrix_shapes(self):
        p = NeuralNetPolicy(hidden_sizes=[16, 8])
        cfg = p.to_cfg()
        weights = cfg["weights"]
        # Layer dims: [BASE_OBS_DIM, 16, 8, 3]
        self.assertEqual(np.array(weights[0]).shape, (16, _N))
        self.assertEqual(np.array(weights[1]).shape, (8, 16))
        self.assertEqual(np.array(weights[2]).shape, (3, 8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
