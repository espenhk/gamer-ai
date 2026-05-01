"""Tests for framework/belief.py — EWMABelief module."""
import unittest

import numpy as np

from framework.belief import EWMABelief


class TestEWMABeliefBasic(unittest.TestCase):
    """Core behaviour of the EWMA belief tracker."""

    def setUp(self):
        self.belief = EWMABelief(n_slots=4, decay_tau=10.0)

    def test_initial_encode_all_zero(self):
        enc = self.belief.encode()
        np.testing.assert_array_equal(enc, np.zeros(8, dtype=np.float32))

    def test_update_sets_value_and_confidence(self):
        obs = np.array([1.0, np.nan, 3.0, np.nan])
        self.belief.update(obs, {})
        enc = self.belief.encode()
        # Slot 0: value=1.0, confidence=1.0
        self.assertAlmostEqual(enc[0], 1.0)
        self.assertAlmostEqual(enc[1], 1.0)
        # Slot 1: never observed
        self.assertAlmostEqual(enc[2], 0.0)
        self.assertAlmostEqual(enc[3], 0.0)
        # Slot 2: value=3.0, confidence=1.0
        self.assertAlmostEqual(enc[4], 3.0)
        self.assertAlmostEqual(enc[5], 1.0)

    def test_project_decays_confidence(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        self.belief.update(obs, {})
        self.belief.project(10.0)  # dt = decay_tau → confidence ≈ e^-1
        enc = self.belief.encode()
        expected_conf = np.exp(-1.0)
        for i in range(4):
            self.assertAlmostEqual(enc[2 * i + 1], expected_conf, places=5)

    def test_scout_then_lose_sight_then_decay(self):
        """Simulate: see slot, lose sight, project forward → decayed confidence."""
        obs_seen = np.array([5.0, np.nan, np.nan, np.nan])
        self.belief.update(obs_seen, {})
        self.assertAlmostEqual(self.belief.confidence[0], 1.0)

        # Project 20s with decay_tau=10 → confidence ≈ e^-2
        self.belief.project(20.0)
        self.assertAlmostEqual(self.belief.confidence[0], np.exp(-2.0), places=5)
        # Value should still be 5.0
        self.assertAlmostEqual(self.belief.values[0], 5.0)

    def test_reset_clears_belief(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        self.belief.update(obs, {})
        self.belief.project(5.0)
        self.belief.reset()
        enc = self.belief.encode()
        np.testing.assert_array_equal(enc, np.zeros(8, dtype=np.float32))


class TestEWMABeliefPerSlotDecay(unittest.TestCase):
    """Per-slot decay_tau configuration."""

    def test_different_tau_per_slot(self):
        taus = np.array([5.0, 20.0])
        belief = EWMABelief(n_slots=2, decay_tau=taus)
        obs = np.array([1.0, 1.0])
        belief.update(obs, {})
        belief.project(10.0)  # dt=10
        # Slot 0: exp(-10/5)=exp(-2), Slot 1: exp(-10/20)=exp(-0.5)
        self.assertAlmostEqual(belief.confidence[0], np.exp(-2.0), places=5)
        self.assertAlmostEqual(belief.confidence[1], np.exp(-0.5), places=5)


class TestEWMABeliefEncodeShape(unittest.TestCase):
    def test_encode_shape(self):
        belief = EWMABelief(n_slots=6)
        self.assertEqual(belief.encode().shape, (12,))


if __name__ == "__main__":
    unittest.main()
