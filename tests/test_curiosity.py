"""Tests for the curiosity-driven exploration modules (issue #24)."""
from __future__ import annotations

import numpy as np
import unittest

from games.tmnf.curiosity import ICM, RND, make_curiosity


class TestICMReward(unittest.TestCase):
    """Intrinsic reward should drop as the model learns a recurring transition."""

    def _fixed_transition(self, obs_dim: int = 6, action_dim: int = 3,
                          seed: int = 0):
        rng = np.random.default_rng(seed)
        s  = rng.standard_normal(obs_dim).astype(np.float32)
        a  = rng.standard_normal(action_dim).astype(np.float32)
        sp = rng.standard_normal(obs_dim).astype(np.float32)
        return s, a, sp

    def test_reward_decreases_after_repeated_updates(self):
        s, a, sp = self._fixed_transition()
        icm = ICM(obs_dim=s.size, action_dim=a.size, feature_dim=4,
                  hidden_size=16, lr=0.05, beta=0.2, seed=42)

        r_initial = icm.reward(s, a, sp)
        for _ in range(500):
            icm.update(s, a, sp)
        r_final = icm.reward(s, a, sp)

        self.assertGreater(r_initial, 0.0)
        self.assertLess(r_final, r_initial * 0.5,
                        f"ICM should learn the recurring transition: "
                        f"initial={r_initial:.5f} final={r_final:.5f}")

    def test_reward_is_nonnegative(self):
        s, a, sp = self._fixed_transition(seed=1)
        icm = ICM(obs_dim=s.size, action_dim=a.size, seed=1)
        for _ in range(20):
            self.assertGreaterEqual(icm.reward(s, a, sp), 0.0)

    def test_action_dim_mismatch_raises(self):
        icm = ICM(obs_dim=5, action_dim=3, seed=0)
        with self.assertRaises(ValueError):
            icm.reward(np.zeros(5), np.zeros(2), np.zeros(5))

    def test_invalid_beta_raises(self):
        with self.assertRaises(ValueError):
            ICM(obs_dim=5, action_dim=3, beta=1.5)

    def test_eta_scales_reward(self):
        s, a, sp = self._fixed_transition(seed=2)
        icm_a = ICM(obs_dim=s.size, action_dim=a.size, eta=1.0, seed=7)
        icm_b = ICM(obs_dim=s.size, action_dim=a.size, eta=4.0, seed=7)
        r_a = icm_a.reward(s, a, sp)
        r_b = icm_b.reward(s, a, sp)
        self.assertAlmostEqual(r_b, 4.0 * r_a, places=5)


class TestRNDReward(unittest.TestCase):
    """RND predictor error should also drop on repeated states."""

    def test_reward_decreases_after_repeated_updates(self):
        rng = np.random.default_rng(0)
        sp = rng.standard_normal(8).astype(np.float32)
        rnd = RND(obs_dim=sp.size, feature_dim=4, hidden_size=16, lr=0.05, seed=3)

        r_initial = rnd.reward(np.zeros_like(sp), np.zeros(3), sp)
        for _ in range(500):
            rnd.update(np.zeros_like(sp), np.zeros(3), sp)
        r_final = rnd.reward(np.zeros_like(sp), np.zeros(3), sp)

        self.assertGreater(r_initial, 0.0)
        self.assertLess(r_final, r_initial * 0.1,
                        f"RND predictor should learn the recurring state: "
                        f"initial={r_initial:.5f} final={r_final:.5f}")

    def test_target_network_is_frozen(self):
        rnd = RND(obs_dim=4, feature_dim=2, hidden_size=8, lr=0.1, seed=0)
        before = [w.copy() for w in rnd.target.weights]
        for _ in range(50):
            rnd.update(np.zeros(4), np.zeros(3), np.ones(4, dtype=np.float32))
        for w_before, w_after in zip(before, rnd.target.weights):
            np.testing.assert_array_equal(w_before, w_after)

    def test_reward_is_nonnegative(self):
        rng = np.random.default_rng(5)
        sp = rng.standard_normal(6).astype(np.float32)
        rnd = RND(obs_dim=sp.size, seed=5)
        self.assertGreaterEqual(rnd.reward(sp, np.zeros(3), sp), 0.0)


class TestMakeCuriosity(unittest.TestCase):

    def test_none_returns_none(self):
        self.assertIsNone(make_curiosity("none", obs_dim=5, action_dim=3))
        self.assertIsNone(make_curiosity("NONE", obs_dim=5, action_dim=3))

    def test_icm_factory(self):
        m = make_curiosity("icm", obs_dim=5, action_dim=3,
                           feature_dim=4, hidden_size=8)
        self.assertIsInstance(m, ICM)

    def test_rnd_factory(self):
        m = make_curiosity("rnd", obs_dim=5, action_dim=3,
                           feature_dim=4, hidden_size=8)
        self.assertIsInstance(m, RND)

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            make_curiosity("count_based", obs_dim=5, action_dim=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
