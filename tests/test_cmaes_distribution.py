"""Pure-math tests for CMAESDistribution in framework/cmaes.py."""
import unittest

import numpy as np

from framework.cmaes import CMAESDistribution


def _make(n=5, lam=10, sigma=0.3, seed=0) -> CMAESDistribution:
    return CMAESDistribution(n, population_size=lam, initial_sigma=sigma, seed=seed)


class TestCMAESDistributionInit(unittest.TestCase):

    def test_n_stored(self):
        d = _make(n=7)
        self.assertEqual(d._n, 7)

    def test_lam_stored(self):
        d = _make(lam=12)
        self.assertEqual(d._lam, 12)

    def test_mu_is_half_lambda(self):
        d = _make(lam=20)
        self.assertEqual(d._mu, 10)

    def test_recombination_weights_sum_to_one(self):
        d = _make(lam=16)
        self.assertAlmostEqual(float(d._weights.sum()), 1.0, places=10)

    def test_recombination_weights_positive_decreasing(self):
        d = _make(lam=10)
        w = d._weights
        for i in range(len(w) - 1):
            self.assertGreater(float(w[i]), float(w[i + 1]))

    def test_initial_sigma_stored(self):
        d = _make(sigma=0.7)
        self.assertAlmostEqual(d._sigma, 0.7)

    def test_covariance_is_identity(self):
        d = _make(n=4)
        np.testing.assert_array_almost_equal(d._C, np.eye(4))

    def test_ps_and_pc_are_zero(self):
        d = _make(n=6)
        np.testing.assert_array_equal(d._ps, np.zeros(6))
        np.testing.assert_array_equal(d._pc, np.zeros(6))

    def test_generation_starts_at_zero(self):
        d = _make()
        self.assertEqual(d._gen, 0)

    def test_mu_eff_positive(self):
        d = _make(lam=20)
        self.assertGreater(d._mu_eff, 1.0)


class TestCMAESDistributionInitializeRandom(unittest.TestCase):

    def test_mean_is_zero_after_initialize(self):
        d = _make(n=5)
        d.initialize_random()
        np.testing.assert_array_equal(d._mean, np.zeros(5))


class TestCMAESDistributionSample(unittest.TestCase):

    def test_sample_returns_lambda_vectors(self):
        d   = _make(n=4, lam=8)
        pop = d.sample()
        self.assertEqual(len(pop), 8)

    def test_sample_vector_shape(self):
        d   = _make(n=6, lam=5)
        pop = d.sample()
        for x in pop:
            self.assertEqual(x.shape, (6,))

    def test_sample_vectors_are_distinct(self):
        d   = _make(n=4, lam=6, seed=1)
        pop = d.sample()
        for i in range(len(pop)):
            for j in range(i + 1, len(pop)):
                self.assertFalse(np.allclose(pop[i], pop[j]))

    def test_sample_fills_pop_xs_and_pop_ys(self):
        d = _make(n=4, lam=6)
        d.sample()
        self.assertEqual(len(d._pop_xs), 6)
        self.assertEqual(len(d._pop_ys), 6)

    def test_sample_reproducible_with_same_seed(self):
        d1 = _make(n=4, lam=5, seed=7)
        d2 = _make(n=4, lam=5, seed=7)
        pop1 = d1.sample()
        pop2 = d2.sample()
        for x1, x2 in zip(pop1, pop2):
            np.testing.assert_array_equal(x1, x2)


class TestCMAESDistributionUpdate(unittest.TestCase):

    def _sample_and_update(self, d, rewards=None):
        d.sample()
        if rewards is None:
            rewards = [float(i) for i in range(d._lam)]
        return d.update(rewards)

    def test_update_returns_tuple(self):
        d = _make(n=3, lam=6)
        result = self._sample_and_update(d)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_update_returns_best_reward(self):
        d   = _make(n=3, lam=6)
        d.sample()
        rewards = [1.0, 5.0, 2.0, 3.0, 0.5, 4.0]
        best_r, _ = d.update(rewards)
        self.assertAlmostEqual(best_r, 5.0)

    def test_update_returns_best_index(self):
        d = _make(n=3, lam=6, seed=0)
        d.sample()
        rewards = [1.0, 5.0, 2.0, 3.0, 0.5, 4.0]
        _, best_idx = d.update(rewards)
        self.assertEqual(best_idx, 1)

    def test_generation_increments_after_update(self):
        d = _make(n=3, lam=6)
        self._sample_and_update(d)
        self.assertEqual(d._gen, 1)

    def test_mean_changes_after_update(self):
        d        = _make(n=5, lam=8, seed=0)
        old_mean = d._mean.copy()
        self._sample_and_update(d)
        self.assertFalse(np.allclose(d._mean, old_mean))

    def test_wrong_reward_count_raises(self):
        d = _make(n=3, lam=6)
        d.sample()
        with self.assertRaises(ValueError):
            d.update([1.0] * 5)

    def test_update_without_sample_raises(self):
        d = _make(n=3, lam=6)
        # Never called sample() — _pop_xs/_pop_ys are empty
        with self.assertRaises(RuntimeError):
            d.update([1.0] * 6)

    def test_sigma_positive_after_update(self):
        d = _make(n=4, lam=8)
        self._sample_and_update(d)
        self.assertGreater(d._sigma, 0.0)

    def test_covariance_symmetric_after_update(self):
        d = _make(n=5, lam=10)
        self._sample_and_update(d)
        np.testing.assert_array_almost_equal(d._C, d._C.T, decimal=12)

    def test_covariance_positive_definite_after_update(self):
        d = _make(n=5, lam=10)
        for _ in range(3):
            self._sample_and_update(d)
        eigvals = np.linalg.eigvalsh(d._C)
        self.assertTrue(np.all(eigvals > 0))

    def test_multiple_generations_change_mean(self):
        d = _make(n=4, lam=8)
        mean_after = []
        for _ in range(5):
            self._sample_and_update(d)
            mean_after.append(d._mean.copy())
        for i in range(len(mean_after) - 1):
            self.assertFalse(np.allclose(mean_after[i], mean_after[i + 1]))


class TestCMAESDistributionConvergence(unittest.TestCase):

    def test_converges_toward_quadratic_minimum(self):
        """After 30 generations the mean should be closer to zero than at start."""
        n      = 5
        lam    = 20
        d      = CMAESDistribution(n, population_size=lam, initial_sigma=1.0, seed=0)
        # Force a known non-zero starting mean
        d._mean = np.array([3.0, -2.0, 1.5, -1.0, 2.5])
        target = np.zeros(n)

        initial_dist = float(np.linalg.norm(d._mean - target))

        for _ in range(30):
            xs = d.sample()
            rewards = [-float(np.sum(x ** 2)) for x in xs]
            d.update(rewards)

        final_dist = float(np.linalg.norm(d._mean - target))
        self.assertLess(final_dist, initial_dist,
                        f"CMA-ES failed to converge: initial={initial_dist:.3f}, "
                        f"final={final_dist:.3f}")


class TestCMAESDistributionSaveLoad(unittest.TestCase):

    def test_save_load_preserves_mean(self):
        import tempfile, os
        d = _make(n=4, lam=8, seed=5)
        for _ in range(3):
            d.sample()
            d.update([float(i) for i in range(8)])

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            d.save_state(path)
            d2 = _make(n=4, lam=8)
            d2.load_state(path)
            np.testing.assert_array_equal(d._mean, d2._mean)
        finally:
            os.unlink(path)

    def test_save_load_preserves_sigma(self):
        import tempfile, os
        d = _make(n=4, lam=8, seed=3)
        d.sample()
        d.update([float(i) for i in range(8)])

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            d.save_state(path)
            d2 = _make(n=4, lam=8)
            d2.load_state(path)
            self.assertAlmostEqual(d._sigma, d2._sigma, places=12)
        finally:
            os.unlink(path)

    def test_save_load_preserves_covariance(self):
        import tempfile, os
        d = _make(n=4, lam=8, seed=2)
        for _ in range(5):
            d.sample()
            d.update([float(i) for i in range(8)])

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            d.save_state(path)
            d2 = _make(n=4, lam=8)
            d2.load_state(path)
            np.testing.assert_array_almost_equal(d._C, d2._C, decimal=12)
        finally:
            os.unlink(path)

    def test_save_load_preserves_generation(self):
        import tempfile, os
        d = _make(n=4, lam=8)
        for _ in range(7):
            d.sample()
            d.update([float(i) for i in range(8)])

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            d.save_state(path)
            d2 = _make(n=4, lam=8)
            d2.load_state(path)
            self.assertEqual(d._gen, d2._gen)
        finally:
            os.unlink(path)

    def test_load_wrong_dimension_raises(self):
        import tempfile, os
        d = _make(n=4, lam=8)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            d.save_state(path)
            d2 = _make(n=7, lam=8)
            with self.assertRaises(ValueError):
                d2.load_state(path)
        finally:
            os.unlink(path)

    def test_evolution_continues_after_load(self):
        """After loading state, evolution should continue from the loaded mean."""
        import tempfile, os
        d = _make(n=4, lam=8, seed=9)
        for _ in range(5):
            d.sample()
            d.update([float(i) for i in range(8)])

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            d.save_state(path)

            d2 = CMAESDistribution(4, population_size=8, seed=99)
            d2.load_state(path)

            # Verify the loaded mean matches
            np.testing.assert_array_equal(d._mean, d2._mean)

            # Verify evolution continues from the loaded state (mean shifts)
            prev_mean = d2._mean.copy()
            d2.sample()
            d2.update([float(i) for i in range(8)])
            self.assertFalse(np.allclose(d2._mean, prev_mean),
                             "Mean did not change after continuing evolution from loaded state")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
