"""Tests for CMAESPolicy in policies.py."""
import unittest

import numpy as np

from games.tmnf.policies import CMAESPolicy, WeightedLinearPolicy


class TestCMAESPolicyInit(unittest.TestCase):

    def test_default_population_size(self):
        policy = CMAESPolicy(population_size=10, initial_sigma=0.3)
        self.assertEqual(policy._lam, 10)

    def test_mu_is_half_lambda(self):
        policy = CMAESPolicy(population_size=20)
        self.assertEqual(policy._mu, 10)

    def test_weights_sum_to_one(self):
        policy = CMAESPolicy(population_size=16)
        self.assertAlmostEqual(float(policy._weights.sum()), 1.0, places=10)

    def test_covariance_is_identity_at_init(self):
        policy = CMAESPolicy(population_size=10)
        np.testing.assert_array_almost_equal(policy._C, np.eye(policy._n))

    def test_initialize_random_sets_zero_mean(self):
        policy = CMAESPolicy(population_size=10)
        policy.initialize_random()
        np.testing.assert_array_equal(policy._mean, np.zeros(policy._n))

    def test_initialize_from_champion_seeds_mean(self):
        policy    = CMAESPolicy(population_size=10, n_lidar_rays=0)
        names     = WeightedLinearPolicy.get_obs_names(0)
        cfg       = {
            "steer_weights": {n: 1.0 for n in names},
            "accel_weights": {n: 0.0 for n in names},
            "brake_weights": {n: 0.0 for n in names},
        }
        champion  = WeightedLinearPolicy.from_cfg(cfg)
        policy.initialize_from_champion(champion)
        expected  = champion.to_flat().astype(np.float64)
        np.testing.assert_array_almost_equal(policy._mean, expected)

    def test_initialize_from_champion_sets_champion(self):
        policy   = CMAESPolicy(population_size=10)
        names    = WeightedLinearPolicy.get_obs_names(0)
        cfg      = {
            "steer_weights": {n: 0.5 for n in names},
            "accel_weights": {n: 0.5 for n in names},
            "brake_weights": {n: 0.0 for n in names},
        }
        champion = WeightedLinearPolicy.from_cfg(cfg)
        policy.initialize_from_champion(champion)
        self.assertIs(policy._champion, champion)


class TestCMAESPolicySampling(unittest.TestCase):

    def test_sample_population_returns_correct_count(self):
        policy = CMAESPolicy(population_size=12)
        policy.initialize_random()
        population = policy.sample_population()
        self.assertEqual(len(population), 12)

    def test_sample_population_returns_weighted_linear_policies(self):
        policy = CMAESPolicy(population_size=8)
        policy.initialize_random()
        population = policy.sample_population()
        for ind in population:
            self.assertIsInstance(ind, WeightedLinearPolicy)

    def test_pop_xs_and_ys_filled_after_sample(self):
        policy = CMAESPolicy(population_size=6)
        policy.initialize_random()
        policy.sample_population()
        self.assertEqual(len(policy._pop_xs), 6)
        self.assertEqual(len(policy._pop_ys), 6)


class TestCMAESPolicyUpdate(unittest.TestCase):

    def _make_and_sample(self, pop=10):
        policy = CMAESPolicy(population_size=pop, initial_sigma=0.5)
        policy.initialize_random()
        policy.sample_population()
        return policy

    def test_update_sets_champion_on_first_call(self):
        policy = self._make_and_sample()
        rewards = list(range(policy._lam))
        policy.update_distribution(rewards)
        self.assertIsNotNone(policy._champion)

    def test_update_returns_true_when_champion_improved(self):
        policy = self._make_and_sample()
        rewards = [float(i) for i in range(policy._lam)]
        improved = policy.update_distribution(rewards)
        self.assertTrue(improved)

    def test_update_returns_false_when_no_improvement(self):
        policy = self._make_and_sample()
        rewards = [100.0] * policy._lam
        policy.update_distribution(rewards)     # sets champion_reward = 100.0

        policy.sample_population()
        improved = policy.update_distribution([50.0] * policy._lam)
        self.assertFalse(improved)

    def test_champion_reward_is_best_seen(self):
        policy = self._make_and_sample()
        rewards = [10.0, 99.0] + [1.0] * (policy._lam - 2)
        policy.update_distribution(rewards)
        self.assertAlmostEqual(policy.champion_reward, 99.0)

    def test_generation_counter_increments(self):
        policy = self._make_and_sample()
        policy.update_distribution([0.0] * policy._lam)
        self.assertEqual(policy._gen, 1)

    def test_wrong_reward_count_raises(self):
        policy = self._make_and_sample()
        with self.assertRaises(ValueError):
            policy.update_distribution([0.0] * (policy._lam - 1))

    def test_update_without_sample_raises(self):
        policy = CMAESPolicy(population_size=6)
        policy.initialize_random()
        # Never called sample_population() — _pop_xs/_pop_ys are empty
        with self.assertRaises(RuntimeError):
            policy.update_distribution([0.0] * 6)

    def test_mean_moves_after_update(self):
        policy = CMAESPolicy(population_size=10, initial_sigma=0.5)
        policy.initialize_random()
        old_mean = policy._mean.copy()
        policy.sample_population()
        # Give high reward to all — any non-zero step should shift the mean
        rewards = [float(i) for i in range(policy._lam)]
        policy.update_distribution(rewards)
        self.assertFalse(np.allclose(policy._mean, old_mean))


class TestCMAESPolicyCallable(unittest.TestCase):

    def test_call_raises_before_any_update(self):
        policy = CMAESPolicy(population_size=6)
        policy.initialize_random()
        obs = np.zeros(21, dtype=np.float32)
        with self.assertRaises(RuntimeError):
            policy(obs)

    def test_call_returns_valid_action_after_update(self):
        policy = CMAESPolicy(population_size=6, initial_sigma=0.3)
        policy.initialize_random()
        policy.sample_population()
        policy.update_distribution([float(i) for i in range(6)])

        from games.tmnf.obs_spec import BASE_OBS_DIM
        obs    = np.zeros(BASE_OBS_DIM, dtype=np.float32)
        action = policy(obs)
        self.assertEqual(action.shape, (3,))
        self.assertGreaterEqual(float(action[0]), -1.0)
        self.assertLessEqual(float(action[0]),  1.0)
        self.assertIn(float(action[1]), (0.0, 1.0))
        self.assertIn(float(action[2]), (0.0, 1.0))


class TestCMAESPolicySerialisation(unittest.TestCase):

    def test_to_cfg_contains_required_keys(self):
        policy = CMAESPolicy(population_size=10, initial_sigma=0.3)
        cfg    = policy.to_cfg()
        for key in ("policy_type", "population_size", "sigma",
                    "n_lidar_rays", "champion_reward"):
            self.assertIn(key, cfg)

    def test_to_cfg_policy_type(self):
        policy = CMAESPolicy(population_size=10)
        self.assertEqual(policy.to_cfg()["policy_type"], "cmaes")

    def test_save_writes_weighted_linear_yaml(self, tmp_path=None):
        import tempfile, os
        policy = CMAESPolicy(population_size=6, initial_sigma=0.3)
        policy.initialize_random()
        policy.sample_population()
        policy.update_distribution([float(i) for i in range(6)])

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            policy.save(path)
            self.assertTrue(os.path.exists(path))
            import yaml
            with open(path) as f:
                cfg = yaml.safe_load(f)
            self.assertIn("steer_weights", cfg)
            self.assertIn("accel_weights", cfg)
            self.assertIn("brake_weights", cfg)
        finally:
            os.unlink(path)


class TestCMAESConvergence(unittest.TestCase):

    def test_converges_toward_quadratic_maximum(self):
        """CMA-ES mean should move toward the maximizer of a quadratic in <=50 generations."""
        from games.tmnf.obs_spec import BASE_OBS_DIM

        policy  = CMAESPolicy(population_size=20, initial_sigma=1.0, n_lidar_rays=0, seed=42)
        policy.initialize_random()

        n_weights = BASE_OBS_DIM * 3
        target    = np.ones(n_weights, dtype=np.float64)

        initial_dist = float(np.linalg.norm(policy._mean - target))

        for _ in range(50):
            individuals = policy.sample_population()
            rewards = [
                -float(np.sum((ind.to_flat().astype(np.float64) - target) ** 2))
                for ind in individuals
            ]
            policy.update_distribution(rewards)

        final_dist = float(np.linalg.norm(policy._mean - target))
        self.assertLess(
            final_dist, initial_dist,
            f"CMA-ES failed to converge: initial_dist={initial_dist:.3f}, "
            f"final_dist={final_dist:.3f}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
