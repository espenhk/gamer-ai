"""Tests for LSTMPolicy and LSTMEvolutionPolicy in games/tmnf/policies.py."""
import unittest

import numpy as np

from policies import LSTMPolicy, LSTMEvolutionPolicy
from games.tmnf.obs_spec import BASE_OBS_DIM


_OBS_DIM = BASE_OBS_DIM


def _zero_obs(n_lidar_rays: int = 0) -> np.ndarray:
    return np.zeros(_OBS_DIM + n_lidar_rays, dtype=np.float32)


def _rand_obs(n_lidar_rays: int = 0, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(_OBS_DIM + n_lidar_rays).astype(np.float32)


# ---------------------------------------------------------------------------
# LSTMPolicy unit tests
# ---------------------------------------------------------------------------

class TestLSTMPolicyStructure(unittest.TestCase):

    def setUp(self):
        self.policy = LSTMPolicy(hidden_size=8, n_lidar_rays=0, seed=0)

    def test_action_shape(self):
        action = self.policy(_zero_obs())
        self.assertEqual(action.shape, (3,))

    def test_steer_in_range(self):
        for _ in range(20):
            action = self.policy(_zero_obs())
            self.assertGreaterEqual(float(action[0]), -1.0)
            self.assertLessEqual(float(action[0]),     1.0)

    def test_accel_binary(self):
        for _ in range(20):
            action = self.policy(_zero_obs())
            self.assertIn(float(action[1]), (0.0, 1.0))

    def test_brake_binary(self):
        for _ in range(20):
            action = self.policy(_zero_obs())
            self.assertIn(float(action[2]), (0.0, 1.0))

    def test_hidden_state_updates_after_call(self):
        h_before = self.policy._h.copy()
        c_before = self.policy._c.copy()
        self.policy(_rand_obs())
        self.assertFalse(np.allclose(self.policy._h, h_before) and
                         np.allclose(self.policy._c, c_before),
                         "Hidden state (h, c) did not change after __call__")

    def test_episode_reset_zeros_hidden_state(self):
        # Drive the hidden state to non-zero values
        for _ in range(5):
            self.policy(_rand_obs())
        self.policy.on_episode_end()
        np.testing.assert_array_equal(self.policy._h, np.zeros(8))
        np.testing.assert_array_equal(self.policy._c, np.zeros(8))

    def test_update_is_noop(self):
        obs    = _zero_obs()
        w_snap = [w.copy() for w in [
            self.policy._W_f, self.policy._W_i,
            self.policy._W_g, self.policy._W_o,
        ]]
        self.policy.update(obs, np.array([0, 1, 0]), 1.0, obs, False)
        for i, w in enumerate([self.policy._W_f, self.policy._W_i,
                                self.policy._W_g, self.policy._W_o]):
            np.testing.assert_array_equal(w, w_snap[i])

    def test_different_history_gives_different_action(self):
        """Same final observation but different prior history → different action."""
        obs_seq  = [_rand_obs(seed=i) for i in range(5)]
        final    = _rand_obs(seed=99)

        policy_a = LSTMPolicy(hidden_size=16, seed=7)
        policy_b = LSTMPolicy(hidden_size=16, seed=7)

        # Feed different history to each
        for obs in obs_seq:
            policy_a(obs)
        # policy_b stays at h=c=0

        action_a = policy_a(final)
        action_b = policy_b(final)

        self.assertFalse(np.allclose(action_a, action_b),
                         "Policies with different histories produced identical actions")


class TestLSTMFlatInterface(unittest.TestCase):

    def setUp(self):
        self.policy = LSTMPolicy(hidden_size=8, n_lidar_rays=0, seed=1)

    def test_flat_dim_correct(self):
        h    = 8
        d    = _OBS_DIM
        c_in = h + d
        expected = 4 * (h * c_in + h) + 3 * h
        self.assertEqual(self.policy.flat_dim, expected)

    def test_to_flat_shape(self):
        flat = self.policy.to_flat()
        self.assertEqual(flat.shape, (self.policy.flat_dim,))

    def test_flat_roundtrip_values(self):
        flat1  = self.policy.to_flat()
        policy2 = self.policy.with_flat(flat1)
        flat2  = policy2.to_flat()
        np.testing.assert_array_almost_equal(flat1, flat2, decimal=6)

    def test_with_flat_zeroed_hidden_state(self):
        """with_flat must produce h=c=0 regardless of original state."""
        # Advance hidden state
        for _ in range(3):
            self.policy(_rand_obs())
        flat     = self.policy.to_flat()
        policy2  = self.policy.with_flat(flat)
        np.testing.assert_array_equal(policy2._h, np.zeros(8))
        np.testing.assert_array_equal(policy2._c, np.zeros(8))

    def test_with_flat_preserves_weights(self):
        flat    = self.policy.to_flat()
        policy2 = self.policy.with_flat(flat)
        np.testing.assert_array_almost_equal(
            self.policy._W_f, policy2._W_f, decimal=6)
        np.testing.assert_array_almost_equal(
            self.policy._W_steer, policy2._W_steer, decimal=6)

    def test_with_flat_wrong_size_raises(self):
        wrong_flat = np.zeros(self.policy.flat_dim + 1, dtype=np.float32)
        with self.assertRaises(ValueError):
            self.policy.with_flat(wrong_flat)

    def test_mutated_differs_from_original(self):
        mutant = self.policy.mutated(scale=0.1)
        flat_o = self.policy.to_flat()
        flat_m = mutant.to_flat()
        self.assertFalse(np.allclose(flat_o, flat_m))

    def test_mutated_same_hidden_size(self):
        mutant = self.policy.mutated(scale=0.5)
        self.assertEqual(mutant._hidden_size, self.policy._hidden_size)
        self.assertEqual(mutant._obs_dim,     self.policy._obs_dim)

    def test_flat_roundtrip_with_lidar(self):
        p    = LSTMPolicy(hidden_size=4, n_lidar_rays=3, seed=5)
        flat = p.to_flat()
        p2   = p.with_flat(flat)
        np.testing.assert_array_almost_equal(flat, p2.to_flat(), decimal=6)


class TestLSTMPolicySerialisation(unittest.TestCase):

    def test_to_cfg_contains_required_keys(self):
        p   = LSTMPolicy(hidden_size=8)
        cfg = p.to_cfg()
        for key in ("policy_type", "hidden_size", "n_lidar_rays", "obs_dim",
                    "W_f", "b_f", "W_i", "b_i", "W_g", "b_g", "W_o", "b_o",
                    "W_steer", "W_accel", "W_brake"):
            self.assertIn(key, cfg)

    def test_policy_type_string(self):
        p = LSTMPolicy(hidden_size=8)
        self.assertEqual(p.to_cfg()["policy_type"], "lstm")

    def test_from_cfg_roundtrip(self):
        p   = LSTMPolicy(hidden_size=8, seed=3)
        cfg = p.to_cfg()
        p2  = LSTMPolicy.from_cfg(cfg)
        np.testing.assert_array_almost_equal(p._W_f,    p2._W_f,    decimal=6)
        np.testing.assert_array_almost_equal(p._W_steer, p2._W_steer, decimal=6)
        self.assertEqual(p2._hidden_size, 8)
        np.testing.assert_array_equal(p2._h, np.zeros(8))

    def test_save_and_reload(self):
        import tempfile, os, yaml
        p = LSTMPolicy(hidden_size=4, seed=42)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            p.save(path)
            with open(path) as f:
                cfg = yaml.safe_load(f)
            self.assertEqual(cfg["policy_type"], "lstm")
            p2 = LSTMPolicy.from_cfg(cfg)
            np.testing.assert_array_almost_equal(p._W_i, p2._W_i, decimal=5)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# LSTMEvolutionPolicy unit tests
# ---------------------------------------------------------------------------

class TestLSTMEvolutionPolicyInit(unittest.TestCase):

    def test_population_size_property(self):
        policy = LSTMEvolutionPolicy(population_size=12)
        self.assertEqual(policy.population_size, 12)

    def test_sigma_property(self):
        policy = LSTMEvolutionPolicy(initial_sigma=0.07)
        self.assertAlmostEqual(policy.sigma, 0.07)

    def test_champion_reward_starts_at_neg_inf(self):
        policy = LSTMEvolutionPolicy()
        self.assertEqual(policy.champion_reward, float("-inf"))

    def test_flat_dim_matches_template(self):
        policy = LSTMEvolutionPolicy(hidden_size=8)
        self.assertEqual(policy._flat_dim, policy._template.flat_dim)

    def test_mu_is_half_lambda(self):
        policy = LSTMEvolutionPolicy(population_size=20)
        self.assertEqual(policy._mu, 10)

    def test_recomb_weights_sum_to_one(self):
        policy = LSTMEvolutionPolicy(population_size=16)
        self.assertAlmostEqual(float(policy._recomb_w.sum()), 1.0, places=10)


class TestLSTMEvolutionPolicySampling(unittest.TestCase):

    def test_sample_population_count(self):
        policy = LSTMEvolutionPolicy(hidden_size=4, population_size=8, seed=0)
        pop    = policy.sample_population()
        self.assertEqual(len(pop), 8)

    def test_sample_population_returns_lstm_policies(self):
        policy = LSTMEvolutionPolicy(hidden_size=4, population_size=6, seed=1)
        for ind in policy.sample_population():
            self.assertIsInstance(ind, LSTMPolicy)

    def test_pop_buffer_fills_on_sample(self):
        policy = LSTMEvolutionPolicy(hidden_size=4, population_size=6, seed=2)
        policy.sample_population()
        self.assertEqual(len(policy._pop), 6)

    def test_sample_produces_distinct_individuals(self):
        policy = LSTMEvolutionPolicy(hidden_size=4, population_size=6, seed=3)
        pop    = policy.sample_population()
        flat_0 = pop[0].to_flat()
        flat_1 = pop[1].to_flat()
        self.assertFalse(np.allclose(flat_0, flat_1))


class TestLSTMEvolutionPolicyUpdate(unittest.TestCase):

    def _setup(self, pop=8, seed=0):
        policy = LSTMEvolutionPolicy(hidden_size=4, population_size=pop,
                                      initial_sigma=0.1, seed=seed)
        policy.sample_population()
        return policy

    def test_update_returns_true_first_time(self):
        policy = self._setup()
        improved = policy.update_distribution([float(i) for i in range(8)])
        self.assertTrue(improved)

    def test_update_sets_champion(self):
        policy = self._setup()
        policy.update_distribution([float(i) for i in range(8)])
        self.assertIsNotNone(policy._champion)
        self.assertIsInstance(policy._champion, LSTMPolicy)

    def test_champion_reward_is_best_seen(self):
        policy = self._setup()
        policy.update_distribution([5.0, 99.0] + [1.0] * 6)
        self.assertAlmostEqual(policy.champion_reward, 99.0)

    def test_update_returns_false_when_no_improvement(self):
        policy = self._setup()
        policy.update_distribution([100.0] * 8)   # sets champion_reward = 100
        policy.sample_population()
        improved = policy.update_distribution([50.0] * 8)
        self.assertFalse(improved)

    def test_mean_shifts_after_update(self):
        policy   = self._setup()
        old_mean = policy._mean.copy()
        policy.update_distribution([float(i) for i in range(8)])
        self.assertFalse(np.allclose(policy._mean, old_mean))

    def test_wrong_reward_count_raises(self):
        policy = self._setup()
        with self.assertRaises(ValueError):
            policy.update_distribution([0.0] * 5)  # wrong count

    def test_update_without_sample_raises(self):
        policy = LSTMEvolutionPolicy(hidden_size=4, population_size=8, seed=0)
        # Never called sample_population() — _pop is empty
        with self.assertRaises(RuntimeError):
            policy.update_distribution([0.0] * 8)

    def test_sigma_adapts(self):
        policy   = self._setup()
        old_sigma = policy.sigma
        policy.update_distribution([float(i) for i in range(8)])
        # sigma should have changed (up or down)
        self.assertNotAlmostEqual(policy.sigma, old_sigma)


class TestLSTMEvolutionPolicyCallable(unittest.TestCase):

    def test_call_raises_before_any_update(self):
        policy = LSTMEvolutionPolicy(hidden_size=4, population_size=4, seed=0)
        policy.sample_population()
        with self.assertRaises(RuntimeError):
            policy(_zero_obs())

    def test_call_returns_valid_action_after_update(self):
        policy = LSTMEvolutionPolicy(hidden_size=4, population_size=4, seed=0)
        policy.sample_population()
        policy.update_distribution([float(i) for i in range(4)])
        action = policy(_zero_obs())
        self.assertEqual(action.shape, (3,))
        self.assertGreaterEqual(float(action[0]), -1.0)
        self.assertLessEqual(float(action[0]),     1.0)
        self.assertIn(float(action[1]), (0.0, 1.0))
        self.assertIn(float(action[2]), (0.0, 1.0))

    def test_on_episode_end_resets_champion_hidden_state(self):
        policy = LSTMEvolutionPolicy(hidden_size=8, population_size=4, seed=0)
        policy.sample_population()
        policy.update_distribution([float(i) for i in range(4)])
        # Drive champion hidden state non-zero
        for _ in range(3):
            policy(_rand_obs())
        policy.on_episode_end()
        np.testing.assert_array_equal(policy._champion._h, np.zeros(8))
        np.testing.assert_array_equal(policy._champion._c, np.zeros(8))


class TestLSTMEvolutionPolicySerialisation(unittest.TestCase):

    def test_to_cfg_keys(self):
        policy = LSTMEvolutionPolicy(hidden_size=4, population_size=6)
        cfg    = policy.to_cfg()
        for key in ("policy_type", "hidden_size", "population_size",
                    "sigma", "n_lidar_rays", "champion_reward"):
            self.assertIn(key, cfg)

    def test_policy_type_string(self):
        policy = LSTMEvolutionPolicy()
        self.assertEqual(policy.to_cfg()["policy_type"], "lstm")

    def test_save_writes_lstm_yaml(self):
        import tempfile, os, yaml
        policy = LSTMEvolutionPolicy(hidden_size=4, population_size=4, seed=0)
        policy.sample_population()
        policy.update_distribution([float(i) for i in range(4)])
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            policy.save(path)
            with open(path) as f:
                cfg = yaml.safe_load(f)
            self.assertEqual(cfg["policy_type"], "lstm")
            self.assertIn("W_f", cfg)
        finally:
            os.unlink(path)

    def test_initialize_from_champion(self):
        champion = LSTMPolicy(hidden_size=8, seed=5)
        policy   = LSTMEvolutionPolicy(hidden_size=8, population_size=4)
        policy.initialize_from_champion(champion)
        np.testing.assert_array_almost_equal(
            policy._mean, champion.to_flat().astype(np.float64), decimal=6)
        self.assertIs(policy._champion, champion)


class TestLSTMEvolutionConvergence(unittest.TestCase):

    def test_mean_moves_toward_target(self):
        """ES mean should move toward the minimizer of a quadratic after 20 generations."""
        policy = LSTMEvolutionPolicy(hidden_size=4, population_size=16,
                                      initial_sigma=0.5, n_lidar_rays=0, seed=42)
        target = np.ones(policy._flat_dim, dtype=np.float64)
        initial_dist = float(np.linalg.norm(policy._mean - target))

        for _ in range(20):
            individuals = policy.sample_population()
            rewards = [
                -float(np.sum((ind.to_flat().astype(np.float64) - target) ** 2))
                for ind in individuals
            ]
            policy.update_distribution(rewards)

        final_dist = float(np.linalg.norm(policy._mean - target))
        self.assertLess(final_dist, initial_dist,
                        f"ES failed to converge: initial={initial_dist:.3f}, "
                        f"final={final_dist:.3f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
