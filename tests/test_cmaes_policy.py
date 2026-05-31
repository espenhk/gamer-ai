"""Tests for CMAESPolicy in framework/cmaes.py."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from framework.cmaes import CMAESPolicy
from framework.policies import WeightedLinearPolicy
from games.tmnf.obs_spec import BASE_OBS_DIM, TMNF_OBS_SPEC

# ---------------------------------------------------------------------------
# TMNF helper: build a CMAESPolicy that produces WeightedLinearPolicy instances
# ---------------------------------------------------------------------------

_OBS_SPEC = TMNF_OBS_SPEC
_N_HEADS = 3  # steer + accel + brake


def _tmnf_factory(flat: np.ndarray, obs_spec) -> WeightedLinearPolicy:
    """Convert a flat [steer|accel|brake] vector into a WeightedLinearPolicy."""
    names = obs_spec.names
    n = obs_spec.dim
    cfg = {
        "steer_weights": {names[i]: float(flat[i]) for i in range(n)},
        "accel_weights": {names[i]: float(flat[n + i]) for i in range(n)},
        "brake_weights": {names[i]: float(flat[2 * n + i]) for i in range(n)},
    }
    return WeightedLinearPolicy.from_cfg(cfg, obs_spec, ["steer", "accel", "brake"])


def _make_tmnf_policy(**kw) -> CMAESPolicy:
    """Create a TMNF-flavoured CMAESPolicy via the framework API."""
    n_params = _OBS_SPEC.dim * _N_HEADS
    defaults = dict(population_size=10, initial_sigma=0.3)
    defaults.update(kw)
    return CMAESPolicy(_OBS_SPEC, _tmnf_factory, n_params, **defaults)


# ---------------------------------------------------------------------------
# Tests — initialisation
# ---------------------------------------------------------------------------


class TestCMAESPolicyInit(unittest.TestCase):
    def test_default_population_size(self):
        policy = _make_tmnf_policy(population_size=10, initial_sigma=0.3)
        self.assertEqual(policy._lam, 10)

    def test_mu_is_half_lambda(self):
        policy = _make_tmnf_policy(population_size=20)
        self.assertEqual(policy._mu, 10)

    def test_weights_sum_to_one(self):
        policy = _make_tmnf_policy(population_size=16)
        self.assertAlmostEqual(float(policy._weights.sum()), 1.0, places=10)

    def test_covariance_is_identity_at_init(self):
        policy = _make_tmnf_policy(population_size=10)
        np.testing.assert_array_almost_equal(policy._C, np.eye(policy._n))

    def test_initialize_random_sets_zero_mean(self):
        policy = _make_tmnf_policy(population_size=10)
        policy.initialize_random()
        np.testing.assert_array_equal(policy._mean, np.zeros(policy._n))

    def test_initialize_from_champion_seeds_mean(self):
        policy = _make_tmnf_policy(population_size=10)
        names = _OBS_SPEC.names
        cfg = {
            "steer_weights": {n: 1.0 for n in names},
            "accel_weights": {n: 0.0 for n in names},
            "brake_weights": {n: 0.0 for n in names},
        }
        champion = WeightedLinearPolicy.from_cfg(cfg, _OBS_SPEC, ["steer", "accel", "brake"])
        policy.initialize_from_champion(champion)
        expected = champion.to_flat().astype(np.float64)
        np.testing.assert_array_almost_equal(policy._mean, expected)

    def test_initialize_from_champion_sets_champion(self):
        policy = _make_tmnf_policy(population_size=10)
        names = _OBS_SPEC.names
        cfg = {
            "steer_weights": {n: 0.5 for n in names},
            "accel_weights": {n: 0.5 for n in names},
            "brake_weights": {n: 0.0 for n in names},
        }
        champion = WeightedLinearPolicy.from_cfg(cfg, _OBS_SPEC, ["steer", "accel", "brake"])
        policy.initialize_from_champion(champion)
        self.assertIs(policy._champion, champion)


# ---------------------------------------------------------------------------
# Tests — sampling
# ---------------------------------------------------------------------------


class TestCMAESPolicySampling(unittest.TestCase):
    def test_sample_population_returns_correct_count(self):
        policy = _make_tmnf_policy(population_size=12)
        policy.initialize_random()
        population = policy.sample_population()
        self.assertEqual(len(population), 12)

    def test_sample_population_returns_weighted_linear_policies(self):
        policy = _make_tmnf_policy(population_size=8)
        policy.initialize_random()
        population = policy.sample_population()
        for ind in population:
            self.assertIsInstance(ind, WeightedLinearPolicy)

    def test_pop_xs_and_ys_filled_after_sample(self):
        policy = _make_tmnf_policy(population_size=6)
        policy.initialize_random()
        policy.sample_population()
        self.assertEqual(len(policy._pop_xs), 6)
        self.assertEqual(len(policy._pop_ys), 6)


# ---------------------------------------------------------------------------
# Tests — update
# ---------------------------------------------------------------------------


class TestCMAESPolicyUpdate(unittest.TestCase):
    def _make_and_sample(self, pop=10):
        policy = _make_tmnf_policy(population_size=pop, initial_sigma=0.5)
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
        policy.update_distribution(rewards)

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
        policy = _make_tmnf_policy(population_size=6)
        policy.initialize_random()
        with self.assertRaises(RuntimeError):
            policy.update_distribution([0.0] * 6)

    def test_mean_moves_after_update(self):
        policy = _make_tmnf_policy(population_size=10, initial_sigma=0.5)
        policy.initialize_random()
        old_mean = policy._mean.copy()
        policy.sample_population()
        rewards = [float(i) for i in range(policy._lam)]
        policy.update_distribution(rewards)
        self.assertFalse(np.allclose(policy._mean, old_mean))


# ---------------------------------------------------------------------------
# Tests — callable
# ---------------------------------------------------------------------------


class TestCMAESPolicyCallable(unittest.TestCase):
    def test_call_raises_before_any_update(self):
        policy = _make_tmnf_policy(population_size=6)
        policy.initialize_random()
        obs = np.zeros(BASE_OBS_DIM, dtype=np.float32)
        with self.assertRaises(RuntimeError):
            policy(obs)

    def test_call_returns_valid_action_after_update(self):
        policy = _make_tmnf_policy(population_size=6, initial_sigma=0.3)
        policy.initialize_random()
        policy.sample_population()
        policy.update_distribution([float(i) for i in range(6)])

        obs = np.zeros(BASE_OBS_DIM, dtype=np.float32)
        action = policy(obs)
        self.assertEqual(action.shape, (3,))
        self.assertGreaterEqual(float(action[0]), -1.0)
        self.assertLessEqual(float(action[0]), 1.0)
        self.assertIn(float(action[1]), (0.0, 1.0))
        self.assertIn(float(action[2]), (0.0, 1.0))


# ---------------------------------------------------------------------------
# Tests — serialisation
# ---------------------------------------------------------------------------


class TestCMAESPolicySerialisation(unittest.TestCase):
    def test_to_cfg_contains_required_keys(self):
        policy = _make_tmnf_policy(population_size=10, initial_sigma=0.3)
        cfg = policy.to_cfg()
        for key in ("policy_type", "population_size", "sigma", "n_params", "champion_reward", "eval_episodes"):
            self.assertIn(key, cfg)

    def test_to_cfg_policy_type(self):
        policy = _make_tmnf_policy(population_size=10)
        self.assertEqual(policy.to_cfg()["policy_type"], "cmaes")

    def test_save_writes_weighted_linear_yaml(self, tmp_path=None):
        policy = _make_tmnf_policy(population_size=6, initial_sigma=0.3)
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


# ---------------------------------------------------------------------------
# Tests — eval_episodes attribute and training loop integration
# ---------------------------------------------------------------------------


class TestCMAESEvalEpisodes(unittest.TestCase):
    def test_eval_episodes_default_is_one(self):
        policy = _make_tmnf_policy(population_size=10)
        self.assertEqual(policy._eval_episodes, 1)

    def test_eval_episodes_stored(self):
        policy = _make_tmnf_policy(population_size=10, eval_episodes=3)
        self.assertEqual(policy._eval_episodes, 3)

    def test_eval_episodes_in_to_cfg(self):
        policy = _make_tmnf_policy(population_size=10, eval_episodes=4)
        cfg = policy.to_cfg()
        self.assertIn("eval_episodes", cfg)
        self.assertEqual(cfg["eval_episodes"], 4)

    def test_eval_episodes_clamped_to_at_least_one(self):
        policy = _make_tmnf_policy(population_size=10, eval_episodes=0)
        self.assertEqual(policy._eval_episodes, 1)

    def _run_one_gen_and_capture(self, pop_size, eval_episodes, rewards_seq):
        """Run one generation and capture rewards passed to update_distribution."""
        from framework.training import _greedy_loop_cmaes

        policy = _make_tmnf_policy(population_size=pop_size, initial_sigma=0.3, eval_episodes=eval_episodes)
        policy.initialize_random()

        rewards_iter = iter(rewards_seq)
        captured = []
        original_fn = policy.update_distribution

        def _capture(rewards):
            captured.append(list(rewards))
            return original_fn(rewards)

        class _SeqEnv:
            def reset(self):
                return np.zeros(BASE_OBS_DIM, dtype=np.float32), {}

            def step(self, action):
                info = {"track_progress": 0.5, "laps_completed": 0, "pos_x": 0.0, "pos_z": 0.0}
                return (np.zeros(BASE_OBS_DIM, dtype=np.float32), next(rewards_iter), True, False, info)

            def get_episode_time_limit(self):
                return None

            def set_episode_time_limit(self, _):
                pass

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            wf = f.name
        try:
            with patch.object(policy, "update_distribution", side_effect=_capture):
                _greedy_loop_cmaes(env=_SeqEnv(), policy=policy, n_generations=1, weights_file=wf)
        finally:
            if os.path.exists(wf):
                os.unlink(wf)

        return captured

    def test_eval_episodes_1_passes_single_reward(self):
        captured = self._run_one_gen_and_capture(
            pop_size=4,
            eval_episodes=1,
            rewards_seq=[10.0, 20.0, 30.0, 40.0],
        )
        self.assertEqual(len(captured), 1)
        self.assertEqual(len(captured[0]), 4)
        np.testing.assert_allclose(captured[0], [10.0, 20.0, 30.0, 40.0])

    def test_eval_episodes_3_averages_rewards(self):
        captured = self._run_one_gen_and_capture(
            pop_size=2,
            eval_episodes=3,
            rewards_seq=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )
        self.assertEqual(len(captured), 1)
        self.assertEqual(len(captured[0]), 2)
        np.testing.assert_allclose(captured[0], [20.0, 50.0])

    def test_eval_episodes_2_correct_reset_count(self):
        from framework.training import _greedy_loop_cmaes

        pop_size = 3
        eval_episodes = 2
        policy = _make_tmnf_policy(population_size=pop_size, initial_sigma=0.3, eval_episodes=eval_episodes)
        policy.initialize_random()

        reset_count = [0]
        rewards_iter = iter([float(i) for i in range(100)])

        class _CountingEnv:
            def reset(self):
                reset_count[0] += 1
                return np.zeros(BASE_OBS_DIM, dtype=np.float32), {}

            def step(self, action):
                info = {"track_progress": 0.5, "laps_completed": 0, "pos_x": 0.0, "pos_z": 0.0}
                return (np.zeros(BASE_OBS_DIM, dtype=np.float32), next(rewards_iter), True, False, info)

            def get_episode_time_limit(self):
                return None

            def set_episode_time_limit(self, _):
                pass

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            wf = f.name
        try:
            _greedy_loop_cmaes(env=_CountingEnv(), policy=policy, n_generations=1, weights_file=wf)
        finally:
            if os.path.exists(wf):
                os.unlink(wf)

        self.assertEqual(reset_count[0], pop_size * eval_episodes)

    def test_self_play_manager_refreshes_opponent(self):
        from framework.training import _greedy_loop_cmaes

        policy = _make_tmnf_policy(population_size=2, initial_sigma=0.3, eval_episodes=1)
        policy.initialize_random()

        class _Env:
            def __init__(self):
                self.set_opponent_policy = MagicMock()

            def reset(self):
                return np.zeros(BASE_OBS_DIM, dtype=np.float32), {}

            def step(self, action):
                info = {"track_progress": 0.5, "laps_completed": 0, "pos_x": 0.0, "pos_z": 0.0}
                return (np.zeros(BASE_OBS_DIM, dtype=np.float32), 1.0, True, False, info)

            def get_episode_time_limit(self):
                return None

            def set_episode_time_limit(self, _):
                pass

        env = _Env()
        manager = MagicMock()
        new_opp = MagicMock()
        manager.step.return_value = new_opp

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            wf = f.name
        try:
            loop = _greedy_loop_cmaes(
                env=env,
                policy=policy,
                n_generations=1,
                weights_file=wf,
                self_play_manager=manager,
            )
        finally:
            if os.path.exists(wf):
                os.unlink(wf)

        manager.step.assert_called_once_with(policy, loop.greedy_sims[0].improved)
        env.set_opponent_policy.assert_called_once_with(new_opp)


# ---------------------------------------------------------------------------
# Tests — convergence
# ---------------------------------------------------------------------------


class TestCMAESConvergence(unittest.TestCase):
    def test_converges_toward_quadratic_maximum(self):
        """CMA-ES mean should move toward the maximizer of a quadratic in ≤50 generations."""
        policy = _make_tmnf_policy(population_size=20, initial_sigma=1.0, seed=42)
        policy.initialize_random()

        n_weights = BASE_OBS_DIM * _N_HEADS
        target = np.ones(n_weights, dtype=np.float64)
        initial_dist = float(np.linalg.norm(policy._mean - target))

        for _ in range(50):
            individuals = policy.sample_population()
            rewards = [-float(np.sum((ind.to_flat().astype(np.float64) - target) ** 2)) for ind in individuals]
            policy.update_distribution(rewards)

        final_dist = float(np.linalg.norm(policy._mean - target))
        self.assertLess(
            final_dist,
            initial_dist,
            f"CMA-ES failed to converge: initial_dist={initial_dist:.3f}, final_dist={final_dist:.3f}",
        )


# ---------------------------------------------------------------------------
# Tests — trainer state
# ---------------------------------------------------------------------------


class TestCMAESTrainerState(unittest.TestCase):
    def _make_trained_policy(self, n_gens: int = 3) -> CMAESPolicy:
        policy = _make_tmnf_policy(population_size=10, initial_sigma=0.5, seed=42)
        policy.initialize_random()
        for _ in range(n_gens):
            policy.sample_population()
            policy.update_distribution([float(i) for i in range(10)])
        return policy

    def test_save_load_roundtrip_all_arrays(self):
        """save_trainer_state → load_trainer_state preserves all distribution arrays
        and the loaded state drives subsequent evolution correctly."""
        policy = self._make_trained_policy()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            policy.save_trainer_state(path)

            policy2 = _make_tmnf_policy(population_size=10, initial_sigma=0.99, seed=99)
            policy2.initialize_random()
            policy2.load_trainer_state(path)

            np.testing.assert_array_equal(policy._mean, policy2._mean)
            np.testing.assert_array_equal(policy._C, policy2._C)
            np.testing.assert_array_equal(policy._B, policy2._B)
            np.testing.assert_array_equal(policy._D, policy2._D)
            np.testing.assert_array_equal(policy._invsqrtC, policy2._invsqrtC)
            np.testing.assert_array_equal(policy._ps, policy2._ps)
            np.testing.assert_array_equal(policy._pc, policy2._pc)
            self.assertAlmostEqual(policy._sigma, policy2._sigma)
            self.assertEqual(policy._gen, policy2._gen)

            prev_mean = policy2._mean.copy()
            policy2.sample_population()
            policy2.update_distribution([float(i) for i in range(10)])
            self.assertEqual(policy2._gen, policy._gen + 1)
            self.assertFalse(np.allclose(policy2._mean, prev_mean))
        finally:
            os.unlink(path)

    def test_load_wrong_dimension_raises(self):
        """Loading state whose n differs from current obs space raises ValueError."""
        policy1 = _make_tmnf_policy(population_size=6)
        policy1.initialize_random()
        policy1.sample_population()
        policy1.update_distribution([float(i) for i in range(6)])

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            policy1.save_trainer_state(path)
            # Different n_params (different obs_spec)
            lidar_spec = _OBS_SPEC.with_lidar(4)
            n_params2 = lidar_spec.dim * _N_HEADS
            policy2 = CMAESPolicy(lidar_spec, _tmnf_factory, n_params2, population_size=6)
            with self.assertRaises(ValueError):
                policy2.load_trainer_state(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
