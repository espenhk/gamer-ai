"""Tests for REINFORCEPolicy in framework/reinforce.py."""
import unittest

import numpy as np

from framework.reinforce import REINFORCEPolicy
from games.tmnf.actions import DISCRETE_ACTIONS
from games.tmnf.obs_spec import BASE_OBS_DIM, TMNF_OBS_SPEC

_OBS_DIM   = BASE_OBS_DIM
_OBS_SPEC  = TMNF_OBS_SPEC
_N_ACTIONS = len(DISCRETE_ACTIONS)
_ACTION_DEC = lambda i: DISCRETE_ACTIONS[i]  # noqa: E731


def _make(**kw) -> REINFORCEPolicy:
    """Create a TMNF-flavoured REINFORCEPolicy via the framework API."""
    defaults = dict(output_dim=_N_ACTIONS)
    defaults.update(kw)
    return REINFORCEPolicy(_OBS_SPEC, _ACTION_DEC, **defaults)


def _zero_obs(n_lidar_rays: int = 0) -> np.ndarray:
    return np.zeros(_OBS_DIM + n_lidar_rays, dtype=np.float32)


def _rand_obs(n_lidar_rays: int = 0, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(
        _OBS_DIM + n_lidar_rays).astype(np.float32)


class TestREINFORCEPolicyStructure(unittest.TestCase):

    def setUp(self):
        self.policy = _make(hidden_sizes=[8, 8], seed=0)

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
            self.assertIn(float(action[1]), (0.0, 0.5, 1.0))

    def test_brake_binary(self):
        for _ in range(20):
            action = self.policy(_zero_obs())
            self.assertIn(float(action[2]), (0.0, 0.5, 1.0))

    def test_action_is_discrete(self):
        """Returned actions must come from DISCRETE_ACTIONS."""
        for _ in range(30):
            action = self.policy(_zero_obs())
            match = any(np.allclose(action, da) for da in DISCRETE_ACTIONS)
            self.assertTrue(match, f"action {action} not in DISCRETE_ACTIONS")

    def test_episode_buffers_fill_on_call(self):
        obs = _zero_obs()
        for _ in range(5):
            self.policy(obs)
            self.policy.update(obs, np.array([0, 1, 0]), 1.0, obs, False)
        self.assertEqual(len(self.policy._ep_grads),   5)
        self.assertEqual(len(self.policy._ep_rewards), 5)

    def test_episode_buffers_clear_on_episode_end(self):
        obs = _zero_obs()
        self.policy(obs)
        self.policy.update(obs, np.array([0, 1, 0]), 1.0, obs, False)
        self.policy.on_episode_end()
        self.assertEqual(len(self.policy._ep_grads),   0)
        self.assertEqual(len(self.policy._ep_rewards), 0)

    def test_on_episode_end_with_empty_buffer_is_noop(self):
        self.policy.on_episode_end()

    def test_weights_match_hidden_sizes(self):
        p    = _make(hidden_sizes=[32, 16])
        dims = [_OBS_DIM, 32, 16, _N_ACTIONS]
        for i, (w, b) in enumerate(zip(p._weights, p._biases)):
            self.assertEqual(w.shape, (dims[i + 1], dims[i]))
            self.assertEqual(b.shape, (dims[i + 1],))


class TestREINFORCEDiscountedReturns(unittest.TestCase):

    def _run_synthetic(self, rewards):
        policy = _make(hidden_sizes=[8], learning_rate=0.1,
                       gamma=0.9, entropy_coeff=0.0, seed=42)
        obs = _zero_obs()
        for _ in range(len(rewards)):
            policy(obs)
        for r in rewards:
            policy.update(obs, np.array([0, 1, 0]), r, obs, False)
        return policy

    def test_buffer_lengths_match(self):
        rewards = [1.0, 2.0, 3.0]
        policy  = self._run_synthetic(rewards)
        self.assertEqual(len(policy._ep_grads),   len(rewards))
        self.assertEqual(len(policy._ep_rewards), len(rewards))

    def test_weights_change_after_update(self):
        policy   = _make(hidden_sizes=[8], learning_rate=0.5,
                         gamma=0.99, entropy_coeff=0.0, seed=1)
        w_before = [w.copy() for w in policy._weights]
        obs      = _rand_obs(seed=42)
        policy(obs)
        policy.update(obs, np.array([0, 1, 0]), 10.0, obs, True)
        policy.on_episode_end()
        for i, (wb, wa) in enumerate(zip(w_before, policy._weights)):
            self.assertFalse(np.allclose(wb, wa),
                             f"weights[{i}] unchanged after gradient step")

    def test_gradient_direction(self):
        """
        After many episodes where action 7 always gets high reward, the
        log-probability of action 7 should increase relative to action 0.
        """
        np.random.seed(123)
        policy  = _make(hidden_sizes=[16], learning_rate=0.05,
                        gamma=1.0, entropy_coeff=0.0,
                        baseline="none", seed=0)
        obs     = _rand_obs(seed=7)
        target  = 7

        def _get_logit_diff():
            probs, _, _ = policy._forward(obs / policy._scales)
            return float(probs[target]) - float(probs[0])

        diff_before = _get_logit_diff()

        for _ in range(200):
            policy(obs)
            l_in, pre_r, probs_ep, _ = policy._ep_grads[-1]
            policy._ep_grads[-1]     = (l_in, pre_r, probs_ep, target)
            policy.update(obs, DISCRETE_ACTIONS[target], 10.0, obs, True)
            policy.on_episode_end()

        diff_after = _get_logit_diff()
        self.assertGreater(diff_after, diff_before,
                           "REINFORCE did not increase probability of rewarded action")


class TestREINFORCEEntropyCoeff(unittest.TestCase):

    def test_zero_entropy_coeff_no_entropy_term(self):
        policy = _make(hidden_sizes=[8], entropy_coeff=0.0, seed=0)
        w_ref  = [w.copy() for w in policy._weights]
        obs    = _rand_obs(seed=1)
        policy(obs)
        policy.update(obs, np.array([0, 1, 0]), 1.0, obs, True)
        policy.on_episode_end()
        changed = any(not np.allclose(w_ref[i], policy._weights[i])
                      for i in range(len(policy._weights)))
        self.assertTrue(changed)

    def test_entropy_coeff_changes_gradient(self):
        obs = _rand_obs(seed=5)

        def _weights_after(entropy_coeff):
            p = _make(hidden_sizes=[8], learning_rate=0.5,
                      gamma=1.0, entropy_coeff=entropy_coeff, seed=77)
            p(obs)
            p.update(obs, np.array([0, 1, 0]), 1.0, obs, True)
            p.on_episode_end()
            return np.concatenate([w.ravel() for w in p._weights])

        w_no_ent  = _weights_after(0.0)
        w_with_ent = _weights_after(1.0)
        self.assertFalse(np.allclose(w_no_ent, w_with_ent))


class TestREINFORCECfgRoundtrip(unittest.TestCase):

    def test_to_cfg_contains_required_keys(self):
        policy = _make(hidden_sizes=[16, 8])
        cfg    = policy.to_cfg()
        for key in ("policy_type", "hidden_sizes", "learning_rate",
                    "gamma", "entropy_coeff", "baseline",
                    "output_dim", "weights", "biases"):
            self.assertIn(key, cfg)

    def test_policy_type_string(self):
        policy = _make()
        self.assertEqual(policy.to_cfg()["policy_type"], "reinforce")

    def test_from_cfg_restores_weights(self):
        policy = _make(hidden_sizes=[8, 4], seed=3)
        cfg    = policy.to_cfg()
        loaded = REINFORCEPolicy.from_cfg(cfg, _OBS_SPEC, _ACTION_DEC)
        for w1, w2 in zip(policy._weights, loaded._weights):
            np.testing.assert_array_equal(w1, w2)
        for b1, b2 in zip(policy._biases, loaded._biases):
            np.testing.assert_array_equal(b1, b2)

    def test_from_cfg_restores_hyperparams(self):
        policy = _make(hidden_sizes=[12], learning_rate=0.005,
                       gamma=0.95, entropy_coeff=0.02,
                       baseline="none")
        cfg    = policy.to_cfg()
        loaded = REINFORCEPolicy.from_cfg(cfg, _OBS_SPEC, _ACTION_DEC)
        self.assertAlmostEqual(loaded._lr,            0.005)
        self.assertAlmostEqual(loaded._gamma,         0.95)
        self.assertAlmostEqual(loaded._entropy_coeff, 0.02)
        self.assertEqual(loaded._baseline_type,       "none")

    def test_save_and_reload(self):
        import tempfile, os, yaml
        policy = _make(hidden_sizes=[8], seed=9)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            policy.save(path)
            with open(path) as f:
                cfg = yaml.safe_load(f)
            loaded = REINFORCEPolicy.from_cfg(cfg, _OBS_SPEC, _ACTION_DEC)
            for w1, w2 in zip(policy._weights, loaded._weights):
                np.testing.assert_array_almost_equal(w1, w2, decimal=5)
        finally:
            os.unlink(path)

    def test_from_cfg_with_lidar(self):
        lidar_spec = _OBS_SPEC.with_lidar(4)
        policy = REINFORCEPolicy(
            lidar_spec, _ACTION_DEC, output_dim=_N_ACTIONS,
            hidden_sizes=[8], seed=2,
        )
        cfg    = policy.to_cfg()
        loaded = REINFORCEPolicy.from_cfg(cfg, lidar_spec, _ACTION_DEC)
        self.assertEqual(loaded._obs_dim, BASE_OBS_DIM + 4)
        obs    = np.zeros(BASE_OBS_DIM + 4, dtype=np.float32)
        action = loaded(obs)
        self.assertEqual(action.shape, (3,))


class TestREINFORCETrainerState(unittest.TestCase):

    def test_save_load_roundtrip_baseline(self):
        """save_trainer_state → load_trainer_state preserves baseline_val."""
        import tempfile, os
        policy = _make(hidden_sizes=[8], seed=0)
        obs = _rand_obs(seed=1)
        for _ in range(3):
            policy(obs)
            policy.update(obs, np.array([0, 1, 0]), 5.0, obs, True)
            policy.on_episode_end()

        self.assertNotEqual(policy._baseline_val, 0.0,
                            "Expected baseline_val to change after training episodes")

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            policy.save_trainer_state(path)

            policy2 = _make(hidden_sizes=[8], seed=99)
            policy2.load_trainer_state(path)

            self.assertAlmostEqual(policy._baseline_val, policy2._baseline_val, places=10)
        finally:
            os.unlink(path)

    def test_load_wrong_obs_dim_raises(self):
        """Loading state with mismatched obs_dim raises ValueError."""
        import tempfile, os
        policy1 = _make()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            policy1.save_trainer_state(path)
            lidar_spec = _OBS_SPEC.with_lidar(4)
            policy2 = REINFORCEPolicy(
                lidar_spec, _ACTION_DEC, output_dim=_N_ACTIONS,
            )
            with self.assertRaises(ValueError):
                policy2.load_trainer_state(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
