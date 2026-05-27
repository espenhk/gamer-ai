"""Tests for PPOPolicy in framework/ppo.py."""

import os
import tempfile
import unittest

import numpy as np
import yaml

from framework.policies import POLICY_REGISTRY
from framework.ppo import PPOPolicy
from games.tmnf.actions import DISCRETE_ACTIONS
from games.tmnf.obs_spec import BASE_OBS_DIM, TMNF_OBS_SPEC

_OBS_DIM = BASE_OBS_DIM
_OBS_SPEC = TMNF_OBS_SPEC
_N_ACTIONS = len(DISCRETE_ACTIONS)
_ACTION_DEC = lambda i: DISCRETE_ACTIONS[i]  # noqa: E731


def _make(**kw) -> PPOPolicy:
    defaults = dict(output_dim=_N_ACTIONS, hidden_sizes=[8, 8], seed=0)
    defaults.update(kw)
    return PPOPolicy(_OBS_SPEC, _ACTION_DEC, **defaults)


def _zero_obs(n_lidar_rays: int = 0) -> np.ndarray:
    return np.zeros(_OBS_DIM + n_lidar_rays, dtype=np.float32)


def _rand_obs(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(_OBS_DIM).astype(np.float32)


def _forced_episode(policy: PPOPolicy, obs: np.ndarray, actions: list[int], rewards: list[float]) -> None:
    """Run one episode, forcing the chosen action index at each step."""
    policy.on_episode_start()
    obs_norm = (obs / policy._scales).astype(np.float32)
    for t, (a, r) in enumerate(zip(actions, rewards)):
        policy(obs)  # populate obs/value buffers and advance RNG
        probs, _, _ = policy._actor_probs(obs_norm, None)
        policy._act_buf[-1] = a
        policy._logp_buf[-1] = float(np.log(probs[a] + 1e-8))
        done = t == len(actions) - 1
        policy.update(obs, DISCRETE_ACTIONS[a], r, obs, done)
    policy.on_episode_end()


class TestPPORegistration(unittest.TestCase):
    def test_registered_framework_wide(self):
        # Importing framework.policies alone must expose "ppo" (it side-effect
        # imports framework.ppo at module bottom).
        self.assertIn("ppo", POLICY_REGISTRY)
        self.assertIs(POLICY_REGISTRY["ppo"], PPOPolicy)

    def test_sc2_incompatible(self):
        import games.sc2.adapter  # noqa: F401 — registers "sc2" action-encoding incompatibility

        ok, hint = PPOPolicy.compatible_with("sc2")
        self.assertFalse(ok)
        self.assertIsInstance(hint, str)
        self.assertIn("sc2_reinforce", hint)

    def test_compatible_with_racing(self):
        ok, hint = PPOPolicy.compatible_with("car_racing")
        self.assertTrue(ok)
        self.assertIsNone(hint)


class TestPPOStructure(unittest.TestCase):
    def setUp(self):
        self.policy = _make()

    def test_action_shape(self):
        self.assertEqual(self.policy(_zero_obs()).shape, (3,))

    def test_action_is_discrete(self):
        for _ in range(30):
            action = self.policy(_zero_obs())
            self.assertTrue(any(np.allclose(action, da) for da in DISCRETE_ACTIONS))

    def test_buffers_fill_on_call_and_update(self):
        obs = _zero_obs()
        for _ in range(5):
            self.policy(obs)
            self.policy.update(obs, np.array([0, 1, 0]), 1.0, obs, False)
        self.assertEqual(len(self.policy._obs_buf), 5)
        self.assertEqual(len(self.policy._rew_buf), 5)
        self.assertEqual(len(self.policy._val_buf), 5)

    def test_buffers_clear_on_episode_end(self):
        obs = _zero_obs()
        self.policy(obs)
        self.policy.update(obs, np.array([0, 1, 0]), 1.0, obs, True)
        self.policy.on_episode_end()
        self.assertEqual(len(self.policy._obs_buf), 0)
        self.assertEqual(len(self.policy._rew_buf), 0)

    def test_empty_episode_end_is_noop(self):
        self.policy.on_episode_end()  # must not raise

    def test_actor_critic_shapes(self):
        p = _make(hidden_sizes=[32, 16])
        actor_dims = [_OBS_DIM, 32, 16, _N_ACTIONS]
        for i, w in enumerate(p._actor_w):
            self.assertEqual(w.shape, (actor_dims[i + 1], actor_dims[i]))
        critic_dims = [_OBS_DIM, 32, 16, 1]
        for i, w in enumerate(p._critic_w):
            self.assertEqual(w.shape, (critic_dims[i + 1], critic_dims[i]))


class TestPPOGae(unittest.TestCase):
    def test_gae_matches_hand_computation(self):
        policy = _make(gamma=0.9, gae_lambda=0.5)
        values = np.array([1.0, 2.0, 3.0])
        rewards = np.array([1.0, 1.0, 1.0])
        adv, ret = policy._compute_gae(values, rewards)
        # delta2 = 1 + 0 - 3 = -2 ; A2 = -2
        # delta1 = 1 + 0.9*3 - 2 = 1.7 ; A1 = 1.7 + 0.45*(-2) = 0.8
        # delta0 = 1 + 0.9*2 - 1 = 1.8 ; A0 = 1.8 + 0.45*0.8 = 2.16
        np.testing.assert_allclose(adv, [2.16, 0.8, -2.0], rtol=1e-6)
        np.testing.assert_allclose(ret, [3.16, 2.8, 1.0], rtol=1e-6)

    def test_gae_lambda_one_is_monte_carlo(self):
        policy = _make(gamma=1.0, gae_lambda=1.0)
        values = np.zeros(3)
        rewards = np.array([1.0, 2.0, 3.0])
        adv, ret = policy._compute_gae(values, rewards)
        # With V=0 and lambda=gamma=1, advantages == returns == reverse-cumsum.
        np.testing.assert_allclose(ret, [6.0, 5.0, 3.0], rtol=1e-6)
        np.testing.assert_allclose(adv, ret, rtol=1e-6)


class TestPPOClipping(unittest.TestCase):
    def _rollout(self):
        # Length-4 rollout with non-degenerate advantages.
        return [3, 7, 1, 5], [1.0, -1.0, 2.0, 0.0]

    def test_clip_inactive_on_single_epoch(self):
        """On the first epoch ratio == 1 for all samples, so clip_range is inert."""
        obs = _rand_obs(1)
        acts, rews = self._rollout()
        p_small = _make(n_epochs=1, clip_range=0.01, learning_rate=0.05, seed=3)
        p_large = _make(n_epochs=1, clip_range=10.0, learning_rate=0.05, seed=3)
        _forced_episode(p_small, obs, acts, rews)
        _forced_episode(p_large, obs, acts, rews)
        for ws, wl in zip(p_small._actor_w, p_large._actor_w):
            np.testing.assert_allclose(ws, wl, rtol=1e-6, atol=1e-7)

    def test_clip_throttles_multi_epoch_drift(self):
        """Over many epochs a tiny clip range limits how far weights drift."""
        obs = _rand_obs(2)
        acts, rews = self._rollout()
        base = _make(seed=5)
        ref = [w.copy() for w in base._actor_w]

        def _drift(clip):
            p = _make(n_epochs=30, clip_range=clip, learning_rate=0.05, seed=5)
            _forced_episode(p, obs, acts, rews)
            return float(np.sqrt(sum(float(((w - r) ** 2).sum()) for w, r in zip(p._actor_w, ref))))

        self.assertLessEqual(_drift(1e-4), _drift(0.5) + 1e-9)


class TestPPOLearning(unittest.TestCase):
    def test_increases_prob_of_rewarded_action(self):
        policy = _make(hidden_sizes=[16], learning_rate=0.02, gamma=0.99, entropy_coeff=0.0, seed=0)
        obs = _rand_obs(7)
        obs_norm = (obs / policy._scales).astype(np.float32)
        target, other = 7, 0

        def _prob_diff():
            probs, _, _ = policy._actor_probs(obs_norm, None)
            return float(probs[target]) - float(probs[other])

        before = _prob_diff()
        for _ in range(150):
            _forced_episode(policy, obs, [target, other], [10.0, -1.0])
        after = _prob_diff()
        self.assertGreater(after, before, "PPO did not raise probability of the rewarded action")


class TestPPOCfgRoundtrip(unittest.TestCase):
    def test_to_cfg_keys(self):
        cfg = _make(hidden_sizes=[16, 8]).to_cfg()
        for key in (
            "policy_type",
            "hidden_sizes",
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_range",
            "n_epochs",
            "entropy_coeff",
            "value_coeff",
            "minibatch_size",
            "output_dim",
            "actor_weights",
            "actor_biases",
            "critic_weights",
            "critic_biases",
        ):
            self.assertIn(key, cfg)
        self.assertEqual(cfg["policy_type"], "ppo")

    def test_from_cfg_restores_weights_and_hyperparams(self):
        policy = _make(hidden_sizes=[8, 4], learning_rate=1e-3, clip_range=0.15, n_epochs=6, seed=3)
        loaded = PPOPolicy.from_cfg(policy.to_cfg(), _OBS_SPEC, _ACTION_DEC)
        for w1, w2 in zip(policy._actor_w, loaded._actor_w):
            np.testing.assert_array_equal(w1, w2)
        for w1, w2 in zip(policy._critic_w, loaded._critic_w):
            np.testing.assert_array_equal(w1, w2)
        self.assertAlmostEqual(loaded._lr, 1e-3)
        self.assertAlmostEqual(loaded._clip, 0.15)
        self.assertEqual(loaded._n_epochs, 6)

    def test_save_and_reload_yaml(self):
        policy = _make(hidden_sizes=[8], seed=9)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            policy.save(path)
            with open(path) as f:
                cfg = yaml.safe_load(f)
            loaded = PPOPolicy.from_cfg(cfg, _OBS_SPEC, _ACTION_DEC)
            for w1, w2 in zip(policy._actor_w, loaded._actor_w):
                np.testing.assert_array_almost_equal(w1, w2, decimal=5)
        finally:
            os.unlink(path)

    def test_from_cfg_obs_dim_mismatch_raises(self):
        cfg = _make().to_cfg()
        with self.assertRaises(ValueError):
            PPOPolicy.from_cfg(cfg, _OBS_SPEC.with_lidar(5), _ACTION_DEC)


class TestPPOTrainerState(unittest.TestCase):
    def test_adam_moment_roundtrip(self):
        policy = _make(hidden_sizes=[8], seed=0)
        _forced_episode(policy, _rand_obs(1), [3, 7], [5.0, -1.0])
        self.assertGreater(policy._adam_t, 0)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            policy.save_trainer_state(path)
            policy2 = _make(hidden_sizes=[8], seed=99)
            policy2.load_trainer_state(path)
            self.assertEqual(policy._adam_t, policy2._adam_t)
            for a, b in zip(policy._am_w, policy2._am_w):
                np.testing.assert_array_equal(a, b)
        finally:
            os.unlink(path)

    def test_load_wrong_obs_dim_raises(self):
        policy = _make()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            policy.save_trainer_state(path)
            policy2 = PPOPolicy(_OBS_SPEC.with_lidar(4), _ACTION_DEC, output_dim=_N_ACTIONS)
            with self.assertRaises(ValueError):
                policy2.load_trainer_state(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
