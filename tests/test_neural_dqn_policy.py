"""Tests for NeuralDQNPolicy and ReplayBuffer in policies.py."""
import unittest

import numpy as np

from games.tmnf.obs_spec import BASE_OBS_DIM
from games.tmnf.policies import (
    NeuralDQNPolicy,
    ReplayBuffer,
    _DISCRETE_ACTIONS,
    _action_to_idx,
)

_N = BASE_OBS_DIM


def _zero_obs() -> np.ndarray:
    return np.zeros(_N, dtype=np.float32)


def _rand_obs() -> np.ndarray:
    return np.random.randn(_N).astype(np.float32)


# ---------------------------------------------------------------------------
# ReplayBuffer tests
# ---------------------------------------------------------------------------

class TestReplayBuffer(unittest.TestCase):

    def test_push_and_len(self):
        buf = ReplayBuffer(maxlen=10)
        self.assertEqual(len(buf), 0)
        buf.push(_zero_obs(), 0, 1.0, _zero_obs(), False)
        self.assertEqual(len(buf), 1)

    def test_circular_eviction(self):
        buf = ReplayBuffer(maxlen=3)
        for i in range(5):
            buf.push(_zero_obs(), i % 9, float(i), _zero_obs(), False)
        self.assertEqual(len(buf), 3)

    def test_sample_shapes(self):
        buf = ReplayBuffer(maxlen=20)
        for _ in range(10):
            buf.push(_rand_obs(), 0, 0.0, _rand_obs(), False)
        obs_b, act_b, rew_b, next_b, done_b = buf.sample(5)
        self.assertEqual(obs_b.shape,  (5, _N))
        self.assertEqual(act_b.shape,  (5,))
        self.assertEqual(rew_b.shape,  (5,))
        self.assertEqual(next_b.shape, (5, _N))
        self.assertEqual(done_b.shape, (5,))
        self.assertEqual(obs_b.dtype,  np.float32)

    def test_sample_without_replacement(self):
        """sample() should never return duplicate entries when buf is large enough."""
        buf = ReplayBuffer(maxlen=100)
        for i in range(20):
            buf.push(_rand_obs(), i % 9, float(i), _rand_obs(), False)
        _, act_b, rew_b, _, _ = buf.sample(10)
        # rewards encode unique step indices — duplicates would repeat a reward
        self.assertEqual(len(set(rew_b.tolist())), 10)

    def test_sample_with_replacement_when_small(self):
        """sample() falls back to replace=True when batch_size > buffer length."""
        buf = ReplayBuffer(maxlen=100)
        for i in range(3):
            buf.push(_rand_obs(), i % 9, float(i), _rand_obs(), False)
        # Requesting more samples than buffer size should not raise
        obs_b, act_b, rew_b, next_b, done_b = buf.sample(10)
        self.assertEqual(obs_b.shape[0], 10)


# ---------------------------------------------------------------------------
# NeuralDQNPolicy structural tests
# ---------------------------------------------------------------------------

class TestNeuralDQNPolicyStructure(unittest.TestCase):

    def _make(self, **kw) -> NeuralDQNPolicy:
        defaults = dict(
            hidden_sizes=[8, 8],
            replay_buffer_size=200,
            batch_size=16,
            min_replay_size=32,
            target_update_freq=10,
            learning_rate=0.01,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=100,
            gamma=0.99,
            n_lidar_rays=0,
        )
        defaults.update(kw)
        return NeuralDQNPolicy(**defaults)

    def test_action_shape_and_range(self):
        p   = self._make(epsilon_start=0.0)
        act = p(_zero_obs())
        self.assertEqual(act.shape, (3,))
        self.assertGreaterEqual(float(act[0]), -1.0)
        self.assertLessEqual(float(act[0]), 1.0)
        self.assertIn(float(act[1]), {0.0, 0.5, 1.0})
        self.assertIn(float(act[2]), {0.0, 0.5, 1.0})

    def test_greedy_action_is_discrete(self):
        """Greedy action must be one of the 25 discrete actions."""
        p   = self._make(epsilon_start=0.0)
        act = p(_zero_obs())
        idx = _action_to_idx(act)
        self.assertIn(idx, range(25))

    def test_random_action_when_epsilon_one(self):
        """With epsilon=1, every call should still return a valid action."""
        p = self._make(epsilon_start=1.0, epsilon_end=1.0)
        for _ in range(20):
            act = p(_zero_obs())
            self.assertIn(_action_to_idx(act), range(25))

    def test_replay_buffer_fills_on_update(self):
        p = self._make()
        for _ in range(10):
            p.update(_zero_obs(), 0, 1.0, _zero_obs(), True)
        self.assertEqual(len(p._replay), 10)

    def test_epsilon_decays_on_update(self):
        p = self._make(epsilon_start=1.0, epsilon_end=0.0, epsilon_decay_steps=10)
        for _ in range(10):
            p.update(_zero_obs(), 0, 0.0, _zero_obs(), True)
        self.assertLess(p._eps, 1.0)

    def test_epsilon_floored_at_epsilon_end(self):
        p = self._make(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=1)
        for _ in range(50):
            p.update(_zero_obs(), 0, 0.0, _zero_obs(), True)
        self.assertGreaterEqual(p._eps, 0.1)

    def test_target_sync_happens(self):
        """Target network weights should match online after a sync."""
        p = self._make(target_update_freq=5, min_replay_size=16)
        # Fill buffer and trigger enough gradient steps to force a sync
        for _ in range(50):
            p.update(_rand_obs(), np.random.randint(25), 0.0, _rand_obs(), False)
        # After sync, target matches online
        for w_on, w_tgt in zip(p._online["weights"], p._target["weights"]):
            np.testing.assert_array_equal(w_on, w_tgt)

    def test_network_weight_shapes(self):
        p = self._make(hidden_sizes=[32, 16])
        # Layer dims: [obs_dim, 32, 16, 25]
        self.assertEqual(p._online["weights"][0].shape, (32, _N))
        self.assertEqual(p._online["weights"][1].shape, (16, 32))
        self.assertEqual(p._online["weights"][2].shape, (25, 16))

    def test_to_cfg_roundtrip(self):
        p   = self._make(epsilon_start=0.0)
        cfg = p.to_cfg()
        p2  = NeuralDQNPolicy.from_cfg(cfg)
        obs = _rand_obs()
        # Greedy actions should match after round-trip
        p._eps  = 0.0
        p2._eps = 0.0
        np.testing.assert_array_equal(p(obs), p2(obs))

    def test_cfg_policy_type_key(self):
        p = self._make()
        self.assertEqual(p.to_cfg()["policy_type"], "neural_dqn")

    def test_on_episode_end_no_crash(self):
        p = self._make()
        p.on_episode_end()   # should not raise

    def test_from_cfg_missing_keys_raises(self):
        """from_cfg() should raise KeyError if weight keys are incomplete."""
        cfg = {"online_weights": [[[1.0]]]}  # missing other required keys
        with self.assertRaises(KeyError):
            NeuralDQNPolicy.from_cfg(cfg)

    def test_from_cfg_shape_mismatch_raises(self):
        """from_cfg() should raise ValueError when obs_dim doesn't match weights."""
        p   = self._make()
        cfg = p.to_cfg()
        # Pass a different n_lidar_rays to cause shape mismatch
        with self.assertRaises(ValueError):
            NeuralDQNPolicy.from_cfg(cfg, n_lidar_rays=5)


# ---------------------------------------------------------------------------
# NeuralDQNPolicy convergence test (2-state bandit MDP)
# ---------------------------------------------------------------------------

class TestNeuralDQNConvergence(unittest.TestCase):
    """
    Single-observation bandit: one action consistently gives +1 reward,
    all others give -0.1.  With gamma=0 the optimal Q-values reduce to
    expected reward per action.  After sufficient training the greedy
    policy must select the rewarding action.
    """

    def test_bandit_convergence(self):
        np.random.seed(42)
        N = BASE_OBS_DIM
        state   = np.zeros(N, dtype=np.float32)
        BEST    = 22         # full accel + straight
        GOOD_R  =  1.0
        BAD_R   = -0.1

        policy = NeuralDQNPolicy(
            hidden_sizes        = [32, 32],
            replay_buffer_size  = 5000,
            batch_size          = 32,
            min_replay_size     = 128,
            target_update_freq  = 25,
            learning_rate       = 0.005,
            epsilon_start       = 1.0,
            epsilon_end         = 1.0,   # keep at 1; we push transitions directly
            epsilon_decay_steps = 1,
            gamma               = 0.0,   # pure bandit — no bootstrapping
            n_lidar_rays        = 0,
        )

        next_obs = np.zeros(N, dtype=np.float32)
        # Cycle through all 25 actions so each is equally represented
        for step in range(12500):
            action_idx = step % 25
            r = GOOD_R if action_idx == BEST else BAD_R
            policy.update(state, action_idx, r, next_obs, done=True)

        # Evaluate greedy
        policy._eps = 0.0
        obs_norm    = (state / policy._scales).astype(np.float32)
        q_vals      = policy._q_values(policy._online, obs_norm)
        greedy_idx  = int(np.argmax(q_vals))

        self.assertEqual(
            greedy_idx, BEST,
            f"Expected greedy action {BEST}, got {greedy_idx}. Q-values: {q_vals.tolist()}"
        )
        # The best action should also have a clearly higher Q-value
        self.assertGreater(float(q_vals[BEST]), float(np.max(np.delete(q_vals, BEST))))


class TestNeuralDQNTrainerState(unittest.TestCase):

    def _make_trained_policy(self) -> NeuralDQNPolicy:
        policy = NeuralDQNPolicy(
            hidden_sizes        = [16, 16],
            replay_buffer_size  = 200,
            batch_size          = 16,
            min_replay_size     = 32,
            target_update_freq  = 10,
            learning_rate       = 0.001,
            epsilon_start       = 1.0,
            epsilon_end         = 0.05,
            epsilon_decay_steps = 100,
            n_lidar_rays        = 0,
        )
        obs  = np.zeros(_N, dtype=np.float32)
        next_obs = np.zeros(_N, dtype=np.float32)
        for i in range(50):
            policy.update(obs, i % 9, float(i % 3), next_obs, done=False)
        return policy

    def test_save_load_roundtrip_replay_buffer(self):
        """Replay buffer length and transition contents survive a save/load cycle."""
        import tempfile, os
        policy = self._make_trained_policy()
        original_buf_len = len(policy._replay)

        # Capture original buffer contents for comparison
        original_buf = list(policy._replay._buf)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            policy.save_trainer_state(path)

            policy2 = NeuralDQNPolicy(
                hidden_sizes        = [16, 16],
                replay_buffer_size  = 200,
                min_replay_size     = 32,
                n_lidar_rays        = 0,
            )
            policy2.load_trainer_state(path)

            self.assertEqual(len(policy2._replay), original_buf_len)

            # Verify actual transition contents match
            restored_buf = list(policy2._replay._buf)
            for orig, rest in zip(original_buf, restored_buf):
                np.testing.assert_array_equal(orig[0], rest[0])   # obs
                self.assertEqual(orig[1], rest[1])                 # action_idx
                self.assertAlmostEqual(orig[2], rest[2])           # reward
                np.testing.assert_array_equal(orig[3], rest[3])   # next_obs
                self.assertEqual(orig[4], rest[4])                 # done
        finally:
            os.unlink(path)

    def test_save_load_roundtrip_adam_moments(self):
        """Adam moments survive a save/load cycle."""
        import tempfile, os
        policy = self._make_trained_policy()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            policy.save_trainer_state(path)

            policy2 = NeuralDQNPolicy(
                hidden_sizes        = [16, 16],
                replay_buffer_size  = 200,
                min_replay_size     = 32,
                n_lidar_rays        = 0,
            )
            policy2.load_trainer_state(path)

            for i in range(len(policy._m_w)):
                np.testing.assert_array_equal(policy._m_w[i], policy2._m_w[i])
                np.testing.assert_array_equal(policy._v_w[i], policy2._v_w[i])
            self.assertEqual(policy._adam_t,      policy2._adam_t)
            self.assertEqual(policy._total_steps, policy2._total_steps)
            self.assertAlmostEqual(policy._eps,   policy2._eps)
        finally:
            os.unlink(path)

    def test_load_wrong_obs_dim_raises(self):
        """Loading state with mismatched obs_dim raises ValueError."""
        import tempfile, os
        policy1 = NeuralDQNPolicy(hidden_sizes=[16], replay_buffer_size=10, n_lidar_rays=0)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            policy1.save_trainer_state(path)
            policy2 = NeuralDQNPolicy(hidden_sizes=[16], replay_buffer_size=10, n_lidar_rays=4)
            with self.assertRaises(ValueError):
                policy2.load_trainer_state(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
