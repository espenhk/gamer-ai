"""Tests for DQNPolicy and ReplayBuffer in framework/dqn.py and framework/replay.py."""

import unittest

import numpy as np

from framework.dqn import DQNPolicy
from framework.replay import ReplayBuffer
from games.tmnf.actions import DISCRETE_ACTIONS, _action_to_idx
from games.tmnf.obs_spec import BASE_OBS_DIM, TMNF_OBS_SPEC

_N = BASE_OBS_DIM

# Alias for test code that referenced the old class name
NeuralDQNPolicy = DQNPolicy

_OBS_SPEC = TMNF_OBS_SPEC


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
        self.assertEqual(obs_b.shape, (5, _N))
        self.assertEqual(act_b.shape, (5,))
        self.assertEqual(rew_b.shape, (5,))
        self.assertEqual(next_b.shape, (5, _N))
        self.assertEqual(done_b.shape, (5,))
        self.assertEqual(obs_b.dtype, np.float32)

    def test_sample_without_replacement(self):
        """sample() should never return duplicate entries when buf is large enough."""
        buf = ReplayBuffer(maxlen=100)
        for i in range(20):
            buf.push(_rand_obs(), i % 9, float(i), _rand_obs(), False)
        _, act_b, rew_b, _, _ = buf.sample(10)
        self.assertEqual(len(set(rew_b.tolist())), 10)

    def test_sample_with_replacement_when_small(self):
        """sample() falls back to replace=True when batch_size > buffer length."""
        buf = ReplayBuffer(maxlen=100)
        for i in range(3):
            buf.push(_rand_obs(), i % 9, float(i), _rand_obs(), False)
        obs_b, act_b, rew_b, next_b, done_b = buf.sample(10)
        self.assertEqual(obs_b.shape[0], 10)


# ---------------------------------------------------------------------------
# DQNPolicy structural tests
# ---------------------------------------------------------------------------


class TestNeuralDQNPolicyStructure(unittest.TestCase):
    def _make(self, **kw) -> DQNPolicy:
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
        )
        defaults.update(kw)
        return DQNPolicy(_OBS_SPEC, DISCRETE_ACTIONS, **defaults)

    def test_action_shape_and_range(self):
        p = self._make(epsilon_start=0.0)
        act = p(_zero_obs())
        self.assertEqual(act.shape, (3,))
        self.assertGreaterEqual(float(act[0]), -1.0)
        self.assertLessEqual(float(act[0]), 1.0)
        self.assertIn(float(act[1]), {0.0, 0.5, 1.0})
        self.assertIn(float(act[2]), {0.0, 0.5, 1.0})

    def test_greedy_action_is_discrete(self):
        """Greedy action must be one of the 25 discrete actions."""
        p = self._make(epsilon_start=0.0)
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
        for _ in range(50):
            p.update(_rand_obs(), np.random.randint(25), 0.0, _rand_obs(), False)
        for w_on, w_tgt in zip(p._online["weights"], p._target["weights"]):
            np.testing.assert_array_equal(w_on, w_tgt)

    def test_network_weight_shapes(self):
        p = self._make(hidden_sizes=[32, 16])
        # Layer dims: [obs_dim, 32, 16, 25]
        self.assertEqual(p._online["weights"][0].shape, (32, _N))
        self.assertEqual(p._online["weights"][1].shape, (16, 32))
        self.assertEqual(p._online["weights"][2].shape, (25, 16))

    def test_to_cfg_roundtrip(self):
        p = self._make(epsilon_start=0.0)
        cfg = p.to_cfg()
        p2 = DQNPolicy.from_cfg(cfg, _OBS_SPEC, DISCRETE_ACTIONS)
        obs = _rand_obs()
        p._eps = 0.0
        p2._eps = 0.0
        np.testing.assert_array_equal(p(obs), p2(obs))

    def test_cfg_policy_type_key(self):
        p = self._make()
        self.assertEqual(p.to_cfg()["policy_type"], "dqn")

    def test_on_episode_end_no_crash(self):
        p = self._make()
        p.on_episode_end()

    def test_from_cfg_missing_keys_raises(self):
        """from_cfg() should raise KeyError if weight keys are incomplete."""
        cfg = {"online_weights": [[[1.0]]]}
        with self.assertRaises(KeyError):
            DQNPolicy.from_cfg(cfg, _OBS_SPEC, DISCRETE_ACTIONS)

    def test_from_cfg_shape_mismatch_raises(self):
        """from_cfg() should raise ValueError when obs_dim doesn't match weights."""
        p = self._make()
        cfg = p.to_cfg()
        # Use a different obs_spec to cause shape mismatch
        from games.tmnf.obs_spec import TMNF_OBS_SPEC

        wrong_spec = TMNF_OBS_SPEC.with_lidar(5)
        with self.assertRaises(ValueError):
            DQNPolicy.from_cfg(cfg, wrong_spec, DISCRETE_ACTIONS)


# ---------------------------------------------------------------------------
# DQNPolicy convergence test (2-state bandit MDP)
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
        state = np.zeros(N, dtype=np.float32)
        BEST = 22  # full accel + straight
        GOOD_R = 1.0
        BAD_R = -0.1

        policy = DQNPolicy(
            _OBS_SPEC,
            DISCRETE_ACTIONS,
            hidden_sizes=[32, 32],
            replay_buffer_size=5000,
            batch_size=32,
            min_replay_size=128,
            target_update_freq=25,
            learning_rate=0.005,
            epsilon_start=1.0,
            epsilon_end=1.0,
            epsilon_decay_steps=1,
            gamma=0.0,
        )

        next_obs = np.zeros(N, dtype=np.float32)
        for step in range(12500):
            action_idx = step % 25
            r = GOOD_R if action_idx == BEST else BAD_R
            policy.update(state, action_idx, r, next_obs, done=True)

        policy._eps = 0.0
        obs_norm = (state / policy._scales).astype(np.float32)
        q_vals = policy._q_values(policy._online, obs_norm)
        greedy_idx = int(np.argmax(q_vals))

        self.assertEqual(
            greedy_idx, BEST, f"Expected greedy action {BEST}, got {greedy_idx}. Q-values: {q_vals.tolist()}"
        )
        self.assertGreater(float(q_vals[BEST]), float(np.max(np.delete(q_vals, BEST))))


class TestNeuralDQNTrainerState(unittest.TestCase):
    def _make_trained_policy(self) -> DQNPolicy:
        policy = DQNPolicy(
            _OBS_SPEC,
            DISCRETE_ACTIONS,
            hidden_sizes=[16, 16],
            replay_buffer_size=200,
            batch_size=16,
            min_replay_size=32,
            target_update_freq=10,
            learning_rate=0.001,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=100,
        )
        obs = np.zeros(_N, dtype=np.float32)
        next_obs = np.zeros(_N, dtype=np.float32)
        for i in range(50):
            policy.update(obs, i % 9, float(i % 3), next_obs, done=False)
        return policy

    def test_save_load_roundtrip_replay_buffer(self):
        """Replay buffer length and transition contents survive a save/load cycle."""
        import os
        import tempfile

        policy = self._make_trained_policy()
        original_buf_len = len(policy._replay)
        original_buf = list(policy._replay._buf)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            policy.save_trainer_state(path)

            policy2 = DQNPolicy(
                _OBS_SPEC,
                DISCRETE_ACTIONS,
                hidden_sizes=[16, 16],
                replay_buffer_size=200,
                min_replay_size=32,
            )
            policy2.load_trainer_state(path)

            self.assertEqual(len(policy2._replay), original_buf_len)
            restored_buf = list(policy2._replay._buf)
            for orig, rest in zip(original_buf, restored_buf):
                np.testing.assert_array_equal(orig[0], rest[0])
                self.assertEqual(orig[1], rest[1])
                self.assertAlmostEqual(orig[2], rest[2])
                np.testing.assert_array_equal(orig[3], rest[3])
                self.assertEqual(orig[4], rest[4])
        finally:
            os.unlink(path)

    def test_save_load_roundtrip_adam_moments(self):
        """Adam moments survive a save/load cycle."""
        import os
        import tempfile

        policy = self._make_trained_policy()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            policy.save_trainer_state(path)

            policy2 = DQNPolicy(
                _OBS_SPEC,
                DISCRETE_ACTIONS,
                hidden_sizes=[16, 16],
                replay_buffer_size=200,
                min_replay_size=32,
            )
            policy2.load_trainer_state(path)

            for i in range(len(policy._m_w)):
                np.testing.assert_array_equal(policy._m_w[i], policy2._m_w[i])
                np.testing.assert_array_equal(policy._v_w[i], policy2._v_w[i])
            self.assertEqual(policy._adam_t, policy2._adam_t)
            self.assertEqual(policy._total_steps, policy2._total_steps)
            self.assertAlmostEqual(policy._eps, policy2._eps)
        finally:
            os.unlink(path)

    def test_load_wrong_obs_dim_raises(self):
        """Loading state with mismatched obs_dim raises ValueError."""
        import os
        import tempfile

        policy1 = DQNPolicy(_OBS_SPEC, DISCRETE_ACTIONS, hidden_sizes=[16], replay_buffer_size=10)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            policy1.save_trainer_state(path)
            wrong_spec = _OBS_SPEC.with_lidar(4)
            policy2 = DQNPolicy(wrong_spec, DISCRETE_ACTIONS, hidden_sizes=[16], replay_buffer_size=10)
            with self.assertRaises(ValueError):
                policy2.load_trainer_state(path)
        finally:
            os.unlink(path)


class TestDoubleDQN(unittest.TestCase):
    def _make(self, **kw) -> DQNPolicy:
        defaults = dict(hidden_sizes=[8], gamma=0.9, min_replay_size=10**9, seed=0)
        defaults.update(kw)
        return DQNPolicy(_OBS_SPEC, DISCRETE_ACTIONS, **defaults)

    def test_flag_defaults(self):
        # double_dqn + Huber + grad-clip default on (SB3-aligned); dueling opt-in.
        p = self._make()
        self.assertTrue(p._double)
        self.assertFalse(p._dueling)
        self.assertTrue(p._huber)
        self.assertEqual(p._max_grad_norm, 10.0)

    def test_double_target_uses_online_argmax_target_value(self):
        """Double DQN bootstrap = Q_target(next)[argmax_a Q_online(next)]."""
        p = self._make(double_dqn=True)
        # Make online and target genuinely different so argmax can disagree.
        p._target["weights"] = [w + 0.5 for w in p._online["weights"]]
        next_norm = (np.stack([_rand_obs(), _rand_obs()]) / p._scales).astype(np.float32)

        got = p._next_state_q(next_norm, None)

        q_online = p._q_values(p._online, next_norm)
        q_target = p._q_values(p._target, next_norm)
        a_star = np.argmax(q_online, axis=1)
        expected = q_target[np.arange(len(a_star)), a_star]
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_double_differs_from_vanilla_when_nets_disagree(self):
        next_norm = (np.stack([_rand_obs(), _rand_obs(), _rand_obs()]) / _OBS_SPEC.scales).astype(np.float32)
        vanilla = self._make(double_dqn=False)
        vanilla._target["weights"] = [w + 0.5 for w in vanilla._online["weights"]]
        van_q = vanilla._next_state_q(next_norm, None)

        dbl = self._make(double_dqn=True)
        dbl._online["weights"] = [w.copy() for w in vanilla._online["weights"]]
        dbl._online["biases"] = [b.copy() for b in vanilla._online["biases"]]
        dbl._target["weights"] = [w.copy() for w in vanilla._target["weights"]]
        dbl._target["biases"] = [b.copy() for b in vanilla._target["biases"]]
        dbl_q = dbl._next_state_q(next_norm, None)

        # Vanilla takes max over target Q (>= any single entry), so the double
        # estimate (target Q at the online-chosen action) cannot exceed it.
        self.assertTrue(np.all(dbl_q <= van_q + 1e-6))
        self.assertFalse(np.allclose(dbl_q, van_q))

    def test_double_to_cfg_roundtrip(self):
        p = self._make(double_dqn=True, epsilon_start=0.0)
        p2 = DQNPolicy.from_cfg(p.to_cfg(), _OBS_SPEC, DISCRETE_ACTIONS)
        self.assertTrue(p2._double)
        p._eps = p2._eps = 0.0
        obs = _rand_obs()
        np.testing.assert_array_equal(p(obs), p2(obs))

    def test_double_learns_bandit(self):
        np.random.seed(0)
        state = np.zeros(_N, dtype=np.float32)
        BEST = 22
        policy = DQNPolicy(
            _OBS_SPEC,
            DISCRETE_ACTIONS,
            double_dqn=True,
            hidden_sizes=[32, 32],
            replay_buffer_size=5000,
            batch_size=32,
            min_replay_size=128,
            target_update_freq=25,
            learning_rate=0.005,
            epsilon_start=1.0,
            epsilon_end=1.0,
            epsilon_decay_steps=1,
            gamma=0.0,
        )
        for step in range(12500):
            a = step % 25
            policy.update(state, a, 1.0 if a == BEST else -0.1, state, done=True)
        policy._eps = 0.0
        q = policy._q_values(policy._online, (state / policy._scales).astype(np.float32))
        self.assertEqual(int(np.argmax(q)), BEST)


class TestDuelingDQN(unittest.TestCase):
    def _make(self, **kw) -> DQNPolicy:
        defaults = dict(hidden_sizes=[8], dueling=True, seed=0)
        defaults.update(kw)
        return DQNPolicy(_OBS_SPEC, DISCRETE_ACTIONS, **defaults)

    def test_value_head_built(self):
        p = self._make(hidden_sizes=[16, 12])
        self.assertEqual(p._online["value_w"].shape, (1, 12))
        self.assertEqual(p._online["value_b"].shape, (1,))
        # advantage stream still produces n_actions, same as the vanilla output.
        self.assertEqual(p._online["weights"][-1].shape, (25, 12))

    def test_q_equals_value_plus_centered_advantage(self):
        p = self._make()
        obs = _rand_obs()
        obs_norm = (obs / p._scales).astype(np.float32)
        q = p._q_values(p._online, obs_norm)
        # Reconstruct A and V manually from the raw forward.
        adv, layer_inputs, _ = p._forward(p._online, obs_norm)
        last_hidden = layer_inputs[-1]
        v = float((last_hidden @ p._online["value_w"].T + p._online["value_b"]).ravel()[0])
        expected = v + (adv - adv.mean())
        np.testing.assert_allclose(q, expected, rtol=1e-6)

    def test_gradient_step_runs_and_updates_value_head(self):
        p = self._make(min_replay_size=4, batch_size=4)
        vw_before = p._online["value_w"].copy()
        for _ in range(20):
            p.update(_rand_obs(), np.random.randint(25), 1.0, _rand_obs(), False)
        self.assertFalse(np.allclose(vw_before, p._online["value_w"]))

    def test_dueling_to_cfg_roundtrip(self):
        p = self._make(epsilon_start=0.0)
        cfg = p.to_cfg()
        self.assertTrue(cfg["dueling"])
        self.assertIn("online_value_w", cfg)
        p2 = DQNPolicy.from_cfg(cfg, _OBS_SPEC, DISCRETE_ACTIONS)
        self.assertTrue(p2._dueling)
        p._eps = p2._eps = 0.0
        obs = _rand_obs()
        np.testing.assert_array_equal(p(obs), p2(obs))

    def test_dueling_trainer_state_preserves_value_head_moments(self):
        import os
        import tempfile

        p = self._make(min_replay_size=4, batch_size=4)
        for _ in range(20):
            p.update(_rand_obs(), np.random.randint(25), 1.0, _rand_obs(), False)
        self.assertFalse(np.allclose(p._m_vw, 0.0), "value-head Adam moment should be non-zero after training")

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            p.save_trainer_state(path)
            p2 = self._make(min_replay_size=4, batch_size=4)
            p2.load_trainer_state(path)
            np.testing.assert_array_equal(p._m_vw, p2._m_vw)
            np.testing.assert_array_equal(p._v_vw, p2._v_vw)
            np.testing.assert_array_equal(p._m_vb, p2._m_vb)
            np.testing.assert_array_equal(p._v_vb, p2._v_vb)
        finally:
            os.unlink(path)

    def test_loading_nondueling_state_into_dueling_raises(self):
        import os
        import tempfile

        vanilla = DQNPolicy(_OBS_SPEC, DISCRETE_ACTIONS, hidden_sizes=[8], replay_buffer_size=10)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            vanilla.save_trainer_state(path)
            dueling = self._make()
            with self.assertRaises(ValueError):
                dueling.load_trainer_state(path)
        finally:
            os.unlink(path)

    def test_dueling_learns_bandit(self):
        np.random.seed(0)
        state = np.zeros(_N, dtype=np.float32)
        BEST = 22
        policy = DQNPolicy(
            _OBS_SPEC,
            DISCRETE_ACTIONS,
            dueling=True,
            hidden_sizes=[32, 32],
            replay_buffer_size=5000,
            batch_size=32,
            min_replay_size=128,
            target_update_freq=25,
            learning_rate=0.005,
            epsilon_start=1.0,
            epsilon_end=1.0,
            epsilon_decay_steps=1,
            gamma=0.0,
        )
        for step in range(12500):
            a = step % 25
            policy.update(state, a, 1.0 if a == BEST else -0.1, state, done=True)
        policy._eps = 0.0
        q = policy._q_values(policy._online, (state / policy._scales).astype(np.float32))
        self.assertEqual(int(np.argmax(q)), BEST)


if __name__ == "__main__":
    unittest.main(verbosity=2)
