"""Tests for framework/obs_memory.py — FrameStack and TimestampedRingBuffer."""
import unittest

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from framework.obs_memory import FrameStack, TimestampedRingBuffer


class _DummyEnv(gym.Env):
    """Minimal Gymnasium env with incrementing observations for testing."""

    def __init__(self, obs_dim: int = 4):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._step_count = 0

    def reset(self, **kwargs):
        self._step_count = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = np.full(self.observation_space.shape, float(self._step_count), dtype=np.float32)
        return obs, 0.0, False, False, {}


class TestFrameStack(unittest.TestCase):

    def test_observation_shape(self):
        env = _DummyEnv(obs_dim=3)
        wrapped = FrameStack(env, k=4)
        obs, _ = wrapped.reset()
        self.assertEqual(obs.shape, (12,))

    def test_reset_fills_with_initial(self):
        env = _DummyEnv(obs_dim=2)
        wrapped = FrameStack(env, k=3)
        obs, _ = wrapped.reset()
        # reset returns zeros; all 3 frames should be zero
        np.testing.assert_array_equal(obs, np.zeros(6, dtype=np.float32))

    def test_step_shifts_frames(self):
        env = _DummyEnv(obs_dim=2)
        wrapped = FrameStack(env, k=2)
        obs, _ = wrapped.reset()
        obs, *_ = wrapped.step(0)
        # Frame 0 is reset obs (zeros), frame 1 is step 1 (ones)
        np.testing.assert_array_equal(obs[:2], np.zeros(2))
        np.testing.assert_array_equal(obs[2:], np.ones(2))

    def test_k_one_is_passthrough(self):
        env = _DummyEnv(obs_dim=3)
        wrapped = FrameStack(env, k=1)
        obs, _ = wrapped.reset()
        self.assertEqual(obs.shape, (3,))

    def test_invalid_k_raises(self):
        env = _DummyEnv()
        with self.assertRaises(ValueError):
            FrameStack(env, k=0)


class TestTimestampedRingBuffer(unittest.TestCase):

    def test_most_recent_zero_padded(self):
        buf = TimestampedRingBuffer(capacity=5, obs_dim=3)
        buf.push(0.0, np.array([1, 2, 3], dtype=np.float32))
        out = buf.most_recent(3)
        self.assertEqual(out.shape, (3, 3))
        # First two rows should be zero-padded
        np.testing.assert_array_equal(out[0], np.zeros(3))
        np.testing.assert_array_equal(out[1], np.zeros(3))
        np.testing.assert_array_equal(out[2], [1, 2, 3])

    def test_clear(self):
        buf = TimestampedRingBuffer(capacity=5, obs_dim=2)
        buf.push(0.0, np.array([1, 2], dtype=np.float32))
        buf.clear()
        self.assertEqual(len(buf), 0)


if __name__ == "__main__":
    unittest.main()
