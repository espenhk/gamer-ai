"""Tests for RunTrace telemetry added for issue #461 (CarRacing analytics).

Covers two additions to framework.training._run_episode():
  - throttle_state entries grew from (accel, brake) to (accel, brake, steer),
    so games can histogram steering alongside gas/brake.
  - a new speed_trace field, sampled from info["speed_ms"] at the same
    cadence as pos_x/pos_z, populated only for games that report speed_ms.
"""

from __future__ import annotations

import unittest

import numpy as np

from framework.training import _TRACE_SAMPLE_EVERY, _run_episode


class _FakePolicy:
    def __call__(self, obs):
        return np.array([0.4, 1.0, 0.0], dtype=np.float32)  # steer, accel, brake

    def update(self, *args, **kwargs):
        pass

    def on_episode_start(self, **kwargs):
        pass


class _FakeEnvWithSpeed:
    """Multi-step env that reports info["speed_ms"] every step (CarRacing-like)."""

    def __init__(self, n_steps: int) -> None:
        self._n_steps = n_steps
        self._i = 0

    def reset(self, *, seed=None, options=None):
        self._i = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._i += 1
        done = self._i >= self._n_steps
        info = {"speed_ms": float(self._i)}
        return np.zeros(4, dtype=np.float32), 1.0, done, False, info


class _FakeEnvWithoutSpeed:
    """Multi-step env whose info dict never mentions speed_ms."""

    def __init__(self, n_steps: int) -> None:
        self._n_steps = n_steps
        self._i = 0

    def reset(self, *, seed=None, options=None):
        self._i = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._i += 1
        done = self._i >= self._n_steps
        return np.zeros(4, dtype=np.float32), 1.0, done, False, {}


class TestThrottleStateIncludesSteer(unittest.TestCase):
    def test_throttle_state_entries_are_three_tuples(self):
        env = _FakeEnvWithoutSpeed(n_steps=3)
        obs, reset_info = env.reset()
        ep = _run_episode(env, _FakePolicy(), obs, reset_info=reset_info)
        self.assertTrue(ep.trace.throttle_state)
        for accel, brake, steer in ep.trace.throttle_state:
            self.assertAlmostEqual(accel, 1.0)
            self.assertAlmostEqual(brake, 0.0)
            self.assertAlmostEqual(steer, 0.4)


class TestSpeedTrace(unittest.TestCase):
    def test_populated_when_env_reports_speed_ms(self):
        n_steps = 3 * _TRACE_SAMPLE_EVERY
        env = _FakeEnvWithSpeed(n_steps=n_steps)
        obs, reset_info = env.reset()
        ep = _run_episode(env, _FakePolicy(), obs, reset_info=reset_info)
        self.assertTrue(ep.trace.speed_trace)
        expected = [float(i) for i in range(1, n_steps + 1) if i % _TRACE_SAMPLE_EVERY == 0]
        self.assertEqual(ep.trace.speed_trace, expected)

    def test_empty_when_env_never_reports_speed_ms(self):
        env = _FakeEnvWithoutSpeed(n_steps=3 * _TRACE_SAMPLE_EVERY)
        obs, reset_info = env.reset()
        ep = _run_episode(env, _FakePolicy(), obs, reset_info=reset_info)
        self.assertEqual(ep.trace.speed_trace, [])


if __name__ == "__main__":
    unittest.main()
