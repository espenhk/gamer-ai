"""Picklable env/policy stubs for test_parallel_eval.py.

Lives in a standalone module so spawn() child processes can import it by
qualified name (``tests._parallel_eval_helpers``).  Conftest puts both
the repo root and ``tests/`` on ``sys.path`` and re-exports them via
``PYTHONPATH`` so the children inherit the search paths.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class DummyEnv:
    """Minimal env: each step pays out a fraction of the policy's reward signal.

    Reward semantics:
      - Each ``reset()`` clears the episode counters.
      - Each ``step(action)`` returns reward = action[0] (i.e. the first element
        of whatever the policy emits) and increments steps.
      - Truncation after ``max_steps`` steps.

    ``crash_on_idx`` lets us simulate a worker dying on a specific candidate
    (the dummy policy passes its individual_idx into action[0] via the
    flat weights, see ``DummyPolicy``).
    """

    def __init__(
        self,
        obs_dim: int = 4,
        max_steps: int = 5,
        crash_on_idx: int | None = None,
    ):
        self.obs_dim = obs_dim
        self.max_steps = max_steps
        self.crash_on_idx = crash_on_idx
        self._steps = 0
        self._time_limit: float | None = None
        self._last_idx: int | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._steps = 0
        # Surface the most recent time-limit so it ends up in the result info
        # dict — lets tests verify that the per-job time limit propagated.
        info = {"time_limit_seen": self._time_limit}
        return np.zeros(self.obs_dim, dtype=np.float32), info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        # Crash check: action[0] is the encoded individual_idx (see DummyPolicy).
        if self.crash_on_idx is not None and int(round(float(a[0]))) == self.crash_on_idx:
            raise RuntimeError(f"intentional crash on individual_idx={self.crash_on_idx}")
        self._steps += 1
        reward = float(a[0]) if len(a) > 0 else 0.0
        terminated = False
        truncated = self._steps >= self.max_steps
        info = {
            "track_progress": self._steps / self.max_steps,
            "time_limit_seen": self._time_limit,
        }
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        return obs, reward, terminated, truncated, info

    def close(self):
        pass

    def get_episode_time_limit(self):
        return self._time_limit

    def set_episode_time_limit(self, seconds: float) -> None:
        self._time_limit = float(seconds)


class DummyPolicy:
    """Policy whose action is a constant taken from its (single-element) flat weight.

    ``with_flat([x])`` returns a fresh policy that emits ``[x, 0, 0]`` every step.
    The trailing zeros are there so ``_run_episode``'s throttle-classification
    branch (which needs ``len(action) >= 3``) takes the cheap path.
    """

    def __init__(self, value: float = 0.0):
        self._value = float(value)

    def __call__(self, obs):
        return np.array([self._value, 0.0, 0.0], dtype=np.float32)

    def with_flat(self, flat):
        return DummyPolicy(value=float(flat[0]))

    def to_flat(self):
        return np.array([self._value], dtype=np.float32)

    def on_episode_start(self, **kwargs):
        pass

    def on_episode_end(self):
        pass

    def update(self, *args, **kwargs):
        pass

    def to_cfg(self):
        return {"value": self._value}


def make_dummy_env() -> DummyEnv:
    """Default picklable env factory."""
    return DummyEnv()


def make_dummy_env_max_steps_2() -> DummyEnv:
    """Variant used in tests that want shorter episodes."""
    return DummyEnv(max_steps=2)


class _CrashingEnvFactory:
    """Picklable env factory parameterised by crash_on_idx (closures don't pickle)."""

    def __init__(self, crash_on_idx: int, max_steps: int = 3):
        self.crash_on_idx = crash_on_idx
        self.max_steps = max_steps

    def __call__(self) -> DummyEnv:
        return DummyEnv(max_steps=self.max_steps, crash_on_idx=self.crash_on_idx)


def make_crashing_env_factory(crash_on_idx: int, max_steps: int = 3):
    return _CrashingEnvFactory(crash_on_idx=crash_on_idx, max_steps=max_steps)
