"""Observation-history wrappers for enriching policy inputs.

FrameStack
    Gymnasium wrapper that stacks the last *K* observations into a single
    flat vector.  The policy sees ``K * obs_dim`` features without any
    code changes.  Useful for velocity estimation and short-horizon
    temporal context.

TimestampedRingBuffer
    Stores the last *N* ``(timestamp, obs)`` pairs and emits a fixed-size
    summary.  Used as a building block for belief modules.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FrameStack(gym.Wrapper):
    """Stack the last *k* observations into a single flat vector.

    The observation space becomes ``Box(shape=(k * obs_dim,))``.
    On ``reset()`` the buffer is pre-filled with copies of the initial
    observation so the policy always sees a full-width input.

    Parameters
    ----------
    env :
        Gymnasium environment to wrap.
    k :
        Number of frames to stack.  ``k=1`` is a no-op pass-through.
    """

    def __init__(self, env: gym.Env, k: int = 4) -> None:
        super().__init__(env)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._k = k
        obs_shape = env.observation_space.shape
        if obs_shape is None or len(obs_shape) != 1:
            raise ValueError(
                "FrameStack requires a 1-D observation space, "
                f"got shape={obs_shape}"
            )
        self._obs_dim = obs_shape[0]
        low = np.tile(env.observation_space.low, k)
        high = np.tile(env.observation_space.high, k)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32,
        )
        self._frames: deque[np.ndarray] = deque(maxlen=k)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self._k):
            self._frames.append(obs.copy())
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs.copy())
        return self._stacked(), reward, terminated, truncated, info

    def _stacked(self) -> np.ndarray:
        return np.concatenate(list(self._frames), dtype=np.float32)


class TimestampedRingBuffer:
    """Fixed-capacity ring buffer storing ``(timestamp, obs)`` pairs.

    Parameters
    ----------
    capacity :
        Maximum entries to keep.
    obs_dim :
        Dimensionality of each observation vector.
    """

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self._capacity = capacity
        self._obs_dim = obs_dim
        self._timestamps: deque[float] = deque(maxlen=capacity)
        self._observations: deque[np.ndarray] = deque(maxlen=capacity)

    def push(self, timestamp: float, obs: np.ndarray) -> None:
        """Record an observation at *timestamp*."""
        self._timestamps.append(timestamp)
        self._observations.append(obs.copy())

    def most_recent(self, k: int) -> np.ndarray:
        """Return the *k* most-recent observations as ``(k, obs_dim)`` array.

        If fewer than *k* entries exist the array is zero-padded at the front.
        """
        buf = list(self._observations)
        n = len(buf)
        out = np.zeros((k, self._obs_dim), dtype=np.float32)
        start = max(0, k - n)
        for i, obs in enumerate(buf[-k:]):
            out[start + i] = obs
        return out

    def clear(self) -> None:
        self._timestamps.clear()
        self._observations.clear()

    def __len__(self) -> int:
        return len(self._timestamps)
