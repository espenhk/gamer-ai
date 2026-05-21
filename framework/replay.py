"""Replay buffers for off-policy RL algorithms.

ReplayBuffer        — circular (obs, action_idx, reward, next_obs, done) buffer
MaskedReplayBuffer  — extends with a per-transition available-actions mask
"""
from __future__ import annotations

from collections import deque

import numpy as np


class ReplayBuffer:
    """Fixed-size circular buffer of (obs, action_idx, reward, next_obs, done) tuples."""

    def __init__(self, maxlen: int) -> None:
        self._buf: deque = deque(maxlen=maxlen)

    def push(
        self,
        obs: np.ndarray,
        action_idx: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((
            obs.copy(), int(action_idx), float(reward), next_obs.copy(), bool(done),
        ))

    def sample(
        self, batch_size: int, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        replace = batch_size > len(self._buf)
        _rng    = rng if rng is not None else np.random
        idxs    = _rng.choice(len(self._buf), size=batch_size, replace=replace)
        batch   = [self._buf[i] for i in idxs]
        obs_b   = np.stack([t[0] for t in batch]).astype(np.float32)
        act_b   = np.array([t[1] for t in batch], dtype=np.int32)
        rew_b   = np.array([t[2] for t in batch], dtype=np.float32)
        next_b  = np.stack([t[3] for t in batch]).astype(np.float32)
        done_b  = np.array([t[4] for t in batch], dtype=np.float32)
        return obs_b, act_b, rew_b, next_b, done_b

    def __len__(self) -> int:
        return len(self._buf)


class MaskedReplayBuffer(ReplayBuffer):
    """Replay buffer that stores an available-actions mask with each transition.

    Adds a boolean ``mask`` column of shape ``(n_actions,)`` to each stored
    tuple.  The mask represents which actions are legal in the *next* state s'
    of the transition; used to restrict ``max_a' Q_target(s', a')`` to legal
    next-state actions so the target network never bootstraps through
    unavailable Q-values.
    """

    def __init__(self, maxlen: int, n_actions: int) -> None:
        super().__init__(maxlen)
        self._n_actions = n_actions

    def push(  # type: ignore[override]
        self,
        obs: np.ndarray,
        action_idx: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        mask: np.ndarray | None = None,
    ) -> None:
        if mask is None:
            mask = np.ones(self._n_actions, dtype=bool)
        self._buf.append((
            obs.copy(), int(action_idx), float(reward),
            next_obs.copy(), bool(done), mask.copy(),
        ))

    def sample(  # type: ignore[override]
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        replace = batch_size > len(self._buf)
        _rng    = rng if rng is not None else np.random
        idxs    = _rng.choice(len(self._buf), size=batch_size, replace=replace)
        batch   = [self._buf[i] for i in idxs]
        obs_b   = np.stack([t[0] for t in batch]).astype(np.float32)
        act_b   = np.array([t[1] for t in batch], dtype=np.int32)
        rew_b   = np.array([t[2] for t in batch], dtype=np.float32)
        next_b  = np.stack([t[3] for t in batch]).astype(np.float32)
        done_b  = np.array([t[4] for t in batch], dtype=np.float32)
        mask_b  = np.stack([t[5] for t in batch])   # (B, n_actions) bool
        return obs_b, act_b, rew_b, next_b, done_b, mask_b
