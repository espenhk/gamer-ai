"""Smoke tests for the Assetto Corsa game integration.

Uses a stub gym env so the test runs on Linux CI without the upstream
assetto-corsa-rl package or the AC binary. Verifies:

- AssettoCorsaEnv constructs against an injected env factory.
- reset() returns an observation matching the obs_spec dimension.
- step() returns the standard 5-tuple with finite reward and the
  canonical info keys (pos_x, pos_z, track_progress, lateral_offset).
- A 5-episode training loop runs end-to-end with WeightedLinearPolicy
  (acceptance criterion #3).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from framework.policies import WeightedLinearPolicy
from games.assetto_corsa.env import AssettoCorsaEnv
from games.assetto_corsa.obs_spec import AC_OBS_SPEC, BASE_OBS_DIM, with_vision
from games.assetto_corsa.reward import RewardCalculator, RewardConfig


# ---------------------------------------------------------------------------
# Stub gym env
# ---------------------------------------------------------------------------

class _StubACEnv:
    """Minimal gym-compatible env that emits deterministic AC-shaped telemetry."""

    def __init__(self, episode_len: int = 8) -> None:
        self.episode_len = episode_len
        self._t = 0

    # Gymnasium reset: (obs, info)
    def reset(self):
        self._t = 0
        obs = self._obs(progress=0.0)
        return obs, {}

    # Gymnasium step: (obs, reward, terminated, truncated, info)
    def step(self, action):
        self._t += 1
        progress = min(1.0, self._t / float(self.episode_len))
        terminated = progress >= 1.0
        return self._obs(progress=progress), 0.0, terminated, False, {}

    def close(self) -> None:
        pass

    def _obs(self, progress: float) -> dict:
        return {
            "speed_ms":        20.0 + progress * 30.0,
            "lateral_offset":  0.5,
            "yaw_error":       0.0,
            "pitch":           0.0,
            "roll":            0.0,
            "track_progress":  progress,
            "steering_angle":  0.0,
            "engine_rpm":      4000.0,
            "gear":            3.0,
            "wheel_slip":      (0.1, 0.1, 0.1, 0.1),
            "angular_velocity":(0.0, 0.0, 0.0),
            "pos_x":           progress * 100.0,
            "pos_z":           0.0,
        }


def _factory(_env_id, **_kwargs):
    return _StubACEnv()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_obs_spec_dimensions_match_base_obs_dim():
    assert AC_OBS_SPEC.dim == BASE_OBS_DIM
    assert with_vision(0).dim == BASE_OBS_DIM
    assert with_vision(4).dim == BASE_OBS_DIM + 4


def test_env_reset_returns_correct_obs_shape():
    env = AssettoCorsaEnv(env_factory=_factory)
    try:
        obs, info = env.reset()
        assert obs.shape == (AC_OBS_SPEC.dim,)
        assert obs.dtype == np.float32
        assert info == {}
    finally:
        env.close()


def test_env_step_returns_5_tuple_with_finite_reward():
    env = AssettoCorsaEnv(env_factory=_factory)
    try:
        env.reset()
        action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (AC_OBS_SPEC.dim,)
        assert math.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        for key in ("pos_x", "pos_z", "track_progress", "lateral_offset",
                    "finished", "termination_reason"):
            assert key in info, f"missing info key: {key}"
    finally:
        env.close()


def test_info_reflects_current_step_not_previous():
    """info dict must contain the current step's state, not _prev_state."""
    env = AssettoCorsaEnv(env_factory=_factory)
    episode_len = 8
    try:
        env.reset()
        action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for t in range(1, episode_len + 1):
            _, _, terminated, _, info = env.step(action)
            expected_progress = min(1.0, t / float(episode_len))
            assert info["track_progress"] == pytest.approx(expected_progress), (
                f"step {t}: info['track_progress'] should be current step's "
                f"progress {expected_progress}, got {info['track_progress']}"
            )
            if terminated:
                break
    finally:
        env.close()


def test_env_terminates_on_finish():
    env = AssettoCorsaEnv(env_factory=_factory)
    try:
        env.reset()
        action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        terminated = False
        for _ in range(20):
            _, _, terminated, _, info = env.step(action)
            if terminated:
                break
        assert terminated
        assert info.get("termination_reason") == "finish"
        assert info.get("laps_completed") == 1
    finally:
        env.close()


def test_env_with_vision_features():
    """Vision features default to zeros when the stub doesn't supply them."""
    env = AssettoCorsaEnv(env_factory=_factory, n_vision=4)
    try:
        obs, _ = env.reset()
        assert obs.shape == (BASE_OBS_DIM + 4,)
        # Last 4 entries are vision; stub emits none → zeros.
        assert np.allclose(obs[-4:], 0.0)
    finally:
        env.close()


def test_reward_calculator_computes_finite_value():
    cfg = RewardConfig()
    calc = RewardCalculator(cfg)
    prev = {"track_progress": 0.0, "lateral_offset": 0.0, "speed_ms": 0.0}
    curr = {"track_progress": 0.1, "lateral_offset": 0.5, "speed_ms": 20.0}
    r = calc.compute(prev, curr, finished=False, elapsed_s=1.0,
                     info={"accelerating": True}, n_ticks=1)
    assert math.isfinite(r)
    # Progress delta dominates (0.1 * progress_weight).
    assert r > 0


def test_five_episode_training_loop_with_linear_policy():
    """Acceptance criterion #3: existing policies trainable for ≥5 episodes."""
    env = AssettoCorsaEnv(env_factory=_factory)
    obs_spec = AC_OBS_SPEC
    policy = WeightedLinearPolicy(obs_spec, ["steer", "accel", "brake"], None)

    try:
        rewards = []
        for _ in range(5):
            obs, _ = env.reset()
            total = 0.0
            for _ in range(50):
                action = policy(obs)
                obs, r, term, trunc, _ = env.step(action)
                total += r
                if term or trunc:
                    break
            rewards.append(total)
        assert len(rewards) == 5
        assert all(math.isfinite(r) for r in rewards)
    finally:
        env.close()
