"""Tests for BeamNGEnv's episode_obs_averages telemetry (issue #462).

Injects a stub ``beamng_gym`` module so the env imports and runs without the
BeamNG binary or the beamng-gym package.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Inject a stub beamng_gym module before games.beamng.env is imported, so the
# module-level try/except succeeds without the real package installed.
if "beamng_gym" not in sys.modules:
    sys.modules["beamng_gym"] = MagicMock()

from games.beamng.env import BeamNGEnv  # noqa: E402
from games.beamng.obs_spec import BASE_OBS_DIM  # noqa: E402


class _StubBeamNGGym:
    """Minimal beamng_gym-style env emitting deterministic 13-dim obs."""

    def __init__(self, episode_len: int = 6) -> None:
        self.episode_len = episode_len
        self._t = 0

    def reset(self):
        self._t = 0
        return self._obs(progress=0.0)

    # beamng_gym step: (obs, reward, done, info) 4-tuple
    def step(self, action):
        self._t += 1
        progress = min(1.0, self._t / float(self.episode_len))
        done = progress >= 1.0
        return self._obs(progress=progress), 0.0, done, {}

    def close(self):
        pass

    def _obs(self, progress: float):
        obs = np.zeros(BASE_OBS_DIM, dtype=np.float32)
        obs[0] = 30.0  # speed_ms
        obs[1] = -1.5  # lateral_offset_m (negative — mean uses abs)
        obs[3] = progress  # track_progress
        obs[6:10] = [1.0, 1.0, 0.0, 0.0]  # two wheels grounded → airborne
        return obs


@pytest.fixture
def env(monkeypatch):
    import games.beamng.env as beamng_env_module

    monkeypatch.setattr(beamng_env_module.beamng_gym, "make", lambda: _StubBeamNGGym())
    return BeamNGEnv()


def _run_episode(env):
    env.reset()
    action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    info = {}
    for _ in range(20):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return info


class TestEpisodeObsAverages:
    def test_terminal_info_carries_obs_averages(self, env):
        info = _run_episode(env)
        avgs = info.get("episode_obs_averages")
        assert avgs is not None
        assert avgs["speed_ms"] == pytest.approx(30.0)
        assert avgs["abs_lateral_offset_m"] == pytest.approx(1.5)
        assert avgs["wheel_0_contact"] == pytest.approx(1.0)
        assert avgs["wheel_2_contact"] == pytest.approx(0.0)
        # Two wheels grounded → not airborne (airborne = ≤1 wheel contact).
        assert avgs["airborne"] == pytest.approx(0.0)

    def test_not_present_on_non_terminal_steps(self, env):
        env.reset()
        action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        _, _, terminated, truncated, info = env.step(action)
        assert not (terminated or truncated)
        assert "episode_obs_averages" not in info

    def test_accumulators_reset_between_episodes(self, env):
        _run_episode(env)
        env.reset()
        assert env._ep_obs_steps == 0
        assert env._ep_obs_sums == {}
        info = _run_episode(env)
        assert info["episode_obs_averages"]["speed_ms"] == pytest.approx(30.0)

    def test_airborne_flagged_when_single_wheel_grounded(self, monkeypatch):
        import games.beamng.env as beamng_env_module

        class _AirborneStub(_StubBeamNGGym):
            def _obs(self, progress: float):
                obs = super()._obs(progress)
                obs[6:10] = [1.0, 0.0, 0.0, 0.0]  # one wheel → airborne
                return obs

        monkeypatch.setattr(beamng_env_module.beamng_gym, "make", lambda: _AirborneStub())
        env = BeamNGEnv()
        info = _run_episode(env)
        assert info["episode_obs_averages"]["airborne"] == pytest.approx(1.0)
