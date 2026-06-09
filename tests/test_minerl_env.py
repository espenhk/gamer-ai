"""Tests for MineRLEnv — mocks the minerl package so tests run without Java."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Inject a stub minerl module before games.minerl.env is imported, so the
# module-level try/except succeeds without the real package installed.
if "minerl" not in sys.modules:
    sys.modules["minerl"] = MagicMock()

from games.minerl.actions import N_ACTIONS  # noqa: E402
from games.minerl.env import _MINERL_FPS, MineRLEnv  # noqa: E402

_OBS_DICT = {"compassAngle": 45.0, "inventory": {"dirt": 3, "log": 1}}
_STEP_RESULT_5 = (_OBS_DICT, 1.0, False, False, {"native_reward": 1.0})


def _mock_inner():
    """Build a minimal mock of the inner gymnasium env."""
    m = MagicMock()
    m.reset.return_value = (_OBS_DICT, {})
    m.step.return_value = _STEP_RESULT_5
    # SimpleNamespace has neither .noop nor .spaces — exercises the fallback
    # paths in _build_noop and _decode_action.
    m.action_space = SimpleNamespace()
    return m


@pytest.fixture
def env():
    with patch("games.minerl.env.gym.make") as mock_make:
        mock_make.return_value = _mock_inner()
        yield MineRLEnv(max_episode_steps=100)


class TestMineRLEnvReset:
    def test_reset_returns_float32_array(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

    def test_reset_shape_matches_obs_dim(self, env):
        obs, _ = env.reset()
        assert obs.shape == (3,)

    def test_reset_obs_values(self, env):
        obs, _ = env.reset()
        assert abs(float(obs[0]) - 45.0 / 180.0) < 1e-5
        assert abs(float(obs[1]) - 3.0 / 64.0) < 1e-5
        assert abs(float(obs[2]) - 1.0 / 64.0) < 1e-5

    def test_reset_clears_step_count(self, env):
        env.reset()
        env._step_count = 50
        env.reset()
        assert env._step_count == 0


class TestMineRLEnvStep:
    def test_step_obs_shape(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(np.array([0.0]))
        assert obs.shape == (3,)

    def test_step_obs_dtype(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(np.array([0.0]))
        assert obs.dtype == np.float32

    def test_step_increments_step_count(self, env):
        env.reset()
        env.step(np.array([0.0]))
        assert env._step_count == 1

    def test_step_truncates_at_max_steps(self, env):
        env.reset()
        env._step_count = 99
        _, _, _, truncated, _ = env.step(np.array([0.0]))
        assert truncated is True

    def test_step_not_truncated_before_limit(self, env):
        env.reset()
        _, _, _, truncated, _ = env.step(np.array([0.0]))
        assert truncated is False

    def test_step_info_contains_native_reward(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.array([1.0]))
        assert "native_reward" in info

    def test_all_action_indices_accepted(self, env):
        for action_idx in range(N_ACTIONS):
            env.reset()
            obs, _, _, _, _ = env.step(np.array([float(action_idx)]))
            assert obs.shape == (3,)


class TestMineRLEnvClose:
    def test_close_propagates_to_inner_env(self, env):
        env.close()
        env._env.close.assert_called_once()


class TestMineRLEnvTimeLimits:
    def test_get_episode_time_limit_matches_constructor(self, env):
        expected = 100.0 / _MINERL_FPS
        assert abs(env.get_episode_time_limit() - expected) < 1e-9

    def test_set_episode_time_limit_updates_max_steps(self, env):
        env.set_episode_time_limit(10.0)
        assert env._max_episode_steps == int(10.0 * _MINERL_FPS)


class TestMineRLEnvImportError:
    def test_import_error_without_minerl(self):
        """Importing env without minerl in sys.modules raises ImportError."""
        import importlib

        saved = sys.modules.pop("games.minerl.env", None)
        saved_minerl = sys.modules.pop("minerl", None)
        try:
            with pytest.raises(ImportError, match="minerl"):
                importlib.import_module("games.minerl.env")
        finally:
            if saved is not None:
                sys.modules["games.minerl.env"] = saved
            if saved_minerl is not None:
                sys.modules["minerl"] = saved_minerl
