"""Tests for the Rocket League environment wrapper.

Validates env spaces, episode logic, and reward routing without requiring
Rocket League or rlgym to be installed (the rlgym import is mocked).
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Mock rlgym before importing env so the module-level ImportError is bypassed.
# ---------------------------------------------------------------------------

_mock_rlgym = MagicMock()
_mock_rlgym_env = MagicMock()
_mock_rlgym.make.return_value = _mock_rlgym_env

_MODULE_PATCHES = {
    "rlgym": _mock_rlgym,
}


def _make_env(**kwargs):
    """Instantiate RocketLeagueEnv with rlgym mocked out."""
    from games.rocket_league.env import RocketLeagueEnv
    from games.rocket_league.reward import RocketLeagueRewardConfig
    reward_config = kwargs.pop("reward_config", None) or RocketLeagueRewardConfig()
    return RocketLeagueEnv(reward_config=reward_config, **kwargs)


class TestRocketLeagueEnvSpaces(unittest.TestCase):
    """Validate observation and action space definitions."""

    @classmethod
    def setUpClass(cls):
        import sys
        cls._old_modules = {k: sys.modules.get(k) for k in _MODULE_PATCHES}
        for mod, mock in _MODULE_PATCHES.items():
            sys.modules[mod] = mock

    @classmethod
    def tearDownClass(cls):
        import sys
        for mod, old in cls._old_modules.items():
            if old is None:
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = old

    def setUp(self):
        from games.rocket_league.obs_spec import BASE_OBS_DIM
        self._base_obs_dim = BASE_OBS_DIM
        self.env = _make_env()

    def test_observation_space_shape(self):
        self.assertEqual(
            self.env.observation_space.shape, (self._base_obs_dim,)
        )

    def test_action_space_shape(self):
        self.assertEqual(self.env.action_space.shape, (8,))

    def test_action_space_low(self):
        np.testing.assert_array_almost_equal(
            self.env.action_space.low,
            [-1., -1., -1., -1., -1., 0., 0., 0.],
        )

    def test_action_space_high(self):
        np.testing.assert_array_almost_equal(
            self.env.action_space.high,
            [1., 1., 1., 1., 1., 1., 1., 1.],
        )


class TestRocketLeagueEnvEpisodeTime(unittest.TestCase):
    """Test the episode time limit API."""

    @classmethod
    def setUpClass(cls):
        import sys
        cls._old_modules = {k: sys.modules.get(k) for k in _MODULE_PATCHES}
        for mod, mock in _MODULE_PATCHES.items():
            sys.modules[mod] = mock

    @classmethod
    def tearDownClass(cls):
        import sys
        for mod, old in cls._old_modules.items():
            if old is None:
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = old

    def test_get_episode_time_limit(self):
        env = _make_env(max_episode_time_s=60.0)
        self.assertEqual(env.get_episode_time_limit(), 60.0)

    def test_set_episode_time_limit(self):
        env = _make_env(max_episode_time_s=60.0)
        env.set_episode_time_limit(120.0)
        self.assertEqual(env.get_episode_time_limit(), 120.0)


class TestRocketLeagueEnvStepLogic(unittest.TestCase):
    """Test step/reset with mocked rlgym client."""

    @classmethod
    def setUpClass(cls):
        import sys
        cls._old_modules = {k: sys.modules.get(k) for k in _MODULE_PATCHES}
        for mod, mock in _MODULE_PATCHES.items():
            sys.modules[mod] = mock

    @classmethod
    def tearDownClass(cls):
        import sys
        for mod, old in cls._old_modules.items():
            if old is None:
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = old

    def setUp(self):
        from games.rocket_league.obs_spec import BASE_OBS_DIM
        self._dim = BASE_OBS_DIM

        self._raw_obs = np.zeros(BASE_OBS_DIM, dtype=np.float32)
        _mock_rlgym_env.reset.return_value = self._raw_obs
        _mock_rlgym_env.step.return_value = (
            self._raw_obs, 0.0, False, {},
        )
        self.env = _make_env()

    def test_reset_returns_obs_and_info(self):
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, (self._dim,))
        self.assertIsInstance(info, dict)

    def test_step_returns_five_tuple(self):
        self.env.reset()
        action = np.zeros(8, dtype=np.float32)
        result = self.env.step(action)
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertEqual(obs.shape, (self._dim,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_info_contains_expected_keys(self):
        self.env.reset()
        _, _, _, _, info = self.env.step(np.zeros(8, dtype=np.float32))
        for key in ("vel_towards_ball", "boosting", "ball_touched",
                    "goal_scored", "goal_conceded", "elapsed_s",
                    "termination_reason"):
            self.assertIn(key, info)

    def test_boost_flag_detected_from_action(self):
        """action[6] > 0.5 should set info['boosting'] = True."""
        self.env.reset()
        action = np.zeros(8, dtype=np.float32)
        action[6] = 1.0  # boost on
        _, _, _, _, info = self.env.step(action)
        self.assertTrue(info["boosting"])

    def test_no_boost_flag(self):
        self.env.reset()
        action = np.zeros(8, dtype=np.float32)
        _, _, _, _, info = self.env.step(action)
        self.assertFalse(info["boosting"])

    def test_goal_scored_sets_termination_reason(self):
        _mock_rlgym_env.step.return_value = (
            self._raw_obs, 0.0, True, {"goal_scored": True},
        )
        self.env.reset()
        _, _, terminated, truncated, info = self.env.step(np.zeros(8))
        self.assertTrue(terminated or info.get("goal_scored"))

    def test_timeout_sets_truncated(self):
        """Elapsed time > max_episode_time_s should truncate the episode."""
        env = _make_env(max_episode_time_s=0.0)
        _mock_rlgym_env.reset.return_value = self._raw_obs
        _mock_rlgym_env.step.return_value = (self._raw_obs, 0.0, False, {})
        env.reset()
        _, _, terminated, truncated, info = env.step(np.zeros(8))
        self.assertTrue(truncated)
        self.assertEqual(info["termination_reason"], "timeout")

    def test_close_calls_underlying_env(self):
        self.env.close()
        _mock_rlgym_env.close.assert_called()

    def test_tick_skip_forwarded_to_rlgym_make(self):
        _make_env(tick_skip=12)
        _mock_rlgym.make.assert_called_with(tick_skip=12, team_size=3, self_play=False)

    def test_make_env_factory_forwards_tick_skip(self):
        from games.rocket_league.env import make_env
        make_env(experiment_dir=".", tick_skip=10)
        _mock_rlgym.make.assert_called_with(tick_skip=10, team_size=3, self_play=False)

    def test_reset_returns_multi_agent_obs_when_rlgym_returns_team_obs(self):
        per_agent_obs = [
            np.full(self._dim, 1.0, dtype=np.float32),
            np.full(self._dim, 2.0, dtype=np.float32),
            np.full(self._dim, 3.0, dtype=np.float32),
        ]
        _mock_rlgym_env.reset.return_value = per_agent_obs
        env = _make_env()
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (3, self._dim))
        self.assertEqual(float(obs[0, 0]), 1.0)
        self.assertEqual(float(obs[1, 0]), 2.0)
        self.assertEqual(float(obs[2, 0]), 3.0)

    def test_step_broadcasts_single_action_to_team_obs(self):
        per_agent_obs = [self._raw_obs.copy(), self._raw_obs.copy(), self._raw_obs.copy()]
        _mock_rlgym_env.reset.return_value = per_agent_obs
        _mock_rlgym_env.step.return_value = (per_agent_obs, 0.0, False, [{}, {}, {}])
        env = _make_env()
        env.reset()
        env.step(np.zeros(8, dtype=np.float32))
        sent_action = _mock_rlgym_env.step.call_args[0][0]
        self.assertEqual(sent_action.shape, (3, 8))

    def test_step_info_contains_per_agent_metrics(self):
        per_agent_obs = [self._raw_obs.copy(), self._raw_obs.copy(), self._raw_obs.copy()]
        _mock_rlgym_env.reset.return_value = per_agent_obs
        _mock_rlgym_env.step.return_value = (per_agent_obs, 0.0, False, [{}, {}, {}])
        env = _make_env()
        env.reset()
        _, _, _, _, info = env.step(np.zeros((3, 8), dtype=np.float32))
        self.assertEqual(len(info["vel_towards_ball_agents"]), 3)
        self.assertEqual(len(info["boosting_agents"]), 3)
        self.assertEqual(info["team_agent_count"], 3)


class TestRocketLeagueActions(unittest.TestCase):
    """Test Rocket League action definitions."""

    def test_discrete_actions_shape(self):
        from games.rocket_league.actions import DISCRETE_ACTIONS
        self.assertEqual(DISCRETE_ACTIONS.shape[1], 8)
        self.assertGreaterEqual(DISCRETE_ACTIONS.shape[0], 9)

    def test_probe_actions_count(self):
        from games.rocket_league.actions import PROBE_ACTIONS
        self.assertEqual(len(PROBE_ACTIONS), 6)
        for action, label in PROBE_ACTIONS:
            self.assertEqual(action.shape, (8,))
            self.assertIsInstance(label, str)

    def test_warmup_action_shape(self):
        from games.rocket_league.actions import WARMUP_ACTION
        self.assertEqual(WARMUP_ACTION.shape, (8,))

    def test_action_bounds_respected(self):
        from games.rocket_league.actions import DISCRETE_ACTIONS, ACTION_LOW, ACTION_HIGH
        self.assertTrue(np.all(DISCRETE_ACTIONS >= ACTION_LOW))
        self.assertTrue(np.all(DISCRETE_ACTIONS <= ACTION_HIGH))


if __name__ == "__main__":
    unittest.main()
