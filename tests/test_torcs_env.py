"""Tests for the TORCS environment wrapper.

These tests validate the env's spaces, episode logic, and termination
without requiring TORCS to be installed (gym_torcs import is lazy).
"""
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from games.torcs.env import TorcsEnv
from games.torcs.reward import TorcsRewardConfig
from games.torcs.obs_spec import BASE_OBS_DIM


class TestTorcsEnvSpaces(unittest.TestCase):
    """Validate observation and action space definitions."""

    def setUp(self):
        # Patch TorcsClient so we don't need TORCS installed.
        with patch("games.torcs.env.TorcsClient"):
            self.env = TorcsEnv()

    def test_observation_space_shape(self):
        self.assertEqual(self.env.observation_space.shape, (BASE_OBS_DIM,))

    def test_action_space_shape(self):
        self.assertEqual(self.env.action_space.shape, (3,))

    def test_action_space_bounds(self):
        np.testing.assert_array_equal(
            self.env.action_space.low, [-1.0, 0.0, 0.0]
        )
        np.testing.assert_array_equal(
            self.env.action_space.high, [1.0, 1.0, 1.0]
        )


class TestTorcsEnvEpisodeTime(unittest.TestCase):
    """Test the episode time limit API."""

    def setUp(self):
        with patch("games.torcs.env.TorcsClient"):
            self.env = TorcsEnv(max_episode_time_s=60.0)

    def test_get_episode_time_limit(self):
        self.assertEqual(self.env.get_episode_time_limit(), 60.0)

    def test_set_episode_time_limit(self):
        self.env.set_episode_time_limit(30.0)
        self.assertEqual(self.env.get_episode_time_limit(), 30.0)


class TestTorcsEnvStepLogic(unittest.TestCase):
    """Test step/reset with mocked client."""

    def setUp(self):
        patcher = patch("games.torcs.env.TorcsClient")
        self.mock_client_cls = patcher.start()
        self.addCleanup(patcher.stop)

        self.mock_client = self.mock_client_cls.return_value
        self.mock_client.reset.return_value = np.zeros(BASE_OBS_DIM, dtype=np.float32)
        self.mock_client.step.return_value = (
            np.zeros(BASE_OBS_DIM, dtype=np.float32),
            0.0,
            False,
            {},
        )
        self.env = TorcsEnv()

    def test_reset_returns_obs_and_info(self):
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, (BASE_OBS_DIM,))
        self.assertIsInstance(info, dict)

    def test_step_returns_five_tuple(self):
        self.env.reset()
        action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        result = self.env.step(action)
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertEqual(obs.shape, (BASE_OBS_DIM,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_crash_terminates(self):
        """Episode should terminate when lateral offset exceeds threshold."""
        cfg = TorcsRewardConfig(crash_threshold_m=5.0)
        with patch("games.torcs.env.TorcsClient"):
            env = TorcsEnv(reward_config=cfg)
        env._client = self.mock_client

        # Lateral offset at index 1 exceeds crash threshold
        obs_far = np.zeros(BASE_OBS_DIM, dtype=np.float32)
        obs_far[1] = 10.0  # lateral_offset_m
        self.mock_client.step.return_value = (obs_far, 0.0, False, {})

        env.reset()
        _, _, terminated, _, info = env.step(np.array([0., 1., 0.]))
        self.assertTrue(terminated)
        self.assertEqual(info["termination_reason"], "crash")

    def test_info_contains_expected_keys(self):
        self.env.reset()
        _, _, _, _, info = self.env.step(np.array([0., 1., 0.]))
        for key in ("speed_ms", "lateral_offset", "track_progress",
                     "termination_reason", "elapsed_s"):
            self.assertIn(key, info)

    def test_close_calls_client_close(self):
        self.env.close()
        self.mock_client.close.assert_called_once()


class TestTorcsEnvActions(unittest.TestCase):
    """Test TORCS action definitions."""

    def test_discrete_actions_shape(self):
        from games.torcs.actions import DISCRETE_ACTIONS
        self.assertEqual(DISCRETE_ACTIONS.shape, (9, 3))

    def test_probe_actions_count(self):
        from games.torcs.actions import PROBE_ACTIONS
        self.assertEqual(len(PROBE_ACTIONS), 6)

    def test_warmup_action_shape(self):
        from games.torcs.actions import WARMUP_ACTION
        self.assertEqual(WARMUP_ACTION.shape, (3,))


if __name__ == "__main__":
    unittest.main()
