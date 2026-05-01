"""Tests for the SC2 environment wrapper.

These tests validate the env's spaces, episode logic, and termination
without requiring PySC2 to be installed (the client is mocked).
"""
import unittest
from unittest.mock import patch

import numpy as np

from games.sc2.env import SC2Env
from games.sc2.obs_spec import BASE_OBS_DIM, LADDER_OBS_DIM
from games.sc2.reward import SC2RewardConfig


class TestSC2EnvSpaces(unittest.TestCase):

    def setUp(self):
        with patch("games.sc2.env.SC2Client"):
            self.env = SC2Env(map_name="MoveToBeacon")

    def test_minigame_observation_space(self):
        self.assertEqual(self.env.observation_space.shape, (BASE_OBS_DIM,))

    def test_action_space_shape(self):
        self.assertEqual(self.env.action_space.shape, (4,))

    def test_action_space_bounds(self):
        # fn_idx range, then x/y/queue in [0,1].
        self.assertEqual(float(self.env.action_space.low[0]), 0.0)
        np.testing.assert_array_equal(
            self.env.action_space.high[1:], [1.0, 1.0, 1.0]
        )


class TestSC2EnvLadderSpaces(unittest.TestCase):

    def test_ladder_obs_space(self):
        with patch("games.sc2.env.SC2Client"):
            env = SC2Env(map_name="Simple64")
        self.assertEqual(env.observation_space.shape, (LADDER_OBS_DIM,))


class TestSC2EnvEpisodeTime(unittest.TestCase):

    def setUp(self):
        with patch("games.sc2.env.SC2Client"):
            self.env = SC2Env(map_name="MoveToBeacon", max_episode_time_s=60.0)

    def test_get_episode_time_limit(self):
        self.assertEqual(self.env.get_episode_time_limit(), 60.0)

    def test_set_episode_time_limit(self):
        self.env.set_episode_time_limit(30.0)
        self.assertEqual(self.env.get_episode_time_limit(), 30.0)


class TestSC2EnvStepLogic(unittest.TestCase):
    """Test step/reset with mocked client."""

    def setUp(self):
        patcher = patch("games.sc2.env.SC2Client")
        self.mock_client_cls = patcher.start()
        self.addCleanup(patcher.stop)

        self.mock_client = self.mock_client_cls.return_value
        self.mock_client.reset.return_value = (
            np.zeros(BASE_OBS_DIM, dtype=np.float32),
            {"score": 0.0, "minerals": 50.0, "vespene": 0.0,
             "food_used": 1.0, "food_cap": 15.0, "army_count": 0.0},
        )
        self.mock_client.step.return_value = (
            np.zeros(BASE_OBS_DIM, dtype=np.float32),
            0.0,
            False,
            {"score": 1.0, "minerals": 50.0, "vespene": 0.0,
             "food_used": 1.0, "food_cap": 15.0, "army_count": 0.0},
        )
        self.env = SC2Env(map_name="MoveToBeacon")

    def test_reset_returns_obs_and_info(self):
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, (BASE_OBS_DIM,))
        self.assertIsInstance(info, dict)
        self.assertIn("score", info)

    def test_step_returns_five_tuple(self):
        self.env.reset()
        action = np.array([1, 0.5, 0.5, 0], dtype=np.float32)
        result = self.env.step(action)
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertEqual(obs.shape, (BASE_OBS_DIM,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_score_delta_reward(self):
        """With default reward config (score_weight=1.0), a +1 score delta
        should produce a reward close to +1 (minus tiny step_penalty)."""
        self.env.reset()
        _, reward, _, _, _ = self.env.step(np.zeros(4, dtype=np.float32))
        # score went 0 -> 1, default step_penalty -0.001 → reward ≈ 0.999.
        self.assertAlmostEqual(reward, 1.0 - 0.001, places=5)

    def test_done_terminates(self):
        """When the client signals done, terminated should be True."""
        self.mock_client.step.return_value = (
            np.zeros(BASE_OBS_DIM, dtype=np.float32),
            1.0,
            True,
            {"score": 5.0, "minerals": 0.0, "vespene": 0.0,
             "food_used": 0.0, "food_cap": 0.0, "army_count": 0.0,
             "player_outcome": 1.0, "is_last": True},
        )
        self.env.reset()
        _, _, terminated, truncated, info = self.env.step(np.zeros(4, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "win")

    def test_loss_outcome(self):
        self.mock_client.step.return_value = (
            np.zeros(BASE_OBS_DIM, dtype=np.float32),
            -1.0,
            True,
            {"score": 0.0, "minerals": 0.0, "vespene": 0.0,
             "food_used": 0.0, "food_cap": 0.0, "army_count": 0.0,
             "player_outcome": -1.0, "is_last": True},
        )
        self.env.reset()
        _, _, terminated, _, info = self.env.step(np.zeros(4, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertEqual(info["termination_reason"], "loss")

    def test_close_calls_client_close(self):
        self.env.close()
        self.mock_client.close.assert_called_once()

    def test_info_contains_required_keys(self):
        """Reward calculator depends on these keys from info."""
        self.env.reset()
        _, _, _, _, info = self.env.step(np.zeros(4, dtype=np.float32))
        for key in ("score", "prev_score", "minerals", "prev_minerals",
                     "vespene", "prev_vespene", "elapsed_s",
                     "termination_reason"):
            self.assertIn(key, info, f"missing key {key}")

    def test_prev_score_threaded_through(self):
        """prev_score in step N+1 should equal score from step N."""
        # Step 1 returns score=10
        self.mock_client.step.return_value = (
            np.zeros(BASE_OBS_DIM, dtype=np.float32),
            0.0,
            False,
            {"score": 10.0, "minerals": 0.0, "vespene": 0.0,
             "food_used": 1.0, "food_cap": 15.0, "army_count": 0.0},
        )
        self.env.reset()
        _, _, _, _, info1 = self.env.step(np.zeros(4, dtype=np.float32))
        self.assertEqual(info1["prev_score"], 0.0)
        self.assertEqual(info1["score"], 10.0)
        # Step 2 returns score=15.
        self.mock_client.step.return_value = (
            np.zeros(BASE_OBS_DIM, dtype=np.float32),
            0.0,
            False,
            {"score": 15.0, "minerals": 0.0, "vespene": 0.0,
             "food_used": 1.0, "food_cap": 15.0, "army_count": 0.0},
        )
        _, _, _, _, info2 = self.env.step(np.zeros(4, dtype=np.float32))
        self.assertEqual(info2["prev_score"], 10.0)
        self.assertEqual(info2["score"], 15.0)


class TestSC2EnvCustomReward(unittest.TestCase):

    def test_custom_reward_config(self):
        with patch("games.sc2.env.SC2Client") as mock_cls:
            mock_cls.return_value.reset.return_value = (
                np.zeros(BASE_OBS_DIM, dtype=np.float32),
                {"score": 0.0, "minerals": 0.0, "vespene": 0.0,
                 "food_used": 0.0, "food_cap": 0.0, "army_count": 0.0},
            )
            mock_cls.return_value.step.return_value = (
                np.zeros(BASE_OBS_DIM, dtype=np.float32),
                0.0,
                False,
                {"score": 5.0, "minerals": 0.0, "vespene": 0.0,
                 "food_used": 0.0, "food_cap": 0.0, "army_count": 0.0},
            )
            cfg = SC2RewardConfig(score_weight=10.0, step_penalty=0.0)
            env = SC2Env(map_name="MoveToBeacon", reward_config=cfg)
            env.reset()
            _, reward, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
            # 5 * 10 = 50.
            self.assertAlmostEqual(reward, 50.0)


if __name__ == "__main__":
    unittest.main()
