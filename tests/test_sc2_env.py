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
        self.mock_client.last_fn_idx = 0
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
        should produce a reward close to +1 minus step_penalty scaled by
        step_mul (default 1)."""
        self.env.reset()
        _, reward, _, _, _ = self.env.step(np.zeros(4, dtype=np.float32))
        # score went 0 -> 1, default step_penalty -0.001 × 1 tick → reward ≈ 0.999.
        self.assertAlmostEqual(reward, 1.0 - 0.001 * 1, places=5)

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
                     "termination_reason", "skipped_frames_this_step",
                     "episode_skipped_frames"):
            self.assertIn(key, info, f"missing key {key}")

    def test_skipped_frames_default_zero_without_game_loop(self):
        """Missing game_loop telemetry should keep skipped-frame counters at zero."""
        self.env.reset()
        _, _, _, _, info = self.env.step(np.zeros(4, dtype=np.float32))
        self.assertEqual(info["skipped_frames_this_step"], 0)
        self.assertEqual(info["episode_skipped_frames"], 0)

    def test_skipped_frames_accumulate_from_game_loop_delta(self):
        """Delta above step_mul counts as skipped frames and accumulates."""
        self.mock_client.reset.return_value = (
            np.zeros(BASE_OBS_DIM, dtype=np.float32),
            {"score": 0.0, "minerals": 50.0, "vespene": 0.0,
             "food_used": 1.0, "food_cap": 15.0, "army_count": 0.0,
             "game_loop": 0.0},
        )
        self.mock_client.step.side_effect = [
            (
                np.zeros(BASE_OBS_DIM, dtype=np.float32),
                0.0,
                False,
                {"score": 1.0, "minerals": 50.0, "vespene": 0.0,
                 "food_used": 1.0, "food_cap": 15.0, "army_count": 0.0,
                 "game_loop": 1.0},   # delta == step_mul → 0 skipped
            ),
            (
                np.zeros(BASE_OBS_DIM, dtype=np.float32),
                0.0,
                False,
                {"score": 2.0, "minerals": 50.0, "vespene": 0.0,
                 "food_used": 1.0, "food_cap": 15.0, "army_count": 0.0,
                 "game_loop": 6.0},   # delta 5 = step_mul 1 + 4 skipped
            ),
        ]
        self.env.reset()
        _, _, _, _, info1 = self.env.step(np.zeros(4, dtype=np.float32))
        _, _, _, _, info2 = self.env.step(np.zeros(4, dtype=np.float32))
        self.assertEqual(info1["skipped_frames_this_step"], 0)
        self.assertEqual(info1["episode_skipped_frames"], 0)
        self.assertEqual(info2["skipped_frames_this_step"], 4)
        self.assertEqual(info2["episode_skipped_frames"], 4)

    def test_step_info_carries_reward_components(self):
        """Issue #128/2b: env populates info['episode_reward_components']."""
        self.env.reset()
        _, _, _, _, info = self.env.step(np.array([0, 0.5, 0.5, 0], dtype=np.float32))
        self.assertIn("episode_reward_components", info)
        comp = info["episode_reward_components"]
        for key in ("score", "economy", "idle_penalty", "idle_bonus",
                    "step_penalty", "terminal"):
            self.assertIn(key, comp)

    def test_reward_components_accumulate_across_steps(self):
        """Repeated steps add their per-step contributions to the totals."""
        self.env.reset()
        action = np.array([0, 0.5, 0.5, 0], dtype=np.float32)
        _, _, _, _, info1 = self.env.step(action)
        _, _, _, _, info2 = self.env.step(action)
        # step_penalty is non-zero by default; second step's accumulator must
        # be at least as negative as the first step's contribution.
        self.assertLessEqual(
            info2["episode_reward_components"]["step_penalty"],
            info1["episode_reward_components"]["step_penalty"],
        )

    def test_reward_components_reset_on_episode_start(self):
        """reset() clears the accumulator so each episode starts fresh."""
        self.env.reset()
        action = np.array([0, 0.5, 0.5, 0], dtype=np.float32)
        self.env.step(action)
        self.env.step(action)
        # Now reset and confirm a single step's totals are smaller than
        # accumulated two steps' worth.
        self.env.reset()
        _, _, _, _, info = self.env.step(action)
        self.assertAlmostEqual(
            info["episode_reward_components"]["step_penalty"],
            self.env._reward_config.step_penalty * self.env._step_mul,
        )

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

    def test_prev_army_and_total_self_hp_threaded_through(self):
        """prev_army_count / prev_total_self_hp seed from reset then update per step."""
        self.mock_client.reset.return_value = (
            np.zeros(BASE_OBS_DIM, dtype=np.float32),
            {"score": 0.0, "minerals": 50.0, "vespene": 0.0,
             "food_used": 1.0, "food_cap": 15.0,
             "army_count": 4.0, "total_self_hp": 110.0},
        )
        self.mock_client.step.side_effect = [
            (
                np.zeros(BASE_OBS_DIM, dtype=np.float32),
                0.0,
                False,
                {"score": 1.0, "minerals": 50.0, "vespene": 0.0,
                 "food_used": 1.0, "food_cap": 15.0,
                 "army_count": 3.0, "total_self_hp": 90.0},
            ),
            (
                np.zeros(BASE_OBS_DIM, dtype=np.float32),
                0.0,
                False,
                {"score": 2.0, "minerals": 50.0, "vespene": 0.0,
                 "food_used": 1.0, "food_cap": 15.0,
                 "army_count": 2.0, "total_self_hp": 70.0},
            ),
        ]
        self.env.reset()

        _, _, _, _, info1 = self.env.step(np.zeros(4, dtype=np.float32))
        self.assertEqual(info1["prev_army_count"], 4.0)
        self.assertEqual(info1["army_count"], 3.0)
        self.assertEqual(info1["prev_total_self_hp"], 110.0)
        self.assertEqual(info1["total_self_hp"], 90.0)

        _, _, _, _, info2 = self.env.step(np.zeros(4, dtype=np.float32))
        self.assertEqual(info2["prev_army_count"], 3.0)
        self.assertEqual(info2["army_count"], 2.0)
        self.assertEqual(info2["prev_total_self_hp"], 90.0)
        self.assertEqual(info2["total_self_hp"], 70.0)

    def test_action_fn_idx_uses_executed_client_action(self):
        """Blocked Attack_screen requests should not suppress passive-under-fire."""
        self.mock_client.reset.return_value = (
            np.zeros(BASE_OBS_DIM, dtype=np.float32),
            {"score": 0.0, "minerals": 50.0, "vespene": 0.0,
             "food_used": 1.0, "food_cap": 15.0, "army_count": 1.0},
        )

        def _step_with_substituted_select_army(_action):
            self.mock_client.last_fn_idx = 1
            return (
                np.zeros(BASE_OBS_DIM, dtype=np.float32),
                0.0,
                False,
                {"score": 0.0, "minerals": 50.0, "vespene": 0.0,
                 "food_used": 1.0, "food_cap": 15.0, "army_count": 1.0,
                 "screen_self_count": 1.0, "screen_enemy_count": 1.0,
                 "screen_self_cx": 32.0, "screen_self_cy": 32.0,
                 "screen_enemy_cx": 42.0, "screen_enemy_cy": 32.0,
                 "self_attack_range_px": 20.0},
            )

        self.mock_client.step.side_effect = _step_with_substituted_select_army
        cfg = SC2RewardConfig(
            score_weight=0.0,
            step_penalty=0.0,
            win_bonus=0.0,
            loss_penalty=0.0,
            economy_weight=0.0,
            move_exploration_bonus=0.0,
            move_repeat_penalty=0.0,
            move_self_penalty=0.0,
            attack_move_bonus=0.0,
            click_attack_bonus=0.0,
            attack_friendly_penalty=0.0,
            passive_under_fire_penalty=-2.0,
        )
        self.env = SC2Env(map_name="MoveToBeacon", reward_config=cfg)

        self.env.reset()
        attack = np.array([3, 0.5, 0.5, 0], dtype=np.float32)
        _, reward, _, _, info = self.env.step(attack)
        self.assertEqual(info["action_fn_idx"], 1)
        self.assertAlmostEqual(info["episode_reward_components"]["passive_under_fire"], -2.0)
        self.assertAlmostEqual(reward, -2.0)


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


class TestSC2EnvEndScreenAnalytics(unittest.TestCase):
    """Tests for the supply-cap / time-series / build-order analytics added
    to SC2Env.step() info at end-of-episode."""

    def _make_env(self, reset_info=None, step_info=None, done=False):
        """Return a mocked SC2Env wired up with the given reset/step info."""
        patcher = patch("games.sc2.env.SC2Client")
        mock_cls = patcher.start()
        self.addCleanup(patcher.stop)

        _base_reset = {
            "score": 0.0, "minerals": 100.0, "vespene": 0.0,
            "food_used": 6.0, "food_cap": 15.0, "army_count": 0.0,
        }
        _base_step = {
            "score": 1.0, "minerals": 150.0, "vespene": 50.0,
            "food_used": 8.0, "food_cap": 15.0, "army_count": 2.0,
            "game_loop": 100.0,
        }
        mock_client = mock_cls.return_value
        mock_client.last_fn_idx = 0
        mock_client.reset.return_value = (
            np.zeros(BASE_OBS_DIM, dtype=np.float32),
            dict(_base_reset, **(reset_info or {})),
        )
        mock_client.step.return_value = (
            np.zeros(BASE_OBS_DIM, dtype=np.float32),
            0.0,
            done,
            dict(_base_step, **(step_info or {})),
        )
        env = SC2Env(map_name="MoveToBeacon")
        return env

    # ---------------------------------------------------------------
    # Series not emitted on non-terminal steps
    # ---------------------------------------------------------------

    def test_series_absent_on_mid_episode_step(self):
        """episode_army_series / resource_series / build_order / supply_capped
        must NOT be in info for non-terminal steps to keep update() lightweight."""
        env = self._make_env(done=False)
        env.reset()
        _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        for key in ("episode_army_series", "episode_resource_series",
                    "episode_build_order", "episode_supply_capped_fraction"):
            self.assertNotIn(key, info, f"{key} should not appear on mid-episode step")

    # ---------------------------------------------------------------
    # Series emitted on terminal step
    # ---------------------------------------------------------------

    def test_series_present_on_terminal_step(self):
        """All four end-screen fields appear in info when the episode ends."""
        env = self._make_env(done=True, step_info={"is_last": True})
        env.reset()
        _, _, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
        self.assertTrue(terminated)
        for key in ("episode_army_series", "episode_resource_series",
                    "episode_build_order", "episode_supply_capped_fraction"):
            self.assertIn(key, info, f"{key} missing from terminal step info")

    # ---------------------------------------------------------------
    # supply_capped_fraction
    # ---------------------------------------------------------------

    def test_supply_capped_fraction_when_capped(self):
        """All steps at food_cap → supply_capped_fraction == 1.0."""
        env = self._make_env(
            done=True,
            step_info={"food_used": 15.0, "food_cap": 15.0, "is_last": True},
        )
        env.reset()
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        self.assertAlmostEqual(info["episode_supply_capped_fraction"], 1.0)

    def test_supply_capped_fraction_when_not_capped(self):
        """food_used < food_cap → supply_capped_fraction == 0.0."""
        env = self._make_env(
            done=True,
            step_info={"food_used": 8.0, "food_cap": 15.0, "is_last": True},
        )
        env.reset()
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        self.assertAlmostEqual(info["episode_supply_capped_fraction"], 0.0)

    # ---------------------------------------------------------------
    # army / resource series
    # ---------------------------------------------------------------

    def test_army_series_contains_step_data(self):
        """Army series has one entry per step with [game_time_s, army_count]."""
        env = self._make_env(
            done=True,
            step_info={"army_count": 5.0, "game_loop": 224.0, "is_last": True},
        )
        env.reset()
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        army = info["episode_army_series"]
        self.assertEqual(len(army), 1)
        self.assertAlmostEqual(army[0][1], 5.0)

    def test_resource_series_sums_minerals_and_vespene(self):
        """Resource series value == minerals + vespene."""
        env = self._make_env(
            done=True,
            step_info={"minerals": 100.0, "vespene": 50.0,
                       "game_loop": 0.0, "is_last": True},
        )
        env.reset()
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        res = info["episode_resource_series"]
        self.assertAlmostEqual(res[0][1], 150.0)

    # ---------------------------------------------------------------
    # Build order: starting units excluded
    # ---------------------------------------------------------------

    def test_starting_units_not_in_build_order(self):
        """Units present in reset info are NOT counted as build-order events."""
        # 6 SCVs present at reset; step returns same 6 → no "SCV built" event.
        env = self._make_env(
            reset_info={"unit_counts": {"SCV": 6.0}},
            done=True,
            step_info={"unit_counts": {"SCV": 6.0}, "is_last": True},
        )
        env.reset()
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        events = info["episode_build_order"]
        self.assertEqual(events, [], "Starting SCVs must not appear in build order")

    def test_new_unit_produces_build_order_event(self):
        """A unit-count increase after reset IS recorded as a build-order event."""
        # Reset: 6 SCVs; step: 7 SCVs → one "SCV" event.
        env = self._make_env(
            reset_info={"unit_counts": {"SCV": 6.0}},
            done=True,
            step_info={"unit_counts": {"SCV": 7.0}, "is_last": True},
        )
        env.reset()
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        events = info["episode_build_order"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][1], "SCV")

    def test_build_order_empty_when_no_unit_counts(self):
        """If the client never provides unit_counts, build_order stays empty."""
        env = self._make_env(done=True, step_info={"is_last": True})
        env.reset()
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        self.assertEqual(info["episode_build_order"], [])


if __name__ == "__main__":
    unittest.main()
