"""Unit tests for _log_new_best_details and _print_episode_summary."""

import logging
import os
import sys
import unittest

_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from framework.training import _log_new_best_details, _print_episode_summary

_TRAINING_LOGGER = logging.getLogger("framework.training")


def _capture_training_logs(fn):
    """Capture log records emitted by framework.training during fn."""
    records = []

    class _Cap(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    cap = _Cap()
    cap.setLevel(logging.DEBUG)
    old_level = _TRAINING_LOGGER.level
    _TRAINING_LOGGER.setLevel(logging.DEBUG)
    _TRAINING_LOGGER.addHandler(cap)
    try:
        fn()
    finally:
        _TRAINING_LOGGER.removeHandler(cap)
        _TRAINING_LOGGER.setLevel(old_level)
    return records


class TestPrintEpisodeSummary(unittest.TestCase):
    def test_terminated_episode_short(self):
        info = {"track_progress": 0.5, "finished": False}
        lines = _capture_training_logs(
            lambda: _print_episode_summary(info, steps=68, total_reward=-1.8, truncated=False)
        )
        self.assertEqual(len(lines), 1)
        self.assertIn("terminated", lines[0])
        self.assertIn("r=", lines[0])
        self.assertIn("steps=", lines[0])

    def test_finished_episode_short(self):
        info = {"finished": True}
        lines = _capture_training_logs(
            lambda: _print_episode_summary(info, steps=100, total_reward=50.0, truncated=False)
        )
        self.assertEqual(len(lines), 1)
        self.assertIn("finished", lines[0])

    def test_truncated_episode_short(self):
        info = {}
        lines = _capture_training_logs(
            lambda: _print_episode_summary(info, steps=200, total_reward=10.0, truncated=True)
        )
        self.assertEqual(len(lines), 1)
        self.assertIn("truncated", lines[0])

    def test_no_laps_or_progress_in_log(self):
        info = {"track_progress": 0.75, "laps_completed": 2, "finished": False}
        lines = _capture_training_logs(
            lambda: _print_episode_summary(info, steps=50, total_reward=5.0, truncated=False)
        )
        self.assertEqual(len(lines), 1)
        self.assertNotIn("laps", lines[0])
        self.assertNotIn("progress", lines[0])

    def test_sc2_outcome_reward_score_logged_as_scalars(self):
        info = {"player_outcome": 1, "raw_reward": 1, "score": 0, "episode_skipped_frames": 3}
        lines = _capture_training_logs(
            lambda: _print_episode_summary(info, steps=240, total_reward=-30.1, truncated=False)
        )
        self.assertEqual(len(lines), 1)
        self.assertIn("outcome=win", lines[0])
        self.assertIn("reward=+1.0", lines[0])
        self.assertIn("score=+0.0", lines[0])
        self.assertIn("skipped_frames=3", lines[0])


class TestLogNewBestDetails(unittest.TestCase):
    def test_empty_info_emits_nothing(self):
        lines = _capture_training_logs(lambda: _log_new_best_details({}, None))
        self.assertEqual(lines, [])

    def test_reward_components_logged(self):
        info = {"episode_reward_components": {"score": 5.0, "idle_bonus": 1.2}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        # score and idle_bonus — no terminal event so win/loss not shown
        self.assertEqual(len(lines), 2)
        all_text = "\n".join(lines)
        self.assertIn("score=", all_text)
        self.assertIn("idle_bonus=", all_text)
        self.assertNotIn("win_bonus=", all_text)
        self.assertNotIn("loss_penalty=", all_text)

    def test_reward_components_zero_omitted(self):
        info = {"episode_reward_components": {"score": 5.0, "economy": 0.0}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        # score only — economy=0 and no terminal event suppress the rest
        self.assertEqual(len(lines), 1)
        self.assertNotIn("economy", lines[0])

    def test_reward_components_prev_comparison(self):
        info = {"episode_reward_components": {"score": 10.0, "idle_bonus": 2.0, "terminal": 0.0}}
        prev = {"episode_reward_components": {"score": 5.0, "idle_bonus": 1.0, "terminal": 0.0}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, prev))
        # terminal=0 in both episodes → win/loss both 0 → neither shown
        self.assertEqual(len(lines), 2)
        all_text = "\n".join(lines)
        self.assertIn("score=+10.0 (prev +5.0)", all_text)
        self.assertIn("idle_bonus=+2.0 (prev +1.0)", all_text)
        self.assertNotIn("win_bonus=", all_text)
        self.assertNotIn("loss_penalty=", all_text)

    def test_reward_components_no_prev_no_comparison(self):
        info = {"episode_reward_components": {"score": 10.0}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        # score only — no terminal event so win/loss not shown
        self.assertEqual(len(lines), 1)
        self.assertIn("score=+10.0", lines[0])

    def test_score_zero_is_still_logged(self):
        info = {"episode_reward_components": {"score": 0.0, "move_self_penalty": -1.0}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        all_text = "\n".join(lines)
        self.assertIn("score=+0.0", all_text)

    def test_terminal_split_into_win_and_loss_components(self):
        info = {
            "episode_reward_components": {"terminal": 100.0, "score": 10.0},
            "player_outcome": 1.0,
        }
        prev = {
            "episode_reward_components": {"terminal": -100.0, "score": 5.0},
            "player_outcome": -1.0,
        }
        lines = _capture_training_logs(lambda: _log_new_best_details(info, prev))
        all_text = "\n".join(lines)
        self.assertIn("score=+10.0 (prev +5.0)", all_text)
        # Current episode is a win (terminal=100) → win_bonus shown; loss_penalty=0 → not shown
        self.assertIn("win_bonus=+100.0 (prev +0.0)", all_text)
        self.assertNotIn("loss_penalty=", all_text)

    def test_win_bonus_shown_only_when_positive(self):
        info = {"episode_reward_components": {"score": 5.0, "terminal": 100.0}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        all_text = "\n".join(lines)
        self.assertIn("win_bonus=+100.0", all_text)
        self.assertNotIn("loss_penalty=", all_text)

    def test_loss_penalty_shown_only_when_negative(self):
        info = {"episode_reward_components": {"score": 0.0, "terminal": -100.0}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        all_text = "\n".join(lines)
        self.assertIn("loss_penalty=-100.0", all_text)
        self.assertNotIn("win_bonus=", all_text)

    def test_no_terminal_hides_win_and_loss(self):
        info = {"episode_reward_components": {"score": 8.0, "terminal": 0.0}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        all_text = "\n".join(lines)
        self.assertNotIn("win_bonus=", all_text)
        self.assertNotIn("loss_penalty=", all_text)

    def test_action_counts_logged(self):
        info = {"episode_action_counts": {0: 30, 1: 10, 2: 60}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        # one log line per action (sorted by descending count: fn2, fn0, fn1)
        self.assertEqual(len(lines), 3)
        all_text = "\n".join(lines)
        # Without a name map the key is stringified: "2=60.0%".
        self.assertIn("2=60.0%", all_text)

    def test_action_counts_uses_name_map(self):
        info = {
            "episode_action_counts": {0: 30, 3: 70},
            "episode_action_name_map": {0: "no_op", 3: "Attack_screen"},
        }
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        all_text = "\n".join(lines)
        self.assertIn("Attack_screen=70.0%", all_text)
        self.assertIn("no_op=30.0%", all_text)
        # Raw integer keys must not appear when a name map is present.
        self.assertNotIn("3=", all_text)

    def test_action_counts_prev_comparison(self):
        info = {"episode_action_counts": {0: 10, 2: 90}}
        prev = {"episode_action_counts": {0: 50, 2: 50}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, prev))
        self.assertEqual(len(lines), 2)
        all_text = "\n".join(lines)
        self.assertIn("prev", all_text)

    def test_tmnf_progress_logged(self):
        info = {"episode_task_metrics": {"progress": "75.0%"}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        self.assertEqual(len(lines), 1)
        self.assertIn("progress=75.0%", lines[0])

    def test_tmnf_lateral_offset_logged(self):
        info = {"episode_task_metrics": {"progress": "50.0%", "mean_lateral": "1.23m"}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        self.assertEqual(len(lines), 1)
        self.assertIn("mean_lateral=1.23m", lines[0])

    def test_tmnf_finish_time_logged_when_finished(self):
        info = {"episode_task_metrics": {"progress": "100.0%", "finish_time": "42.5s"}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        self.assertEqual(len(lines), 1)
        self.assertIn("finish_time=42.5s", lines[0])

    def test_tmnf_finish_time_not_logged_when_not_finished(self):
        # TMNF only includes "finish_time" when finished; test that absence is honoured.
        info = {"episode_task_metrics": {"progress": "80.0%"}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        self.assertNotIn("finish_time", lines[0])

    def test_tmnf_prev_comparison(self):
        info = {"episode_task_metrics": {"progress": "90.0%", "mean_lateral": "0.50m"}}
        prev = {"episode_task_metrics": {"progress": "70.0%", "mean_lateral": "0.80m"}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, prev))
        self.assertEqual(len(lines), 1)
        self.assertIn("(prev 70.0%)", lines[0])
        self.assertIn("(prev 0.80m)", lines[0])

    def test_kill_stats_logged(self):
        info = {
            "episode_killed_value_units": 250.0,
            "episode_killed_value_structures": 100.0,
        }
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        self.assertEqual(len(lines), 1)
        self.assertIn("kills:", lines[0])
        self.assertIn("units=250", lines[0])
        self.assertIn("structures=100", lines[0])

    def test_kill_stats_prev_comparison(self):
        info = {"episode_killed_value_units": 300.0, "episode_killed_value_structures": 0.0}
        prev = {"episode_killed_value_units": 200.0, "episode_killed_value_structures": 0.0}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, prev))
        self.assertEqual(len(lines), 1)
        self.assertIn("(prev 200)", lines[0])

    def test_kill_stats_absent_when_not_in_info(self):
        info = {"episode_reward_components": {"score": 5.0}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        for line in lines:
            self.assertNotIn("kills:", line)

    def test_kill_stats_suppressed_when_both_zero(self):
        """kills: line not emitted when both units and structures killed are zero."""
        info = {
            "episode_killed_value_units": 0.0,
            "episode_killed_value_structures": 0.0,
        }
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        for line in lines:
            self.assertNotIn("kills:", line)

    def test_game_state_averages_logged(self):
        info = {"episode_obs_averages": {"army_count": 4.5, "food_used": 10.0, "screen_enemy_count": 3.2}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        # one log line per non-zero metric (army_count, food_used, screen_enemy_count)
        self.assertEqual(len(lines), 3)
        all_text = "\n".join(lines)
        self.assertIn("army_count=4.5", all_text)

    def test_game_state_zero_values_omitted(self):
        info = {"episode_obs_averages": {"army_count": 0.0, "screen_enemy_count": 0.0}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        self.assertEqual(lines, [])

    def test_game_state_prev_comparison(self):
        info = {"episode_obs_averages": {"army_count": 5.0}}
        prev = {"episode_obs_averages": {"army_count": 3.0}}
        lines = _capture_training_logs(lambda: _log_new_best_details(info, prev))
        self.assertEqual(len(lines), 1)
        self.assertIn("(prev 3.0)", lines[0])

    def test_all_groups_emit_multiple_lines(self):
        info = {
            "episode_reward_components": {"score": 5.0, "idle_bonus": 1.0},
            "episode_action_counts": {0: 20, 2: 80},
            "episode_task_metrics": {"progress": "60.0%"},
            "episode_killed_value_units": 100.0,
            "episode_killed_value_structures": 0.0,
            "episode_obs_averages": {"army_count": 3.0},
        }
        lines = _capture_training_logs(lambda: _log_new_best_details(info, None))
        # 2 reward components + 2 actions + 1 task metric + 1 kills + 1 game-state = 7
        # (no terminal event → win/loss both 0 → not shown)
        self.assertEqual(len(lines), 7)


if __name__ == "__main__":
    unittest.main()
