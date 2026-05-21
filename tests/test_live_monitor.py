"""Tests for framework.live_monitor helper logic."""

from __future__ import annotations

import unittest
from collections import deque

from framework.live_monitor import (
    _classify_observation_features,
    _derive_step_components,
    _fmt_action,
    _rolling_means,
    _reward_sort_key,
)


class TestRewardComponents(unittest.TestCase):
    def test_derive_from_step_components(self):
        info = {"step_reward_components": {"progress": 1.5, "speed": 0.2}}
        step, current = _derive_step_components(info, reward=2.0, prev_episode_components=None)
        self.assertEqual(step["progress"], 1.5)
        self.assertEqual(step["speed"], 0.2)
        self.assertEqual(step["total_reward"], 2.0)
        self.assertEqual(current, {})

    def test_derive_from_episode_cumulative(self):
        prev = {"progress": 10.0, "speed": 2.0}
        info = {"episode_reward_components": {"progress": 12.5, "speed": 2.2}}
        step, current = _derive_step_components(info, reward=2.7, prev_episode_components=prev)
        self.assertAlmostEqual(step["progress"], 2.5)
        self.assertAlmostEqual(step["speed"], 0.2)
        self.assertAlmostEqual(step["total_reward"], 2.7)
        self.assertEqual(current, {"progress": 12.5, "speed": 2.2})

    def test_rolling_means_window(self):
        history: dict[str, deque[float]] = {}
        for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
            means = _rolling_means(history, {"total_reward": value}, window=5)
        self.assertAlmostEqual(means["total_reward"], 3.0)
        means = _rolling_means(history, {"total_reward": 11.0}, window=5)
        self.assertAlmostEqual(means["total_reward"], (2.0 + 3.0 + 4.0 + 5.0 + 11.0) / 5.0)


class TestObservationClassification(unittest.TestCase):
    def test_classifies_xy_indexed_and_quads(self):
        names = [
            "move_x", "move_y",
            "lidar_0", "lidar_1", "lidar_2",
            "screen_enemy_NE", "screen_enemy_NW", "screen_enemy_SE", "screen_enemy_SW",
            "speed_ms",
        ]
        groups = _classify_observation_features(names)
        self.assertIn(("move", "move_x", "move_y"), groups.xy_pairs)
        self.assertIn("lidar", [base for base, _ in groups.indexed])
        self.assertIn("screen_enemy", [base for base, _ in groups.quads])
        speed_idx = names.index("speed_ms")
        self.assertIn(speed_idx, groups.scalar_idxs)


class TestDisplayHelpers(unittest.TestCase):
    def test_reward_sort_key_keeps_known_order_and_sorts_unknowns_alphabetically(self):
        names = ["z_bonus", "step_penalty", "a_bonus", "progress_weight"]
        self.assertEqual(
            sorted(names, key=_reward_sort_key),
            ["progress_weight", "step_penalty", "a_bonus", "z_bonus"],
        )

    def test_fmt_action_truncates_long_vectors_after_six_values(self):
        self.assertEqual(
            _fmt_action([1, 2, 3, 4, 5, 6, 7]),
            "[+1.00, +2.00, +3.00, +4.00, +5.00, +6.00…]",
        )

    def test_fmt_action_formats_tmnf_style_controls(self):
        self.assertEqual(
            _fmt_action([-0.5, 1.0, 0.25]),
            "accel 100% / brake 25% | steer left 50%",
        )

    def test_fmt_action_formats_straight_steer(self):
        self.assertEqual(
            _fmt_action([0.0, 0.5, 0.0]),
            "accel 50% | steer straight",
        )

    def test_fmt_action_formats_brake_only_controls(self):
        self.assertEqual(
            _fmt_action([0.25, 0.0, 0.8]),
            "brake 80% | steer right 25%",
        )

    def test_fmt_action_treats_small_pedal_values_as_effectively_zero(self):
        self.assertEqual(
            _fmt_action([0.0, 0.01, 0.8]),
            "brake 80% | steer straight",
        )
        self.assertEqual(
            _fmt_action([0.0, 0.8, 0.01]),
            "accel 80% | steer straight",
        )


if __name__ == "__main__":
    unittest.main()
