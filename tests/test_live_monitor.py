"""Tests for framework.live_monitor helper logic."""

from __future__ import annotations

import unittest
from collections import deque

from framework.live_monitor import (
    _classify_observation_features,
    _derive_step_components,
    _rolling_means,
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


if __name__ == "__main__":
    unittest.main()
