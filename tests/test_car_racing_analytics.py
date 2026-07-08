"""Tests for CarRacing-specific analytics output (issue #482 follow-up).

Covers the "Reward Moving Average" section/plot added so a SAC/PPO run's
average reward can be checked against CarRacing-v2's published "solved"
benchmark (average reward >= 900 over 100 consecutive episodes) without
manually computing a rolling mean.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from framework.analytics import ExperimentData, GreedySimResult
from games.car_racing.analytics import (
    SOLVED_REWARD_THRESHOLD,
    SOLVED_WINDOW_EPISODES,
    save_experiment_results,
)


def _make_data(rewards: list[float]) -> ExperimentData:
    greedy_sims = [
        GreedySimResult(
            sim=i + 1,
            reward=r,
            improved=(i == 0 or r > max(rewards[:i], default=float("-inf"))),
            throttle_counts=[5, 10, 30],
            total_steps=200,
        )
        for i, r in enumerate(rewards)
    ]
    return ExperimentData(
        experiment_name="car-racing-smoke",
        probe_results=[],
        cold_start_restarts=[],
        greedy_sims=greedy_sims,
        probe_floor=None,
        weights_file="policy_weights.yaml",
        reward_config_file="reward_config.yaml",
        training_params={},
        timings={
            "start": "s",
            "end": "e",
            "total_s": 1.0,
            "probe_s": None,
            "cold_start_s": None,
            "greedy_s": 1.0,
        },
    )


class TestSolvedBenchmarkConstants(unittest.TestCase):
    def test_matches_published_benchmark(self):
        self.assertEqual(SOLVED_REWARD_THRESHOLD, 900.0)
        self.assertEqual(SOLVED_WINDOW_EPISODES, 100)


class TestCarRacingSaveExperimentResults(unittest.TestCase):
    def test_report_includes_reward_moving_average_section(self):
        data = _make_data([100.0, 200.0, 300.0])
        with tempfile.TemporaryDirectory() as tmp:
            save_experiment_results(data, tmp)
            report = Path(tmp, "results.md").read_text(encoding="utf-8")
            self.assertIn("## Reward Moving Average", report)
            self.assertIn("reward_moving_average.png", report)
            self.assertIn("not yet solved", report)
            self.assertTrue(Path(tmp, "reward_moving_average.png").exists())

    def test_reports_solved_when_threshold_met(self):
        data = _make_data([950.0] * 5)
        with tempfile.TemporaryDirectory() as tmp:
            save_experiment_results(data, tmp)
            report = Path(tmp, "results.md").read_text(encoding="utf-8")
            self.assertIn("**solved**", report)

    def test_no_moving_average_section_without_greedy_sims(self):
        data = _make_data([])
        with tempfile.TemporaryDirectory() as tmp:
            save_experiment_results(data, tmp)
            report = Path(tmp, "results.md").read_text(encoding="utf-8")
            self.assertNotIn("Reward Moving Average", report)


if __name__ == "__main__":
    unittest.main()
