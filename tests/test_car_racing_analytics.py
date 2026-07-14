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

from framework.analytics import ExperimentData, GreedySimResult, RunTrace
from games.car_racing.analytics import (
    SOLVED_REWARD_THRESHOLD,
    SOLVED_WINDOW_EPISODES,
    plot_action_histograms,
    plot_speed_trace,
    save_experiment_results,
)


def _make_trace(reward: float, n_steps: int = 20, with_speed: bool = False) -> RunTrace:
    return RunTrace(
        pos_x=[],
        pos_z=[],
        throttle_state=[(0.8, 0.0, 0.3)] * n_steps,
        total_reward=reward,
        speed_trace=[float(i) for i in range(n_steps)] if with_speed else [],
    )


def _make_data(rewards: list[float], with_trace: bool = False, with_speed: bool = False) -> ExperimentData:
    greedy_sims = [
        GreedySimResult(
            sim=i + 1,
            reward=r,
            improved=(i == 0 or r > max(rewards[:i], default=float("-inf"))),
            throttle_counts=[5, 10, 30],
            total_steps=200,
            trace=_make_trace(r, with_speed=with_speed) if with_trace else None,
            termination_reason="finish" if i % 2 == 0 else "crash",
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

    def test_report_includes_action_and_termination_plots(self):
        data = _make_data([100.0, 200.0, 300.0], with_trace=True, with_speed=True)
        with tempfile.TemporaryDirectory() as tmp:
            save_experiment_results(data, tmp)
            report = Path(tmp, "results.md").read_text(encoding="utf-8")
            self.assertIn("greedy_action_dist.png", report)
            self.assertIn("greedy_best_run.png", report)
            self.assertIn("termination_reasons.png", report)
            self.assertIn("action_histograms.png", report)
            self.assertIn("speed_trace.png", report)
            for fname in (
                "greedy_action_dist.png",
                "greedy_best_run.png",
                "termination_reasons.png",
                "action_histograms.png",
                "speed_trace.png",
            ):
                self.assertTrue(Path(tmp, fname).exists(), fname)

    def test_report_omits_speed_and_histogram_plots_without_trace_data(self):
        data = _make_data([100.0, 200.0], with_trace=False)
        with tempfile.TemporaryDirectory() as tmp:
            save_experiment_results(data, tmp)
            report = Path(tmp, "results.md").read_text(encoding="utf-8")
            self.assertNotIn("action_histograms.png", report)
            self.assertNotIn("speed_trace.png", report)
            self.assertNotIn("greedy_best_run.png", report)
            self.assertFalse(Path(tmp, "action_histograms.png").exists())
            self.assertFalse(Path(tmp, "speed_trace.png").exists())
            self.assertFalse(Path(tmp, "greedy_best_run.png").exists())


class TestPlotActionHistograms(unittest.TestCase):
    def test_creates_file_with_trace(self):
        data = _make_data([10.0, 20.0], with_trace=True)
        with tempfile.TemporaryDirectory() as tmp:
            plot_action_histograms(data, tmp)
            self.assertTrue(Path(tmp, "action_histograms.png").exists())

    def test_no_crash_without_trace(self):
        data = _make_data([10.0, 20.0], with_trace=False)
        with tempfile.TemporaryDirectory() as tmp:
            plot_action_histograms(data, tmp)
            self.assertFalse(Path(tmp, "action_histograms.png").exists())

    def test_no_crash_with_no_sims(self):
        data = _make_data([])
        with tempfile.TemporaryDirectory() as tmp:
            plot_action_histograms(data, tmp)
            self.assertFalse(Path(tmp, "action_histograms.png").exists())


class TestPlotSpeedTrace(unittest.TestCase):
    def test_creates_file_with_speed_data(self):
        data = _make_data([10.0, 20.0], with_trace=True, with_speed=True)
        with tempfile.TemporaryDirectory() as tmp:
            plot_speed_trace(data, tmp)
            self.assertTrue(Path(tmp, "speed_trace.png").exists())

    def test_no_crash_without_speed_data(self):
        data = _make_data([10.0, 20.0], with_trace=True, with_speed=False)
        with tempfile.TemporaryDirectory() as tmp:
            plot_speed_trace(data, tmp)
            self.assertFalse(Path(tmp, "speed_trace.png").exists())


if __name__ == "__main__":
    unittest.main()
