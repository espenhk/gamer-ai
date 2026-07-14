"""Tests for Rocket League-specific analytics (issue #466).

Exercises each RL plot function and the results.md report with synthetic
ExperimentData — no Rocket League binary or rlgym install needed.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from framework.analytics import ExperimentData, GreedySimResult
from games.rocket_league.analytics import (
    plot_ball_pursuit,
    plot_ball_touches,
    plot_boost_usage,
    plot_match_results,
    save_experiment_results,
)


def _obs_averages(boost: float = 0.4, touches: float = 3.0) -> dict:
    return {
        "boost_amount": boost,
        "dist_to_ball": 2000.0,
        "vel_towards_ball": 250.0,
        "ball_touches": touches,
    }


_REASONS = ["goal_scored", "goal_conceded", "timeout", "goal_scored"]


def _make_data(n_sims: int = 4, with_obs_averages: bool = True) -> ExperimentData:
    greedy_sims = [
        GreedySimResult(
            sim=i + 1,
            reward=10.0 * (i + 1),
            improved=True,
            throttle_counts=[0, 0, 0],
            total_steps=300,
            termination_reason=_REASONS[i % len(_REASONS)],
            obs_averages=_obs_averages(boost=0.3 + 0.1 * i, touches=float(i)) if with_obs_averages else None,
        )
        for i in range(n_sims)
    ]
    return ExperimentData(
        experiment_name="rl-analytics-test",
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


class TestPlotMatchResults(unittest.TestCase):
    def test_writes_plot_from_termination_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(plot_match_results(_make_data(), tmp))
            self.assertTrue(Path(tmp, "rl_match_results.png").exists())

    def test_skips_without_greedy_sims(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(plot_match_results(_make_data(n_sims=0), tmp))
            self.assertFalse(Path(tmp, "rl_match_results.png").exists())

    def test_unknown_reason_counts_as_draw(self):
        data = _make_data()
        for s in data.greedy_sims:
            s.termination_reason = "something_else"
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(plot_match_results(data, tmp))


class TestPlotBallTouches(unittest.TestCase):
    def test_writes_plot_when_obs_averages_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(plot_ball_touches(_make_data(), tmp))
            self.assertTrue(Path(tmp, "rl_ball_touches.png").exists())

    def test_skips_without_obs_averages(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(plot_ball_touches(_make_data(with_obs_averages=False), tmp))


class TestPlotBoostUsage(unittest.TestCase):
    def test_writes_plot_when_obs_averages_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(plot_boost_usage(_make_data(), tmp))
            self.assertTrue(Path(tmp, "rl_boost_usage.png").exists())

    def test_skips_without_obs_averages(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(plot_boost_usage(_make_data(with_obs_averages=False), tmp))


class TestPlotBallPursuit(unittest.TestCase):
    def test_writes_plot_when_obs_averages_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(plot_ball_pursuit(_make_data(), tmp))
            self.assertTrue(Path(tmp, "rl_ball_pursuit.png").exists())

    def test_skips_without_obs_averages(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(plot_ball_pursuit(_make_data(with_obs_averages=False), tmp))

    def test_skips_when_pursuit_keys_absent(self):
        data = _make_data()
        for s in data.greedy_sims:
            s.obs_averages = {"boost_amount": 0.5}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(plot_ball_pursuit(data, tmp))


class TestSaveExperimentResults(unittest.TestCase):
    def test_report_includes_rl_plots_when_data_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_experiment_results(_make_data(), tmp)
            report = Path(tmp, "results.md").read_text(encoding="utf-8")
            self.assertIn("## Rocket League Plots", report)
            for fname in (
                "rl_match_results.png",
                "rl_ball_touches.png",
                "rl_boost_usage.png",
                "rl_ball_pursuit.png",
            ):
                self.assertIn(fname, report)
                self.assertTrue(Path(tmp, fname).exists(), fname)

    def test_match_results_present_even_without_obs_averages(self):
        # termination_reason is always recorded, so the headline plot still fires.
        with tempfile.TemporaryDirectory() as tmp:
            save_experiment_results(_make_data(with_obs_averages=False), tmp)
            report = Path(tmp, "results.md").read_text(encoding="utf-8")
            self.assertIn("rl_match_results.png", report)
            self.assertNotIn("rl_ball_touches.png", report)
            self.assertNotIn("rl_boost_usage.png", report)
            self.assertNotIn("rl_ball_pursuit.png", report)

    def test_no_rl_section_without_greedy_sims(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_experiment_results(_make_data(n_sims=0), tmp)
            report = Path(tmp, "results.md").read_text(encoding="utf-8")
            self.assertNotIn("## Rocket League Plots", report)


if __name__ == "__main__":
    unittest.main()
