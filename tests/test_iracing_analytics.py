"""Tests for iRacing-specific analytics output."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from framework.analytics import ExperimentData, GreedySimResult
from games.iracing.analytics import (
    _iracing_summary_md,
    plot_greedy_action_dist,
    plot_lap_time_improvement,
    plot_obs_avg_panels,
    plot_termination_reasons,
    save_experiment_results,
)


def _make_data(
    *,
    finish_times: list[float | None] | None = None,
    obs_averages: list[dict | None] | None = None,
) -> ExperimentData:
    if finish_times is None:
        finish_times = [85.3, None, 82.1, 79.8]
    greedy_sims = []
    for i, ft in enumerate(finish_times):
        avgs = obs_averages[i] if obs_averages else None
        greedy_sims.append(
            GreedySimResult(
                sim=i + 1,
                reward=float(100 - (ft or 200)),
                improved=(i == 0 or (ft is not None and ft < min(f for f in finish_times[:i] if f))),
                throttle_counts=[10, 5, 30],
                total_steps=45,
                finish_time_s=ft,
                termination_reason="finish" if ft is not None else "crash",
                obs_averages=avgs,
            )
        )
    return ExperimentData(
        experiment_name="iracing-smoke",
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
        track="laguna_seca",
    )


class TestIRacingLapTimeImprovement(unittest.TestCase):
    def test_plot_written_when_some_sims_finish(self):
        data = _make_data(finish_times=[85.3, None, 82.1, 79.8])
        with tempfile.TemporaryDirectory() as tmp:
            result = plot_lap_time_improvement(data, tmp)
            self.assertTrue(result)
            self.assertTrue(Path(tmp, "iracing_lap_time_improvement.png").exists())

    def test_returns_false_when_no_finished_laps(self):
        data = _make_data(finish_times=[None, None])
        with tempfile.TemporaryDirectory() as tmp:
            result = plot_lap_time_improvement(data, tmp)
            self.assertFalse(result)

    def test_returns_false_for_empty_sims(self):
        data = _make_data(finish_times=[])
        with tempfile.TemporaryDirectory() as tmp:
            result = plot_lap_time_improvement(data, tmp)
            self.assertFalse(result)


class TestIRacingActionDist(unittest.TestCase):
    def test_plot_written(self):
        data = _make_data()
        with tempfile.TemporaryDirectory() as tmp:
            result = plot_greedy_action_dist(data, tmp)
            self.assertTrue(result)
            self.assertTrue(Path(tmp, "iracing_action_dist.png").exists())


class TestIRacingTerminationReasons(unittest.TestCase):
    def test_plot_written(self):
        data = _make_data()
        with tempfile.TemporaryDirectory() as tmp:
            result = plot_termination_reasons(data, tmp)
            self.assertTrue(result)
            self.assertTrue(Path(tmp, "iracing_termination_reasons.png").exists())


class TestIRacingObsAveragePanels(unittest.TestCase):
    def test_skipped_when_obs_averages_absent(self):
        data = _make_data()
        with tempfile.TemporaryDirectory() as tmp:
            written = plot_obs_avg_panels(data, tmp)
            self.assertEqual(written, [])

    def test_tire_temp_panel_written_when_obs_averages_present(self):
        avgs = [
            {
                "tire_temp_fl": 90.0,
                "tire_temp_fr": 91.0,
                "tire_temp_rl": 88.0,
                "tire_temp_rr": 89.0,
                "fuel_pct": 0.8,
                "rpm": 5000.0,
                "brake_bias": 0.55,
            }
        ] * 4
        data = _make_data(obs_averages=avgs)
        with tempfile.TemporaryDirectory() as tmp:
            written = plot_obs_avg_panels(data, tmp)
            self.assertIn("iracing_tire_temps.png", written)
            self.assertIn("iracing_fuel.png", written)
            self.assertIn("iracing_rpm.png", written)
            self.assertIn("iracing_brake_bias.png", written)
            self.assertTrue(Path(tmp, "iracing_tire_temps.png").exists())


class TestIRacingSummaryMd(unittest.TestCase):
    def test_includes_finish_rate_and_best_lap(self):
        data = _make_data(finish_times=[85.3, None, 82.1, 79.8])
        md = _iracing_summary_md(data)
        self.assertIn("## iRacing Metrics", md)
        self.assertIn("Finish rate", md)
        self.assertIn("Best lap time", md)
        self.assertIn("79.800 s", md)

    def test_no_lap_stats_when_all_dnf(self):
        data = _make_data(finish_times=[None, None])
        md = _iracing_summary_md(data)
        self.assertIn("Finish rate", md)
        self.assertNotIn("Best lap time", md)


class TestIRacingSaveExperimentResults(unittest.TestCase):
    def test_report_includes_iracing_sections_and_headline_plot(self):
        data = _make_data(finish_times=[85.3, None, 82.1, 79.8])
        with tempfile.TemporaryDirectory() as tmp:
            save_experiment_results(data, tmp)
            report = Path(tmp, "results.md").read_text(encoding="utf-8")
            self.assertIn("## iRacing Metrics", report)
            self.assertIn("Finish rate", report)
            self.assertIn("iracing_lap_time_improvement.png", report)
            self.assertIn("iracing_action_dist.png", report)
            self.assertIn("iracing_termination_reasons.png", report)
            self.assertTrue(Path(tmp, "iracing_lap_time_improvement.png").exists())
            self.assertTrue(Path(tmp, "iracing_action_dist.png").exists())


if __name__ == "__main__":
    unittest.main()
