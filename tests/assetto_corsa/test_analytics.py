"""Tests for Assetto Corsa-specific analytics (issue #464).

Exercises each AC plot function and the results.md report with synthetic
ExperimentData — no Assetto Corsa binary or assetto-corsa-rl install needed.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from framework.analytics import ExperimentData, GreedySimResult
from games.assetto_corsa.analytics import (
    plot_centerline_distribution,
    plot_lap_time_progression,
    plot_rpm_gear,
    plot_wheel_slip,
    save_experiment_results,
)


def _obs_averages(rpm: float = 4500.0, gear: float = 3.2, slip: float = 0.1) -> dict:
    return {
        "speed_ms": 40.0,
        "abs_lateral_offset_m": 1.5,
        "engine_rpm": rpm,
        "gear": gear,
        "wheel_0_slip": slip,
        "wheel_1_slip": slip + 0.02,
        "wheel_2_slip": slip + 0.05,
        "wheel_3_slip": slip + 0.07,
    }


def _make_data(
    n_sims: int = 4,
    with_obs_averages: bool = True,
    with_finish_times: bool = True,
) -> ExperimentData:
    greedy_sims = [
        GreedySimResult(
            sim=i + 1,
            reward=100.0 * (i + 1),
            improved=True,
            throttle_counts=[5, 10, 30],
            total_steps=200,
            final_track_progress=0.25 * (i + 1),
            termination_reason="finish" if i == n_sims - 1 else "timeout",
            finish_time_s=(140.0 - i) if (with_finish_times and i >= n_sims // 2) else None,
            mean_abs_lateral_offset=1.0 + 0.1 * i,
            obs_averages=_obs_averages(rpm=4000.0 + 100 * i) if with_obs_averages else None,
        )
        for i in range(n_sims)
    ]
    return ExperimentData(
        experiment_name="ac-analytics-test",
        probe_results=[],
        cold_start_restarts=[],
        greedy_sims=greedy_sims,
        probe_floor=None,
        weights_file="policy_weights.yaml",
        reward_config_file="nonexistent_reward_config.yaml",
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


class TestPlotWheelSlip(unittest.TestCase):
    def test_writes_plot_when_obs_averages_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(plot_wheel_slip(_make_data(), tmp))
            self.assertTrue(Path(tmp, "ac_wheel_slip.png").exists())

    def test_skips_without_obs_averages(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(plot_wheel_slip(_make_data(with_obs_averages=False), tmp))
            self.assertFalse(Path(tmp, "ac_wheel_slip.png").exists())

    def test_skips_when_slip_keys_absent(self):
        data = _make_data()
        for s in data.greedy_sims:
            s.obs_averages = {"engine_rpm": 4000.0}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(plot_wheel_slip(data, tmp))
            self.assertFalse(Path(tmp, "ac_wheel_slip.png").exists())


class TestPlotRpmGear(unittest.TestCase):
    def test_writes_plot_when_obs_averages_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(plot_rpm_gear(_make_data(), tmp))
            self.assertTrue(Path(tmp, "ac_rpm_gear.png").exists())

    def test_skips_without_obs_averages(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(plot_rpm_gear(_make_data(with_obs_averages=False), tmp))


class TestPlotCenterlineDistribution(unittest.TestCase):
    def test_writes_plot_from_mean_abs_lateral_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(plot_centerline_distribution(_make_data(), tmp))
            self.assertTrue(Path(tmp, "ac_centerline_dist.png").exists())

    def test_falls_back_to_obs_averages(self):
        data = _make_data()
        for s in data.greedy_sims:
            s.mean_abs_lateral_offset = None
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(plot_centerline_distribution(data, tmp))

    def test_skips_without_any_source(self):
        data = _make_data(with_obs_averages=False)
        for s in data.greedy_sims:
            s.mean_abs_lateral_offset = None
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(plot_centerline_distribution(data, tmp))


class TestPlotLapTimeProgression(unittest.TestCase):
    def test_writes_plot_when_finish_times_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(plot_lap_time_progression(_make_data(), tmp))
            self.assertTrue(Path(tmp, "ac_lap_times.png").exists())

    def test_skips_without_finished_sims(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(plot_lap_time_progression(_make_data(with_finish_times=False), tmp))

    def test_draws_par_line_from_reward_config(self):
        data = _make_data()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp, "reward_config.yaml")
            cfg.write_text("par_time_s: 150.0\n", encoding="utf-8")
            data.reward_config_file = str(cfg)
            self.assertTrue(plot_lap_time_progression(data, tmp))
            self.assertTrue(Path(tmp, "ac_lap_times.png").exists())


class TestSaveExperimentResults(unittest.TestCase):
    def test_report_includes_ac_plots_when_data_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_experiment_results(_make_data(), tmp)
            report = Path(tmp, "results.md").read_text(encoding="utf-8")
            self.assertIn("## Assetto Corsa Plots", report)
            self.assertIn("ac_wheel_slip.png", report)
            self.assertIn("ac_rpm_gear.png", report)
            self.assertIn("ac_centerline_dist.png", report)
            self.assertIn("ac_lap_times.png", report)
            self.assertIn("greedy_progress.png", report)
            self.assertIn("termination_reasons.png", report)
            for fname in (
                "ac_wheel_slip.png",
                "ac_rpm_gear.png",
                "ac_centerline_dist.png",
                "ac_lap_times.png",
                "greedy_progress.png",
                "termination_reasons.png",
                "results.md",
            ):
                self.assertTrue(Path(tmp, fname).exists(), fname)

    def test_report_omits_ac_section_without_game_data(self):
        data = _make_data(with_obs_averages=False, with_finish_times=False)
        for s in data.greedy_sims:
            s.mean_abs_lateral_offset = None
        with tempfile.TemporaryDirectory() as tmp:
            save_experiment_results(data, tmp)
            report = Path(tmp, "results.md").read_text(encoding="utf-8")
            self.assertNotIn("## Assetto Corsa Plots", report)
            self.assertNotIn("ac_wheel_slip.png", report)

    def test_best_run_link_gated_on_file_existence(self):
        # No trace on any sim → plot_greedy_best_run writes nothing → no link.
        with tempfile.TemporaryDirectory() as tmp:
            save_experiment_results(_make_data(), tmp)
            report = Path(tmp, "results.md").read_text(encoding="utf-8")
            self.assertNotIn("greedy_best_run.png", report)
            self.assertFalse(Path(tmp, "greedy_best_run.png").exists())


if __name__ == "__main__":
    unittest.main()
