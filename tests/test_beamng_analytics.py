"""Tests for BeamNG-specific analytics (issue #462).

Exercises each BeamNG plot function and the results.md report with synthetic
ExperimentData — no BeamNG binary or beamng_gym install needed.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from framework.analytics import ExperimentData, GreedySimResult
from games.beamng.analytics import (
    plot_airborne,
    plot_centerline_distribution,
    plot_lap_time_progression,
    plot_mean_speed,
    save_experiment_results,
)


def _obs_averages(speed: float = 30.0, airborne: float = 0.1) -> dict:
    return {
        "speed_ms": speed,
        "abs_lateral_offset_m": 1.2,
        "wheel_0_contact": 0.95,
        "wheel_1_contact": 0.95,
        "wheel_2_contact": 0.9,
        "wheel_3_contact": 0.9,
        "airborne": airborne,
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
            termination_reason="finish" if i == n_sims - 1 else "crash",
            finish_time_s=(110.0 - i) if (with_finish_times and i >= n_sims // 2) else None,
            mean_abs_lateral_offset=1.0 + 0.1 * i,
            obs_averages=_obs_averages(speed=25.0 + i) if with_obs_averages else None,
        )
        for i in range(n_sims)
    ]
    return ExperimentData(
        experiment_name="beamng-analytics-test",
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


class TestPlotAirborne(unittest.TestCase):
    def test_writes_plot_when_obs_averages_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(plot_airborne(_make_data(), tmp))
            self.assertTrue(Path(tmp, "beamng_airborne.png").exists())

    def test_skips_without_obs_averages(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(plot_airborne(_make_data(with_obs_averages=False), tmp))
            self.assertFalse(Path(tmp, "beamng_airborne.png").exists())

    def test_skips_when_airborne_and_contact_keys_absent(self):
        data = _make_data()
        for s in data.greedy_sims:
            s.obs_averages = {"speed_ms": 30.0}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(plot_airborne(data, tmp))


class TestPlotMeanSpeed(unittest.TestCase):
    def test_writes_plot_when_speed_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(plot_mean_speed(_make_data(), tmp))
            self.assertTrue(Path(tmp, "beamng_mean_speed.png").exists())

    def test_skips_without_obs_averages(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(plot_mean_speed(_make_data(with_obs_averages=False), tmp))


class TestPlotCenterlineDistribution(unittest.TestCase):
    def test_writes_plot_from_mean_abs_lateral_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(plot_centerline_distribution(_make_data(), tmp))
            self.assertTrue(Path(tmp, "beamng_centerline_dist.png").exists())

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
            self.assertTrue(Path(tmp, "beamng_lap_times.png").exists())

    def test_skips_without_finished_sims(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(plot_lap_time_progression(_make_data(with_finish_times=False), tmp))

    def test_draws_par_line_from_reward_config(self):
        data = _make_data()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp, "reward_config.yaml")
            cfg.write_text("par_time_s: 120.0\n", encoding="utf-8")
            data.reward_config_file = str(cfg)
            self.assertTrue(plot_lap_time_progression(data, tmp))

    def test_bad_reward_config_skips_par_line_without_crashing(self):
        for bad_content in (
            "par_time_s: not-a-number\n",  # non-numeric value
            "::: not yaml {{{\n",  # unparseable YAML
        ):
            data = _make_data()
            with tempfile.TemporaryDirectory() as tmp:
                cfg = Path(tmp, "reward_config.yaml")
                cfg.write_text(bad_content, encoding="utf-8")
                data.reward_config_file = str(cfg)
                self.assertTrue(plot_lap_time_progression(data, tmp))
                self.assertTrue(Path(tmp, "beamng_lap_times.png").exists())


class TestSaveExperimentResults(unittest.TestCase):
    def test_report_includes_beamng_plots_when_data_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_experiment_results(_make_data(), tmp)
            report = Path(tmp, "results.md").read_text(encoding="utf-8")
            self.assertIn("## BeamNG Plots", report)
            self.assertIn("beamng_airborne.png", report)
            self.assertIn("beamng_mean_speed.png", report)
            self.assertIn("beamng_centerline_dist.png", report)
            self.assertIn("beamng_lap_times.png", report)
            self.assertIn("greedy_progress.png", report)
            self.assertIn("termination_reasons.png", report)
            for fname in (
                "beamng_airborne.png",
                "beamng_mean_speed.png",
                "beamng_centerline_dist.png",
                "beamng_lap_times.png",
                "greedy_progress.png",
                "termination_reasons.png",
                "results.md",
            ):
                self.assertTrue(Path(tmp, fname).exists(), fname)

    def test_report_omits_beamng_section_without_game_data(self):
        data = _make_data(with_obs_averages=False, with_finish_times=False)
        for s in data.greedy_sims:
            s.mean_abs_lateral_offset = None
        with tempfile.TemporaryDirectory() as tmp:
            save_experiment_results(data, tmp)
            report = Path(tmp, "results.md").read_text(encoding="utf-8")
            self.assertNotIn("## BeamNG Plots", report)
            self.assertNotIn("beamng_airborne.png", report)

    def test_best_run_link_gated_on_file_existence(self):
        # No trace on any sim → plot_greedy_best_run writes nothing → no link.
        with tempfile.TemporaryDirectory() as tmp:
            save_experiment_results(_make_data(), tmp)
            report = Path(tmp, "results.md").read_text(encoding="utf-8")
            self.assertNotIn("greedy_best_run.png", report)
            self.assertFalse(Path(tmp, "greedy_best_run.png").exists())


if __name__ == "__main__":
    unittest.main()
