"""Tests for the TORCS analytics module."""
import os
import tempfile
import unittest

import numpy as np

from framework.analytics import (
    ExperimentData,
    RunTrace,
    ProbeResult,
    ColdStartSimResult,
    ColdStartRestartResult,
    GreedySimResult,
)
from games.torcs.analytics import (
    save_experiment_results,
    save_torcs_plots,
    save_grid_summary,
    plot_greedy_action_dist,
    plot_greedy_progress,
    plot_termination_reasons,
    plot_weight_heatmap,
    plot_weight_evolution,
    plot_cold_start_action_dist,
    plot_gs_comparison_progress,
)


def _make_trace(reward=10.0, n_steps=20):
    """Build a RunTrace with throttle data (no pos_x/pos_z for TORCS)."""
    return RunTrace(
        pos_x=[],
        pos_z=[],
        throttle_state=[(0.8, 0.0)] * n_steps,
        total_reward=reward,
    )


def _make_experiment(
    name="torcs_test",
    n_probes=6,
    n_restarts=3,
    n_greedy=10,
    weights_file="/tmp/torcs_test_weights.yaml",
    reward_cfg_file="/tmp/torcs_test_reward.yaml",
) -> ExperimentData:
    """Build a minimal ExperimentData for testing."""
    probes = [
        ProbeResult(
            action_idx=i,
            action_name=f"action_{i}",
            reward=float(i * 10),
            trace=_make_trace(float(i * 10)),
        )
        for i in range(n_probes)
    ]

    restarts = []
    for r in range(n_restarts):
        sims = [
            ColdStartSimResult(
                sim=s + 1,
                reward=float(r * 5 + s),
                throttle_counts=[2, 3, 15],
                total_steps=20,
                trace=_make_trace(float(r * 5 + s)),
                termination_reason="timeout",
            )
            for s in range(3)
        ]
        restarts.append(ColdStartRestartResult(
            restart=r + 1,
            sims=sims,
            best_reward=max(s.reward for s in sims),
            beat_probe_floor=r > 0,
        ))

    greedy = [
        GreedySimResult(
            sim=i + 1,
            reward=float(i * 3),
            improved=(i % 3 == 0),
            throttle_counts=[1, 2, 17],
            total_steps=20,
            trace=_make_trace(float(i * 3)),
            weights=None,
            final_track_progress=0.1 * i,
            laps_completed=0,
            mutation_scale=0.05,
            termination_reason="timeout" if i < 8 else "crash",
        )
        for i in range(n_greedy)
    ]

    return ExperimentData(
        experiment_name=name,
        probe_results=probes,
        cold_start_restarts=restarts,
        greedy_sims=greedy,
        probe_floor=10.0,
        weights_file=weights_file,
        reward_config_file=reward_cfg_file,
        training_params={"n_sims": 10, "policy_type": "genetic"},
        timings={
            "start": "2026-01-01T00:00:00",
            "end": "2026-01-01T01:00:00",
            "total_s": 3600.0,
            "probe_s": 120.0,
            "cold_start_s": 600.0,
            "greedy_s": 2880.0,
        },
        track="torcs",
    )


class TestTorcsAnalyticsPlots(unittest.TestCase):
    """Smoke-test that TORCS-specific plots run without error."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data = _make_experiment()

    def test_plot_greedy_action_dist(self):
        plot_greedy_action_dist(self.data, self.tmpdir)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "greedy_action_dist.png")))

    def test_plot_greedy_progress(self):
        plot_greedy_progress(self.data, self.tmpdir)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "greedy_progress.png")))

    def test_plot_termination_reasons(self):
        plot_termination_reasons(self.data, self.tmpdir)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "termination_reasons.png")))

    def test_plot_cold_start_action_dist(self):
        plot_cold_start_action_dist(self.data, self.tmpdir)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "cold_start_action_dist.png")))

    def test_plot_weight_heatmap_no_file(self):
        """Should not crash when weights file doesn't exist."""
        plot_weight_heatmap(self.data, self.tmpdir)
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "policy_weights_heatmap.png")))

    def test_plot_weight_evolution_no_weights(self):
        """Should not crash when greedy sims have no weights."""
        plot_weight_evolution(self.data, self.tmpdir)
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "greedy_weight_evolution.png")))

    def test_save_torcs_plots_no_crash(self):
        """save_torcs_plots should run without error."""
        save_torcs_plots(self.data, self.tmpdir)

    def test_save_experiment_results_generates_report(self):
        """save_experiment_results should generate a results.md file."""
        save_experiment_results(self.data, self.tmpdir)
        report = os.path.join(self.tmpdir, "results.md")
        self.assertTrue(os.path.exists(report))
        with open(report) as f:
            content = f.read()
        self.assertIn("torcs_test", content)
        self.assertIn("TORCS", content)

    def test_empty_experiment_no_crash(self):
        """Should handle empty experiment (no probes, no restarts, no greedy)."""
        data = ExperimentData(
            experiment_name="empty_torcs",
            probe_results=[],
            cold_start_restarts=[],
            greedy_sims=[],
            probe_floor=None,
            weights_file="/tmp/nonexistent.yaml",
            reward_config_file="/tmp/nonexistent.yaml",
            training_params={},
            timings={"start": "x", "end": "x", "total_s": 0.0},
            track="torcs",
        )
        save_experiment_results(data, self.tmpdir)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "results.md")))


class TestTorcsGridSummary(unittest.TestCase):
    """Smoke-test the grid search summary for TORCS."""

    def test_save_grid_summary(self):
        tmpdir = tempfile.mkdtemp()
        data1 = _make_experiment(name="combo_1")
        data2 = _make_experiment(name="combo_2")
        runs = [("combo_1", data1), ("combo_2", data2)]
        save_grid_summary(runs, ["n_sims"], tmpdir, "gs_test")
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "summary.md")))
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "comparison_rewards.png")))

    def test_gs_comparison_progress(self):
        tmpdir = tempfile.mkdtemp()
        data1 = _make_experiment(name="combo_1")
        data2 = _make_experiment(name="combo_2")
        runs = [("combo_1", data1), ("combo_2", data2)]
        plot_gs_comparison_progress(runs, tmpdir)
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "comparison_progress.png")))


if __name__ == "__main__":
    unittest.main()
