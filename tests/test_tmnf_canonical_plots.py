"""Tests for the TMNF-specific canonical-score plots added for issue #481.

Covers:
- games.tmnf.analytics.plot_greedy_canonical(): renders per-experiment chart
  when any greedy sim has a canonical score, no-op otherwise.
- games.tmnf.analytics.plot_gs_comparison_canonical_progress(): cross-run
  best-so-far chart, no-op when no run has canonical scores.
- save_tmnf_plots() calls plot_greedy_canonical() when greedy sims exist.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from framework.analytics import ExperimentData, GreedySimResult
from games.tmnf.analytics import (
    plot_greedy_canonical,
    plot_gs_comparison_canonical_progress,
    save_tmnf_plots,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sim(sim: int, improved: bool = False, canonical_score: float | None = None) -> GreedySimResult:
    return GreedySimResult(
        sim=sim,
        reward=float(sim),
        improved=improved,
        throttle_counts=[10, 10, 80],
        total_steps=100,
        canonical_score=canonical_score,
    )


def _make_experiment(sims: list, name: str = "test_exp") -> ExperimentData:
    return ExperimentData(
        experiment_name=name,
        probe_results=[],
        cold_start_restarts=[],
        greedy_sims=sims,
        probe_floor=None,
        weights_file="/tmp/w.yaml",
        reward_config_file="/tmp/r.yaml",
        training_params={},
        timings={"start": "2024-01-01", "end": "2024-01-02", "total_s": 100.0, "greedy_s": 90.0},
    )


# ---------------------------------------------------------------------------
# plot_greedy_canonical
# ---------------------------------------------------------------------------


class TestPlotGreedyCanonical(unittest.TestCase):
    def test_renders_when_canonical_scores_present(self):
        sims = [
            _make_sim(1, canonical_score=100.0),
            _make_sim(2, improved=True, canonical_score=200.0),
        ]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_greedy_canonical(data, d)
            self.assertIn("greedy_canonical.png", os.listdir(d))

    def test_no_op_when_no_canonical_scores(self):
        sims = [_make_sim(i) for i in range(1, 4)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_greedy_canonical(data, d)
            self.assertEqual(os.listdir(d), [])

    def test_no_op_when_no_sims(self):
        data = _make_experiment([])
        with tempfile.TemporaryDirectory() as d:
            plot_greedy_canonical(data, d)
            self.assertEqual(os.listdir(d), [])

    def test_save_tmnf_plots_includes_canonical(self):
        sims = [_make_sim(1, canonical_score=150.0)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            save_tmnf_plots(data, d)
            self.assertIn("greedy_canonical.png", os.listdir(d))


# ---------------------------------------------------------------------------
# plot_gs_comparison_canonical_progress
# ---------------------------------------------------------------------------


class TestPlotGsComparisonCanonicalProgress(unittest.TestCase):
    def test_renders_when_eligible(self):
        sims_a = [_make_sim(i, canonical_score=100.0 * i) for i in range(1, 4)]
        sims_b = [_make_sim(i, canonical_score=50.0 * i) for i in range(1, 3)]
        runs = [
            ("exp_a", _make_experiment(sims_a, "exp_a")),
            ("exp_b", _make_experiment(sims_b, "exp_b")),
        ]
        with tempfile.TemporaryDirectory() as d:
            plot_gs_comparison_canonical_progress(runs, d)
            self.assertIn("comparison_canonical_progress.png", os.listdir(d))

    def test_no_op_when_no_run_has_canonical_scores(self):
        sims_a = [_make_sim(i) for i in range(1, 4)]
        runs = [("exp_a", _make_experiment(sims_a, "exp_a"))]
        with tempfile.TemporaryDirectory() as d:
            plot_gs_comparison_canonical_progress(runs, d)
            self.assertEqual(os.listdir(d), [])

    def test_no_op_when_no_runs(self):
        with tempfile.TemporaryDirectory() as d:
            plot_gs_comparison_canonical_progress([], d)
            self.assertEqual(os.listdir(d), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
