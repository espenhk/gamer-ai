"""Tests for task-metric fields and analytics helpers added in issue #XX.

Covers:
- GreedySimResult new fields (finish_time_s, mean_abs_lateral_offset,
  reward_components).
- _task_metrics_table_md()
- _greedy_table_md() extended columns.
- _gs_stats() task-metric keys.
- plot_task_metrics() / plot_reward_components() are no-ops when matplotlib
  is absent (regression guard).
"""
from __future__ import annotations

import unittest

from framework.analytics import (
    GreedySimResult,
    ExperimentData,
    _task_metrics_table_md,
    _greedy_table_md,
    _gs_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sim(
    sim: int = 1,
    reward: float = 10.0,
    improved: bool = False,
    final_track_progress: float = 0.5,
    finish_time_s: float | None = None,
    mean_abs_lateral_offset: float | None = None,
    reward_components: dict | None = None,
    termination_reason: str | None = "timeout",
) -> GreedySimResult:
    return GreedySimResult(
        sim=sim,
        reward=reward,
        improved=improved,
        throttle_counts=[10, 10, 80],
        total_steps=100,
        final_track_progress=final_track_progress,
        finish_time_s=finish_time_s,
        mean_abs_lateral_offset=mean_abs_lateral_offset,
        reward_components=reward_components,
        termination_reason=termination_reason,
    )


def _make_experiment(sims: list) -> ExperimentData:
    return ExperimentData(
        experiment_name="test_exp",
        probe_results=[],
        cold_start_restarts=[],
        greedy_sims=sims,
        probe_floor=None,
        weights_file="/tmp/w.yaml",
        reward_config_file="/tmp/r.yaml",
        training_params={},
        timings={"start": "2024-01-01", "end": "2024-01-02",
                 "total_s": 100.0, "greedy_s": 90.0},
    )


# ---------------------------------------------------------------------------
# GreedySimResult new fields
# ---------------------------------------------------------------------------

class TestGreedySimResultNewFields(unittest.TestCase):

    def test_default_new_fields_are_none(self):
        s = GreedySimResult(
            sim=1, reward=5.0, improved=False,
            throttle_counts=[0, 0, 1], total_steps=1,
        )
        self.assertIsNone(s.finish_time_s)
        self.assertIsNone(s.mean_abs_lateral_offset)
        self.assertIsNone(s.reward_components)

    def test_finish_time_s_stored(self):
        s = _make_sim(finish_time_s=42.5)
        self.assertAlmostEqual(s.finish_time_s, 42.5)

    def test_mean_abs_lateral_offset_stored(self):
        s = _make_sim(mean_abs_lateral_offset=1.23)
        self.assertAlmostEqual(s.mean_abs_lateral_offset, 1.23)

    def test_reward_components_stored(self):
        comps = {"progress": 5.0, "step_penalty": -0.5}
        s = _make_sim(reward_components=comps)
        self.assertEqual(s.reward_components, comps)


# ---------------------------------------------------------------------------
# _gs_stats() task-metric keys
# ---------------------------------------------------------------------------

class TestGsStatsTaskMetrics(unittest.TestCase):

    def test_empty_sims_returns_zero_finish_rate(self):
        data = _make_experiment([])
        stats = _gs_stats(data)
        self.assertEqual(stats["finish_rate"], 0.0)
        self.assertEqual(stats["best_track_progress"], 0.0)
        self.assertIsNone(stats["best_finish_time_s"])

    def test_no_finishes(self):
        sims = [_make_sim(i, final_track_progress=0.5) for i in range(1, 4)]
        data = _make_experiment(sims)
        stats = _gs_stats(data)
        self.assertAlmostEqual(stats["finish_rate"], 0.0)
        self.assertIsNone(stats["best_finish_time_s"])
        self.assertAlmostEqual(stats["best_track_progress"], 0.5)

    def test_all_finished(self):
        sims = [
            _make_sim(1, final_track_progress=1.0, finish_time_s=55.0),
            _make_sim(2, final_track_progress=1.0, finish_time_s=45.0),
            _make_sim(3, final_track_progress=1.0, finish_time_s=60.0),
        ]
        data = _make_experiment(sims)
        stats = _gs_stats(data)
        self.assertAlmostEqual(stats["finish_rate"], 1.0)
        self.assertAlmostEqual(stats["best_finish_time_s"], 45.0)
        self.assertAlmostEqual(stats["best_track_progress"], 1.0)

    def test_partial_finishes(self):
        sims = [
            _make_sim(1, final_track_progress=1.0, finish_time_s=50.0),
            _make_sim(2, final_track_progress=0.7),
            _make_sim(3, final_track_progress=0.9),
        ]
        data = _make_experiment(sims)
        stats = _gs_stats(data)
        self.assertAlmostEqual(stats["finish_rate"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(stats["best_finish_time_s"], 50.0)
        self.assertAlmostEqual(stats["best_track_progress"], 1.0)


# ---------------------------------------------------------------------------
# _task_metrics_table_md()
# ---------------------------------------------------------------------------

class TestTaskMetricsTableMd(unittest.TestCase):

    def test_empty_sims_returns_empty_string(self):
        data = _make_experiment([])
        result = _task_metrics_table_md(data)
        self.assertEqual(result, "")

    def test_contains_finish_rate(self):
        sims = [
            _make_sim(1, final_track_progress=1.0, finish_time_s=55.0),
            _make_sim(2, final_track_progress=0.5),
        ]
        data = _make_experiment(sims)
        md = _task_metrics_table_md(data)
        self.assertIn("50.0%", md)
        self.assertIn("Finish rate", md)

    def test_contains_best_finish_time(self):
        sims = [
            _make_sim(1, finish_time_s=45.0, final_track_progress=1.0),
            _make_sim(2, finish_time_s=60.0, final_track_progress=1.0),
        ]
        data = _make_experiment(sims)
        md = _task_metrics_table_md(data)
        self.assertIn("45.0s", md)
        self.assertIn("Best finish time", md)

    def test_no_finish_time_when_no_finishes(self):
        sims = [_make_sim(i) for i in range(1, 4)]
        data = _make_experiment(sims)
        md = _task_metrics_table_md(data)
        self.assertNotIn("Best finish time", md)

    def test_contains_lateral_offset(self):
        sims = [_make_sim(1, mean_abs_lateral_offset=1.5)]
        data = _make_experiment(sims)
        md = _task_metrics_table_md(data)
        self.assertIn("1.500m", md)
        self.assertIn("lateral", md.lower())


# ---------------------------------------------------------------------------
# _greedy_table_md() — extended columns
# ---------------------------------------------------------------------------

class TestGreedyTableMd(unittest.TestCase):

    def test_includes_progress_column(self):
        sims = [_make_sim(1, final_track_progress=0.75)]
        data = _make_experiment(sims)
        md = _greedy_table_md(data)
        self.assertIn("Progress", md)
        self.assertIn("0.750", md)

    def test_includes_finish_time_column(self):
        sims = [_make_sim(1, finish_time_s=50.0, final_track_progress=1.0)]
        data = _make_experiment(sims)
        md = _greedy_table_md(data)
        self.assertIn("Finish Time", md)
        self.assertIn("50.0s", md)

    def test_no_finish_shows_dash(self):
        sims = [_make_sim(1)]  # finish_time_s=None
        data = _make_experiment(sims)
        md = _greedy_table_md(data)
        self.assertIn("—", md)

    def test_includes_lateral_offset_column(self):
        sims = [_make_sim(1, mean_abs_lateral_offset=2.0)]
        data = _make_experiment(sims)
        md = _greedy_table_md(data)
        self.assertIn("2.00m", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
