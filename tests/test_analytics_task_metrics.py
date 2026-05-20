"""Tests for task-metric fields and analytics helpers added in the
'Normalize training rewards' issue.

Covers:
- GreedySimResult new fields (finish_time_s, mean_abs_lateral_offset,
  reward_components).
- _task_metrics_table_md()
- _greedy_table_md() extended columns.
- _gs_stats() task-metric keys.
- plot_task_metrics() / plot_reward_components() are no-ops when matplotlib
  is absent (regression guard).
- plot_reward_component_breakdown(): diverging stacked bar per sim.
"""
from __future__ import annotations

import os
import tempfile
import unittest

from framework.analytics import (
    GreedySimResult,
    ExperimentData,
    _task_metrics_table_md,
    _greedy_table_md,
    _summary_md,
    _gs_stats,
    plot_reward_component_breakdown,
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


# ---------------------------------------------------------------------------
# plot_gs_reward_trajectories — cross-run normalized reward chart (issue #182)
# ---------------------------------------------------------------------------

class TestPlotGsRewardTrajectories(unittest.TestCase):
    """plot_gs_reward_trajectories should write comparison_reward_trajectories.png
    and be included in the summary.md generated by save_grid_summary."""

    def _make_runs(self) -> list:
        sims_a = [_make_sim(i, reward=float(i)) for i in range(1, 6)]
        sims_b = [_make_sim(i, reward=float(-i)) for i in range(1, 4)]
        exp_a = _make_experiment(sims_a)
        exp_b = _make_experiment(sims_b)
        exp_a.experiment_name = "exp_a"
        exp_b.experiment_name = "exp_b"
        return [("exp_a", exp_a), ("exp_b", exp_b)]

    def test_trajectory_chart_written_by_save_grid_summary(self):
        import tempfile, os
        from framework.analytics import save_grid_summary
        runs = self._make_runs()
        with tempfile.TemporaryDirectory() as d:
            save_grid_summary(runs, [], d, "gs_traj")
            self.assertIn("comparison_reward_trajectories.png", os.listdir(d))

    def test_trajectory_chart_referenced_in_summary_md(self):
        import tempfile, os
        from framework.analytics import save_grid_summary
        runs = self._make_runs()
        with tempfile.TemporaryDirectory() as d:
            save_grid_summary(runs, [], d, "gs_traj")
            with open(os.path.join(d, "summary.md"), encoding="utf-8") as f:
                md = f.read()
            self.assertIn("comparison_reward_trajectories.png", md)

    def test_no_crash_with_empty_sims(self):
        import tempfile
        from framework.analytics import save_grid_summary
        exp_empty = _make_experiment([])
        with tempfile.TemporaryDirectory() as d:
            save_grid_summary([("empty", exp_empty)], [], d, "gs_empty")
            # No trajectory chart expected when there are no sims, but no crash.
            # summary.md should still be created.
            import os
            self.assertIn("summary.md", os.listdir(d))


# ---------------------------------------------------------------------------
# save_grid_summary — task_metric_fn / task_metric_fmt plugin (issue #209)
# ---------------------------------------------------------------------------

class TestSaveGridSummaryTaskMetric(unittest.TestCase):

    def _make_runs(self, progresses_by_name: dict) -> list:
        runs = []
        for name, progresses in progresses_by_name.items():
            sims = [_make_sim(sim=i, reward=p * 10, improved=True,
                              final_track_progress=p)
                    for i, p in enumerate(progresses)]
            exp = _make_experiment(sims)
            exp.experiment_name = name
            runs.append((name, exp))
        return runs

    def test_default_uses_track_progress_with_4f_format(self):
        import tempfile, os
        from framework.analytics import save_grid_summary
        runs = self._make_runs({"exp_a": [0.7], "exp_b": [0.3]})
        with tempfile.TemporaryDirectory() as d:
            save_grid_summary(runs, [], d, "gs")
            md = open(os.path.join(d, "summary.md"), encoding="utf-8").read()
        self.assertIn("Best Track Progress", md)
        self.assertIn("0.7000", md)
        self.assertNotIn("70.0%", md)

    def test_custom_metric_fn_replaces_label_and_value(self):
        import tempfile, os
        from framework.analytics import save_grid_summary
        runs = self._make_runs({"exp_a": [0.0]})
        with tempfile.TemporaryDirectory() as d:
            save_grid_summary(runs, [], d, "gs",
                              task_metric_fn=lambda _: 0.8,
                              task_metric_label="Win Rate",
                              task_metric_fmt="{:.1%}".format)
            md = open(os.path.join(d, "summary.md"), encoding="utf-8").read()
        self.assertIn("Win Rate", md)
        self.assertIn("80.0%", md)
        self.assertNotIn("Best Track Progress", md)

    def test_custom_metric_drives_ranking(self):
        import tempfile, os
        from framework.analytics import save_grid_summary
        # exp_b has higher custom metric despite lower track progress
        runs = self._make_runs({"exp_a": [0.9], "exp_b": [0.1]})
        metric_values = {"exp_a": 0.2, "exp_b": 0.8}
        runs_by_name = dict(runs)

        def metric_fn(data):
            return metric_values[data.experiment_name]

        with tempfile.TemporaryDirectory() as d:
            save_grid_summary(runs, [], d, "gs",
                              task_metric_fn=metric_fn,
                              task_metric_label="Score",
                              task_metric_fmt="{:.2f}".format)
            md = open(os.path.join(d, "summary.md"), encoding="utf-8").read()
        # exp_b should appear before exp_a in rankings
        self.assertLess(md.index("## 1. exp_b"), md.index("## 2. exp_a"))

    def test_custom_fmt_without_fn_applies_to_track_progress(self):
        import tempfile, os
        from framework.analytics import save_grid_summary
        runs = self._make_runs({"exp_a": [0.5]})
        with tempfile.TemporaryDirectory() as d:
            save_grid_summary(runs, [], d, "gs",
                              task_metric_fmt="{:.1%}".format)
            md = open(os.path.join(d, "summary.md"), encoding="utf-8").read()
        self.assertIn("50.0%", md)

    def test_summary_includes_code_versions_section_and_per_run_version(self):
        import tempfile, os
        from framework.analytics import save_grid_summary

        runs = self._make_runs({"exp_a": [0.5], "exp_b": [0.4]})
        runs[0][1].code_version = "0.1.2+gabc1234"
        runs[1][1].code_version = "0.1.3+gdef5678"

        with tempfile.TemporaryDirectory() as d:
            save_grid_summary(runs, [], d, "gs")
            md = open(os.path.join(d, "summary.md"), encoding="utf-8").read()

        self.assertIn("## Code Versions", md)
        self.assertIn("`0.1.2+gabc1234`", md)
        self.assertIn("`0.1.3+gdef5678`", md)
        self.assertIn("| Code version | `0.1.2+gabc1234` |", md)


class TestRunSummaryCodeVersion(unittest.TestCase):
    def test_summary_md_has_dedicated_code_version_section(self):
        exp = _make_experiment([_make_sim(1)])
        exp.code_version = "0.1.2+gabc1234"
        md = _summary_md(exp)
        self.assertIn("### Code Version", md)
        self.assertIn("`0.1.2+gabc1234`", md)


# ---------------------------------------------------------------------------
# plot_reward_component_breakdown (issue #252)
# ---------------------------------------------------------------------------

class TestPlotRewardComponentBreakdown(unittest.TestCase):

    def test_renders_to_file(self):
        sims = [
            _make_sim(1, reward_components={"progress": 10.0, "step_penalty": -1.0}),
            _make_sim(2, reward_components={"progress": 12.0, "step_penalty": -1.5},
                      improved=True),
            _make_sim(3, reward_components={"progress": 8.0, "step_penalty": -0.5}),
        ]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_reward_component_breakdown(data, d)
            self.assertIn("reward_component_breakdown.png", os.listdir(d))

    def test_skips_when_no_component_data(self):
        sims = [_make_sim(i) for i in range(1, 4)]  # reward_components=None
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_reward_component_breakdown(data, d)
            self.assertNotIn("reward_component_breakdown.png", os.listdir(d))

    def test_skips_when_no_sims(self):
        data = _make_experiment([])
        with tempfile.TemporaryDirectory() as d:
            plot_reward_component_breakdown(data, d)
            self.assertEqual(os.listdir(d), [])

    def test_skips_when_all_components_zero(self):
        sims = [_make_sim(1, reward_components={"progress": 0.0, "step_penalty": 0.0})]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_reward_component_breakdown(data, d)
            self.assertNotIn("reward_component_breakdown.png", os.listdir(d))

    def test_positive_only_components(self):
        sims = [_make_sim(1, reward_components={"progress": 15.0, "bonus": 5.0})]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_reward_component_breakdown(data, d)
            self.assertIn("reward_component_breakdown.png", os.listdir(d))

    def test_negative_only_components(self):
        sims = [_make_sim(1, reward_components={"step_penalty": -2.0, "crash": -10.0})]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_reward_component_breakdown(data, d)
            self.assertIn("reward_component_breakdown.png", os.listdir(d))

    def test_partial_none_sims_use_zero_for_missing_keys(self):
        sims = [
            _make_sim(1, reward_components={"progress": 10.0}),
            _make_sim(2),  # reward_components=None → treated as all-zero
        ]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_reward_component_breakdown(data, d)
            self.assertIn("reward_component_breakdown.png", os.listdir(d))


if __name__ == "__main__":
    unittest.main(verbosity=2)
