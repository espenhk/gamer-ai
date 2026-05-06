"""Tests for the SC2-specific analytics module (issue improvements 2a-2i).

Covers:
- SUPPORTS_THROTTLE / SUPPORTS_PATH flags
- plot_action_frequency: renders without error; skips when no data
- plot_obs_averages: renders without error; skips when no data
- plot_spatial_heatmap: renders without error; skips when no data
- plot_outcome_breakdown: renders without error; skips when no data
- plot_supply_capped: renders without error; skips when no data
- plot_resource_series: renders without error; skips when no data
- plot_army_count: renders without error; skips when no data
- plot_build_order: renders without error; skips when no data
- GreedySimResult new optional fields: action_counts, obs_averages, xy_hist,
  supply_capped_fraction, build_order, army_count_series, resource_series
- save_experiment_results: completes without error; writes results.md
"""
from __future__ import annotations

import os
import tempfile
import unittest

from framework.analytics import ExperimentData, GreedySimResult
from games.sc2.analytics import (
    SUPPORTS_PATH,
    SUPPORTS_THROTTLE,
    plot_action_frequency,
    plot_army_count,
    plot_build_order,
    plot_obs_averages,
    plot_outcome_breakdown,
    plot_resource_series,
    plot_spatial_heatmap,
    plot_supply_capped,
    save_experiment_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sim(
    sim: int = 1,
    reward: float = 10.0,
    improved: bool = False,
    action_counts: dict | None = None,
    obs_averages: dict | None = None,
    xy_hist: list | None = None,
    termination_reason: str | None = "timeout",
    reward_components: dict | None = None,
    supply_capped_fraction: float | None = None,
    build_order: list | None = None,
    army_count_series: list | None = None,
    resource_series: list | None = None,
) -> GreedySimResult:
    return GreedySimResult(
        sim=sim,
        reward=reward,
        improved=improved,
        throttle_counts=[0, 0, 0],
        total_steps=100,
        action_counts=action_counts,
        obs_averages=obs_averages,
        xy_hist=xy_hist,
        termination_reason=termination_reason,
        reward_components=reward_components,
        supply_capped_fraction=supply_capped_fraction,
        build_order=build_order,
        army_count_series=army_count_series,
        resource_series=resource_series,
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
        timings={
            "start": "2024-01-01", "end": "2024-01-02",
            "total_s": 100.0, "greedy_s": 90.0,
        },
    )


def _xy_hist(val: int = 1) -> list:
    """8×8 histogram with uniform counts for testing."""
    return [[val] * 8 for _ in range(8)]


# ---------------------------------------------------------------------------
# Flag constants
# ---------------------------------------------------------------------------

class TestSC2AnalyticsFlags(unittest.TestCase):

    def test_supports_throttle_is_false(self):
        self.assertFalse(SUPPORTS_THROTTLE)

    def test_supports_path_is_false(self):
        self.assertFalse(SUPPORTS_PATH)


# ---------------------------------------------------------------------------
# GreedySimResult new fields
# ---------------------------------------------------------------------------

class TestGreedySimResultNewFields(unittest.TestCase):

    def test_action_counts_default_none(self):
        s = GreedySimResult(
            sim=1, reward=5.0, improved=False,
            throttle_counts=[0, 0, 0], total_steps=1,
        )
        self.assertIsNone(s.action_counts)

    def test_obs_averages_default_none(self):
        s = GreedySimResult(
            sim=1, reward=5.0, improved=False,
            throttle_counts=[0, 0, 0], total_steps=1,
        )
        self.assertIsNone(s.obs_averages)

    def test_xy_hist_default_none(self):
        s = GreedySimResult(
            sim=1, reward=5.0, improved=False,
            throttle_counts=[0, 0, 0], total_steps=1,
        )
        self.assertIsNone(s.xy_hist)

    def test_action_counts_stored(self):
        counts = {0: 10, 1: 5, 2: 85}
        s = _make_sim(action_counts=counts)
        self.assertEqual(s.action_counts, counts)

    def test_obs_averages_stored(self):
        avgs = {"army_count": 3.5, "minerals": 200.0}
        s = _make_sim(obs_averages=avgs)
        self.assertEqual(s.obs_averages, avgs)

    def test_xy_hist_stored(self):
        hist = _xy_hist(3)
        s = _make_sim(xy_hist=hist)
        self.assertEqual(s.xy_hist, hist)


# ---------------------------------------------------------------------------
# plot_action_frequency
# ---------------------------------------------------------------------------

class TestPlotActionFrequency(unittest.TestCase):

    def test_renders_to_file(self):
        sims = [
            _make_sim(1, action_counts={0: 10, 1: 5, 2: 85}),
            _make_sim(2, action_counts={0: 5, 1: 10, 2: 85}, improved=True),
            _make_sim(3, action_counts={2: 100}),
        ]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_action_frequency(data, d)
            self.assertIn("action_frequency.png", os.listdir(d))

    def test_skips_when_no_action_counts(self):
        sims = [_make_sim(i) for i in range(1, 4)]  # action_counts=None
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_action_frequency(data, d)
            self.assertNotIn("action_frequency.png", os.listdir(d))

    def test_skips_when_no_sims(self):
        data = _make_experiment([])
        with tempfile.TemporaryDirectory() as d:
            plot_action_frequency(data, d)
            self.assertEqual(os.listdir(d), [])

    def test_single_fn_idx_only(self):
        """Only one action type — entropy should be 0."""
        sims = [_make_sim(1, action_counts={2: 100})]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_action_frequency(data, d)
            self.assertIn("action_frequency.png", os.listdir(d))


# ---------------------------------------------------------------------------
# plot_obs_averages
# ---------------------------------------------------------------------------

class TestPlotObsAverages(unittest.TestCase):

    def test_renders_to_file(self):
        sims = [
            _make_sim(1, obs_averages={"army_count": 2.0, "minerals": 150.0}),
            _make_sim(2, obs_averages={"army_count": 3.0, "minerals": 200.0},
                      improved=True),
            _make_sim(3, obs_averages={"army_count": 4.0, "minerals": 180.0}),
        ]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_obs_averages(data, d)
            self.assertIn("obs_averages.png", os.listdir(d))

    def test_skips_when_no_obs_averages(self):
        sims = [_make_sim(i) for i in range(1, 4)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_obs_averages(data, d)
            self.assertNotIn("obs_averages.png", os.listdir(d))

    def test_skips_when_all_zero(self):
        """All values 0 → no active features → nothing written."""
        sims = [_make_sim(1, obs_averages={"army_count": 0.0})]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_obs_averages(data, d)
            self.assertNotIn("obs_averages.png", os.listdir(d))

    def test_unknown_feature_key_does_not_crash(self):
        """Unexpected feature keys are silently skipped (no label → omitted)."""
        sims = [_make_sim(1, obs_averages={"army_count": 5.0,
                                           "some_new_feature": 99.0})]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_obs_averages(data, d)
            self.assertIn("obs_averages.png", os.listdir(d))


# ---------------------------------------------------------------------------
# plot_spatial_heatmap
# ---------------------------------------------------------------------------

class TestPlotSpatialHeatmap(unittest.TestCase):

    def test_renders_to_file(self):
        sims = [
            _make_sim(1, xy_hist=_xy_hist(5)),
            _make_sim(2, xy_hist=_xy_hist(10)),
        ]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_spatial_heatmap(data, d)
            self.assertIn("spatial_heatmap.png", os.listdir(d))

    def test_skips_when_no_xy_hist(self):
        sims = [_make_sim(i) for i in range(1, 4)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_spatial_heatmap(data, d)
            self.assertNotIn("spatial_heatmap.png", os.listdir(d))

    def test_skips_when_all_zero_hist(self):
        sims = [_make_sim(1, xy_hist=_xy_hist(0))]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_spatial_heatmap(data, d)
            self.assertNotIn("spatial_heatmap.png", os.listdir(d))

    def test_partial_xy_hist_none_sims_ignored(self):
        sims = [
            _make_sim(1, xy_hist=_xy_hist(3)),
            _make_sim(2),                      # xy_hist=None
        ]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_spatial_heatmap(data, d)
            self.assertIn("spatial_heatmap.png", os.listdir(d))


# ---------------------------------------------------------------------------
# plot_outcome_breakdown
# ---------------------------------------------------------------------------

class TestPlotOutcomeBreakdown(unittest.TestCase):

    def test_renders_to_file(self):
        sims = [
            _make_sim(1, termination_reason="timeout"),
            _make_sim(2, termination_reason="win"),
            _make_sim(3, termination_reason="loss"),
            _make_sim(4, termination_reason="finish"),
        ]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_outcome_breakdown(data, d)
            self.assertIn("outcome_breakdown.png", os.listdir(d))

    def test_skips_when_all_none_reasons(self):
        sims = [_make_sim(i, termination_reason=None) for i in range(1, 4)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_outcome_breakdown(data, d)
            self.assertNotIn("outcome_breakdown.png", os.listdir(d))

    def test_skips_when_no_sims(self):
        data = _make_experiment([])
        with tempfile.TemporaryDirectory() as d:
            plot_outcome_breakdown(data, d)
            self.assertEqual(os.listdir(d), [])

    def test_ladder_win_loss(self):
        sims = [
            _make_sim(1, termination_reason="win"),
            _make_sim(2, termination_reason="loss"),
        ]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_outcome_breakdown(data, d)
            self.assertIn("outcome_breakdown.png", os.listdir(d))


# ---------------------------------------------------------------------------
# save_experiment_results — integration
# ---------------------------------------------------------------------------

class TestSaveExperimentResults(unittest.TestCase):

    def _make_full_sims(self) -> list:
        series_pts = [[float(i) * 0.5, 100.0 + i * 5] for i in range(20)]
        army_pts   = [[float(i) * 0.5, float(i // 3)] for i in range(20)]
        build_evt  = [[2.0, "Marine"], [5.0, "SCV"], [8.0, "Marine"]]
        return [
            _make_sim(1, reward=5.0,
                      action_counts={0: 20, 1: 5, 2: 75},
                      obs_averages={"army_count": 2.0, "minerals": 100.0},
                      xy_hist=_xy_hist(4),
                      termination_reason="timeout",
                      reward_components={"score": 5.0, "step_penalty": -0.5},
                      supply_capped_fraction=0.3,
                      resource_series=series_pts,
                      army_count_series=army_pts,
                      build_order=build_evt),
            _make_sim(2, reward=8.0, improved=True,
                      action_counts={0: 5, 1: 10, 2: 85},
                      obs_averages={"army_count": 3.0, "minerals": 150.0},
                      xy_hist=_xy_hist(6),
                      termination_reason="finish",
                      reward_components={"score": 8.5, "step_penalty": -0.5},
                      supply_capped_fraction=0.1,
                      resource_series=series_pts,
                      army_count_series=army_pts,
                      build_order=build_evt),
        ]

    def test_writes_results_md(self):
        data = _make_experiment(self._make_full_sims())
        with tempfile.TemporaryDirectory() as d:
            save_experiment_results(data, d)
            self.assertIn("results.md", os.listdir(d))

    def test_writes_sc2_plots(self):
        data = _make_experiment(self._make_full_sims())
        with tempfile.TemporaryDirectory() as d:
            save_experiment_results(data, d)
            files = os.listdir(d)
            self.assertIn("greedy_rewards.png", files)
            self.assertIn("reward_trajectory.png", files)
            self.assertIn("action_frequency.png", files)
            self.assertIn("obs_averages.png", files)
            self.assertIn("spatial_heatmap.png", files)
            self.assertIn("outcome_breakdown.png", files)
            self.assertIn("supply_capped.png", files)
            self.assertIn("resource_series.png", files)
            self.assertIn("army_count.png", files)
            self.assertIn("build_order.png", files)

    def test_results_md_mentions_game(self):
        data = _make_experiment(self._make_full_sims())
        with tempfile.TemporaryDirectory() as d:
            save_experiment_results(data, d)
            with open(os.path.join(d, "results.md")) as f:
                md = f.read()
            self.assertIn("StarCraft 2", md)

    def test_no_crash_with_empty_sims(self):
        data = _make_experiment([])
        with tempfile.TemporaryDirectory() as d:
            save_experiment_results(data, d)
            self.assertIn("results.md", os.listdir(d))

    def test_no_racing_plots_written(self):
        """No TMNF/TORCS-specific files should appear in the output dir."""
        data = _make_experiment(self._make_full_sims())
        with tempfile.TemporaryDirectory() as d:
            save_experiment_results(data, d)
            files = set(os.listdir(d))
            racing_files = {
                "greedy_best_run.png",
                "greedy_weight_evolution.png",
                "cold_start_best_run.png",
            }
            self.assertTrue(
                files.isdisjoint(racing_files),
                f"Racing-specific files found: {files & racing_files}",
            )


# ---------------------------------------------------------------------------
# GreedySimResult new fields (supply_capped_fraction, build_order, series)
# ---------------------------------------------------------------------------

class TestGreedySimResultEndScreenFields(unittest.TestCase):

    def test_supply_capped_fraction_default_none(self):
        s = GreedySimResult(
            sim=1, reward=5.0, improved=False,
            throttle_counts=[0, 0, 0], total_steps=1,
        )
        self.assertIsNone(s.supply_capped_fraction)

    def test_build_order_default_none(self):
        s = GreedySimResult(
            sim=1, reward=5.0, improved=False,
            throttle_counts=[0, 0, 0], total_steps=1,
        )
        self.assertIsNone(s.build_order)

    def test_army_count_series_default_none(self):
        s = GreedySimResult(
            sim=1, reward=5.0, improved=False,
            throttle_counts=[0, 0, 0], total_steps=1,
        )
        self.assertIsNone(s.army_count_series)

    def test_resource_series_default_none(self):
        s = GreedySimResult(
            sim=1, reward=5.0, improved=False,
            throttle_counts=[0, 0, 0], total_steps=1,
        )
        self.assertIsNone(s.resource_series)

    def test_fields_stored_correctly(self):
        bo = [[1.0, "Marine"], [5.0, "SCV"]]
        army = [[0.5, 0.0], [1.0, 1.0]]
        res  = [[0.5, 200.0], [1.0, 180.0]]
        s = _make_sim(
            supply_capped_fraction=0.25,
            build_order=bo,
            army_count_series=army,
            resource_series=res,
        )
        self.assertAlmostEqual(s.supply_capped_fraction, 0.25)
        self.assertEqual(s.build_order, bo)
        self.assertEqual(s.army_count_series, army)
        self.assertEqual(s.resource_series, res)


# ---------------------------------------------------------------------------
# plot_supply_capped
# ---------------------------------------------------------------------------

class TestPlotSupplyCapped(unittest.TestCase):

    def test_renders_to_file(self):
        sims = [
            _make_sim(1, supply_capped_fraction=0.1),
            _make_sim(2, supply_capped_fraction=0.4, improved=True),
            _make_sim(3, supply_capped_fraction=0.7),
        ]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_supply_capped(data, d)
            self.assertIn("supply_capped.png", os.listdir(d))

    def test_skips_when_all_none(self):
        sims = [_make_sim(i) for i in range(1, 4)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_supply_capped(data, d)
            self.assertNotIn("supply_capped.png", os.listdir(d))

    def test_skips_when_no_sims(self):
        data = _make_experiment([])
        with tempfile.TemporaryDirectory() as d:
            plot_supply_capped(data, d)
            self.assertEqual(os.listdir(d), [])

    def test_zero_fraction_renders(self):
        """All zeros is valid (supply was never capped)."""
        sims = [_make_sim(1, supply_capped_fraction=0.0)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_supply_capped(data, d)
            self.assertIn("supply_capped.png", os.listdir(d))


# ---------------------------------------------------------------------------
# plot_resource_series
# ---------------------------------------------------------------------------

_SERIES = [[float(i) * 0.5, 100.0 + i * 10] for i in range(20)]
_ARMY   = [[float(i) * 0.5, float(i // 3)]   for i in range(20)]
_BUILD  = [[2.0, "Marine"], [4.0, "SCV"], [7.0, "Marine"]]


class TestPlotResourceSeries(unittest.TestCase):

    def test_renders_to_file(self):
        sims = [
            _make_sim(1, resource_series=_SERIES),
            _make_sim(2, resource_series=_SERIES, improved=True),
        ]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_resource_series(data, d)
            self.assertIn("resource_series.png", os.listdir(d))

    def test_uses_best_sim(self):
        """Only one file written even with multiple sims."""
        sims = [
            _make_sim(1, resource_series=_SERIES),
            _make_sim(2, resource_series=_SERIES, improved=True),
            _make_sim(3, resource_series=_SERIES),
        ]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_resource_series(data, d)
            self.assertIn("resource_series.png", os.listdir(d))

    def test_skips_when_no_series(self):
        sims = [_make_sim(i) for i in range(1, 4)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_resource_series(data, d)
            self.assertNotIn("resource_series.png", os.listdir(d))

    def test_skips_when_no_sims(self):
        data = _make_experiment([])
        with tempfile.TemporaryDirectory() as d:
            plot_resource_series(data, d)
            self.assertEqual(os.listdir(d), [])

    def test_falls_back_to_last_sim_when_none_improved(self):
        """If no sim is improved, the last sim's series is shown."""
        sims = [_make_sim(1, resource_series=_SERIES)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_resource_series(data, d)
            self.assertIn("resource_series.png", os.listdir(d))


# ---------------------------------------------------------------------------
# plot_army_count
# ---------------------------------------------------------------------------

class TestPlotArmyCount(unittest.TestCase):

    def test_renders_to_file(self):
        sims = [_make_sim(1, army_count_series=_ARMY, improved=True)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_army_count(data, d)
            self.assertIn("army_count.png", os.listdir(d))

    def test_skips_when_no_series(self):
        sims = [_make_sim(i) for i in range(1, 4)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_army_count(data, d)
            self.assertNotIn("army_count.png", os.listdir(d))

    def test_skips_when_no_sims(self):
        data = _make_experiment([])
        with tempfile.TemporaryDirectory() as d:
            plot_army_count(data, d)
            self.assertEqual(os.listdir(d), [])


# ---------------------------------------------------------------------------
# plot_build_order
# ---------------------------------------------------------------------------

class TestPlotBuildOrder(unittest.TestCase):

    def test_renders_to_file(self):
        sims = [_make_sim(1, build_order=_BUILD, improved=True)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_build_order(data, d)
            self.assertIn("build_order.png", os.listdir(d))

    def test_skips_when_no_build_order(self):
        sims = [_make_sim(i) for i in range(1, 4)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_build_order(data, d)
            self.assertNotIn("build_order.png", os.listdir(d))

    def test_skips_when_no_sims(self):
        data = _make_experiment([])
        with tempfile.TemporaryDirectory() as d:
            plot_build_order(data, d)
            self.assertEqual(os.listdir(d), [])

    def test_single_unit_type(self):
        """Build order with one unit type renders without crash."""
        events = [[float(i), "Marine"] for i in range(5)]
        sims = [_make_sim(1, build_order=events)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_build_order(data, d)
            self.assertIn("build_order.png", os.listdir(d))

    def test_multiple_unit_types(self):
        """Build order with multiple unit types renders correctly."""
        events = [[1.0, "Marine"], [2.0, "SCV"], [3.0, "Zergling"],
                  [4.0, "Marine"], [5.0, "SCV"]]
        sims = [_make_sim(1, build_order=events, improved=True)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_build_order(data, d)
            self.assertIn("build_order.png", os.listdir(d))


if __name__ == "__main__":
    unittest.main(verbosity=2)
