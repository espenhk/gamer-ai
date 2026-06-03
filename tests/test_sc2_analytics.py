"""Tests for the SC2-specific analytics module (issue improvements 2a-2i).

Covers:
- SUPPORTS_THROTTLE / SUPPORTS_PATH flags
- plot_action_frequency: renders without error; skips when no data
- plot_obs_averages: renders without error; skips when no data
- plot_spatial_heatmap: renders without error; skips when no data
- plot_outcome_breakdown: renders without error; skips when no data
- plot_skipped_frames: renders without error; skips when no data
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
from unittest import mock

import games.sc2.analytics as sc2_analytics
from framework.analytics import ExperimentData, GreedySimResult
from games.sc2.analytics import (
    SUPPORTS_PATH,
    SUPPORTS_THROTTLE,
    plot_action_frequency,
    plot_army_count,
    plot_build_order,
    plot_gs_reward_component_breakdown,
    plot_obs_averages,
    plot_outcome_breakdown,
    plot_resource_series,
    plot_skipped_frames,
    plot_spatial_heatmap,
    plot_supply_capped,
    save_experiment_results,
    save_grid_summary,
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
    skipped_frames: int | None = None,
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
        skipped_frames=skipped_frames,
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
            "start": "2024-01-01",
            "end": "2024-01-02",
            "total_s": 100.0,
            "greedy_s": 90.0,
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
            sim=1,
            reward=5.0,
            improved=False,
            throttle_counts=[0, 0, 0],
            total_steps=1,
        )
        self.assertIsNone(s.action_counts)

    def test_obs_averages_default_none(self):
        s = GreedySimResult(
            sim=1,
            reward=5.0,
            improved=False,
            throttle_counts=[0, 0, 0],
            total_steps=1,
        )
        self.assertIsNone(s.obs_averages)

    def test_xy_hist_default_none(self):
        s = GreedySimResult(
            sim=1,
            reward=5.0,
            improved=False,
            throttle_counts=[0, 0, 0],
            total_steps=1,
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

    def test_skipped_frames_default_none(self):
        s = GreedySimResult(
            sim=1,
            reward=5.0,
            improved=False,
            throttle_counts=[0, 0, 0],
            total_steps=1,
        )
        self.assertIsNone(s.skipped_frames)

    def test_skipped_frames_stored(self):
        s = _make_sim(skipped_frames=7)
        self.assertEqual(s.skipped_frames, 7)


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

    def test_no_op_only_skips_plot(self):
        """Issue #382: all-no_op action counts → nothing to plot → file not written."""
        sims = [_make_sim(1, action_counts={0: 500})]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_action_frequency(data, d)
            self.assertNotIn("action_frequency.png", os.listdir(d))

    def test_no_op_excluded_from_plot(self):
        """Issue #382: no_op (fn_idx 0) counts are stripped before rendering."""
        sims = [_make_sim(1, action_counts={0: 9000, 2: 100})]
        data = _make_experiment(sims)
        captured: dict = {}

        def _capture(fig, _path):
            # The legend labels of the stacked bar (panel 1) must not include no_op.
            ax = fig.axes[0]
            legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
            captured["labels"] = legend_texts

        with mock.patch.object(sc2_analytics, "_save", side_effect=_capture):
            plot_action_frequency(data, "/tmp")

        self.assertIn("labels", captured)
        self.assertNotIn("no_op", captured["labels"])
        self.assertIn("Move_screen", captured["labels"])


# ---------------------------------------------------------------------------
# plot_obs_averages
# ---------------------------------------------------------------------------


class TestPlotObsAverages(unittest.TestCase):
    def test_renders_to_file(self):
        sims = [
            _make_sim(1, obs_averages={"army_count": 2.0, "minerals": 150.0}),
            _make_sim(2, obs_averages={"army_count": 3.0, "minerals": 200.0}, improved=True),
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
        sims = [_make_sim(1, obs_averages={"army_count": 5.0, "some_new_feature": 99.0})]
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
            _make_sim(2),  # xy_hist=None
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
# plot_skipped_frames
# ---------------------------------------------------------------------------


class TestPlotSkippedFrames(unittest.TestCase):
    def test_renders_to_file(self):
        sims = [
            _make_sim(1, skipped_frames=0),
            _make_sim(2, skipped_frames=4, improved=True),
            _make_sim(3, skipped_frames=10),
        ]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_skipped_frames(data, d)
            self.assertIn("skipped_frames.png", os.listdir(d))

    def test_skips_when_all_none(self):
        sims = [_make_sim(i) for i in range(1, 4)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_skipped_frames(data, d)
            self.assertNotIn("skipped_frames.png", os.listdir(d))

    def test_zero_only_still_renders(self):
        sims = [_make_sim(1, skipped_frames=0)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_skipped_frames(data, d)
            self.assertIn("skipped_frames.png", os.listdir(d))

    def test_negative_skipped_frames_clamped_to_zero_for_plotting(self):
        sims = [_make_sim(1, skipped_frames=-5)]
        data = _make_experiment(sims)
        captured = {}

        def _capture_save(fig, _path):
            captured["heights"] = [patch.get_height() for patch in fig.axes[0].patches]

        with mock.patch.object(sc2_analytics, "_save", side_effect=_capture_save):
            plot_skipped_frames(data, "/tmp")
        self.assertEqual(captured["heights"], [0])


# ---------------------------------------------------------------------------
# save_experiment_results — integration
# ---------------------------------------------------------------------------


class TestSaveExperimentResults(unittest.TestCase):
    def _make_full_sims(self) -> list:
        series_pts = [[float(i) * 0.5, 100.0 + i * 5] for i in range(20)]
        army_pts = [[float(i) * 0.5, float(i // 3)] for i in range(20)]
        build_evt = [[2.0, "Marine"], [5.0, "SCV"], [8.0, "Marine"]]
        return [
            _make_sim(
                1,
                reward=5.0,
                action_counts={0: 20, 1: 5, 2: 75},
                obs_averages={"army_count": 2.0, "minerals": 100.0},
                xy_hist=_xy_hist(4),
                termination_reason="timeout",
                reward_components={"score": 5.0, "step_penalty": -0.5},
                skipped_frames=3,
                supply_capped_fraction=0.3,
                resource_series=series_pts,
                army_count_series=army_pts,
                build_order=build_evt,
            ),
            _make_sim(
                2,
                reward=8.0,
                improved=True,
                action_counts={0: 5, 1: 10, 2: 85},
                obs_averages={"army_count": 3.0, "minerals": 150.0},
                xy_hist=_xy_hist(6),
                termination_reason="finish",
                reward_components={"score": 8.5, "step_penalty": -0.5},
                skipped_frames=1,
                supply_capped_fraction=0.1,
                resource_series=series_pts,
                army_count_series=army_pts,
                build_order=build_evt,
            ),
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
            self.assertIn("skipped_frames.png", files)
            self.assertIn("supply_capped.png", files)
            self.assertIn("resource_series.png", files)
            self.assertIn("army_count.png", files)
            self.assertIn("build_order.png", files)
            self.assertIn("reward_component_breakdown.png", files)

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


class TestSaveGridSummary(unittest.TestCase):
    def test_passes_sc2_extra_plots_hook_to_framework(self):
        with tempfile.TemporaryDirectory() as d:
            data = _make_experiment([_make_sim(sim=1, reward=1.0)], name="exp_hook")
            with mock.patch.object(sc2_analytics, "_framework_save_grid_summary") as m:
                save_grid_summary([("exp_hook", data)], [], d, "gs_test")

            self.assertIn("extra_plots_fn", m.call_args.kwargs)
            self.assertTrue(callable(m.call_args.kwargs["extra_plots_fn"]))

    def test_normalises_rewards_from_components_before_summary(self):
        # score is scaled by score_weight; step_penalty is scaled by its weight (floor=0.001).
        # Here step_penalty weight=-2.0 (well above floor), so divided by 2.0.
        with tempfile.TemporaryDirectory() as d:
            reward_cfg_path = os.path.join(d, "reward_config.yaml")
            with open(reward_cfg_path, "w", encoding="utf-8") as f:
                f.write("score_weight: 100.0\nstep_penalty: -2.0\n")

            sim = _make_sim(
                sim=1,
                reward=496.0,
                reward_components={"score": 500.0, "step_penalty": -4.0},
            )
            data = _make_experiment([sim], name="exp_a")
            data.reward_config_file = reward_cfg_path

            with mock.patch.object(sc2_analytics, "_framework_save_grid_summary") as m:
                save_grid_summary([("exp_a", data)], ["score_weight"], d, "gs_test")

            forwarded_runs = m.call_args.args[0]
            forwarded_sim = forwarded_runs[0][1].greedy_sims[0]
            # score: 500.0/100.0 = 5.0; step_penalty: -4.0/2.0 = -2.0 → total = 3.0
            self.assertAlmostEqual(forwarded_sim.reward, 3.0, places=6)
            self.assertEqual(
                forwarded_sim.reward_components,
                {"score": 5.0, "step_penalty": -2.0},
            )

    def test_falls_back_to_raw_reward_without_components(self):
        with tempfile.TemporaryDirectory() as d:
            sim = _make_sim(sim=1, reward=12.5, reward_components=None)
            data = _make_experiment([sim], name="exp_b")

            with mock.patch.object(sc2_analytics, "_framework_save_grid_summary") as m:
                save_grid_summary([("exp_b", data)], [], d, "gs_test")

            forwarded_runs = m.call_args.args[0]
            forwarded_sim = forwarded_runs[0][1].greedy_sims[0]
            self.assertAlmostEqual(forwarded_sim.reward, 12.5, places=6)

    def test_normalises_multiple_sims_independently(self):
        # step_penalty weight=-2.0 (above floor), so divided by 2.0 as normal.
        with tempfile.TemporaryDirectory() as d:
            reward_cfg_path = os.path.join(d, "reward_config.yaml")
            with open(reward_cfg_path, "w", encoding="utf-8") as f:
                f.write("score_weight: 10.0\nstep_penalty: -2.0\n")

            sims = [
                _make_sim(sim=1, reward=20.0, reward_components={"score": 20.0}),
                _make_sim(sim=2, reward=-6.0, reward_components={"step_penalty": -6.0}),
            ]
            data = _make_experiment(sims, name="exp_c")
            data.reward_config_file = reward_cfg_path

            with mock.patch.object(sc2_analytics, "_framework_save_grid_summary") as m:
                save_grid_summary([("exp_c", data)], [], d, "gs_test")

            forwarded_runs = m.call_args.args[0]
            forwarded_sims = forwarded_runs[0][1].greedy_sims
            # sim1: score 20/10 = 2.0; sim2: step_penalty -6/2.0 = -3.0
            self.assertAlmostEqual(forwarded_sims[0].reward, 2.0, places=6)
            self.assertAlmostEqual(forwarded_sims[1].reward, -3.0, places=6)

    def test_malformed_reward_config_yaml_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            reward_cfg_path = os.path.join(d, "reward_config.yaml")
            with open(reward_cfg_path, "w", encoding="utf-8") as f:
                f.write("score_weight: [1.0\n")

            sim = _make_sim(sim=1, reward=5.0, reward_components={"score": 5.0})
            data = _make_experiment([sim], name="exp_d")
            data.reward_config_file = reward_cfg_path

            with mock.patch.object(sc2_analytics, "_framework_save_grid_summary") as m:
                save_grid_summary([("exp_d", data)], [], d, "gs_test")

            forwarded_runs = m.call_args.args[0]
            forwarded_sim = forwarded_runs[0][1].greedy_sims[0]
            self.assertAlmostEqual(forwarded_sim.reward, 5.0, places=6)

    def test_non_mapping_reward_config_yaml_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            reward_cfg_path = os.path.join(d, "reward_config.yaml")
            with open(reward_cfg_path, "w", encoding="utf-8") as f:
                f.write("[]\n")

            sim = _make_sim(sim=1, reward=5.0, reward_components={"score": 5.0})
            data = _make_experiment([sim], name="exp_e")
            data.reward_config_file = reward_cfg_path

            with mock.patch.object(sc2_analytics, "_framework_save_grid_summary") as m:
                save_grid_summary([("exp_e", data)], [], d, "gs_test")

            forwarded_runs = m.call_args.args[0]
            forwarded_sim = forwarded_runs[0][1].greedy_sims[0]
            self.assertAlmostEqual(forwarded_sim.reward, 5.0, places=6)

    def test_non_numeric_reward_weight_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            reward_cfg_path = os.path.join(d, "reward_config.yaml")
            with open(reward_cfg_path, "w", encoding="utf-8") as f:
                f.write("score_weight: foo\n")

            sim = _make_sim(sim=1, reward=5.0, reward_components={"score": 5.0})
            data = _make_experiment([sim], name="exp_f")
            data.reward_config_file = reward_cfg_path

            with mock.patch.object(sc2_analytics, "_framework_save_grid_summary") as m:
                save_grid_summary([("exp_f", data)], [], d, "gs_test")

            forwarded_runs = m.call_args.args[0]
            forwarded_sim = forwarded_runs[0][1].greedy_sims[0]
            self.assertAlmostEqual(forwarded_sim.reward, 5.0, places=6)

    def test_scout_component_does_not_emit_unmapped_warning(self):
        with tempfile.TemporaryDirectory() as d:
            sim = _make_sim(sim=1, reward=2.0, reward_components={"scout": 2.0})
            data = _make_experiment([sim], name="exp_g")

            with (
                mock.patch.object(sc2_analytics, "_framework_save_grid_summary"),
                mock.patch.object(sc2_analytics.logger, "warning") as warn,
            ):
                save_grid_summary([("exp_g", data)], [], d, "gs_test")

            self.assertTrue(all("unmapped reward component" not in str(call.args[0]) for call in warn.call_args_list))

    def test_scout_explore_component_does_not_emit_unmapped_warning(self):
        # The reward calculator's map-exploration term ("scout_explore") is a
        # distinct, normally-scaled component from the belief "scout" reward.
        with tempfile.TemporaryDirectory() as d:
            sim = _make_sim(sim=1, reward=2.0, reward_components={"scout_explore": 2.0})
            data = _make_experiment([sim], name="exp_g2")

            with (
                mock.patch.object(sc2_analytics, "_framework_save_grid_summary"),
                mock.patch.object(sc2_analytics.logger, "warning") as warn,
            ):
                save_grid_summary([("exp_g2", data)], [], d, "gs_test")

            self.assertTrue(all("unmapped reward component" not in str(call.args[0]) for call in warn.call_args_list))

    def test_step_penalty_with_weight_below_1_passes_through_raw(self):
        """step_penalty weight < 1.0 → scale=1.0 → raw value passes through unchanged.

        The raw contribution already encodes the weight (raw = steps × weight),
        so dividing by the weight would amplify it 1000× and swamp other signals.
        """
        with tempfile.TemporaryDirectory() as d:
            reward_cfg_path = os.path.join(d, "reward_config.yaml")
            with open(reward_cfg_path, "w", encoding="utf-8") as f:
                f.write("step_penalty: -0.001\n")

            sim = _make_sim(
                sim=1,
                reward=-0.5,
                reward_components={"step_penalty": -0.5},
            )
            data = _make_experiment([sim], name="exp_sp")
            data.reward_config_file = reward_cfg_path

            with mock.patch.object(sc2_analytics, "_framework_save_grid_summary") as m:
                save_grid_summary([("exp_sp", data)], [], d, "gs_test")

            forwarded_runs = m.call_args.args[0]
            forwarded_sim = forwarded_runs[0][1].greedy_sims[0]
            # scale = max(0.001, 1.0) = 1.0 → -0.5 / 1.0 = -0.5 (not the amplified -500)
            self.assertAlmostEqual(forwarded_sim.reward, -0.5, places=6)

    def test_idle_penalty_with_weight_below_1_passes_through_raw(self):
        """idle_penalty weight < 1.0 → scale=1.0 → raw value passes through unchanged."""
        with tempfile.TemporaryDirectory() as d:
            reward_cfg_path = os.path.join(d, "reward_config.yaml")
            with open(reward_cfg_path, "w", encoding="utf-8") as f:
                f.write("idle_penalty: -0.001\n")

            sim = _make_sim(
                sim=1,
                reward=-0.3,
                reward_components={"idle_penalty": -0.3},
            )
            data = _make_experiment([sim], name="exp_ip")
            data.reward_config_file = reward_cfg_path

            with mock.patch.object(sc2_analytics, "_framework_save_grid_summary") as m:
                save_grid_summary([("exp_ip", data)], [], d, "gs_test")

            forwarded_runs = m.call_args.args[0]
            forwarded_sim = forwarded_runs[0][1].greedy_sims[0]
            # scale = max(0.001, 1.0) = 1.0 → -0.3 / 1.0 = -0.3 (not -300)
            self.assertAlmostEqual(forwarded_sim.reward, -0.3, places=6)

    def test_sub_1_weights_compared_by_proportional_raw_contribution(self):
        """Two sub-1.0 weights (0.001 vs 0.002) both use scale=1.0, so runs are
        compared by their raw contributions — the larger weight naturally produces
        a larger raw value, preserving the correct relative ordering.
        """
        with tempfile.TemporaryDirectory() as d:
            reward_cfg_path = os.path.join(d, "reward_config.yaml")
            with open(reward_cfg_path, "w", encoding="utf-8") as f:
                # weight=0.0001 — still sub-1.0 — should also pass through raw
                f.write("step_penalty: -0.0001\n")

            sim = _make_sim(
                sim=1,
                reward=-0.5,
                reward_components={"step_penalty": -0.5},
            )
            data = _make_experiment([sim], name="exp_small")
            data.reward_config_file = reward_cfg_path

            with mock.patch.object(sc2_analytics, "_framework_save_grid_summary") as m:
                save_grid_summary([("exp_small", data)], [], d, "gs_test")

            forwarded_runs = m.call_args.args[0]
            forwarded_sim = forwarded_runs[0][1].greedy_sims[0]
            # scale = max(0.0001, 1.0) = 1.0 → -0.5 / 1.0 = -0.5
            self.assertAlmostEqual(forwarded_sim.reward, -0.5, places=6)

    def test_tiny_weight_does_not_make_positive_reward_negative(self):
        """Realistic SC2 sim: idle_bonus >> step_penalty → normalized reward stays positive."""
        with tempfile.TemporaryDirectory() as d:
            reward_cfg_path = os.path.join(d, "reward_config.yaml")
            with open(reward_cfg_path, "w", encoding="utf-8") as f:
                f.write("idle_bonus: 0.5\nstep_penalty: -0.001\nscore_weight: 1.0\n")

            # Mirrors a real SC2 sim: 648 idle steps, 1504 total steps, score=-9.
            sim = _make_sim(
                sim=1,
                reward=313.5,
                reward_components={
                    "score": -9.0,
                    "idle_bonus": 324.0,  # 648 idle steps × 0.5
                    "step_penalty": -1.504,  # 1504 steps × 0.001
                },
            )
            data = _make_experiment([sim], name="exp_real")
            data.reward_config_file = reward_cfg_path

            with mock.patch.object(sc2_analytics, "_framework_save_grid_summary") as m:
                save_grid_summary([("exp_real", data)], [], d, "gs_test")

            forwarded_runs = m.call_args.args[0]
            forwarded_sim = forwarded_runs[0][1].greedy_sims[0]
            # score:       -9 / max(1.0, 1.0) = -9.0
            # idle_bonus: 324 / max(0.5, 1.0) = 324.0   (sub-1.0 → raw)
            # step_penalty: -1.504 / max(0.001, 1.0) = -1.504  (sub-1.0 → raw)
            # total = -9.0 + 324.0 - 1.504 = 313.496  (positive ✓, not the old -865)
            self.assertGreater(forwarded_sim.reward, 0.0)
            self.assertAlmostEqual(forwarded_sim.reward, 313.496, places=2)

    def test_new_penalty_components_are_normalized_via_config_weights(self):
        with tempfile.TemporaryDirectory() as d:
            reward_cfg_path = os.path.join(d, "reward_config.yaml")
            with open(reward_cfg_path, "w", encoding="utf-8") as f:
                f.write("unit_loss_penalty: -3.0\ndamage_taken_penalty: -4.0\npassive_under_fire_penalty: -2.0\n")

            sim = _make_sim(
                sim=1,
                reward=-23.0,
                reward_components={
                    "unit_loss": -9.0,
                    "damage_taken": -8.0,
                    "passive_under_fire": -6.0,
                },
            )
            data = _make_experiment([sim], name="exp_penalties")
            data.reward_config_file = reward_cfg_path

            with (
                mock.patch.object(sc2_analytics, "_framework_save_grid_summary") as m,
                mock.patch.object(sc2_analytics.logger, "warning") as warn,
            ):
                save_grid_summary([("exp_penalties", data)], [], d, "gs_test")

            forwarded_runs = m.call_args.args[0]
            forwarded_sim = forwarded_runs[0][1].greedy_sims[0]
            self.assertAlmostEqual(forwarded_sim.reward, -8.0, places=6)
            warn.assert_not_called()

    def test_writes_sc2_cross_run_summary_charts(self):
        with tempfile.TemporaryDirectory() as d:
            runs = [
                (
                    "exp_a",
                    _make_experiment(
                        [
                            _make_sim(
                                sim=1,
                                reward=5.0,
                                action_counts={0: 20, 2: 80},
                                termination_reason="win",
                                skipped_frames=2,
                                supply_capped_fraction=0.2,
                                xy_hist=_xy_hist(3),
                            ),
                            _make_sim(
                                sim=2,
                                reward=6.0,
                                action_counts={0: 50, 2: 50},
                                termination_reason="finish",
                                skipped_frames=1,
                                supply_capped_fraction=0.1,
                                xy_hist=_xy_hist(4),
                            ),
                        ],
                        name="exp_a",
                    ),
                ),
                (
                    "exp_b",
                    _make_experiment(
                        [
                            _make_sim(
                                sim=1,
                                reward=1.0,
                                action_counts={0: 100},
                                termination_reason="loss",
                                skipped_frames=0,
                                supply_capped_fraction=0.7,
                                xy_hist=_xy_hist(1),
                            ),
                        ],
                        name="exp_b",
                    ),
                ),
            ]
            save_grid_summary(runs, [], d, "gs_test")

            files = set(os.listdir(d))
            self.assertIn("comparison_action_entropy.png", files)
            self.assertIn("comparison_outcomes.png", files)
            self.assertIn("comparison_skipped_frames.png", files)
            self.assertIn("comparison_supply_capped.png", files)
            self.assertIn("comparison_spatial_heatmap.png", files)

            with open(os.path.join(d, "summary.md"), encoding="utf-8") as f:
                summary = f.read()
            self.assertIn("SC2-specific cross-run charts", summary)
            self.assertIn("comparison_action_entropy.png", summary)
            self.assertIn("comparison_outcomes.png", summary)
            self.assertIn("comparison_skipped_frames.png", summary)
            self.assertIn("comparison_supply_capped.png", summary)
            self.assertIn("comparison_spatial_heatmap.png", summary)


# ---------------------------------------------------------------------------
# GreedySimResult new fields (supply_capped_fraction, build_order, series)
# ---------------------------------------------------------------------------


class TestGreedySimResultEndScreenFields(unittest.TestCase):
    def test_supply_capped_fraction_default_none(self):
        s = GreedySimResult(
            sim=1,
            reward=5.0,
            improved=False,
            throttle_counts=[0, 0, 0],
            total_steps=1,
        )
        self.assertIsNone(s.supply_capped_fraction)

    def test_build_order_default_none(self):
        s = GreedySimResult(
            sim=1,
            reward=5.0,
            improved=False,
            throttle_counts=[0, 0, 0],
            total_steps=1,
        )
        self.assertIsNone(s.build_order)

    def test_army_count_series_default_none(self):
        s = GreedySimResult(
            sim=1,
            reward=5.0,
            improved=False,
            throttle_counts=[0, 0, 0],
            total_steps=1,
        )
        self.assertIsNone(s.army_count_series)

    def test_resource_series_default_none(self):
        s = GreedySimResult(
            sim=1,
            reward=5.0,
            improved=False,
            throttle_counts=[0, 0, 0],
            total_steps=1,
        )
        self.assertIsNone(s.resource_series)

    def test_fields_stored_correctly(self):
        bo = [[1.0, "Marine"], [5.0, "SCV"]]
        army = [[0.5, 0.0], [1.0, 1.0]]
        res = [[0.5, 200.0], [1.0, 180.0]]
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
_ARMY = [[float(i) * 0.5, float(i // 3)] for i in range(20)]
_BUILD = [[2.0, "Marine"], [4.0, "SCV"], [7.0, "Marine"]]


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
        events = [[1.0, "Marine"], [2.0, "SCV"], [3.0, "Zergling"], [4.0, "Marine"], [5.0, "SCV"]]
        sims = [_make_sim(1, build_order=events, improved=True)]
        data = _make_experiment(sims)
        with tempfile.TemporaryDirectory() as d:
            plot_build_order(data, d)
            self.assertIn("build_order.png", os.listdir(d))


# ---------------------------------------------------------------------------
# _sc2_task_metric and _GS_SUCCESS_REASONS (issue #209)
# ---------------------------------------------------------------------------


class TestSC2TaskMetric(unittest.TestCase):
    def test_empty_sims_returns_zero(self):
        data = _make_experiment([])
        self.assertEqual(sc2_analytics._sc2_task_metric(data), 0.0)

    def test_win_and_finish_are_successes(self):
        sims = [
            _make_sim(1, termination_reason="win"),
            _make_sim(2, termination_reason="finish"),
            _make_sim(3, termination_reason="timeout"),
        ]
        data = _make_experiment(sims)
        self.assertAlmostEqual(sc2_analytics._sc2_task_metric(data), 2 / 3)

    def test_loss_timeout_none_other_not_counted(self):
        sims = [
            _make_sim(1, termination_reason="loss"),
            _make_sim(2, termination_reason="timeout"),
            _make_sim(3, termination_reason=None),
            _make_sim(4, termination_reason="other"),
        ]
        data = _make_experiment(sims)
        self.assertEqual(sc2_analytics._sc2_task_metric(data), 0.0)

    def test_all_wins(self):
        sims = [_make_sim(i, termination_reason="win") for i in range(5)]
        data = _make_experiment(sims)
        self.assertAlmostEqual(sc2_analytics._sc2_task_metric(data), 1.0)

    def test_success_reasons_constant_covers_win_and_finish(self):
        self.assertIn("win", sc2_analytics._GS_SUCCESS_REASONS)
        self.assertIn("finish", sc2_analytics._GS_SUCCESS_REASONS)
        self.assertNotIn("loss", sc2_analytics._GS_SUCCESS_REASONS)
        self.assertNotIn("timeout", sc2_analytics._GS_SUCCESS_REASONS)


# ---------------------------------------------------------------------------
# plot_gs_reward_component_breakdown (issue #252)
# ---------------------------------------------------------------------------


class TestPlotGsRewardComponentBreakdown(unittest.TestCase):
    def _runs_with_components(self) -> list:
        sims_a = [
            _make_sim(1, reward_components={"score": 5.0, "step_penalty": -0.5}),
            _make_sim(2, reward_components={"score": 7.0, "step_penalty": -0.8}, improved=True),
        ]
        sims_b = [
            _make_sim(1, reward_components={"score": 2.0, "step_penalty": -0.3}),
        ]
        exp_a = _make_experiment(sims_a, name="exp_a")
        exp_b = _make_experiment(sims_b, name="exp_b")
        return [("exp_a", exp_a), ("exp_b", exp_b)]

    def test_renders_to_file(self):
        runs = self._runs_with_components()
        with tempfile.TemporaryDirectory() as d:
            plot_gs_reward_component_breakdown(runs, d)
            self.assertIn("comparison_reward_breakdown.png", os.listdir(d))

    def test_skips_when_no_component_data(self):
        runs = [
            ("exp_a", _make_experiment([_make_sim(1)])),
            ("exp_b", _make_experiment([_make_sim(1)])),
        ]
        with tempfile.TemporaryDirectory() as d:
            plot_gs_reward_component_breakdown(runs, d)
            self.assertNotIn("comparison_reward_breakdown.png", os.listdir(d))

    def test_skips_when_empty_runs(self):
        with tempfile.TemporaryDirectory() as d:
            plot_gs_reward_component_breakdown([], d)
            self.assertEqual(os.listdir(d), [])

    def test_skips_when_all_components_zero(self):
        sims = [_make_sim(1, reward_components={"score": 0.0})]
        runs = [("exp_a", _make_experiment(sims))]
        with tempfile.TemporaryDirectory() as d:
            plot_gs_reward_component_breakdown(runs, d)
            self.assertNotIn("comparison_reward_breakdown.png", os.listdir(d))

    def test_single_experiment(self):
        sims = [_make_sim(1, reward_components={"score": 3.0, "step_penalty": -0.2})]
        runs = [("only_exp", _make_experiment(sims, name="only_exp"))]
        with tempfile.TemporaryDirectory() as d:
            plot_gs_reward_component_breakdown(runs, d)
            self.assertIn("comparison_reward_breakdown.png", os.listdir(d))

    def test_written_by_save_grid_summary(self):
        runs = self._runs_with_components()
        with tempfile.TemporaryDirectory() as d:
            save_grid_summary(runs, [], d, "gs_breakdown")
            self.assertIn("comparison_reward_breakdown.png", os.listdir(d))

    def test_linked_in_summary_md(self):
        runs = self._runs_with_components()
        with tempfile.TemporaryDirectory() as d:
            save_grid_summary(runs, [], d, "gs_breakdown")
            with open(os.path.join(d, "summary.md"), encoding="utf-8") as f:
                md = f.read()
            self.assertIn("comparison_reward_breakdown.png", md)


class TestSaveGridSummaryTaskMetricPassthrough(unittest.TestCase):
    """SC2 save_grid_summary passes task_metric_fn and task_metric_fmt to framework."""

    def test_passes_task_metric_fn_to_framework(self):
        with tempfile.TemporaryDirectory() as d:
            data = _make_experiment([_make_sim(sim=1, reward=1.0)], name="exp")
            with mock.patch.object(sc2_analytics, "_framework_save_grid_summary") as m:
                save_grid_summary([("exp", data)], [], d, "gs")
            kwargs = m.call_args.kwargs
            self.assertIn("task_metric_fn", kwargs)
            self.assertTrue(callable(kwargs["task_metric_fn"]))

    def test_passes_task_metric_fmt_to_framework(self):
        with tempfile.TemporaryDirectory() as d:
            data = _make_experiment([_make_sim(sim=1, reward=1.0)], name="exp")
            with mock.patch.object(sc2_analytics, "_framework_save_grid_summary") as m:
                save_grid_summary([("exp", data)], [], d, "gs")
            kwargs = m.call_args.kwargs
            self.assertIn("task_metric_fmt", kwargs)
            self.assertTrue(callable(kwargs["task_metric_fmt"]))

    def test_task_metric_fmt_formats_as_percentage(self):
        with tempfile.TemporaryDirectory() as d:
            data = _make_experiment([_make_sim(sim=1, reward=1.0)], name="exp")
            with mock.patch.object(sc2_analytics, "_framework_save_grid_summary") as m:
                save_grid_summary([("exp", data)], [], d, "gs")
            fmt = m.call_args.kwargs["task_metric_fmt"]
            self.assertEqual(fmt(0.75), "75.0%")


if __name__ == "__main__":
    unittest.main(verbosity=2)
