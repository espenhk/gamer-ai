"""Tests for redo_analytics.py — re-generate analytics from experiment_data.json."""
from __future__ import annotations

import os
import tempfile
import unittest

from framework.analytics import (
    ExperimentData,
    GreedySimResult,
    RunTrace,
    save_experiment_data_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(
    name: str = "test_exp",
    track: str = "a03_centerline",
    reward: float = 100.0,
    training_params: dict | None = None,
    weights_file: str | None = None,
    reward_config_file: str | None = None,
) -> ExperimentData:
    params = training_params or {"speed": 10.0, "n_sims": 5, "mutation_scale": 0.05}
    return ExperimentData(
        experiment_name=name,
        probe_results=[],
        cold_start_restarts=[],
        greedy_sims=[
            GreedySimResult(
                sim=1, reward=reward, improved=True,
                throttle_counts=[5, 10, 85], total_steps=100,
                trace=RunTrace(
                    pos_x=[0.0, 1.0], pos_z=[0.0, 2.0],
                    throttle_state=[(0.0, 1.0)], total_reward=reward,
                ),
            ),
        ],
        probe_floor=None,
        weights_file=weights_file or f"experiments/{track}/{name}/policy_weights.yaml",
        reward_config_file=reward_config_file or f"experiments/{track}/{name}/reward_config.yaml",
        training_params=params,
        timings={"start": "2026-01-01 00:00:00", "end": "2026-01-01 01:00:00",
                 "total_s": 3600.0, "greedy_s": 3600.0},
        track=track,
    )


def _write_experiment(base_dir: str, name: str, **kwargs) -> str:
    """Write experiment_data.json to base_dir/name/results/ and return the experiment dir."""
    experiment_dir = os.path.join(base_dir, name)
    results_dir = os.path.join(experiment_dir, "results")
    os.makedirs(results_dir)

    # Write minimal YAML files inside the experiment dir so analytics can run
    # without FileNotFoundError (e.g. plot_weight_heatmap opens weights_file).
    weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
    reward_config_file = os.path.join(experiment_dir, "reward_config.yaml")
    with open(weights_file, "w") as f:
        f.write("{}\n")  # no steer_weights key → heatmap skips gracefully
    with open(reward_config_file, "w") as f:
        f.write("progress_weight: 10000.0\n")

    data = _make_data(
        name,
        weights_file=weights_file,
        reward_config_file=reward_config_file,
        **kwargs,
    )
    save_experiment_data_json(data, results_dir)
    return experiment_dir


# ---------------------------------------------------------------------------
# Game detection
# ---------------------------------------------------------------------------

class TestDetectGame(unittest.TestCase):
    def test_sc2_detected_by_map_name(self):
        from redo_analytics import _detect_game
        self.assertEqual(_detect_game({"map_name": "MoveToBeacon", "n_sims": 50}), "sc2")

    def test_sc2_detected_by_agent_race(self):
        from redo_analytics import _detect_game
        self.assertEqual(_detect_game({"agent_race": "terran", "n_sims": 50}), "sc2")

    def test_tmnf_default_for_racing_params(self):
        from redo_analytics import _detect_game
        self.assertEqual(_detect_game({"speed": 10.0, "n_sims": 100, "mutation_scale": 0.05}), "tmnf")

    def test_empty_params_defaults_to_tmnf(self):
        from redo_analytics import _detect_game
        self.assertEqual(_detect_game({}), "tmnf")


# ---------------------------------------------------------------------------
# Analytics loader
# ---------------------------------------------------------------------------

class TestLoadAnalyticsFns(unittest.TestCase):
    def test_tmnf_returns_callables(self):
        from redo_analytics import _load_analytics_fns
        save_exp, save_grid = _load_analytics_fns("tmnf")
        self.assertTrue(callable(save_exp))
        self.assertTrue(callable(save_grid))

    def test_unknown_game_falls_back_gracefully(self):
        from redo_analytics import _load_analytics_fns
        save_exp, save_grid = _load_analytics_fns("nonexistent_game")
        self.assertTrue(callable(save_exp))
        self.assertTrue(callable(save_grid))


# ---------------------------------------------------------------------------
# redo_analytics: single experiment
# ---------------------------------------------------------------------------

class TestRedoAnalyticsSingle(unittest.TestCase):
    def test_regenerates_results_md(self):
        from redo_analytics import redo_analytics

        with tempfile.TemporaryDirectory() as tmpdir:
            d = _write_experiment(tmpdir, "exp1")
            redo_analytics([d], game="tmnf")
            self.assertTrue(os.path.exists(os.path.join(d, "results", "results.md")))

    def test_regenerates_greedy_rewards_plot(self):
        from redo_analytics import redo_analytics

        with tempfile.TemporaryDirectory() as tmpdir:
            d = _write_experiment(tmpdir, "exp1")
            redo_analytics([d], game="tmnf")
            self.assertTrue(os.path.exists(os.path.join(d, "results", "greedy_rewards.png")))

    def test_no_summary_for_single_without_summary_name(self):
        from redo_analytics import redo_analytics

        with tempfile.TemporaryDirectory() as tmpdir:
            d = _write_experiment(tmpdir, "exp1")
            redo_analytics([d], game="tmnf")
            # No summary directory should be created.
            entries = os.listdir(tmpdir)
            summary_dirs = [e for e in entries if "__summary" in e]
            self.assertEqual(summary_dirs, [])

    def test_summary_produced_when_summary_name_given(self):
        from redo_analytics import redo_analytics

        with tempfile.TemporaryDirectory() as tmpdir:
            d = _write_experiment(tmpdir, "exp1")
            summary_dir = os.path.join(tmpdir, "my_summary")
            redo_analytics([d], game="tmnf", summary_name="my_summary",
                           summary_dir=summary_dir)
            self.assertTrue(os.path.exists(os.path.join(summary_dir, "summary.md")))

    def test_missing_experiment_dir_is_skipped(self):
        from redo_analytics import redo_analytics

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_dir = os.path.join(tmpdir, "nonexistent")
            os.makedirs(bad_dir)  # dir exists but has no results/
            # Should not raise; just log error and return.
            redo_analytics([bad_dir], game="tmnf")

    def test_no_individual_without_summary_name_raises(self):
        from redo_analytics import redo_analytics

        with tempfile.TemporaryDirectory() as tmpdir:
            d = _write_experiment(tmpdir, "exp1")
            with self.assertRaises(ValueError):
                redo_analytics([d], game="tmnf", no_individual=True)


# ---------------------------------------------------------------------------
# redo_analytics: multiple experiments
# ---------------------------------------------------------------------------

class TestRedoAnalyticsMultiple(unittest.TestCase):
    def test_summary_md_written(self):
        from redo_analytics import redo_analytics

        with tempfile.TemporaryDirectory() as tmpdir:
            d1 = _write_experiment(tmpdir, "exp__ms0.05", training_params={"mutation_scale": 0.05, "n_sims": 5})
            d2 = _write_experiment(tmpdir, "exp__ms0.1",  training_params={"mutation_scale": 0.10, "n_sims": 5})
            summary_dir = os.path.join(tmpdir, "summary")
            redo_analytics([d1, d2], game="tmnf", summary_name="test_summary",
                           summary_dir=summary_dir)
            self.assertTrue(os.path.exists(os.path.join(summary_dir, "summary.md")))

    def test_summary_contains_experiment_names(self):
        from redo_analytics import redo_analytics

        with tempfile.TemporaryDirectory() as tmpdir:
            d1 = _write_experiment(tmpdir, "alpha", training_params={"mutation_scale": 0.05, "n_sims": 5})
            d2 = _write_experiment(tmpdir, "beta",  training_params={"mutation_scale": 0.10, "n_sims": 5})
            summary_dir = os.path.join(tmpdir, "summary")
            redo_analytics([d1, d2], game="tmnf", summary_name="gs",
                           summary_dir=summary_dir)
            with open(os.path.join(summary_dir, "summary.md")) as f:
                content = f.read()
            self.assertIn("alpha", content)
            self.assertIn("beta", content)

    def test_individual_results_regenerated(self):
        from redo_analytics import redo_analytics

        with tempfile.TemporaryDirectory() as tmpdir:
            d1 = _write_experiment(tmpdir, "exp1", training_params={"mutation_scale": 0.05, "n_sims": 5})
            d2 = _write_experiment(tmpdir, "exp2", training_params={"mutation_scale": 0.10, "n_sims": 5})
            redo_analytics([d1, d2], game="tmnf")
            self.assertTrue(os.path.exists(os.path.join(d1, "results", "results.md")))
            self.assertTrue(os.path.exists(os.path.join(d2, "results", "results.md")))

    def test_no_individual_skips_results_md(self):
        from redo_analytics import redo_analytics

        with tempfile.TemporaryDirectory() as tmpdir:
            d1 = _write_experiment(tmpdir, "exp1", training_params={"mutation_scale": 0.05, "n_sims": 5})
            d2 = _write_experiment(tmpdir, "exp2", training_params={"mutation_scale": 0.10, "n_sims": 5})
            summary_dir = os.path.join(tmpdir, "summary")
            # Remove any pre-existing results.md to verify it isn't created.
            for d in [d1, d2]:
                rmd = os.path.join(d, "results", "results.md")
                if os.path.exists(rmd):
                    os.remove(rmd)
            redo_analytics([d1, d2], game="tmnf", summary_name="s",
                           summary_dir=summary_dir, no_individual=True)
            self.assertFalse(os.path.exists(os.path.join(d1, "results", "results.md")))
            self.assertFalse(os.path.exists(os.path.join(d2, "results", "results.md")))
            self.assertTrue(os.path.exists(os.path.join(summary_dir, "summary.md")))

    def test_summary_dir_inferred_from_common_parent(self):
        from redo_analytics import redo_analytics

        with tempfile.TemporaryDirectory() as tmpdir:
            d1 = _write_experiment(tmpdir, "exp1", training_params={"mutation_scale": 0.05, "n_sims": 5})
            d2 = _write_experiment(tmpdir, "exp2", training_params={"mutation_scale": 0.10, "n_sims": 5})
            redo_analytics([d1, d2], game="tmnf", summary_name="inferred")
            expected = os.path.join(tmpdir, "inferred__summary", "summary.md")
            self.assertTrue(os.path.exists(expected))

    def test_default_summary_name_is_combined(self):
        from redo_analytics import redo_analytics

        with tempfile.TemporaryDirectory() as tmpdir:
            d1 = _write_experiment(tmpdir, "exp1", training_params={"mutation_scale": 0.05, "n_sims": 5})
            d2 = _write_experiment(tmpdir, "exp2", training_params={"mutation_scale": 0.10, "n_sims": 5})
            redo_analytics([d1, d2], game="tmnf")
            expected = os.path.join(tmpdir, "combined__summary", "summary.md")
            self.assertTrue(os.path.exists(expected))

    def test_skips_missing_experiments_gracefully(self):
        from redo_analytics import redo_analytics

        with tempfile.TemporaryDirectory() as tmpdir:
            d1 = _write_experiment(tmpdir, "good")
            bad = os.path.join(tmpdir, "bad")
            os.makedirs(bad)  # no results/ inside
            summary_dir = os.path.join(tmpdir, "summary")
            redo_analytics([d1, bad], game="tmnf", summary_name="s",
                           summary_dir=summary_dir)
            with open(os.path.join(summary_dir, "summary.md")) as f:
                content = f.read()
            self.assertIn("1 experiments", content)

    def test_auto_detect_sc2_game(self):
        from redo_analytics import redo_analytics

        sc2_params = {"map_name": "MoveToBeacon", "agent_race": "terran",
                      "n_sims": 5, "step_mul": 8}
        with tempfile.TemporaryDirectory() as tmpdir:
            d = _write_experiment(tmpdir, "sc2_exp", training_params=sc2_params)
            # Should not raise; SC2 analytics will be used.
            redo_analytics([d])
            self.assertTrue(os.path.exists(os.path.join(d, "results", "results.md")))
