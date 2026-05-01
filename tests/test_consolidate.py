"""Tests for grid-search consolidation: save/load experiment data + consolidate CLI."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import pytest

from framework.analytics import (
    ExperimentData,
    GreedySimResult,
    RunTrace,
    save_experiment_data_json,
    load_experiment_data,
)


def _make_experiment_data(name: str = "test_exp", track: str = "a03_centerline",
                          reward: float = 100.0, mutation_scale: float = 0.05) -> ExperimentData:
    return ExperimentData(
        experiment_name=name,
        probe_results=[],
        cold_start_restarts=[],
        greedy_sims=[
            GreedySimResult(
                sim=1, reward=reward, improved=True,
                throttle_counts=[5, 10, 85], total_steps=100,
                trace=RunTrace(pos_x=[0.0, 1.0], pos_z=[0.0, 2.0],
                               throttle_state=[(0, 1)], total_reward=reward),
            ),
        ],
        probe_floor=None,
        weights_file=f"experiments/{track}/{name}/policy_weights.yaml",
        reward_config_file=f"experiments/{track}/{name}/reward_config.yaml",
        training_params={"speed": 10.0, "n_sims": 5, "mutation_scale": mutation_scale},
        timings={"greedy_s": 42.0},
        track=track,
    )


class TestSaveAndLoadExperimentData(unittest.TestCase):
    def test_round_trip(self):
        data = _make_experiment_data("round_trip_test", reward=123.4)
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = os.path.join(tmpdir, "my_experiment")
            results_dir = os.path.join(experiment_dir, "results")
            os.makedirs(results_dir)
            save_experiment_data_json(data, results_dir)

            self.assertTrue(os.path.exists(os.path.join(results_dir, "experiment_data.json")))

            loaded = load_experiment_data(experiment_dir)
            self.assertEqual(loaded.experiment_name, "round_trip_test")
            self.assertEqual(len(loaded.greedy_sims), 1)
            self.assertAlmostEqual(loaded.greedy_sims[0].reward, 123.4)
            self.assertEqual(loaded.training_params["speed"], 10.0)
            self.assertEqual(loaded.track, "a03_centerline")

    def test_load_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                load_experiment_data(tmpdir)

    def test_json_is_valid(self):
        data = _make_experiment_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "results")
            path = save_experiment_data_json(data, results_dir)
            with open(path) as f:
                parsed = json.load(f)
            self.assertEqual(parsed["experiment_name"], "test_exp")

    def test_creates_results_dir_if_missing(self):
        data = _make_experiment_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "does_not_exist", "results")
            save_experiment_data_json(data, results_dir)
            self.assertTrue(os.path.exists(os.path.join(results_dir, "experiment_data.json")))


class TestConsolidate(unittest.TestCase):
    def _write_experiment(self, base_dir: str, name: str, **kwargs) -> str:
        data = _make_experiment_data(name, **kwargs)
        experiment_dir = os.path.join(base_dir, name)
        results_dir = os.path.join(experiment_dir, "results")
        os.makedirs(results_dir)
        save_experiment_data_json(data, results_dir)
        return experiment_dir

    def test_consolidate_produces_summary(self):
        from grid_search import _consolidate

        with tempfile.TemporaryDirectory() as tmpdir:
            dir1 = self._write_experiment(tmpdir, "exp__ms0.05", mutation_scale=0.05)
            dir2 = self._write_experiment(tmpdir, "exp__ms0.1", mutation_scale=0.1)

            summary_dir = os.path.join(tmpdir, "test_summary")
            _consolidate([dir1, dir2], "test_summary", summary_dir)

            summary_path = os.path.join(summary_dir, "summary.md")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path) as f:
                content = f.read()
            self.assertIn("test_summary", content)
            self.assertIn("2 experiments", content)

    def test_consolidate_infers_summary_dir(self):
        from grid_search import _consolidate

        with tempfile.TemporaryDirectory() as tmpdir:
            dir1 = self._write_experiment(tmpdir, "a")
            dir2 = self._write_experiment(tmpdir, "b")

            _consolidate([dir1, dir2], "my_consolidated", None)

            expected = os.path.join(tmpdir, "my_consolidated__summary", "summary.md")
            self.assertTrue(os.path.exists(expected))

    def test_consolidate_skips_missing(self):
        from grid_search import _consolidate

        with tempfile.TemporaryDirectory() as tmpdir:
            dir1 = self._write_experiment(tmpdir, "good_exp")
            bad_dir = os.path.join(tmpdir, "missing_exp")
            os.makedirs(bad_dir)

            summary_dir = os.path.join(tmpdir, "summary")
            _consolidate([dir1, bad_dir], "summary", summary_dir)

            summary_path = os.path.join(summary_dir, "summary.md")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path) as f:
                content = f.read()
            self.assertIn("1 experiments", content)

    def test_consolidate_detects_varied_keys(self):
        from grid_search import _consolidate

        with tempfile.TemporaryDirectory() as tmpdir:
            dir1 = self._write_experiment(tmpdir, "exp1", mutation_scale=0.05)
            dir2 = self._write_experiment(tmpdir, "exp2", mutation_scale=0.2)

            summary_dir = os.path.join(tmpdir, "summary")
            _consolidate([dir1, dir2], "summary", summary_dir)

            with open(os.path.join(summary_dir, "summary.md")) as f:
                content = f.read()
            self.assertIn("mutation_scale", content)
