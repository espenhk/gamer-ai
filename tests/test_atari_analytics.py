"""Tests for Atari-specific analytics output."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from framework.analytics import ExperimentData, GreedySimResult
from games.atari.analytics import save_experiment_results


class TestAtariAnalytics(unittest.TestCase):
    def test_results_report_includes_return_length_and_action_artifacts(self):
        data = ExperimentData(
            experiment_name="atari-smoke",
            probe_results=[],
            cold_start_restarts=[],
            greedy_sims=[
                GreedySimResult(
                    sim=1,
                    reward=1.0,
                    improved=True,
                    throttle_counts=[0, 0, 0],
                    total_steps=12,
                    action_counts={0: 2, 1: 10},
                ),
                GreedySimResult(
                    sim=2,
                    reward=3.0,
                    improved=True,
                    throttle_counts=[0, 0, 0],
                    total_steps=20,
                    action_counts={1: 5, 2: 15},
                ),
            ],
            probe_floor=None,
            weights_file="policy_weights.yaml",
            reward_config_file="reward_config.yaml",
            training_params={},
            timings={"start": "s", "end": "e", "total_s": 1.0, "probe_s": None, "cold_start_s": None, "greedy_s": 1.0},
            track="Pong-v5",
        )

        with tempfile.TemporaryDirectory() as tmp:
            save_experiment_results(data, tmp)
            report = Path(tmp, "results.md").read_text(encoding="utf-8")

            self.assertIn("## Atari Metrics", report)
            self.assertIn("Best episode return", report)
            self.assertIn("Mean episode length", report)
            self.assertIn("atari_episode_returns.png", report)
            self.assertIn("atari_episode_lengths.png", report)
            self.assertIn("atari_action_histogram.png", report)
            self.assertTrue(Path(tmp, "atari_episode_returns.png").exists())
            self.assertTrue(Path(tmp, "atari_episode_lengths.png").exists())
            self.assertTrue(Path(tmp, "atari_action_histogram.png").exists())


if __name__ == "__main__":
    unittest.main()
