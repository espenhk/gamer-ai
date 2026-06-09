"""Tests for train_rl returning slim_for_summary() data (issue #451).

Verifies that the ExperimentData returned by train_rl() has bulk per-sim
fields stripped so grid searches don't accumulate unbounded series data.
"""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import MagicMock, patch

from framework.analytics import ExperimentData, GreedySimResult


def _make_full_sim(sim: int = 1) -> GreedySimResult:
    """Build a GreedySimResult with every bulk field populated."""
    return GreedySimResult(
        sim=sim,
        reward=10.0,
        improved=True,
        throttle_counts=[1, 2, 3],
        total_steps=50,
        trace=object(),
        weights={"a": 1.0},
        army_count_series=[[0.0, 5], [1.0, 6]],
        resource_series=[[0.0, 100], [1.0, 120]],
        build_order=[["0.5", "Marine"]],
        obs_averages={"speed_ms": 0.5},
        xy_hist=[[0] * 8 for _ in range(8)],
    )


class TestSlimForSummaryUnit(unittest.TestCase):
    """Direct tests for ExperimentData.slim_for_summary()."""

    def _make_data(self, n_sims: int = 2) -> ExperimentData:
        return ExperimentData(
            experiment_name="test",
            probe_results=[object(), object()],
            cold_start_restarts=[object()],
            greedy_sims=[_make_full_sim(i) for i in range(1, n_sims + 1)],
            probe_floor=0.0,
            weights_file="w.yaml",
            reward_config_file="r.yaml",
            training_params={},
            timings={},
        )

    def test_probe_results_cleared(self):
        data = self._make_data()
        slim = data.slim_for_summary()
        self.assertEqual(slim.probe_results, [])

    def test_cold_start_restarts_cleared(self):
        data = self._make_data()
        slim = data.slim_for_summary()
        self.assertEqual(slim.cold_start_restarts, [])

    def test_greedy_sims_count_preserved(self):
        data = self._make_data(n_sims=3)
        slim = data.slim_for_summary()
        self.assertEqual(len(slim.greedy_sims), 3)

    def test_army_count_series_stripped(self):
        data = self._make_data()
        slim = data.slim_for_summary()
        for s in slim.greedy_sims:
            self.assertIsNone(s.army_count_series)

    def test_resource_series_stripped(self):
        data = self._make_data()
        slim = data.slim_for_summary()
        for s in slim.greedy_sims:
            self.assertIsNone(s.resource_series)

    def test_trace_stripped(self):
        data = self._make_data()
        slim = data.slim_for_summary()
        for s in slim.greedy_sims:
            self.assertIsNone(s.trace)

    def test_weights_stripped(self):
        data = self._make_data()
        slim = data.slim_for_summary()
        for s in slim.greedy_sims:
            self.assertIsNone(s.weights)

    def test_scalar_fields_preserved(self):
        data = self._make_data()
        slim = data.slim_for_summary()
        for s in slim.greedy_sims:
            self.assertEqual(s.reward, 10.0)
            self.assertTrue(s.improved)
            self.assertEqual(s.total_steps, 50)

    def test_original_data_unchanged(self):
        data = self._make_data()
        _ = data.slim_for_summary()
        # slim_for_summary must not mutate the original
        for s in data.greedy_sims:
            self.assertIsNotNone(s.army_count_series)
            self.assertIsNotNone(s.trace)


class TestTrainRLReturnIsSlim(unittest.TestCase):
    """train_rl() must return slim_for_summary() data, not the full ExperimentData."""

    @patch("framework.training._greedy_loop_genetic")
    @patch("framework.training._maybe_build_evaluator", return_value=None)
    @patch("framework.training.make_live_monitor", return_value=None)
    def test_returned_data_has_empty_probe_results(self, _lm, _eval, mock_loop):
        from framework.run_config import GameSpec, RunConfig
        from framework.training import GreedyLoopResult, train_rl
        from games.car_racing.actions import DISCRETE_ACTIONS
        from games.car_racing.obs_spec import CAR_RACING_OBS_SPEC as OBS_SPEC

        mock_loop.return_value = GreedyLoopResult(
            policy=MagicMock(),
            best_reward=0.0,
            greedy_sims=[_make_full_sim(1)],
            early_stopped=False,
            early_stop_sim=None,
        )

        mock_env = MagicMock()
        mock_env.observation_space = MagicMock()
        mock_env.action_space = MagicMock()

        training_params = {
            "policy_type": "genetic",
            "n_sims": 1,
            "speed": 1.0,
            "in_game_episode_s": 10.0,
            "mutation_scale": 0.05,
            "mutation_share": 1.0,
            "adaptive_mutation": False,
            "patience": 0,
            "policy_params": {"population_size": 2, "elite_k": 1},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            weights_file = os.path.join(tmpdir, "w.yaml")
            game = GameSpec(
                experiment_name="slim_test",
                track="car_racing",
                make_env_fn=lambda: mock_env,
                obs_spec=OBS_SPEC,
                head_names=["action"],
                discrete_actions=DISCRETE_ACTIONS,
                weights_file=weights_file,
                reward_config_file="",
                save_results_fn=None,
                game_name="car_racing",
            )
            config = RunConfig.from_training_params(training_params)
            result = train_rl(game, config, no_interrupt=True)

        self.assertEqual(result.probe_results, [])
        self.assertEqual(result.cold_start_restarts, [])
        for s in result.greedy_sims:
            self.assertIsNone(s.army_count_series)
            self.assertIsNone(s.trace)
            self.assertIsNone(s.weights)


if __name__ == "__main__":
    unittest.main()
