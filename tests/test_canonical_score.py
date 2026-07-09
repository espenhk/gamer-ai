"""Tests for compute_canonical_score and the canonical-score plumbing added
for cross-experiment reward comparability (issue #481).

Covers:
- compute_canonical_score() formula.
- RunTrace new fields (track_progress, finished, mean_lateral_offset_m,
  canonical_score) default to None; GreedySimResult.canonical_score default
  None, from_dict() round-trip.
- _gs_stats() best_canonical_score key.
- plot_gs_comparison_canonical(): renders when eligible, no-op otherwise.
- distributed/protocol.py round-trip of the new RunTrace/GreedySimResult
  fields, including old-JSON backward compatibility (missing keys -> None).
- games/tmnf/analytics.py: plot_greedy_canonical() and
  plot_gs_comparison_canonical_progress() render / no-op correctly.
- framework.training._run_episode() only populates the canonical-score
  fields when the env's info dict actually reports track_progress, so
  non-racing games (SC2, Atari, ...) get None rather than a fake 0.0.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from framework.analytics import (
    ExperimentData,
    GreedySimResult,
    RunTrace,
    _gs_stats,
    compute_canonical_score,
    plot_gs_comparison_canonical,
)
from framework.training import _run_episode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sim(
    sim: int = 1,
    reward: float = 10.0,
    improved: bool = False,
    canonical_score: float | None = None,
) -> GreedySimResult:
    return GreedySimResult(
        sim=sim,
        reward=reward,
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
# compute_canonical_score
# ---------------------------------------------------------------------------


class TestComputeCanonicalScore(unittest.TestCase):
    def test_full_clean_lap(self):
        score = compute_canonical_score(track_progress=1.0, finished=True, mean_lateral_offset_m=0.0)
        self.assertAlmostEqual(score, 1500.0)

    def test_half_lap_crash(self):
        score = compute_canonical_score(track_progress=0.5, finished=False, mean_lateral_offset_m=2.0)
        self.assertAlmostEqual(score, 500.0 - 8.0)

    def test_finished_flag_adds_bonus(self):
        base = compute_canonical_score(track_progress=1.0, finished=False, mean_lateral_offset_m=0.0)
        finished = compute_canonical_score(track_progress=1.0, finished=True, mean_lateral_offset_m=0.0)
        self.assertAlmostEqual(finished - base, 500.0)

    def test_lateral_offset_penalty_is_quadratic(self):
        score_2m = compute_canonical_score(0.0, False, 2.0)
        score_4m = compute_canonical_score(0.0, False, 4.0)
        # Quadratic penalty: doubling offset quadruples the penalty.
        self.assertAlmostEqual(score_2m, -8.0)
        self.assertAlmostEqual(score_4m, -32.0)

    def test_zero_progress_no_finish_no_offset_is_zero(self):
        self.assertAlmostEqual(compute_canonical_score(0.0, False, 0.0), 0.0)


# ---------------------------------------------------------------------------
# RunTrace / GreedySimResult new fields
# ---------------------------------------------------------------------------


class TestRunTraceNewFields(unittest.TestCase):
    def test_defaults_are_none(self):
        trace = RunTrace(pos_x=[], pos_z=[], throttle_state=[], total_reward=0.0)
        self.assertIsNone(trace.track_progress)
        self.assertIsNone(trace.finished)
        self.assertIsNone(trace.mean_lateral_offset_m)
        self.assertIsNone(trace.canonical_score)

    def test_fields_stored(self):
        trace = RunTrace(
            pos_x=[],
            pos_z=[],
            throttle_state=[],
            total_reward=0.0,
            track_progress=0.75,
            finished=False,
            mean_lateral_offset_m=1.2,
            canonical_score=742.3,
        )
        self.assertAlmostEqual(trace.track_progress, 0.75)
        self.assertFalse(trace.finished)
        self.assertAlmostEqual(trace.mean_lateral_offset_m, 1.2)
        self.assertAlmostEqual(trace.canonical_score, 742.3)


class TestGreedySimResultCanonicalScore(unittest.TestCase):
    def test_default_is_none(self):
        s = GreedySimResult(sim=1, reward=5.0, improved=False, throttle_counts=[0, 0, 1], total_steps=1)
        self.assertIsNone(s.canonical_score)

    def test_stored(self):
        s = _make_sim(canonical_score=1234.5)
        self.assertAlmostEqual(s.canonical_score, 1234.5)

    def test_from_dict_reads_canonical_score(self):
        d = {
            "sim": 1,
            "reward": 5.0,
            "improved": True,
            "throttle_counts": [0, 0, 1],
            "total_steps": 1,
            "canonical_score": 999.0,
        }
        s = GreedySimResult.from_dict(d)
        self.assertAlmostEqual(s.canonical_score, 999.0)

    def test_from_dict_defaults_canonical_score_to_none(self):
        d = {
            "sim": 1,
            "reward": 5.0,
            "improved": True,
            "throttle_counts": [0, 0, 1],
            "total_steps": 1,
        }
        s = GreedySimResult.from_dict(d)
        self.assertIsNone(s.canonical_score)


# ---------------------------------------------------------------------------
# _gs_stats() best_canonical_score
# ---------------------------------------------------------------------------


class TestGsStatsCanonicalScore(unittest.TestCase):
    def test_empty_sims_returns_none(self):
        data = _make_experiment([])
        stats = _gs_stats(data)
        self.assertIsNone(stats["best_canonical_score"])

    def test_no_canonical_scores_returns_none(self):
        sims = [_make_sim(i) for i in range(1, 4)]
        data = _make_experiment(sims)
        stats = _gs_stats(data)
        self.assertIsNone(stats["best_canonical_score"])

    def test_best_canonical_score_is_max(self):
        sims = [
            _make_sim(1, canonical_score=100.0),
            _make_sim(2, canonical_score=500.0),
            _make_sim(3, canonical_score=250.0),
        ]
        data = _make_experiment(sims)
        stats = _gs_stats(data)
        self.assertAlmostEqual(stats["best_canonical_score"], 500.0)

    def test_partial_canonical_scores_ignores_none(self):
        sims = [
            _make_sim(1, canonical_score=None),
            _make_sim(2, canonical_score=300.0),
        ]
        data = _make_experiment(sims)
        stats = _gs_stats(data)
        self.assertAlmostEqual(stats["best_canonical_score"], 300.0)


# ---------------------------------------------------------------------------
# plot_gs_comparison_canonical
# ---------------------------------------------------------------------------


class TestPlotGsComparisonCanonical(unittest.TestCase):
    def test_renders_when_eligible(self):
        runs = [
            ("exp_a", {"best_canonical_score": 100.0}),
            ("exp_b", {"best_canonical_score": 200.0}),
        ]
        with tempfile.TemporaryDirectory() as d:
            plot_gs_comparison_canonical(runs, d)
            self.assertIn("comparison_canonical.png", os.listdir(d))

    def test_no_op_when_no_run_has_canonical_score(self):
        runs = [("exp_a", {"best_canonical_score": None})]
        with tempfile.TemporaryDirectory() as d:
            plot_gs_comparison_canonical(runs, d)
            self.assertNotIn("comparison_canonical.png", os.listdir(d))

    def test_no_op_when_no_runs(self):
        with tempfile.TemporaryDirectory() as d:
            plot_gs_comparison_canonical([], d)
            self.assertEqual(os.listdir(d), [])


# ---------------------------------------------------------------------------
# save_grid_summary wires in the canonical bar chart + table column
# ---------------------------------------------------------------------------


class TestSaveGridSummaryCanonical(unittest.TestCase):
    def test_canonical_chart_written_when_eligible(self):
        from framework.analytics import save_grid_summary

        sims_a = [_make_sim(1, reward=10.0, canonical_score=1200.0)]
        sims_b = [_make_sim(1, reward=5.0, canonical_score=800.0)]
        runs = [("exp_a", _make_experiment(sims_a, "exp_a")), ("exp_b", _make_experiment(sims_b, "exp_b"))]
        with tempfile.TemporaryDirectory() as d:
            save_grid_summary(runs, [], d, "gs_canon")
            self.assertIn("comparison_canonical.png", os.listdir(d))
            with open(os.path.join(d, "summary.md"), encoding="utf-8") as f:
                md = f.read()
            self.assertIn("Best Canonical", md)
            self.assertIn("+1200.0", md)
            # PR #490 review: the chart file existing isn't enough -- summary.md
            # must actually embed/link it so it's not easy to miss.
            self.assertIn("![Canonical score comparison](comparison_canonical.png)", md)

    def test_no_canonical_chart_when_ineligible(self):
        from framework.analytics import save_grid_summary

        sims = [_make_sim(1, reward=10.0)]  # canonical_score=None
        runs = [("exp_a", _make_experiment(sims, "exp_a"))]
        with tempfile.TemporaryDirectory() as d:
            save_grid_summary(runs, [], d, "gs_no_canon")
            self.assertNotIn("comparison_canonical.png", os.listdir(d))
            with open(os.path.join(d, "summary.md"), encoding="utf-8") as f:
                md = f.read()
            self.assertNotIn("comparison_canonical.png", md)


# ---------------------------------------------------------------------------
# distributed/protocol.py round-trip
# ---------------------------------------------------------------------------


class TestProtocolCanonicalRoundTrip(unittest.TestCase):
    def _make_experiment_data(self) -> ExperimentData:
        return ExperimentData(
            experiment_name="canon_test",
            probe_results=[],
            cold_start_restarts=[],
            greedy_sims=[
                GreedySimResult(
                    sim=1,
                    reward=123.4,
                    improved=True,
                    throttle_counts=[5, 10, 85],
                    total_steps=100,
                    trace=RunTrace(
                        pos_x=[0.0, 1.0],
                        pos_z=[0.0, 2.0],
                        throttle_state=[(0, 1)],
                        total_reward=123.4,
                        track_progress=1.0,
                        finished=True,
                        mean_lateral_offset_m=0.5,
                        canonical_score=1499.5,
                    ),
                    canonical_score=1499.5,
                ),
                GreedySimResult(
                    sim=2,
                    reward=10.0,
                    improved=False,
                    throttle_counts=[10, 10, 80],
                    total_steps=100,
                    # canonical_score / trace left as None.
                ),
            ],
            probe_floor=None,
            weights_file="",
            reward_config_file="",
            training_params={},
            timings={},
        )

    def test_round_trip_preserves_canonical_fields(self):
        from distributed.protocol import experiment_from_dict, experiment_to_json

        original = self._make_experiment_data()
        recovered = experiment_from_dict(json.loads(experiment_to_json(original)))

        s0 = recovered.greedy_sims[0]
        self.assertAlmostEqual(s0.canonical_score, 1499.5)
        self.assertIsNotNone(s0.trace)
        self.assertAlmostEqual(s0.trace.track_progress, 1.0)
        self.assertTrue(s0.trace.finished)
        self.assertAlmostEqual(s0.trace.mean_lateral_offset_m, 0.5)
        self.assertAlmostEqual(s0.trace.canonical_score, 1499.5)

        s1 = recovered.greedy_sims[1]
        self.assertIsNone(s1.canonical_score)
        self.assertIsNone(s1.trace)

    def test_old_json_without_canonical_fields_deserializes_to_none(self):
        """Old experiment_data.json predating this feature has no canonical
        keys at all; the loader must default them to None rather than raise."""
        from distributed.protocol import experiment_from_dict

        old_dict = {
            "experiment_name": "old_run",
            "probe_results": [],
            "cold_start_restarts": [],
            "greedy_sims": [
                {
                    "sim": 1,
                    "reward": 42.0,
                    "improved": True,
                    "throttle_counts": [1, 2, 3],
                    "total_steps": 10,
                    "trace": {
                        "pos_x": [0.0],
                        "pos_z": [0.0],
                        "throttle_state": [],
                        "total_reward": 42.0,
                        # No track_progress / finished / mean_lateral_offset_m / canonical_score.
                    },
                    # No canonical_score.
                }
            ],
            "probe_floor": None,
            "weights_file": "",
            "reward_config_file": "",
            "training_params": {},
            "timings": {},
        }
        recovered = experiment_from_dict(old_dict)
        s0 = recovered.greedy_sims[0]
        self.assertIsNone(s0.canonical_score)
        self.assertIsNotNone(s0.trace)
        self.assertIsNone(s0.trace.track_progress)
        self.assertIsNone(s0.trace.finished)
        self.assertIsNone(s0.trace.mean_lateral_offset_m)
        self.assertIsNone(s0.trace.canonical_score)


# ---------------------------------------------------------------------------
# _run_episode() gating: canonical fields only populated when the env
# actually reports track_progress (issue #481 review fix)
# ---------------------------------------------------------------------------


class _FakePolicy:
    def __call__(self, obs):
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def update(self, *args, **kwargs):
        pass

    def on_episode_start(self, **kwargs):
        pass


class _FakeEnvWithProgress:
    """One-step env whose info dict reports track_progress (TMNF-like)."""

    def reset(self, *, seed=None, options=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        info = {"track_progress": 0.5, "finished": False, "mean_abs_lateral_offset": 1.5}
        return np.zeros(4, dtype=np.float32), 1.0, True, False, info


class _FakeEnvWithoutProgress:
    """One-step env whose info dict never mentions track_progress (SC2-like)."""

    def reset(self, *, seed=None, options=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        info = {"score": 10.0}
        return np.zeros(4, dtype=np.float32), 1.0, True, False, info


class _FakeEnvWithPerStepOffset:
    """Multi-step env reporting per-step lateral_offset but no aggregated
    mean_abs_lateral_offset (TORCS/BeamNG/iRacing/Assetto-like)."""

    def __init__(self, offsets: list) -> None:
        self._offsets = offsets
        self._i = 0

    def reset(self, *, seed=None, options=None):
        self._i = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        offset = self._offsets[self._i]
        self._i += 1
        done = self._i >= len(self._offsets)
        info = {"track_progress": self._i / len(self._offsets), "finished": done, "lateral_offset": offset}
        return np.zeros(4, dtype=np.float32), 1.0, done, False, info


class _FakeEnvWithMeanOffsetOnly:
    """track_progress present, no per-step lateral_offset at all, but a
    pre-aggregated mean_abs_lateral_offset on the terminal step (fallback path)."""

    def reset(self, *, seed=None, options=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        info = {"track_progress": 1.0, "finished": True, "mean_abs_lateral_offset": 3.0}
        return np.zeros(4, dtype=np.float32), 1.0, True, False, info


class TestRunEpisodeCanonicalGating(unittest.TestCase):
    def test_populates_canonical_fields_when_track_progress_present(self):
        env = _FakeEnvWithProgress()
        obs, reset_info = env.reset()
        ep = _run_episode(env, _FakePolicy(), obs, reset_info=reset_info)
        self.assertIsNotNone(ep.trace.canonical_score)
        self.assertAlmostEqual(ep.trace.track_progress, 0.5)
        self.assertFalse(ep.trace.finished)
        self.assertAlmostEqual(ep.trace.mean_lateral_offset_m, 1.5)

    def test_leaves_canonical_fields_none_when_track_progress_absent(self):
        """Regression test: a game whose info dict never sets track_progress
        (e.g. SC2, Atari) must get None, not a fake 0.0, for all four
        canonical-score fields -- otherwise the is-not-None no-op guards on
        the canonical plots/stats treat it as real data."""
        env = _FakeEnvWithoutProgress()
        obs, reset_info = env.reset()
        ep = _run_episode(env, _FakePolicy(), obs, reset_info=reset_info)
        self.assertIsNone(ep.trace.track_progress)
        self.assertIsNone(ep.trace.finished)
        self.assertIsNone(ep.trace.mean_lateral_offset_m)
        self.assertIsNone(ep.trace.canonical_score)

    def test_accumulates_mean_lateral_offset_from_per_step_info(self):
        """Regression test (PR #490 review): non-TMNF racing envs (TORCS,
        BeamNG, iRacing, Assetto Corsa) report per-step info["lateral_offset"]
        but never an aggregated mean_abs_lateral_offset. _run_episode must
        accumulate the per-step values itself rather than silently treating
        the offset as 0.0 for those games."""
        offsets = [1.0, 3.0, 2.0]
        env = _FakeEnvWithPerStepOffset(offsets)
        obs, reset_info = env.reset()
        ep = _run_episode(env, _FakePolicy(), obs, reset_info=reset_info)
        self.assertAlmostEqual(ep.trace.mean_lateral_offset_m, sum(offsets) / len(offsets))
        expected_canonical = compute_canonical_score(1.0, True, sum(offsets) / len(offsets))
        self.assertAlmostEqual(ep.trace.canonical_score, expected_canonical)

    def test_falls_back_to_mean_abs_lateral_offset_when_no_per_step_samples(self):
        env = _FakeEnvWithMeanOffsetOnly()
        obs, reset_info = env.reset()
        ep = _run_episode(env, _FakePolicy(), obs, reset_info=reset_info)
        self.assertAlmostEqual(ep.trace.mean_lateral_offset_m, 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
