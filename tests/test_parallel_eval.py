"""Unit tests for framework.parallel_eval (issue #229).

Uses picklable dummy env / policy from tests._parallel_eval_helpers so the
spawn-based workers can rebuild them in fresh interpreters.

PYTHONPATH is set in this module so spawn children inherit the search paths
they need to ``import tests._parallel_eval_helpers``.
"""
from __future__ import annotations

import os
import sys

# Make spawn children able to find the helpers module + framework package.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_existing = os.environ.get("PYTHONPATH", "")
_paths = [_REPO_ROOT] + ([_existing] if _existing else [])
os.environ["PYTHONPATH"] = os.pathsep.join(_paths)

import numpy as np
import pytest

from framework.parallel_eval import (
    Candidate,
    EpisodeResult,
    ParallelEvaluator,
)
from tests._parallel_eval_helpers import (
    DummyPolicy,
    make_crashing_env_factory,
    make_dummy_env,
)


# Keep workers cheap — these are in-process dummies, no SC2 binary involved.
_FAST_KW = dict(
    worker_start_stagger_s=0.0,
    worker_warmup_timeout_s=5.0,
    per_episode_timeout_s=5.0,
)


def _serial_eval(candidates: list[Candidate], make_env_fn, template) -> dict[int, float]:
    """Reference implementation: run candidates one at a time in this process."""
    env = make_env_fn()
    out: dict[int, float] = {}
    try:
        for c in candidates:
            policy = template.with_flat(c.flat_weights)
            ep_rewards: list[float] = []
            for _ in range(c.eval_episodes):
                obs, _info = env.reset()
                total = 0.0
                while True:
                    action = policy(obs)
                    obs, reward, terminated, truncated, _info = env.step(action)
                    total += reward
                    if terminated or truncated:
                        break
                ep_rewards.append(total)
            out[c.individual_idx] = float(np.mean(ep_rewards))
    finally:
        env.close()
    return out


class TestParallelEvaluator:
    """Pool-wiring, crash recovery, shutdown, determinism."""

    def test_returns_results_sorted_by_individual_idx(self):
        """Submit candidates in non-monotonic order; results come back sorted."""
        template = DummyPolicy()
        evaluator = ParallelEvaluator(
            n_workers=2,
            make_env_fn=make_dummy_env,
            template_policy=template,
            **_FAST_KW,
        )
        try:
            # Reverse-order submission.
            candidates = [
                Candidate(individual_idx=4, flat_weights=np.array([4.0]), eval_episodes=1),
                Candidate(individual_idx=1, flat_weights=np.array([1.0]), eval_episodes=1),
                Candidate(individual_idx=3, flat_weights=np.array([3.0]), eval_episodes=1),
                Candidate(individual_idx=0, flat_weights=np.array([0.0]), eval_episodes=1),
                Candidate(individual_idx=2, flat_weights=np.array([2.0]), eval_episodes=1),
            ]
            results = evaluator.evaluate(candidates)

            assert [r.individual_idx for r in results] == [0, 1, 2, 3, 4]
            assert all(isinstance(r, EpisodeResult) for r in results)
            assert all(not r.failed for r in results)
        finally:
            evaluator.close()

    def test_matches_serial_reference(self):
        """Parallel and serial evaluation produce the same rewards per individual."""
        template = DummyPolicy()
        candidates = [
            Candidate(individual_idx=i, flat_weights=np.array([float(i) * 0.5]),
                      eval_episodes=1)
            for i in range(6)
        ]
        serial = _serial_eval(candidates, make_dummy_env, template)

        evaluator = ParallelEvaluator(
            n_workers=3,
            make_env_fn=make_dummy_env,
            template_policy=template,
            **_FAST_KW,
        )
        try:
            results = evaluator.evaluate(candidates)
        finally:
            evaluator.close()

        for r in results:
            assert r.reward == pytest.approx(serial[r.individual_idx])

    def test_eval_episodes_means_reward(self):
        """eval_episodes>1: returned reward is the per-episode mean."""
        template = DummyPolicy()
        candidate = Candidate(
            individual_idx=0,
            flat_weights=np.array([2.0]),
            eval_episodes=4,
        )
        evaluator = ParallelEvaluator(
            n_workers=1,
            make_env_fn=make_dummy_env,
            template_policy=template,
            **_FAST_KW,
        )
        try:
            results = evaluator.evaluate([candidate])
        finally:
            evaluator.close()

        # DummyEnv pays out action[0] per step for 5 steps → episode reward = 10.0.
        # Mean over 4 episodes is still 10.0.
        assert len(results) == 1
        assert results[0].reward == pytest.approx(10.0)
        # total_steps is summed across episodes (5 × 4).
        assert results[0].total_steps == 20

    def test_worker_crash_isolates_failure(self):
        """A worker crash on one individual still lets the other results come through."""
        template = DummyPolicy()
        # Env crashes on the candidate that emits action[0] ≈ 99.
        env_factory = make_crashing_env_factory(crash_on_idx=99, max_steps=3)

        evaluator = ParallelEvaluator(
            n_workers=2,
            make_env_fn=env_factory,
            template_policy=template,
            **_FAST_KW,
        )
        try:
            candidates = [
                Candidate(individual_idx=0, flat_weights=np.array([1.0])),
                Candidate(individual_idx=1, flat_weights=np.array([99.0])),  # boom
                Candidate(individual_idx=2, flat_weights=np.array([2.0])),
                Candidate(individual_idx=3, flat_weights=np.array([3.0])),
            ]
            results = evaluator.evaluate(candidates)
        finally:
            evaluator.close()

        assert [r.individual_idx for r in results] == [0, 1, 2, 3]
        crashed = results[1]
        assert crashed.failed is True
        assert crashed.reward == float("-inf")
        # All non-crashing individuals must come back even though one worker died.
        for r in (results[0], results[2], results[3]):
            assert not r.failed
            assert r.reward != float("-inf")

    def test_close_joins_all_workers(self):
        """After close(), no worker is alive."""
        evaluator = ParallelEvaluator(
            n_workers=3,
            make_env_fn=make_dummy_env,
            template_policy=DummyPolicy(),
            **_FAST_KW,
        )
        # Workers spawn even with no jobs.
        assert all(w.is_alive() for w in evaluator.workers)
        evaluator.close()
        # Join is bounded; everyone should be down within a few seconds.
        for w in evaluator.workers:
            assert not w.is_alive(), f"worker {w.name} still alive after close()"

    def test_close_is_idempotent(self):
        """Calling close() twice is safe (atexit also calls it)."""
        evaluator = ParallelEvaluator(
            n_workers=1,
            make_env_fn=make_dummy_env,
            template_policy=DummyPolicy(),
            **_FAST_KW,
        )
        evaluator.close()
        evaluator.close()  # should be a no-op

    def test_evaluate_after_close_raises(self):
        evaluator = ParallelEvaluator(
            n_workers=1,
            make_env_fn=make_dummy_env,
            template_policy=DummyPolicy(),
            **_FAST_KW,
        )
        evaluator.close()
        with pytest.raises(RuntimeError):
            evaluator.evaluate([Candidate(0, np.array([0.0]), 1)])

    def test_empty_candidate_list_returns_empty(self):
        evaluator = ParallelEvaluator(
            n_workers=2,
            make_env_fn=make_dummy_env,
            template_policy=DummyPolicy(),
            **_FAST_KW,
        )
        try:
            assert evaluator.evaluate([]) == []
        finally:
            evaluator.close()

    def test_n_workers_zero_rejected(self):
        with pytest.raises(ValueError):
            ParallelEvaluator(
                n_workers=0,
                make_env_fn=make_dummy_env,
                template_policy=DummyPolicy(),
                **_FAST_KW,
            )

    def test_more_candidates_than_workers(self):
        """8 jobs on 2 workers: every job still gets evaluated exactly once."""
        template = DummyPolicy()
        evaluator = ParallelEvaluator(
            n_workers=2,
            make_env_fn=make_dummy_env,
            template_policy=template,
            **_FAST_KW,
        )
        try:
            candidates = [
                Candidate(individual_idx=i, flat_weights=np.array([float(i)]),
                          eval_episodes=1)
                for i in range(8)
            ]
            results = evaluator.evaluate(candidates)
        finally:
            evaluator.close()

        assert [r.individual_idx for r in results] == list(range(8))
        # DummyEnv pays out action[0] per step × max_steps=5.
        for i, r in enumerate(results):
            assert r.reward == pytest.approx(float(i) * 5.0)

    def test_determinism_same_seed(self):
        """Two evaluators with the same seed produce identical results for identical inputs."""
        template = DummyPolicy()
        candidates = [
            Candidate(individual_idx=i, flat_weights=np.array([float(i) * 0.25]),
                      eval_episodes=1)
            for i in range(4)
        ]

        ev1 = ParallelEvaluator(
            n_workers=2, make_env_fn=make_dummy_env, template_policy=template,
            base_seed=42, **_FAST_KW,
        )
        try:
            r1 = ev1.evaluate(candidates)
        finally:
            ev1.close()

        ev2 = ParallelEvaluator(
            n_workers=2, make_env_fn=make_dummy_env, template_policy=template,
            base_seed=42, **_FAST_KW,
        )
        try:
            r2 = ev2.evaluate(candidates)
        finally:
            ev2.close()

        for a, b in zip(r1, r2):
            assert a.reward == pytest.approx(b.reward)

    def test_episode_time_limit_broadcast(self):
        """Passing episode_time_limit_s reaches workers' set_episode_time_limit hook."""
        # We can't easily inspect a child's env state from the parent.
        # Instead, just verify the call path doesn't crash and results still
        # come back — the underlying DummyEnv accepts set_episode_time_limit.
        template = DummyPolicy()
        evaluator = ParallelEvaluator(
            n_workers=2, make_env_fn=make_dummy_env, template_policy=template,
            **_FAST_KW,
        )
        try:
            candidates = [
                Candidate(individual_idx=i, flat_weights=np.array([1.0]),
                          eval_episodes=1)
                for i in range(2)
            ]
            results = evaluator.evaluate(candidates, episode_time_limit_s=30.0)
        finally:
            evaluator.close()
        assert len(results) == 2
        assert all(not r.failed for r in results)


class TestMaybeBuildEvaluator:
    """Validation of train_rl's evaluator-construction guard (issue #229)."""

    def test_returns_none_when_n_workers_is_one(self):
        from framework.training import _maybe_build_evaluator
        result = _maybe_build_evaluator(
            n_workers=1, policy_type="sc2_genetic", loop_kind="genetic",
            policy=object(), make_env_fn=make_dummy_env,
            training_params={}, in_game_episode_s=10.0,
        )
        assert result is None

    def test_rejects_non_population_policy(self):
        from framework.training import _maybe_build_evaluator
        with pytest.raises(ValueError, match="population-based"):
            _maybe_build_evaluator(
                n_workers=4, policy_type="hill_climbing", loop_kind=None,
                policy=object(), make_env_fn=make_dummy_env,
                training_params={}, in_game_episode_s=10.0,
            )

    def test_rejects_q_learning_loop(self):
        from framework.training import _maybe_build_evaluator
        with pytest.raises(ValueError, match="population-based"):
            _maybe_build_evaluator(
                n_workers=2, policy_type="epsilon_greedy", loop_kind="q_learning",
                policy=object(), make_env_fn=make_dummy_env,
                training_params={}, in_game_episode_s=10.0,
            )

    def test_caps_n_workers_at_population_size(self, caplog):
        """n_workers=8 with population_size=4 should cap at 4 and warn."""
        from framework.training import _maybe_build_evaluator

        # Stand-in for a policy with a ._template and a population_size of 4.
        class _FakePop:
            population_size = 4
            _template = DummyPolicy()

        with caplog.at_level("WARNING"):
            evaluator = _maybe_build_evaluator(
                n_workers=8, policy_type="sc2_cmaes", loop_kind="cmaes",
                policy=_FakePop(), make_env_fn=make_dummy_env,
                training_params={"worker_start_stagger_s": 0.0,
                                 "worker_warmup_timeout_s": 5.0},
                in_game_episode_s=2.0,
            )
        try:
            assert evaluator.n_workers == 4
            assert any("capping at 4" in r.message for r in caplog.records)
        finally:
            evaluator.close()


def test_smoke_sc2_parallel_evaluator():
    """Opt-in integration smoke test for SC2 binary spawn path (issue #229).

    Skipped unless RUN_SC2_TESTS=1 — the actual PySC2 binary takes
    10-20s to start and isn't installed in CI.
    """
    if os.environ.get("RUN_SC2_TESTS") != "1":
        pytest.skip("Set RUN_SC2_TESTS=1 to exercise the SC2 binary spawn path.")

    # This branch is left for the (manual) dogfood run described in
    # plans/issue-229-sc2-multi-instance-intra-run.md §7 step 5.
    pytest.skip("SC2 smoke test stub — populate when wiring is in place.")
