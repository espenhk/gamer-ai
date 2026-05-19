"""
Parallel intra-run evaluation for population-based policies.

Spawns N persistent worker processes, each holding one game env instance.
Coordinates work distribution via multiprocessing.Queue.

Each worker keeps its env alive across generations (binary startup is the
dominant cost for SC2) and reconstructs the candidate policy per job via
``template_policy.with_flat(weights)``.

Public API:
  - ParallelEvaluator(n_workers, make_env_fn, template_policy, ...)
  - evaluator.evaluate(candidates, *, warmup_action=None, warmup_steps=0,
                       episode_time_limit_s=None) -> results
  - evaluator.close(): tear down all workers

Generation-synchronous: ``evaluate`` blocks until all candidates have
results.  See plans/issue-229-sc2-multi-instance-intra-run.md.
"""
from __future__ import annotations

import atexit
import dataclasses
import logging
import multiprocessing as mp
import queue as queue_mod
import time
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Candidate:
    """A single individual to evaluate."""
    individual_idx: int
    flat_weights: np.ndarray
    eval_episodes: int = 1


@dataclasses.dataclass
class EpisodeResult:
    """Result from evaluating one candidate over eval_episodes episodes."""
    individual_idx: int
    reward: float          # mean over eval_episodes
    info: dict             # info dict from the last episode (for logging)
    trace: Any             # RunTrace | None from the last episode
    total_steps: int       # summed across eval_episodes
    failed: bool = False   # True if the worker raised on this candidate


# Sentinel used on the job queue to tell a worker to shut down.
_JOB_SENTINEL = "__shutdown__"


@dataclasses.dataclass
class _WorkerError:
    """Wire message: worker hit a fatal error and is about to exit."""
    worker_id: int
    individual_idx: int | None
    reason: str


class ParallelEvaluator:
    """
    Manages N worker processes, each with one persistent env instance.

    Coordinates via multiprocessing.Queue: main process pushes Candidate jobs,
    workers push EpisodeResult back, main process gathers and returns.
    """

    def __init__(
        self,
        n_workers: int,
        make_env_fn: Callable[[], Any],
        template_policy: Any,
        *,
        worker_start_stagger_s: float = 5.0,
        worker_warmup_timeout_s: float = 90.0,
        per_episode_timeout_s: float = 120.0,
        base_seed: int = 0,
    ):
        """
        Spawn n_workers child processes.

        Args:
            n_workers: number of parallel env instances.
            make_env_fn: zero-arg picklable factory returning a BaseGameEnv.
            template_policy: a picklable policy carrying obs_spec / head_names
                etc.; per-candidate policies are built via
                ``template_policy.with_flat(flat_weights)``.
            worker_start_stagger_s: sleep between spawning children (lets
                PySC2's port-picker / absl.flags settle).
            worker_warmup_timeout_s: per-worker startup budget; folded into
                the per-job timeout in evaluate().
            per_episode_timeout_s: budget per episode (in_game_episode_s × 2
                is a sensible default).
            base_seed: each worker's RNG seed = base_seed + worker_id.
        """
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")

        self.n_workers = n_workers
        self.make_env_fn = make_env_fn
        self.template_policy = template_policy
        self.worker_start_stagger_s = worker_start_stagger_s
        self.worker_warmup_timeout_s = worker_warmup_timeout_s
        self.per_episode_timeout_s = per_episode_timeout_s
        self.base_seed = base_seed

        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self.job_queue: mp.Queue = ctx.Queue()
        self.result_queue: mp.Queue = ctx.Queue()
        self.workers: list[mp.Process] = []
        self._closed = False

        logger.info(
            "[ParallelEvaluator] spawning %d workers (stagger=%.1fs, warmup=%.1fs)",
            n_workers, worker_start_stagger_s, worker_warmup_timeout_s,
        )

        for worker_id in range(n_workers):
            p = ctx.Process(
                target=_worker_main,
                args=(
                    self.job_queue,
                    self.result_queue,
                    make_env_fn,
                    template_policy,
                    worker_id,
                    base_seed + worker_id,
                ),
                daemon=True,
                name=f"ParallelEval-Worker-{worker_id}",
            )
            p.start()
            self.workers.append(p)
            if worker_id < n_workers - 1 and worker_start_stagger_s > 0:
                time.sleep(worker_start_stagger_s)

        atexit.register(self.close)

    def evaluate(
        self,
        candidates: list[Candidate],
        *,
        warmup_action: np.ndarray | None = None,
        warmup_steps: int = 0,
        episode_time_limit_s: float | None = None,
    ) -> list[EpisodeResult]:
        """
        Evaluate all candidates in parallel.

        Returns results sorted by individual_idx regardless of arrival order.
        Missing results (timeout / worker crash) are returned with
        reward=-inf and failed=True.
        """
        if not candidates:
            return []
        if self._closed:
            raise RuntimeError("evaluate() called after close()")

        max_eps = max(c.eval_episodes for c in candidates)
        per_job_budget = (
            self.worker_warmup_timeout_s
            + self.per_episode_timeout_s * max_eps * 2.0
        )

        # Attach the per-generation episode_time_limit_s to every job rather
        # than broadcasting N copies of a control message on the shared
        # queue: mp.Queue is multi-consumer, so a fast worker can drain
        # several copies and leave another worker un-updated for this
        # generation.  Each worker applies the limit before each candidate.
        for cand in candidates:
            self.job_queue.put((cand, warmup_action, warmup_steps,
                                episode_time_limit_s))

        n_active_workers = self.n_workers
        results: dict[int, EpisodeResult] = {}
        expected_indices = {c.individual_idx for c in candidates}

        # Worst case: all jobs serialised onto one surviving worker.
        overall_deadline = time.time() + per_job_budget * len(candidates)

        while len(results) < len(candidates) and n_active_workers > 0:
            remaining = overall_deadline - time.time()
            if remaining <= 0:
                logger.error(
                    "[ParallelEvaluator] timeout: %d/%d results received",
                    len(results), len(candidates),
                )
                break
            try:
                msg = self.result_queue.get(timeout=remaining)
            except queue_mod.Empty:
                logger.error(
                    "[ParallelEvaluator] result queue empty after %.1fs; "
                    "%d/%d received",
                    per_job_budget * len(candidates),
                    len(results), len(candidates),
                )
                break

            if isinstance(msg, _WorkerError):
                logger.warning(
                    "[ParallelEvaluator] worker %d died (%s)",
                    msg.worker_id, msg.reason,
                )
                n_active_workers -= 1
                if msg.individual_idx is not None and msg.individual_idx not in results:
                    results[msg.individual_idx] = _failed_result(msg.individual_idx)
                continue

            assert isinstance(msg, EpisodeResult)
            results[msg.individual_idx] = msg

        # Fill in placeholders for any individuals nobody reported on.
        for idx in expected_indices - set(results.keys()):
            logger.warning(
                "[ParallelEvaluator] no result for individual %d; assigning -inf", idx,
            )
            results[idx] = _failed_result(idx)

        return sorted(results.values(), key=lambda r: r.individual_idx)

    def close(self):
        """Send sentinels and join all workers."""
        if self._closed:
            return
        self._closed = True
        logger.info("[ParallelEvaluator] closing %d workers", self.n_workers)

        # Drain any straggler results so workers don't block on a full queue.
        try:
            while True:
                self.result_queue.get_nowait()
        except queue_mod.Empty:
            pass

        for _ in range(self.n_workers):
            try:
                self.job_queue.put(_JOB_SENTINEL)
            except Exception:
                pass

        for worker in self.workers:
            if not worker.is_alive():
                continue
            worker.join(timeout=10.0)
            if worker.is_alive():
                logger.warning(
                    "[ParallelEvaluator] worker %s did not join; terminating",
                    worker.name,
                )
                worker.terminate()
                worker.join(timeout=2.0)

        try:
            self.job_queue.close()
            self.result_queue.close()
        except Exception:
            pass


def _failed_result(idx: int) -> EpisodeResult:
    return EpisodeResult(
        individual_idx=idx,
        reward=float("-inf"),
        info={},
        trace=None,
        total_steps=0,
        failed=True,
    )


def _worker_main(
    job_queue: mp.Queue,
    result_queue: mp.Queue,
    make_env_fn: Callable[[], Any],
    template_policy: Any,
    worker_id: int,
    seed: int,
) -> None:
    """Worker entry point — runs in a spawned child process."""
    env = None
    current_idx: int | None = None
    try:
        # Match the import path used inside train_rl; cheaper than passing
        # _run_episode through pickle (it captures lots of module state).
        from framework.training import _run_episode

        # Seed numpy in this worker so any stochastic policy step / env
        # reset uses an independent stream per worker.
        np.random.seed(seed)

        logger.info("[Worker %d] building env (seed=%d)", worker_id, seed)
        env = make_env_fn()
        logger.info("[Worker %d] ready", worker_id)

        last_time_limit_s: float | None = None
        while True:
            msg = job_queue.get()
            if msg == _JOB_SENTINEL:
                logger.info("[Worker %d] sentinel received; exiting", worker_id)
                break

            candidate, warmup_action, warmup_steps, episode_time_limit_s = msg
            current_idx = candidate.individual_idx

            # Apply per-job episode time limit before episodes start.  We
            # only re-apply when the value actually changed to spare
            # otherwise-redundant calls on long generations.
            if (
                episode_time_limit_s is not None
                and episode_time_limit_s != last_time_limit_s
                and hasattr(env, "set_episode_time_limit")
            ):
                env.set_episode_time_limit(episode_time_limit_s)
                last_time_limit_s = episode_time_limit_s

            individual = template_policy.with_flat(candidate.flat_weights)

            ep_rewards: list[float] = []
            last_info: dict = {}
            last_trace: Any = None
            total_steps = 0

            for _ in range(candidate.eval_episodes):
                obs, reset_info = env.reset()
                reward, info, _, steps, trace = _run_episode(
                    env, individual, obs,
                    warmup_action=warmup_action, warmup_steps=warmup_steps,
                    reset_info=reset_info,
                )
                ep_rewards.append(reward)
                last_info = info
                last_trace = trace
                total_steps += steps

            mean_reward = float(sum(ep_rewards) / len(ep_rewards))
            result_queue.put(EpisodeResult(
                individual_idx=candidate.individual_idx,
                reward=mean_reward,
                info=last_info,
                trace=last_trace,
                total_steps=total_steps,
            ))
            current_idx = None

    except Exception as e:
        logger.exception("[Worker %d] crashed: %s", worker_id, e)
        try:
            result_queue.put(_WorkerError(
                worker_id=worker_id,
                individual_idx=current_idx,
                reason=f"{type(e).__name__}: {e}",
            ))
        except Exception:
            pass

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
