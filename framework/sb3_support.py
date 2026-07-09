"""Training-loop glue for the Stable-Baselines3-backed policies.

SB3 algorithms own their own training loop (``model.learn(total_timesteps)``),
so instead of the framework's per-step ``update`` path they are driven here.
``run_sb3_loop`` wraps the game's :class:`~framework.base_env.BaseGameEnv`
(Gymnasium-compatible) for SB3, runs ``learn``, records one
:class:`~framework.analytics.GreedySimResult` per completed episode via a
callback, and returns a :class:`~framework.training.GreedyLoopResult` so the
standard analytics path works unchanged.

The heavy SB3 imports live inside the functions, so importing this module is
cheap — it is only imported when a ``LOOP_TYPE == "sb3"`` policy actually runs.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


def _atomic_save(save_fn: Callable[[str], None], final_path: str) -> None:
    """Call ``save_fn(tmp_path)`` then atomically replace *final_path*.

    SB3's ``save`` / ``save_replay_buffer`` write directly to the given path;
    an interruption mid-write would corrupt it and break resume. Writing to a
    ``.tmp`` sibling first and swapping it in with ``os.replace`` (atomic on
    POSIX and Windows) means the previous checkpoint stays valid until the new
    one is fully written. (SB3's ``open_path`` only force-appends its own
    suffix when the given path has *no* suffix at all, so the ``.tmp``-suffixed
    path is never mangled.)
    """
    tmp_path = f"{final_path}.tmp"
    try:
        save_fn(tmp_path)
        os.replace(tmp_path, final_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


class DiscretizeActionWrapper(gym.Wrapper):
    """Expose a ``Discrete(n)`` action space over a fixed table of actions.

    DQN-family algorithms (QR-DQN) need a discrete action space, but the games
    expose a continuous ``Box``.  This maps a discrete index to the matching row
    of ``discrete_actions`` (the same table the tabular policies use) before
    forwarding to the wrapped env.
    """

    def __init__(self, env: gym.Env, discrete_actions: np.ndarray) -> None:
        super().__init__(env)
        self._actions = np.asarray(discrete_actions, dtype=np.float32)
        if self._actions.ndim != 2 or len(self._actions) == 0:
            raise ValueError("DiscretizeActionWrapper requires a non-empty 2-D discrete_actions table.")
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def step(self, action):
        idx = int(np.asarray(action).reshape(-1)[0])
        return self.env.step(self._actions[idx])


def _save_checkpoint(model, *, weights_file: str, off_policy: bool) -> None:
    """Persist the latest training state (model + replay buffer) for crash recovery.

    Distinct from the best-reward snapshot: this is overwritten on every
    checkpoint tick regardless of episode reward, so a resume picks up the
    actual training state as of the last checkpoint rather than a possibly
    stale best-episode snapshot.
    """
    from framework.sb3_policies import _checkpoint_zip_path, _replay_buffer_path

    _atomic_save(model.save, _checkpoint_zip_path(weights_file))
    if off_policy:
        _atomic_save(model.save_replay_buffer, _replay_buffer_path(weights_file))


def _make_sim_recorder(greedy_sims: list, *, weights_file: str, patience: int, checkpoint_freq: int, off_policy: bool):
    """Build an SB3 callback that records per-episode telemetry, saves the best
    model, and periodically checkpoints the full training state for resume."""
    from stable_baselines3.common.callbacks import BaseCallback

    from framework.analytics import GreedySimResult
    from framework.sb3_policies import _model_zip_path

    best_path = _model_zip_path(weights_file)

    class _SimRecorder(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.best_reward = float("-inf")
            self.best_sim: int | None = None
            self.sims_since_improve = 0
            self.early_stopped = False
            self.early_stop_sim: int | None = None
            self._last_checkpoint_step = 0

        def _on_training_start(self) -> None:
            # On resume, num_timesteps already reflects steps from prior
            # invocations; anchor the cadence to that so the first _on_step
            # doesn't immediately re-checkpoint (num_timesteps - 0 would
            # already exceed checkpoint_freq).
            self._last_checkpoint_step = self.num_timesteps

        def _on_step(self) -> bool:
            if checkpoint_freq > 0 and self.num_timesteps - self._last_checkpoint_step >= checkpoint_freq:
                _save_checkpoint(self.model, weights_file=weights_file, off_policy=off_policy)
                self._last_checkpoint_step = self.num_timesteps
                logger.debug("[sb3] checkpoint saved at %d timesteps", self.num_timesteps)
            for info in self.locals.get("infos", []):
                ep = info.get("episode") if isinstance(info, dict) else None
                if ep is None:
                    continue
                sim_idx = len(greedy_sims)
                reward = float(ep["r"])
                total_steps = int(ep["l"])
                improved = reward > self.best_reward
                if improved:
                    self.best_reward = reward
                    self.best_sim = sim_idx
                    self.sims_since_improve = 0
                    _atomic_save(self.model.save, best_path)
                else:
                    self.sims_since_improve += 1
                greedy_sims.append(
                    GreedySimResult(
                        sim=sim_idx,
                        reward=reward,
                        improved=improved,
                        throttle_counts=[0, 0, 0],
                        total_steps=total_steps,
                    )
                )
                logger.info("ep %d  r=%+.1f  steps=%d%s", sim_idx, reward, total_steps, "  *best*" if improved else "")
                if patience > 0 and self.sims_since_improve >= patience:
                    self.early_stopped = True
                    self.early_stop_sim = sim_idx
                    logger.info("[sb3] early stop at ep %d (no improvement for %d eps)", sim_idx, patience)
                    return False
            return True

    return _SimRecorder()


def run_sb3_loop(
    *,
    env,
    policy,
    n_sims: int,
    weights_file: str,
    training_params: dict,
    patience: int = 0,
    warmup_action: Any = None,
    warmup_steps: int = 0,
    live_monitor: Any = None,
    log_stats_every_n_sims: int = 0,
):
    """Drive an SB3-backed policy and return a GreedyLoopResult.

    ``warmup_action`` / ``live_monitor`` are accepted for dispatch parity but
    not applied — SB3 owns the rollout loop, so the framework's forced-warmup
    and live-monitor hooks do not participate in an SB3 run.
    """
    from stable_baselines3.common.monitor import Monitor

    from framework.sb3_policies import _model_zip_path
    from framework.training import GreedyLoopResult

    if warmup_action is not None and warmup_steps > 0:
        logger.info("[sb3] warmup (action forcing) is not applied under the SB3 loop; ignoring.")

    wrapped = env
    if getattr(policy, "REQUIRES_DISCRETE", False):
        if policy._discrete_actions is None:
            raise ValueError(f"policy_type={policy.POLICY_TYPE!r} needs a discrete action table but none was provided.")
        wrapped = DiscretizeActionWrapper(wrapped, policy._discrete_actions)
    wrapped = Monitor(wrapped)

    target_timesteps = policy.total_timesteps(n_sims)
    model = policy.build_model(wrapped)
    resumed = bool(getattr(policy, "_resume", False))
    already_done = int(model.num_timesteps) if resumed else 0
    remaining_timesteps = max(0, target_timesteps - already_done)
    if resumed:
        logger.info(
            "[sb3] %s — resuming at %d/%d timesteps (%d remaining, algo=%s)",
            policy.POLICY_TYPE,
            already_done,
            target_timesteps,
            remaining_timesteps,
            policy.SB3_ALGO,
        )
    else:
        logger.info(
            "[sb3] %s — training for %d timesteps (algo=%s)", policy.POLICY_TYPE, target_timesteps, policy.SB3_ALGO
        )

    greedy_sims: list = []
    checkpoint_freq = int(policy._params.get("checkpoint_freq", 10_000))
    recorder = _make_sim_recorder(
        greedy_sims,
        weights_file=weights_file,
        patience=patience,
        checkpoint_freq=checkpoint_freq,
        off_policy=getattr(policy, "OFF_POLICY", False),
    )

    if remaining_timesteps > 0:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=recorder,
            progress_bar=False,
            reset_num_timesteps=not resumed,
        )
    else:
        logger.info("[sb3] already at or past target total_timesteps=%d; skipping training.", target_timesteps)
    policy.set_model(model)

    # Checkpoint the final training state so a subsequent resume picks up
    # exactly where this run left off, even if it ended cleanly.
    _save_checkpoint(model, weights_file=weights_file, off_policy=getattr(policy, "OFF_POLICY", False))

    # Persist YAML metadata; the callback already saved the best-scoring model
    # zip whenever an episode improved on it this invocation.
    import yaml

    with open(weights_file, "w") as f:
        yaml.dump(policy.to_cfg(), f, default_flow_style=False, sort_keys=False)
    best_path = _model_zip_path(weights_file)
    if recorder.best_sim is None:
        # No episode completed this invocation (e.g. training was skipped
        # because the resumed run already hit its target, or the remaining
        # budget ended mid-episode). Only fall back to writing the current
        # model as the "best" snapshot if none exists yet — otherwise this
        # would clobber a genuine champion with the (not necessarily better)
        # checkpoint state the run resumed from.
        if not os.path.exists(best_path):
            _atomic_save(model.save, best_path)
        best_reward = float("-inf") if not greedy_sims else max(s.reward for s in greedy_sims)
    else:
        best_reward = recorder.best_reward

    return GreedyLoopResult(
        policy=policy,
        best_reward=best_reward,
        greedy_sims=greedy_sims,
        early_stopped=recorder.early_stopped,
        early_stop_sim=recorder.early_stop_sim,
    )
