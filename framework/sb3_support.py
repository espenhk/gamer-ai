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
from typing import Any

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


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


def _make_sim_recorder(greedy_sims: list, *, weights_file: str, patience: int):
    """Build an SB3 callback that records per-episode telemetry + saves the best model."""
    from stable_baselines3.common.callbacks import BaseCallback

    from framework.analytics import GreedySimResult
    from framework.sb3_policies import _model_zip_path

    zip_path = _model_zip_path(weights_file)

    class _SimRecorder(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.best_reward = float("-inf")
            self.best_sim: int | None = None
            self.sims_since_improve = 0
            self.early_stopped = False
            self.early_stop_sim: int | None = None

        def _on_step(self) -> bool:
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
                    self.model.save(zip_path)
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

    total_timesteps = policy.total_timesteps(n_sims)
    logger.info("[sb3] %s — training for %d timesteps (algo=%s)", policy.POLICY_TYPE, total_timesteps, policy.SB3_ALGO)

    model = policy.build_model(wrapped)
    greedy_sims: list = []
    recorder = _make_sim_recorder(greedy_sims, weights_file=weights_file, patience=patience)

    model.learn(total_timesteps=total_timesteps, callback=recorder, progress_bar=False)
    policy.set_model(model)

    # Persist YAML metadata; the callback already saved the best-scoring model
    # zip.  If no episode ever completed, save the final model so an artifact
    # exists for inference.
    import yaml

    with open(weights_file, "w") as f:
        yaml.dump(policy.to_cfg(), f, default_flow_style=False, sort_keys=False)
    if recorder.best_sim is None:
        model.save(_model_zip_path(weights_file))
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
