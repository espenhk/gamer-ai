"""AssettoCorsaEnv — Gymnasium environment for Assetto Corsa.

Wraps an upstream ACClient (which itself wraps the assetto-corsa-rl gym
environment) to satisfy the framework's BaseGameEnv interface. The action
space matches TMNF's ([steer, accel, brake]) so framework policies are
drop-in compatible.

Observation
-----------
See games.assetto_corsa.obs_spec.AC_OBS_SPEC for the full feature list.
Vision features (when enabled) are appended at the end and exposed via
``AC_OBS_SPEC.with_vision(n_vision)``.

Action
------
Box([-1, 0, 0], [1, 1, 1], shape=(3,), dtype=float32)
    [0] steer  ∈ [-1, 1]
    [1] accel  ∈ [0, 1]   (thresholded at 0.5 → bool)
    [2] brake  ∈ [0, 1]   (thresholded at 0.5 → bool)

Termination
-----------
Terminated: client reports finished, or |lateral_offset| > crash_threshold_m
Truncated:  elapsed > max_episode_time_s
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from gymnasium import spaces

from framework.base_env import BaseGameEnv
from games.assetto_corsa.clients.ac_client import ACClient, ACStepState, DEFAULT_ENV_ID
from games.assetto_corsa.obs_spec import AC_OBS_SPEC, with_vision
from games.assetto_corsa.reward import RewardCalculator, RewardConfig

logger = logging.getLogger(__name__)


_DEFAULT_REWARD_CONFIG = os.path.join(
    os.path.dirname(__file__), "config", "reward_config.yaml"
)


class AssettoCorsaEnv(BaseGameEnv):
    """Gymnasium environment for Assetto Corsa reinforcement learning.

    Parameters
    ----------
    reward_config:
        RewardConfig instance. If None, loaded from the package default.
    max_episode_time_s:
        Wall-clock seconds before the episode is truncated.
    n_vision:
        Number of vision-distance features appended to the observation
        (0 = vision disabled).
    env_id:
        Upstream gym env id (defaults to ``"AssettoCorsa-v0"``).
    env_factory:
        Optional callable used in place of ``gymnasium.make``. Tests
        inject a stub here so no real AC binary is required.
    env_kwargs:
        Extra keyword arguments forwarded to the env factory.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_config: RewardConfig | None = None,
        max_episode_time_s: float = 150.0,
        n_vision: int = 0,
        env_id: str = DEFAULT_ENV_ID,
        env_factory: Callable[..., Any] | None = None,
        env_kwargs: dict | None = None,
    ) -> None:
        super().__init__()

        self._reward_config = reward_config or RewardConfig.from_yaml(_DEFAULT_REWARD_CONFIG)
        self._max_episode_time_s = max_episode_time_s
        self._n_vision = n_vision
        self._reward_calc = RewardCalculator(self._reward_config)

        self._obs_spec = with_vision(n_vision)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_spec.dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._client = ACClient(env_id=env_id, env_factory=env_factory,
                                env_kwargs=env_kwargs)

        self._prev_state: ACStepState | None = None
        self._elapsed_s: float = 0.0
        self._episode_start_s: float = 0.0
        self._laps_completed: int = 0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        step = self._client.reset()
        self._prev_state = step
        self._elapsed_s = 0.0
        self._episode_start_s = time.monotonic()
        self._laps_completed = 0
        obs = self._build_obs(step)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        step = self._client.step(np.asarray(action, dtype=np.float32))
        self._elapsed_s = time.monotonic() - self._episode_start_s

        finished = bool(step.finished) or (
            step.track_progress is not None and step.track_progress >= 1.0
        )
        crashed = (
            step.lateral_offset is not None
            and abs(step.lateral_offset) > self._reward_config.crash_threshold_m
        )
        time_over = self._elapsed_s > self._max_episode_time_s

        accelerating = bool(float(action[1]) >= 0.5) if len(action) >= 2 else False

        reward = self._reward_calc.compute(
            prev_state = self._prev_state,
            curr_state = step,
            finished   = finished,
            elapsed_s  = self._elapsed_s,
            info       = {"accelerating": accelerating},
            n_ticks    = 1,
        )

        terminated = finished or crashed or step.terminated
        truncated  = step.truncated or (time_over and not terminated)

        if finished:
            termination_reason: str | None = "finish"
            self._laps_completed += 1
        elif crashed:
            termination_reason = "crash"
        elif time_over:
            termination_reason = "timeout"
        elif step.terminated:
            termination_reason = "env"
        else:
            termination_reason = None

        info = self._get_game_info()
        info.update({
            "finished":        finished,
            "laps_completed":  self._laps_completed,
            "elapsed_s":       self._elapsed_s,
            "termination_reason": termination_reason,
        })

        self._prev_state = step
        obs = self._build_obs(step)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # BaseGameEnv hooks
    # ------------------------------------------------------------------

    def get_episode_time_limit(self) -> float:
        return self._max_episode_time_s

    def set_episode_time_limit(self, seconds: float) -> None:
        self._max_episode_time_s = seconds

    def _get_game_info(self) -> dict:
        s = self._prev_state
        if s is None:
            return {}
        return {
            "pos_x":          s.pos_x,
            "pos_z":          s.pos_z,
            "track_progress": s.track_progress or 0.0,
            "lateral_offset": s.lateral_offset or 0.0,
        }

    def _build_obs(self, step: ACStepState) -> np.ndarray:
        wheels = step.wheel_slip
        ang    = step.angular_velocity
        state = np.array(
            [
                step.speed_ms,
                step.lateral_offset or 0.0,
                step.yaw_error,
                step.pitch,
                step.roll,
                step.track_progress or 0.0,
                step.steering_angle,
                step.engine_rpm,
                step.gear,
                wheels[0], wheels[1], wheels[2], wheels[3],
                ang[0],    ang[1],    ang[2],
            ],
            dtype=np.float32,
        )
        if self._n_vision > 0:
            vis = step.vision
            if vis is None or vis.size != self._n_vision:
                vis = np.zeros(self._n_vision, dtype=np.float32)
            state = np.concatenate([state, vis.astype(np.float32)])
        return state


def make_env(
    experiment_dir: str | Path,
    speed: float = 1.0,
    in_game_episode_s: float = 150.0,
    n_vision: int = 0,
    env_factory: Callable[..., Any] | None = None,
) -> AssettoCorsaEnv:
    """Factory mirroring games.tmnf.env.make_env.

    *speed* is accepted for API parity with TMNF; AC's gym wrapper does not
    expose a real-time-acceleration knob at this layer.
    """
    experiment_dir = Path(experiment_dir)
    reward_cfg_path = experiment_dir / "reward_config.yaml"
    reward_config = (
        RewardConfig.from_yaml(str(reward_cfg_path))
        if reward_cfg_path.exists()
        else RewardConfig.from_yaml(_DEFAULT_REWARD_CONFIG)
    )
    return AssettoCorsaEnv(
        reward_config       = reward_config,
        max_episode_time_s  = in_game_episode_s,
        n_vision            = n_vision,
        env_factory         = env_factory,
    )
