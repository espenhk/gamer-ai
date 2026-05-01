"""
TorcsEnv — Gymnasium environment wrapping TORCS via gym_torcs for RL training.

Observation space  (BASE_OBS_DIM floats, dtype float32)
-------------------------------------------------------
  See games.torcs.obs_spec.TORCS_OBS_SPEC for the full list.
  Summary:
    [0]   speed_ms          — vehicle speed in m/s
    [1]   lateral_offset_m  — metres from track centre
    [2]   yaw_error_rad     — heading error vs track direction
    [3]   track_progress    — fraction of lap completed [0, 1]
    [4]   rpm               — engine RPM
    [5-8] wheel_N_spin      — wheel spin velocities (rad/s)
    [9-17] track_edge_N     — rangefinder track-edge distances
    [18]  track_position    — normalised track position [-1, 1]

Action space
------------
  Box([-1, 0, 0], [1, 1, 1], shape=(3,), dtype=float32)
    [0] steer  — steering input in [-1, 1]
    [1] accel  — throttle in [0, 1]
    [2] brake  — braking  in [0, 1]

Episode lifecycle
-----------------
  reset() → launch/relaunch TORCS → return initial obs
  step()  → set action, read sensors, compute reward
  Terminated when: track_progress wraps (lap finished)
               or: |lateral_offset| > crash_threshold_m
  Truncated  when: elapsed_time > max_episode_time_s
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from framework.base_env import BaseGameEnv
from games.torcs.client import TorcsClient
from games.torcs.obs_spec import BASE_OBS_DIM
from games.torcs.reward import TorcsRewardConfig, TorcsRewardCalculator

logger = logging.getLogger(__name__)


class TorcsEnv(BaseGameEnv):
    """Gymnasium environment for TORCS reinforcement learning.

    Parameters
    ----------
    reward_config :
        TorcsRewardConfig instance.  If None, uses Python defaults.
    max_episode_time_s :
        Wall-clock seconds before the episode is truncated.
    vision :
        If True, request pixel observations from TORCS.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_config: TorcsRewardConfig | None = None,
        max_episode_time_s: float = 120.0,
        vision: bool = False,
    ) -> None:
        super().__init__()

        self._reward_config = reward_config or TorcsRewardConfig()
        self._max_episode_time_s = max_episode_time_s
        self._reward_calc = TorcsRewardCalculator(self._reward_config)

        obs_dim = BASE_OBS_DIM
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._client = TorcsClient(vision=vision)

        # Episode tracking
        self._prev_obs: np.ndarray | None = None
        self._prev_progress: float = 0.0
        self._elapsed_s: float = 0.0
        self._episode_start_s: float = 0.0
        self._step_count: int = 0

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

        # Relaunch every reset to avoid TORCS memory leak.
        obs = self._client.reset(relaunch=True)

        self._prev_obs = obs
        self._prev_progress = float(obs[3])  # track_progress index
        self._elapsed_s = 0.0
        self._episode_start_s = time.monotonic()
        self._step_count = 0
        self._reward_calc.reset()

        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, _torcs_reward, done, _info = self._client.step(action)

        self._step_count += 1
        self._elapsed_s = time.monotonic() - self._episode_start_s

        speed = float(obs[0])
        lateral_offset = float(obs[1])
        curr_progress = float(obs[3])
        accelerating = bool(float(action[1]) >= 0.5)

        # Detect lap completion: progress wraps from near 1 back to near 0.
        progress_delta = curr_progress - self._prev_progress
        finished = progress_delta < -0.5

        # Crash detection: too far from centre.
        crashed = abs(lateral_offset) > self._reward_config.crash_threshold_m

        # Time limit.
        time_over = self._elapsed_s > self._max_episode_time_s

        info = {
            "speed_ms": speed,
            "lateral_offset": lateral_offset,
            "track_progress": curr_progress,
            "prev_progress": self._prev_progress,
            "accelerating": accelerating,
            "finished": finished,
            "elapsed_s": self._elapsed_s,
            "termination_reason": None,
        }

        reward = self._reward_calc.compute(
            prev_state=None,
            curr_state=None,
            finished=finished,
            elapsed_s=self._elapsed_s,
            info=info,
        )

        terminated = finished or crashed or done
        truncated = time_over and not terminated

        if finished:
            info["termination_reason"] = "finish"
        elif crashed:
            info["termination_reason"] = "crash"
        elif done:
            info["termination_reason"] = "done"
        elif time_over:
            info["termination_reason"] = "timeout"

        self._prev_obs = obs
        self._prev_progress = curr_progress

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # BaseGameEnv API
    # ------------------------------------------------------------------

    def _build_obs(self, step: Any) -> np.ndarray:
        """Not used directly — obs comes from client.step()."""
        return np.zeros(BASE_OBS_DIM, dtype=np.float32)

    def get_episode_time_limit(self) -> float:
        return self._max_episode_time_s

    def set_episode_time_limit(self, seconds: float) -> None:
        self._max_episode_time_s = seconds


def make_env(
    experiment_dir: str | Path,
    max_episode_time_s: float = 120.0,
    vision: bool = False,
) -> TorcsEnv:
    """Factory that wires up a TorcsEnv from an experiment directory.

    Loads reward_config.yaml from *experiment_dir* if it exists.
    """
    experiment_dir = Path(experiment_dir)
    reward_cfg_path = experiment_dir / "reward_config.yaml"
    if reward_cfg_path.exists():
        reward_config = TorcsRewardConfig.from_yaml(str(reward_cfg_path))
    else:
        reward_config = TorcsRewardConfig()
    return TorcsEnv(
        reward_config=reward_config,
        max_episode_time_s=max_episode_time_s,
        vision=vision,
    )
