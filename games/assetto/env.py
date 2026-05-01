"""Assetto Corsa Gymnasium environment.

Requires the optional ``assettocorsa`` package (Assetto Corsa Python bindings).
Install with::

    pip install assettocorsa

If ``assettocorsa`` is not installed, importing this module raises
``ImportError``.  The entry point in ``main.py`` converts that to a
``ValueError`` with a helpful message.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

# Optional dependency — fails loudly if not installed so the caller can
# convert the ImportError into a descriptive ValueError.
try:
    import assettocorsa  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        "assettocorsa is not installed.  Install the Assetto Corsa Python bridge "
        "with:\n    pip install assettocorsa\n"
        "Assetto Corsa (commercial) must also be installed separately."
    ) from _exc

import numpy as np
from gymnasium import spaces

from framework.base_env import BaseGameEnv
from games.assetto.obs_spec import BASE_OBS_DIM
from games.assetto.reward import AssettoCorsaRewardConfig, AssettoCorsaRewardCalculator

logger = logging.getLogger(__name__)


class AssettoCorsaEnv(BaseGameEnv):
    """Gymnasium environment wrapping Assetto Corsa.

    Parameters
    ----------
    reward_config :
        AssettoCorsaRewardConfig instance.  If None, uses Python defaults.
    max_episode_time_s :
        Wall-clock seconds before the episode is truncated.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_config: AssettoCorsaRewardConfig | None = None,
        max_episode_time_s: float = 120.0,
    ) -> None:
        super().__init__()

        self._reward_config = reward_config or AssettoCorsaRewardConfig()
        self._max_episode_time_s = max_episode_time_s
        self._reward_calc = AssettoCorsaRewardCalculator(self._reward_config)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(BASE_OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._client = assettocorsa.Client()

        self._prev_progress: float = 0.0
        self._elapsed_s: float = 0.0
        self._episode_start_s: float = 0.0
        self._step_count: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        raw_obs = self._client.reset()
        obs = self._parse_obs(raw_obs)

        self._prev_progress = 0.0
        self._elapsed_s = 0.0
        self._episode_start_s = time.monotonic()
        self._step_count = 0
        self._reward_calc.reset()

        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        raw_obs, _r, done, _info = self._client.step(action)
        obs = self._parse_obs(raw_obs)

        self._step_count += 1
        self._elapsed_s = time.monotonic() - self._episode_start_s

        speed = float(obs[0])
        lateral_offset = float(obs[1])
        curr_progress = float(obs[3])
        accelerating = bool(float(action[1]) >= 0.5)

        finished = done and curr_progress >= 0.99
        crashed = abs(lateral_offset) > self._reward_config.crash_threshold_m
        time_over = self._elapsed_s > self._max_episode_time_s

        info: dict[str, Any] = {
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

        terminated = finished or crashed or (done and not time_over)
        truncated = time_over and not terminated

        if finished:
            info["termination_reason"] = "finish"
        elif crashed:
            info["termination_reason"] = "crash"
        elif done:
            info["termination_reason"] = "done"
        elif time_over:
            info["termination_reason"] = "timeout"

        self._prev_progress = curr_progress

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._client.close()

    def _build_obs(self, step: Any) -> np.ndarray:
        return np.zeros(BASE_OBS_DIM, dtype=np.float32)

    def _parse_obs(self, raw_obs: Any) -> np.ndarray:
        arr = np.zeros(BASE_OBS_DIM, dtype=np.float32)
        if raw_obs is not None:
            raw = np.asarray(raw_obs, dtype=np.float32).ravel()
            n = min(len(raw), BASE_OBS_DIM)
            arr[:n] = raw[:n]
        return arr

    def get_episode_time_limit(self) -> float:
        return self._max_episode_time_s

    def set_episode_time_limit(self, seconds: float) -> None:
        self._max_episode_time_s = seconds


def make_env(
    experiment_dir: str | Path,
    max_episode_time_s: float = 120.0,
) -> AssettoCorsaEnv:
    """Factory that wires up an AssettoCorsaEnv from an experiment directory."""
    experiment_dir = Path(experiment_dir)
    reward_cfg_path = experiment_dir / "reward_config.yaml"
    if reward_cfg_path.exists():
        reward_config = AssettoCorsaRewardConfig.from_yaml(str(reward_cfg_path))
    else:
        reward_config = AssettoCorsaRewardConfig()
    return AssettoCorsaEnv(
        reward_config=reward_config,
        max_episode_time_s=max_episode_time_s,
    )
