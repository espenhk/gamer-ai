"""iRacing Gymnasium environment.

Wraps the iRacing telemetry API via ``pyirsdk`` (Python iRacing SDK).
Install with::

    pip install pyirsdk

Phase 1 is telemetry-only (read-only): the environment reads live car
state but does **not** inject actions.  Action injection (Phase 2) will
use vJoy or similar virtual controller.

If ``pyirsdk`` is not installed, importing this module raises
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
    import irsdk  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        "pyirsdk is not installed.  Install the iRacing Python SDK with:\n"
        "    pip install pyirsdk\n"
        "iRacing (Windows) must also be running with telemetry enabled."
    ) from _exc

import numpy as np
from gymnasium import spaces

from framework.base_env import BaseGameEnv
from games.iracing.obs_spec import BASE_OBS_DIM
from games.iracing.reward import IRacingRewardConfig, IRacingRewardCalculator

logger = logging.getLogger(__name__)


class IRacingEnv(BaseGameEnv):
    """Gymnasium environment wrapping iRacing telemetry via pyirsdk.

    Parameters
    ----------
    reward_config :
        IRacingRewardConfig instance.  If None, uses Python defaults.
    max_episode_time_s :
        Wall-clock seconds before the episode is truncated.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_config: IRacingRewardConfig | None = None,
        max_episode_time_s: float = 120.0,
    ) -> None:
        super().__init__()

        self._reward_config = reward_config or IRacingRewardConfig()
        self._max_episode_time_s = max_episode_time_s
        self._reward_calc = IRacingRewardCalculator(self._reward_config)

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

        self._ir = irsdk.IRSDK()
        self._connected = False
        self._prev_progress: float = 0.0
        self._elapsed_s: float = 0.0
        self._episode_start_s: float = 0.0
        self._step_count: int = 0

    def _ensure_connected(self) -> None:
        if not self._connected:
            if self._ir.startup():
                self._connected = True
                logger.info("Connected to iRacing telemetry")
            else:
                raise RuntimeError(
                    "Cannot connect to iRacing.  Is the simulator running?"
                )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._ensure_connected()

        self._prev_progress = 0.0
        self._elapsed_s = 0.0
        self._episode_start_s = time.monotonic()
        self._step_count = 0
        self._reward_calc.reset()

        obs = self._read_telemetry()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Phase 1: telemetry-only — action is ignored (no injection yet).
        self._step_count += 1
        self._elapsed_s = time.monotonic() - self._episode_start_s

        obs = self._read_telemetry()
        speed = float(obs[0])
        lateral_offset = float(obs[1])
        curr_progress = float(obs[2])
        is_off_track = bool(self._ir["CarIdxOnPitRoad"] is not None and False)

        finished = curr_progress >= 1.0
        crashed = abs(lateral_offset) > self._reward_config.crash_threshold_m
        time_over = self._elapsed_s > self._max_episode_time_s

        info: dict[str, Any] = {
            "speed_ms": speed,
            "lateral_offset": lateral_offset,
            "track_progress": curr_progress,
            "prev_progress": self._prev_progress,
            "is_off_track": is_off_track,
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

        terminated = finished or crashed
        truncated = time_over and not terminated

        if finished:
            info["termination_reason"] = "finish"
        elif crashed:
            info["termination_reason"] = "crash"
        elif time_over:
            info["termination_reason"] = "timeout"

        self._prev_progress = curr_progress

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._connected:
            self._ir.shutdown()
            self._connected = False

    def _build_obs(self, step: Any) -> np.ndarray:
        return self._read_telemetry()

    def _read_telemetry(self) -> np.ndarray:
        """Read current telemetry frame from iRacing into obs vector."""
        obs = np.zeros(BASE_OBS_DIM, dtype=np.float32)

        def _get(key: str, default: float = 0.0) -> float:
            val = self._ir[key]
            return float(val) if val is not None else default

        obs[0] = _get("Speed")
        obs[1] = _get("CarIdxLapDistPct")  # approximate lateral offset
        obs[2] = _get("LapDistPct")
        obs[3] = _get("YawRate")
        obs[4] = _get("RPM")
        obs[5] = _get("Gear")
        obs[6] = _get("FuelLevelPct")
        obs[7] = _get("Throttle")
        obs[8] = _get("Brake")
        obs[9] = _get("SteeringWheelAngle")
        # Tire loads (LF, RF, LR, RR)
        obs[10] = _get("LFtempCL")
        obs[11] = _get("RFtempCL")
        obs[12] = _get("LRtempCL")
        obs[13] = _get("RRtempCL")
        # Tire temps
        obs[14] = _get("LFtempCL")
        obs[15] = _get("RFtempCL")
        obs[16] = _get("LRtempCL")
        obs[17] = _get("RRtempCL")
        obs[18] = _get("dcBrakeBias")
        obs[19] = _get("LapCurrentLapTime")
        obs[20] = _get("LapBestLapTime")

        return obs

    def get_episode_time_limit(self) -> float:
        return self._max_episode_time_s

    def set_episode_time_limit(self, seconds: float) -> None:
        self._max_episode_time_s = seconds


def make_env(
    experiment_dir: str | Path,
    max_episode_time_s: float = 120.0,
) -> IRacingEnv:
    """Factory that wires up an IRacingEnv from an experiment directory."""
    experiment_dir = Path(experiment_dir)
    reward_cfg_path = experiment_dir / "reward_config.yaml"
    if reward_cfg_path.exists():
        reward_config = IRacingRewardConfig.from_yaml(str(reward_cfg_path))
    else:
        reward_config = IRacingRewardConfig()
    return IRacingEnv(
        reward_config=reward_config,
        max_episode_time_s=max_episode_time_s,
    )
