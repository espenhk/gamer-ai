"""CarRacing Gymnasium environment.

Wraps the ``CarRacing-v2`` environment from gymnasium.  Requires the
``box2d-py`` and ``pygame`` extras::

    pip install gymnasium[box2d]

If those packages are not installed, importing this module raises
``ImportError``.  The entry point in ``main.py`` converts that to a
``ValueError`` with a helpful message.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

# Optional dependency — raises ImportError if box2d extras not installed.
try:
    import gymnasium as gym
    from gymnasium.envs.box2d import CarRacing  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        "CarRacing-v2 requires optional gymnasium box2d extras.  Install with:\n"
        "    pip install gymnasium[box2d]\n"
        "This also requires pygame and box2d-py."
    ) from _exc

from gymnasium import spaces

from framework.base_env import BaseGameEnv
from games.car_racing.obs_spec import BASE_OBS_DIM
from games.car_racing.reward import CarRacingRewardConfig, CarRacingRewardCalculator

logger = logging.getLogger(__name__)


class CarRacingEnv(BaseGameEnv):
    """Gymnasium wrapper around CarRacing-v2.

    Converts the pixel observation into a compact feature vector compatible
    with the WeightedLinearPolicy framework.

    Parameters
    ----------
    reward_config :
        CarRacingRewardConfig instance.  If None, uses Python defaults.
    max_episode_steps :
        Maximum steps per episode (gymnasium default: 1000).
    continuous :
        If True, use continuous action space (default).  If False, use
        discrete actions.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        reward_config: CarRacingRewardConfig | None = None,
        max_episode_steps: int = 1000,
        continuous: bool = True,
    ) -> None:
        super().__init__()

        self._reward_config = reward_config or CarRacingRewardConfig()
        self._reward_calc = CarRacingRewardCalculator(self._reward_config)
        self._max_episode_steps = max_episode_steps

        self._env = gym.make(
            "CarRacing-v2",
            continuous=continuous,
            max_episode_steps=max_episode_steps,
        )

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

        self._step_count: int = 0
        self._prev_info: dict = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        _raw_obs, info = self._env.reset(seed=seed, options=options)
        self._step_count = 0
        self._prev_info = info
        self._reward_calc.reset()

        return np.zeros(BASE_OBS_DIM, dtype=np.float32), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # CarRacing-v2 uses [steer, gas, brake] in [-1,1] × [0,1] × [0,1].
        car_action = np.array([
            float(action[0]),           # steer
            float(action[1]),           # gas
            float(action[2]),           # brake
        ], dtype=np.float32)

        _raw_obs, native_reward, terminated, truncated, info = self._env.step(car_action)

        self._step_count += 1

        info["native_reward"] = float(native_reward)
        info.setdefault("termination_reason", None)

        if terminated:
            info["termination_reason"] = "finish"
        elif truncated:
            info["termination_reason"] = "timeout"

        reward = self._reward_calc.compute(
            prev_state=None,
            curr_state=None,
            finished=terminated,
            elapsed_s=self._step_count / 50.0,  # approx 50 steps/s
            info=info,
        )

        obs = np.zeros(BASE_OBS_DIM, dtype=np.float32)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._env.close()

    def _build_obs(self, step: Any) -> np.ndarray:
        return np.zeros(BASE_OBS_DIM, dtype=np.float32)

    def get_episode_time_limit(self) -> float:
        return float(self._max_episode_steps) / 50.0

    def set_episode_time_limit(self, seconds: float) -> None:
        self._max_episode_steps = int(seconds * 50)


def make_env(
    experiment_dir: str | Path,
    max_episode_time_s: float = 20.0,
) -> CarRacingEnv:
    """Factory that wires up a CarRacingEnv from an experiment directory."""
    experiment_dir = Path(experiment_dir)
    reward_cfg_path = experiment_dir / "reward_config.yaml"
    if reward_cfg_path.exists():
        reward_config = CarRacingRewardConfig.from_yaml(str(reward_cfg_path))
    else:
        reward_config = CarRacingRewardConfig()
    max_steps = int(max_episode_time_s * 50)
    return CarRacingEnv(
        reward_config=reward_config,
        max_episode_steps=max_steps,
    )
