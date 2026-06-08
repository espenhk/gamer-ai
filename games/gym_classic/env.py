"""Gymnasium classic-control environment wrapper.

Supports CartPole-v1, MountainCar-v0, Acrobot-v1, Pendulum-v1, and
LunarLander-v2.  Select via the ``map_name`` training parameter.

Requires ``gymnasium[classic_control]`` (and ``gymnasium[box2d]`` for
LunarLander-v2)::

    pip install "gymnasium[classic_control]"
    pip install "gymnasium[box2d]"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as _exc:
    raise ImportError(
        "gym_classic requires gymnasium.  Install with:\n"
        "    pip install 'gymnasium[classic_control]'\n"
        "(LunarLander-v2 also needs: pip install 'gymnasium[box2d]')"
    ) from _exc

from framework.base_env import BaseGameEnv
from games.gym_classic.actions import PENDULUM_TORQUE_SCALE, get_n_actions, is_continuous
from games.gym_classic.obs_spec import get_obs_spec
from games.gym_classic.reward import GymClassicRewardCalculator, GymClassicRewardConfig

logger = logging.getLogger(__name__)

# Simulated steps-per-second used to convert episode_time_s to max_steps.
_STEPS_PER_SECOND: int = 50

_DEFAULT_MAP: str = "CartPole-v1"


class GymClassicEnv(BaseGameEnv):
    """Gymnasium wrapper around a classic-control environment.

    Parameters
    ----------
    map_name :
        Gymnasium env id (e.g. ``"CartPole-v1"``).
    reward_config :
        Reward shaping config.  If None uses Python defaults.
    max_episode_steps :
        Maximum steps per episode.
    render_mode :
        Gymnasium render mode (``"human"`` / ``"rgb_array"`` / None).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        map_name: str = _DEFAULT_MAP,
        reward_config: GymClassicRewardConfig | None = None,
        max_episode_steps: int = 500,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self._map_name = map_name
        self._reward_config = reward_config or GymClassicRewardConfig()
        self._reward_calc = GymClassicRewardCalculator(self._reward_config)
        self._max_episode_steps = max_episode_steps
        self._obs_spec = get_obs_spec(map_name)
        self._n_actions = get_n_actions(map_name)
        self._is_continuous_env = is_continuous(map_name)

        kwargs: dict[str, Any] = {}
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        if max_episode_steps is not None:
            kwargs["max_episode_steps"] = max_episode_steps

        self._env = gym.make(map_name, **kwargs)
        self._step_count: int = 0

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_spec.dim,),
            dtype=np.float32,
        )
        # Expose a 1D Box for the framework's policy contract.
        # Continuous policies output action[0] in [-1, 1]; the env maps this
        # to the gym action.  Tabular policies emit integer-valued floats.
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        raw_obs, info = self._env.reset(seed=seed, options=options)
        self._step_count = 0
        self._reward_calc.reset()
        return self._build_obs(raw_obs), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        gym_action = self._action_to_gym(action)
        raw_obs, native_reward, terminated, truncated, info = self._env.step(gym_action)
        self._step_count += 1
        info["native_reward"] = float(native_reward)
        reward = self._reward_calc.compute(
            prev_state=None,
            curr_state=None,
            finished=terminated,
            elapsed_s=self._step_count / _STEPS_PER_SECOND,
            info=info,
        )
        return self._build_obs(raw_obs), reward, terminated, truncated, info

    def close(self) -> None:
        self._env.close()

    def _build_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """Normalize raw gymnasium observation by obs_spec scales."""
        obs = np.asarray(raw_obs, dtype=np.float32)
        scales = self._obs_spec.scales
        # obs may have fewer dims than spec if env returns partial obs;
        # truncate or pad to match.
        n = min(len(obs), len(scales))
        result = np.zeros(len(scales), dtype=np.float32)
        result[:n] = obs[:n] / scales[:n]
        return result

    def _action_to_gym(self, action: Any) -> Any:
        """Map a framework action vector to the native gym action.

        Mirrors the Atari adapter's logic:
          - Integer-valued floats (tabular/DQN discrete_actions rows)
            are used directly as discrete action indices.
          - Continuous [-1, 1] values (evolutionary/gradient policies)
            are linearly mapped to [0, n_actions − 1] for discrete envs,
            or scaled to the torque range for Pendulum.
        """
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        val = float(arr[0]) if arr.size > 0 else 0.0

        if self._is_continuous_env:
            # Pendulum: scale [-1, 1] output to torque range.
            torque = np.clip(val, -1.0, 1.0) * PENDULUM_TORQUE_SCALE
            return np.array([torque], dtype=np.float32)

        # Discrete env: map float to integer action index.
        _EPS = 1e-4
        if abs(val - round(val)) < _EPS and val >= 0.0:
            # Integer-valued (from tabular DISCRETE_ACTIONS row)
            idx = int(round(val))
        elif -1.0 <= val <= 1.0 and self._n_actions > 1:
            # Continuous head → linear map to [0, n_actions − 1]
            idx = int(round((val + 1.0) * 0.5 * (self._n_actions - 1)))
        else:
            idx = int(round(val))

        return int(np.clip(idx, 0, self._n_actions - 1))

    def get_episode_time_limit(self) -> float:
        return float(self._max_episode_steps) / _STEPS_PER_SECOND

    def set_episode_time_limit(self, seconds: float) -> None:
        self._max_episode_steps = int(seconds * _STEPS_PER_SECOND)


def make_env(
    experiment_dir: str | Path,
    map_name: str = _DEFAULT_MAP,
    max_episode_time_s: float = 10.0,
    render_mode: str | None = None,
) -> GymClassicEnv:
    """Factory that wires up a GymClassicEnv from an experiment directory."""
    experiment_dir = Path(experiment_dir)
    reward_cfg_path = experiment_dir / "reward_config.yaml"
    if reward_cfg_path.exists():
        reward_config = GymClassicRewardConfig.from_yaml(str(reward_cfg_path))
    else:
        reward_config = GymClassicRewardConfig()
    max_steps = int(max_episode_time_s * _STEPS_PER_SECOND)
    return GymClassicEnv(
        map_name=map_name,
        reward_config=reward_config,
        max_episode_steps=max_steps,
        render_mode=render_mode,
    )
