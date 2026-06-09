"""MineRL environment wrapper (Phase 1: vector observations).

Wraps a MineRL Gymnasium-compatible environment and converts the obs dict
into a compact feature vector compatible with the framework's policy zoo.

Requires the ``minerl`` package and Java 8+::

    pip install minerl

See ``games/minerl/README.md`` for full setup instructions.

If ``minerl`` is not installed, importing this module raises ``ImportError``.
The entry point in ``main.py`` converts that to a ``ValueError`` with a
helpful message.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

# Optional dependency — raises ImportError if minerl is not installed.
try:
    import minerl  # noqa: F401
    import gymnasium as gym
except ImportError as _exc:
    raise ImportError(
        "MineRL requires the 'minerl' package and Java 8+.  Install with:\n"
        "    pip install minerl\n"
        "See games/minerl/README.md for full setup instructions."
    ) from _exc

from gymnasium import spaces

from framework.base_env import BaseGameEnv
from games.minerl.actions import N_ACTIONS, _ACTION_OVERRIDES
from games.minerl.obs_spec import BASE_OBS_DIM
from games.minerl.reward import MineRLRewardCalculator, MineRLRewardConfig

logger = logging.getLogger(__name__)

#: Default frames per second for MineRL envs (used for time limit conversion).
_MINERL_FPS: float = 20.0


class MineRLEnv(BaseGameEnv):
    """Gymnasium wrapper around a MineRL environment.

    Converts the MineRL obs dict into a compact feature vector (compass angle
    + inventory counts) compatible with the ``WeightedLinearPolicy`` framework.

    Parameters
    ----------
    map_name :
        MineRL environment ID (e.g. ``"MineRLNavigateDense-v0"``).
    reward_config :
        ``MineRLRewardConfig`` instance.  If None, uses Python defaults.
    max_episode_steps :
        Maximum steps per episode.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        map_name: str = "MineRLNavigateDense-v0",
        reward_config: MineRLRewardConfig | None = None,
        max_episode_steps: int = 2400,
    ) -> None:
        super().__init__()

        self._map_name = map_name
        self._reward_config = reward_config or MineRLRewardConfig()
        self._reward_calc = MineRLRewardCalculator(self._reward_config)
        self._max_episode_steps = max_episode_steps

        self._env = gym.make(map_name, max_episode_steps=max_episode_steps)
        self._noop_action: dict = self._build_noop()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(BASE_OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._step_count: int = 0

    # ------------------------------------------------------------------
    # BaseGameEnv contract
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        obs_dict, info = self._env.reset(seed=seed, options=options)
        self._step_count = 0
        self._reward_calc.reset()
        return self._flatten_obs(obs_dict), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_idx = int(round(float(action[0]) if hasattr(action, "__len__") else float(action)))
        action_idx = max(0, min(action_idx, N_ACTIONS - 1))
        minerl_action = self._decode_action(action_idx)

        result = self._env.step(minerl_action)
        # Handle both gymnasium 5-tuple and legacy MineRL 4-tuple returns.
        if len(result) == 5:
            obs_dict, native_reward, terminated, truncated, info = result
        else:
            obs_dict, native_reward, done, info = result
            terminated, truncated = done, False

        self._step_count += 1
        if self._step_count >= self._max_episode_steps:
            truncated = True

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
            elapsed_s=self._step_count / _MINERL_FPS,
            info=info,
        )

        return self._flatten_obs(obs_dict), reward, terminated, truncated, info

    def close(self) -> None:
        self._env.close()

    def _build_obs(self, step: Any) -> np.ndarray:
        return np.zeros(BASE_OBS_DIM, dtype=np.float32)

    def get_episode_time_limit(self) -> float:
        return float(self._max_episode_steps) / _MINERL_FPS

    def set_episode_time_limit(self, seconds: float) -> None:
        self._max_episode_steps = max(1, int(seconds * _MINERL_FPS))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_noop(self) -> dict:
        """Return a no-op action dict from the env's action space."""
        if hasattr(self._env.action_space, "noop"):
            return self._env.action_space.noop()
        return {
            "forward": 0,
            "back": 0,
            "left": 0,
            "right": 0,
            "jump": 0,
            "sprint": 0,
            "sneak": 0,
            "attack": 0,
            "camera": [0.0, 0.0],
            "place": "none",
            "craft": "none",
            "equip": "none",
            "nearbyCraft": "none",
            "nearbySmelt": "none",
        }

    def _decode_action(self, action_idx: int) -> dict:
        """Convert a discrete action index to a MineRL action dict."""
        act = dict(self._noop_action)
        act.update(_ACTION_OVERRIDES[action_idx])
        # Only pass keys present in the env's action space to avoid KeyErrors.
        if hasattr(self._env.action_space, "spaces"):
            act = {k: v for k, v in act.items() if k in self._env.action_space.spaces}
        return act

    def _flatten_obs(self, obs_dict: dict) -> np.ndarray:
        """Flatten MineRL obs dict into a float32 array matching MINERL_OBS_SPEC."""
        compass = float(obs_dict.get("compassAngle", 0.0))
        inv = obs_dict.get("inventory", {})
        inv_dirt = float(inv.get("dirt", 0))
        inv_log = float(inv.get("log", 0))
        return np.array(
            [compass / 180.0, inv_dirt / 64.0, inv_log / 64.0],
            dtype=np.float32,
        )


def make_env(
    experiment_dir: str | Path,
    map_name: str = "MineRLNavigateDense-v0",
    max_episode_time_s: float = 120.0,
) -> MineRLEnv:
    """Factory that wires up a MineRLEnv from an experiment directory."""
    experiment_dir = Path(experiment_dir)
    reward_cfg_path = experiment_dir / "reward_config.yaml"
    if reward_cfg_path.exists():
        reward_config = MineRLRewardConfig.from_yaml(str(reward_cfg_path))
    else:
        reward_config = MineRLRewardConfig()
    max_steps = max(1, int(max_episode_time_s * _MINERL_FPS))
    return MineRLEnv(
        map_name=map_name,
        reward_config=reward_config,
        max_episode_steps=max_steps,
    )
