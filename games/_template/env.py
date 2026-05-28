"""<GAME_TITLE> Gymnasium environment.

Wraps the game SDK and exposes a Gymnasium ``Env`` interface.  Keep imports
of the game's SDK **inside** functions / methods (not at module top) so
that CI can import this module on platforms where the SDK is unavailable.

Copy this file into ``games/<name>/env.py`` and fill in the blanks.  Read
``docs/framework/base_env.md`` for the full protocol.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from framework.base_env import BaseGameEnv

logger = logging.getLogger(__name__)

# Import your obs_spec here — update the module path after copying.
# from games.<name>.obs_spec import BASE_OBS_DIM, <NAME>_OBS_SPEC


class _TemplateEnv(BaseGameEnv):
    """Gymnasium wrapper around <GAME_TITLE>.

    Replace this class with your own, e.g. ``MyGameEnv``.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_time_s: float = 60.0) -> None:
        super().__init__()

        self._max_episode_time_s = max_episode_time_s

        # TODO: set observation_space shape to match your obs_spec
        obs_dim = 1  # replace with BASE_OBS_DIM
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # TODO: define your action space
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        raise NotImplementedError("Connect to the game, reset the episode, return (obs, info)")

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        raise NotImplementedError("Advance one step, return (obs, reward, terminated, truncated, info)")

    def close(self) -> None:
        """Release any resources held by the environment."""
        pass

    def _build_obs(self, step: Any) -> np.ndarray:
        raise NotImplementedError("Convert raw game state into a flat numpy obs vector")

    def get_episode_time_limit(self) -> float:
        return self._max_episode_time_s

    def set_episode_time_limit(self, seconds: float) -> None:
        self._max_episode_time_s = seconds


def make_env(
    experiment_dir: str | Path,
    max_episode_time_s: float = 60.0,
) -> _TemplateEnv:
    """Factory that wires up a game env from an experiment directory.

    Called by ``build_game_spec()`` in the adapter.
    """
    raise NotImplementedError("Instantiate and return your env here")
