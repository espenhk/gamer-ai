"""Abstract base class for game environment integrations.

Game integrations subclass BaseGameEnv and implement the three abstract
methods.  The framework layer only ever holds a reference of type BaseGameEnv
and never imports from games/.

Relationship to Gymnasium
--------------------------
BaseGameEnv inherits gymnasium.Env so game envs are drop-in compatible with
standard RL tooling (SB3, Gymnasium wrappers, etc.).  Concrete subclasses
still define observation_space and action_space in __init__ as usual.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np


class BaseGameEnv(gym.Env, ABC):
    """Abstract base for game-specific Gymnasium environments.

    The framework training loop interacts exclusively via this interface.
    Game integrations live in games/<name>/ and inherit this class.

    Concrete subclasses must implement:

    _build_obs(step)
        Build the float32 observation array from the current game state.
        The shape must match self.observation_space.

    _get_game_info()
        Return a dict of game-specific metrics for the current step.
        These are merged into the Gymnasium step *info* dict and are
        available to game-specific analytics without any framework coupling.

        Recommended keys for TMNF-like racing games:
            "pos_x", "pos_z" — for bird's-eye trajectory plots
            "track_progress" — fraction of track completed [0, 1]
            "laps_completed" — cumulative lap count
            "lateral_offset" — metres from centreline

    The base class provides no-op default implementations that return
    empty values so subclasses can implement incrementally.
    """

    # ------------------------------------------------------------------
    # Abstract interface (game integrations implement these)
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_obs(self, step: Any) -> np.ndarray:
        """Build and return the observation vector from the current game state."""

    def _get_game_info(self) -> dict:
        """Return game-specific metrics to include in the step info dict.

        Default implementation returns an empty dict.  Override in game
        integrations to expose metrics (positions, progress, etc.) that
        analytics and logging can consume without the framework knowing
        about game-specific concepts.
        """
        return {}
