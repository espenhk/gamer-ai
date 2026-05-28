"""<GAME_TITLE> reward configuration and calculator.

Implement ``RewardCalculator.compute()`` to return a shaped reward signal
for each environment step.  Default knobs go in
``config/reward_config.yaml``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import yaml


@dataclasses.dataclass
class RewardConfig:
    """Reward hyperparameters — add fields matching ``reward_config.yaml``."""

    # Example field — replace with your game's reward knobs.
    step_penalty: float = -0.1

    @classmethod
    def from_yaml(cls, path: str) -> "RewardConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RewardCalculator:
    """Stateful reward calculator for episodes."""

    def __init__(self, config: RewardConfig) -> None:
        self._cfg = config

    def reset(self) -> None:
        """Reset any per-episode state."""
        pass

    def compute(
        self,
        prev_state: Any,
        curr_state: Any,
        finished: bool,
        elapsed_s: float,
        info: dict,
    ) -> float:
        """Return the shaped reward for the current transition.

        Parameters
        ----------
        prev_state : game-specific state from the previous step (or None).
        curr_state : game-specific state from the current step (or None).
        finished : whether the episode just ended normally.
        elapsed_s : wall-clock seconds elapsed in this episode.
        info : the ``info`` dict returned by the environment step.
        """
        raise NotImplementedError("Compute and return a float reward")
