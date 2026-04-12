"""Abstract base class for reward calculators.

Game integrations subclass RewardCalculatorBase and implement compute().
The framework training loop calls compute() after every env.step() without
knowing anything about the game-specific reward signals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RewardCalculatorBase(ABC):
    """Abstract base for game-specific reward calculators.

    The framework layer only ever holds a reference of type
    RewardCalculatorBase and calls compute() after each step.

    Concrete implementations live in games/<name>/reward.py.
    """

    @abstractmethod
    def compute(
        self,
        prev_state: Any,
        curr_state: Any,
        finished: bool,
        elapsed_s: float,
        info: dict,
        n_ticks: int = 1,
    ) -> float:
        """Return the scalar reward for this RL step.

        Parameters
        ----------
        prev_state:
            Game-specific state snapshot from the *previous* step.
        curr_state:
            Game-specific state snapshot from the *current* step.
        finished:
            True if the game signalled episode completion (e.g. crossed
            the finish line or reached the goal).
        elapsed_s:
            Wall-clock seconds elapsed in the current episode.
        info:
            The info dict produced by env._get_game_info() for this step.
            Game-specific reward signals can read from here instead of
            taking bespoke positional parameters.
        n_ticks:
            Number of game ticks covered by this RL step (≥ 1).
            Tick-rate-independent rewards should be scaled by n_ticks.
        """

    def reset(self) -> None:
        """Called at the start of each episode.  Override if stateful."""
