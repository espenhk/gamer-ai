"""CarRacing reward configuration and calculator."""
from __future__ import annotations

import dataclasses
from typing import Any

import yaml


@dataclasses.dataclass
class CarRacingRewardConfig:
    """Reward hyperparameters for CarRacing training.

    CarRacing-v2 provides its own dense reward signal (tiles visited per step).
    This config adds optional shaping on top of the native reward.
    """

    native_reward_scale: float = 1.0
    step_penalty: float = -0.1
    finish_bonus: float = 100.0
    crash_threshold_m: float = 25.0

    @classmethod
    def from_yaml(cls, path: str) -> "CarRacingRewardConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class CarRacingRewardCalculator:
    """Stateful reward calculator for CarRacing episodes."""

    def __init__(self, config: CarRacingRewardConfig) -> None:
        self._cfg = config

    def reset(self) -> None:
        pass

    def compute(
        self,
        prev_state: Any,
        curr_state: Any,
        finished: bool,
        elapsed_s: float,
        info: dict,
    ) -> float:
        reward = self._cfg.native_reward_scale * info.get("native_reward", 0.0)
        reward += self._cfg.step_penalty
        if finished:
            reward += self._cfg.finish_bonus
        return float(reward)
