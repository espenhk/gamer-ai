"""MineRL reward configuration and calculator."""

from __future__ import annotations

import dataclasses
from typing import Any

import yaml


@dataclasses.dataclass
class MineRLRewardConfig:
    """Reward hyperparameters for MineRL training.

    MineRL environments provide a native dense or sparse reward signal.
    This config scales the native signal and adds optional shaping terms.
    """

    native_reward_scale: float = 1.0
    step_penalty: float = -0.001
    finish_bonus: float = 100.0

    @classmethod
    def from_yaml(cls, path: str) -> "MineRLRewardConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MineRLRewardCalculator:
    """Stateful reward calculator for MineRL episodes."""

    def __init__(self, config: MineRLRewardConfig) -> None:
        self._cfg = config
        self._total_native_reward: float = 0.0

    def reset(self) -> None:
        self._total_native_reward = 0.0

    def compute(
        self,
        prev_state: Any,
        curr_state: Any,
        finished: bool,
        elapsed_s: float,
        info: dict,
    ) -> float:
        native = float(info.get("native_reward", 0.0))
        self._total_native_reward += native

        reward = self._cfg.native_reward_scale * native
        reward += self._cfg.step_penalty
        if finished:
            reward += self._cfg.finish_bonus
        return float(reward)
