"""Gymnasium classic-control reward configuration and calculator."""

from __future__ import annotations

import dataclasses
from typing import Any

import yaml


@dataclasses.dataclass
class GymClassicRewardConfig:
    """Reward hyperparameters for gym_classic training.

    The Gymnasium classic-control envs provide good dense reward signals
    natively; this config adds optional shaping on top.
    """

    native_reward_scale: float = 1.0
    step_penalty: float = 0.0

    @classmethod
    def from_yaml(cls, path: str) -> "GymClassicRewardConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class GymClassicRewardCalculator:
    """Stateful reward calculator for gym_classic episodes."""

    def __init__(self, config: GymClassicRewardConfig) -> None:
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
        return float(reward)
