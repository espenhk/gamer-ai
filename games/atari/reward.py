"""Atari reward configuration and calculator.

Atari games emit a native score delta each step (positive when the agent
gains points, negative when it loses them).  Two optional shaping knobs:

* ``clip_sign`` — clip the per-step reward to ``{-1, 0, 1}``.  This is the
  classic DQN-paper trick that makes very-large-score games (Asteroids,
  Q*bert) comparable to small-score ones (Pong, Boxing) and is on by
  default behind a flag.
* ``step_penalty`` — flat per-step cost; useful to discourage stalling
  policies in games where the score doesn't penalise idling.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import yaml


@dataclasses.dataclass
class AtariRewardConfig:
    """Reward hyperparameters for Atari training."""

    native_reward_scale: float = 1.0
    clip_sign: bool = False
    step_penalty: float = 0.0

    @classmethod
    def from_yaml(cls, path: str) -> "AtariRewardConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class AtariRewardCalculator:
    """Stateless reward shaper for Atari episodes."""

    def __init__(self, config: AtariRewardConfig) -> None:
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
        native = float(info.get("native_reward", 0.0))
        if self._cfg.clip_sign and native != 0.0:
            native = float(np.sign(native))
        reward = self._cfg.native_reward_scale * native
        reward += self._cfg.step_penalty
        return float(reward)
