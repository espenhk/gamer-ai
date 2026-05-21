"""iRacing reward configuration and calculator.

Phase 1 (telemetry-only) reward: lap-time improvement, centerline adherence,
and off-track penalty.  The iRacing safety-rating system provides an
off-track signal via the ``is_off_track`` telemetry field.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import yaml


@dataclasses.dataclass
class IRacingRewardConfig:
    """Reward hyperparameters for iRacing training."""

    progress_weight: float = 10000.0
    centerline_weight: float = -0.1
    centerline_exp: float = 2.0
    speed_weight: float = 0.05
    step_penalty: float = -0.05
    finish_bonus: float = 5000.0
    off_track_penalty: float = -10.0
    lap_time_improvement_bonus: float = 100.0
    crash_threshold_m: float = 25.0

    @classmethod
    def from_yaml(cls, path: str) -> "IRacingRewardConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class IRacingRewardCalculator:
    """Stateful reward calculator for iRacing episodes."""

    def __init__(self, config: IRacingRewardConfig) -> None:
        self._cfg = config
        self._prev_progress: float = 0.0

    def reset(self) -> None:
        self._prev_progress = 0.0

    def compute(
        self,
        prev_state: Any,
        curr_state: Any,
        finished: bool,
        elapsed_s: float,
        info: dict,
    ) -> float:
        cfg = self._cfg
        reward = 0.0

        curr_progress = info.get("track_progress", 0.0)
        progress_delta = curr_progress - self._prev_progress
        reward += cfg.progress_weight * max(progress_delta, 0.0)

        lateral_offset = info.get("lateral_offset", 0.0)
        reward += cfg.centerline_weight * abs(lateral_offset) ** cfg.centerline_exp

        speed = info.get("speed_ms", 0.0)
        reward += cfg.speed_weight * speed

        reward += cfg.step_penalty

        if info.get("is_off_track", False):
            reward += cfg.off_track_penalty

        if finished:
            reward += cfg.finish_bonus

        self._prev_progress = curr_progress
        return float(reward)
