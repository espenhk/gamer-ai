"""BeamNG reward configuration and calculator.

Optional dependency: beamng_gym — only required when actually connecting to
the BeamNG.drive simulator.  This module itself has no external dependencies.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import yaml


@dataclasses.dataclass
class BeamNGRewardConfig:
    """Reward hyperparameters for BeamNG training."""

    progress_weight: float = 10000.0
    centerline_weight: float = -0.1
    centerline_exp: float = 2.0
    speed_weight: float = 0.05
    step_penalty: float = -0.05
    finish_bonus: float = 5000.0
    finish_time_weight: float = -5.0
    par_time_s: float = 120.0
    accel_bonus: float = 0.5
    crash_threshold_m: float = 25.0

    @classmethod
    def from_yaml(cls, path: str) -> "BeamNGRewardConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class BeamNGRewardCalculator:
    """Stateful reward calculator for BeamNG episodes."""

    def __init__(self, config: BeamNGRewardConfig) -> None:
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

        if info.get("accelerating", False):
            reward += cfg.accel_bonus

        if finished:
            reward += cfg.finish_bonus
            time_diff = elapsed_s - cfg.par_time_s
            reward += cfg.finish_time_weight * time_diff

        self._prev_progress = curr_progress
        return float(reward)
