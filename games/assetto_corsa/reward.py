"""Assetto Corsa reward calculator.

Subclasses the framework's RewardCalculatorBase. Reward signals are read
from a state dict (the ACStepState produced by the gym wrapper) so this
module has no dependency on TMNF-specific data classes.

The signal mix is intentionally simple — extra terms (engine RPM bonus,
slip penalty, etc.) can be added later via the optional kwargs in
RewardConfig without changing the public interface.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import yaml

from framework.base_reward import RewardCalculatorBase


@dataclass
class RewardConfig:
    """Reward weights for the AC environment.

    Defaults are progress-dominant, mirroring the TMNF defaults so policies
    trained on one game transfer reasonably to the other.
    """

    progress_weight:    float = 1000.0
    centerline_weight:  float = -0.5
    centerline_exp:     float = 2.0
    speed_weight:       float = 0.05
    step_penalty:       float = -0.05
    finish_bonus:       float = 500.0
    finish_time_weight: float = -1.0
    par_time_s:         float = 150.0
    accel_bonus:        float = 0.5
    crash_threshold_m:  float = 25.0

    @classmethod
    def from_yaml(cls, path: str) -> "RewardConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - valid
        if unknown:
            raise ValueError(
                f"{path}: unknown reward config keys: {sorted(unknown)}\n"
                f"Valid keys: {sorted(valid)}"
            )
        return cls(**data)


class RewardCalculator(RewardCalculatorBase):
    """Linear-weighted reward computed from telemetry fields.

    Stateless; the framework calls compute() on every RL step.
    """

    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def compute(
        self,
        prev_state: Any,
        curr_state: Any,
        finished: bool,
        elapsed_s: float,
        info: dict,
        n_ticks: int = 1,
    ) -> float:
        cfg = self.config
        reward = 0.0

        prev_progress = self._get(prev_state, "track_progress")
        curr_progress = self._get(curr_state, "track_progress")
        if prev_progress is not None and curr_progress is not None:
            reward += (curr_progress - prev_progress) * cfg.progress_weight

        lateral = self._get(curr_state, "lateral_offset")
        if lateral is not None:
            reward += (
                cfg.centerline_weight * abs(lateral) ** cfg.centerline_exp * n_ticks
            )

        speed = self._get(curr_state, "speed_ms")
        if speed is not None:
            reward += cfg.speed_weight * speed * n_ticks

        if bool(info.get("accelerating", False)):
            reward += cfg.accel_bonus * n_ticks

        reward += cfg.step_penalty * n_ticks

        if finished:
            reward += cfg.finish_bonus
            reward += cfg.finish_time_weight * (elapsed_s - cfg.par_time_s)

        return reward

    @staticmethod
    def _get(state: Any, key: str) -> Any:
        """Read *key* from a dict-like or attr-like state object."""
        if state is None:
            return None
        if isinstance(state, dict):
            return state.get(key)
        return getattr(state, key, None)
