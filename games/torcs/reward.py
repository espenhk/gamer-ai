"""TORCS-specific reward calculator.

Uses the same structure as the TMNF reward calculator but adapted for the
TORCS sensor set.  The reward is computed from the flat observation vector
and episode metadata rather than from game-engine state objects.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import yaml

from framework.base_reward import RewardCalculatorBase


@dataclass
class TorcsRewardConfig:
    """Reward weights for the TORCS environment.

    Python defaults are kept for lightweight unit tests.
    Training should load YAML via ``TorcsRewardConfig.from_yaml()``.

    Parameters
    ----------
    progress_weight :
        Reward proportional to distance-raced delta.
    centerline_weight :
        Penalty for lateral deviation from track centre (negative).
    centerline_exp :
        Exponent for the centerline penalty (2 = quadratic).
    speed_weight :
        Small reward per m/s.
    step_penalty :
        Tiny negative reward every tick.
    finish_bonus :
        One-time bonus at lap completion.
    finish_time_weight :
        Bonus/penalty relative to ``par_time_s``.
    par_time_s :
        Reference lap time (seconds).
    accel_bonus :
        Flat reward per step when throttle is pressed.
    crash_threshold_m :
        Episode ends when ``|lateral_offset| > threshold``.
    """

    progress_weight:    float = 10.0
    centerline_weight:  float = -0.5
    centerline_exp:     float = 2.0
    speed_weight:       float = 0.05
    step_penalty:       float = -0.01
    finish_bonus:       float = 100.0
    finish_time_weight: float = -0.1
    par_time_s:         float = 120.0
    accel_bonus:        float = 0.5
    crash_threshold_m:  float = 8.0

    @classmethod
    def from_yaml(cls, path: str) -> TorcsRewardConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid_keys = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - valid_keys
        if unknown:
            raise ValueError(
                f"{path}: unknown reward config keys: {sorted(unknown)}\n"
                f"Valid keys: {sorted(valid_keys)}"
            )
        return cls(**data)


class TorcsRewardCalculator(RewardCalculatorBase):
    """Reward computation for the TORCS environment.

    Stateless — all information is passed in via ``compute()``.
    """

    def __init__(self, config: TorcsRewardConfig) -> None:
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
        """Compute reward for one RL step.

        Here ``prev_state`` and ``curr_state`` are dicts with keys matching
        the info dict produced by :class:`games.torcs.env.TorcsEnv`.

        Expected keys in *info* (and in states):
            ``speed_ms``          — current speed (m/s)
            ``lateral_offset``    — metres from track centre
            ``track_progress``    — fraction of lap completed [0, 1]
            ``prev_progress``     — previous step's progress
            ``accelerating``      — bool, whether throttle was pressed
        """
        cfg = self.config
        reward = 0.0

        # Progress: reward for advancing along the track.
        prev_progress = info.get("prev_progress", 0.0)
        curr_progress = info.get("track_progress", 0.0)
        delta = curr_progress - prev_progress
        # Handle lap wrap-around (progress going from ~1 back to ~0).
        if delta < -0.5:
            delta += 1.0
        reward += delta * cfg.progress_weight

        # Centerline: quadratic penalty for lateral deviation.
        lateral = info.get("lateral_offset", 0.0)
        reward += cfg.centerline_weight * abs(lateral) ** cfg.centerline_exp * n_ticks

        # Speed: small reward for going fast.
        speed = info.get("speed_ms", 0.0)
        reward += cfg.speed_weight * speed * n_ticks

        # Acceleration bonus.
        if info.get("accelerating", False):
            reward += cfg.accel_bonus * n_ticks

        # Time cost.
        reward += cfg.step_penalty * n_ticks

        # Finish bonus.
        if finished:
            reward += cfg.finish_bonus
            over_par = elapsed_s - cfg.par_time_s
            reward += cfg.finish_time_weight * over_par

        return reward
