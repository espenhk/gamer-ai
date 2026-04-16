from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import yaml

from framework.base_reward import RewardCalculatorBase
from games.tmnf.state import StateData


@dataclass
class RewardConfig:
    """
    All reward weights in one place.

    Signs:
        Positive weights add reward (encourage the behaviour).
        Negative weights subtract reward (penalise the behaviour).

    The canonical training values live in config/reward_config.yaml (copied into
    each experiment directory on first run).

    Python defaults are kept for backward compatibility and lightweight unit tests.
    Training should still load YAML via RewardConfig.from_yaml().

    Parameters
    ----------
    progress_weight:
        Reward proportional to how far along the track the car advanced this step.
        Large value because this is the primary objective.
    centerline_weight:
        Penalty coefficient for lateral deviation from the centreline (negative).
        Penalty = centerline_weight * |lateral_offset| ** centerline_exp
    centerline_exp:
        Exponent for the centeline penalty (default 2 = quadratic).
        Small drifts are forgiven; large drifts are heavily penalised.
    speed_weight:
        Small reward per m/s to break ties and encourage not braking unnecessarily.
    step_penalty:
        Tiny negative reward every tick so the agent prefers finishing fast.
    finish_bonus:
        One-time bonus when track_progress reaches 1.0.
    finish_time_weight:
        Additional bonus/penalty relative to par_time_s.
        Negative weight means slower = more negative reward.
    par_time_s:
        Reference lap time in seconds for the finish_time_weight calculation.
    accel_bonus:
        Flat reward every step the throttle is pressed.
        Prevents the policy from preferring coast actions.
    airborne_penalty:
        Applied when the car has ≤1 wheel in contact AND vertical_offset ≤ 0
        (below or beside the centreline — not a legitimate jump).
    lidar_wall_weight:
        Penalty = lidar_wall_weight * (1 - min_ray)^2, where min_ray is the
        nearest wall distance normalised to [0, 1].  Set to 0.0 when n_lidar_rays=0.
    crash_threshold_m:
        The env ends the episode when |lateral_offset| exceeds this (metres).
    """

    progress_weight:    float = 10.0
    centerline_weight:  float = -0.5
    centerline_exp:     float = 2.0
    speed_weight:       float = 0.05
    step_penalty:       float = -0.01
    finish_bonus:       float = 100.0
    finish_time_weight: float = -0.1
    par_time_s:         float = 60.0
    accel_bonus:        float = 0.5
    airborne_penalty:   float = -1.0
    lidar_wall_weight:  float = 0.0
    crash_threshold_m:  float = 25.0
    track_name:         str   = "a03"
    centerline_path:    str   = "tracks/a03_centerline.npy"

    @classmethod
    def from_yaml(cls, path: str) -> RewardConfig:
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


class RewardCalculator(RewardCalculatorBase):
    """Stateless reward computation — call compute() every RL step."""

    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def compute(
        self,
        prev_state: StateData,
        curr_state: StateData,
        finished: bool,
        elapsed_s: float,
        info: dict,
        n_ticks: int = 1,
    ) -> float:
        """Compute reward for one RL step.

        Game-specific signals are passed through *info* rather than as extra
        positional arguments so that the signature matches RewardCalculatorBase:

            info["accelerating"]  bool    — whether the throttle was pressed
            info["lidar_rays"]    ndarray — LIDAR wall-distance rays (optional)
        """
        cfg          = self.config
        accelerating = bool(info.get("accelerating", False))
        lidar_rays   = info.get("lidar_rays", None)
        reward       = 0.0

        # Progress: reward for advancing along the track this step.
        if curr_state.track_progress is not None and prev_state.track_progress is not None:
            delta = curr_state.track_progress - prev_state.track_progress
            reward += delta * cfg.progress_weight

        # Centerline: quadratic penalty for lateral deviation.
        # Scaled by n_ticks: the car was off-centreline for this many game ticks.
        if curr_state.lateral_offset is not None:
            reward += (
                cfg.centerline_weight * abs(curr_state.lateral_offset) ** cfg.centerline_exp
                * n_ticks
            )

        # Speed: small reward for going fast.
        # Scaled by n_ticks: the car was travelling at this speed for this many ticks.
        reward += cfg.speed_weight * curr_state.velocity.magnitude() * n_ticks

        # Acceleration bonus: nudge the policy away from coasting.
        # Scaled by n_ticks because the action was held for that many game ticks.
        if accelerating:
            reward += cfg.accel_bonus * n_ticks

        # Time cost: constant small penalty per tick, scaled by ticks covered.
        reward += cfg.step_penalty * n_ticks

        # Finish: one-time bonus + time-relative bonus.
        if finished:
            reward += cfg.finish_bonus
            over_par = elapsed_s - cfg.par_time_s
            reward += cfg.finish_time_weight * over_par  # negative if slow

        # Airborne penalty: only when below or beside the centerline.
        # Scaled by n_ticks: the car was airborne for this many game ticks.
        if curr_state.vertical_offset is not None:
            wheels_in_contact = sum(w.contact for w in curr_state.wheels)
            airborne = wheels_in_contact <= 1
            # vertical_offset > 0 → car is above centerline → legitimate jump → no penalty
            if airborne and curr_state.vertical_offset <= 0.0:
                reward += cfg.airborne_penalty * n_ticks

        # Lidar wall proximity: quadratic penalty for the nearest detected wall.
        if (
            lidar_rays is not None
            and len(lidar_rays) > 0
            and cfg.lidar_wall_weight != 0.0
        ):
            min_ray = float(np.min(lidar_rays))
            reward += cfg.lidar_wall_weight * (1.0 - min_ray) ** 2

        return reward
