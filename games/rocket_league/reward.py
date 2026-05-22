"""Rocket League reward configuration and calculator.

Reward signal
-------------
Dense shaping:
  - velocity towards ball  (``vel_to_ball_weight``)
  - boost pickup bonus     (``boost_weight``)
  - ball touch bonus       (``touch_bonus``)
  - per-step time cost     (``step_penalty``)

Sparse terminal:
  - goal scored            (+``goal_weight``)
  - goal conceded          (−``concede_penalty``)
"""

from __future__ import annotations

import dataclasses
from typing import Any

import yaml


@dataclasses.dataclass
class RocketLeagueRewardConfig:
    """Reward hyperparameters for Rocket League training.

    Parameters
    ----------
    vel_to_ball_weight :
        Reward proportional to the car's velocity component directed towards
        the ball each step.  Encourages the agent to chase the ball.
    boost_weight :
        Bonus applied each step when the agent is currently boosting
        (``action[6] > 0.5``).  Can be negative to discourage boost waste.
    touch_bonus :
        One-time bonus awarded the first step the car touches the ball in an
        episode (detected via ``info["ball_touched"]``).
    goal_weight :
        Reward added when the agent scores a goal.
    concede_penalty :
        Penalty (positive value → subtracted) when the opponent scores.
    step_penalty :
        Per-step time cost, encourages the agent to act efficiently.
    """

    vel_to_ball_weight: float = 0.01
    boost_weight: float = 0.0
    touch_bonus: float = 1.0
    goal_weight: float = 10.0
    concede_penalty: float = 5.0
    step_penalty: float = -0.001

    @classmethod
    def from_yaml(cls, path: str) -> "RocketLeagueRewardConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        unknown = [k for k in d if k not in cls.__dataclass_fields__]
        if unknown:
            raise ValueError(
                f"Unknown reward config key(s): {unknown}. "
                f"Valid keys: {list(cls.__dataclass_fields__)}"
            )
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RocketLeagueRewardCalculator:
    """Stateful reward calculator for one Rocket League episode.

    Parameters
    ----------
    config :
        Reward hyperparameters.

    Usage
    -----
    Call ``reset()`` at the start of each episode, then ``compute()`` each
    step.  The calculator keeps episode-level state (e.g. whether the ball
    has been touched this episode) so that touch bonuses fire only once.
    """

    def __init__(self, config: RocketLeagueRewardConfig) -> None:
        self._cfg = config
        self._touched_ball_this_ep: bool = False

    def reset(self) -> None:
        """Reset episode-level state."""
        self._touched_ball_this_ep = False

    def compute(
        self,
        prev_state: Any,
        curr_state: Any,
        finished: bool,
        elapsed_s: float,
        info: dict,
    ) -> float:
        """Compute step reward.

        Parameters
        ----------
        prev_state, curr_state :
            Raw game states (unused; observations are read from *info*).
        finished :
            True when the episode ended due to a goal or timeout.
        elapsed_s :
            Seconds elapsed in the episode (unused by this calculator but
            kept for API compatibility).
        info :
            Step info dict from ``RocketLeagueEnv.step()``.  Expected keys:

            ``vel_towards_ball`` — scalar velocity (UU/s) directed at ball.
            ``boosting``         — bool, whether boost is active this step.
            ``ball_touched``     — bool, car touched ball this step.
            ``goal_scored``      — bool, agent scored a goal this step.
            ``goal_conceded``    — bool, opponent scored this step.
        """
        cfg = self._cfg
        reward = 0.0

        # Dense: velocity towards ball shaping.
        vel_towards = float(info.get("vel_towards_ball", 0.0))
        reward += cfg.vel_to_ball_weight * vel_towards

        # Dense: boost usage.
        if info.get("boosting", False):
            reward += cfg.boost_weight

        # Dense: first touch in this episode.
        if info.get("ball_touched", False) and not self._touched_ball_this_ep:
            reward += cfg.touch_bonus
            self._touched_ball_this_ep = True

        # Sparse: goal events.
        if info.get("goal_scored", False):
            reward += cfg.goal_weight
        if info.get("goal_conceded", False):
            reward -= cfg.concede_penalty

        # Time cost.
        reward += cfg.step_penalty

        return float(reward)
