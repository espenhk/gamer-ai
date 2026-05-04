"""StarCraft 2 reward calculator.

For minigames the canonical signal is the cumulative ``score`` returned by
PySC2 itself — so the calculator's main job is to compute the score *delta*
each step and add small shaping terms.  For the ladder game stub the same
signal is used as a placeholder; richer reward shaping (kill credit, mineral
income, supply lead) is left for the follow-up issue that adds learning.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import yaml

from framework.base_reward import RewardCalculatorBase


@dataclass
class SC2RewardConfig:
    """Reward weights for the SC2 environment.

    Python defaults match a sensible baseline for ``MoveToBeacon``.  Other
    minigames will typically want a different ``score_weight`` because the
    environment-provided score scales differ (e.g. mineral count vs.
    beacon-touch count).

    Parameters
    ----------
    score_weight :
        Coefficient on the ``score`` delta returned by PySC2 each step.
    win_bonus :
        One-shot bonus when the player reward signals a win (>0 from PySC2).
    loss_penalty :
        One-shot penalty when the player reward signals a loss (<0).
    step_penalty :
        Tiny negative reward every tick — discourages indefinite no-op.
    idle_penalty :
        Per-step penalty when ``army_count == 0 and food_used < food_cap``;
        used by ``BuildMarines`` to discourage doing nothing.
    idle_bonus :
        Per-step bonus awarded when the agent issues ``no_op`` *and* friendly
        units are within combat range of an enemy on the screen (issue #127).
        Default ``0.0`` — opt-in.  Makes the "stand still so units can shoot"
        lesson learnable rather than only discoverable through luck.  Requires
        the screen summary features (``screen_self_count`` / ``screen_enemy_count``
        / centroids) populated by the client; with the default obs preset
        these are always present.
    economy_weight :
        Coefficient on (minerals + vespene) delta.  Useful for economy
        minigames.  Set to 0 for pure-combat minigames.
    """

    score_weight:    float = 1.0
    win_bonus:       float = 100.0
    loss_penalty:    float = -100.0
    step_penalty:    float = -0.001
    idle_penalty:    float = 0.0
    idle_bonus:      float = 0.0
    economy_weight:  float = 0.0

    @classmethod
    def from_yaml(cls, path: str) -> SC2RewardConfig:
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


class SC2RewardCalculator(RewardCalculatorBase):
    """Reward computation for the SC2 environment.

    Stateless — episode-state derivatives (``prev_score``) are passed in via
    the ``info`` dict produced by :class:`games.sc2.env.SC2Env`.

    Expected keys in ``info``:
        ``score``         — current cumulative environment score
        ``prev_score``    — previous step's environment score
        ``minerals``, ``vespene`` — current totals
        ``prev_minerals``, ``prev_vespene`` — previous totals
        ``army_count``, ``food_used``, ``food_cap``
        ``player_outcome`` — None / +1 / -1 (only set on the final step)
        ``action_fn_idx`` — fn_idx of the action issued this step (for ``idle_bonus``)
        ``screen_self_count`` / ``screen_enemy_count`` — friendly / enemy
            pixel counts on screen (for ``idle_bonus`` combat-range check)
        ``screen_self_cx`` / ``screen_self_cy`` / ``screen_enemy_cx`` /
            ``screen_enemy_cy`` — centroids in screen pixels
    """

    # Maximum centroid-distance (in screen pixels) for friendly units to be
    # considered "in combat range" for the ``idle_bonus`` shaping reward.
    # 0.4 of the 64-pixel default screen ≈ 25 pixels — roughly Marine range.
    _COMBAT_RANGE_PX: float = 25.0

    def __init__(self, config: SC2RewardConfig) -> None:
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

        # Score delta — primary signal for minigames.
        prev_score = info.get("prev_score", 0.0)
        curr_score = info.get("score", 0.0)
        reward += cfg.score_weight * (curr_score - prev_score)

        # Economy delta (optional — typically 0 for pure-combat minigames).
        if cfg.economy_weight != 0.0:
            prev_min = info.get("prev_minerals", 0.0)
            curr_min = info.get("minerals", 0.0)
            prev_vesp = info.get("prev_vespene", 0.0)
            curr_vesp = info.get("vespene", 0.0)
            reward += cfg.economy_weight * (
                (curr_min - prev_min) + (curr_vesp - prev_vesp)
            )

        # Idle penalty: nothing built and supply slack — encourages building.
        if cfg.idle_penalty != 0.0:
            army = info.get("army_count", 0.0)
            food_used = info.get("food_used", 0.0)
            food_cap = info.get("food_cap", 0.0)
            if army == 0 and food_used < food_cap:
                reward += cfg.idle_penalty * n_ticks

        # Idle bonus (issue #127): reward standing still when units are in
        # combat range of an enemy.  Encourages letting units shoot rather
        # than chasing.  Requires fn_idx == 0 (no_op) AND non-zero friendly
        # and enemy presence on screen with their centroids close together.
        if cfg.idle_bonus != 0.0 and info.get("action_fn_idx") == 0:
            self_count  = info.get("screen_self_count", 0.0)
            enemy_count = info.get("screen_enemy_count", 0.0)
            if self_count > 0 and enemy_count > 0:
                dx = float(info.get("screen_self_cx", 0.0)) - float(info.get("screen_enemy_cx", 0.0))
                dy = float(info.get("screen_self_cy", 0.0)) - float(info.get("screen_enemy_cy", 0.0))
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= self._COMBAT_RANGE_PX:
                    reward += cfg.idle_bonus * n_ticks

        # Time cost.
        reward += cfg.step_penalty * n_ticks

        # Terminal win/loss bonus (only set when the env signals an outcome).
        outcome = info.get("player_outcome")
        if finished and outcome is not None:
            if outcome > 0:
                reward += cfg.win_bonus
            elif outcome < 0:
                reward += cfg.loss_penalty

        return reward
