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
        units are within effective attack range of an enemy on the screen
        (issue #127).  The gate applies at 95% of max range so idling exactly
        at the edge is not rewarded. Default ``0.0`` — opt-in.  Makes the
        "stand still so units can shoot" lesson learnable rather than only
        discoverable through luck. Requires screen summary features and, for
        unit-aware range gating, ``self_attack_range_px`` from the client.
    attack_move_bonus :
        Per-step bonus awarded when the agent issues ``Attack_screen``
        (fn_idx 3) *and* the click target is **not** on a visible enemy unit
        (i.e. the attack lands on empty ground — an A-move command).
        Encourages the unit to patrol-attack rather than plain-move past
        enemies.  Default ``0.0`` — opt-in.
    click_attack_bonus :
        Per-step bonus awarded when the agent issues ``Attack_screen``
        (fn_idx 3) *and* the click target is close to the enemy centroid on
        screen — i.e. a direct click-to-attack on a visible enemy unit.
        Should typically be set slightly higher than ``attack_move_bonus`` to
        prefer precision targeting.  Subject to ``click_attack_cooldown_steps``
        to discourage very rapid target-switching.  Default ``0.0`` — opt-in.
    click_attack_cooldown_steps :
        Minimum env steps that must elapse before a ``click_attack_bonus`` is
        awarded again when the agent switches to a *new* attack target (one
        that is more than ``_CLICK_ATTACK_RADIUS_FRAC`` of the screen away
        from the previous target).  Targeting the *same* unit (same vicinity)
        always fires immediately — the cooldown only penalises rapidly bouncing
        between different enemies.  Default ``8``.
    economy_weight :
        Coefficient on (minerals + vespene) delta.  Useful for economy
        minigames.  Set to 0 for pure-combat minigames.
    move_exploration_bonus :
        Per-step bonus for issuing ``Move_screen`` commands whose target is at
        least ``SC2RewardCalculator._MOVE_MIN_MEANINGFUL_FRAC`` of the screen
        away from the previous move target.  Bonus scales linearly with
        distance up to ``_MOVE_EXPLORATION_NORM``.  Sub-threshold moves receive
        no bonus — this prevents stutter-stepping (tiny back-and-forth moves)
        from farming exploration rewards.
    move_repeat_penalty :
        Per-step penalty when a ``Move_screen`` command targets a point that is
        less than ``_MOVE_MIN_MEANINGFUL_FRAC`` of the screen away from the
        previous move target (covers both exact repeats and tiny stutter
        steps).
    move_self_penalty :
        Per-step penalty when a ``Move_screen`` command targets the centroid of
        currently-visible friendly units (a common "keep moving to where we
        already are" failure mode).
    attack_friendly_penalty :
        Per-step penalty when the agent issues ``Attack_screen`` (fn_idx 3)
        *and* the click target lands on or near the centroid of currently-visible
        **friendly** units.  In-game this issues an attack command against an ally
        (friendly fire), causing units to shoot team-mates to death.  The penalty
        fires whenever ``screen_self_count > 0`` and the target pixel is within
        ``_ATTACK_SELF_RADIUS_FRAC`` of the friendly centroid.  Set to a large
        negative default to strongly discourage this behaviour.
    unit_loss_penalty :
        Penalty per unit lost this step (army_count drop).  Applied when
        ``army_count < prev_army_count``.  Scaled by the number of units lost
        so losing two units at once yields twice the penalty.  Default ``0.0``
        — opt-in.  Requires ``prev_army_count`` in info (set by ``SC2Env``).
    damage_taken_penalty :
        Penalty per raw HP+shield point lost this step across all visible
        friendly units.  Computed from ``feature_units`` health+shield sum
        delta (``prev_total_self_hp - total_self_hp``).  Only visible units
        contribute, so units that leave the camera frame can cause false
        positives; keep the weight small.  Default ``0.0`` — opt-in.
    passive_under_fire_penalty :
        Per-step penalty applied when enemy units are within effective attack
        range of friendly units **and** the agent did not issue ``Attack_screen``
        (fn_idx 3).  Encourages the agent to fight back rather than run or idle
        while taking fire.  Uses the same range gate as ``idle_bonus``
        (``self_attack_range_px`` from info, or the screen-fraction fallback)
        but without the 95% inside-margin so the penalty fires at the full
        attack range.  Default ``0.0`` — opt-in.
    small_selection_bonus :
        Per-step bonus for issuing unit-targeted commands with a small active
        selection (``selected_count == 1`` or ``selected_count`` less than 50%
        of ``visible_self_unit_count``).  Encourages micro-style control of one
        or a few units rather than always commanding the full visible army.
        Default ``0.0`` — opt-in.
    """

    score_weight: float = 1.0
    win_bonus: float = 100.0
    loss_penalty: float = -100.0
    step_penalty: float = -0.001
    idle_penalty: float = 0.0
    idle_bonus: float = 0.0
    attack_move_bonus: float = 0.0
    click_attack_bonus: float = 0.0
    click_attack_cooldown_steps: int = 8
    economy_weight: float = 0.0
    move_exploration_bonus: float = 0.01
    move_repeat_penalty: float = -0.02
    move_self_penalty: float = -0.01
    attack_friendly_penalty: float = -5.0
    unit_loss_penalty: float = 0.0
    damage_taken_penalty: float = 0.0
    passive_under_fire_penalty: float = 0.0
    small_selection_bonus: float = 0.0

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

    Holds a small amount of per-episode state to implement the
    ``click_attack_bonus`` cooldown (tracking the last click-attack target
    so rapid target-switching is not rewarded).  Call :meth:`reset` at the
    start of each episode (``SC2Env`` does this automatically).

    Expected keys in ``info``:
        ``score``         — current cumulative environment score
        ``prev_score``    — previous step's environment score
        ``minerals``, ``vespene`` — current totals
        ``prev_minerals``, ``prev_vespene`` — previous totals
        ``army_count``, ``food_used``, ``food_cap``
        ``player_outcome`` — None / +1 / -1 (only set on the final step)
        ``action_fn_idx`` — fn_idx of the action issued this step
        ``action_target_x`` / ``action_target_y`` — normalised [0, 1] screen
            target coordinates of the action (used by ``attack_move_bonus``
            and ``click_attack_bonus`` to classify the attack type)
        ``prev_move_target_x`` / ``prev_move_target_y`` — previous move target
            in [0, 1], or None if no previous move target is known
        ``screen_self_count`` / ``screen_enemy_count`` — friendly / enemy
            pixel counts on screen
        ``screen_self_cx`` / ``screen_self_cy`` / ``screen_enemy_cx`` /
            ``screen_enemy_cy`` — centroids in screen pixels
        ``screen_size`` — screen feature-layer side length (default 64)
        ``self_attack_range_px`` — optional estimated max friendly attack
            range in pixels (from SC2 client feature_units)
        ``prev_army_count`` / ``army_count`` — friendly army counts for the
            unit-loss penalty (set by SC2Env)
        ``prev_total_self_hp`` / ``total_self_hp`` — summed health+shield of
            visible friendly units; used by the damage-taken penalty (set by
            SC2Client / SC2Env)
        ``selected_count`` / ``visible_self_unit_count`` — active selection
            size and visible friendly unit count for the small-selection bonus
    """

    # Fallback max attack range (fraction of screen) when unit-specific range
    # metadata is unavailable.
    _DEFAULT_COMBAT_RANGE_FRAC: float = 20.0 / 64.0
    # Idle bonus only applies slightly inside max range to avoid "edge of range"
    # stalling behavior.
    _IDLE_RANGE_INSIDE_FRAC: float = 0.95
    # Minimum move distance (as a screen fraction) that counts as a
    # "meaningful" move.  Below this threshold the exploration bonus is
    # withheld and the repeat penalty is applied instead.  At the default
    # 64-pixel screen this corresponds to ~6 px — more than a single
    # stutter step but less than a typical tactical repositioning.
    _MOVE_MIN_MEANINGFUL_FRAC: float = 6.0 / 64.0
    # Radius (as a screen fraction) considered "targeting where my units are".
    _MOVE_SELF_RADIUS_FRAC: float = 6.0 / 64.0
    # Distance normaliser for movement exploration bonus.
    _MOVE_EXPLORATION_NORM: float = 0.5

    # Radius (as screen fraction) within which a click target must fall to be
    # classified as "on an enemy unit" rather than "attack-move to ground".
    # At the 64-pixel default this is 8 px — roughly one unit sprite.
    _CLICK_ATTACK_RADIUS_FRAC: float = 8.0 / 64.0

    # Radius (as screen fraction) within which an Attack_screen target is
    # considered to be on a friendly unit — triggers attack_friendly_penalty.
    # Matches _CLICK_ATTACK_RADIUS_FRAC so the same sprite-footprint heuristic
    # applies consistently to both ally and enemy targeting: one unit sprite ≈ 8 px
    # at the 64-pixel default.  If you widen click-attack detection you will
    # likely want to widen friendly-fire detection equally.
    _ATTACK_SELF_RADIUS_FRAC: float = 8.0 / 64.0

    def __init__(self, config: SC2RewardConfig) -> None:
        self.config = config
        self._last_click_x: float | None = None
        self._last_click_y: float | None = None
        self._last_click_step: int = -1
        self._step_count: int = 0

    def reset(self) -> None:
        """Clear per-episode state at the start of a new episode."""
        self._last_click_x = None
        self._last_click_y = None
        self._last_click_step = -1
        self._step_count = 0

    def compute(
        self,
        prev_state: Any,
        curr_state: Any,
        finished: bool,
        elapsed_s: float,
        info: dict,
        n_ticks: int = 1,
    ) -> float:
        return self.compute_with_components(
            prev_state,
            curr_state,
            finished,
            elapsed_s,
            info,
            n_ticks,
        )[0]

    def compute_with_components(
        self,
        prev_state: Any,
        curr_state: Any,
        finished: bool,
        elapsed_s: float,
        info: dict,
        n_ticks: int = 1,
    ) -> tuple[float, dict[str, float]]:
        """Return ``(reward, components)`` for this step (issue #128/2b).

        ``components`` exposes a per-term breakdown so analytics can
        attribute reward to ``score``, ``economy``, ``idle_penalty``,
        ``idle_bonus``, ``move_exploration``, ``move_repeat_penalty``,
        ``move_self_penalty``, ``attack_move_bonus``, ``click_attack_bonus``,
        ``attack_friendly_penalty``, ``unit_loss``, ``damage_taken``,
        ``passive_under_fire``, ``small_selection``, ``step_penalty`` and
        ``terminal``
        separately.
        """
        cfg = self.config
        components: dict[str, float] = {}

        self._step_count += 1

        # Score delta — primary signal for minigames.
        prev_score = info.get("prev_score", 0.0)
        curr_score = info.get("score", 0.0)
        components["score"] = float(cfg.score_weight * (curr_score - prev_score))

        # Economy delta (optional — typically 0 for pure-combat minigames).
        if cfg.economy_weight != 0.0:
            prev_min = info.get("prev_minerals", 0.0)
            curr_min = info.get("minerals", 0.0)
            prev_vesp = info.get("prev_vespene", 0.0)
            curr_vesp = info.get("vespene", 0.0)
            components["economy"] = float(
                cfg.economy_weight * ((curr_min - prev_min) + (curr_vesp - prev_vesp))
            )
        else:
            components["economy"] = 0.0

        # Idle penalty: nothing built and supply slack — encourages building.
        idle_pen = 0.0
        if cfg.idle_penalty != 0.0:
            army = info.get("army_count", 0.0)
            food_used = info.get("food_used", 0.0)
            food_cap = info.get("food_cap", 0.0)
            if army == 0 and food_used < food_cap:
                idle_pen = cfg.idle_penalty * n_ticks
        components["idle_penalty"] = float(idle_pen)

        # Idle bonus (issue #127): reward standing still when units are in
        # effective attack range of an enemy. If the client provides
        # self_attack_range_px, use that unit-aware threshold; otherwise use a
        # conservative screen-size-scaled fallback.
        idle_bonus = 0.0
        if cfg.idle_bonus != 0.0 and info.get("action_fn_idx") == 0:
            self_count = info.get("screen_self_count", 0.0)
            enemy_count = info.get("screen_enemy_count", 0.0)
            if self_count > 0 and enemy_count > 0:
                dx = float(info.get("screen_self_cx", 0.0)) - float(
                    info.get("screen_enemy_cx", 0.0)
                )
                dy = float(info.get("screen_self_cy", 0.0)) - float(
                    info.get("screen_enemy_cy", 0.0)
                )
                dist = (dx * dx + dy * dy) ** 0.5
                screen_size = float(info.get("screen_size", 64))
                max_attack_range_px = float(
                    info.get(
                        "self_attack_range_px",
                        self._DEFAULT_COMBAT_RANGE_FRAC * screen_size,
                    )
                )
                idle_gate_px = max(0.0, max_attack_range_px) * self._IDLE_RANGE_INSIDE_FRAC
                if dist <= idle_gate_px:
                    idle_bonus = cfg.idle_bonus * n_ticks
        components["idle_bonus"] = float(idle_bonus)

        # Move shaping: encourage varied move targets and
        # discourage repeatedly targeting where units already are.
        move_exploration = 0.0
        move_repeat_penalty = 0.0
        move_self_penalty = 0.0
        if info.get("action_fn_idx") == 2:
            x = float(info.get("action_target_x", 0.5))
            y = float(info.get("action_target_y", 0.5))

            prev_x = info.get("prev_move_target_x")
            prev_y = info.get("prev_move_target_y")
            if prev_x is not None and prev_y is not None:
                dx_prev = x - float(prev_x)
                dy_prev = y - float(prev_y)
                dist_prev = (dx_prev * dx_prev + dy_prev * dy_prev) ** 0.5
                if dist_prev >= self._MOVE_MIN_MEANINGFUL_FRAC:
                    if cfg.move_exploration_bonus != 0.0:
                        novelty = min(1.0, dist_prev / self._MOVE_EXPLORATION_NORM)
                        move_exploration = (
                            cfg.move_exploration_bonus * novelty * n_ticks
                        )
                else:
                    if cfg.move_repeat_penalty != 0.0:
                        move_repeat_penalty = cfg.move_repeat_penalty * n_ticks

            if cfg.move_self_penalty != 0.0:
                self_count = float(info.get("screen_self_count", 0.0))
                if self_count > 0:
                    screen_size = max(1.0, float(info.get("screen_size", 64)))
                    self_x = float(info.get("screen_self_cx", 0.0)) / screen_size
                    self_y = float(info.get("screen_self_cy", 0.0)) / screen_size
                    dx_self = x - self_x
                    dy_self = y - self_y
                    dist_self = (dx_self * dx_self + dy_self * dy_self) ** 0.5
                    if dist_self <= self._MOVE_SELF_RADIUS_FRAC:
                        move_self_penalty = cfg.move_self_penalty * n_ticks
        components["move_exploration"] = float(move_exploration)
        components["move_repeat_penalty"] = float(move_repeat_penalty)
        components["move_self_penalty"] = float(move_self_penalty)

        # Unit loss: penalty per army unit that died this step.
        unit_loss = 0.0
        if cfg.unit_loss_penalty != 0.0:
            prev_army = float(info.get("prev_army_count", 0.0))
            curr_army = float(info.get("army_count", 0.0))
            units_lost = max(0.0, prev_army - curr_army)
            unit_loss = cfg.unit_loss_penalty * units_lost
        components["unit_loss"] = float(unit_loss)

        # Damage taken: penalty per HP+shield point lost across visible friendly units.
        damage_taken = 0.0
        if cfg.damage_taken_penalty != 0.0:
            prev_hp = float(info.get("prev_total_self_hp", 0.0))
            curr_hp = float(info.get("total_self_hp", 0.0))
            hp_lost = max(0.0, prev_hp - curr_hp)
            damage_taken = cfg.damage_taken_penalty * hp_lost
        components["damage_taken"] = float(damage_taken)

        # Passive under fire: enemies in attack range but agent not attacking.
        passive_under_fire = 0.0
        if cfg.passive_under_fire_penalty != 0.0 and info.get("action_fn_idx") != 3:
            self_count = info.get("screen_self_count", 0.0)
            enemy_count = info.get("screen_enemy_count", 0.0)
            if self_count > 0 and enemy_count > 0:
                dx = float(info.get("screen_self_cx", 0.0)) - float(info.get("screen_enemy_cx", 0.0))
                dy = float(info.get("screen_self_cy", 0.0)) - float(info.get("screen_enemy_cy", 0.0))
                dist = (dx * dx + dy * dy) ** 0.5
                screen_size = float(info.get("screen_size", 64))
                max_attack_range_px = float(
                    info.get(
                        "self_attack_range_px",
                        self._DEFAULT_COMBAT_RANGE_FRAC * screen_size,
                    )
                )
                if dist <= max_attack_range_px:
                    passive_under_fire = cfg.passive_under_fire_penalty * n_ticks
        components["passive_under_fire"] = float(passive_under_fire)

        # Reward small active selections when issuing unit-targeted commands.
        small_selection = 0.0
        if cfg.small_selection_bonus != 0.0 and info.get("action_fn_idx") in (2, 3, 5):
            selected_count = float(info.get("selected_count", 0.0))
            visible_self = float(info.get("visible_self_unit_count", 0.0))
            if selected_count > 0.0 and visible_self > 0.0:
                if selected_count <= 1.0 or selected_count < (0.5 * visible_self):
                    small_selection = cfg.small_selection_bonus * n_ticks
        components["small_selection"] = float(small_selection)

        # Attack bonuses: split Attack_screen into attack-move (ground target)
        # and click-to-attack (target on/near a visible enemy unit).
        attack_move_bonus = 0.0
        click_attack_bonus = 0.0
        if info.get("action_fn_idx") == 3:
            screen_size = float(info.get("screen_size", 64))
            # Use (screen_size - 1) to match the client's pixel conversion:
            # x_screen = int(clip(norm, 0, 1) * (screen_size - 1))
            # Centroids from games.sc2.client.SC2Client._centroid() are raw
            # pixel indices in [0, screen_size-1].
            scale = max(1.0, screen_size - 1.0)
            enemy_count = info.get("screen_enemy_count", 0.0)
            tx_norm = float(info.get("action_target_x", 0.5))
            ty_norm = float(info.get("action_target_y", 0.5))
            tx_px = tx_norm * scale
            ty_px = ty_norm * scale
            ecx = float(info.get("screen_enemy_cx", 0.0))
            ecy = float(info.get("screen_enemy_cy", 0.0))
            click_radius_px = self._CLICK_ATTACK_RADIUS_FRAC * screen_size

            # "Click to attack": target pixel is within click_radius of enemy
            # centroid and at least one enemy is visible.
            on_enemy = (
                enemy_count > 0
                and ((tx_px - ecx) ** 2 + (ty_px - ecy) ** 2) ** 0.5 <= click_radius_px
            )

            if on_enemy and cfg.click_attack_bonus != 0.0:
                # Check cooldown: only blocked when the agent switches to a
                # *different* target quickly (same target always passes).
                steps_since = self._step_count - self._last_click_step
                if self._last_click_x is None:
                    same_target = True
                else:
                    dx = tx_px - self._last_click_x
                    dy = (
                        ty_px - self._last_click_y
                    )  # _last_click_x/_last_click_y are set together
                    same_target = (dx * dx + dy * dy) ** 0.5 <= click_radius_px
                if same_target or steps_since >= cfg.click_attack_cooldown_steps:
                    click_attack_bonus = cfg.click_attack_bonus * n_ticks
                # Always update the tracked target on a click-attack so the
                # cooldown window resets relative to the most recent action.
                self._last_click_x = tx_px
                self._last_click_y = ty_px
                self._last_click_step = self._step_count
            elif not on_enemy and cfg.attack_move_bonus != 0.0 and enemy_count > 0:
                # Attack-move to ground while enemies are visible.
                attack_move_bonus = cfg.attack_move_bonus * n_ticks

        components["attack_move_bonus"] = float(attack_move_bonus)
        components["click_attack_bonus"] = float(click_attack_bonus)

        # Friendly-fire penalty: Attack_screen aimed at own units.
        attack_friendly_penalty = 0.0
        if cfg.attack_friendly_penalty != 0.0 and info.get("action_fn_idx") == 3:
            self_count = float(info.get("screen_self_count", 0.0))
            if self_count > 0:
                screen_size = max(1.0, float(info.get("screen_size", 64)))
                # Use (screen_size - 1) to match client pixel conversion.
                scale = max(1.0, screen_size - 1.0)
                scx = float(info.get("screen_self_cx", 0.0))
                scy = float(info.get("screen_self_cy", 0.0))
                tx_norm = float(info.get("action_target_x", 0.5))
                ty_norm = float(info.get("action_target_y", 0.5))
                tx_px = tx_norm * scale
                ty_px = ty_norm * scale
                dx = tx_px - scx
                dy = ty_px - scy
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= self._ATTACK_SELF_RADIUS_FRAC * screen_size:
                    attack_friendly_penalty = cfg.attack_friendly_penalty * n_ticks
        components["attack_friendly_penalty"] = float(attack_friendly_penalty)

        # Time cost.
        components["step_penalty"] = float(cfg.step_penalty * n_ticks)

        # Terminal win/loss bonus (only set when the env signals an outcome).
        terminal = 0.0
        outcome = info.get("player_outcome")
        if finished and outcome is not None:
            if outcome > 0:
                terminal = cfg.win_bonus
            elif outcome < 0:
                terminal = cfg.loss_penalty
        components["terminal"] = float(terminal)

        reward = float(sum(components.values()))
        return reward, components
