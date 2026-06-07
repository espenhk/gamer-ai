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
from games.sc2.actions import FN_IDX_TO_CATEGORY
from games.sc2.tech_tree import PRECONDITIONS


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
    idle_worker_penalty :
        Per-step penalty scaled by the number of idle workers reported by
        PySC2 (``player.idle_worker_count``).  Having idle workers is almost
        always inefficient in StarCraft 2 — workers should be mining or
        building.  The penalty fires whenever ``idle_worker_count > 0``; the
        full penalty is ``idle_worker_penalty × idle_worker_count × n_ticks``.
        Default ``0.0`` — opt-in.  Recommended for economy maps and ladder
        runs where worker efficiency matters (e.g. ``-0.05`` to ``-0.5``
        per idle worker per step).
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
        Per-step bonus awarded when a ``Move_screen`` command is issued and
        the current friendly-unit centroid is in a grid cell that is
        currently **unexplored**.  The screen is cut into a
        ``move_exploration_grid_size`` × ``move_exploration_grid_size`` grid
        (default 8×8 → cells ~1/8 of the screen wide).  A cell becomes
        explored as soon as the centroid enters it (regardless of action
        type) and stays explored while the centroid keeps refreshing it; it
        **expires** ``move_exploration_decay_steps`` env steps after the
        centroid last left it, becoming eligible to be rewarded again on a
        later return.  Units must be visible (``screen_self_count > 0``).
        This blocks the command-spam exploit — a stationary centroid keeps
        refreshing its own cell so spamming moves from one spot earns nothing
        — while the decay prevents the bonus from going permanently silent
        once the whole screen has been covered (which otherwise makes
        freezing in place optimal).
    move_exploration_grid_size :
        Side length of the square screen grid used by
        ``move_exploration_bonus`` (number of cells per axis).  Default ``8``
        (cells ~1/8 of the screen).  Higher = finer cells / smaller meaningful
        relocation; lower = coarser.
    move_exploration_decay_steps :
        Number of env steps after which an explored cell expires and may be
        rewarded again on a return visit (measured in env steps, like
        ``click_attack_cooldown_steps``).  ``0`` disables decay, restoring the
        permanent "once per cell per episode" behaviour.  Larger values keep
        an area "explored" for longer, reducing how often re-visits are
        rewarded.
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
    attack_bonus :
        Per-step bonus awarded whenever the agent issues ``Attack_screen``
        (fn_idx 3), regardless of whether the target is on an enemy unit
        (click-to-attack) or on open ground (A-move).  A simpler alternative
        to enabling both ``attack_move_bonus`` and ``click_attack_bonus``
        separately; all three can be active at once.
        Default ``0.0`` — opt-in.
    early_random_action_bonus :
        Per-step bonus for trying a previously-unseen non-no-op action
        function id during the early part of each episode. This explicitly
        rewards broad early action-space exploration instead of repeatedly
        exploiting one action that happened to work first.
        Default ``0.0`` — opt-in.
    early_random_action_window_steps :
        Number of episode steps from reset in which
        ``early_random_action_bonus`` may fire. Outside this window the bonus
        is disabled. Default ``250``.
    new_action_unlock_bonus :
        One-shot bonus per fn_idx that appears in ``available_fn_ids`` for
        the first time in an episode, restricted to actions whose tech-tree
        preconditions include at least one required building.  The bonus fires
        the first time the action is *fully executable* — meaning the tech-tree
        prerequisite building exists, the correct unit type is selected, and the
        action is affordable — not strictly at the moment the prerequisite
        building completes (e.g. ``Build_Barracks_screen`` first becomes
        available when a ``SupplyDepot`` exists *and* an SCV is selected and
        minerals are sufficient).  Selection-only actions (``Move_screen``,
        ``Attack_screen``, basic training) and always-available actions
        (``no_op``, ``select_army``) do not trigger the bonus.  The bonus fires
        once per qualifying fn_idx per episode; each subsequent step where that
        fn_idx appears earns no additional reward.  Default ``0.0`` — opt-in.
        Recommended starting range: ``1.0–10.0`` (much larger than per-step
        shaping terms so the tech-unlock signal is clearly visible to the
        policy).
    new_action_usage_bonus :
        Per-step bonus when the agent actually *issues* a tech-gated fn_idx
        that has already been unlocked this episode (i.e. appeared in
        ``available_fn_ids`` at least once).  The bonus fires for the first
        ``new_action_usage_max_uses`` uses of each qualifying fn_idx per
        episode and is then silenced for the rest of that episode.  This
        complements ``new_action_unlock_bonus``: unlocking a tech earns one
        large reward; then using it consistently earns smaller shaping rewards
        that encourage following through on the investment.  The same fn_idx
        filter applies — only tech-gated (building-prerequisite) actions
        qualify; ``no_op``, selection-only, and always-available actions are
        excluded.  Can be enabled independently of ``new_action_unlock_bonus``.
        Default ``0.0`` — opt-in.  Recommended starting range: ``0.1–2.0``
        (keep smaller than ``new_action_unlock_bonus`` since it fires many
        times).
    new_action_usage_max_uses :
        Maximum number of times per fn_idx per episode that
        ``new_action_usage_bonus`` fires.  After this many uses the bonus is
        silenced for that fn_idx for the remainder of the episode, so the usage
        shaping does not dominate the reward indefinitely.  Default ``50``.
    resource_banking_penalty :
        Per-step penalty proportional to the total excess resources above
        ``mineral_banking_threshold`` and ``gas_banking_threshold`` (issue #372).
        Agents tend to hoard minerals/gas rather than spending them on buildings
        or units; a small negative coefficient nudges them to invest.
        Penalty each step = ``resource_banking_penalty × (max(0, minerals −
        mineral_banking_threshold) + max(0, vespene − gas_banking_threshold))
        × n_ticks``.  Default ``0.0`` — opt-in.  Recommended range:
        ``-0.0001`` to ``-0.001``.
    mineral_banking_threshold :
        Minerals above this level are considered "banked" for the
        ``resource_banking_penalty``.  Default ``300.0``.
    gas_banking_threshold :
        Vespene above this level is considered "banked" for the
        ``resource_banking_penalty``.  Default ``200.0``.
    supply_block_penalty :
        Per-step penalty while supply-blocked — ``food_used >= food_cap`` and
        ``food_cap < 200`` (not yet at the hard cap).  Being capped halts
        production, the most common ladder macro failure.  Default ``0.0`` —
        opt-in.  Recommended range ``-0.05`` to ``-0.5``.
    supply_growth_bonus :
        Bonus per point of ``food_cap`` increase between steps (build supply
        structures or expand — a new base also raises the cap).  Only increases
        are rewarded.  Default ``0.0`` — opt-in.  Recommended ``0.5–3.0``.
    worker_growth_bonus :
        Bonus per point of ``food_workers`` increase (train workers).  Only
        increases are rewarded; pairs with ``idle_worker_penalty``.  Default
        ``0.0`` — opt-in.  Recommended ``0.5–3.0``.
    army_growth_bonus :
        Bonus per point of ``food_army`` increase (produce combat units).  Only
        increases are rewarded — losses are handled by ``unit_loss_penalty``.
        Default ``0.0`` — opt-in.  Recommended ``0.5–3.0``.
    tech_building_bonus :
        One-shot bonus the first time each friendly structure *type* is seen
        this episode (from ``owned_building_names``) — climbing the build tree.
        Fires once per structure type per episode.  Default ``0.0`` — opt-in.
        Recommended ``2.0–10.0``.
    expansion_bonus :
        One-shot bonus each time the friendly town-hall count
        (``townhall_count``) reaches a new episode maximum — i.e. an expansion.
        Counted from currently-visible town halls, so a base built fully
        off-camera may be missed; the running maximum keeps the signal monotonic
        (the starting base is never rewarded).  Default ``0.0`` — opt-in.
        Recommended ``5.0–25.0``.
    scout_bonus :
        Bonus proportional to the increase in ``minimap_explored_frac`` between
        steps — revealing previously-unseen map.  Reward each step =
        ``scout_bonus × max(0, explored_frac − prev_explored_frac)`` with the
        fractions in ``[0, 1]``.  Captures scouting that screen-local
        ``move_exploration_bonus`` cannot.  Default ``0.0`` — opt-in.  The
        per-step delta is small, so the weight is large; recommended ``5.0–50.0``.
    build_train_bonus :
        Per-step bonus when the agent issues a "build" or "train" category
        action (``FN_IDX_TO_CATEGORY[fn_idx] in {"build", "train"}``).  Build
        actions include constructing supply depots, barracks, refineries, and
        other structures; train actions include producing SCVs, marines, and
        other units.  Encourages the agent to invest its mineral income into
        production rather than cycling through move/attack commands — the most
        common failure mode when the agent is stuck in the early game.  Fires
        once per env step when any build or train fn_idx is issued; scaled by
        ``n_ticks`` for consistency with other per-step terms.  Default
        ``0.0`` — opt-in.  Recommended starting range: ``0.2–1.0`` (keep
        smaller than ``supply_growth_bonus`` / ``worker_growth_bonus`` so the
        shaping term does not dominate outcome-driven signals).
    build_repeat_penalty :
        Per-step penalty when the agent issues the same build fn_idx on two
        consecutive env steps.  Discourages placement-spam loops where the
        agent repeatedly tries to build at an invalid location and the
        "Can't find placement location" rejection earns no penalty from the
        reward function.  The penalty fires on the second and later consecutive
        repetitions; the first issuance of any build fn_idx is always
        unpenalised.  Any intervening non-build action (move, train, no-op,
        select) resets the counter.  Combined with ``build_train_bonus``, a
        weight of ``-build_train_bonus`` makes the net reward exactly zero for
        a repeated build, breaking the positive feedback loop while leaving the
        first issuance fully rewarded.  Default ``0.0`` — opt-in.  Recommended
        range: ``-0.3`` to ``-1.0``.
    """

    score_weight: float = 1.0
    win_bonus: float = 100.0
    loss_penalty: float = -100.0
    step_penalty: float = -0.001
    idle_penalty: float = 0.0
    idle_worker_penalty: float = 0.0
    idle_bonus: float = 0.0
    attack_move_bonus: float = 0.0
    click_attack_bonus: float = 0.0
    click_attack_cooldown_steps: int = 8
    economy_weight: float = 0.0
    move_exploration_bonus: float = 0.01
    move_exploration_grid_size: int = 8
    move_exploration_decay_steps: int = 50
    move_repeat_penalty: float = -0.02
    move_self_penalty: float = -0.01
    attack_friendly_penalty: float = -5.0
    unit_loss_penalty: float = 0.0
    damage_taken_penalty: float = 0.0
    passive_under_fire_penalty: float = 0.0
    small_selection_bonus: float = 0.0
    attack_bonus: float = 0.0
    early_random_action_bonus: float = 0.0
    early_random_action_window_steps: int = 250
    new_action_unlock_bonus: float = 0.0
    new_action_usage_bonus: float = 0.0
    new_action_usage_max_uses: int = 50
    resource_banking_penalty: float = 0.0
    mineral_banking_threshold: float = 300.0
    gas_banking_threshold: float = 200.0
    supply_block_penalty: float = 0.0
    supply_growth_bonus: float = 0.0
    worker_growth_bonus: float = 0.0
    army_growth_bonus: float = 0.0
    tech_building_bonus: float = 0.0
    expansion_bonus: float = 0.0
    scout_bonus: float = 0.0
    build_train_bonus: float = 0.0
    build_repeat_penalty: float = 0.0

    @classmethod
    def from_yaml(cls, path: str) -> SC2RewardConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid_keys = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - valid_keys
        if unknown:
            raise ValueError(f"{path}: unknown reward config keys: {sorted(unknown)}\nValid keys: {sorted(valid_keys)}")
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
        ``idle_worker_count`` — idle workers this step (issue #358)
        ``player_outcome`` — None / +1 / -1 (only set on the final step)
        ``action_fn_idx`` — fn_idx of the action issued this step
        ``action_target_x`` / ``action_target_y`` — normalised [0, 1] screen
            target coordinates of the action (used by ``attack_move_bonus``
            and ``click_attack_bonus`` to classify the attack type)
        ``prev_move_target_x`` / ``prev_move_target_y`` — previous move target
            in [0, 1], or None if no previous move target is known (used by
            ``move_repeat_penalty`` only)
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
        ``food_used`` / ``food_cap`` — supply used / cap, for the supply-block
            penalty and supply-growth bonus
        ``food_workers`` / ``food_army`` — worker / army supply, for the
            worker-growth and army-growth bonuses
        ``owned_building_names`` — cumulative set of friendly structure type
            names seen this episode, for the tech-building bonus
        ``townhall_count`` — number of currently-visible friendly town halls,
            for the expansion bonus
        ``minimap_explored_frac`` — fraction of the minimap explored, for the
            scout bonus
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

    # fn_ids whose PRECONDITIONS include at least one required building —
    # the only actions for which new_action_unlock_bonus fires.
    _TECH_GATED_FN_IDS: frozenset[int] = frozenset(
        fn_idx for fn_idx, prec in PRECONDITIONS.items() if prec.required_buildings
    )

    def __init__(self, config: SC2RewardConfig) -> None:
        self.config = config
        self._last_click_x: float | None = None
        self._last_click_y: float | None = None
        self._last_click_step: int = -1
        self._last_non_noop_fn_idx: int | None = None
        self._last_non_noop_target_x: float = 0.5
        self._last_non_noop_target_y: float = 0.5
        self._step_count: int = 0
        # cell -> env step on which the centroid was last seen in that cell.
        self._visited_unit_cells: dict[tuple[int, int], int] = {}
        self._seen_action_fns: set[int] = set()
        # tech-gated fn_ids seen so far this episode (for new_action_unlock_bonus
        # and new_action_usage_bonus).
        self._unlocked_tech_fn_ids: set[int] = set()
        # fn_idx -> number of times issued after being unlocked this episode.
        self._new_action_usage_counts: dict[int, int] = {}
        # Progression trackers: previous supply / worker / army / explored
        # values (``None`` until the first step so the baseline is never
        # rewarded), cumulative structure types seen, and the running max town
        # hall count for the one-shot tech-building / expansion bonuses.
        self._prev_food_cap: float | None = None
        self._prev_food_workers: float | None = None
        self._prev_food_army: float | None = None
        self._prev_explored_frac: float | None = None
        self._seen_structures: set[str] = set()
        self._max_townhalls: int | None = None
        self._last_build_fn_idx: int | None = None

    def reset(self) -> None:
        """Clear per-episode state at the start of a new episode."""
        self._last_click_x = None
        self._last_click_y = None
        self._last_click_step = -1
        self._last_non_noop_fn_idx = None
        self._last_non_noop_target_x = 0.5
        self._last_non_noop_target_y = 0.5
        self._step_count = 0
        self._visited_unit_cells = {}
        self._seen_action_fns = set()
        self._unlocked_tech_fn_ids = set()
        self._new_action_usage_counts = {}
        self._prev_food_cap = None
        self._prev_food_workers = None
        self._prev_food_army = None
        self._prev_explored_frac = None
        self._seen_structures = set()
        self._max_townhalls = None
        self._last_build_fn_idx = None

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
        ``idle_worker_penalty``, ``idle_bonus``, ``move_exploration``,
        ``move_repeat_penalty``, ``move_self_penalty``, ``attack_move_bonus``,
        ``click_attack_bonus``, ``attack_bonus``, ``attack_friendly_penalty``,
        ``early_random_action``, ``new_action_unlock``, ``new_action_usage``,
        ``build_train``, ``build_repeat_penalty``, ``unit_loss``,
        ``damage_taken``, ``passive_under_fire``, ``small_selection``,
        ``resource_banking``, ``supply_block``, ``supply_growth``,
        ``worker_growth``, ``army_growth``, ``tech_building``, ``expansion``,
        ``scout_explore``, ``step_penalty`` and ``terminal`` separately.
        """
        cfg = self.config
        components: dict[str, float] = {}

        self._step_count += 1
        raw_fn_idx = int(info.get("action_fn_idx", 0))
        raw_target_x = float(info.get("action_target_x", 0.5))
        raw_target_y = float(info.get("action_target_y", 0.5))
        if raw_fn_idx != 0:
            self._last_non_noop_fn_idx = raw_fn_idx
            self._last_non_noop_target_x = raw_target_x
            self._last_non_noop_target_y = raw_target_y
        carried_attack_noop = raw_fn_idx == 0 and self._last_non_noop_fn_idx == 3
        combat_fn_idx = 3 if carried_attack_noop else raw_fn_idx
        combat_target_x = self._last_non_noop_target_x if carried_attack_noop else raw_target_x
        combat_target_y = self._last_non_noop_target_y if carried_attack_noop else raw_target_y

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
            components["economy"] = float(cfg.economy_weight * ((curr_min - prev_min) + (curr_vesp - prev_vesp)))
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

        # Idle worker penalty (issue #358): penalise each idle worker.
        idle_worker_pen = 0.0
        if cfg.idle_worker_penalty != 0.0:
            idle_workers = float(info.get("idle_worker_count", 0.0))
            if idle_workers > 0:
                idle_worker_pen = cfg.idle_worker_penalty * idle_workers * n_ticks
        components["idle_worker_penalty"] = float(idle_worker_pen)

        # Idle bonus (issue #127): reward standing still when units are in
        # effective attack range of an enemy. If the client provides
        # self_attack_range_px, use that unit-aware threshold; otherwise use a
        # conservative screen-size-scaled fallback.
        idle_bonus = 0.0
        if cfg.idle_bonus != 0.0 and combat_fn_idx == 0:
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
                idle_gate_px = max(0.0, max_attack_range_px) * self._IDLE_RANGE_INSIDE_FRAC
                if dist <= idle_gate_px:
                    idle_bonus = cfg.idle_bonus * n_ticks
        components["idle_bonus"] = float(idle_bonus)

        # Early random-action encouragement: reward unseen non-no-op fn_idx
        # choices in the first N episode steps.
        early_random_action = 0.0
        if cfg.early_random_action_bonus != 0.0:
            early_window_steps = max(0, int(cfg.early_random_action_window_steps))
            if self._step_count <= early_window_steps:
                current_fn_idx = int(info.get("action_fn_idx", 0))
                if current_fn_idx != 0 and current_fn_idx not in self._seen_action_fns:
                    early_random_action = cfg.early_random_action_bonus * n_ticks
        current_fn_idx = int(info.get("action_fn_idx", 0))
        if current_fn_idx != 0:
            self._seen_action_fns.add(current_fn_idx)
        components["early_random_action"] = float(early_random_action)

        # New tech-tree unlock bonus: reward once per qualifying fn_idx that
        # appears for the first time this episode.  Only fn_ids with at least
        # one required_building in PRECONDITIONS are eligible (selection-only
        # and always-available actions are excluded).
        # _unlocked_tech_fn_ids is also needed by new_action_usage_bonus, so
        # update it whenever either bonus is active.
        new_action_unlock = 0.0
        if cfg.new_action_unlock_bonus != 0.0 or cfg.new_action_usage_bonus != 0.0:
            available = info.get("available_fn_ids") or set()
            tech_available = available & self._TECH_GATED_FN_IDS
            newly_unlocked = tech_available - self._unlocked_tech_fn_ids
            if newly_unlocked and cfg.new_action_unlock_bonus != 0.0:
                new_action_unlock = cfg.new_action_unlock_bonus * len(newly_unlocked)
            self._unlocked_tech_fn_ids |= tech_available
        components["new_action_unlock"] = float(new_action_unlock)

        # New action usage bonus: reward the agent for actually issuing a
        # tech-gated action up to new_action_usage_max_uses times per episode.
        new_action_usage = 0.0
        if cfg.new_action_usage_bonus != 0.0:
            used_fn_idx = int(info.get("action_fn_idx", 0))
            if used_fn_idx != 0 and used_fn_idx in self._unlocked_tech_fn_ids:
                max_uses = max(0, int(cfg.new_action_usage_max_uses))
                count = self._new_action_usage_counts.get(used_fn_idx, 0)
                if count < max_uses:
                    new_action_usage = cfg.new_action_usage_bonus * n_ticks
                    self._new_action_usage_counts[used_fn_idx] = count + 1
        components["new_action_usage"] = float(new_action_usage)

        # Build/train bonus (issue #416): reward issuing build or train actions.
        build_train = 0.0
        if cfg.build_train_bonus != 0.0:
            bt_fn_idx = int(info.get("action_fn_idx", 0))
            if FN_IDX_TO_CATEGORY.get(bt_fn_idx) in ("build", "train"):
                build_train = cfg.build_train_bonus * n_ticks
        components["build_train"] = float(build_train)

        # Build-repeat penalty: discourage consecutive same-fn_idx build spam.
        # The first issuance of any build fn_idx is unpenalised; any subsequent
        # repetition of the *same* build fn_idx without an intervening non-build
        # action fires the penalty.  Catches "Can't find placement location"
        # loops where failed placements earn no consequence from the reward.
        build_repeat = 0.0
        if FN_IDX_TO_CATEGORY.get(raw_fn_idx) == "build":
            if cfg.build_repeat_penalty != 0.0 and self._last_build_fn_idx == raw_fn_idx:
                build_repeat = cfg.build_repeat_penalty * n_ticks
            self._last_build_fn_idx = raw_fn_idx
        else:
            self._last_build_fn_idx = None
        components["build_repeat_penalty"] = float(build_repeat)

        self_count = float(info.get("screen_self_count", 0.0))
        newly_visited_unit_cell = False
        if self_count > 0:
            grid_size = max(1, int(cfg.move_exploration_grid_size))
            decay_steps = int(cfg.move_exploration_decay_steps)
            screen_size = max(1.0, float(info.get("screen_size", 64)))
            cx = float(info.get("screen_self_cx", 0.0))
            cy = float(info.get("screen_self_cy", 0.0))
            cell_x = min(grid_size - 1, int(cx / screen_size * grid_size))
            cell_y = min(grid_size - 1, int(cy / screen_size * grid_size))
            cell = (cell_x, cell_y)
            last_visit = self._visited_unit_cells.get(cell)
            # A cell counts as "newly explored" when the centroid has never
            # been in it, or when it was last seen there more than
            # decay_steps env steps ago (the area went stale and the units
            # have now returned).  A stationary centroid refreshes last_visit
            # every step, so it never re-triggers — preserving the
            # anti-stationary-spam guarantee.  decay_steps <= 0 disables
            # expiry entirely (permanent once-per-episode behaviour).
            if last_visit is None:
                newly_visited_unit_cell = True
            elif decay_steps > 0 and (self._step_count - last_visit) > decay_steps:
                newly_visited_unit_cell = True
            self._visited_unit_cells[cell] = self._step_count

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
                if dist_prev < self._MOVE_MIN_MEANINGFUL_FRAC:
                    if cfg.move_repeat_penalty != 0.0:
                        move_repeat_penalty = cfg.move_repeat_penalty * n_ticks

            if cfg.move_exploration_bonus != 0.0 and newly_visited_unit_cell:
                move_exploration = cfg.move_exploration_bonus * n_ticks

            if cfg.move_self_penalty != 0.0:
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
        if cfg.passive_under_fire_penalty != 0.0 and combat_fn_idx != 3:
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
        if combat_fn_idx == 3:
            screen_size = float(info.get("screen_size", 64))
            # Use (screen_size - 1) to match the client's pixel conversion:
            # x_screen = int(clip(norm, 0, 1) * (screen_size - 1))
            # Centroids from games.sc2.client.SC2Client._centroid() are raw
            # pixel indices in [0, screen_size-1].
            scale = max(1.0, screen_size - 1.0)
            enemy_count = info.get("screen_enemy_count", 0.0)
            tx_norm = combat_target_x
            ty_norm = combat_target_y
            tx_px = tx_norm * scale
            ty_px = ty_norm * scale
            ecx = float(info.get("screen_enemy_cx", 0.0))
            ecy = float(info.get("screen_enemy_cy", 0.0))
            click_radius_px = self._CLICK_ATTACK_RADIUS_FRAC * screen_size

            # "Click to attack": target pixel is within click_radius of enemy
            # centroid and at least one enemy is visible.
            on_enemy = enemy_count > 0 and ((tx_px - ecx) ** 2 + (ty_px - ecy) ** 2) ** 0.5 <= click_radius_px

            if on_enemy and cfg.click_attack_bonus != 0.0:
                # Check cooldown: only blocked when the agent switches to a
                # *different* target quickly (same target always passes).
                steps_since = self._step_count - self._last_click_step
                if self._last_click_x is None:
                    same_target = True
                else:
                    dx = tx_px - self._last_click_x
                    dy = ty_px - self._last_click_y  # _last_click_x/_last_click_y are set together
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

        # General attack bonus: fires on any Attack_screen regardless of target.
        attack_bonus = 0.0
        if cfg.attack_bonus != 0.0 and combat_fn_idx == 3:
            attack_bonus = cfg.attack_bonus * n_ticks
        components["attack_bonus"] = float(attack_bonus)

        # Friendly-fire penalty: Attack_screen aimed at own units.
        attack_friendly_penalty = 0.0
        if cfg.attack_friendly_penalty != 0.0 and combat_fn_idx == 3:
            self_count = float(info.get("screen_self_count", 0.0))
            if self_count > 0:
                screen_size = max(1.0, float(info.get("screen_size", 64)))
                # Use (screen_size - 1) to match client pixel conversion.
                scale = max(1.0, screen_size - 1.0)
                scx = float(info.get("screen_self_cx", 0.0))
                scy = float(info.get("screen_self_cy", 0.0))
                tx_norm = combat_target_x
                ty_norm = combat_target_y
                tx_px = tx_norm * scale
                ty_px = ty_norm * scale
                dx = tx_px - scx
                dy = ty_px - scy
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= self._ATTACK_SELF_RADIUS_FRAC * screen_size:
                    attack_friendly_penalty = cfg.attack_friendly_penalty * n_ticks
        components["attack_friendly_penalty"] = float(attack_friendly_penalty)

        # Resource banking penalty (issue #372): penalise hoarding excess minerals/gas.
        resource_banking = 0.0
        if cfg.resource_banking_penalty != 0.0:
            curr_min = float(info.get("minerals", 0.0))
            curr_vesp = float(info.get("vespene", 0.0))
            excess_min = max(0.0, curr_min - cfg.mineral_banking_threshold)
            excess_vesp = max(0.0, curr_vesp - cfg.gas_banking_threshold)
            resource_banking = cfg.resource_banking_penalty * (excess_min + excess_vesp) * n_ticks
        components["resource_banking"] = float(resource_banking)

        # --- Progression shaping (macro economy / tech / map control) ---
        food_used = float(info.get("food_used", 0.0))
        food_cap = float(info.get("food_cap", 0.0))
        food_workers = float(info.get("food_workers", 0.0))
        food_army = float(info.get("food_army", 0.0))

        # Supply-block penalty: production stalls when capped (but not at 200).
        supply_block = 0.0
        if cfg.supply_block_penalty != 0.0 and food_cap > 0.0 and food_cap < 200.0 and food_used >= food_cap:
            supply_block = cfg.supply_block_penalty * n_ticks
        components["supply_block"] = float(supply_block)

        # Supply-growth bonus: reward raising the supply cap (depots / expansions).
        supply_growth = 0.0
        if cfg.supply_growth_bonus != 0.0 and self._prev_food_cap is not None:
            supply_growth = cfg.supply_growth_bonus * max(0.0, food_cap - self._prev_food_cap)
        self._prev_food_cap = food_cap
        components["supply_growth"] = float(supply_growth)

        # Worker-growth bonus: reward training workers (economy).
        worker_growth = 0.0
        if cfg.worker_growth_bonus != 0.0 and self._prev_food_workers is not None:
            worker_growth = cfg.worker_growth_bonus * max(0.0, food_workers - self._prev_food_workers)
        self._prev_food_workers = food_workers
        components["worker_growth"] = float(worker_growth)

        # Army-growth bonus: reward producing combat units.
        army_growth = 0.0
        if cfg.army_growth_bonus != 0.0 and self._prev_food_army is not None:
            army_growth = cfg.army_growth_bonus * max(0.0, food_army - self._prev_food_army)
        self._prev_food_army = food_army
        components["army_growth"] = float(army_growth)

        # Tech-building bonus: one-shot per newly-seen friendly structure type.
        tech_building = 0.0
        if cfg.tech_building_bonus != 0.0:
            owned = info.get("owned_building_names") or set()
            newly_seen = set(owned) - self._seen_structures
            if newly_seen:
                tech_building = cfg.tech_building_bonus * len(newly_seen)
            self._seen_structures |= set(owned)
        components["tech_building"] = float(tech_building)

        # Expansion bonus: one-shot when town-hall count reaches a new max.
        expansion = 0.0
        townhall_count = int(info.get("townhall_count", 0))
        if cfg.expansion_bonus != 0.0 and self._max_townhalls is not None and townhall_count > self._max_townhalls:
            expansion = cfg.expansion_bonus * (townhall_count - self._max_townhalls)
        if self._max_townhalls is None:
            self._max_townhalls = townhall_count
        else:
            self._max_townhalls = max(self._max_townhalls, townhall_count)
        components["expansion"] = float(expansion)

        # Scout bonus: reward revealing previously-unseen map (explored delta).
        # Emitted as "scout_explore" to avoid colliding with the belief
        # system's intrinsic "scout" component accumulated in SC2Env.
        scout = 0.0
        explored_frac = float(info.get("minimap_explored_frac", 0.0))
        if cfg.scout_bonus != 0.0 and self._prev_explored_frac is not None:
            scout = cfg.scout_bonus * max(0.0, explored_frac - self._prev_explored_frac)
        self._prev_explored_frac = explored_frac
        components["scout_explore"] = float(scout)

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
