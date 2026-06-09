"""Thin wrapper around ``pysc2.env.sc2_env.SC2Env``.

Provides ``reset()`` / ``step()`` / ``close()`` returning the flat
``np.ndarray`` observation expected by :class:`games.sc2.env.SC2Env`,
plus an info dict with the raw scalars the reward calculator needs.

PySC2 import is lazy: importing this module does not pull pysc2 in, so
unit tests can mock the client without installing the SC2 binary.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from framework.obs_spec import ObsSpec
from games.sc2.actions import FUNCTION_IDS, action_to_function_call, fn_ids_for_race
from games.sc2.obs_spec import get_spec
from games.sc2.tech_tree import (
    GEYSER_NAMES,
    PRECONDITIONS,
    STRUCTURE_NAMES,
    TOWNHALL_NAMES,
    WORKER_NAMES,
    SelectionReq,
    fn_idx_satisfied,
)

# fn_ids that build a vespene-extraction structure on top of a geyser.
# 22 = Build_Refinery_screen (Terran), 51 = Build_Assimilator_screen (Protoss),
# 82 = Build_Extractor_screen (Zerg). These three must be targeted on a visible
# vespene geyser; the client snaps their target coordinates onto the nearest
# one before calling PySC2 (issue #402).
REFINERY_FN_IDS: frozenset[int] = frozenset({22, 51, 82})

logger = logging.getLogger(__name__)

# Reverse mapping from FUNCTION_IDS name → our fn_idx key.
# Used in _timestep_to_obs_info() to convert PySC2 available_actions IDs
# into our fn_idx values (0-5); built once at import time.
_FN_NAME_TO_IDX: dict[str, int] = {v: k for k, v in FUNCTION_IDS.items()}

# ---------------------------------------------------------------------------
# Spatial feature layer normalisation scales
# ---------------------------------------------------------------------------
# Values taken from PySC2 feature layer documentation.  Unknown layers default
# to 1.0 (no normalisation), so 0–max values map to ~[0, 1].
_LAYER_SCALE: dict[str, float] = {
    "player_relative": 4.0,
    "selected": 1.0,
    "unit_type": 1917.0,
    "height_map": 255.0,
    "unit_hit_points": 255.0,
    "unit_shields": 255.0,
    "unit_density": 16.0,
    "unit_density_aa": 255.0,
    "effects": 16.0,
    "visibility_map": 2.0,
    "unit_energy": 255.0,
    "creep": 1.0,
    "power": 1.0,
    "pathable": 1.0,
    "buildable": 1.0,
}

# Approximate SC2 weapon ranges in game units for common combat units.
# PySC2 exposes unit IDs via pysc2.lib.units but does not expose weapon ranges
# directly, so we keep a curated name->range table and map it to unit IDs
# lazily at runtime.
#
# Reference for checking/updating ranges:
# - Blizzard s2client protocol unit weapon schema:
#   https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/data.proto
# - Community unit-stat tables:
#   https://liquipedia.net/starcraft2/Unit_Statistics_(Legacy_of_the_Void)
_UNIT_ATTACK_RANGE_GU: dict[str, float] = {
    "Marine": 5.0,
    "SCV": 0.1,
    "Marauder": 6.0,
    "Reaper": 5.0,
    "Ghost": 6.0,
    "Hellion": 5.0,
    "Cyclone": 7.0,
    "WidowMine": 5.0,
    "SiegeTank": 7.0,
    "SiegeTankSieged": 13.0,
    "Thor": 7.0,
    "ThorAP": 7.0,
    "VikingAssault": 6.0,
    "Banshee": 6.0,
    "Battlecruiser": 6.0,
    "Liberator": 5.0,
    "LiberatorAG": 10.0,
    "Zealot": 0.1,
    "Adept": 4.0,
    "Sentry": 5.0,
    "HighTemplar": 6.0,
    "DarkTemplar": 0.1,
    "Immortal": 6.0,
    "Colossus": 7.0,
    "Archon": 3.0,
    "Phoenix": 5.0,
    "VoidRay": 6.0,
    "Carrier": 8.0,
    "Tempest": 14.0,
    "Oracle": 4.0,
    "Mothership": 7.0,
    "Zergling": 0.1,
    "Drone": 0.1,
    "Probe": 0.1,
    "Stalker": 6.0,
    "Roach": 4.0,
    "Mutalisk": 3.0,
    "Queen": 5.0,
    "Hydralisk": 6.0,
    "Ravager": 6.0,
    "Baneling": 0.1,
    "Infestor": 0.1,
    "Ultralisk": 1.0,
    "Corruptor": 6.0,
    "BroodLord": 10.0,
    "SpineCrawler": 7.0,
    "SporeCrawler": 7.0,
    "LurkerMP": 8.0,
    "LurkerMPBurrowed": 10.0,
}
_MARINE_RANGE_GU: float = 5.0
_MARINE_RANGE_PX_AT_64: float = 20.0

# Lazy cache: maps PySC2 native function ID → our fn_idx (0-5 in FUNCTION_IDS).
# Built on first use when pysc2 is available.
_pysc2_id_to_fn_idx: dict[int, int] | None = None

_SAVE_REPLAY_TIMEOUT_S: float = 5.0


# ---------------------------------------------------------------------------
# PySC2 unit-row column indices
# ---------------------------------------------------------------------------
# Mirrors pysc2.lib.features.FeatureUnit (IntEnum).  Used for feature_units
# rows AND for single_select / multi_select rows, which share the same
# leading columns (unit_type at 0, alliance at 1).  Named constants keep
# every unit-row access self-documenting and in one place — changing the
# layout only requires updating this class.


class _FU:
    unit_type = 0
    alliance = 1
    health = 2
    shield = 3
    x = 12
    y = 13
    weapon_cooldown = 25


# ---------------------------------------------------------------------------
# Lazy PySC2 field-name helpers
# ---------------------------------------------------------------------------
# PySC2 exposes structured field names via NamedNumpyArray descriptors.
# We source these lazily to avoid a hard pysc2 import (unit tests run without
# it) and fall back to our hardcoded lists when pysc2 is not installed.
# Caches are populated once on first use and then reused.

_score_field_names_cache: tuple[str, ...] | None = None
_player_field_names_cache: tuple[str, ...] | None = None


def _get_score_field_names() -> tuple[str, ...]:
    """Return ``score_cumulative`` field names, sourced from PySC2 if available.

    The PySC2 ``ScoreCumulative`` namedtuple defines these fields; we rename
    ``score`` to ``score_total`` to avoid confusion with the reward signal.
    When pysc2 is not installed (e.g. in unit tests) we fall back to our
    hardcoded list, which is kept in sync with the PySC2 ordering.
    If a future PySC2 version adds or renames fields, the live path picks up
    the change automatically; the fallback path would then need a manual
    update to stay in sync.
    """
    global _score_field_names_cache
    if _score_field_names_cache is None:
        try:
            from pysc2.lib import features as pysc2_features  # type: ignore[import-untyped]

            raw = pysc2_features.ScoreCumulative._fields
            _score_field_names_cache = tuple("score_total" if f == "score" else f for f in raw)
        except (ImportError, AttributeError):
            _score_field_names_cache = (
                "score_total",
                "idle_production_time",
                "idle_worker_time",
                "total_value_units",
                "total_value_structures",
                "killed_value_units",
                "killed_value_structures",
                "collected_minerals",
                "collected_vespene",
                "collection_rate_minerals",
                "collection_rate_vespene",
                "spent_minerals",
                "spent_vespene",
            )
    return _score_field_names_cache


def _get_player_field_names() -> tuple[str, ...]:
    """Return ``player`` vector field names, sourced from PySC2 if available.

    Excludes ``player_id`` (fixed per game; only meaningful in multi-agent
    settings).  When pysc2 is not installed we fall back to a hardcoded list
    matching the current PySC2 ``Player`` namedtuple minus ``player_id``.
    If a future PySC2 version adds new player fields, the live path picks them
    up; the fallback list would need a manual update in that case.
    """
    global _player_field_names_cache
    if _player_field_names_cache is None:
        try:
            from pysc2.lib import features as pysc2_features  # type: ignore[import-untyped]

            _player_field_names_cache = tuple(f for f in pysc2_features.Player._fields if f != "player_id")
        except (ImportError, AttributeError):
            _player_field_names_cache = (
                "minerals",
                "vespene",
                "food_used",
                "food_cap",
                "army_count",
                "idle_worker_count",
                "warp_gate_count",
                "larva_count",
                "food_workers",
                "food_army",
            )
    return _player_field_names_cache


def _get_pysc2_id_to_fn_idx() -> dict[int, int]:
    """Build and cache a mapping from PySC2 native function ID → our fn_idx.

    Imports ``pysc2.lib.actions`` lazily so that callers without PySC2
    installed (unit tests) can import this module without errors.
    """
    global _pysc2_id_to_fn_idx
    if _pysc2_id_to_fn_idx is None:
        try:
            from pysc2.lib import actions as pysc2_actions  # type: ignore[import-untyped]

            _pysc2_id_to_fn_idx = {}
            for fn_idx, name in FUNCTION_IDS.items():
                fn_obj = getattr(pysc2_actions.FUNCTIONS, name, None)
                if fn_obj is not None:
                    _pysc2_id_to_fn_idx[int(fn_obj.id)] = fn_idx
        except Exception:
            _pysc2_id_to_fn_idx = {}
    return _pysc2_id_to_fn_idx


class SC2Client:
    """Manages a ``pysc2.env.sc2_env.SC2Env`` session.

    Parameters
    ----------
    map_name :
        PySC2 map name (e.g. ``MoveToBeacon`` or ``Simple64``).
    step_mul :
        Game-tick multiplier per env step (default 8 ≈ 0.5 sec real-time).
    screen_size :
        Square feature-screen resolution (default 64).
    minimap_size :
        Square feature-minimap resolution (default 64).
    agent_race :
        Race string (``"random"``, ``"protoss"``, ``"terran"``, ``"zerg"``).
    bot_difficulty :
        Bot difficulty for 1v1 maps; ignored for minigames.
    visualize :
        If True, render the PySC2 visualizer window.
    realtime :
        If True, synchronise env steps with wall-clock time so the game
        runs at natural game speed rather than as fast as possible.
        Useful for evaluation / watch sessions.
    play_mode :
        If True, set up a Human + Agent session instead of Agent (+ Bot).
        The human plays via the standard SC2 UI; the agent acts via PySC2.
    self_play :
        If True, set up two Agent players for self-play.  An
        ``opponent_policy`` must be provided to control the second agent.
        The primary agent's observation and reward are returned from
        ``reset()`` / ``step()``; the opponent acts from its own
        observation each step, mirroring AlphaGo-style self-play.
    opponent_policy :
        Callable ``(obs) -> action`` used to control the second agent when
        *self_play* is True.  Ignored when *self_play* is False.
    extreme_random_run_count :
        Number of episodes at the start of *this client's* lifetime during
        which every ``step()`` call replaces the incoming policy action with a
        random valid action.  The counter ``_episodes_started`` is
        per-instance, so the budget is **per-worker** (parallel evaluation
        spawns one client per worker) and **per-candidate** (population-based
        training such as CMA-ES / genetic creates a fresh client per
        candidate).  When derived from ``n_sims * fraction`` in the adapter
        the effective total across all workers/candidates scales accordingly —
        document or account for this when choosing the fraction.
    """

    def __init__(
        self,
        map_name: str,
        step_mul: int = 1,
        screen_size: int = 64,
        minimap_size: int = 64,
        agent_race: str = "random",
        bot_difficulty: str = "very_easy",
        visualize: bool = False,
        realtime: bool = False,
        screen_layers: list[str] | None = None,
        minimap_layers: list[str] | None = None,
        play_mode: bool = False,
        obs_spec_preset: str | None = None,
        store_minimap_vis: bool = False,
        extreme_random_run_count: int = 0,
        self_play: bool = False,
        opponent_policy: Any = None,
    ) -> None:
        self._map_name = map_name
        self._step_mul = step_mul
        self._screen_size = screen_size
        self._minimap_size = minimap_size
        self._agent_race = agent_race
        self._bot_difficulty = bot_difficulty
        self._visualize = visualize
        self._realtime = realtime
        self._play_mode = play_mode
        self._self_play = self_play
        self._opponent_policy = opponent_policy
        self._opponent_obs: np.ndarray | None = None
        self._store_minimap_vis = store_minimap_vis
        self._extreme_random_run_count = max(0, int(extreme_random_run_count))
        self._episodes_started = 0
        self._rng = np.random.default_rng()
        self._screen_layers: list[str] = list(screen_layers or [])
        self._minimap_layers: list[str] = list(minimap_layers or [])
        self._sc2_env: Any = None
        self._is_ladder = self._detect_ladder(map_name)
        self._obs_spec_preset = obs_spec_preset
        # Spec drives flat-vector assembly — feature_block extractors fill a
        # name-indexed dict from the timestep, then __call__-time we project
        # onto self._spec.names to produce the flat ndarray.
        self._spec: ObsSpec = get_spec(map_name, preset=obs_spec_preset)
        self._obs_names = self._spec.names
        # Preset-gating flags: skip expensive feature extractors whose output
        # is not present in the active obs spec.  Checked at __init__ time so
        # there is zero per-step branching overhead beyond a single bool test.
        _obs_name_set = frozenset(self._obs_names)
        # Ladder and rich presets include features absent from minigame.
        self._use_ladder_obs: bool = "food_workers" in _obs_name_set
        # Rich preset adds per-unit-type counts, quadrants, top-k enemies, etc.
        self._use_rich_obs: bool = "screen_self_NE_count" in _obs_name_set
        self._cumulative_score: float = 0.0
        self._explored_mask: np.ndarray | None = None
        self._available_actions: set[int] | None = None
        # Internal fn_idx mask — race ∩ PySC2 ∩ tech-tree ∩ selection.
        # Single source of truth: extreme-random sampling, policy masks
        # (via info["available_fn_ids"]), and the deferred-action resolver
        # all read from this.
        self._available_fn_ids: set[int] | None = None
        self._selected_count: float = 0.0
        self._last_fn_idx: int = 0  # for last_fn_* one-hot in rich preset
        # Lookup table for unit-type ids → label, populated lazily so unit
        # tests don't import pysc2.lib.units at module load.
        self._unit_type_id_to_name: dict[int, str] | None = None
        # Lookup table for unit-type ids → race label ("terran"/"protoss"/"zerg").
        # Used to infer a race-consistent available-fn mask when PySC2
        # available_actions mapping is missing or overly broad.
        self._unit_type_id_to_race: dict[int, str] | None = None
        # Lookup table for unit-type ids → attack range (game units), used by
        # self_attack_range_px for idle-bonus gating.
        self._unit_type_id_to_attack_range_gu: dict[int, float] | None = None
        # Tech-tree state cached from the latest timestep.
        self._owned_buildings: frozenset[str] = frozenset()
        # Accumulates every building seen this episode so that buildings that
        # scroll off-screen don't vanish from the tech-tree mask (issue #356 H2).
        self._owned_buildings_seen: frozenset[str] = frozenset()
        self._completed_upgrades: frozenset[str] = frozenset()
        # Current resource counts cached from the latest timestep (issue #357).
        self._minerals: float = 0.0
        self._vespene: float = 0.0
        # Cache for _compute_available_fn_ids() — skip the per-fn_idx tech-tree
        # loop when owned_buildings / completed_upgrades / selected_unit_types
        # and the candidate set are all unchanged since last step (issue #356 H3).
        self._fn_ids_cache_keys: tuple | None = None
        self._fn_ids_cache_result: set[int] = set()
        # Currently-selected unit-type name (None when nothing selected or
        # the selection is mixed across types).
        # Set of currently-selected unit-type names.  Multi-type
        # selections (e.g. ``select_army`` on a mixed Marine+Marauder
        # army) keep both names so ``ANY_UNIT`` actions and ``OF_TYPE``
        # actions whose target matches either type stay satisfied.
        self._selected_unit_types: frozenset[str] = frozenset()
        # Cached screen (x, y) per friendly unit-type name, populated from
        # feature_units each step.  The deferred-action resolver uses this
        # to issue select_point on the right worker/building/unit when the
        # policy's chosen action requires a different selection.
        self._screen_xy_by_unit_type: dict[str, tuple[int, int]] = {}
        # Visible neutral vespene-geyser screen positions, refreshed every step
        # from feature_units. The Build_Refinery/Assimilator/Extractor snap
        # (issue #402) picks the nearest entry to the policy's requested
        # target. Geysers that already have a refinery built on them disappear
        # from the neutral list (the friendly refinery takes their place), so
        # this implicitly excludes already-occupied geysers.
        self._geyser_screen_positions: list[tuple[int, int]] = []
        # 1-slot FIFO for the deferred-action queue. When the resolver auto-
        # emits a selection this step, the original action is stored here
        # and replayed on the next step().
        self._deferred_action: np.ndarray | None = None
        # Flags set each step so env.py can update action counts correctly.
        # last_was_deferred: True when the deferred queue fired (the fn_idx
        #   was already counted at the request step; env must not count again).
        # last_was_extreme_random: True when the extreme-random phase replaced
        #   the policy action (count the sampled fn_idx, not the policy's).
        # last_extreme_random_sampled_fn_idx: the fn_idx drawn by the extreme-
        #   random sampler *before* _resolve_action() ran.  When resolve emits
        #   a select_* and defers the real action, the executed fn_idx is only
        #   the selector; this stores the intended sampled fn_idx so the env
        #   can attribute it correctly (issue #467 follow-up).
        self._last_was_deferred: bool = False
        self._last_was_extreme_random: bool = False
        self._last_extreme_random_sampled_fn_idx: int = 0
        # Wall-clock timestamp of the last "current game state" debug dump.
        # Logged every ~10 s when DEBUG logging is enabled so the user can
        # eyeball units / buildings / upgrades / valid action set without
        # tailing 22.4 obs/s of raw step logs.  (Issue #346 follow-up.)
        self._last_state_log_wall_s: float | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, dict]:
        """Initialise the SC2 env and return the first observation + info."""
        self._episodes_started += 1
        if self._sc2_env is None:
            self._sc2_env = self._make_sc2_env()
        elif self._is_ladder:
            # PySC2's _restart() only calls leave() when there are multiple
            # controllers, but a 1-Agent + Bot game has a single controller.
            # Without leave(), the SC2 binary stays on the post-game end
            # screen and may reject the RequestCreateGame that PySC2's
            # _create_join() issues next.  Explicitly leave the current game
            # so the binary returns to the lobby before reset() proceeds.
            self._leave_ladder_game()
        timesteps = self._sc2_env.reset()
        self._cumulative_score = 0.0
        self._explored_mask = None
        self._available_actions = None
        self._available_fn_ids = None
        self._selected_count = 0.0
        self._last_fn_idx = 0
        self._owned_buildings = frozenset()
        self._owned_buildings_seen = frozenset()
        self._completed_upgrades = frozenset()
        self._selected_unit_types = frozenset()
        self._minerals = 0.0
        self._vespene = 0.0
        self._fn_ids_cache_keys = None
        self._fn_ids_cache_result = set()
        self._screen_xy_by_unit_type = {}
        self._geyser_screen_positions = []
        self._deferred_action = None
        self._last_state_log_wall_s = None
        if self._self_play and len(timesteps) > 1:
            self._opponent_obs, _ = self._timestep_to_obs_info(timesteps[1])
        return self._timestep_to_obs_info(timesteps[0])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Apply an action and return ``(obs, score, done, info)``.

        The middle return value is the raw PySC2 reward signal — for
        minigames this is the score increment; for ladder maps it is the
        terminal +1 / -1 / 0.  The reward calculator computes the actual
        training reward in :class:`games.sc2.env.SC2Env`.

        Action resolution order (issue #346):

        1. **Deferred action** — if the previous step's resolver injected
           a selection and queued the original action, replay the original
           now so the agent's intent actually executes.
        2. **Extreme-random override** — early-episode exploration phase
           samples uniformly from the current ``_available_fn_ids`` mask
           (race ∩ PySC2 ∩ tech-tree ∩ selection).
        3. **Resolve selection** — if the chosen action requires a
           specific selection type that doesn't match the current
           selection, this step's call is replaced by the appropriate
           ``select_*`` and the original action goes into the deferred
           slot for the next step.
        """
        self._last_was_deferred = False
        self._last_was_extreme_random = False
        if self._deferred_action is not None:
            # The selector was already emitted last step; send the original
            # action directly without re-resolving.  If the selection still
            # doesn't match, PySC2 will no-op it — but we must not re-defer,
            # or we get an infinite select_army oscillation (issue #356: H1).
            # select_army is always "available" in PySC2 even with zero army
            # units, so without this guard the deferred action would never
            # actually execute.
            action = self._deferred_action
            self._deferred_action = None
            self._last_was_deferred = True
        else:
            if self._is_extreme_random_phase():
                action = self._sample_extreme_random_action()
                self._last_was_extreme_random = True
                # Capture the sampled fn_idx *before* _resolve_action() runs.
                # If resolve emits a select_* this tick and defers the real
                # action, _last_fn_idx will be the selector's fn_idx, not the
                # intended train/build/attack — storing the pre-resolve value
                # lets env.py attribute the count to the sampled intent.
                self._last_extreme_random_sampled_fn_idx = int(action[0])
            action, deferred = self._resolve_action(action)
            self._deferred_action = deferred

        fn_call = self._action_to_call(action)
        if self._self_play and self._opponent_policy is not None:
            # Self-play: opponent acts from its own observation.
            opp_obs = (
                self._opponent_obs
                if self._opponent_obs is not None
                else np.zeros(len(self._obs_names), dtype=np.float32)
            )
            opp_action = self._opponent_policy(opp_obs)
            opp_fn_call = self._action_to_call(np.asarray(opp_action, dtype=np.float32))
            timesteps = self._sc2_env.step([fn_call, opp_fn_call])
            # Update cached opponent observation for the next step.
            if len(timesteps) > 1:
                self._opponent_obs, _ = self._timestep_to_obs_info(timesteps[1])
        else:
            timesteps = self._sc2_env.step([fn_call])
        timestep = timesteps[0]
        obs, info = self._timestep_to_obs_info(timestep)
        done = bool(timestep.last())
        score = float(getattr(timestep, "reward", 0.0) or 0.0)
        return obs, score, done, info

    def close(self) -> None:
        if self._sc2_env is not None:
            self._sc2_env.close()
            self._sc2_env = None

    def _leave_ladder_game(self) -> None:
        """Leave the current ladder game so the SC2 binary returns to the lobby.

        PySC2's ``SC2Env._restart()`` only issues ``RequestLeaveGame`` when
        there are multiple controllers.  For a single-Agent + Bot game there
        is one controller, so PySC2 skips ``leave()`` and calls
        ``_create_join()`` directly.  Without the leave, the SC2 binary stays
        on the post-game end screen and may reject ``RequestCreateGame``,
        preventing automatic restart between episodes.

        Calling ``leave()`` here moves the binary back to the ``launched``
        state before PySC2's ``reset()`` triggers ``_create_join()``.
        If ``leave()`` fails (e.g. the controller is already in a disconnected
        state), the env is closed and recreated as a fallback.
        """
        try:
            controllers = getattr(self._sc2_env, "_controllers", None) or []
            for controller in controllers:
                controller.leave()
        except Exception as exc:
            logger.warning(
                "SC2Client: leave() before ladder reset raised %s; closing the SC2 env to force a clean restart.",
                exc,
            )
            self._sc2_env.close()
            self._sc2_env = self._make_sc2_env()

    def save_replay(self, replay_dir: str, prefix: str) -> str | None:
        """Save the most recently played episode as an SC2 replay file.

        Returns the path to the saved file, or None when the SC2 process is
        not running or the save fails.
        """
        if self._sc2_env is None:
            return None
        try:
            os.makedirs(replay_dir, exist_ok=True)
            result: dict[str, Any] = {"path": None, "exc": None}

            def _save() -> None:
                try:
                    result["path"] = self._sc2_env.save_replay(replay_dir, prefix=prefix)
                except Exception as exc:
                    result["exc"] = exc

            worker = threading.Thread(
                target=_save,
                name="sc2-save-replay",
                daemon=True,
            )
            worker.start()
            worker.join(timeout=_SAVE_REPLAY_TIMEOUT_S)
            if worker.is_alive():
                logger.warning(
                    "SC2Client.save_replay timed out after %.1fs; skipping.",
                    _SAVE_REPLAY_TIMEOUT_S,
                )
                return None
            if result["exc"] is not None:
                raise result["exc"]
            return result["path"]
        except Exception as exc:
            logger.warning("SC2Client.save_replay failed: %s", exc)
            return None

    @property
    def last_fn_idx(self) -> int:
        """fn_idx of the action actually sent to PySC2 in the last step.

        Reflects fallback substitutions: if the policy requested a blocked
        action and the client substituted ``select_army`` or ``no_op``,
        this returns the substituted fn_idx, not the requested one.
        """
        return self._last_fn_idx

    @property
    def last_was_deferred(self) -> bool:
        """True when the deferred-action queue fired on the last step.

        The fn_idx was already counted at the step the policy requested it;
        ``env.step()`` must not increment the count a second time.
        """
        return self._last_was_deferred

    @property
    def last_was_extreme_random(self) -> bool:
        """True when the extreme-random phase replaced the policy action.

        The policy was completely bypassed; ``env.step()`` should count the
        sampled fn_idx (``last_extreme_random_sampled_fn_idx``), not the
        policy's no_op request or the executed selector fn_idx.
        """
        return self._last_was_extreme_random

    @property
    def last_extreme_random_sampled_fn_idx(self) -> int:
        """fn_idx drawn by the extreme-random sampler on the last step.

        Only meaningful when ``last_was_extreme_random`` is True.  Captures
        the sampled intent *before* ``_resolve_action()`` ran, so if resolve
        emitted a ``select_*`` this tick and deferred the real action,
        ``env.step()`` can still attribute the count to the sampled fn_idx
        (e.g. train/build/attack) rather than to the selector or no_op.
        """
        return self._last_extreme_random_sampled_fn_idx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_ladder(map_name: str) -> bool:
        from games.sc2.obs_spec import MINIGAME_NAMES

        return map_name not in MINIGAME_NAMES

    def _make_sc2_env(self) -> Any:
        try:
            from pysc2.env import sc2_env  # type: ignore[import-untyped]
            from pysc2.lib import features  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pysc2 is required for the StarCraft 2 game integration.  "
                "Install it with:  poetry install --with sc2  "
                "(and download the Blizzard SC2 binary + maps separately — "
                "see CLAUDE.md for setup instructions)."
            ) from exc

        from absl import flags as _absl_flags  # type: ignore[import-untyped]

        if not _absl_flags.FLAGS.is_parsed():
            _absl_flags.FLAGS([""])

        # Issue #254: serialise map-file reads across all SC2 binaries
        # running on this host (distributed local workers, parallel-eval
        # workers, etc.) so they don't race on the same .SC2Map file.
        from games.sc2.map_access_gate import acquire_map_access_slot

        acquire_map_access_slot()

        if self._play_mode:
            # Human (via SC2 UI) vs AI agent.  PySC2 only takes step actions
            # for Agent slots; Human actions come from the game client directly.
            agents = [
                sc2_env.Human(self._race(sc2_env)),
                sc2_env.Agent(self._race(sc2_env), "ai_agent"),
            ]
        elif self._self_play:
            # Two AI agents for self-play.  Both are Agent slots so PySC2
            # expects one action list entry per agent.
            agents = [
                sc2_env.Agent(self._race(sc2_env), "rl_agent"),
                sc2_env.Agent(self._race(sc2_env), "opponent_agent"),
            ]
        else:
            agents = [sc2_env.Agent(self._race(sc2_env), "rl_agent")]
            if self._is_ladder:
                agents.append(
                    sc2_env.Bot(
                        self._race(sc2_env),
                        self._difficulty(sc2_env),
                    )
                )

        return sc2_env.SC2Env(
            map_name=self._map_name,
            players=agents,
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=self._screen_size,
                    minimap=self._minimap_size,
                ),
                use_feature_units=True,
            ),
            step_mul=self._step_mul,
            game_steps_per_episode=0,
            visualize=self._visualize,
            realtime=self._realtime,
            disable_fog=False,
        )

    def _race(self, sc2_env_mod: Any) -> Any:
        return getattr(sc2_env_mod.Race, self._agent_race, sc2_env_mod.Race.random)

    def _difficulty(self, sc2_env_mod: Any) -> Any:
        return getattr(
            sc2_env_mod.Difficulty,
            self._bot_difficulty,
            sc2_env_mod.Difficulty.very_easy,
        )

    def _resolve_action(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Resolve selection preconditions, optionally deferring the action.

        Issue #346: actions like ``Build_SupplyDepot_screen`` and
        ``Train_Marine_quick`` require specific selection types (a worker
        for builds, the producer building for trains).  If the currently-
        selected unit type doesn't match the action's ``selection_target``,
        emit the appropriate ``select_*`` *this* tick and return the
        original action for the *next* tick.

        Returns ``(executed_action, deferred_action_or_None)``.
        """
        fn_idx = int(action[0])
        pre = PRECONDITIONS.get(fn_idx)
        if pre is None or pre.required_selection == SelectionReq.NONE:
            return action, None

        # ANY_UNIT — any non-building unit selected is fine.
        if pre.required_selection == SelectionReq.ANY_UNIT:
            if self._selected_count >= 1.0:
                return action, None
            selector = self._pick_any_unit_selector()
            if selector is None:
                return action, None  # no recourse; PySC2 will no-op it
            return selector, action

        # OF_TYPE — selection must contain at least one type in selection_target.
        if self._selected_unit_types & pre.selection_target:
            return action, None

        selector = self._pick_typed_selector(pre.selection_target)
        if selector is None:
            # Mask should have prevented this. Fall through to the action;
            # PySC2 will no-op it. Log so the gap is visible.
            logger.debug(
                "_resolve_action: no selector available for %s (target=%s, screen_cache=%s)",
                FUNCTION_IDS.get(fn_idx, "?"),
                sorted(pre.selection_target),
                sorted(self._screen_xy_by_unit_type.keys()),
            )
            return action, None
        return selector, action

    def _pick_any_unit_selector(self) -> np.ndarray | None:
        """Return a ``select_*`` action that picks any visible friendly unit.

        Prefers ``select_army`` when PySC2 reports it available; otherwise
        falls back to ``select_point`` on any non-worker, non-structure
        unit cached from ``feature_units``.
        """
        from games.sc2.actions import WARMUP_ACTION  # select_army at fn_idx=1

        select_army_pysc2_id = self._pysc2_fn_id("select_army")
        if self._available_actions is None or (
            select_army_pysc2_id is not None and select_army_pysc2_id in self._available_actions
        ):
            return WARMUP_ACTION.copy()
        # No army — try select_point on any cached friendly unit.
        for name, (sx, sy) in self._screen_xy_by_unit_type.items():
            if name in WORKER_NAMES:
                continue
            return self._select_point_action(sx, sy)
        # Last-ditch: select any worker.
        for name in WORKER_NAMES:
            xy = self._screen_xy_by_unit_type.get(name)
            if xy is not None:
                return self._select_point_action(*xy)
        return None

    def _pick_typed_selector(self, target_names: frozenset[str]) -> np.ndarray | None:
        """Return a ``select_*`` action that selects a unit/building of an
        accepted type.

        For workers, prefers ``select_idle_worker`` (cheap, no screen-target)
        when PySC2 reports it available and there's an idle worker; otherwise
        falls back to ``select_point`` on any cached worker (mining or
        building — issue #346 specifically requires non-idle workers be
        selectable).  For non-workers, uses ``select_point`` on the cached
        screen location of any unit in ``target_names``.
        """
        is_worker_target = bool(target_names & WORKER_NAMES)

        if is_worker_target:
            # Prefer select_idle_worker (cheap, no screen-target).
            siw_id = self._pysc2_fn_id("select_idle_worker")
            if siw_id is not None and (self._available_actions is None or siw_id in self._available_actions):
                return np.array([4, 0.5, 0.5, 0], dtype=np.float32)
            # Otherwise select_point on any visible worker (mining/building OK).
            for name in target_names:
                xy = self._screen_xy_by_unit_type.get(name)
                if xy is not None:
                    return self._select_point_action(*xy)
            return None

        # Non-worker target: scan cached positions for one of the accepted
        # types and select_point on it.
        for name in target_names:
            xy = self._screen_xy_by_unit_type.get(name)
            if xy is not None:
                return self._select_point_action(*xy)
        return None

    def _select_point_action(self, sx: int, sy: int) -> np.ndarray:
        x_norm = float(sx) / max(self._screen_size - 1, 1)
        y_norm = float(sy) / max(self._screen_size - 1, 1)
        return np.array([6, x_norm, y_norm, 0], dtype=np.float32)

    def _pysc2_fn_id(self, name: str) -> int | None:
        """Return the raw PySC2 function id for *name*, or None if pysc2 unavailable."""
        try:
            from pysc2.lib import actions as pysc2_actions  # type: ignore[import-untyped]
        except ImportError:
            return None
        fn = getattr(pysc2_actions.FUNCTIONS, name, None)
        return int(fn.id) if fn is not None else None

    def _snap_refinery_to_geyser(self, action: np.ndarray) -> np.ndarray:
        """Snap Build_Refinery/Assimilator/Extractor targets onto a real geyser.

        Refineries can only be placed on a vespene geyser, but the policy
        outputs continuous screen coordinates that almost never line up with
        one.  When the action's fn_idx is in :data:`REFINERY_FN_IDS` and at
        least one visible neutral geyser is cached this step, we rewrite the
        target ``(x, y)`` to the nearest geyser's normalised screen position.
        Actions with other fn_ids — and refinery actions taken on screens
        with no visible geyser — pass through unchanged (issue #402).
        """
        fn_idx = int(action[0])
        if fn_idx not in REFINERY_FN_IDS or not self._geyser_screen_positions:
            return action
        scale = max(self._screen_size - 1, 1)
        tx_px = float(np.clip(action[1], 0.0, 1.0)) * scale
        ty_px = float(np.clip(action[2], 0.0, 1.0)) * scale
        best = min(
            self._geyser_screen_positions,
            key=lambda p: (p[0] - tx_px) ** 2 + (p[1] - ty_px) ** 2,
        )
        # Offset by 0.5 px before normalising so that downstream int-truncation
        # (``int(x_norm * (screen_size - 1))`` in ``action_to_function_call``)
        # lands on the geyser's pixel instead of one off due to float rounding.
        snapped = action.copy()
        snapped[1] = min(1.0, (float(best[0]) + 0.5) / scale)
        snapped[2] = min(1.0, (float(best[1]) + 0.5) / scale)
        return snapped

    def _action_to_call(self, action: np.ndarray) -> Any:
        """Translate a 4-vector action to a PySC2 ``FunctionCall``.

        Selection preconditions are handled upstream by ``_resolve_action``;
        the tech-tree mask plus the deferred-action queue should make this
        function's input always-executable. When PySC2 still reports the
        action as unavailable (rare — e.g. resource gates or a stale
        observation race) we issue ``no_op`` rather than substituting,
        keeping this layer simple and deterministic.

        Refinery-style actions also pass through
        :meth:`_snap_refinery_to_geyser`, which rewrites the target
        coordinates onto the nearest visible vespene geyser so the placement
        is actually legal in-game (issue #402).
        """
        from pysc2.lib import actions as pysc2_actions  # type: ignore[import-untyped]

        action = self._snap_refinery_to_geyser(action)
        fn_call = action_to_function_call(action, self._screen_size, self._minimap_size)

        fn_idx = int(action[0])
        fn_name = FUNCTION_IDS.get(fn_idx, "no_op")

        if self._available_actions is not None and int(fn_call.function) not in self._available_actions:
            logger.debug(
                "Action %s unavailable in PySC2 mask; issuing no_op.",
                fn_name,
            )
            self._last_fn_idx = 0
            return pysc2_actions.FunctionCall(int(pysc2_actions.FUNCTIONS.no_op.id), [])

        self._last_fn_idx = fn_idx

        if fn_name != "no_op":
            x_screen = int(np.clip(action[1], 0.0, 1.0) * (self._screen_size - 1))
            y_screen = int(np.clip(action[2], 0.0, 1.0) * (self._screen_size - 1))
            queue = int(np.clip(round(float(action[3])), 0, 1))
            if fn_name in ("select_army", "select_idle_worker"):
                logger.debug("Action: %s", fn_name)
            else:
                logger.debug(
                    "Action: %s  screen=(%d, %d)  queue=%d",
                    fn_name,
                    x_screen,
                    y_screen,
                    queue,
                )

        return fn_call

    def _is_extreme_random_phase(self) -> bool:
        return self._episodes_started > 0 and self._episodes_started <= self._extreme_random_run_count

    def _sample_extreme_random_action(self) -> np.ndarray:
        """Sample uniformly from the fully-filtered internal fn_idx mask.

        Reads ``self._available_fn_ids`` (race ∩ PySC2 ∩ tech-tree ∩
        selection) rather than raw PySC2 ``available_actions``; this is
        what makes issue #346's "Build_FusionCore at game start" no
        longer reachable in the random phase.  Falls back to ``no_op``
        (fn_idx=0) when the mask is empty (e.g. very first tick before
        any observation has populated the mask).
        """
        if self._available_fn_ids:
            valid_fn_ids = sorted(self._available_fn_ids)
            fn_idx = int(self._rng.choice(valid_fn_ids))
        else:
            fn_idx = 0
        return np.array(
            [
                fn_idx,
                float(self._rng.random()),
                float(self._rng.random()),
                float(self._rng.integers(0, 2)),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Observation flattening
    # ------------------------------------------------------------------

    def _timestep_to_obs_info(self, timestep: Any) -> tuple[np.ndarray, dict]:
        """Convert a PySC2 TimeStep into ``(flat_obs, info)``.

        Builds a name-indexed feature dict from PySC2 fields then projects
        onto ``self._spec.names`` to produce the flat observation.  Each
        feature group has its own extractor so unit tests can target them.
        """
        ob = timestep.observation
        feat_screen = self._safe_array(ob, "feature_screen")
        feat_minimap = self._safe_array(ob, "feature_minimap")

        feats: dict[str, float] = {}
        feats.update(self._player_features(ob))
        feats.update(self._selected_features(ob))
        self._selected_count = float(feats.get("selected_count", 0.0))
        feats.update(self._screen_summary_features(feat_screen))
        feats.update(self._minimap_summary_features(feat_minimap))
        # Ladder preset adds economy/minimap/score/HP/topk/alerts features.
        if self._use_ladder_obs:
            feats.update(self._score_features(ob))
            feats.update(self._screen_hp_features(feat_screen))
            feats.update(self._topk_enemy_features(feat_screen))
            feats.update(self._alerts_features(ob))
        # _per_unit_type_features is always needed: info["unit_counts"] drives
        # build-order tracking in SC2Env regardless of obs preset.
        feats.update(self._per_unit_type_features(ob))
        # _update_unit_screen_positions is always needed for action resolution.
        self._update_unit_screen_positions(ob)
        # Rich preset adds quadrants, action features, enemy types, etc.
        if self._use_rich_obs:
            feats.update(self._quadrant_features(feat_screen))
            feats.update(self._available_actions_features(ob))
            feats.update(self._last_action_features())
            feats.update(self._enemy_unit_type_features(ob))
            feats.update(self._shield_energy_features(feat_screen))
            feats.update(self._creep_features(feat_minimap))
            feats.update(self._economy_pipeline_features(ob))
            feats.update(self._screen_visibility_features(feat_screen))
            feats.update(self._screen_antiair_features(feat_screen))
            feats.update(self._weapon_cooldown_features(ob))

        # game_loop scalar — present on both ladder and rich.
        game_loop_arr = self._safe_array(ob, "game_loop")
        game_loop = float(game_loop_arr[0]) if game_loop_arr is not None and game_loop_arr.size > 0 else 0.0
        feats["game_loop"] = game_loop

        # Project feature dict → ordered ndarray driven by the active spec.
        flat = np.array(
            [float(feats.get(name, 0.0)) for name in self._obs_names],
            dtype=np.float32,
        )

        # Build the info dict — score deltas + reward inputs.
        prev_score = self._cumulative_score
        score_arr = self._safe_array(ob, "score_cumulative")
        if score_arr is not None and score_arr.size > 0:
            cumulative = float(score_arr[0])
        else:
            cumulative = prev_score + float(getattr(timestep, "reward", 0.0) or 0.0)
        self._cumulative_score = cumulative

        # Track raw PySC2 available_actions for downstream layer checks.
        avail_arr = self._safe_array(ob, "available_actions")
        if avail_arr is not None:
            self._available_actions = set(avail_arr.tolist())
        else:
            self._available_actions = None

        # Compute the full internal mask: race ∩ PySC2 ∩ tech-tree ∩ selection ∩ resources.
        # Single source of truth — read by extreme-random sampler, policies
        # (via info["available_fn_ids"]), and the deferred-action resolver.
        self._owned_buildings_seen = self._owned_buildings_seen | self._compute_owned_buildings(ob)
        self._owned_buildings = self._owned_buildings_seen
        self._completed_upgrades = self._compute_completed_upgrades(ob)
        self._selected_unit_types = self._compute_selected_unit_types(ob)
        self._minerals = feats.get("minerals", 0.0)
        self._vespene = feats.get("vespene", 0.0)
        available_fn_ids = self._compute_available_fn_ids(ob)
        self._available_fn_ids = available_fn_ids

        # player_outcome is only meaningful for ladder maps where PySC2 emits
        # a terminal +1 / -1 / 0.  For minigames timestep.reward is a per-step
        # score delta, not a win/loss signal, so we leave it as None.
        if timestep.last() and self._is_ladder:
            player_outcome: float | None = float(getattr(timestep, "reward", 0.0) or 0.0)
        else:
            player_outcome = None

        info = {
            "score": cumulative,
            "prev_score": prev_score,
            "minerals": feats.get("minerals", 0.0),
            "vespene": feats.get("vespene", 0.0),
            "prev_minerals": 0.0,  # filled in by env on subsequent steps
            "prev_vespene": 0.0,
            "food_used": feats.get("food_used", 0.0),
            "food_cap": feats.get("food_cap", 0.0),
            "food_workers": feats.get("food_workers", 0.0),
            "food_army": feats.get("food_army", 0.0),
            "army_count": feats.get("army_count", 0.0),
            "idle_worker_count": feats.get("idle_worker_count", 0.0),
            # Progression-reward inputs (supply/economy growth, tech, scouting).
            "minimap_explored_frac": feats.get("minimap_explored_frac", 0.0),
            "owned_building_names": self._owned_buildings,
            "townhall_count": self._townhall_count(ob),
            "killed_value_units": feats.get("killed_value_units", 0.0),
            "killed_value_structures": feats.get("killed_value_structures", 0.0),
            "player_outcome": player_outcome,
            "is_last": bool(timestep.last()),
            "available_fn_ids": available_fn_ids,
            "game_loop": game_loop,
            # Screen summary used by reward shaping (idle_bonus, #127).
            "screen_self_count": feats.get("screen_self_count", 0.0),
            "screen_enemy_count": feats.get("screen_enemy_count", 0.0),
            "screen_self_cx": feats.get("screen_self_cx", 0.0),
            "screen_self_cy": feats.get("screen_self_cy", 0.0),
            "screen_enemy_cx": feats.get("screen_enemy_cx", 0.0),
            "screen_enemy_cy": feats.get("screen_enemy_cy", 0.0),
            "selected_count": feats.get("selected_count", 0.0),
            "visible_self_unit_count": self._visible_self_unit_count(ob),
            # Per-unit-type counts for build-order tracking (analytics).
            # Keys follow _RICH_UNIT_TYPES naming: "Marine", "SCV", etc.
            "unit_counts": {name: feats.get(f"unit_count_{name}", 0.0) for name in self._get_rich_unit_types()},
        }
        self_attack_range_px = self._self_attack_range_px(ob)
        if self_attack_range_px is not None:
            info["self_attack_range_px"] = self_attack_range_px
        info["total_self_hp"] = self._total_self_hp(ob)
        info["self_weapon_cooldown_mean"] = feats.get("self_weapon_cooldown_mean", 0.0)

        # Periodic human-readable state dump for tech-tree debugging
        # (issue #346 follow-up).  Logged at DEBUG, throttled to ~10 s
        # wall-clock so the log isn't drowned in 22.4 lines/sec.
        self._maybe_log_state_dump(feats)

        # Raw minimap visibility layer — only stored when the belief module is
        # active (store_minimap_vis=True) to avoid the per-step payload cost
        # of carrying a full H×W array when belief is disabled.
        if self._store_minimap_vis:
            info["minimap_vis"] = self._extract_visibility(feat_minimap)

        # Spatial obs: stack selected screen + minimap layers into (C, H, W).
        if self._screen_layers or self._minimap_layers:
            feat_screen = self._safe_array(ob, "feature_screen")
            channels: list[np.ndarray] = []
            for name in self._screen_layers:
                layer = self._extract_named_layer(feat_screen, name)
                scale = _LAYER_SCALE.get(name, 1.0)
                if layer is not None:
                    channels.append((layer / scale).astype(np.float32))
                else:
                    channels.append(np.zeros((self._screen_size, self._screen_size), dtype=np.float32))
            if self._minimap_layers:
                feat_minimap = self._safe_array(ob, "feature_minimap")
                for name in self._minimap_layers:
                    layer = self._extract_named_layer(feat_minimap, name)
                    scale = _LAYER_SCALE.get(name, 1.0)
                    if layer is not None:
                        layer = (layer / scale).astype(np.float32)
                        if layer.shape != (self._screen_size, self._screen_size):
                            layer = self._resize_layer(layer, self._screen_size)
                        channels.append(layer)
                    else:
                        channels.append(np.zeros((self._screen_size, self._screen_size), dtype=np.float32))
            if channels:
                info["spatial_obs"] = np.stack(channels, axis=0)

        return flat, info

    # ------------------------------------------------------------------
    # Feature-block extractors — each emits a {name: float} dict matching
    # the per-block ObsDim definitions in games/sc2/obs_spec.py.  A given
    # extractor is allowed to return more keys than any one preset uses;
    # _timestep_to_obs_info projects onto self._obs_names at the end.
    # ------------------------------------------------------------------

    def _player_features(self, ob: Any) -> dict[str, float]:
        return _player_features(ob)

    def _selected_features(self, ob: Any) -> dict[str, float]:
        return _selected_features(ob)

    def _screen_summary_features(self, feat_screen: np.ndarray | None) -> dict[str, float]:
        return _screen_summary_features(feat_screen)

    def _minimap_summary_features(self, feat_minimap: np.ndarray | None) -> dict[str, float]:
        out, self._explored_mask = _minimap_summary_features(feat_minimap, self._explored_mask)
        return out

    def _score_features(self, ob: Any) -> dict[str, float]:
        return _score_features(ob)

    def _screen_hp_features(self, feat_screen: np.ndarray | None) -> dict[str, float]:
        return _screen_hp_features(feat_screen)

    def _topk_enemy_features(self, feat_screen: np.ndarray | None) -> dict[str, float]:
        return _topk_enemy_features(feat_screen)

    def _per_unit_type_features(self, ob: Any) -> dict[str, float]:
        out, self._unit_type_id_to_name = _per_unit_type_features(ob, self._unit_type_id_to_name)
        return out

    def _quadrant_features(self, feat_screen: np.ndarray | None) -> dict[str, float]:
        return _quadrant_features(feat_screen)

    def _available_actions_features(self, ob: Any) -> dict[str, float]:
        return _available_actions_features(ob)

    def _last_action_features(self) -> dict[str, float]:
        return _last_action_features(self._last_fn_idx)

    def _enemy_unit_type_features(self, ob: Any) -> dict[str, float]:
        out, self._unit_type_id_to_name = _enemy_unit_type_features(ob, self._unit_type_id_to_name)
        return out

    def _shield_energy_features(self, feat_screen: np.ndarray | None) -> dict[str, float]:
        return _shield_energy_features(feat_screen)

    def _creep_features(self, feat_minimap: np.ndarray | None) -> dict[str, float]:
        return _creep_features(feat_minimap)

    def _economy_pipeline_features(self, ob: Any) -> dict[str, float]:
        return _economy_pipeline_features(ob)

    def _screen_visibility_features(self, feat_screen: np.ndarray | None) -> dict[str, float]:
        """Fraction of screen tiles currently fully visible (visibility_map == 2).

        Complements ``minimap_visible_frac`` (which measures the full-map
        coverage) by capturing how much of the *current camera view* is
        un-fogged.  PySC2 feature_screen ``visibility_map`` values:
        0 = hidden, 1 = fogged, 2 = visible.
        """
        return _screen_visibility_features(feat_screen)

    def _screen_antiair_features(self, feat_screen: np.ndarray | None) -> dict[str, float]:
        """Mean anti-air unit density across the screen.

        PySC2 ``unit_density_aa`` is a spatial layer where each tile holds the
        count of anti-air units present.  Aggregating to a mean scalar gives a
        compact air-threat signal without requiring the full spatial grid.
        """
        return _screen_antiair_features(feat_screen)

    def _alerts_features(self, ob: Any) -> dict[str, float]:
        """Number of active PySC2 alerts this step.

        ``obs.observation["alerts"]`` is a variable-size tensor emitted when
        the player is under major attack (see PySC2 docs: "Actions and
        Observations → Structured → Alerts").  The array is usually empty and
        at most 2 entries long.  We collapse it to a single count scalar so the
        policy receives a direct "under attack" signal without needing to handle
        variable-size arrays.
        """
        return _alerts_features(ob)

    def _weapon_cooldown_features(self, ob: Any) -> dict[str, float]:
        """Mean weapon cooldown for friendly units from ``feature_units``.

        PySC2 ``feature_units`` rows have ``weapon_cooldown`` at column 25.
        A mean of zero means all weapons are ready; higher values indicate
        units that fired recently and cannot shoot again immediately.  Only
        units with alliance == 1 (self) are included.
        """
        return _weapon_cooldown_features(ob)

    def _self_attack_range_px(self, ob: Any) -> float | None:
        """Approximate max friendly attack range in screen pixels."""
        feat_units = self._safe_array(ob, "feature_units")
        if feat_units is None or feat_units.size == 0:
            return None
        if feat_units.ndim != 2 or feat_units.shape[1] < _FU.alliance + 1:
            return None
        if self._unit_type_id_to_attack_range_gu is None:
            self._unit_type_id_to_attack_range_gu = self._build_attack_range_lookup()

        max_range_gu = 0.0
        for row, owner in zip(feat_units, feat_units[:, _FU.alliance]):
            if int(owner) != 1:
                continue
            max_range_gu = max(
                max_range_gu,
                self._unit_type_id_to_attack_range_gu.get(int(row[_FU.unit_type]), 0.0),
            )
        if max_range_gu <= 0.0:
            return None

        scale = _MARINE_RANGE_PX_AT_64 / _MARINE_RANGE_GU * (float(self._screen_size) / 64.0)
        return float(max_range_gu * scale)

    def _total_self_hp(self, ob: Any) -> float:
        """Sum of health+shield for all visible friendly units from ``feature_units``.

        Used by the damage-taken penalty: when the sum drops step-over-step,
        friendly units absorbed damage.  Only units currently on-screen are
        counted (feature_units limitation), so units walking off-camera will
        cause the sum to drop without actual damage — keep ``damage_taken_penalty``
        weights small to account for this noise.

        """
        feat_units = self._safe_array(ob, "feature_units")
        if feat_units is None or feat_units.size == 0:
            return 0.0
        if feat_units.ndim != 2 or feat_units.shape[1] < _FU.shield + 1:
            return 0.0
        self_mask = feat_units[:, _FU.alliance] == 1
        if not self_mask.any():
            return 0.0
        hp = feat_units[self_mask, _FU.health].astype(np.float32)
        shields = feat_units[self_mask, _FU.shield].astype(np.float32)
        return float((hp + shields).sum())

    def _visible_self_unit_count(self, ob: Any) -> float:
        """Count visible friendly units from ``feature_units``."""
        feat_units = self._safe_array(ob, "feature_units")
        if feat_units is None or feat_units.size == 0:
            return 0.0
        if feat_units.ndim != 2 or feat_units.shape[1] < _FU.alliance + 1:
            return 0.0
        return float((feat_units[:, _FU.alliance] == 1).sum())

    def _update_unit_screen_positions(self, ob: Any) -> None:
        """Cache screen (x, y) per friendly unit-type name from ``feature_units``.

        Column indices come from :class:`_FU`.

        Populates ``self._screen_xy_by_unit_type`` (replaces the
        worker-only cache used before issue #346).  The deferred-action
        resolver uses this to issue ``select_point`` on the right
        worker/building/unit when the agent's chosen action requires a
        different selection.

        Also refreshes ``self._geyser_screen_positions`` with every visible
        neutral vespene-geyser position (alliance=3) so the Build_Refinery
        snap (issue #402) can pick the nearest geyser to the policy's
        requested target.

        First-seen-wins per friendly unit-type name; ties are resolved
        arbitrarily.  Geyser positions are kept in full (every visible
        geyser is a distinct candidate).
        """
        self._screen_xy_by_unit_type = {}
        self._geyser_screen_positions = []
        feat_units = self._safe_array(ob, "feature_units")
        if feat_units is None or feat_units.size == 0:
            return
        if feat_units.ndim != 2 or feat_units.shape[1] < _FU.y + 1:
            return
        if self._unit_type_id_to_name is None:
            self._unit_type_id_to_name = self._build_unit_type_lookup()
        for row in feat_units:
            alliance = int(row[_FU.alliance])
            name = self._unit_type_id_to_name.get(int(row[_FU.unit_type]))
            if name is None:
                continue
            if alliance == 1:
                if name in self._screen_xy_by_unit_type:
                    continue
                self._screen_xy_by_unit_type[name] = (int(row[_FU.x]), int(row[_FU.y]))
            elif alliance == 3 and name in GEYSER_NAMES:
                self._geyser_screen_positions.append((int(row[_FU.x]), int(row[_FU.y])))

    def _compute_owned_buildings(self, ob: Any) -> frozenset[str]:
        """Return the set of friendly structure/building names visible this step.

        Returns frozenset of pysc2.lib.units member names. Used as the
        ``owned_buildings`` input to the tech-tree filter.
        """
        feat_units = self._safe_array(ob, "feature_units")
        if feat_units is None or feat_units.size == 0:
            return frozenset()
        if feat_units.ndim != 2 or feat_units.shape[1] < _FU.alliance + 1:
            return frozenset()
        if self._unit_type_id_to_name is None:
            self._unit_type_id_to_name = self._build_unit_type_lookup()
        names: set[str] = set()
        for row in feat_units:
            if int(row[_FU.alliance]) != 1:
                continue
            name = self._unit_type_id_to_name.get(int(row[_FU.unit_type]))
            if name is not None and name in STRUCTURE_NAMES:
                names.add(name)
        return frozenset(names)

    def _townhall_count(self, ob: Any) -> int:
        """Return the number of currently-visible friendly town halls.

        Counts friendly ``feature_units`` whose type name is a town hall
        (CommandCenter / Nexus / Hatchery and their morphs).  Used by the
        expansion reward; only on-screen units are visible, so a base built
        fully off-camera may be undercounted — the reward tracks a running
        maximum to keep the expansion signal monotonic.
        """
        feat_units = self._safe_array(ob, "feature_units")
        if feat_units is None or feat_units.size == 0:
            return 0
        if feat_units.ndim != 2 or feat_units.shape[1] < _FU.alliance + 1:
            return 0
        if self._unit_type_id_to_name is None:
            self._unit_type_id_to_name = self._build_unit_type_lookup()
        count = 0
        for row in feat_units:
            if int(row[_FU.alliance]) != 1:
                continue
            name = self._unit_type_id_to_name.get(int(row[_FU.unit_type]))
            if name is not None and name in TOWNHALL_NAMES:
                count += 1
        return count

    def _compute_completed_upgrades(self, ob: Any) -> frozenset[str]:
        """Return the set of completed upgrade/research names this step.

        Reads PySC2's ``upgrades`` field (int array of UpgradeID values)
        and maps to names via ``pysc2.lib.upgrades.Upgrades`` enum.
        Returns frozenset of upgrade names (empty if pysc2 unavailable).
        """
        upgrades_arr = self._safe_array(ob, "upgrades")
        if upgrades_arr is None or upgrades_arr.size == 0:
            return frozenset()
        lookup = self._upgrade_id_to_name()
        if not lookup:
            return frozenset()
        names: set[str] = set()
        for uid in upgrades_arr.tolist():
            name = lookup.get(int(uid))
            if name is not None:
                names.add(name)
        return frozenset(names)

    def _compute_selected_unit_types(self, ob: Any) -> frozenset[str]:
        """Return the set of unit-type names currently in the selection.

        Reads ``single_select`` and ``multi_select``.  Empty selection →
        empty frozenset.  Mixed-type selections (e.g. ``select_army`` on
        a Marine + Marauder army) return all distinct types so that
        ``ANY_UNIT`` precondition checks stay satisfied and ``OF_TYPE``
        checks succeed whenever at least one selected unit matches the
        action's target — PySC2 will route the command to the
        compatible units in the selection.
        """
        names: set[str] = set()
        for key in ("single_select", "multi_select"):
            arr = self._safe_array(ob, key)
            if arr is None or arr.size == 0:
                continue
            if arr.ndim == 1:
                rows = arr[np.newaxis, :]
            elif arr.ndim == 2 and arr.shape[1] >= 1:
                rows = arr
            else:
                continue
            if self._unit_type_id_to_name is None:
                self._unit_type_id_to_name = self._build_unit_type_lookup()
            for row in rows:
                name = self._unit_type_id_to_name.get(int(row[_FU.unit_type]))
                if name is not None:
                    names.add(name)
        return frozenset(names)

    def _compute_available_fn_ids(self, ob: Any) -> set[int]:
        """Compute the per-step internal fn_idx mask.

        Order:
          1. Race mask (configured race; falls back to inferred race when
             configured is "random").
          2. PySC2 ``available_actions`` (mapped to internal fn_idx).
          3. Tech-tree filter: building prereqs, upgrade prereqs,
             selection-type requirements, and resource costs (see
             :func:`games.sc2.tech_tree.fn_idx_satisfied`).

        Note: the resource filter uses the agent's current mineral/vespene
        bank as of this step.  It does not account for resources already
        committed by a deferred/queued action in the same decision cycle, so
        back-to-back build commands can still over-draw resources if the mask
        is consulted again before the first command executes.
        """
        configured = (self._agent_race or "random").lower()
        if configured == "random":
            inferred = self._infer_fn_ids_from_units(ob)
            if inferred is not None:
                race_set: set[int] = set(inferred)
            else:
                race_set = set(fn_ids_for_race("random"))
        else:
            race_set = set(fn_ids_for_race(configured))

        pysc2_mapped: set[int] | None = None
        if self._available_actions is not None:
            id_map = _get_pysc2_id_to_fn_idx()
            if id_map:
                pysc2_mapped = {id_map[pid] for pid in self._available_actions if pid in id_map}

        candidate = race_set if pysc2_mapped is None else race_set & pysc2_mapped

        # Tech-tree + resource filter — exact game-state check.  Skipped when
        # the unit-type lookup is empty (PySC2 unavailable, e.g. unit tests):
        # without unit-type names we can't determine owned_buildings or
        # selected_unit_types, so the tech filter would drop everything
        # that requires a specific selection.  Fall back to race ∩ PySC2.
        if not self._unit_type_id_to_name:
            out = set(candidate)
            if WORKER_NAMES & self._selected_unit_types:
                out.discard(4)
            return out

        # Accessible types: all unit/building types the agent could select
        # this step — owned buildings (persistent across the episode) plus
        # every unit type currently visible on screen.  Passing this instead
        # of the *current* selection means training actions (e.g.
        # Train_Marine_quick) appear in the mask whenever the producer
        # building exists, not only when it happens to be selected.  The
        # deferred-action resolver auto-emits a select_point when the policy
        # picks such an action — but only if the target has a cached screen
        # position.  Buildings in owned_buildings that are off-screen will
        # appear in the mask yet no-op that tick; this is an intentional
        # trade-off favouring a stable mask over camera-driven flicker.
        visible_unit_types = frozenset(self._screen_xy_by_unit_type.keys())
        accessible_types = self._owned_buildings | visible_unit_types

        # Cache: skip the full fn_idx_satisfied loop when the game-state inputs
        # are unchanged (issue #356 H3).  Minerals and vespene are included
        # because PR #357 gates build/train actions on affordability.
        cache_keys = (
            frozenset(candidate),
            self._owned_buildings,
            self._completed_upgrades,
            visible_unit_types,
            self._minerals,
            self._vespene,
        )
        if cache_keys == self._fn_ids_cache_keys:
            out = set(self._fn_ids_cache_result)
        else:
            result = {
                fn_idx
                for fn_idx in candidate
                if fn_idx_satisfied(
                    fn_idx,
                    self._owned_buildings,
                    self._completed_upgrades,
                    accessible_types,
                    self._minerals,
                    self._vespene,
                )
            }
            self._fn_ids_cache_keys = cache_keys
            self._fn_ids_cache_result = result
            out = set(result)

        # Issue #383: suppress select_idle_worker (fn_idx 4) when a worker is
        # already in the current selection.  The policy would otherwise waste
        # steps re-selecting an idle worker it already has, preventing it from
        # acting on the build/harvest actions that are already unblocked.
        if WORKER_NAMES & self._selected_unit_types:
            out.discard(4)
        return out

    _upgrade_id_to_name_cache: dict[int, str] | None = None

    @classmethod
    def _upgrade_id_to_name(cls) -> dict[int, str]:
        """Lazy-built lookup table from PySC2 UpgradeID → name."""
        if cls._upgrade_id_to_name_cache is None:
            try:
                from pysc2.lib import upgrades as pysc2_upgrades  # type: ignore[import-untyped]
            except ImportError:
                cls._upgrade_id_to_name_cache = {}
            else:
                # PySC2 exposes Upgrades as an IntEnum (Upgrades) with
                # members like Stimpack=15, CombatShield=16, etc. Fall back
                # to an empty dict if the structure changes.
                enum_cls = getattr(pysc2_upgrades, "Upgrades", None) or getattr(pysc2_upgrades, "Upgrade", None)
                if enum_cls is None:
                    cls._upgrade_id_to_name_cache = {}
                else:
                    cls._upgrade_id_to_name_cache = {int(m.value): m.name for m in enum_cls}
        return cls._upgrade_id_to_name_cache

    # ------------------------------------------------------------------
    # Periodic state dump (issue #346 follow-up)
    # ------------------------------------------------------------------

    _STATE_LOG_INTERVAL_S: float = 10.0

    def _maybe_log_state_dump(self, feats: dict[str, float]) -> None:
        """Log a readable game-state dump every ~10 s of wall-clock time.

        Gated on DEBUG log level — the dump is skipped entirely when the
        logger isn't at DEBUG, so there's no per-step cost in production.
        Logs:

        - Currently owned units (friendly, counted from feature_units).
        - Currently owned buildings (friendly structures).
        - Completed upgrades / research.
        - Possible actions (the internal ``_available_fn_ids`` mask),
          each annotated with the unit/building type that needs to be
          selected.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        now = time.monotonic()
        last = self._last_state_log_wall_s
        if last is not None and (now - last) < self._STATE_LOG_INTERVAL_S:
            return
        self._last_state_log_wall_s = now

        units_str = self._format_owned_units(feats)
        buildings_str = self._format_set_or_dash(self._owned_buildings)
        upgrades_str = self._format_set_or_dash(self._completed_upgrades)
        actions_str = self._format_available_actions(self._available_fn_ids)
        if not self._selected_unit_types:
            selected_str = "(nothing selected)"
        else:
            selected_str = ", ".join(sorted(self._selected_unit_types))

        logger.debug(
            "SC2 state @ game_loop=%d:\n"
            "  units      : %s\n"
            "  buildings  : %s\n"
            "  upgrades   : %s\n"
            "  selected   : %s\n"
            "  actions    : %s",
            int(feats.get("game_loop", 0.0)),
            units_str,
            buildings_str,
            upgrades_str,
            selected_str,
            actions_str,
        )

    @staticmethod
    def _format_set_or_dash(items: frozenset[str] | set[str]) -> str:
        if not items:
            return "-"
        return ", ".join(sorted(items))

    def _format_owned_units(self, feats: dict[str, float]) -> str:
        """Render friendly unit counts as e.g. ``Marine x4, SCV x12``.

        Pulls per-unit-type counts from the feature dict the env already
        builds (no extra obs traversal).
        """
        counts: list[tuple[str, int]] = []
        for key, value in feats.items():
            if not key.startswith("unit_count_"):
                continue
            n = int(value)
            if n <= 0:
                continue
            name = key[len("unit_count_") :]
            counts.append((name, n))
        if not counts:
            return "-"
        counts.sort()
        return ", ".join(f"{name} x{n}" for name, n in counts)

    def _format_available_actions(self, available_fn_ids: set[int] | None) -> str:
        """Render the available-fn mask as a multi-line list with selection hints.

        Each line: ``  - <fn_name> (<selection requirement>)``.  Selection
        requirements are pulled from
        :data:`games.sc2.tech_tree.PRECONDITIONS` so the user can see at
        a glance which actions are currently waiting on a select.
        """
        if not available_fn_ids:
            return "(none)"
        lines: list[str] = []
        for fn_idx in sorted(available_fn_ids):
            name = FUNCTION_IDS.get(fn_idx, f"fn_{fn_idx}")
            hint = self._selection_hint(fn_idx)
            lines.append(f"\n    - {name} ({hint})")
        return "".join(lines)

    @staticmethod
    def _selection_hint(fn_idx: int) -> str:
        pre = PRECONDITIONS.get(fn_idx)
        if pre is None:
            return "no requirement"
        if pre.required_selection == SelectionReq.NONE:
            return "no selection needed"
        if pre.required_selection == SelectionReq.ANY_UNIT:
            return "select any unit"
        targets = pre.selection_target
        if not targets:
            return "select unknown"
        return "select " + " or ".join(sorted(targets))

    @staticmethod
    def _build_attack_range_lookup() -> dict[int, float]:
        """Build {unit_type_id: attack_range_gu} for range-aware reward shaping."""
        try:
            from pysc2.lib import units as pysc2_units  # type: ignore[import-untyped]
        except ImportError:
            return {}
        lookup: dict[int, float] = {}
        races = (
            getattr(pysc2_units, "Terran", None),
            getattr(pysc2_units, "Protoss", None),
            getattr(pysc2_units, "Zerg", None),
            getattr(pysc2_units, "Neutral", None),
        )
        for race in races:
            if race is None:
                continue
            for member in race:
                r = _UNIT_ATTACK_RANGE_GU.get(member.name)
                if r is not None:
                    lookup[int(member.value)] = float(r)
        return lookup

    @staticmethod
    def _build_unit_type_lookup() -> dict[int, str]:
        """Build {unit_type_id: unit_name} for all SC2 units and structures.

        Previously restricted to _RICH_UNIT_TYPES (8 combat units) for the
        rich obs preset; issue #346 needs the full lookup so the tech-tree
        filter can recognise structures (CommandCenter, Barracks, Starport,
        …) and morph parents (Lair, Hive, SiegeTankSieged, …).

        Uses pysc2.lib.units when available; falls back to an empty dict so
        unit tests that don't install pysc2 still work (no tech filtering
        in that case — the mask reduces to the race ∩ PySC2 intersection
        as before).
        """
        try:
            from pysc2.lib import units as pysc2_units  # type: ignore[import-untyped]
        except ImportError:
            return {}

        lookup: dict[int, str] = {}
        races = (
            getattr(pysc2_units, "Terran", None),
            getattr(pysc2_units, "Protoss", None),
            getattr(pysc2_units, "Zerg", None),
            getattr(pysc2_units, "Neutral", None),
        )
        for race in races:
            if race is None:
                continue
            for member in race:
                lookup[int(member.value)] = member.name
        return lookup

    @staticmethod
    def _build_unit_type_race_lookup() -> dict[int, str]:
        """Build {unit_type_id: race_label} from pysc2.lib.units enums."""
        try:
            from pysc2.lib import units as pysc2_units  # type: ignore[import-untyped]
        except ImportError:
            return {}
        lookup: dict[int, str] = {}
        race_groups = (
            ("terran", getattr(pysc2_units, "Terran", None)),
            ("protoss", getattr(pysc2_units, "Protoss", None)),
            ("zerg", getattr(pysc2_units, "Zerg", None)),
        )
        for race_name, group in race_groups:
            if group is None:
                continue
            for member in group:
                lookup[int(member.value)] = race_name
        return lookup

    def _infer_fn_ids_from_units(self, ob: Any) -> set[int] | None:
        """Infer a race-consistent fn-id set from visible/selected own units.

        This supplements PySC2 available_actions mapping with unit/building-type
        context so obviously cross-race actions are masked out.
        """
        if self._unit_type_id_to_race is None:
            self._unit_type_id_to_race = self._build_unit_type_race_lookup()
        if not self._unit_type_id_to_race:
            return None

        race_votes: set[str] = set()
        feat_units = self._safe_array(ob, "feature_units")
        if (
            feat_units is not None
            and feat_units.size > 0
            and feat_units.ndim == 2
            and feat_units.shape[1] >= _FU.alliance + 1
        ):
            for row in feat_units:
                if int(row[_FU.alliance]) != 1:
                    continue
                race = self._unit_type_id_to_race.get(int(row[_FU.unit_type]))
                if race is not None:
                    race_votes.add(race)

        for key in ("single_select", "multi_select"):
            selected = self._safe_array(ob, key)
            if selected is None or selected.size == 0:
                continue
            if selected.ndim == 1:
                if selected.shape[0] < 1:
                    continue
                rows = selected[np.newaxis, :]
            elif selected.ndim == 2 and selected.shape[1] >= 1:
                rows = selected
            else:
                continue
            for row in rows:
                race = self._unit_type_id_to_race.get(int(row[_FU.unit_type]))
                if race is not None:
                    race_votes.add(race)

        if len(race_votes) != 1:
            return None
        inferred_race = next(iter(race_votes))
        return set(fn_ids_for_race(inferred_race))

    @staticmethod
    def _get_rich_unit_types() -> tuple:
        """Return the _RICH_UNIT_TYPES tuple (lazy import avoids circular deps)."""
        from games.sc2.obs_spec import _RICH_UNIT_TYPES

        return _RICH_UNIT_TYPES

    # ------------------------------------------------------------------
    # PySC2 observation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_player(ob: Any) -> dict:
        """PySC2 ``ob['player']`` is a NamedNumpyArray indexed by feature name.

        Field names are sourced from ``pysc2.lib.features.Player._fields``
        (lazily, so tests without pysc2 still work).  ``player_id`` is excluded
        because it is fixed per game and not useful as a policy input.
        """
        player = ob.get("player") if hasattr(ob, "get") else None
        if player is None:
            return {}
        result = {}
        for k in _get_player_field_names():
            try:
                v = player[k]
            except (KeyError, IndexError, TypeError):
                v = getattr(player, k, 0)
            result[k] = float(v) if v is not None else 0.0
        return result

    @staticmethod
    def _safe_array(ob: Any, key: str) -> np.ndarray | None:
        """Look up a key in the timestep observation, return None if missing."""
        try:
            value = ob[key] if hasattr(ob, "__getitem__") else None
        except (KeyError, IndexError, TypeError):
            value = None
        if value is None:
            value = getattr(ob, key, None)
        if value is None:
            return None
        if not isinstance(value, np.ndarray):
            try:
                value = np.asarray(value)
            except (TypeError, ValueError):
                return None
        return value

    @staticmethod
    def _extract_player_relative(feat: np.ndarray | None, screen: bool) -> np.ndarray | None:
        """PySC2 feature_screen / feature_minimap layers are indexable by name.

        We dig out the ``player_relative`` channel (1 = self, 4 = enemy).
        Returns None if the feature is unavailable.
        """
        if feat is None:
            return None
        # NamedNumpyArray-style access supports string indexing.
        try:
            return np.asarray(feat["player_relative"])
        except (KeyError, IndexError, TypeError, ValueError):
            pass
        # Fall back to the canonical channel index.
        # PySC2's SCREEN_FEATURES.player_relative.index = 5
        # PySC2's MINIMAP_FEATURES.player_relative.index = 5
        if feat.ndim == 3 and feat.shape[0] > 5:
            return np.asarray(feat[5])
        return None

    @staticmethod
    def _extract_visibility(feat: np.ndarray | None) -> np.ndarray | None:
        """Extract minimap ``visibility_map`` (0=hidden, 1=fogged, 2=visible)."""
        if feat is None:
            return None
        try:
            return np.asarray(feat["visibility_map"])
        except (KeyError, IndexError, TypeError, ValueError):
            pass
        # Canonical index 1.
        if feat.ndim == 3 and feat.shape[0] > 1:
            return np.asarray(feat[1])
        return None

    @staticmethod
    def _extract_named_layer(feat: np.ndarray | None, name: str) -> np.ndarray | None:
        """Extract a named feature layer from a PySC2 NamedNumpyArray."""
        if feat is None:
            return None
        try:
            return np.asarray(feat[name], dtype=np.float32)
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    @staticmethod
    def _resize_layer(layer: np.ndarray, target_size: int) -> np.ndarray:
        """Bilinear resize a 2-D feature layer to (target_size, target_size).

        Used to bring minimap layers to the same resolution as screen layers
        when ``minimap_size != screen_size``.
        """
        h, w = layer.shape
        if h == target_size and w == target_size:
            return layer
        row_idx = np.linspace(0, h - 1, target_size)
        col_idx = np.linspace(0, w - 1, target_size)
        r0 = np.floor(row_idx).astype(int).clip(0, h - 2)
        r1 = (r0 + 1).clip(0, h - 1)
        c0 = np.floor(col_idx).astype(int).clip(0, w - 2)
        c1 = (c0 + 1).clip(0, w - 1)
        dr = (row_idx - r0).astype(np.float32)
        dc = (col_idx - c0).astype(np.float32)
        out = (
            layer[np.ix_(r0, c0)] * np.outer(1 - dr, 1 - dc)
            + layer[np.ix_(r0, c1)] * np.outer(1 - dr, dc)
            + layer[np.ix_(r1, c0)] * np.outer(dr, 1 - dc)
            + layer[np.ix_(r1, c1)] * np.outer(dr, dc)
        )
        return out.astype(np.float32)

    @staticmethod
    def _centroid(mask: np.ndarray) -> tuple[float, float]:
        if mask.sum() == 0:
            return 0.0, 0.0
        ys, xs = np.where(mask)
        return float(np.mean(xs)), float(np.mean(ys))


# ---------------------------------------------------------------------------
# Shared, stateless observation extraction (issue #350)
# ---------------------------------------------------------------------------
# The functions below are the single implementation of the PySC2-TimeStep →
# flat-observation projection.  ``SC2Client._timestep_to_obs_info`` delegates
# its obs-building half to :func:`extract_flat_obs`, and the per-block instance
# methods on ``SC2Client`` are thin wrappers around the matching free function
# here.  The offline replay reader (BC, issue #349) imports these directly, so
# training and behaviour cloning never drift.
#
# Low-level PySC2 accessors (``_safe_array`` / ``_centroid`` / …) remain
# ``@staticmethod`` on ``SC2Client`` and are reused by reference; cross-step
# state that the live client accumulates is threaded explicitly via
# :class:`_ObsExtractState` rather than read off ``self``.


@dataclass
class _ObsExtractState:
    """Cross-step state needed to reproduce the live client's exact obs.

    A fresh instance is fine for a single-frame, history-free extraction
    (the replay reader can keep one per episode); ``SC2Client`` seeds one
    from ``self`` each step and writes the mutated fields back.
    """

    last_fn_idx: int = 0
    explored_mask: np.ndarray | None = None
    unit_type_lookup: dict[int, str] | None = None


def _player_features(ob: Any) -> dict[str, float]:
    player = SC2Client._safe_player(ob)
    return {
        "minerals": float(player.get("minerals", 0.0)),
        "vespene": float(player.get("vespene", 0.0)),
        "food_used": float(player.get("food_used", 0.0)),
        "food_cap": float(player.get("food_cap", 0.0)),
        "army_count": float(player.get("army_count", 0.0)),
        "idle_worker_count": float(player.get("idle_worker_count", 0.0)),
        "warp_gate_count": float(player.get("warp_gate_count", 0.0)),
        "larva_count": float(player.get("larva_count", 0.0)),
        "food_workers": float(player.get("food_workers", 0.0)),
        "food_army": float(player.get("food_army", 0.0)),
    }


def _selected_features(ob: Any) -> dict[str, float]:
    selected = SC2Client._safe_array(ob, "single_select")
    if selected is None or selected.size == 0:
        multi = SC2Client._safe_array(ob, "multi_select")
        if multi is not None and multi.size > 0:
            selected = multi
    if selected is not None and selected.size > 0:
        count = float(selected.shape[0]) if selected.ndim >= 2 else 1.0
        try:
            hp_col = selected[:, 2] if selected.ndim >= 2 else selected[2:3]
            avg_hp = float(np.mean(hp_col))
        except (IndexError, ValueError):
            avg_hp = 0.0
        try:
            shield_col = selected[:, 3] if selected.ndim >= 2 else selected[3:4]
            avg_shields = float(np.mean(shield_col))
        except (IndexError, ValueError):
            avg_shields = 0.0
        try:
            energy_col = selected[:, 4] if selected.ndim >= 2 else selected[4:5]
            avg_energy = float(np.mean(energy_col))
        except (IndexError, ValueError):
            avg_energy = 0.0
    else:
        count, avg_hp, avg_shields, avg_energy = 0.0, 0.0, 0.0, 0.0
    return {
        "selected_count": count,
        "selected_avg_hp": avg_hp,
        "selected_avg_shields": avg_shields,
        "selected_avg_energy": avg_energy,
    }


def _screen_summary_features(feat_screen: np.ndarray | None) -> dict[str, float]:
    out = {
        "screen_self_count": 0.0,
        "screen_enemy_count": 0.0,
        "screen_self_cx": 0.0,
        "screen_self_cy": 0.0,
        "screen_enemy_cx": 0.0,
        "screen_enemy_cy": 0.0,
    }
    layer = SC2Client._extract_player_relative(feat_screen, screen=True)
    if layer is None:
        return out
    self_mask = layer == 1
    enemy_mask = layer == 4
    out["screen_self_count"] = float(self_mask.sum())
    out["screen_enemy_count"] = float(enemy_mask.sum())
    out["screen_self_cx"], out["screen_self_cy"] = SC2Client._centroid(self_mask)
    out["screen_enemy_cx"], out["screen_enemy_cy"] = SC2Client._centroid(enemy_mask)
    return out


def _minimap_summary_features(
    feat_minimap: np.ndarray | None, explored_mask: np.ndarray | None
) -> tuple[dict[str, float], np.ndarray | None]:
    out = {
        "minimap_self_count": 0.0,
        "minimap_enemy_count": 0.0,
        "minimap_enemy_cx": 0.0,
        "minimap_enemy_cy": 0.0,
        "minimap_visible_frac": 0.0,
        "minimap_explored_frac": 0.0,
        "minimap_camera_x": 0.0,
        "minimap_camera_y": 0.0,
    }
    if feat_minimap is None:
        return out, explored_mask
    layer = SC2Client._extract_player_relative(feat_minimap, screen=False)
    if layer is not None:
        out["minimap_self_count"] = float((layer == 1).sum())
        enemy_mask = layer == 4
        out["minimap_enemy_count"] = float(enemy_mask.sum())
        out["minimap_enemy_cx"], out["minimap_enemy_cy"] = SC2Client._centroid(enemy_mask)
    visible = SC2Client._extract_visibility(feat_minimap)
    if visible is not None:
        out["minimap_visible_frac"] = float((visible == 2).sum()) / max(visible.size, 1)
        if explored_mask is None:
            explored_mask = (visible > 0).astype(bool)
        else:
            explored_mask |= visible > 0
        out["minimap_explored_frac"] = float(explored_mask.sum()) / max(explored_mask.size, 1)
    camera = SC2Client._extract_named_layer(feat_minimap, "camera")
    if camera is not None:
        cmask = camera > 0
        out["minimap_camera_x"], out["minimap_camera_y"] = SC2Client._centroid(cmask)
    return out, explored_mask


def _score_features(ob: Any) -> dict[str, float]:
    # Field names are sourced from pysc2.lib.features.ScoreCumulative._fields
    # (lazily) so we never duplicate them.  'score' is renamed 'score_total'
    # to avoid confusion with the per-step reward signal.
    names = _get_score_field_names()
    # Retrieve the raw value without coercing to ndarray so that PySC2's
    # NamedNumpyArray field-name access is preserved for the primary path.
    raw = None
    try:
        raw = ob["score_cumulative"] if hasattr(ob, "__getitem__") else None
    except (KeyError, IndexError, TypeError):
        pass
    if raw is None:
        raw = getattr(ob, "score_cumulative", None)
    if raw is None:
        return {n: 0.0 for n in names}
    # Precompute the positional-fallback array once, outside the per-field
    # loop.  For plain ndarray inputs every named-access attempt raises an
    # exception; detecting that here avoids 13 × 2 exception catches per
    # call and keeps the hot path clean.
    pos_arr: np.ndarray | None
    try:
        pos_arr = raw if isinstance(raw, np.ndarray) else np.asarray(raw)
        if pos_arr.ndim < 1:
            pos_arr = None
    except (TypeError, ValueError):
        pos_arr = None

    out: dict[str, float] = {}
    for i, n in enumerate(names):
        # Prefer field-name access for robustness against PySC2 schema
        # changes.  The rename score → score_total means we also try the
        # original PySC2 name ("score") for that entry.  Deduplicate when
        # the two names are identical (every field except score_total).
        pysc2_name = "score" if n == "score_total" else n
        attrs = (pysc2_name,) if pysc2_name == n else (pysc2_name, n)
        v = None
        for attr in attrs:
            try:
                v = float(raw[attr])
                break
            except (KeyError, IndexError, TypeError, ValueError):
                pass
        # Fall back to positional index when named access is unavailable.
        if v is None and pos_arr is not None and i < pos_arr.size:
            try:
                v = float(pos_arr[i])
            except (IndexError, TypeError, ValueError):
                pass
        out[n] = v if v is not None else 0.0
    return out


def _screen_hp_features(feat_screen: np.ndarray | None) -> dict[str, float]:
    out = {
        "screen_unit_density_mean": 0.0,
        "screen_self_hp_mean": 0.0,
        "screen_enemy_hp_mean": 0.0,
    }
    if feat_screen is None:
        return out
    density = SC2Client._extract_named_layer(feat_screen, "unit_density")
    if density is not None:
        out["screen_unit_density_mean"] = float(density.mean())
    hp = SC2Client._extract_named_layer(feat_screen, "unit_hit_points")
    rel = SC2Client._extract_player_relative(feat_screen, screen=True)
    if hp is not None and rel is not None:
        self_mask = rel == 1
        enemy_mask = rel == 4
        if self_mask.any():
            out["screen_self_hp_mean"] = float(hp[self_mask].mean())
        if enemy_mask.any():
            out["screen_enemy_hp_mean"] = float(hp[enemy_mask].mean())
    return out


def _topk_enemy_features(feat_screen: np.ndarray | None) -> dict[str, float]:
    # Counts within radii 8 and 24 of the friendly centroid plus the top-3
    # closest enemies' relative positions and HP ratios.
    out = {"topk_enemy_within_8": 0.0, "topk_enemy_within_24": 0.0}
    for i in range(3):
        out[f"topk_enemy_{i}_rel_x"] = 0.0
        out[f"topk_enemy_{i}_rel_y"] = 0.0
        out[f"topk_enemy_{i}_hp_ratio"] = 0.0
    if feat_screen is None:
        return out
    rel = SC2Client._extract_player_relative(feat_screen, screen=True)
    if rel is None:
        return out
    self_mask = rel == 1
    enemy_mask = rel == 4
    if not self_mask.any() or not enemy_mask.any():
        return out
    scx, scy = SC2Client._centroid(self_mask)
    ys, xs = np.where(enemy_mask)
    dx = xs.astype(np.float32) - scx
    dy = ys.astype(np.float32) - scy
    dist = np.sqrt(dx * dx + dy * dy)
    out["topk_enemy_within_8"] = float((dist <= 8.0).sum())
    out["topk_enemy_within_24"] = float((dist <= 24.0).sum())

    order = np.argsort(dist)[:3]
    hp_layer = SC2Client._extract_named_layer(feat_screen, "unit_hit_points_ratio")
    for k, idx in enumerate(order):
        out[f"topk_enemy_{k}_rel_x"] = float(dx[idx])
        out[f"topk_enemy_{k}_rel_y"] = float(dy[idx])
        if hp_layer is not None:
            # HP ratio layer is 0–255; normalise to [0, 1].
            out[f"topk_enemy_{k}_hp_ratio"] = float(hp_layer[ys[idx], xs[idx]]) / 255.0
    return out


def _per_unit_type_features(
    ob: Any, unit_type_lookup: dict[int, str] | None
) -> tuple[dict[str, float], dict[int, str] | None]:
    # Initialise every rich-preset unit type to zero.
    from games.sc2.obs_spec import _RICH_UNIT_TYPES

    out = {f"unit_count_{name}": 0.0 for name in _RICH_UNIT_TYPES}
    feat_units = SC2Client._safe_array(ob, "feature_units")
    if feat_units is None or feat_units.size == 0:
        return out, unit_type_lookup
    if feat_units.ndim != 2 or feat_units.shape[1] < _FU.alliance + 1:
        return out, unit_type_lookup
    if unit_type_lookup is None:
        unit_type_lookup = SC2Client._build_unit_type_lookup()
    for row, owner in zip(feat_units, feat_units[:, _FU.alliance]):
        if int(owner) != 1:
            continue
        unit_id = int(row[_FU.unit_type])
        name = unit_type_lookup.get(unit_id)
        if name is None:
            continue
        key = f"unit_count_{name}"
        if key in out:
            out[key] += 1.0
    return out, unit_type_lookup


def _quadrant_features(feat_screen: np.ndarray | None) -> dict[str, float]:
    out = {
        "screen_self_NE_count": 0.0,
        "screen_self_NW_count": 0.0,
        "screen_self_SE_count": 0.0,
        "screen_self_SW_count": 0.0,
        "screen_enemy_NE_count": 0.0,
        "screen_enemy_NW_count": 0.0,
        "screen_enemy_SE_count": 0.0,
        "screen_enemy_SW_count": 0.0,
    }
    rel = SC2Client._extract_player_relative(feat_screen, screen=True)
    if rel is None:
        return out
    h, w = rel.shape
    mid_y, mid_x = h // 2, w // 2
    for tag, value in (("self", 1), ("enemy", 4)):
        mask = rel == value
        ne = mask[:mid_y, mid_x:]
        nw = mask[:mid_y, :mid_x]
        se = mask[mid_y:, mid_x:]
        sw = mask[mid_y:, :mid_x]
        out[f"screen_{tag}_NE_count"] = float(ne.sum())
        out[f"screen_{tag}_NW_count"] = float(nw.sum())
        out[f"screen_{tag}_SE_count"] = float(se.sum())
        out[f"screen_{tag}_SW_count"] = float(sw.sum())
    return out


def _available_actions_features(ob: Any) -> dict[str, float]:
    n = len(FUNCTION_IDS)
    out = {f"available_fn_{i}": 0.0 for i in range(n)}
    avail = SC2Client._safe_array(ob, "available_actions")
    if avail is None:
        return out
    avail_set = set(int(x) for x in avail.tolist()) if avail.size > 0 else set()
    # Use the module-level cache to avoid per-step PySC2 attribute lookups
    # and repeated lazy imports.  _get_pysc2_id_to_fn_idx() resolves
    # PySC2 function metadata exactly once and caches the result.
    id_to_fn_idx = _get_pysc2_id_to_fn_idx()
    for pysc2_id, fn_idx in id_to_fn_idx.items():
        if pysc2_id in avail_set:
            out[f"available_fn_{fn_idx}"] = 1.0
    return out


def _last_action_features(last_fn_idx: int) -> dict[str, float]:
    n = len(FUNCTION_IDS)
    out = {f"last_fn_{i}": 0.0 for i in range(n)}
    if 0 <= last_fn_idx < n:
        out[f"last_fn_{last_fn_idx}"] = 1.0
    return out


def _enemy_unit_type_features(
    ob: Any, unit_type_lookup: dict[int, str] | None
) -> tuple[dict[str, float], dict[int, str] | None]:
    from games.sc2.obs_spec import _RICH_UNIT_TYPES

    out = {f"enemy_count_{name}": 0.0 for name in _RICH_UNIT_TYPES}
    feat_units = SC2Client._safe_array(ob, "feature_units")
    if feat_units is None or feat_units.size == 0:
        return out, unit_type_lookup
    if feat_units.ndim != 2 or feat_units.shape[1] < _FU.alliance + 1:
        return out, unit_type_lookup
    if unit_type_lookup is None:
        unit_type_lookup = SC2Client._build_unit_type_lookup()
    # PySC2 alliance values: 0=none/background, 1=self, 2=ally, 3=neutral, 4=enemy.
    for row, owner in zip(feat_units, feat_units[:, _FU.alliance]):
        if int(owner) != 4:
            continue
        name = unit_type_lookup.get(int(row[_FU.unit_type]))
        key = f"enemy_count_{name}"
        if name is not None and key in out:
            out[key] += 1.0
    return out, unit_type_lookup


def _shield_energy_features(feat_screen: np.ndarray | None) -> dict[str, float]:
    out = {
        "screen_self_shield_mean": 0.0,
        "screen_enemy_shield_mean": 0.0,
        "screen_self_energy_mean": 0.0,
    }
    if feat_screen is None:
        return out
    rel = SC2Client._extract_player_relative(feat_screen, screen=True)
    if rel is None:
        return out
    self_mask = rel == 1
    enemy_mask = rel == 4
    shield = SC2Client._extract_named_layer(feat_screen, "unit_shields")
    if shield is not None:
        if self_mask.any():
            out["screen_self_shield_mean"] = float(shield[self_mask].mean())
        if enemy_mask.any():
            out["screen_enemy_shield_mean"] = float(shield[enemy_mask].mean())
    energy = SC2Client._extract_named_layer(feat_screen, "unit_energy")
    if energy is not None and self_mask.any():
        out["screen_self_energy_mean"] = float(energy[self_mask].mean())
    return out


def _creep_features(feat_minimap: np.ndarray | None) -> dict[str, float]:
    out = {"minimap_creep_frac": 0.0}
    creep = SC2Client._extract_named_layer(feat_minimap, "creep")
    if creep is not None:
        out["minimap_creep_frac"] = float((creep > 0).sum()) / max(creep.size, 1)
    return out


def _economy_pipeline_features(ob: Any) -> dict[str, float]:
    out = {"upgrade_count": 0.0, "build_queue_size": 0.0, "cargo_count": 0.0}
    upgrades = SC2Client._safe_array(ob, "upgrades")
    if upgrades is not None:
        out["upgrade_count"] = float(upgrades.size)
    build_queue = SC2Client._safe_array(ob, "build_queue")
    if build_queue is not None and build_queue.ndim >= 1:
        out["build_queue_size"] = float(build_queue.shape[0] if build_queue.ndim >= 2 else build_queue.size)
    cargo = SC2Client._safe_array(ob, "cargo")
    if cargo is not None and cargo.ndim >= 1:
        out["cargo_count"] = float(cargo.shape[0] if cargo.ndim >= 2 else cargo.size)
    return out


def _screen_visibility_features(feat_screen: np.ndarray | None) -> dict[str, float]:
    out = {"screen_visibility_frac": 0.0}
    if feat_screen is None:
        return out
    vis = SC2Client._extract_named_layer(feat_screen, "visibility_map")
    if vis is not None:
        out["screen_visibility_frac"] = float((vis == 2).sum()) / max(vis.size, 1)
    return out


def _screen_antiair_features(feat_screen: np.ndarray | None) -> dict[str, float]:
    out = {"screen_unit_density_aa_mean": 0.0}
    if feat_screen is None:
        return out
    aa = SC2Client._extract_named_layer(feat_screen, "unit_density_aa")
    if aa is not None:
        out["screen_unit_density_aa_mean"] = float(aa.mean())
    return out


def _alerts_features(ob: Any) -> dict[str, float]:
    alerts = SC2Client._safe_array(ob, "alerts")
    return {"alert_count": float(alerts.size) if alerts is not None else 0.0}


def _weapon_cooldown_features(ob: Any) -> dict[str, float]:
    out = {"self_weapon_cooldown_mean": 0.0}
    feat_units = SC2Client._safe_array(ob, "feature_units")
    if feat_units is None or feat_units.size == 0:
        return out
    if feat_units.ndim != 2 or feat_units.shape[1] < _FU.weapon_cooldown + 1:
        return out
    self_mask = feat_units[:, _FU.alliance] == 1
    if not self_mask.any():
        return out
    cooldowns = feat_units[self_mask, _FU.weapon_cooldown].astype(np.float32)
    out["self_weapon_cooldown_mean"] = float(cooldowns.mean())
    return out


def extract_flat_obs(
    timestep: Any,
    obs_names: tuple[str, ...] | list[str],
    *,
    state: _ObsExtractState | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Project a PySC2 ``TimeStep`` onto a flat observation vector.

    This is the single source of truth for SC2 observation building, shared
    by the live :class:`SC2Client` and the offline replay reader (issue #350)
    so training and behaviour cloning use identical features.

    Parameters
    ----------
    timestep :
        A PySC2 ``TimeStep`` (anything exposing ``.observation``).
    obs_names :
        The ordered feature names of the active ``ObsSpec`` (``spec.names``).
        Names absent from the computed feature dict project to ``0.0`` — this
        is the standard "missing key → 0.0" migration path.
    state :
        Optional :class:`_ObsExtractState` carrying cross-step state
        (explored-mask accumulation, last-action one-hot index, unit-type
        lookup cache).  A fresh state is created when omitted; pass a
        persistent instance to reproduce the live client's exact,
        history-dependent output.  Mutated in place.

    Returns
    -------
    (flat, feats) :
        ``flat`` is the float32 vector projected onto ``obs_names``; ``feats``
        is the full ``{name: value}`` dict (the live client reuses it to build
        its info dict).
    """
    if state is None:
        state = _ObsExtractState()

    ob = timestep.observation
    feat_screen = SC2Client._safe_array(ob, "feature_screen")
    feat_minimap = SC2Client._safe_array(ob, "feature_minimap")

    feats: dict[str, float] = {}
    feats.update(_player_features(ob))
    feats.update(_selected_features(ob))
    feats.update(_screen_summary_features(feat_screen))
    minimap_feats, state.explored_mask = _minimap_summary_features(feat_minimap, state.explored_mask)
    feats.update(minimap_feats)
    feats.update(_score_features(ob))
    feats.update(_screen_hp_features(feat_screen))
    feats.update(_topk_enemy_features(feat_screen))
    unit_feats, state.unit_type_lookup = _per_unit_type_features(ob, state.unit_type_lookup)
    feats.update(unit_feats)
    feats.update(_quadrant_features(feat_screen))
    feats.update(_available_actions_features(ob))
    feats.update(_last_action_features(state.last_fn_idx))
    enemy_feats, state.unit_type_lookup = _enemy_unit_type_features(ob, state.unit_type_lookup)
    feats.update(enemy_feats)
    feats.update(_shield_energy_features(feat_screen))
    feats.update(_creep_features(feat_minimap))
    feats.update(_economy_pipeline_features(ob))
    feats.update(_screen_visibility_features(feat_screen))
    feats.update(_screen_antiair_features(feat_screen))
    feats.update(_weapon_cooldown_features(ob))
    feats.update(_alerts_features(ob))

    # game_loop scalar — present on both ladder and rich.
    game_loop_arr = SC2Client._safe_array(ob, "game_loop")
    game_loop = float(game_loop_arr[0]) if game_loop_arr is not None and game_loop_arr.size > 0 else 0.0
    feats["game_loop"] = game_loop

    # Project feature dict → ordered ndarray driven by the active spec.
    flat = np.array(
        [float(feats.get(name, 0.0)) for name in obs_names],
        dtype=np.float32,
    )
    return flat, feats
