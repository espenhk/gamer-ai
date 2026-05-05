"""Thin wrapper around ``pysc2.env.sc2_env.SC2Env``.

Provides ``reset()`` / ``step()`` / ``close()`` returning the flat
``np.ndarray`` observation expected by :class:`games.sc2.env.SC2Env`,
plus an info dict with the raw scalars the reward calculator needs.

PySC2 import is lazy: importing this module does not pull pysc2 in, so
unit tests can mock the client without installing the SC2 binary.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from framework.obs_spec import ObsSpec
from games.sc2.actions import FUNCTION_IDS, action_to_function_call
from games.sc2.obs_spec import get_spec

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
    "player_relative":    4.0,
    "selected":           1.0,
    "unit_type":       1917.0,
    "height_map":       255.0,
    "unit_hit_points":  255.0,
    "unit_shields":     255.0,
    "unit_density":      16.0,
    "unit_density_aa":  255.0,
    "effects":           16.0,
    "visibility_map":     2.0,
    "unit_energy":      255.0,
    "creep":              1.0,
    "power":              1.0,
    "pathable":           1.0,
    "buildable":          1.0,
}

# Lazy cache: maps PySC2 native function ID → our fn_idx (0-5 in FUNCTION_IDS).
# Built on first use when pysc2 is available.
_pysc2_id_to_fn_idx: dict[int, int] | None = None


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
    play_mode :
        If True, set up a Human + Agent session instead of Agent (+ Bot).
        The human plays via the standard SC2 UI; the agent acts via PySC2.
    """

    def __init__(
        self,
        map_name: str,
        step_mul: int = 8,
        screen_size: int = 64,
        minimap_size: int = 64,
        agent_race: str = "random",
        bot_difficulty: str = "very_easy",
        visualize: bool = False,
        screen_layers: list[str] | None = None,
        minimap_layers: list[str] | None = None,
        play_mode: bool = False,
        obs_spec_preset: str | None = None,
    ) -> None:
        self._map_name = map_name
        self._step_mul = step_mul
        self._screen_size = screen_size
        self._minimap_size = minimap_size
        self._agent_race = agent_race
        self._bot_difficulty = bot_difficulty
        self._visualize = visualize
        self._play_mode = play_mode
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
        self._cumulative_score: float = 0.0
        self._explored_mask: np.ndarray | None = None
        self._available_actions: set[int] | None = None
        self._last_fn_idx: int = 0  # for last_fn_* one-hot in rich preset
        # Lookup table for unit-type ids → label, populated lazily so unit
        # tests don't import pysc2.lib.units at module load.
        self._unit_type_id_to_name: dict[int, str] | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, dict]:
        """Initialise the SC2 env and return the first observation + info."""
        if self._sc2_env is None:
            self._sc2_env = self._make_sc2_env()
        timesteps = self._sc2_env.reset()
        self._cumulative_score = 0.0
        self._explored_mask = None
        self._available_actions = None
        self._last_fn_idx = 0
        return self._timestep_to_obs_info(timesteps[0])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Apply an action and return ``(obs, score, done, info)``.

        The middle return value is the raw PySC2 reward signal — for
        minigames this is the score increment; for ladder maps it is the
        terminal +1 / -1 / 0.  The reward calculator computes the actual
        training reward in :class:`games.sc2.env.SC2Env`.
        """
        fn_call = self._action_to_call(action)
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
            _absl_flags.FLAGS([''])

        if self._play_mode:
            # Human (via SC2 UI) vs AI agent.  PySC2 only takes step actions
            # for Agent slots; Human actions come from the game client directly.
            agents = [
                sc2_env.Human(self._race(sc2_env)),
                sc2_env.Agent(self._race(sc2_env), "ai_agent"),
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

    def _action_to_call(self, action: np.ndarray) -> Any:
        """Translate a 4-vector action to a PySC2 ``FunctionCall``.

        When the requested function is not available (PySC2 enforces
        preconditions like "have units selected"), substitute either
        ``select_army`` or ``no_op`` depending on context.

        Issues #121 / #124: if the policy issues a unit-targeted action
        (``Move_screen`` / ``Attack_screen`` / ``Harvest_Gather_screen``)
        but no army is selected, fall back to ``select_army`` rather than
        ``no_op``.  Otherwise the agent appears to "idle" — the next step
        has the same observation, the policy emits the same blocked
        action, and PySC2 keeps no-op'ing it until the policy stochastically
        elects ``select_army`` itself.  Auto-selecting closes that gap and
        ensures the very next step can actually move.
        """
        from pysc2.lib import actions as pysc2_actions  # type: ignore[import-untyped]

        fn_call = action_to_function_call(action, self._screen_size)

        fn_idx = int(action[0])
        fn_name = FUNCTION_IDS.get(fn_idx, "no_op")

        if (
            self._available_actions is not None
            and int(fn_call.function) not in self._available_actions
        ):
            select_army_id = int(pysc2_actions.FUNCTIONS.select_army.id)
            unit_targeted = fn_name in (
                "Move_screen", "Attack_screen", "Harvest_Gather_screen",
            )
            if unit_targeted and select_army_id in self._available_actions:
                logger.debug(
                    "Action %s blocked; auto-selecting army (#124).", fn_name,
                )
                # Reflect the executed action, not the requested one, so the
                # rich preset's last_fn_* one-hot stays consistent.
                self._last_fn_idx = 1  # FUNCTION_IDS index for select_army
                return pysc2_actions.FunctionCall(select_army_id, [[0]])
            logger.debug(
                "Action %s blocked (not in available_actions); substituting no_op.",
                fn_name,
            )
            self._last_fn_idx = 0      # FUNCTION_IDS index for no_op
            return pysc2_actions.FunctionCall(
                int(pysc2_actions.FUNCTIONS.no_op.id), []
            )

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
                    fn_name, x_screen, y_screen, queue,
                )

        return fn_call

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
        feat_screen  = self._safe_array(ob, "feature_screen")
        feat_minimap = self._safe_array(ob, "feature_minimap")

        feats: dict[str, float] = {}
        feats.update(self._player_features(ob))
        feats.update(self._selected_features(ob))
        feats.update(self._screen_summary_features(feat_screen))
        feats.update(self._minimap_summary_features(feat_minimap))
        feats.update(self._score_features(ob))
        feats.update(self._screen_hp_features(feat_screen))
        feats.update(self._topk_enemy_features(feat_screen))
        feats.update(self._per_unit_type_features(ob))
        feats.update(self._quadrant_features(feat_screen))
        feats.update(self._available_actions_features(ob))
        feats.update(self._last_action_features())

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

        # Track available actions for precondition checking in _action_to_call.
        avail_arr = self._safe_array(ob, "available_actions")
        available_fn_ids: set[int] | None = None
        if avail_arr is not None:
            self._available_actions = set(avail_arr.tolist())
            id_map = _get_pysc2_id_to_fn_idx()
            if id_map:
                available_fn_ids = {
                    id_map[pid]
                    for pid in self._available_actions
                    if pid in id_map
                }

        # player_outcome is only meaningful for ladder maps where PySC2 emits
        # a terminal +1 / -1 / 0.  For minigames timestep.reward is a per-step
        # score delta, not a win/loss signal, so we leave it as None.
        if timestep.last() and self._is_ladder:
            player_outcome: float | None = float(
                getattr(timestep, "reward", 0.0) or 0.0
            )
        else:
            player_outcome = None

        info = {
            "score": cumulative,
            "prev_score": prev_score,
            "minerals": feats.get("minerals", 0.0),
            "vespene": feats.get("vespene", 0.0),
            "prev_minerals": 0.0,   # filled in by env on subsequent steps
            "prev_vespene": 0.0,
            "food_used": feats.get("food_used", 0.0),
            "food_cap": feats.get("food_cap", 0.0),
            "army_count": feats.get("army_count", 0.0),
            "player_outcome": player_outcome,
            "is_last": bool(timestep.last()),
            "available_fn_ids": available_fn_ids,
            "game_loop": game_loop,
            # Screen summary used by reward shaping (idle_bonus, #127).
            "screen_self_count":  feats.get("screen_self_count", 0.0),
            "screen_enemy_count": feats.get("screen_enemy_count", 0.0),
            "screen_self_cx":     feats.get("screen_self_cx", 0.0),
            "screen_self_cy":     feats.get("screen_self_cy", 0.0),
            "screen_enemy_cx":    feats.get("screen_enemy_cx", 0.0),
            "screen_enemy_cy":    feats.get("screen_enemy_cy", 0.0),
        }

        # Spatial obs: stack selected screen + minimap layers into (C, H, W).
        if self._screen_layers or self._minimap_layers:
            channels: list[np.ndarray] = []
            for name in self._screen_layers:
                layer = self._extract_named_layer(feat_screen, name)
                scale = _LAYER_SCALE.get(name, 1.0)
                if layer is not None:
                    channels.append((layer / scale).astype(np.float32))
                else:
                    channels.append(
                        np.zeros((self._screen_size, self._screen_size), dtype=np.float32)
                    )
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
                        channels.append(
                            np.zeros((self._screen_size, self._screen_size), dtype=np.float32)
                        )
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
        player = self._safe_player(ob)
        return {
            "minerals":          float(player.get("minerals", 0.0)),
            "vespene":           float(player.get("vespene", 0.0)),
            "food_used":         float(player.get("food_used", 0.0)),
            "food_cap":          float(player.get("food_cap", 0.0)),
            "army_count":        float(player.get("army_count", 0.0)),
            "idle_worker_count": float(player.get("idle_worker_count", 0.0)),
            "warp_gate_count":   float(player.get("warp_gate_count", 0.0)),
            "larva_count":       float(player.get("larva_count", 0.0)),
            "food_workers":      float(player.get("food_workers", 0.0)),
            "food_army":         float(player.get("food_army", 0.0)),
        }

    def _selected_features(self, ob: Any) -> dict[str, float]:
        selected = self._safe_array(ob, "single_select")
        if selected is None or selected.size == 0:
            multi = self._safe_array(ob, "multi_select")
            if multi is not None and multi.size > 0:
                selected = multi
        if selected is not None and selected.size > 0:
            count = float(selected.shape[0]) if selected.ndim >= 2 else 1.0
            try:
                hp_col = selected[:, 2] if selected.ndim >= 2 else selected[2:3]
                avg_hp = float(np.mean(hp_col))
            except (IndexError, ValueError):
                avg_hp = 0.0
        else:
            count, avg_hp = 0.0, 0.0
        return {"selected_count": count, "selected_avg_hp": avg_hp}

    def _screen_summary_features(self, feat_screen: np.ndarray | None) -> dict[str, float]:
        out = {
            "screen_self_count":  0.0,
            "screen_enemy_count": 0.0,
            "screen_self_cx":     0.0,
            "screen_self_cy":     0.0,
            "screen_enemy_cx":    0.0,
            "screen_enemy_cy":    0.0,
        }
        layer = self._extract_player_relative(feat_screen, screen=True)
        if layer is None:
            return out
        self_mask  = layer == 1
        enemy_mask = layer == 4
        out["screen_self_count"]  = float(self_mask.sum())
        out["screen_enemy_count"] = float(enemy_mask.sum())
        out["screen_self_cx"], out["screen_self_cy"]   = self._centroid(self_mask)
        out["screen_enemy_cx"], out["screen_enemy_cy"] = self._centroid(enemy_mask)
        return out

    def _minimap_summary_features(self, feat_minimap: np.ndarray | None) -> dict[str, float]:
        out = {
            "minimap_self_count":   0.0,
            "minimap_enemy_count":  0.0,
            "minimap_visible_frac": 0.0,
            "minimap_explored_frac":0.0,
            "minimap_camera_x":     0.0,
            "minimap_camera_y":     0.0,
        }
        if feat_minimap is None:
            return out
        layer = self._extract_player_relative(feat_minimap, screen=False)
        if layer is not None:
            out["minimap_self_count"]  = float((layer == 1).sum())
            out["minimap_enemy_count"] = float((layer == 4).sum())
        visible = self._extract_visibility(feat_minimap)
        if visible is not None:
            out["minimap_visible_frac"] = float((visible == 2).sum()) / max(visible.size, 1)
            if self._explored_mask is None:
                self._explored_mask = (visible > 0).astype(bool)
            else:
                self._explored_mask |= (visible > 0)
            out["minimap_explored_frac"] = float(self._explored_mask.sum()) / max(self._explored_mask.size, 1)
        camera = self._extract_named_layer(feat_minimap, "camera")
        if camera is not None:
            cmask = camera > 0
            out["minimap_camera_x"], out["minimap_camera_y"] = self._centroid(cmask)
        return out

    def _score_features(self, ob: Any) -> dict[str, float]:
        score_arr = self._safe_array(ob, "score_cumulative")
        names = (
            "score_total", "idle_production_time", "idle_worker_time",
            "total_value_units", "total_value_structures",
            "killed_value_units", "killed_value_structures",
            "collected_minerals", "collected_vespene",
            "collection_rate_minerals", "collection_rate_vespene",
            "spent_minerals", "spent_vespene",
        )
        if score_arr is None:
            return {n: 0.0 for n in names}
        out: dict[str, float] = {}
        for i, n in enumerate(names):
            out[n] = float(score_arr[i]) if i < score_arr.size else 0.0
        return out

    def _screen_hp_features(self, feat_screen: np.ndarray | None) -> dict[str, float]:
        out = {
            "screen_unit_density_mean": 0.0,
            "screen_self_hp_mean":      0.0,
            "screen_enemy_hp_mean":     0.0,
        }
        if feat_screen is None:
            return out
        density = self._extract_named_layer(feat_screen, "unit_density")
        if density is not None:
            out["screen_unit_density_mean"] = float(density.mean())
        hp = self._extract_named_layer(feat_screen, "unit_hit_points")
        rel = self._extract_player_relative(feat_screen, screen=True)
        if hp is not None and rel is not None:
            self_mask  = rel == 1
            enemy_mask = rel == 4
            if self_mask.any():
                out["screen_self_hp_mean"] = float(hp[self_mask].mean())
            if enemy_mask.any():
                out["screen_enemy_hp_mean"] = float(hp[enemy_mask].mean())
        return out

    def _topk_enemy_features(self, feat_screen: np.ndarray | None) -> dict[str, float]:
        # Counts within radii 8 and 24 of the friendly centroid plus the top-3
        # closest enemies' relative positions and HP ratios.
        out = {"topk_enemy_within_8": 0.0, "topk_enemy_within_24": 0.0}
        for i in range(3):
            out[f"topk_enemy_{i}_rel_x"] = 0.0
            out[f"topk_enemy_{i}_rel_y"] = 0.0
            out[f"topk_enemy_{i}_hp_ratio"] = 0.0
        if feat_screen is None:
            return out
        rel = self._extract_player_relative(feat_screen, screen=True)
        if rel is None:
            return out
        self_mask  = rel == 1
        enemy_mask = rel == 4
        if not self_mask.any() or not enemy_mask.any():
            return out
        scx, scy = self._centroid(self_mask)
        ys, xs = np.where(enemy_mask)
        dx = xs.astype(np.float32) - scx
        dy = ys.astype(np.float32) - scy
        dist = np.sqrt(dx * dx + dy * dy)
        out["topk_enemy_within_8"]  = float((dist <= 8.0).sum())
        out["topk_enemy_within_24"] = float((dist <= 24.0).sum())

        order = np.argsort(dist)[:3]
        hp_layer = self._extract_named_layer(feat_screen, "unit_hit_points_ratio")
        for k, idx in enumerate(order):
            out[f"topk_enemy_{k}_rel_x"] = float(dx[idx])
            out[f"topk_enemy_{k}_rel_y"] = float(dy[idx])
            if hp_layer is not None:
                # HP ratio layer is 0–255; normalise to [0, 1].
                out[f"topk_enemy_{k}_hp_ratio"] = float(hp_layer[ys[idx], xs[idx]]) / 255.0
        return out

    def _per_unit_type_features(self, ob: Any) -> dict[str, float]:
        # Initialise every rich-preset unit type to zero.
        from games.sc2.obs_spec import _RICH_UNIT_TYPES
        out = {f"unit_count_{name}": 0.0 for name in _RICH_UNIT_TYPES}
        feat_units = self._safe_array(ob, "feature_units")
        if feat_units is None or feat_units.size == 0:
            return out
        # PySC2's feature_units rows have unit_type at column 0 and owner
        # (player relative) at column 1 in standard schemas; tolerate
        # missing columns by short-circuiting on shape.
        if feat_units.ndim != 2 or feat_units.shape[1] < 2:
            return out
        if self._unit_type_id_to_name is None:
            self._unit_type_id_to_name = self._build_unit_type_lookup()
        owners = feat_units[:, 1]
        # PySC2 owner values: 1 = self, others (4 = enemy) excluded for the
        # friendly count.
        for row, owner in zip(feat_units, owners):
            if int(owner) != 1:
                continue
            unit_id = int(row[0])
            name = self._unit_type_id_to_name.get(unit_id)
            if name is None:
                continue
            key = f"unit_count_{name}"
            if key in out:
                out[key] += 1.0
        return out

    def _quadrant_features(self, feat_screen: np.ndarray | None) -> dict[str, float]:
        out = {
            "screen_self_NE_count":  0.0, "screen_self_NW_count":  0.0,
            "screen_self_SE_count":  0.0, "screen_self_SW_count":  0.0,
            "screen_enemy_NE_count": 0.0, "screen_enemy_NW_count": 0.0,
            "screen_enemy_SE_count": 0.0, "screen_enemy_SW_count": 0.0,
        }
        rel = self._extract_player_relative(feat_screen, screen=True)
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

    def _available_actions_features(self, ob: Any) -> dict[str, float]:
        n = len(FUNCTION_IDS)
        out = {f"available_fn_{i}": 0.0 for i in range(n)}
        avail = self._safe_array(ob, "available_actions")
        if avail is None:
            return out
        avail_set = set(int(x) for x in avail.tolist()) if avail.size > 0 else set()
        # Map our FUNCTION_IDS table indices to PySC2 function IDs lazily.
        try:
            from pysc2.lib import actions as pysc2_actions  # type: ignore[import-untyped]
        except ImportError:
            return out
        for i, name in FUNCTION_IDS.items():
            fn = getattr(pysc2_actions.FUNCTIONS, name, None)
            if fn is not None and int(fn.id) in avail_set:
                out[f"available_fn_{i}"] = 1.0
        return out

    def _last_action_features(self) -> dict[str, float]:
        n = len(FUNCTION_IDS)
        out = {f"last_fn_{i}": 0.0 for i in range(n)}
        if 0 <= self._last_fn_idx < n:
            out[f"last_fn_{self._last_fn_idx}"] = 1.0
        return out

    @staticmethod
    def _build_unit_type_lookup() -> dict[int, str]:
        """Build {unit_type_id: race_label} for the rich preset's unit-type counts.

        Uses pysc2.lib.units when available; falls back to an empty dict so
        unit tests that don't install pysc2 still work (the rich preset will
        report zero unit-type counts in that case).
        """
        try:
            from pysc2.lib import units as pysc2_units  # type: ignore[import-untyped]
        except ImportError:
            return {}
        from games.sc2.obs_spec import _RICH_UNIT_TYPES
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
                if member.name in _RICH_UNIT_TYPES:
                    lookup[int(member.value)] = member.name
        return lookup

    # ------------------------------------------------------------------
    # PySC2 observation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_player(ob: Any) -> dict:
        """PySC2 ``ob['player']`` is a NamedNumpyArray indexed by feature name."""
        player = ob.get("player") if hasattr(ob, "get") else None
        if player is None:
            return {}
        # NamedNumpyArray: support both attribute access and dict-like access.
        result = {}
        keys = (
            "minerals", "vespene", "food_used", "food_cap",
            "army_count", "idle_worker_count", "warp_gate_count", "larva_count",
            "food_workers", "food_army",
        )
        for k in keys:
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
