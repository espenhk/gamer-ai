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

from games.sc2.actions import FUNCTION_IDS, action_to_function_call, pysc2_ids_to_internal_fn_idx
from games.sc2.obs_spec import (
    LADDER_OBS_NAMES,
    OBS_NAMES,
    SC2_LADDER_OBS_SPEC,
    SC2_MINIGAME_OBS_SPEC,
)

logger = logging.getLogger(__name__)

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
        self._spec = (
            SC2_LADDER_OBS_SPEC if self._is_ladder else SC2_MINIGAME_OBS_SPEC
        )
        self._obs_names = (
            LADDER_OBS_NAMES if self._is_ladder else OBS_NAMES
        )
        self._cumulative_score: float = 0.0
        self._explored_mask: np.ndarray | None = None
        self._available_actions: set[int] | None = None

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

        Falls back to ``no_op`` if the requested function is not currently
        available (PySC2 enforces preconditions like "have units selected").
        """
        from pysc2.lib import actions as pysc2_actions  # type: ignore[import-untyped]

        fn_call = action_to_function_call(action, self._screen_size)

        fn_idx = int(action[0])
        fn_name = FUNCTION_IDS.get(fn_idx, "no_op")

        if (
            self._available_actions is not None
            and int(fn_call.function) not in self._available_actions
        ):
            logger.debug(
                "Action %s blocked (not in available_actions); substituting no_op.",
                fn_name,
            )
            return pysc2_actions.FunctionCall(
                int(pysc2_actions.FUNCTIONS.no_op.id), []
            )

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

        Tolerates missing fields gracefully so the same code path works for
        minigames (no minimap visibility tracking) and ladder maps (with
        fog of war).
        """
        ob = timestep.observation
        player = self._safe_player(ob)

        minerals   = float(player.get("minerals", 0))
        vespene    = float(player.get("vespene", 0))
        food_used  = float(player.get("food_used", 0))
        food_cap   = float(player.get("food_cap", 0))
        army_count = float(player.get("army_count", 0))
        idle_workers = float(player.get("idle_worker_count", 0))
        warp_gates = float(player.get("warp_gate_count", 0))
        larva      = float(player.get("larva_count", 0))

        # Selected-unit summary.
        selected = self._safe_array(ob, "single_select")
        if selected is None or selected.size == 0:
            multi = self._safe_array(ob, "multi_select")
            if multi is not None and multi.size > 0:
                selected = multi
        if selected is not None and selected.size > 0:
            sel_count = float(selected.shape[0]) if selected.ndim >= 2 else 1.0
            # Column index 2 in PySC2 single/multi select = current health.
            try:
                hp_col = selected[:, 2] if selected.ndim >= 2 else selected[2:3]
                sel_avg_hp = float(np.mean(hp_col))
            except (IndexError, ValueError):
                sel_avg_hp = 0.0
        else:
            sel_count = 0.0
            sel_avg_hp = 0.0

        # Spatial summaries from the screen feature layer.
        screen_self_count, screen_enemy_count = 0.0, 0.0
        screen_self_cx, screen_self_cy = 0.0, 0.0
        screen_enemy_cx, screen_enemy_cy = 0.0, 0.0
        feat_screen = self._safe_array(ob, "feature_screen")
        player_relative_screen = self._extract_player_relative(feat_screen, screen=True)
        if player_relative_screen is not None:
            self_mask = player_relative_screen == 1
            enemy_mask = player_relative_screen == 4
            screen_self_count = float(self_mask.sum())
            screen_enemy_count = float(enemy_mask.sum())
            screen_self_cx, screen_self_cy = self._centroid(self_mask)
            screen_enemy_cx, screen_enemy_cy = self._centroid(enemy_mask)

        minigame_features = [
            minerals, vespene, food_used, food_cap, army_count,
            sel_count, sel_avg_hp,
            screen_self_count, screen_enemy_count,
            screen_self_cx, screen_self_cy,
            screen_enemy_cx, screen_enemy_cy,
        ]

        game_loop_arr = self._safe_array(ob, "game_loop")
        game_loop = float(game_loop_arr[0]) if game_loop_arr is not None and game_loop_arr.size > 0 else 0.0

        if not self._is_ladder:
            flat = np.array(minigame_features, dtype=np.float32)
        else:
            # Ladder extras: minimap-level stats.
            mmap = self._safe_array(ob, "feature_minimap")
            player_relative_mm = self._extract_player_relative(mmap, screen=False)
            visible_mm = self._extract_visibility(mmap)
            mm_self_count, mm_enemy_count = 0.0, 0.0
            visible_frac, explored_frac = 0.0, 0.0
            if player_relative_mm is not None:
                mm_self_count = float((player_relative_mm == 1).sum())
                mm_enemy_count = float((player_relative_mm == 4).sum())
            if visible_mm is not None:
                visible_frac = float((visible_mm == 2).sum()) / max(visible_mm.size, 1)
                if self._explored_mask is None:
                    self._explored_mask = (visible_mm > 0).astype(bool)
                else:
                    self._explored_mask |= (visible_mm > 0)
                explored_frac = float(self._explored_mask.sum()) / max(self._explored_mask.size, 1)

            ladder_features = minigame_features + [
                idle_workers, warp_gates, larva,
                mm_self_count, mm_enemy_count,
                visible_frac, explored_frac,
                game_loop,
            ]
            flat = np.array(ladder_features, dtype=np.float32)

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
        if avail_arr is not None:
            self._available_actions = set(avail_arr.tolist())

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
            "minerals": minerals,
            "vespene": vespene,
            "prev_minerals": 0.0,   # filled in by env on subsequent steps
            "prev_vespene": 0.0,
            "food_used": food_used,
            "food_cap": food_cap,
            "army_count": army_count,
            "player_outcome": player_outcome,
            "is_last": bool(timestep.last()),
<<<<<<< HEAD
            "available_fn_ids": set(self._available_actions) if self._available_actions is not None else set(),
            "game_loop": game_loop,
=======
            "available_fn_ids": pysc2_ids_to_internal_fn_idx(self._available_actions) if self._available_actions is not None else set(),
>>>>>>> 39dbbe5 (fix: correct Copilot review issues in SC2NeuralDQNPolicy)
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
