"""SC2Env — Gymnasium environment wrapping PySC2 for RL training.

Observation space
-----------------
Either ``SC2_MINIGAME_OBS_SPEC`` (13 dims) or ``SC2_LADDER_OBS_SPEC`` (21
dims) depending on the map.  See :mod:`games.sc2.obs_spec`.

Action space
------------
``Box(low=[0, 0, 0, 0], high=[N_FUNCS-1, 1, 1, 1], shape=(4,), dtype=float32)``

  [0] fn_idx — integer in ``[0, len(FUNCTION_IDS)-1]`` (cast to int at exec)
  [1] x      — normalised screen target x in ``[0, 1]``
  [2] y      — normalised screen target y in ``[0, 1]``
  [3] queue  — 0 or 1 (queue the order)

The discrete-action policies use :data:`games.sc2.actions.DISCRETE_ACTIONS`
which is a 9×4 grid in this Box space.

Episode lifecycle
-----------------
``reset()`` → first PySC2 timestep → flat obs, info
``step()``  → apply FunctionCall, read next timestep, compute reward
Terminated when PySC2 marks ``timestep.last()``.
Truncated when wall-clock elapsed > ``max_episode_time_s``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from framework.base_env import BaseGameEnv
from framework.belief import EWMABelief
from framework.info_gain import RegionStalenessTracker
from games.sc2.actions import DISCRETE_ACTIONS, FUNCTION_IDS
from games.sc2.apm_limiter import ApmLimiter
from games.sc2.client import SC2Client
from games.sc2.obs_spec import MINIGAME_NAMES, get_spec
from games.sc2.reward import SC2RewardCalculator, SC2RewardConfig

# PySC2 runs at 22.4 game ticks per real second.
_SC2_TICKS_PER_S: float = 22.4
# Record one army/resource series point every N steps to bound memory.
# At step_mul=1 a 10-min game produces ~13 400 steps; sampling at 10 gives
# ~1 340 points per series — enough resolution for analytics charts.
_SERIES_SAMPLE_RATE: int = 10

logger = logging.getLogger(__name__)


class SC2Env(BaseGameEnv):
    """Gymnasium environment for StarCraft 2 reinforcement learning.

    Parameters
    ----------
    map_name :
        PySC2 map name (e.g. ``MoveToBeacon``, ``Simple64``).
    reward_config :
        :class:`SC2RewardConfig` instance.  If None, uses Python defaults.
    max_episode_time_s :
        Wall-clock seconds before the episode is truncated.
    step_mul :
        Game-tick multiplier per env step.  SC2 runs at 22.4 ticks/second, so
        one env step advances ``step_mul / 22.4`` seconds of *game* time.  The
        default of 8 is ~357 ms of game time per step, or a theoretical ceiling
        of ~168 game-time APM if the policy acts every step.

        **Interaction with max_apm:** the APM limiter measures *wall-clock*
        time (``time.monotonic()``), not game time.  PySC2 runs faster than
        real-time by default (no ``realtime=True`` is set), so env steps arrive
        faster in wall-clock terms than the game-time formula implies.  As a
        result, setting ``max_apm`` to a value that looks safe relative to the
        168-APM game-time ceiling can still trigger throttling in practice.
        For example, with ``step_mul=8`` and training running at 2× real-time
        speed, the agent attempts ~336 actions per wall-clock minute; a
        ``max_apm=300`` limit would throttle roughly 1 in 9 of those actions.
        If you do not want APM limiting, leave ``max_apm`` unset.
    screen_size, minimap_size :
        Square feature-layer resolutions.
    agent_race :
        Race string (``"random"``, ``"protoss"``, ``"terran"``, ``"zerg"``).
    bot_difficulty :
        Bot difficulty for 1v1 maps; ignored for minigames.
    visualize :
        If True, render the PySC2 visualizer.
    screen_layers :
        PySC2 feature_screen layer names to stack as spatial obs channels.
        When non-empty the observation space becomes a ``Dict`` with keys
        ``"flat"`` (the existing vector) and ``"spatial"`` (C × H × W).
    minimap_layers :
        PySC2 feature_minimap layer names appended after screen_layers.
    max_apm :
        Optional maximum actions per minute enforced by a token-bucket
        limiter measured in **game time** (derived from PySC2's ``game_loop``
        counter, not wall-clock time).  When the budget is exhausted,
        non-no-op actions are replaced with ``no_op``.  ``None`` (default)
        disables limiting.  No-op actions (``fn_idx == 0``) are always free.

        Because the limiter uses game time, ``max_apm=300`` means exactly
        300 actions per in-game minute regardless of whether training runs
        faster or slower than real-time.
    apm_burst_s :
        Token-bucket burst window in seconds.  Controls how many seconds'
        worth of tokens can accumulate before the limiter kicks in.
        Defaults to ``2.0`` — short bursts are fine, but the agent cannot
        spend its entire per-minute budget instantaneously.
    extreme_random_run_count :
        Number of initial episodes to execute in extreme-random mode. In this
        mode each step samples a random *valid* function id and random spatial
        target coordinates.
    realtime :
        If True, synchronise env steps with wall-clock time so the game
        advances at natural game pace rather than as fast as possible.
        Useful for evaluation / watch sessions.  Forwarded verbatim to
        the underlying :class:`~games.sc2.client.SC2Client`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_name: str = "MoveToBeacon",
        reward_config: SC2RewardConfig | None = None,
        max_episode_time_s: float = 120.0,
        step_mul: int = 1,
        screen_size: int = 64,
        minimap_size: int = 64,
        agent_race: str = "random",
        bot_difficulty: str = "very_easy",
        visualize: bool = False,
        screen_layers: list[str] | None = None,
        minimap_layers: list[str] | None = None,
        obs_spec_preset: str | None = None,
        enable_belief: bool = False,
        max_apm: int | None = None,
        apm_burst_s: float = 2.0,
        extreme_random_run_count: int = 0,
        realtime: bool = False,
        self_play: bool = False,
        opponent_policy: Any = None,
    ) -> None:
        super().__init__()

        self._map_name = map_name
        self._is_ladder = map_name not in MINIGAME_NAMES
        self._reward_config = reward_config or SC2RewardConfig()
        self._max_episode_time_s = max_episode_time_s
        self._step_mul = step_mul
        self._screen_size = screen_size
        self._reward_calc = SC2RewardCalculator(self._reward_config)
        self._screen_layers: list[str] = list(screen_layers or [])
        self._minimap_layers: list[str] = list(minimap_layers or [])
        self._use_spatial = bool(self._screen_layers or self._minimap_layers)
        self._obs_spec_preset = obs_spec_preset

        spec = get_spec(map_name, preset=obs_spec_preset)

        self._belief_cfg: dict | None = None
        self._belief: EWMABelief | None = None
        self._info_gain: RegionStalenessTracker | None = None
        self._prev_game_loop: float = 0.0
        if enable_belief:
            from games.sc2.belief_schema import (
                extend_obs_spec,
                load_belief_config,
                make_belief,
                make_info_gain,
            )

            _belief_cfg_path = Path(__file__).parent / "config" / "belief_config.yaml"
            self._belief_cfg = load_belief_config(_belief_cfg_path)
            self._belief = make_belief(self._belief_cfg)
            self._info_gain = make_info_gain(self._belief_cfg)
            spec = extend_obs_spec(spec, self._belief_cfg)
        flat_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(spec.dim,),
            dtype=np.float32,
        )
        if self._use_spatial:
            n_channels = len(self._screen_layers) + len(self._minimap_layers)
            self.observation_space = spaces.Dict(
                {
                    "flat": flat_space,
                    "spatial": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(n_channels, screen_size, screen_size),
                        dtype=np.float32,
                    ),
                }
            )
            self._spatial_shape = (n_channels, screen_size, screen_size)
        else:
            self.observation_space = flat_space
            self._spatial_shape = None

        n_funcs = max(FUNCTION_IDS) + 1
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([float(n_funcs - 1), 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._client = SC2Client(
            map_name=map_name,
            step_mul=step_mul,
            screen_size=screen_size,
            minimap_size=minimap_size,
            agent_race=agent_race,
            bot_difficulty=bot_difficulty,
            visualize=visualize,
            realtime=realtime,
            screen_layers=self._screen_layers,
            minimap_layers=self._minimap_layers,
            obs_spec_preset=obs_spec_preset,
            store_minimap_vis=enable_belief,
            extreme_random_run_count=extreme_random_run_count,
            self_play=self_play,
            opponent_policy=opponent_policy,
        )

        # APM limiter — None when no limit is requested.
        self._apm_limiter: ApmLimiter | None = ApmLimiter(max_apm, burst_s=apm_burst_s) if max_apm is not None else None

        # Episode tracking
        self._prev_obs: np.ndarray | None = None
        self._prev_minerals: float = 0.0
        self._prev_vespene: float = 0.0
        self._prev_score: float = 0.0
        self._prev_army_count: float = 0.0
        self._prev_total_self_hp: float = 0.0
        self._prev_move_target: tuple[float, float] | None = None
        self._elapsed_s: float = 0.0
        self._episode_start_s: float = 0.0
        self._step_count: int = 0
        self._ep_reward_components: dict[str, float] = {}
        # Per-episode action and observation tracking (analytics 2a/2c/2d).
        self._ep_action_counts: dict[int, int] = {}
        self._ep_obs_sums: dict[str, float] = {}
        self._ep_obs_step_count: int = 0
        self._ep_xy_hist: np.ndarray = np.zeros((8, 8), dtype=np.int64)
        self._ep_apm_throttled_steps: int = 0
        self._apm_game_time_s: float = 0.0
        self._prev_step_game_loop: float | None = None
        self._ep_skipped_frames: int = 0
        # Per-episode SC2 end-screen analytics (supply cap, time-series, build order).
        self._ep_supply_capped_steps: int = 0
        self._ep_army_series: list = []  # [[game_time_s, army_count], ...]
        self._ep_resource_series: list = []  # [[game_time_s, minerals+vespene], ...]
        self._ep_build_order: list = []  # [[game_time_s, unit_name], ...]
        self._ep_prev_unit_counts: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray | dict, dict]:
        super().reset(seed=seed)

        flat_obs, info = self._client.reset()
        obs = self._make_obs(flat_obs, info)

        self._prev_minerals = info.get("minerals", 0.0)
        self._prev_vespene = info.get("vespene", 0.0)
        self._prev_score = info.get("score", 0.0)
        self._prev_army_count = info.get("army_count", 0.0)
        self._prev_total_self_hp = info.get("total_self_hp", 0.0)
        self._prev_move_target = None
        self._elapsed_s = 0.0
        self._episode_start_s = time.monotonic()
        self._step_count = 0
        self._ep_reward_components = {}
        self._ep_action_counts = {}
        self._ep_obs_sums = {}
        self._ep_obs_step_count = 0
        self._ep_xy_hist = np.zeros((8, 8), dtype=np.int64)
        self._ep_apm_throttled_steps = 0
        self._prev_step_game_loop = self._safe_float_or_none(info.get("game_loop"))
        self._apm_game_time_s = (self._prev_step_game_loop or 0.0) / _SC2_TICKS_PER_S
        self._ep_skipped_frames = 0
        self._ep_supply_capped_steps = 0
        self._ep_army_series = []
        self._ep_resource_series = []
        self._ep_build_order = []
        # Seed from the reset observation so units already present at episode
        # start (e.g. starting SCVs) are not recorded as "built" build-order
        # events on the first step.
        self._ep_prev_unit_counts = dict(info.get("unit_counts") or {})
        self._reward_calc.reset()
        self._prev_game_loop = float(info.get("game_loop", 0.0))

        if self._apm_limiter is not None:
            self._apm_limiter.reset(self._apm_game_time_s)

        if self._belief is not None:
            self._belief.reset()
            self._info_gain.reset()
            _benc = self._belief.encode()
            _senc = self._info_gain.staleness().astype(np.float32)
            if self._use_spatial:
                obs = {"flat": np.concatenate([obs["flat"], _benc, _senc]), "spatial": obs["spatial"]}
            else:
                obs = np.concatenate([obs, _benc, _senc])

        self._prev_obs = obs
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray | dict, float, bool, bool, dict]:
        # APM limiting: replace non-no-op actions with no_op when budget is
        # exhausted.  Uses game time (derived from game_loop) so max_apm is
        # expressed in game-time APM regardless of training speed.
        _fn_idx_requested = int(action[0]) if len(action) > 0 else 0
        if self._apm_limiter is not None and not self._apm_limiter.allow(
            self._apm_game_time_s,
            _fn_idx_requested,
            protect_burst_budget=True,
        ):
            action = DISCRETE_ACTIONS[0].copy()
            _apm_throttled = True
            self._ep_apm_throttled_steps += 1
        else:
            _apm_throttled = False

        flat_obs, _raw_reward, done, info = self._client.step(action)
        obs = self._make_obs(flat_obs, info)

        self._step_count += 1
        self._elapsed_s = time.monotonic() - self._episode_start_s

        # Override the prev-* entries the client cannot know about.
        info["prev_minerals"] = self._prev_minerals
        info["prev_vespene"] = self._prev_vespene
        info["prev_score"] = self._prev_score
        info["prev_army_count"] = self._prev_army_count
        info["prev_total_self_hp"] = self._prev_total_self_hp
        info["elapsed_s"] = self._elapsed_s
        info["raw_reward"] = float(_raw_reward)
        # Action fn_idx and target coords — required by reward shaping.
        # Use the action actually sent to PySC2 (after client-side fallback
        # substitutions), not the policy-requested fn_idx, so blocked
        # Attack_screen/Move_screen requests do not masquerade as executed.
        _executed_fn_idx = int(getattr(self._client, "last_fn_idx", -1))
        # action_target_x/y are normalised [0, 1] screen coordinates used by
        # click_attack_bonus to distinguish "click on unit" from "attack move".
        info["action_fn_idx"] = _executed_fn_idx
        info["action_target_x"] = float(np.clip(action[1], 0.0, 1.0)) if len(action) > 1 else 0.5
        info["action_target_y"] = float(np.clip(action[2], 0.0, 1.0)) if len(action) > 2 else 0.5
        if self._prev_move_target is not None:
            info["prev_move_target_x"] = float(self._prev_move_target[0])
            info["prev_move_target_y"] = float(self._prev_move_target[1])
        else:
            info["prev_move_target_x"] = None
            info["prev_move_target_y"] = None
        # Screen size threading — reward shaping uses it to scale pixel-based
        # thresholds for non-default screen resolutions.
        info["screen_size"] = self._screen_size
        # APM throttle metadata — useful for diagnostics and analytics.
        info["apm_throttled"] = _apm_throttled
        info["episode_apm_throttled_steps"] = self._ep_apm_throttled_steps
        # Realtime skip-frame diagnostics: if game_loop advanced more than the
        # expected step_mul since the previous action, the excess is counted as
        # skipped frames (action arrived too late for those loops).
        _skipped_frames_this_step = 0
        _curr_game_loop = self._safe_float_or_none(info.get("game_loop"))
        if _curr_game_loop is not None and self._prev_step_game_loop is not None:
            _delta = max(0.0, _curr_game_loop - self._prev_step_game_loop)
            _skipped_frames_this_step = int(max(0.0, np.floor(_delta - float(self._step_mul) + 1e-6)))
            self._ep_skipped_frames += _skipped_frames_this_step
            self._apm_game_time_s += _delta / _SC2_TICKS_PER_S
        else:
            self._apm_game_time_s += self._step_mul / _SC2_TICKS_PER_S
        if _curr_game_loop is not None:
            self._prev_step_game_loop = _curr_game_loop
        info["skipped_frames_this_step"] = _skipped_frames_this_step
        info["episode_skipped_frames"] = self._ep_skipped_frames

        time_over = self._elapsed_s > self._max_episode_time_s
        finished = done

        reward, step_components = self._reward_calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=finished,
            elapsed_s=self._elapsed_s,
            info=info,
            n_ticks=self._step_mul,
        )
        # Accumulate per-component totals across the episode so analytics
        # can attribute reward to score / economy / penalties separately
        # (issue #128/2b).
        for k, v in step_components.items():
            self._ep_reward_components[k] = self._ep_reward_components.get(k, 0.0) + float(v)
        info["episode_reward_components"] = dict(self._ep_reward_components)

        # Track per-episode action counts (analytics 2a) using the policy-
        # requested fn_idx so the chart reflects intent rather than the
        # intermediate select_* injected by the deferred-action resolver
        # (issue #383).  Reward shaping and debug metadata still use the
        # executed fn_idx via info["action_fn_idx"].
        self._ep_action_counts[_fn_idx_requested] = self._ep_action_counts.get(_fn_idx_requested, 0) + 1
        # Track 8×8 spatial-target histogram (analytics 2d).
        if len(action) >= 3:
            _xi = min(7, int(float(np.clip(action[1], 0.0, 1.0)) * 8))
            _yi = min(7, int(float(np.clip(action[2], 0.0, 1.0)) * 8))
            self._ep_xy_hist[_yi, _xi] += 1
        # Track key game-state feature averages (analytics 2c).
        for _feat in (
            "army_count",
            "food_used",
            "food_cap",
            "minerals",
            "vespene",
            "screen_self_count",
            "screen_enemy_count",
        ):
            _v = info.get(_feat)
            if _v is not None:
                self._ep_obs_sums[_feat] = self._ep_obs_sums.get(_feat, 0.0) + float(_v)
        self._ep_obs_step_count += 1

        # Track SC2 end-screen analytics: supply cap, time-series, build order.
        _game_time_s = float(info.get("game_loop", 0.0)) / _SC2_TICKS_PER_S
        _food_used = info.get("food_used", 0.0)
        _food_cap = info.get("food_cap", 0.0)
        if _food_cap > 0 and _food_used >= _food_cap:
            self._ep_supply_capped_steps += 1
        _army_count = info.get("army_count", 0.0)
        _resources = info.get("minerals", 0.0) + info.get("vespene", 0.0)
        _is_terminal = info.get("is_last", False) or info.get("time_over", False)
        if self._ep_obs_step_count % _SERIES_SAMPLE_RATE == 0 or _is_terminal:
            self._ep_army_series.append([_game_time_s, _army_count])
            self._ep_resource_series.append([_game_time_s, _resources])
        # Build-order: detect unit-count increases from client's unit_counts dict.
        _unit_counts = info.get("unit_counts") or {}
        for _uname, _ucount in _unit_counts.items():
            _prev = self._ep_prev_unit_counts.get(_uname, 0.0)
            if _ucount > _prev:
                self._ep_build_order.append([_game_time_s, _uname])
            self._ep_prev_unit_counts[_uname] = _ucount

        terminated = finished
        truncated = time_over and not terminated

        # Only emit the large per-episode series into info at end of episode so
        # mid-episode policy.update() calls are not burdened with copying/
        # serialising O(n) lists on every step.
        if terminated or truncated:
            info["episode_supply_capped_fraction"] = self._ep_supply_capped_steps / self._ep_obs_step_count
            info["episode_army_series"] = self._ep_army_series
            info["episode_resource_series"] = self._ep_resource_series
            info["episode_build_order"] = self._ep_build_order
            info["episode_action_counts"] = dict(self._ep_action_counts)
            info["episode_action_name_map"] = FUNCTION_IDS
            info["episode_xy_hist"] = self._ep_xy_hist.tolist()
            info["episode_obs_averages"] = {k: v / self._ep_obs_step_count for k, v in self._ep_obs_sums.items()}
            # Kill stats: score_cumulative counters are cumulative and reset
            # each episode, so the final value equals the episode total.
            info["episode_killed_value_units"] = info.get("killed_value_units", 0.0)
            info["episode_killed_value_structures"] = info.get("killed_value_structures", 0.0)

        if finished:
            outcome = info.get("player_outcome")
            if outcome is not None and outcome > 0:
                info["termination_reason"] = "win"
            elif outcome is not None and outcome < 0:
                info["termination_reason"] = "loss"
            else:
                info["termination_reason"] = "finish"
        elif time_over:
            info["termination_reason"] = "timeout"
        else:
            info["termination_reason"] = None

        self._prev_minerals = info.get("minerals", 0.0)
        self._prev_vespene = info.get("vespene", 0.0)
        self._prev_score = info.get("score", 0.0)
        self._prev_army_count = info.get("army_count", 0.0)
        self._prev_total_self_hp = info.get("total_self_hp", 0.0)
        if info["action_fn_idx"] == 2:
            self._prev_move_target = (info["action_target_x"], info["action_target_y"])

        if self._belief is not None:
            game_loop = float(info.get("game_loop", 0.0))
            dt_s = (game_loop - self._prev_game_loop) / _SC2_TICKS_PER_S
            self._prev_game_loop = game_loop

            grid = self._belief_cfg["region_grid"]
            n_rows, n_cols = int(grid[0]), int(grid[1])
            n_slots = n_rows * n_cols

            vis_raw = info.get("minimap_vis")
            belief_obs = np.full(n_slots, np.nan, dtype=np.float64)
            visible_slots = np.zeros(n_slots, dtype=bool)

            if vis_raw is not None:
                vis = np.asarray(vis_raw, dtype=np.float32)
                if vis.ndim == 2:
                    h, w = vis.shape
                    # Guard: minimap must be at least as large as the region grid.
                    if h >= n_rows and w >= n_cols:
                        rs = h // n_rows
                        cs = w // n_cols
                        trimmed = vis[: n_rows * rs, : n_cols * cs]
                        pooled = trimmed.reshape(n_rows, rs, n_cols, cs).max(axis=(1, 3))
                        visible_slots = pooled.flatten() >= 2.0
                        belief_obs = np.where(visible_slots, pooled.flatten() / 2.0, np.nan)

            self._belief.project(max(dt_s, 0.0))
            self._belief.update(belief_obs, {})
            self._info_gain.update(
                belief_obs,
                {"time_s": game_loop / _SC2_TICKS_PER_S, "visible_slots": visible_slots},
            )
            scout_reward = self._info_gain.intrinsic_reward()

            _benc = self._belief.encode().astype(np.float32)
            _senc = self._info_gain.staleness().astype(np.float32)
            if self._use_spatial:
                obs = {"flat": np.concatenate([obs["flat"], _benc, _senc]), "spatial": obs["spatial"]}
            else:
                obs = np.concatenate([obs, _benc, _senc])

            reward += scout_reward
            self._ep_reward_components["scout"] = self._ep_reward_components.get("scout", 0.0) + float(scout_reward)
            # Refresh the snapshot so episode_reward_components includes scout.
            info["episode_reward_components"] = dict(self._ep_reward_components)

        self._prev_obs = obs
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._client.close()

    def save_replay(self, replay_dir: str, prefix: str) -> str | None:
        """Save the most recently played episode as an SC2 replay file."""
        return self._client.save_replay(replay_dir, prefix=prefix)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_obs(self, flat_obs: np.ndarray, info: dict) -> np.ndarray | dict:
        """Wrap flat_obs into a dict observation when spatial layers are active."""
        if not self._use_spatial:
            return flat_obs
        spatial = info.get("spatial_obs")
        if spatial is None:
            spatial = np.zeros(self._spatial_shape, dtype=np.float32)
        return {"flat": flat_obs, "spatial": spatial}

    @staticmethod
    def _safe_float_or_none(value: Any) -> float | None:
        """Convert *value* to float or return None when conversion is invalid."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # BaseGameEnv API
    # ------------------------------------------------------------------

    def _build_obs(self, step: Any) -> np.ndarray:
        """Not used directly — obs comes from the client's reset/step."""
        if self._use_spatial:
            return np.zeros(self.observation_space["flat"].shape, dtype=np.float32)
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_game_info(self) -> dict:
        return {
            "map_name": self._map_name,
            "is_ladder": self._is_ladder,
            "step_count": self._step_count,
            "elapsed_s": self._elapsed_s,
        }

    def get_episode_time_limit(self) -> float:
        return self._max_episode_time_s

    def set_episode_time_limit(self, seconds: float) -> None:
        self._max_episode_time_s = seconds

    def set_opponent_policy(self, policy: Any) -> None:
        """Set (or replace) the opponent policy for self-play.

        The opponent acts from its own observation every step, mirroring
        AlphaGo-style self-play.  Only effective when the env was created
        with ``self_play=True``.
        """
        self._client._opponent_policy = policy


def make_env(
    experiment_dir: str | Path,
    map_name: str = "MoveToBeacon",
    max_episode_time_s: float = 120.0,
    step_mul: int = 1,
    screen_size: int = 64,
    minimap_size: int = 64,
    agent_race: str = "random",
    bot_difficulty: str = "very_easy",
    visualize: bool = False,
    screen_layers: list[str] | None = None,
    minimap_layers: list[str] | None = None,
    obs_spec_preset: str | None = None,
    enable_belief: bool = False,
    max_apm: int | None = None,
    apm_burst_s: float = 2.0,
    extreme_random_run_count: int = 0,
    self_play: bool = False,
    opponent_policy: Any = None,
) -> SC2Env:
    """Factory that wires up an :class:`SC2Env` from an experiment directory.

    Loads ``reward_config.yaml`` from *experiment_dir* if it exists.
    """
    experiment_dir = Path(experiment_dir)
    reward_cfg_path = experiment_dir / "reward_config.yaml"
    if reward_cfg_path.exists():
        reward_config = SC2RewardConfig.from_yaml(str(reward_cfg_path))
    else:
        reward_config = SC2RewardConfig()
    return SC2Env(
        map_name=map_name,
        reward_config=reward_config,
        max_episode_time_s=max_episode_time_s,
        step_mul=step_mul,
        screen_size=screen_size,
        minimap_size=minimap_size,
        agent_race=agent_race,
        bot_difficulty=bot_difficulty,
        visualize=visualize,
        screen_layers=screen_layers,
        minimap_layers=minimap_layers,
        obs_spec_preset=obs_spec_preset,
        enable_belief=enable_belief,
        max_apm=max_apm,
        apm_burst_s=apm_burst_s,
        extreme_random_run_count=extreme_random_run_count,
        self_play=self_play,
        opponent_policy=opponent_policy,
    )
