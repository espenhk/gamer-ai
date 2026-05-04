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
from games.sc2.actions import FUNCTION_IDS
from games.sc2.client import SC2Client
from games.sc2.obs_spec import MINIGAME_NAMES, get_spec
from games.sc2.reward import SC2RewardCalculator, SC2RewardConfig

VALID_OBS_PRESETS = ("minigame", "ladder", "rich")

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
        Game-tick multiplier per env step.
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
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_name: str = "MoveToBeacon",
        reward_config: SC2RewardConfig | None = None,
        max_episode_time_s: float = 120.0,
        step_mul: int = 8,
        screen_size: int = 64,
        minimap_size: int = 64,
        agent_race: str = "random",
        bot_difficulty: str = "very_easy",
        visualize: bool = False,
        screen_layers: list[str] | None = None,
        minimap_layers: list[str] | None = None,
        obs_spec_preset: str | None = None,
    ) -> None:
        super().__init__()

        self._map_name = map_name
        self._is_ladder = map_name not in MINIGAME_NAMES
        self._reward_config = reward_config or SC2RewardConfig()
        self._max_episode_time_s = max_episode_time_s
        self._step_mul = step_mul
        self._reward_calc = SC2RewardCalculator(self._reward_config)
        self._screen_layers: list[str] = list(screen_layers or [])
        self._minimap_layers: list[str] = list(minimap_layers or [])
        self._use_spatial = bool(self._screen_layers or self._minimap_layers)
        self._obs_spec_preset = obs_spec_preset

        spec = get_spec(map_name, preset=obs_spec_preset)
        flat_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(spec.dim,),
            dtype=np.float32,
        )
        if self._use_spatial:
            n_channels = len(self._screen_layers) + len(self._minimap_layers)
            self.observation_space = spaces.Dict({
                "flat": flat_space,
                "spatial": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(n_channels, screen_size, screen_size),
                    dtype=np.float32,
                ),
            })
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
            screen_layers=self._screen_layers,
            minimap_layers=self._minimap_layers,
            obs_spec_preset=obs_spec_preset,
        )

        # Episode tracking
        self._prev_obs: np.ndarray | None = None
        self._prev_minerals: float = 0.0
        self._prev_vespene: float = 0.0
        self._prev_score: float = 0.0
        self._elapsed_s: float = 0.0
        self._episode_start_s: float = 0.0
        self._step_count: int = 0
        self._ep_reward_components: dict[str, float] = {}

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

        self._prev_obs = obs
        self._prev_minerals = info.get("minerals", 0.0)
        self._prev_vespene = info.get("vespene", 0.0)
        self._prev_score = info.get("score", 0.0)
        self._elapsed_s = 0.0
        self._episode_start_s = time.monotonic()
        self._step_count = 0
        self._ep_reward_components = {}
        self._reward_calc.reset()

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray | dict, float, bool, bool, dict]:
        flat_obs, _raw_reward, done, info = self._client.step(action)
        obs = self._make_obs(flat_obs, info)

        self._step_count += 1
        self._elapsed_s = time.monotonic() - self._episode_start_s

        # Override the prev-* entries the client cannot know about.
        info["prev_minerals"] = self._prev_minerals
        info["prev_vespene"] = self._prev_vespene
        info["prev_score"] = self._prev_score
        info["elapsed_s"] = self._elapsed_s
        # Action fn_idx — required by reward shaping (e.g. idle_bonus, #127).
        info["action_fn_idx"] = int(action[0]) if len(action) > 0 else -1

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
            self._ep_reward_components[k] = (
                self._ep_reward_components.get(k, 0.0) + float(v)
            )
        info["episode_reward_components"] = dict(self._ep_reward_components)

        terminated = finished
        truncated = time_over and not terminated

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

        self._prev_obs = obs
        self._prev_minerals = info.get("minerals", 0.0)
        self._prev_vespene = info.get("vespene", 0.0)
        self._prev_score = info.get("score", 0.0)

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._client.close()

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

    # ------------------------------------------------------------------
    # BaseGameEnv API
    # ------------------------------------------------------------------

    def _build_obs(self, step: Any) -> np.ndarray:
        """Not used directly — obs comes from the client's reset/step."""
        if self._use_spatial:
            return np.zeros(
                self.observation_space["flat"].shape, dtype=np.float32
            )
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


def make_env(
    experiment_dir: str | Path,
    map_name: str = "MoveToBeacon",
    max_episode_time_s: float = 120.0,
    step_mul: int = 8,
    screen_size: int = 64,
    minimap_size: int = 64,
    agent_race: str = "random",
    bot_difficulty: str = "very_easy",
    visualize: bool = False,
    screen_layers: list[str] | None = None,
    minimap_layers: list[str] | None = None,
    obs_spec_preset: str | None = None,
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
    )
