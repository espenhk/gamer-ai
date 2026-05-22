"""Rocket League Gymnasium environment.

Wraps RLGym (``rlgym``) which hooks into Rocket League via Bakkesmod.

Requirements
------------
- Rocket League (Steam or Epic Games, Windows only).
- Bakkesmod + RLGym plugin — follow the install guide in
  ``games/rocket_league/README.md``.
- Python package: ``pip install rlgym``

If ``rlgym`` is not installed, importing this module raises ``ImportError``
with setup guidance.

Observation
-----------
142 float32 values — see ``games.rocket_league.obs_spec`` for the full spec.

Action
------
Box([-1,-1,-1,-1,-1, 0, 0, 0], [1,1,1,1,1,1,1,1], dtype=float32)
  [0] throttle, [1] steer, [2] pitch, [3] yaw, [4] roll,
  [5] jump, [6] boost, [7] handbrake

Episode termination
-------------------
- *Goal scored or conceded* — sparse terminal event from RLGym.
- *Timeout* — ``elapsed_s > max_episode_time_s``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

# Optional dependency — fails loudly if not installed.
try:
    import rlgym  # noqa: F401
    _RLGYM_AVAILABLE = True
except ImportError as _exc:
    raise ImportError(
        "rlgym is not installed.  Install the Rocket League gym with:\n"
        "    pip install rlgym\n"
        "Rocket League (commercial, Windows) + Bakkesmod must also be installed.\n"
        "See games/rocket_league/README.md for full setup instructions."
    ) from _exc

from gymnasium import spaces

from framework.base_env import BaseGameEnv
from games.rocket_league.obs_spec import BASE_OBS_DIM
from games.rocket_league.reward import RocketLeagueRewardConfig, RocketLeagueRewardCalculator

logger = logging.getLogger(__name__)

# RLGym arena constants (Unreal Units)
_BALL_GOAL_Y_OPP = 5120.0   # opponent goal Y (orange goal, positive side)
_BALL_GOAL_Y_OWN = -5120.0  # own goal Y (blue goal, negative side)
_ARENA_DIAG = 13272.0


class RocketLeagueEnv(BaseGameEnv):
    """Gymnasium environment wrapping Rocket League via RLGym.

    Parameters
    ----------
    reward_config :
        ``RocketLeagueRewardConfig`` instance.  If *None*, uses defaults.
    max_episode_time_s :
        Seconds before the episode is truncated.
    tick_skip :
        Number of physics frames to advance per ``step()`` call (default 8,
        matching the RLGym default).  Affects observation granularity and
        effective APM.
    """

    metadata = {"render_modes": []}
    _TEAM_AGENT_COUNT = 3

    def __init__(
        self,
        reward_config: RocketLeagueRewardConfig | None = None,
        max_episode_time_s: float = 300.0,
        tick_skip: int = 8,
    ) -> None:
        super().__init__()

        self._reward_config = reward_config or RocketLeagueRewardConfig()
        self._max_episode_time_s = max_episode_time_s
        self._tick_skip = tick_skip
        self._reward_calc = RocketLeagueRewardCalculator(self._reward_config)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(BASE_OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1., -1., -1., -1., -1., 0., 0., 0.], dtype=np.float32),
            high=np.array([ 1.,  1.,  1.,  1.,  1., 1., 1., 1.], dtype=np.float32),
            dtype=np.float32,
        )

        # Build RLGym environment.
        self._env = rlgym.make(
            tick_skip=tick_skip,
            team_size=self._TEAM_AGENT_COUNT,
            self_play=False,
        )

        self._episode_start_s: float = 0.0
        self._elapsed_s: float = 0.0
        self._prev_obs: np.ndarray = np.zeros(BASE_OBS_DIM, dtype=np.float32)
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        raw_obs = self._env.reset()
        obs = self._parse_obs(raw_obs)

        self._episode_start_s = time.monotonic()
        self._elapsed_s = 0.0
        self._step_count = 0
        self._prev_obs = obs.copy()
        self._reward_calc.reset()

        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1 and self._prev_obs.ndim == 2:
            action = np.repeat(action[np.newaxis, :], self._prev_obs.shape[0], axis=0)

        raw_obs, _rlgym_reward, done, info_raw = self._env.step(action)
        obs = self._parse_obs(raw_obs)

        self._step_count += 1
        self._elapsed_s = time.monotonic() - self._episode_start_s

        # Extract game events from RLGym info.
        info_rows: list[dict] = (
            [row for row in info_raw if isinstance(row, dict)]
            if isinstance(info_raw, (list, tuple))
            else [info_raw] if isinstance(info_raw, dict)
            else []
        )
        primary_info = info_rows[0] if info_rows else {}

        goal_scored = any(
            bool(row.get("TimeoutException", False) is False and row.get("goal_scored", False))
            for row in info_rows
        ) if info_rows else bool(
            primary_info.get("TimeoutException", False) is False
            and primary_info.get("goal_scored", False)
        )
        goal_conceded = any(bool(row.get("goal_conceded", False)) for row in info_rows)
        ball_touched = any(bool(row.get("ball_touched", False)) for row in info_rows)

        action_rows = action if action.ndim == 2 else action[np.newaxis, :]
        boosting_agents = [bool(float(a[6]) > 0.5) for a in action_rows]
        boosting = bool(any(boosting_agents))

        # Velocity towards ball (derived from obs).
        obs_rows = obs if obs.ndim == 2 else obs[np.newaxis, :]
        vel_towards_ball_agents = [self._compute_vel_towards_ball(o) for o in obs_rows]
        vel_towards_ball = float(np.mean(vel_towards_ball_agents))

        time_over = self._elapsed_s >= self._max_episode_time_s
        terminated = bool(done and not time_over)
        truncated = bool(time_over and not terminated)

        termination_reason: str | None = None
        if goal_scored:
            termination_reason = "goal_scored"
        elif goal_conceded:
            termination_reason = "goal_conceded"
        elif terminated:
            termination_reason = "done"
        elif truncated:
            termination_reason = "timeout"

        info: dict[str, Any] = {
            "vel_towards_ball": vel_towards_ball,
            "boosting": boosting,
            "ball_touched": ball_touched,
            "goal_scored": goal_scored,
            "goal_conceded": goal_conceded,
            "elapsed_s": self._elapsed_s,
            "termination_reason": termination_reason,
            "vel_towards_ball_agents": vel_towards_ball_agents,
            "boosting_agents": boosting_agents,
            "team_agent_count": int(obs_rows.shape[0]),
        }

        reward = self._reward_calc.compute(
            prev_state=None,
            curr_state=None,
            finished=terminated or truncated,
            elapsed_s=self._elapsed_s,
            info=info,
        )

        self._prev_obs = obs.copy()
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # BaseGameEnv abstract method
    # ------------------------------------------------------------------

    def _build_obs(self, step: Any) -> np.ndarray:
        return np.zeros(BASE_OBS_DIM, dtype=np.float32)

    # ------------------------------------------------------------------
    # Episode time limit hooks
    # ------------------------------------------------------------------

    def get_episode_time_limit(self) -> float:
        return self._max_episode_time_s

    def set_episode_time_limit(self, seconds: float) -> None:
        self._max_episode_time_s = seconds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_obs(self, raw_obs: Any) -> np.ndarray:
        """Convert raw RLGym observation to our fixed-size vector(s)."""
        if isinstance(raw_obs, (list, tuple)):
            if len(raw_obs) == 0:
                return np.zeros(BASE_OBS_DIM, dtype=np.float32)
            rows = [self._parse_obs_row(row) for row in raw_obs]
            return np.stack(rows, axis=0) if len(rows) > 1 else rows[0]

        arr = np.asarray(raw_obs, dtype=np.float32)
        if arr.ndim == 2:
            rows = [self._parse_obs_row(row) for row in arr]
            return np.stack(rows, axis=0) if len(rows) > 1 else rows[0]
        return self._parse_obs_row(raw_obs)

    def _parse_obs_row(self, raw_obs: Any) -> np.ndarray:
        """Convert one agent's raw observation to our fixed-size vector."""
        arr = np.zeros(BASE_OBS_DIM, dtype=np.float32)
        if raw_obs is not None:
            raw = np.asarray(raw_obs, dtype=np.float32).ravel()
            n = min(len(raw), BASE_OBS_DIM)
            arr[:n] = raw[:n]
        return arr

    def _compute_vel_towards_ball(self, obs: np.ndarray) -> float:
        """Return the car's velocity component directed towards the ball (UU/s)."""
        # car velocity: indices 3–5
        car_vel = obs[3:6]
        # relative ball pos: indices 117–119 (already ball_pos - car_pos)
        rel_ball = obs[117:120]
        dist = float(np.linalg.norm(rel_ball))
        if dist < 1e-6:
            return 0.0
        ball_dir = rel_ball / dist
        return float(np.dot(car_vel, ball_dir))


def make_env(
    experiment_dir: str | Path,
    max_episode_time_s: float = 300.0,
    tick_skip: int = 8,
) -> RocketLeagueEnv:
    """Factory that wires up a ``RocketLeagueEnv`` from an experiment directory.

    Loads ``reward_config.yaml`` from *experiment_dir* if it exists.
    """
    experiment_dir = Path(experiment_dir)
    reward_cfg_path = experiment_dir / "reward_config.yaml"
    if reward_cfg_path.exists():
        reward_config = RocketLeagueRewardConfig.from_yaml(str(reward_cfg_path))
    else:
        reward_config = RocketLeagueRewardConfig()
    return RocketLeagueEnv(
        reward_config=reward_config,
        max_episode_time_s=max_episode_time_s,
        tick_skip=int(tick_skip),
    )
