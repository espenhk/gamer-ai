"""CarRacing Gymnasium environment.

Wraps the ``CarRacing-v3`` environment from gymnasium.  Requires the
``box2d-py`` and ``pygame`` extras::

    pip install gymnasium[box2d]

If those packages are not installed, importing this module raises
``ImportError``.  The entry point in ``main.py`` converts that to a
``ValueError`` with a helpful message.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

# Optional dependency — raises ImportError if box2d extras not installed.
try:
    import gymnasium as gym
    from gymnasium.envs.box2d import CarRacing  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        "CarRacing-v3 requires optional gymnasium box2d extras.  Install with:\n"
        "    pip install gymnasium[box2d]\n"
        "This also requires pygame and box2d-py."
    ) from _exc

from gymnasium import spaces

from framework.base_env import BaseGameEnv
from games.car_racing.obs_spec import LOOKAHEAD_STEPS, build_car_racing_obs_spec_from_steps
from games.car_racing.reward import CarRacingRewardCalculator, CarRacingRewardConfig

logger = logging.getLogger(__name__)


def _wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians to [-pi, pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class CarRacingEnv(BaseGameEnv):
    """Gymnasium wrapper around CarRacing-v3.

    Converts the pixel observation into a compact feature vector compatible
    with the WeightedLinearPolicy framework: car-physics features (speed,
    angular velocity, wheel spin, control inputs) plus track-relative
    perception features (lateral offset, heading error, progress, and a
    lookahead schedule of upcoming curvature) derived from
    ``env.unwrapped.track`` — the centreline checkpoint list CarRacing-v3
    already builds internally. Mirrors the TMNF obs_spec.py pattern
    (see games/tmnf/obs_spec.py) so a driving agent can anticipate turns
    instead of only reacting to its own physics state.

    Parameters
    ----------
    reward_config :
        CarRacingRewardConfig instance.  If None, uses Python defaults.
    max_episode_steps :
        Maximum steps per episode (gymnasium default: 1000).
    continuous :
        If True, use continuous action space (default).  If False, use
        discrete actions.
    lookahead_steps :
        Checkpoint-index offsets ahead of the car's nearest centreline point
        to include in the observation. If None, uses the legacy 3-point
        default (see games/car_racing/obs_spec.py::LOOKAHEAD_STEPS).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        reward_config: CarRacingRewardConfig | None = None,
        max_episode_steps: int = 1000,
        continuous: bool = True,
        lookahead_steps: list[int] | None = None,
    ) -> None:
        super().__init__()

        self._reward_config = reward_config or CarRacingRewardConfig()
        self._reward_calc = CarRacingRewardCalculator(self._reward_config)
        self._max_episode_steps = max_episode_steps

        # Resolve the lookahead schedule once so obs_dim and the observation
        # space agree (mirrors games/tmnf/env.py, issue #493).
        self._lookahead_steps = lookahead_steps if lookahead_steps is not None else list(LOOKAHEAD_STEPS)
        self._obs_dim = build_car_racing_obs_spec_from_steps(self._lookahead_steps).dim

        self._env = gym.make(
            "CarRacing-v3",
            continuous=continuous,
            max_episode_steps=max_episode_steps,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._step_count: int = 0
        self._prev_info: dict = {}
        self._last_action: tuple[float, float, float] = (0.0, 0.0, 0.0)
        # Cached centreline, resolved on reset() from env.unwrapped.track:
        # (alpha, beta, x, y) per checkpoint -> track_beta (heading) / track_xy (position).
        self._track_beta: np.ndarray | None = None
        self._track_xy: np.ndarray | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        _raw_obs, info = self._env.reset(seed=seed, options=options)
        self._step_count = 0
        self._prev_info = info
        self._last_action = (0.0, 0.0, 0.0)
        self._reward_calc.reset()

        track = np.array(self._env.unwrapped.track, dtype=np.float64)
        self._track_beta = track[:, 1]
        self._track_xy = track[:, 2:4]

        return self._build_obs(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # CarRacing-v2 uses [steer, gas, brake] in [-1,1] × [0,1] × [0,1].
        car_action = np.array(
            [
                float(action[0]),  # steer
                float(action[1]),  # gas
                float(action[2]),  # brake
            ],
            dtype=np.float32,
        )

        self._last_action = (float(action[0]), float(action[1]), float(action[2]))

        _raw_obs, native_reward, terminated, truncated, info = self._env.step(car_action)

        self._step_count += 1

        info["native_reward"] = float(native_reward)
        car = self._env.unwrapped.car
        vx, vy = car.hull.linearVelocity
        info["speed_ms"] = float(np.hypot(vx, vy))
        info.setdefault("termination_reason", None)

        if terminated:
            # CarRacing-v3 sets terminated=True both on finishing the lap and on
            # driving out of the playfield; info["lap_finished"] (set by the env
            # itself alongside `terminated`) is what actually distinguishes the two.
            info["termination_reason"] = "finish" if info.get("lap_finished", True) else "crash"
        elif truncated:
            info["termination_reason"] = "timeout"

        reward = self._reward_calc.compute(
            prev_state=None,
            curr_state=None,
            finished=info["termination_reason"] == "finish",
            elapsed_s=self._step_count / 50.0,  # approx 50 steps/s
            info=info,
        )

        obs = self._build_obs()
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._env.close()

    def _build_obs(self) -> np.ndarray:
        car = self._env.unwrapped.car
        vx, vy = car.hull.linearVelocity
        speed = float(np.hypot(vx, vy))
        angular_vel = float(car.hull.angularVelocity)
        wheel_ang = [float(w.omega) for w in car.wheels]
        steer, gas, brake = self._last_action

        car_x, car_y = car.hull.position
        car_pos = np.array([car_x, car_y], dtype=np.float64)
        car_angle = float(car.hull.angle)
        lateral_offset, yaw_error, track_progress, lookahead = self._track_features(car_pos, car_angle)

        obs = [speed, angular_vel, *wheel_ang, steer, gas, brake, lateral_offset, yaw_error, track_progress]
        for lat, yaw in lookahead:
            obs.append(lat)
            obs.append(yaw)
        return np.array(obs, dtype=np.float32)

    def _track_features(
        self, car_pos: np.ndarray, car_angle: float
    ) -> tuple[float, float, float, list[tuple[float, float]]]:
        """Compute track-relative perception features from the cached
        centreline (set on reset()): signed lateral offset from the
        centreline, heading error vs. track direction, lap progress
        fraction, and (lateral offset, heading change) for each configured
        lookahead checkpoint. Mirrors games/tmnf's centreline projection."""
        n_lookahead = len(self._lookahead_steps)
        if self._track_xy is None or len(self._track_xy) == 0:
            return 0.0, 0.0, 0.0, [(0.0, 0.0)] * n_lookahead

        deltas = self._track_xy - car_pos
        nearest_idx = int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))
        n = len(self._track_xy)

        track_beta = float(self._track_beta[nearest_idx])
        track_point = self._track_xy[nearest_idx]
        # Perpendicular ("right") of the track heading, in the same rotation
        # convention as track_beta (CarRacing spawns the car with
        # hull.angle == track[0].beta, so this frame is consistent with the
        # car's own angle without needing to know Box2D's forward-axis choice).
        track_right = np.array([np.sin(track_beta), -np.cos(track_beta)])

        lateral_offset = float(np.dot(car_pos - track_point, track_right))
        yaw_error = _wrap_to_pi(track_beta - car_angle)
        track_progress = nearest_idx / n

        lookahead: list[tuple[float, float]] = []
        for step in self._lookahead_steps:
            j = (nearest_idx + step) % n
            waypoint = self._track_xy[j]
            waypoint_beta = float(self._track_beta[j])
            lat = float(np.dot(waypoint - car_pos, track_right))
            yaw = _wrap_to_pi(waypoint_beta - track_beta)
            lookahead.append((lat, yaw))

        return lateral_offset, yaw_error, track_progress, lookahead

    def get_episode_time_limit(self) -> float:
        return float(self._max_episode_steps) / 50.0

    def set_episode_time_limit(self, seconds: float) -> None:
        self._max_episode_steps = int(seconds * 50)


def make_env(
    experiment_dir: str | Path,
    max_episode_time_s: float = 20.0,
    lookahead_steps: list[int] | None = None,
) -> CarRacingEnv:
    """Factory that wires up a CarRacingEnv from an experiment directory."""
    experiment_dir = Path(experiment_dir)
    reward_cfg_path = experiment_dir / "reward_config.yaml"
    if reward_cfg_path.exists():
        reward_config = CarRacingRewardConfig.from_yaml(str(reward_cfg_path))
    else:
        reward_config = CarRacingRewardConfig()
    max_steps = int(max_episode_time_s * 50)
    return CarRacingEnv(
        reward_config=reward_config,
        max_episode_steps=max_steps,
        lookahead_steps=lookahead_steps,
    )
