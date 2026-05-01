"""Thin wrapper around the assetto-corsa-rl gym environment.

The upstream package is imported lazily so the rest of the codebase
(including unit tests on Linux CI) can run without it installed.

ACStepState is the dict-like state object the env and reward calculator
consume. It exposes the telemetry fields each layer reads as plain
attributes so we can swap raw obs encodings without touching downstream
code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


# Default upstream env id, per issue #79.
DEFAULT_ENV_ID = "AssettoCorsa-v0"


@dataclass
class ACStepState:
    """Normalised view of one AC env step.

    Fields mirror the AC_OBS_SPEC vocabulary so env._build_obs() and
    RewardCalculator.compute() can read them by name.
    """

    raw_obs: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict
    speed_ms: float = 0.0
    lateral_offset: float | None = None
    vertical_offset: float | None = None
    yaw_error: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    track_progress: float | None = None
    steering_angle: float = 0.0
    engine_rpm: float = 0.0
    gear: float = 0.0
    wheel_slip: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pos_x: float = 0.0
    pos_z: float = 0.0
    finished: bool = False
    vision: np.ndarray | None = None


class ACClient:
    """Synchronous wrapper that converts AC gym output to ACStepState.

    The constructor delegates env creation to a factory callable so tests
    can inject a stub without monkey-patching ``gymnasium.make``.
    """

    def __init__(
        self,
        env_id: str = DEFAULT_ENV_ID,
        env_factory: Callable[..., Any] | None = None,
        env_kwargs: dict | None = None,
    ) -> None:
        self._env_kwargs = env_kwargs or {}
        if env_factory is None:
            env_factory = _default_env_factory
        self._env = env_factory(env_id, **self._env_kwargs)

    # ------------------------------------------------------------------
    # Gym passthroughs
    # ------------------------------------------------------------------

    def reset(self) -> ACStepState:
        result = self._env.reset()
        # Gymnasium returns (obs, info); legacy gym returns just obs.
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        return self._to_step_state(obs, reward=0.0, terminated=False,
                                   truncated=False, info=info)

    def step(self, action: np.ndarray) -> ACStepState:
        result = self._env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            # Legacy gym 4-tuple
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = False
        return self._to_step_state(obs, float(reward), bool(terminated),
                                   bool(truncated), info)

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            logger.warning("AC env close() raised; ignoring", exc_info=True)

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def _to_step_state(
        self,
        obs: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> ACStepState:
        """Map raw obs/info into ACStepState by best-effort field lookup.

        AC gym wrappers vary in exact keys; we accept both flat dicts and
        ``info``-only telemetry. Missing fields default to neutral values.
        """
        info = dict(info) if info else {}
        getter = _make_getter(obs, info)

        wheel_slip = getter(
            ["wheel_slip", "tyre_slip", "tire_slip"],
            default=(0.0, 0.0, 0.0, 0.0),
        )
        if not isinstance(wheel_slip, (list, tuple, np.ndarray)):
            wheel_slip = (float(wheel_slip),) * 4
        wheel_slip = tuple(float(x) for x in list(wheel_slip)[:4])
        if len(wheel_slip) < 4:
            wheel_slip = wheel_slip + (0.0,) * (4 - len(wheel_slip))

        angular = getter(["angular_velocity", "ang_vel"], default=(0.0, 0.0, 0.0))
        if not isinstance(angular, (list, tuple, np.ndarray)):
            angular = (0.0, 0.0, 0.0)
        angular = tuple(float(x) for x in list(angular)[:3])
        if len(angular) < 3:
            angular = angular + (0.0,) * (3 - len(angular))

        finished = bool(info.get("finished", False)) or (
            terminated and bool(info.get("lap_completed", False))
        )

        return ACStepState(
            raw_obs        = obs,
            reward         = reward,
            terminated     = terminated,
            truncated      = truncated,
            info           = info,
            speed_ms       = float(getter(["speed_ms", "speed"], default=0.0)),
            lateral_offset = _as_float_or_none(getter(["lateral_offset", "lateral_offset_m"])),
            vertical_offset= _as_float_or_none(getter(["vertical_offset", "vertical_offset_m"])),
            yaw_error      = float(getter(["yaw_error", "yaw_error_rad"], default=0.0)),
            pitch          = float(getter(["pitch", "pitch_rad"], default=0.0)),
            roll           = float(getter(["roll", "roll_rad"], default=0.0)),
            track_progress = _as_float_or_none(getter(["track_progress", "normalized_position"])),
            steering_angle = float(getter(["steering_angle", "steer"], default=0.0)),
            engine_rpm     = float(getter(["engine_rpm", "rpm"], default=0.0)),
            gear           = float(getter(["gear"], default=0.0)),
            wheel_slip     = wheel_slip,  # type: ignore[arg-type]
            angular_velocity = angular,   # type: ignore[arg-type]
            pos_x          = float(getter(["pos_x", "x"], default=0.0)),
            pos_z          = float(getter(["pos_z", "z"], default=0.0)),
            finished       = finished,
            vision         = _maybe_array(getter(["vision", "lidar"])),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_env_factory(env_id: str, **kwargs: Any) -> Any:
    """Lazy import of gymnasium / assetto_corsa_rl.

    Importing these at module-load time would break Linux CI where the
    upstream package isn't installed; importing here keeps the AC package
    introspectable everywhere.
    """
    try:
        # Importing assetto_corsa_rl registers the AssettoCorsa-v0 env id.
        import assetto_corsa_rl  # noqa: F401
    except ImportError:
        logger.debug("assetto_corsa_rl not installed; gym.make may still find the env.")
    try:
        import gymnasium as gym  # type: ignore
    except ImportError:
        import gym  # type: ignore
    return gym.make(env_id, **kwargs)


def _make_getter(obs: Any, info: dict) -> Callable[..., Any]:
    """Return a function that resolves the first matching key from obs/info."""

    def get(keys: list[str], default: Any = None) -> Any:
        for k in keys:
            if isinstance(obs, dict) and k in obs:
                return obs[k]
            if k in info:
                return info[k]
            if hasattr(obs, k):
                return getattr(obs, k)
        return default

    return get


def _as_float_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _maybe_array(v: Any) -> np.ndarray | None:
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float32)
    return arr if arr.size > 0 else None
