"""Thin wrapper around gym_torcs providing a uniform client interface.

The TorcsClient encapsulates the ``gym_torcs.TorcsEnv`` instance and exposes
``reset()`` / ``step()`` / ``close()`` methods that return data in the flat
numpy format expected by :class:`games.torcs.env.TorcsEnv`.

The client translates the TORCS observation dictionary into a fixed-length
``np.ndarray`` matching :data:`games.torcs.obs_spec.TORCS_OBS_SPEC`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Selected track sensor indices — we pick 9 representative rangefinder angles
# out of the 19 that TORCS provides (approx. -90, -60, -30, -10, 0, 10, 30, 60, 90 deg).
_TRACK_SENSOR_INDICES = [0, 3, 6, 8, 9, 10, 12, 15, 18]


class TorcsClient:
    """Manages a ``gym_torcs.TorcsEnv`` session.

    Parameters
    ----------
    vision : bool
        If True, request pixel observations from TORCS (64×64 image).
        Default is False (sensor-only mode).
    throttle : bool
        If True, expose a 2-dim continuous action (steer + accel).  If False,
        the action is steering only.  Default True.
    gear_change : bool
        If True, let the agent control gear shifting.  Default False (auto).
    """

    def __init__(
        self,
        vision: bool = False,
        throttle: bool = True,
        gear_change: bool = False,
    ) -> None:
        self._vision = vision
        self._throttle = throttle
        self._gear_change = gear_change
        self._torcs_env: Any = None  # lazy — created on first reset()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, relaunch: bool = True) -> np.ndarray:
        """Reset the TORCS environment and return the initial observation.

        Parameters
        ----------
        relaunch : bool
            When True the TORCS process is killed and relaunched to avoid
            the memory leak documented in gym_torcs.  Recommended for long
            training runs.
        """
        if self._torcs_env is None:
            try:
                from gym_torcs import TorcsEnv as _GymTorcsEnv  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "gym_torcs is required for the TORCS game integration.  "
                    "Install it with:  pip install git+https://github.com/ugo-nama-kun/gym_torcs.git"
                ) from exc
            self._torcs_env = _GymTorcsEnv(
                vision=self._vision,
                throttle=self._throttle,
                gear_change=self._gear_change,
            )
            obs = self._torcs_env.reset()
        else:
            obs = self._torcs_env.reset(relaunch=relaunch)
        return self._flatten_obs(obs)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Take one step in the environment.

        Parameters
        ----------
        action : np.ndarray
            A (3,) array ``[steer, accel, brake]``, each in ``[-1, 1]`` or
            ``[0, 1]``.  Mapped to the TORCS action format internally.

        Returns
        -------
        obs : np.ndarray
            Flat observation vector of shape ``(BASE_OBS_DIM,)``.
        reward : float
            Reward returned by gym_torcs (unused by our reward calculator).
        done : bool
            True if the episode ended.
        info : dict
            Extra metadata.
        """
        torcs_action = self._map_action(action)
        obs, reward, done, info = self._torcs_env.step(torcs_action)
        return self._flatten_obs(obs), reward, done, info

    def close(self) -> None:
        """Shut down the TORCS process."""
        if self._torcs_env is not None:
            self._torcs_env.end()
            self._torcs_env = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flatten_obs(self, obs: Any) -> np.ndarray:
        """Convert a gym_torcs observation object to a flat float32 array.

        The order matches :data:`games.torcs.obs_spec.TORCS_OBS_SPEC`.
        """
        speed = getattr(obs, "speedX", 0.0) * 300.0  # gym_torcs normalises by 300
        track_pos = getattr(obs, "trackPos", 0.0)
        lateral_offset = track_pos * 5.0  # rough conversion: trackPos is in [-1,1], track width ~10 m
        angle = getattr(obs, "angle", 0.0)
        dist_raced = getattr(obs, "distRaced", 0.0)
        track_length = max(getattr(obs, "trackLength", 1.0), 1.0)
        progress = (dist_raced / track_length) % 1.0
        rpm = getattr(obs, "rpm", 0.0)

        wheel_spin = getattr(obs, "wheelSpinVel", np.zeros(4))
        if not isinstance(wheel_spin, np.ndarray):
            wheel_spin = np.array(wheel_spin, dtype=np.float32)
        wheel_spin = wheel_spin[:4]  # ensure exactly 4

        track_sensors_raw = getattr(obs, "track", np.zeros(19))
        if not isinstance(track_sensors_raw, np.ndarray):
            track_sensors_raw = np.array(track_sensors_raw, dtype=np.float32)
        track_edges = track_sensors_raw[_TRACK_SENSOR_INDICES]

        flat = np.array(
            [
                speed,
                lateral_offset,
                angle,
                progress,
                rpm,
                *wheel_spin,
                *track_edges,
                track_pos,
            ],
            dtype=np.float32,
        )
        return flat

    @staticmethod
    def _map_action(action: np.ndarray) -> np.ndarray:
        """Map ``[steer, accel, brake]`` to the format gym_torcs expects.

        gym_torcs with ``throttle=True`` expects ``[steer, accel]``.
        Braking is applied by sending negative accel values.
        """
        steer = float(np.clip(action[0], -1.0, 1.0))
        accel = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))
        # gym_torcs uses a single accel dimension: positive = throttle,
        # negative = brake.  Subtract brake to produce a net signal.
        net_accel = accel - brake
        return np.array([steer, net_accel], dtype=np.float32)
