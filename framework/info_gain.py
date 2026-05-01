"""Information-gain modules for partially observable environments.

The *information-gain drive* complements the belief module (``belief.py``)
by encoding **what the agent does not know** and rewarding it for filling
gaps — the intrinsic motivation to scout.

InfoGainModule
    Abstract base class.  Concrete implementations track per-slot
    staleness and emit an intrinsic reward when a stale slot is freshened.

RegionStalenessTracker
    Partitions the observable space into a grid of slots.  Each slot has
    a ``last_seen_t`` timestamp.  ``staleness[i]`` rises linearly from 0
    to 1 over ``scout_horizon_s`` seconds; never-observed slots start at
    1.0 (maximum staleness).  Intrinsic reward fires on stale→fresh
    transitions.

Tunable parameters are kept in a config dict (or YAML) so they can be
set per-experiment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class InfoGainModule(ABC):
    """Abstract interface for information-gain / scouting-drive tracking."""

    @abstractmethod
    def reset(self) -> None:
        """Clear state (call at episode start)."""

    @abstractmethod
    def update(self, obs: np.ndarray, info: dict) -> None:
        """Record which slots/regions have been observed at the current time."""

    @abstractmethod
    def staleness(self) -> np.ndarray:
        """Per-slot staleness in ``[0, 1]``.

        ``0`` = just observed, ``1`` = never observed or fully expired.
        """

    @abstractmethod
    def intrinsic_reward(self) -> float:
        """Scalar intrinsic reward emitted on stale→fresh transitions."""


class RegionStalenessTracker(InfoGainModule):
    """Grid-based staleness tracker.

    Parameters
    ----------
    n_rows, n_cols :
        Spatial partitioning resolution (e.g. 8×8 for minimap).
    scout_horizon_s :
        Seconds after which a slot reaches full staleness.
    stale_threshold :
        Staleness fraction above which a rediscovery is rewarded.
    scout_drive_weight :
        Coefficient on intrinsic reward.  **0.0 disables** the drive.
    never_seen_bonus :
        Multiplier on intrinsic reward for first-time discoveries vs
        re-scouting.
    """

    def __init__(
        self,
        n_rows: int = 8,
        n_cols: int = 8,
        scout_horizon_s: float = 60.0,
        stale_threshold: float = 0.5,
        scout_drive_weight: float = 0.1,
        never_seen_bonus: float = 2.0,
    ) -> None:
        self._n_rows = n_rows
        self._n_cols = n_cols
        self._n_slots = n_rows * n_cols
        self._scout_horizon_s = max(scout_horizon_s, 1e-6)
        self._stale_threshold = stale_threshold
        self._scout_drive_weight = scout_drive_weight
        self._never_seen_bonus = never_seen_bonus

        # Per-slot state
        self._last_seen_t: np.ndarray = np.full(self._n_slots, np.nan, dtype=np.float64)
        self._current_time: float = 0.0
        self._prev_staleness: np.ndarray = np.ones(self._n_slots, dtype=np.float64)
        self._pending_reward: float = 0.0

    # -- Properties --------------------------------------------------------

    @property
    def n_slots(self) -> int:
        return self._n_slots

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self._n_rows, self._n_cols)

    # -- InfoGainModule interface ------------------------------------------

    def reset(self) -> None:
        self._last_seen_t[:] = np.nan
        self._current_time = 0.0
        self._prev_staleness[:] = 1.0
        self._pending_reward = 0.0

    def update(self, obs: np.ndarray, info: dict) -> None:
        """Update staleness from an observation.

        Expected ``info`` keys:

        * ``"time_s"`` — current episode time in seconds.
        * ``"visible_slots"`` — 1-D bool array of length ``n_slots``
          indicating which slots are currently visible.
        """
        self._current_time = float(info.get("time_s", self._current_time))
        visible = info.get("visible_slots")
        if visible is None:
            return

        visible = np.asarray(visible, dtype=bool)
        if visible.shape != (self._n_slots,):
            return

        old_staleness = self._compute_staleness()

        # Mark visible slots as seen *now*.
        self._last_seen_t[visible] = self._current_time

        new_staleness = self._compute_staleness()

        # Reward for stale→fresh transitions.
        reward = 0.0
        for i in range(self._n_slots):
            if old_staleness[i] > self._stale_threshold and new_staleness[i] <= self._stale_threshold:
                bonus = self._never_seen_bonus if np.isnan(self._prev_staleness[i]) and old_staleness[i] >= 1.0 else 1.0
                reward += bonus

        self._pending_reward = reward * self._scout_drive_weight
        self._prev_staleness = new_staleness.copy()

    def staleness(self) -> np.ndarray:
        return self._compute_staleness()

    def intrinsic_reward(self) -> float:
        r = self._pending_reward
        self._pending_reward = 0.0
        return r

    # -- Internal ----------------------------------------------------------

    def _compute_staleness(self) -> np.ndarray:
        """Compute per-slot staleness from last-seen timestamps."""
        s = np.ones(self._n_slots, dtype=np.float64)
        observed = ~np.isnan(self._last_seen_t)
        if observed.any():
            dt = self._current_time - self._last_seen_t[observed]
            s[observed] = np.clip(dt / self._scout_horizon_s, 0.0, 1.0)
        return s
