"""Belief modules for partially observable environments.

A belief state persists information about partially-observed entities whose
state evolves while unobserved.  Each slot stores a *value* and a
*confidence* that decays over time without new evidence.

BeliefModule
    Abstract base class.  Concrete implementations must provide
    ``reset``, ``update``, ``project``, and ``encode``.

EWMABelief
    Exponentially-weighted decay of last-known feature values.
    Confidence is ``exp(-dt / decay_tau)`` per slot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BeliefModule(ABC):
    """Abstract interface for belief-state tracking."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all belief state (call at episode start)."""

    @abstractmethod
    def update(self, obs: np.ndarray, info: dict) -> None:
        """Ingest new observation + game info; update internal belief."""

    @abstractmethod
    def project(self, dt_seconds: float) -> None:
        """Advance belief in time without new evidence (for unobserved entities)."""

    @abstractmethod
    def encode(self) -> np.ndarray:
        """Return fixed-size vector of ``(value, confidence)`` pairs.

        Shape is ``(2 * n_slots,)`` — interleaved ``[v0, c0, v1, c1, ...]``.
        Confidence is a function of time since last evidence, governed by
        ``decay_tau``.
        """


class EWMABelief(BeliefModule):
    """Exponentially-weighted moving-average belief tracker.

    Each *slot* stores the last-known feature value and a confidence that
    decays as ``exp(-dt / decay_tau)``.

    Parameters
    ----------
    n_slots :
        Number of independent belief slots.
    decay_tau :
        Decay time constant (seconds).  Can be a scalar (same for all
        slots) or a 1-D array of per-slot values.
    """

    def __init__(
        self,
        n_slots: int,
        decay_tau: float | np.ndarray = 30.0,
    ) -> None:
        self._n_slots = n_slots
        if np.isscalar(decay_tau):
            self._decay_tau = np.full(n_slots, float(decay_tau), dtype=np.float64)
        else:
            self._decay_tau = np.asarray(decay_tau, dtype=np.float64)
            if self._decay_tau.shape != (n_slots,):
                raise ValueError(
                    f"decay_tau shape {self._decay_tau.shape} != ({n_slots},)"
                )

        self._values: np.ndarray = np.zeros(n_slots, dtype=np.float64)
        self._confidence: np.ndarray = np.zeros(n_slots, dtype=np.float64)
        self._dt_accum: np.ndarray = np.zeros(n_slots, dtype=np.float64)

    # -- Properties --------------------------------------------------------

    @property
    def n_slots(self) -> int:
        return self._n_slots

    @property
    def values(self) -> np.ndarray:
        return self._values.copy()

    @property
    def confidence(self) -> np.ndarray:
        return self._confidence.copy()

    # -- BeliefModule interface --------------------------------------------

    def reset(self) -> None:
        self._values[:] = 0.0
        self._confidence[:] = 0.0
        self._dt_accum[:] = 0.0

    def update(self, obs: np.ndarray, info: dict) -> None:
        """Update belief from observation.

        ``obs`` is expected to be a 1-D array of length ``n_slots``.
        Non-NaN entries are treated as fresh evidence; NaN entries
        indicate that the slot is currently unobserved.
        """
        obs = np.asarray(obs, dtype=np.float64)
        observed = ~np.isnan(obs)
        self._values[observed] = obs[observed]
        self._confidence[observed] = 1.0
        self._dt_accum[observed] = 0.0

    def project(self, dt_seconds: float) -> None:
        self._dt_accum += dt_seconds
        safe_tau = np.where(self._decay_tau > 0, self._decay_tau, 1.0)
        self._confidence = np.exp(-self._dt_accum / safe_tau)

    def encode(self) -> np.ndarray:
        out = np.empty(2 * self._n_slots, dtype=np.float32)
        out[0::2] = self._values
        out[1::2] = self._confidence
        return out
