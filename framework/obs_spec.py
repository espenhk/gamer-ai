"""Generic observation space definition for RL environments.

ObsDim  — a single named, scaled feature dimension.
ObsSpec — an ordered, validated collection of ObsDim entries.

Game integrations create their own ObsSpec instance in games/<name>/obs_spec.py.
The framework uses ObsSpec for policy weight serialisation and obs normalisation
without knowing anything about the specific game.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObsDim:
    """A single observation feature dimension."""
    name: str
    scale: float
    description: str


class ObsSpec:
    """An ordered, indexed collection of ObsDim entries.

    All framework code that needs feature names, scales, or dimensionality
    should operate on an ObsSpec rather than bare lists.

    Example usage:
        spec = ObsSpec([ObsDim("x", 1.0, "x position"), ...])
        obs_normalised = raw_obs / spec.scales
    """

    def __init__(self, dims: list[ObsDim]) -> None:
        self._dims = list(dims)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Total number of features."""
        return len(self._dims)

    @property
    def names(self) -> list[str]:
        """Ordered list of feature names."""
        return [d.name for d in self._dims]

    @property
    def scales(self) -> np.ndarray:
        """Float32 array of divisor scales, shape (dim,)."""
        return np.array([d.scale for d in self._dims], dtype=np.float32)

    @property
    def dims(self) -> list[ObsDim]:
        """Ordered list of all ObsDim entries."""
        return list(self._dims)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def with_lidar(self, n_rays: int) -> "ObsSpec":
        """Return a new ObsSpec extended with *n_rays* LIDAR dimensions.

        LIDAR values are already in ~[0, 1] so their scale is 1.0.
        Returns self unchanged when n_rays == 0.
        """
        if n_rays == 0:
            return self
        lidar_dims = [
            ObsDim(f"lidar_{i}", 1.0, "LIDAR ray wall-distance [0, 1]")
            for i in range(n_rays)
        ]
        return ObsSpec(self._dims + lidar_dims)

    def __repr__(self) -> str:
        return f"ObsSpec(dim={self.dim}, names={self.names[:3]}...)"

    def __len__(self) -> int:
        return self.dim
