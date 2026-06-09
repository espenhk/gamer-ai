"""MineRL observation space definition (Phase 1: vector obs only).

Phase 1 uses a compact feature vector derived from the MineRL obs dict.
Pixel observations (``pov``) are not included; that is deferred to Phase 2
once a CNN policy is available for non-SC2 games.

Supported environments:
  - MineRLNavigateDense-v0  (compass angle + dirt inventory)
  - MineRLTreechop-v0       (log inventory)
"""

from __future__ import annotations

import numpy as np

from framework.obs_spec import ObsDim, ObsSpec

_MINERL_DIMS: list[ObsDim] = [
    ObsDim("compass_angle", 180.0, "Compass bearing to goal in degrees [-180, 180]"),
    ObsDim("inventory_dirt", 64.0, "Dirt blocks in inventory [0, 64]"),
    ObsDim("inventory_log", 64.0, "Log blocks in inventory [0, 64]"),
]

#: The canonical MineRL observation spec.
MINERL_OBS_SPEC: ObsSpec = ObsSpec(_MINERL_DIMS)

#: Number of base observation features.
BASE_OBS_DIM: int = MINERL_OBS_SPEC.dim

#: Ordered list of feature names.
OBS_NAMES: list[str] = MINERL_OBS_SPEC.names

#: Float32 scale array, shape (BASE_OBS_DIM,).
OBS_SCALES: np.ndarray = MINERL_OBS_SPEC.scales

#: Plain OBS_SPEC list for callers that iterate over ObsDim entries.
OBS_SPEC: list[ObsDim] = _MINERL_DIMS
