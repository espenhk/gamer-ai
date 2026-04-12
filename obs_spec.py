"""Backward-compatibility shim.

All symbols previously defined here have moved to:
  framework.obs_spec   — ObsDim, ObsSpec classes
  games.tmnf.obs_spec  — TMNF-specific OBS_SPEC data and derived constants

Import from those modules directly in new code.
"""

from framework.obs_spec import ObsDim, ObsSpec  # noqa: F401
from games.tmnf.obs_spec import (  # noqa: F401
    OBS_SPEC,
    TMNF_OBS_SPEC,
    BASE_OBS_DIM,
    OBS_NAMES,
    OBS_SCALES,
    N_LOOKAHEAD,
    LOOKAHEAD_STEPS,
    obs_names_with_lidar,
    obs_scales_with_lidar,
)
