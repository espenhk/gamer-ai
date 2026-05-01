"""TORCS-specific observation space definition.

This is the single source of truth for all observation features produced by
the TORCS game integration.  Import TORCS_OBS_SPEC (or its derived constants)
rather than redefining feature lists elsewhere.

The framework layer receives an ObsSpec at construction time and never
references TORCS features by name — this module is only imported by TORCS code.
"""

from __future__ import annotations

import numpy as np

from framework.obs_spec import ObsDim, ObsSpec


# ---------------------------------------------------------------------------
# TORCS observation spec — 19 base features
# ---------------------------------------------------------------------------
# TORCS exposes a rich sensor set via the SCR (Simulated Car Racing) server.
# The features below are chosen to align as closely as possible with the TMNF
# observation vector while adding TORCS-specific track-edge sensors.

_TORCS_DIMS: list[ObsDim] = [
    ObsDim("speed_ms",          50.0,    "Vehicle speed in m/s (longitudinal)"),
    ObsDim("lateral_offset_m",   5.0,    "Metres from track centre (neg=left, pos=right)"),
    ObsDim("yaw_error_rad",      3.14159, "Track heading minus car heading, [−π, π]"),
    ObsDim("track_progress",     1.0,    "Fraction of lap completed, [0, 1]"),
    ObsDim("rpm",             10000.0,   "Engine RPM"),
    ObsDim("wheel_0_spin",      200.0,   "Front-left wheel spin velocity (rad/s)"),
    ObsDim("wheel_1_spin",      200.0,   "Front-right wheel spin velocity (rad/s)"),
    ObsDim("wheel_2_spin",      200.0,   "Rear-left wheel spin velocity (rad/s)"),
    ObsDim("wheel_3_spin",      200.0,   "Rear-right wheel spin velocity (rad/s)"),
    # Track edge sensors: distances to track edges at various angles.
    # TORCS provides 19 rangefinder-like sensors; we pick 10 representative ones.
    ObsDim("track_edge_0",     200.0,    "Track edge distance at -90°"),
    ObsDim("track_edge_1",     200.0,    "Track edge distance at -60°"),
    ObsDim("track_edge_2",     200.0,    "Track edge distance at -30°"),
    ObsDim("track_edge_3",     200.0,    "Track edge distance at -10°"),
    ObsDim("track_edge_4",     200.0,    "Track edge distance at 0° (ahead)"),
    ObsDim("track_edge_5",     200.0,    "Track edge distance at 10°"),
    ObsDim("track_edge_6",     200.0,    "Track edge distance at 30°"),
    ObsDim("track_edge_7",     200.0,    "Track edge distance at 60°"),
    ObsDim("track_edge_8",     200.0,    "Track edge distance at 90°"),
    ObsDim("track_position",     1.0,    "Normalised track position [-1, 1] (centre=0)"),
]

#: The canonical TORCS observation spec.
TORCS_OBS_SPEC: ObsSpec = ObsSpec(_TORCS_DIMS)

# ---------------------------------------------------------------------------
# Derived constants — mirror the style used by games.tmnf.obs_spec.
# ---------------------------------------------------------------------------

#: Number of base observation features.
BASE_OBS_DIM: int = TORCS_OBS_SPEC.dim

#: Ordered list of feature names for the base observation.
OBS_NAMES: list[str] = TORCS_OBS_SPEC.names

#: Float32 scale array for the base observation, shape (BASE_OBS_DIM,).
OBS_SCALES: np.ndarray = TORCS_OBS_SPEC.scales

# Expose the plain OBS_SPEC list for callers that iterate over ObsDim entries.
OBS_SPEC: list[ObsDim] = _TORCS_DIMS
