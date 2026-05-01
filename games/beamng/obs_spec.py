"""BeamNG-specific observation space definition.

This is the single source of truth for all observation features produced by
the BeamNG game integration.

The framework layer receives an ObsSpec at construction time and never
references BeamNG features by name — this module is only imported by BeamNG code.
"""

from __future__ import annotations

import numpy as np

from framework.obs_spec import ObsDim, ObsSpec


# ---------------------------------------------------------------------------
# BeamNG observation spec
# ---------------------------------------------------------------------------

_BEAMNG_DIMS: list[ObsDim] = [
    ObsDim("speed_ms",          50.0,    "Vehicle speed in m/s"),
    ObsDim("lateral_offset_m",   5.0,    "Metres from track centre (neg=left, pos=right)"),
    ObsDim("yaw_error_rad",      3.14159, "Track heading minus car heading, [−π, π]"),
    ObsDim("track_progress",     1.0,    "Fraction of track completed, [0, 1]"),
    ObsDim("pitch_rad",          0.3,    "Nose-up/down rotation"),
    ObsDim("roll_rad",           0.3,    "Tilt left/right"),
    ObsDim("wheel_0_contact",    1.0,    "Front-left wheel ground contact (0 or 1)"),
    ObsDim("wheel_1_contact",    1.0,    "Front-right wheel ground contact (0 or 1)"),
    ObsDim("wheel_2_contact",    1.0,    "Rear-left wheel ground contact (0 or 1)"),
    ObsDim("wheel_3_contact",    1.0,    "Rear-right wheel ground contact (0 or 1)"),
    ObsDim("angular_vel_x",      5.0,    "Angular velocity x (rad/s)"),
    ObsDim("angular_vel_y",      5.0,    "Angular velocity y (rad/s)"),
    ObsDim("angular_vel_z",      5.0,    "Angular velocity z (rad/s)"),
]

#: The canonical BeamNG observation spec.
BEAMNG_OBS_SPEC: ObsSpec = ObsSpec(_BEAMNG_DIMS)

#: Number of base observation features.
BASE_OBS_DIM: int = BEAMNG_OBS_SPEC.dim

#: Ordered list of feature names.
OBS_NAMES: list[str] = BEAMNG_OBS_SPEC.names

#: Float32 scale array, shape (BASE_OBS_DIM,).
OBS_SCALES: np.ndarray = BEAMNG_OBS_SPEC.scales

#: Plain OBS_SPEC list for callers that iterate over ObsDim entries.
OBS_SPEC: list[ObsDim] = _BEAMNG_DIMS
