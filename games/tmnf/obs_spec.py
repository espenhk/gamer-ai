"""TMNF-specific observation space definition.

This is the single source of truth for all observation features produced by
the TMNF game integration. Import TMNF_OBS_SPEC (or its derived constants)
rather than redefining feature lists elsewhere.

The framework layer receives an ObsSpec at construction time and never
references TMNF features by name — this module is only imported by TMNF code.
"""

from __future__ import annotations

import numpy as np

from framework.obs_spec import ObsDim, ObsSpec


# ---------------------------------------------------------------------------
# Lookahead configuration
# ---------------------------------------------------------------------------

# Number of waypoints ahead to include in the observation.
N_LOOKAHEAD: int = 3
# Centerline indices (relative to nearest point) for each lookahead slot.
LOOKAHEAD_STEPS: list[int] = [10, 25, 50]


# ---------------------------------------------------------------------------
# TMNF observation spec — 21 base features (no LIDAR)
# ---------------------------------------------------------------------------

_TMNF_DIMS: list[ObsDim] = [
    ObsDim("speed_ms",          50.0,    "Vehicle speed in m/s"),
    ObsDim("lateral_offset_m",   5.0,    "Metres from centreline (neg=left, pos=right)"),
    ObsDim("vertical_offset_m",  2.0,    "Metres above (+) / below (-) centreline"),
    ObsDim("yaw_error_rad",      3.14159, "Track heading minus car heading, [−π, π]"),
    ObsDim("pitch_rad",          0.3,    "Nose-up/down rotation"),
    ObsDim("roll_rad",           0.3,    "Tilt left/right"),
    ObsDim("track_progress",     1.0,    "Fraction of track completed, [0, 1]"),
    ObsDim("turning_rate",   65536.0,    "Raw TMInterface steer value, ±65536"),
    ObsDim("wheel_0_contact",    1.0,    "Front-left wheel ground contact (0 or 1)"),
    ObsDim("wheel_1_contact",    1.0,    "Front-right wheel ground contact (0 or 1)"),
    ObsDim("wheel_2_contact",    1.0,    "Rear-left wheel ground contact (0 or 1)"),
    ObsDim("wheel_3_contact",    1.0,    "Rear-right wheel ground contact (0 or 1)"),
    ObsDim("angular_vel_x",      5.0,    "Roll rate (rad/s)"),
    ObsDim("angular_vel_y",      5.0,    "Yaw rate (rad/s)"),
    ObsDim("angular_vel_z",      5.0,    "Pitch rate (rad/s)"),
    # Lookahead: (lateral offset m, heading change rad) at each upcoming waypoint.
    ObsDim("lookahead_10_lat",   5.0,    "Lateral offset 10 pts ahead (m)"),
    ObsDim("lookahead_10_yaw",   3.14,   "Heading change 10 pts ahead (rad)"),
    ObsDim("lookahead_25_lat",   5.0,    "Lateral offset 25 pts ahead (m)"),
    ObsDim("lookahead_25_yaw",   3.14,   "Heading change 25 pts ahead (rad)"),
    ObsDim("lookahead_50_lat",   5.0,    "Lateral offset 50 pts ahead (m)"),
    ObsDim("lookahead_50_yaw",   3.14,   "Heading change 50 pts ahead (rad)"),
]

#: The canonical TMNF observation spec (no LIDAR).
#: Use `TMNF_OBS_SPEC.with_lidar(n)` to get a spec extended with LIDAR rays.
TMNF_OBS_SPEC: ObsSpec = ObsSpec(_TMNF_DIMS)


# ---------------------------------------------------------------------------
# Legacy derived constants — kept for backward-compat with code that imports
# these names directly.  Prefer using TMNF_OBS_SPEC.dim / .names / .scales.
# ---------------------------------------------------------------------------

#: Number of base observation features (no LIDAR).
BASE_OBS_DIM: int = TMNF_OBS_SPEC.dim

#: Ordered list of feature names for the base observation.
OBS_NAMES: list[str] = TMNF_OBS_SPEC.names

#: Float32 scale array for the base observation, shape (BASE_OBS_DIM,).
OBS_SCALES: np.ndarray = TMNF_OBS_SPEC.scales

# Expose the plain OBS_SPEC list for callers that iterate over ObsDim entries.
OBS_SPEC: list[ObsDim] = _TMNF_DIMS


def obs_names_with_lidar(n_lidar_rays: int) -> list[str]:
    """Return feature names extended with *n_lidar_rays* LIDAR names."""
    return TMNF_OBS_SPEC.with_lidar(n_lidar_rays).names


def obs_scales_with_lidar(n_lidar_rays: int) -> np.ndarray:
    """Return scale array extended for *n_lidar_rays* LIDAR rays (scale=1)."""
    return TMNF_OBS_SPEC.with_lidar(n_lidar_rays).scales
