"""Canonical observation space definition for the TMNF RL environment.

A single ObsDim entry per feature is the one source of truth for:
  - feature name   (used as YAML key in policy weight files)
  - scale          (divisor that maps raw values to ~[-1, 1])
  - description    (used to generate docstrings and debug output)

Adding a new observation dimension means adding one entry here.
BASE_OBS_DIM, OBS_NAMES, and OBS_SCALES are all derived automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObsDim:
    name: str
    scale: float
    description: str


# Lookahead configuration — how many waypoints ahead to observe, and at which
# centerline indices relative to the current nearest point.
N_LOOKAHEAD: int = 3
LOOKAHEAD_STEPS: list[int] = [10, 25, 50]

OBS_SPEC: list[ObsDim] = [
    ObsDim("speed_ms",          50.0,    "Vehicle speed in m/s"),
    ObsDim("lateral_offset_m",   5.0,    "Metres from centreline (neg=left, pos=right)"),
    ObsDim("vertical_offset_m",  2.0,    "Metres above (+) / below (-) centreline"),
    ObsDim("yaw_error_rad",      3.14159, "Track heading minus car heading, [-\u03c0, \u03c0]"),
    ObsDim("pitch_rad",          0.3,    "Nose-up/down rotation"),
    ObsDim("roll_rad",           0.3,    "Tilt left/right"),
    ObsDim("track_progress",     1.0,    "Fraction of track completed, [0, 1]"),
    ObsDim("turning_rate",   65536.0,    "Raw TMInterface steer value, \u00b165536"),
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

# Derived constants — import these instead of hardcoding 15 / list-literals elsewhere.
BASE_OBS_DIM: int = len(OBS_SPEC)
OBS_NAMES: list[str] = [d.name for d in OBS_SPEC]
OBS_SCALES: np.ndarray = np.array([d.scale for d in OBS_SPEC], dtype=np.float32)


def obs_names_with_lidar(n_lidar_rays: int) -> list[str]:
    return OBS_NAMES + [f"lidar_{i}" for i in range(n_lidar_rays)]


def obs_scales_with_lidar(n_lidar_rays: int) -> np.ndarray:
    """Return the full scale vector. LIDAR values are already ~[0,1] so scale=1."""
    return np.concatenate([OBS_SCALES, np.ones(n_lidar_rays, dtype=np.float32)])
