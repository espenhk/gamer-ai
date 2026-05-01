"""Assetto Corsa observation space.

Single source of truth for AC observation features. Mirrors the layout of
games.tmnf.obs_spec but uses telemetry signals exposed by the AC gym
wrapper. Vision features (when enabled) are appended at the end so all
linear policies handle the variable-length observation transparently via
ObsSpec.
"""

from __future__ import annotations

from framework.obs_spec import ObsDim, ObsSpec


# ---------------------------------------------------------------------------
# Base AC observation dims
# ---------------------------------------------------------------------------

_AC_DIMS: list[ObsDim] = [
    ObsDim("speed_ms",          50.0,    "Vehicle speed in m/s"),
    ObsDim("lateral_offset_m",   5.0,    "Metres from track centreline (neg=left, pos=right)"),
    ObsDim("yaw_error_rad",      3.14159, "Track heading minus car heading, [−π, π]"),
    ObsDim("pitch_rad",          0.3,    "Nose-up/down rotation"),
    ObsDim("roll_rad",           0.3,    "Tilt left/right"),
    ObsDim("track_progress",     1.0,    "Fraction of track completed, [0, 1]"),
    ObsDim("steering_angle",     1.0,    "Current steering input in [-1, 1]"),
    ObsDim("engine_rpm",      8000.0,    "Engine RPM"),
    ObsDim("gear",               6.0,    "Current gear (0=R, 1=N, 2..7 forward)"),
    ObsDim("wheel_0_slip",       1.0,    "Front-left wheel slip ratio"),
    ObsDim("wheel_1_slip",       1.0,    "Front-right wheel slip ratio"),
    ObsDim("wheel_2_slip",       1.0,    "Rear-left wheel slip ratio"),
    ObsDim("wheel_3_slip",       1.0,    "Rear-right wheel slip ratio"),
    ObsDim("angular_vel_x",      5.0,    "Roll rate (rad/s)"),
    ObsDim("angular_vel_y",      5.0,    "Yaw rate (rad/s)"),
    ObsDim("angular_vel_z",      5.0,    "Pitch rate (rad/s)"),
]

#: The canonical AC observation spec (no vision features).
#: Use ``AC_OBS_SPEC.with_vision(n)`` to extend with vision rays.
AC_OBS_SPEC: ObsSpec = ObsSpec(_AC_DIMS)


#: Number of base observation features (no vision).
BASE_OBS_DIM: int = AC_OBS_SPEC.dim


def with_vision(n_vision: int) -> ObsSpec:
    """Return an AC ObsSpec extended with *n_vision* vision-distance features.

    Vision features are normalised to ~[0, 1] so their scale is 1.0.
    Returns the base spec unchanged when n_vision == 0.
    """
    if n_vision == 0:
        return AC_OBS_SPEC
    extra = [
        ObsDim(f"vision_{i}", 1.0, "Vision distance feature ~[0, 1]")
        for i in range(n_vision)
    ]
    return ObsSpec(_AC_DIMS + extra)
