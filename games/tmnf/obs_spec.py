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
# Lookahead configuration (issue #493)
# ---------------------------------------------------------------------------
#
# Legacy hardcoded defaults, kept for backward compatibility: 3 waypoints at
# centerline indices [10, 25, 50] (unevenly spaced, ~20/50/100 m ahead at the
# default 2 m point spacing set by games/tmnf/tools/build_centerline.py).
#
# Both counts and spacing are now configurable via training_params.yaml's
# `n_lookahead_points` / `lookahead_step_spacing` (see build_lookahead_steps()
# and build_tmnf_obs_spec() below) so a longer/denser lookahead can be swept
# in a grid search without touching this module. Leaving both unset reproduces
# the legacy list exactly.

# Number of waypoints ahead to include in the observation (legacy default).
N_LOOKAHEAD: int = 3
# Centerline indices (relative to nearest point) for each lookahead slot
# (legacy default — unevenly spaced).
LOOKAHEAD_STEPS: list[int] = [10, 25, 50]


def build_lookahead_steps(
    n_lookahead_points: int | None = None,
    lookahead_step_spacing: int | None = None,
) -> list[int]:
    """Return the centerline-index offsets for each lookahead slot.

    With both arguments left at ``None``, returns the legacy hardcoded
    ``LOOKAHEAD_STEPS`` list unchanged (backward compatible). Passing either
    argument switches to an evenly-spaced schedule: *n_lookahead_points*
    waypoints spaced *lookahead_step_spacing* centerline indices apart
    (``spacing, 2*spacing, ..., n*spacing``), defaulting the other argument to
    its legacy-equivalent value (``N_LOOKAHEAD`` / ``20``) when omitted.
    """
    if n_lookahead_points is None and lookahead_step_spacing is None:
        return list(LOOKAHEAD_STEPS)
    n = n_lookahead_points if n_lookahead_points is not None else N_LOOKAHEAD
    spacing = lookahead_step_spacing if lookahead_step_spacing is not None else 20
    return [spacing * (i + 1) for i in range(n)]


# ---------------------------------------------------------------------------
# TMNF observation spec — 21 base features (no LIDAR) with the legacy 3-point
# lookahead; use build_tmnf_obs_spec() for a custom lookahead schedule.
# ---------------------------------------------------------------------------

_BASE_DIMS: list[ObsDim] = [
    ObsDim("speed_ms", 50.0, "Vehicle speed in m/s"),
    ObsDim("lateral_offset_m", 5.0, "Metres from centreline (neg=left, pos=right)"),
    ObsDim("vertical_offset_m", 2.0, "Metres above (+) / below (-) centreline"),
    ObsDim("yaw_error_rad", 3.14159, "Track heading minus car heading, [−π, π]"),
    ObsDim("pitch_rad", 0.3, "Nose-up/down rotation"),
    ObsDim("roll_rad", 0.3, "Tilt left/right"),
    ObsDim("track_progress", 1.0, "Fraction of track completed, [0, 1]"),
    ObsDim("turning_rate", 65536.0, "Raw TMInterface steer value, ±65536"),
    ObsDim("wheel_0_contact", 1.0, "Front-left wheel ground contact (0 or 1)"),
    ObsDim("wheel_1_contact", 1.0, "Front-right wheel ground contact (0 or 1)"),
    ObsDim("wheel_2_contact", 1.0, "Rear-left wheel ground contact (0 or 1)"),
    ObsDim("wheel_3_contact", 1.0, "Rear-right wheel ground contact (0 or 1)"),
    ObsDim("angular_vel_x", 5.0, "Roll rate (rad/s)"),
    ObsDim("angular_vel_y", 5.0, "Yaw rate (rad/s)"),
    ObsDim("angular_vel_z", 5.0, "Pitch rate (rad/s)"),
]


def _lookahead_dims(lookahead_steps: list[int]) -> list[ObsDim]:
    """Build the interleaved (lateral offset, heading change) ObsDim pairs
    for each entry in *lookahead_steps*, in order."""
    dims: list[ObsDim] = []
    for step in lookahead_steps:
        dims.append(ObsDim(f"lookahead_{step}_lat", 5.0, f"Lateral offset {step} pts ahead (m)"))
        dims.append(ObsDim(f"lookahead_{step}_yaw", 3.14, f"Heading change {step} pts ahead (rad)"))
    return dims


def build_tmnf_obs_spec_from_steps(lookahead_steps: list[int]) -> ObsSpec:
    """Build a TMNF ObsSpec (no LIDAR) from an already-resolved lookahead
    step list. Used by env.py, which receives the resolved list (rather than
    the raw n_lookahead_points/lookahead_step_spacing config knobs) so the
    obs dimensionality it computes always matches build_lookahead_steps()."""
    return ObsSpec(_BASE_DIMS + _lookahead_dims(lookahead_steps))


def build_tmnf_obs_spec(
    n_lookahead_points: int | None = None,
    lookahead_step_spacing: int | None = None,
) -> ObsSpec:
    """Build a TMNF ObsSpec (no LIDAR) for the given lookahead configuration.

    Leaving both arguments ``None`` reproduces the legacy 21-dim spec
    (``TMNF_OBS_SPEC``) exactly. See build_lookahead_steps() for how the
    arguments map to centerline-index offsets.
    """
    steps = build_lookahead_steps(n_lookahead_points, lookahead_step_spacing)
    return build_tmnf_obs_spec_from_steps(steps)


#: The canonical TMNF observation spec (no LIDAR), using the legacy 3-point lookahead.
#: Use `TMNF_OBS_SPEC.with_lidar(n)` to get a spec extended with LIDAR rays, or
#: `build_tmnf_obs_spec(...)` for a custom lookahead schedule.
TMNF_OBS_SPEC: ObsSpec = build_tmnf_obs_spec()
_TMNF_DIMS: list[ObsDim] = TMNF_OBS_SPEC.dims


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
