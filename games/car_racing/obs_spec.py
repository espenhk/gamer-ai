"""CarRacing (gymnasium) observation space definition.

CarRacing-v3 uses a 96x96x3 pixel observation by default.  This integration
extracts a compact feature vector instead, for compatibility with the
WeightedLinearPolicy framework: car-physics features (speed, angular
velocity, wheel spin, current control inputs) plus track-relative
perception features mirroring TMNF's obs_spec.py (lateral offset from
centreline, heading error, track progress, and a lookahead schedule of
upcoming curvature) so the agent can anticipate turns instead of only
reacting to its own physics state.

The track-relative features are derived at runtime (games/car_racing/env.py)
from ``env.unwrapped.track``, the list of ``(alpha, beta, x, y)`` centreline
checkpoints CarRacing-v3 already builds internally.
"""

from __future__ import annotations

import numpy as np

from framework.obs_spec import ObsDim, ObsSpec

# ---------------------------------------------------------------------------
# Lookahead configuration — mirrors games/tmnf/obs_spec.py's pattern.
# ---------------------------------------------------------------------------
#
# Offsets are in track-checkpoint indices (not metres): CarRacing-v3 tracks
# have ~280-320 checkpoints spaced ~3.5 units apart around the full loop, so
# these defaults cover roughly one to a few tiles ahead.

# Number of waypoints ahead to include in the observation (legacy default).
N_LOOKAHEAD: int = 3
# Track-checkpoint-index offsets (relative to the nearest point) for each
# lookahead slot (legacy default).
LOOKAHEAD_STEPS: list[int] = [5, 15, 30]


def build_lookahead_steps(
    n_lookahead_points: int | None = None,
    lookahead_step_spacing: int | None = None,
) -> list[int]:
    """Return the checkpoint-index offsets for each lookahead slot.

    With both arguments left at ``None``, returns the legacy hardcoded
    ``LOOKAHEAD_STEPS`` list unchanged (backward compatible). Passing either
    argument switches to an evenly-spaced schedule: *n_lookahead_points*
    waypoints spaced *lookahead_step_spacing* checkpoint indices apart
    (``spacing, 2*spacing, ..., n*spacing``), defaulting the other argument
    to its legacy-equivalent value (``N_LOOKAHEAD`` / ``10``) when omitted.
    """
    if n_lookahead_points is None and lookahead_step_spacing is None:
        return list(LOOKAHEAD_STEPS)
    n = n_lookahead_points if n_lookahead_points is not None else N_LOOKAHEAD
    spacing = lookahead_step_spacing if lookahead_step_spacing is not None else 10
    return [spacing * (i + 1) for i in range(n)]


# ---------------------------------------------------------------------------
# Base observation dims — car-physics features (always present) plus the
# track-relative perception features (issue: car_racing had no track
# perception at all — the agent could only react, never anticipate a turn).
# ---------------------------------------------------------------------------

_BASE_DIMS: list[ObsDim] = [
    ObsDim("speed", 100.0, "Vehicle speed (raw Box2D hull velocity magnitude; policies normalise via ObsSpec.scales)"),
    ObsDim("angular_vel", 10.0, "Angular velocity of the car body"),
    ObsDim("wheel_0_ang", 300.0, "Front-left wheel angular velocity"),
    ObsDim("wheel_1_ang", 300.0, "Front-right wheel angular velocity"),
    ObsDim("wheel_2_ang", 300.0, "Rear-left wheel angular velocity"),
    ObsDim("wheel_3_ang", 300.0, "Rear-right wheel angular velocity"),
    ObsDim("steering", 1.0, "Current steering input [-1, 1]"),
    ObsDim("gas", 1.0, "Current gas input [0, 1]"),
    ObsDim("brake", 1.0, "Current brake input [0, 1]"),
    ObsDim("lateral_offset_m", 5.0, "Signed distance from track centreline (neg=left, pos=right)"),
    ObsDim("yaw_error_rad", 3.14159, "Track heading minus car heading, [-pi, pi]"),
    ObsDim("track_progress", 1.0, "Fraction of the lap's centreline checkpoints passed, [0, 1]"),
]


def _lookahead_dims(lookahead_steps: list[int]) -> list[ObsDim]:
    """Build the interleaved (lateral offset, heading change) ObsDim pairs
    for each entry in *lookahead_steps*, in order."""
    dims: list[ObsDim] = []
    for step in lookahead_steps:
        dims.append(ObsDim(f"lookahead_{step}_lat", 5.0, f"Lateral offset {step} checkpoints ahead (track units)"))
        dims.append(ObsDim(f"lookahead_{step}_yaw", 3.14159, f"Heading change {step} checkpoints ahead (rad)"))
    return dims


def build_car_racing_obs_spec_from_steps(lookahead_steps: list[int]) -> ObsSpec:
    """Build a CarRacing ObsSpec from an already-resolved lookahead step
    list. Used by env.py, which receives the resolved list (rather than the
    raw n_lookahead_points/lookahead_step_spacing config knobs) so the obs
    dimensionality it computes always matches build_lookahead_steps()."""
    return ObsSpec(_BASE_DIMS + _lookahead_dims(lookahead_steps))


def build_car_racing_obs_spec(
    n_lookahead_points: int | None = None,
    lookahead_step_spacing: int | None = None,
) -> ObsSpec:
    """Build a CarRacing ObsSpec for the given lookahead configuration.

    Leaving both arguments ``None`` reproduces the legacy spec
    (``CAR_RACING_OBS_SPEC``) exactly. See build_lookahead_steps() for how
    the arguments map to checkpoint-index offsets.
    """
    steps = build_lookahead_steps(n_lookahead_points, lookahead_step_spacing)
    return build_car_racing_obs_spec_from_steps(steps)


#: The canonical CarRacing observation spec, using the default 3-point lookahead.
CAR_RACING_OBS_SPEC: ObsSpec = build_car_racing_obs_spec()
_CAR_RACING_DIMS: list[ObsDim] = CAR_RACING_OBS_SPEC.dims

# ---------------------------------------------------------------------------
# Legacy derived constants — kept for backward-compat with code that imports
# these names directly.  Prefer using CAR_RACING_OBS_SPEC.dim / .names / .scales.
# ---------------------------------------------------------------------------

#: Number of base observation features.
BASE_OBS_DIM: int = CAR_RACING_OBS_SPEC.dim

#: Ordered list of feature names.
OBS_NAMES: list[str] = CAR_RACING_OBS_SPEC.names

#: Float32 scale array, shape (BASE_OBS_DIM,).
OBS_SCALES: np.ndarray = CAR_RACING_OBS_SPEC.scales

#: Plain OBS_SPEC list for callers that iterate over ObsDim entries.
OBS_SPEC: list[ObsDim] = _CAR_RACING_DIMS
