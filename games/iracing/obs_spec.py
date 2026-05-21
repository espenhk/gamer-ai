"""iRacing observation space definition.

Telemetry-driven observation vector sourced from pyirsdk.  This is the
single source of truth for all observation features produced by the
iRacing game integration.

The framework layer receives an ObsSpec at construction time and never
references iRacing features by name — this module is only imported by
iRacing code.
"""

from __future__ import annotations

import numpy as np

from framework.obs_spec import ObsDim, ObsSpec


_IRACING_DIMS: list[ObsDim] = [
    ObsDim("speed_ms",           100.0,  "Vehicle speed in m/s"),
    ObsDim("lateral_offset_m",     5.0,  "Metres from track centre (neg=left, pos=right)"),
    ObsDim("track_progress",       1.0,  "Fraction of track completed, [0, 1]"),
    ObsDim("yaw_error_rad",        3.14159, "Track heading minus car heading, [−π, π]"),
    ObsDim("rpm",               8000.0,  "Engine RPM"),
    ObsDim("gear",                 6.0,  "Current gear (0=neutral, -1=reverse)"),
    ObsDim("fuel_pct",             1.0,  "Fuel level as fraction [0, 1]"),
    ObsDim("throttle",             1.0,  "Throttle input [0, 1]"),
    ObsDim("brake",                1.0,  "Brake input [0, 1]"),
    ObsDim("steering",             1.0,  "Steering input [-1, 1]"),
    ObsDim("tire_load_fl",      5000.0,  "Front-left tyre load (N)"),
    ObsDim("tire_load_fr",      5000.0,  "Front-right tyre load (N)"),
    ObsDim("tire_load_rl",      5000.0,  "Rear-left tyre load (N)"),
    ObsDim("tire_load_rr",      5000.0,  "Rear-right tyre load (N)"),
    ObsDim("tire_temp_fl",       150.0,  "Front-left tyre surface temp (°C)"),
    ObsDim("tire_temp_fr",       150.0,  "Front-right tyre surface temp (°C)"),
    ObsDim("tire_temp_rl",       150.0,  "Rear-left tyre surface temp (°C)"),
    ObsDim("tire_temp_rr",       150.0,  "Rear-right tyre surface temp (°C)"),
    ObsDim("brake_bias",           1.0,  "Brake bias front/rear [0, 1]"),
    ObsDim("lap_time_s",         120.0,  "Current lap elapsed time (s)"),
    ObsDim("best_lap_time_s",    120.0,  "Session best lap time (s), 0 if none"),
]

#: The canonical iRacing observation spec.
IRACING_OBS_SPEC: ObsSpec = ObsSpec(_IRACING_DIMS)

#: Number of base observation features.
BASE_OBS_DIM: int = IRACING_OBS_SPEC.dim

#: Ordered list of feature names.
OBS_NAMES: list[str] = IRACING_OBS_SPEC.names

#: Float32 scale array, shape (BASE_OBS_DIM,).
OBS_SCALES: np.ndarray = IRACING_OBS_SPEC.scales

#: Plain OBS_SPEC list for callers that iterate over ObsDim entries.
OBS_SPEC: list[ObsDim] = _IRACING_DIMS
