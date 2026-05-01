"""CarRacing (gymnasium) observation space definition.

CarRacing-v2 uses a 96×96×3 pixel observation by default.  We add a compact
feature-vector mode (speed + angular velocity + progress estimate) for
compatibility with the WeightedLinearPolicy framework.
"""

from __future__ import annotations

import numpy as np

from framework.obs_spec import ObsDim, ObsSpec


_CAR_RACING_DIMS: list[ObsDim] = [
    ObsDim("speed",          100.0,  "Vehicle speed (normalised)"),
    ObsDim("angular_vel",     10.0,  "Angular velocity of the car body"),
    ObsDim("wheel_0_ang",    100.0,  "Front-left wheel angular velocity"),
    ObsDim("wheel_1_ang",    100.0,  "Front-right wheel angular velocity"),
    ObsDim("wheel_2_ang",    100.0,  "Rear-left wheel angular velocity"),
    ObsDim("wheel_3_ang",    100.0,  "Rear-right wheel angular velocity"),
    ObsDim("steering",         1.0,  "Current steering input [-1, 1]"),
    ObsDim("gas",              1.0,  "Current gas input [0, 1]"),
    ObsDim("brake",            1.0,  "Current brake input [0, 1]"),
]

#: The canonical CarRacing observation spec.
CAR_RACING_OBS_SPEC: ObsSpec = ObsSpec(_CAR_RACING_DIMS)

#: Number of base observation features.
BASE_OBS_DIM: int = CAR_RACING_OBS_SPEC.dim

#: Ordered list of feature names.
OBS_NAMES: list[str] = CAR_RACING_OBS_SPEC.names

#: Float32 scale array, shape (BASE_OBS_DIM,).
OBS_SCALES: np.ndarray = CAR_RACING_OBS_SPEC.scales

#: Plain OBS_SPEC list for callers that iterate over ObsDim entries.
OBS_SPEC: list[ObsDim] = _CAR_RACING_DIMS
