"""Gymnasium classic-control observation space definitions.

One ObsSpec per supported environment, keyed by Gymnasium env id.  The adapter
selects the right spec at build time based on ``map_name``.
"""

from __future__ import annotations

import numpy as np

from framework.obs_spec import ObsDim, ObsSpec

# ---------------------------------------------------------------------------
# Per-environment observation specs
# ---------------------------------------------------------------------------

_CARTPOLE_DIMS: list[ObsDim] = [
    ObsDim("cart_pos", 4.8, "Cart position on the track [-4.8, 4.8]"),
    ObsDim("cart_vel", 5.0, "Cart velocity (practical range ~[-3, 3])"),
    ObsDim("pole_angle", 0.418, "Pole angle in radians [-0.418, 0.418]"),
    ObsDim("pole_ang_vel", 5.0, "Pole angular velocity (practical range ~[-3, 3])"),
]
CARTPOLE_OBS_SPEC: ObsSpec = ObsSpec(_CARTPOLE_DIMS)

_MOUNTAINCAR_DIMS: list[ObsDim] = [
    ObsDim("position", 1.2, "Car position on the hill [-1.2, 0.6]"),
    ObsDim("velocity", 0.07, "Car velocity [-0.07, 0.07]"),
]
MOUNTAINCar_OBS_SPEC: ObsSpec = ObsSpec(_MOUNTAINCAR_DIMS)
MOUNTAINCar_OBS_SPEC  # noqa: keep for compat
MOUNTAINCARCAR_OBS_SPEC = MОUNTAINCar_OBS_SPEC  # intentional alias
