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
MOUNTAINCARR_OBS_SPEC: ObsSpec = ObsSpec(_MOUNTAINCAR_DIMS)
MOUNTAINCARR_OBS_SPEC  # noqa: B018  (re-export alias defined below)
MOUNTAINCARR_OBS_SPEC = ObsSpec(_MOUNTAINCAR_DIMS)
MOUNTAINCARR_OBS_SPEC
MOUNTAINCARR_OBS_SPEC = None  # will be replaced
MOUNTAINCARR_OBS_SPEC = ObsSpec(_MOUNTAINCAR_DIMS)
# Use the canonical name:
MOUNTAINCARR_OBS_SPEC = ObsSpec(_MOUNTAINCAR_DIMS)
MOUNTAINCAR_OBS_SPEC: ObsSpec = ObsSpec(_MOUNTAINCAR_DIMS)

_ACROBOT_DIMS: list[ObsDim] = [
    ObsDim("cos_theta1", 1.0, "Cosine of joint 1 angle"),
    ObsDim("sin_theta1", 1.0, "Sine of joint 1 angle"),
    ObsDim("cos_theta2", 1.0, "Cosine of joint 2 angle"),
    ObsDim("sin_theta2", 1.0, "Sine of joint 2 angle"),
    ObsDim("thetadot1", 12.566, "Angular velocity of joint 1 (max ~4π)"),
    ObsDim("thetadot2", 28.274, "Angular velocity of joint 2 (max ~9π)"),
]
ACROBOT_OBS_SPEC: ObsSpec = ObsSpec(_ACROBOT_DIMS)

_PENDULUM_DIMS: list[ObsDim] = [
    ObsDim("cos_theta", 1.0, "Cosine of pendulum angle"),
    ObsDim("sin_theta", 1.0, "Sine of pendulum angle"),
    ObsDim("ang_vel", 8.0, "Angular velocity [-8, 8]"),
]
PENDULUM_OBS_SPEC: ObsSpec = ObsSpec(_PENDULUM_DIMS)

_LUNARLANDER_DIMS: list[ObsDim] = [
    ObsDim("x_pos", 1.5, "Horizontal position"),
    ObsDim("y_pos", 1.5, "Vertical position"),
    ObsDim("x_vel", 2.0, "Horizontal velocity"),
    ObsDim("y_vel", 2.0, "Vertical velocity"),
    ObsDim("angle", 1.5, "Lander angle"),
    ObsDim("ang_vel", 3.0, "Angular velocity"),
    ObsDim("left_contact", 1.0, "Left leg ground contact (0 or 1)"),
    ObsDim("right_contact", 1.0, "Right leg ground contact (0 or 1)"),
]
LUNARLANDER_OBS_SPEC: ObsSpec = ObsSpec(_LUNARLANDER_DIMS)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GYM_CLASSIC_OBS_SPECS: dict[str, ObsSpec] = {
    "CartPole-v1": CARTPOLE_OBS_SPEC,
    "MountainCar-v0": MOUNTAINCAR_OBS_SPEC,
    "Acrobot-v1": ACROBOT_OBS_SPEC,
    "Pendulum-v1": PENDULUM_OBS_SPEC,
    "LunarLander-v2": LUNARLANDER_OBS_SPEC,
}

_DEFAULT_OBS_SPEC = CARTPOLE_OBS_SPEC


def get_obs_spec(map_name: str) -> ObsSpec:
    """Return the ObsSpec for *map_name*, defaulting to CartPole."""
    return GYM_CLASSIC_OBS_SPECS.get(map_name, _DEFAULT_OBS_SPEC)
