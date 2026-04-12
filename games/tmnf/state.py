"""TMNF game state data classes.

Vec3, Quat, WheelState — pure data/math; no TMInterface coupling.
StateData              — extracts and enriches raw TMInterface state objects.
steer_percent()        — converts a steer percentage to TMInterface integer encoding.
"""

from __future__ import annotations

import math
from typing import Any

from games.tmnf.constants import STEER_SCALE
from games.tmnf.obs_spec import LOOKAHEAD_STEPS


class Vec3:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def magnitude(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def compute_speed(self) -> float:
        """Backward-compatible alias for magnitude()."""
        return self.magnitude()


class Quat:
    # TMInterface stores quat as [w, x, y, z]
    def __init__(self, w: float, x: float, y: float, z: float) -> None:
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def yaw(self) -> float:
        """Rotation around Y axis (left/right), in radians."""
        siny_cosp = 2 * (self.w * self.y + self.z * self.x)
        cosy_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        return math.atan2(siny_cosp, cosy_cosp)

    def pitch(self) -> float:
        """Rotation around X axis (nose up/down), in radians."""
        sinp = 2 * (self.w * self.x - self.y * self.z)
        if abs(sinp) >= 1:
            return math.copysign(math.pi / 2, sinp)
        return math.asin(sinp)

    def roll(self) -> float:
        """Rotation around Z axis (tilt left/right), in radians."""
        sinr_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosr_cosp = 1 - 2 * (self.z * self.z + self.x * self.x)
        return math.atan2(sinr_cosp, cosr_cosp)

    def to_euler_degrees(self) -> Vec3:
        """Returns yaw, pitch, roll as a Vec3 in degrees."""
        return Vec3(
            math.degrees(self.yaw()),
            math.degrees(self.pitch()),
            math.degrees(self.roll()),
        )


class WheelState:
    def __init__(self, contact: bool, sliding: bool) -> None:
        self.contact = contact
        self.sliding = sliding


class StateData:
    def __init__(self, state: Any, centerline: Any | None = None,
                 hint_idx: int | None = None) -> None:
        dyna = state.dyna.current_state  # type: ignore[attr-defined]
        mobil = state.scene_mobil        # type: ignore[attr-defined]
        wheels = state.simulation_wheels # type: ignore[attr-defined]

        pos = state.dyna.current_state.position  # type: ignore[attr-defined]
        self.position = Vec3(pos[0], pos[1], pos[2])

        self.velocity = Vec3(dyna.linear_speed[0], dyna.linear_speed[1], dyna.linear_speed[2])
        self.rotation = Quat(dyna.quat[0], dyna.quat[1], dyna.quat[2], dyna.quat[3])
        self.rotation_euler = self.rotation.to_euler_degrees()
        self.angular_velocity = Vec3(dyna.angular_speed[0], dyna.angular_speed[1], dyna.angular_speed[2])

        self.gear = mobil.engine.gear
        self.turning_rate = mobil.turning_rate

        self.wheels = [
            WheelState(
                contact=wheels[i].real_time_state.has_ground_contact,
                sliding=wheels[i].real_time_state.is_sliding,
            )
            for i in range(4)
        ]

        self.track_progress = None
        self.lateral_offset = None
        self.vertical_offset = None
        self.track_forward = None   # unit np.ndarray of track direction at car position
        self._centerline_idx = None  # nearest centerline point index (for windowed search)
        self.lookahead: list[tuple[float, float]] = [(0.0, 0.0)] * 3
        if centerline is not None:
            (self.track_progress, self.lateral_offset, self.vertical_offset,
             self.track_forward, self._centerline_idx) = centerline.project_with_forward(
                self.position, hint_idx=hint_idx
            )
            self.lookahead = [
                centerline.project_ahead(self.position, self._centerline_idx, s)
                for s in LOOKAHEAD_STEPS
            ]

    def __str__(self) -> str:
        contact_str = " ".join(str(int(w.contact)) for w in self.wheels)
        sliding_str = " ".join(str(int(w.sliding)) for w in self.wheels)

        track_str = ""
        if self.track_progress is not None:
            lat_side = "left" if self.lateral_offset < 0 else "right"
            track_str = (
                f"Track progress : {self.track_progress:8.4f}   ({self.track_progress * 100:.1f}%)\n"
                f"Lateral offset : {self.lateral_offset:7.2f} m  ({lat_side})\n"
                f"Vertical offset: {self.vertical_offset:7.2f} m\n\n"
            )

        return (
            f"========================================================\n"
            f"Speed       : {self.velocity.magnitude():7.2f} m/s  \n\n"
            f"Velocity (X): {self.velocity.x:7.2f}\n"
            f"Velocity (Y): {self.velocity.y:7.2f}\n"
            f"Velocity (Z): {self.velocity.z:7.2f}\n\n"
            f"Rotation (X): {self.rotation.pitch():7.2f} (pitch)\n"
            f"Rotation (Y): {self.rotation.yaw():7.2f} (yaw)\n"
            f"Rotation (Z): {self.rotation.roll():7.2f} (roll)\n\n"
            f"Ang.Vel. (X): {self.angular_velocity.x:7.2f}\n"
            f"Ang.Vel. (Y): {self.angular_velocity.y:7.2f}\n"
            f"Ang.Vel. (Z): {self.angular_velocity.z:7.2f}\n\n"
            f"Gear        : {self.gear:7d}\n"
            f"TurningRate : {self.turning_rate:7.2f}\n\n"
            f"{track_str}"
            f"Wh. contact : {contact_str}\n"
            f"Wh. sliding : {sliding_str}\n"
            f"========================================================\n"
        )


def steer_percent(pct: int) -> int:  # -100 = full left, 0 = straight, 100 = full right
    return int(pct / 100 * STEER_SCALE)
