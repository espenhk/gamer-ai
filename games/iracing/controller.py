"""iRacing action injection controllers.

Provides a pluggable controller abstraction for sending steer/throttle/brake
commands to iRacing.

``NullController``
    Default (Phase 1) — discards actions and does **not** inject them.

``VJoyController``
    Phase 2 — injects actions via a `vJoy <https://github.com/jshafer817/vJoy>`_
    virtual joystick device using the ``pyvjoy`` Python binding.

Usage::

    # Telemetry-only (default):
    ctrl = NullController()

    # Live action injection:
    ctrl = VJoyController(device_id=1)

    ctrl.send(steer=-0.3, throttle=0.8, brake=0.0)
    ctrl.reset()
    ctrl.close()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


# ── vJoy axis constants ──────────────────────────────────────────────
# vJoy axes range from 1 (0x1) to 32768 (0x8000).
_VJOY_AXIS_MIN: int = 1
_VJOY_AXIS_MAX: int = 32768
_VJOY_AXIS_MID: int = (_VJOY_AXIS_MIN + _VJOY_AXIS_MAX) // 2


class BaseController(ABC):
    """Abstract controller interface for iRacing action injection."""

    @abstractmethod
    def send(self, steer: float, throttle: float, brake: float) -> None:
        """Send a single control frame to the game.

        Parameters
        ----------
        steer : float
            Steering input in [-1, 1] (negative = left, positive = right).
        throttle : float
            Throttle input in [0, 1].
        brake : float
            Brake input in [0, 1].
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset all axes to neutral (steer centred, throttle/brake zero)."""

    @abstractmethod
    def close(self) -> None:
        """Release any held resources (e.g. vJoy device handle)."""


class NullController(BaseController):
    """No-op controller for telemetry-only mode (Phase 1).

    Actions are silently discarded.  This is the default when
    ``action_mode`` is ``"telemetry_only"``.
    """

    def send(self, steer: float, throttle: float, brake: float) -> None:
        pass

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class VJoyController(BaseController):
    """Injects steer/throttle/brake into iRacing via a vJoy virtual joystick.

    Requires:
    - vJoy driver installed (https://github.com/jshafer817/vJoy)
    - ``pyvjoy`` Python package (``pip install pyvjoy``)
    - iRacing configured to accept the vJoy device as a controller

    Parameters
    ----------
    device_id : int
        vJoy device number (1-based).  Default ``1``.
    steer_axis : int
        HID axis used for steering.  Default ``0x30`` (wAxis / X axis).
    throttle_axis : int
        HID axis used for throttle.  Default ``0x31`` (wAxisY / Y axis).
    brake_axis : int
        HID axis used for brake.  Default ``0x32`` (wAxisZ / Z axis).
    """

    # HID usage IDs for the three default vJoy axes.
    HID_USAGE_X: int = 0x30
    HID_USAGE_Y: int = 0x31
    HID_USAGE_Z: int = 0x32

    def __init__(
        self,
        device_id: int = 1,
        steer_axis: int | None = None,
        throttle_axis: int | None = None,
        brake_axis: int | None = None,
    ) -> None:
        try:
            import pyvjoy  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pyvjoy is not installed.  Install it with:\n"
                "    pip install pyvjoy\n"
                "vJoy driver must also be installed: "
                "https://github.com/jshafer817/vJoy"
            ) from exc

        self._device_id = device_id
        self._steer_axis = steer_axis if steer_axis is not None else self.HID_USAGE_X
        self._throttle_axis = (
            throttle_axis if throttle_axis is not None else self.HID_USAGE_Y
        )
        self._brake_axis = brake_axis if brake_axis is not None else self.HID_USAGE_Z
        self._pyvjoy = pyvjoy

        self._joy = pyvjoy.VJoyDevice(device_id)
        logger.info("VJoyController: acquired vJoy device %d", device_id)

    def send(self, steer: float, throttle: float, brake: float) -> None:
        steer_val = _float_to_vjoy_axis(float(np.clip(steer, -1.0, 1.0)))
        throttle_val = _float_to_vjoy_axis_unipolar(
            float(np.clip(throttle, 0.0, 1.0))
        )
        brake_val = _float_to_vjoy_axis_unipolar(
            float(np.clip(brake, 0.0, 1.0))
        )

        self._joy.set_axis(self._steer_axis, steer_val)
        self._joy.set_axis(self._throttle_axis, throttle_val)
        self._joy.set_axis(self._brake_axis, brake_val)

    def reset(self) -> None:
        self._joy.set_axis(self._steer_axis, _VJOY_AXIS_MID)
        self._joy.set_axis(self._throttle_axis, _VJOY_AXIS_MIN)
        self._joy.set_axis(self._brake_axis, _VJOY_AXIS_MIN)

    def close(self) -> None:
        try:
            self.reset()
        except Exception:
            logger.warning(
                "VJoyController: failed to reset axes on close", exc_info=True
            )
        logger.info("VJoyController: released vJoy device %d", self._device_id)


# ── Axis conversion helpers ──────────────────────────────────────────


def _float_to_vjoy_axis(value: float) -> int:
    """Map a bipolar float in [-1, 1] to vJoy axis range [1, 32768].

    -1.0 → 1  (full left),  0.0 → ~16384 (centre),  1.0 → 32768 (full right).
    """
    clamped = float(np.clip(value, -1.0, 1.0))
    normalised = (clamped + 1.0) / 2.0  # [0, 1]
    return int(round(_VJOY_AXIS_MIN + normalised * (_VJOY_AXIS_MAX - _VJOY_AXIS_MIN)))


def _float_to_vjoy_axis_unipolar(value: float) -> int:
    """Map a unipolar float in [0, 1] to vJoy axis range [1, 32768].

    0.0 → 1 (minimum),  1.0 → 32768 (maximum).
    """
    clamped = float(np.clip(value, 0.0, 1.0))
    return int(round(_VJOY_AXIS_MIN + clamped * (_VJOY_AXIS_MAX - _VJOY_AXIS_MIN)))


def make_controller(action_mode: str, **kwargs) -> BaseController:
    """Factory that creates the appropriate controller for the given mode.

    Parameters
    ----------
    action_mode : str
        ``"telemetry_only"`` for Phase 1 (no injection), or
        ``"live"`` for Phase 2 (vJoy injection).
    **kwargs
        Forwarded to the controller constructor (e.g. ``device_id``).

    Returns
    -------
    BaseController
    """
    if action_mode == "telemetry_only":
        return NullController()
    if action_mode == "live":
        return VJoyController(**kwargs)
    raise ValueError(
        f"Unknown action_mode {action_mode!r}.  "
        f"Use 'telemetry_only' or 'live'."
    )
