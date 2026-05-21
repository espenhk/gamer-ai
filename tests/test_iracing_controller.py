"""Tests for the iRacing controller module.

Validates:
  - NullController is a safe no-op (telemetry-only mode)
  - VJoyController axis-conversion math
  - make_controller factory routing
  - Axis value boundary conditions
"""

from __future__ import annotations

import builtins
import pytest
import sys

from games.iracing.controller import (
    BaseController,
    NullController,
    VJoyController,
    _float_to_vjoy_axis,
    _float_to_vjoy_axis_unipolar,
    _VJOY_AXIS_MIN,
    _VJOY_AXIS_MAX,
    _VJOY_AXIS_MID,
    make_controller,
)


# ── NullController ───────────────────────────────────────────────────


class TestNullController:
    """NullController must be a safe no-op."""

    def test_implements_base(self):
        ctrl = NullController()
        assert isinstance(ctrl, BaseController)

    def test_send_does_not_raise(self):
        ctrl = NullController()
        ctrl.send(steer=-1.0, throttle=0.5, brake=0.3)
        ctrl.send(steer=0.0, throttle=0.0, brake=0.0)

    def test_reset_does_not_raise(self):
        NullController().reset()

    def test_close_does_not_raise(self):
        NullController().close()


# ── Axis conversion helpers ──────────────────────────────────────────


class TestFloatToVJoyAxis:
    """Bipolar [-1, 1] → [1, 32768] mapping."""

    def test_negative_one(self):
        assert _float_to_vjoy_axis(-1.0) == _VJOY_AXIS_MIN

    def test_zero(self):
        assert _float_to_vjoy_axis(0.0) == _VJOY_AXIS_MID

    def test_positive_one(self):
        assert _float_to_vjoy_axis(1.0) == _VJOY_AXIS_MAX

    def test_clamps_below(self):
        assert _float_to_vjoy_axis(-5.0) == _VJOY_AXIS_MIN

    def test_clamps_above(self):
        assert _float_to_vjoy_axis(5.0) == _VJOY_AXIS_MAX

    def test_half_right(self):
        val = _float_to_vjoy_axis(0.5)
        assert _VJOY_AXIS_MID < val < _VJOY_AXIS_MAX

    def test_half_left(self):
        val = _float_to_vjoy_axis(-0.5)
        assert _VJOY_AXIS_MIN < val < _VJOY_AXIS_MID


class TestFloatToVJoyAxisUnipolar:
    """Unipolar [0, 1] → [1, 32768] mapping."""

    def test_zero(self):
        assert _float_to_vjoy_axis_unipolar(0.0) == _VJOY_AXIS_MIN

    def test_one(self):
        assert _float_to_vjoy_axis_unipolar(1.0) == _VJOY_AXIS_MAX

    def test_clamps_below(self):
        assert _float_to_vjoy_axis_unipolar(-1.0) == _VJOY_AXIS_MIN

    def test_clamps_above(self):
        assert _float_to_vjoy_axis_unipolar(2.0) == _VJOY_AXIS_MAX

    def test_half(self):
        val = _float_to_vjoy_axis_unipolar(0.5)
        assert _VJOY_AXIS_MIN < val < _VJOY_AXIS_MAX


# ── make_controller factory ──────────────────────────────────────────


class TestMakeController:
    def test_telemetry_only_returns_null(self):
        ctrl = make_controller("telemetry_only")
        assert isinstance(ctrl, NullController)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown action_mode"):
            make_controller("bogus")

    def test_live_mode_without_pyvjoy_raises_import_error(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pyvjoy":
                raise ImportError("No module named 'pyvjoy'")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "pyvjoy", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pyvjoy"):
            make_controller("live")


# ── VJoyController (mock-based) ──────────────────────────────────────


class _FakeVJoyDevice:
    """Lightweight stand-in for pyvjoy.VJoyDevice."""

    def __init__(self, device_id: int) -> None:
        self.device_id = device_id
        self.axes: dict[int, int] = {}

    def set_axis(self, axis_id: int, value: int) -> None:
        self.axes[axis_id] = value


class TestVJoyControllerWithMock:
    """Test VJoyController logic using a mocked pyvjoy module."""

    @pytest.fixture()
    def ctrl(self, monkeypatch):
        """Create a VJoyController with a fake pyvjoy module."""
        import types

        fake_pyvjoy = types.ModuleType("pyvjoy")
        fake_pyvjoy.VJoyDevice = _FakeVJoyDevice  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pyvjoy", fake_pyvjoy)
        return VJoyController(device_id=1)

    def test_send_sets_all_axes(self, ctrl):
        ctrl.send(steer=0.0, throttle=1.0, brake=0.0)
        assert ctrl._joy.axes[VJoyController.HID_USAGE_X] == _VJOY_AXIS_MID
        assert ctrl._joy.axes[VJoyController.HID_USAGE_Y] == _VJOY_AXIS_MAX
        assert ctrl._joy.axes[VJoyController.HID_USAGE_Z] == _VJOY_AXIS_MIN

    def test_send_full_left_full_brake(self, ctrl):
        ctrl.send(steer=-1.0, throttle=0.0, brake=1.0)
        assert ctrl._joy.axes[VJoyController.HID_USAGE_X] == _VJOY_AXIS_MIN
        assert ctrl._joy.axes[VJoyController.HID_USAGE_Y] == _VJOY_AXIS_MIN
        assert ctrl._joy.axes[VJoyController.HID_USAGE_Z] == _VJOY_AXIS_MAX

    def test_send_clamps_out_of_range(self, ctrl):
        ctrl.send(steer=5.0, throttle=-1.0, brake=3.0)
        assert ctrl._joy.axes[VJoyController.HID_USAGE_X] == _VJOY_AXIS_MAX
        assert ctrl._joy.axes[VJoyController.HID_USAGE_Y] == _VJOY_AXIS_MIN
        assert ctrl._joy.axes[VJoyController.HID_USAGE_Z] == _VJOY_AXIS_MAX

    def test_reset_centres_steer_and_zeros_pedals(self, ctrl):
        ctrl.send(steer=0.7, throttle=0.9, brake=0.3)
        ctrl.reset()
        assert ctrl._joy.axes[VJoyController.HID_USAGE_X] == _VJOY_AXIS_MID
        assert ctrl._joy.axes[VJoyController.HID_USAGE_Y] == _VJOY_AXIS_MIN
        assert ctrl._joy.axes[VJoyController.HID_USAGE_Z] == _VJOY_AXIS_MIN

    def test_close_resets_axes(self, ctrl):
        ctrl.send(steer=1.0, throttle=1.0, brake=1.0)
        ctrl.close()
        assert ctrl._joy.axes[VJoyController.HID_USAGE_X] == _VJOY_AXIS_MID
        assert ctrl._joy.axes[VJoyController.HID_USAGE_Y] == _VJOY_AXIS_MIN
        assert ctrl._joy.axes[VJoyController.HID_USAGE_Z] == _VJOY_AXIS_MIN
