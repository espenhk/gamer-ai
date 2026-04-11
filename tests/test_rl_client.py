"""Tests for RLClient.on_run_step() action application.

tminterface is a Windows-only, non-PyPI library so we stub it out at the
sys.modules level before any client code is imported.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# ---------------------------------------------------------------------------
# Stub out tminterface so the module can be imported without the real library.
# ---------------------------------------------------------------------------
def _make_stub(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

_tm      = _make_stub("tminterface")
_client  = _make_stub("tminterface.client")
_iface   = _make_stub("tminterface.interface")
_structs = _make_stub("tminterface.structs")

class _FakeClient:
    def __init__(self): pass

_client.Client     = _FakeClient
_iface.TMInterface = MagicMock

# ---------------------------------------------------------------------------
# Now we can safely import the module under test.
# ---------------------------------------------------------------------------
from clients.rl_client import RLClient, StepState, _DEFAULT_ACTION, ACTIONS  # noqa: E402


def _make_state_data(track_progress=0.5, lateral_offset=0.0, speed=10.0):
    """Build a StateData-like mock for injection into on_run_step."""
    from helpers import make_state_data
    return make_state_data(
        track_progress=track_progress,
        lateral_offset=lateral_offset,
        speed=(speed, 0.0, 0.0),
    )


def _make_client():
    """Instantiate RLClient with a mocked Centerline (no file I/O)."""
    with patch("clients.rl_client.Centerline", return_value=MagicMock()):
        return RLClient(centerline_file="fake.npy", speed=1.0)


class TestSetInputStateCalled(unittest.TestCase):
    """Verify set_input_state is called on every normal running tick."""

    def setUp(self):
        self.client = _make_client()
        self.client._running = True
        self.client._finish_respawn_pending = False
        self.client._simulation_finish_delivered = False
        self.state_data = _make_state_data()

    def _run_step(self, iface, action, time_ms=1000):
        """Execute one on_run_step with the given action, mocking away StateData and yaw."""
        self.client.set_action(action)
        with patch("clients.rl_client.StateData", return_value=self.state_data), \
             patch.object(self.client, "_compute_yaw_error", return_value=0.0):
            self.client.on_run_step(iface, time_ms)

    def _iface(self):
        iface = MagicMock()
        iface.get_simulation_state.return_value = MagicMock()  # raw game state (not used directly)
        return iface

    def test_set_input_state_called_on_normal_tick(self):
        """set_input_state must be called every tick when _running=True."""
        iface = self._iface()
        self._run_step(iface, np.array([0.5, 1.0, 0.0], dtype=np.float32))
        iface.set_input_state.assert_called_once()

    def test_steer_accel_brake_values_correct(self):
        """set_input_state must receive the correct steer/accel/brake from action."""
        iface = self._iface()
        # steer=0.5 → int(0.5 * 65536)=32768, accel=True (1.0>=0.5), brake=False
        self._run_step(iface, np.array([0.5, 1.0, 0.0], dtype=np.float32))
        iface.set_input_state.assert_called_once_with(
            accelerate=True,
            brake=False,
            steer=32768,
        )

    def test_brake_action(self):
        """brake=1.0, accel=0.0 → brake=True, accelerate=False."""
        iface = self._iface()
        self._run_step(iface, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        iface.set_input_state.assert_called_once_with(
            accelerate=False,
            brake=True,
            steer=0,
        )

    def test_full_left_steer(self):
        """steer=-1.0 should map to -65536."""
        iface = self._iface()
        self._run_step(iface, np.array([-1.0, 0.0, 0.0], dtype=np.float32))
        iface.set_input_state.assert_called_once_with(
            accelerate=False,
            brake=False,
            steer=-65536,
        )

    def test_steer_clamped_beyond_range(self):
        """Steer values outside [-1, 1] are clipped before mapping."""
        iface = self._iface()
        self._run_step(iface, np.array([-2.0, 0.0, 0.0], dtype=np.float32))
        iface.set_input_state.assert_called_once_with(
            accelerate=False,
            brake=False,
            steer=-65536,
        )

    def test_set_input_state_not_called_when_finish_respawn_pending(self):
        """When _finish_respawn_pending=True, give_up fires and set_input_state does not."""
        self.client._finish_respawn_pending = True
        iface = self._iface()
        self._run_step(iface, _DEFAULT_ACTION.copy())
        iface.give_up.assert_called_once()
        iface.set_input_state.assert_not_called()

    def test_finish_respawn_resets_running_flag(self):
        """After the respawn path fires, _running must become False."""
        self.client._finish_respawn_pending = True
        iface = self._iface()
        self._run_step(iface, _DEFAULT_ACTION.copy())
        self.assertFalse(self.client._running)


class TestDefaultAction(unittest.TestCase):
    def test_shape_and_dtype(self):
        self.assertEqual(_DEFAULT_ACTION.shape, (3,))
        self.assertEqual(_DEFAULT_ACTION.dtype, np.float32)

    def test_coast_straight(self):
        """Default action should be coast straight: [steer=0, accel=0, brake=0]."""
        np.testing.assert_array_equal(_DEFAULT_ACTION, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
