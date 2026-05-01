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
from games.tmnf.clients.rl_client import RLClient, StepState, _DEFAULT_ACTION, ACTIONS  # noqa: E402


def _make_state_data(track_progress=0.5, lateral_offset=0.0, speed=10.0):
    """Build a StateData-like mock for injection into on_run_step."""
    from helpers import make_state_data
    return make_state_data(
        track_progress=track_progress,
        lateral_offset=lateral_offset,
        speed=(speed, 0.0, 0.0),
    )


def _make_client(action_window_ticks=1):
    """Instantiate RLClient with a mocked Centerline (no file I/O)."""
    with patch("games.tmnf.clients.rl_client.Centerline", return_value=MagicMock()):
        return RLClient(centerline_file="fake.npy", speed=1.0,
                        action_window_ticks=action_window_ticks)

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
        with patch("games.tmnf.clients.rl_client.StateData", return_value=self.state_data), \
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


class TestActionWindow(unittest.TestCase):
    """Verify action windowing emits states only every N ticks."""

    def setUp(self):
        self.client = _make_client(action_window_ticks=5)
        self.client._running = True
        self.client._finish_respawn_pending = False
        self.state_data = _make_state_data()

    def _iface(self):
        iface = MagicMock()
        iface.get_simulation_state.return_value = MagicMock()
        return iface

    def _run_step(self, iface, time_ms=1000):
        """Execute one on_run_step with the default action."""
        self.client.set_action(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        with patch("games.tmnf.clients.rl_client.StateData", return_value=self.state_data), \
             patch.object(self.client, "_compute_yaw_error", return_value=0.0):
            self.client.on_run_step(iface, time_ms)

    def test_no_state_emitted_before_window_complete(self):
        """State queue should stay empty for first N-1 ticks."""
        iface = self._iface()
        for i in range(4):
            self._run_step(iface, time_ms=i * 10)
        self.assertTrue(self.client._state_queue.empty())

    def test_state_emitted_on_window_complete(self):
        """State should be emitted on the Nth tick (window complete)."""
        iface = self._iface()
        for i in range(5):
            self._run_step(iface, time_ms=i * 10)
        self.assertFalse(self.client._state_queue.empty())

    def test_ticks_this_step_reflects_window_size(self):
        """ticks_this_step in emitted state should equal action_window_ticks."""
        iface = self._iface()
        for i in range(5):
            self._run_step(iface, time_ms=i * 10)
        step_state = self.client._state_queue.get_nowait()
        self.assertEqual(step_state.ticks_this_step, 5)

    def test_action_applied_every_tick(self):
        """set_input_state must be called every tick even within a window."""
        iface = self._iface()
        for i in range(5):
            self._run_step(iface, time_ms=i * 10)
        self.assertEqual(iface.set_input_state.call_count, 5)

    def test_done_emits_immediately(self):
        """When a hard crash is detected mid-window, state should emit immediately."""
        client = _make_client(action_window_ticks=10)
        client._running = True
        client._finish_respawn_pending = False
        # Simulate a hard crash state (lateral offset > 50m)
        crash_state = _make_state_data(lateral_offset=60.0)

        iface = self._iface()
        client.set_action(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        with patch("games.tmnf.clients.rl_client.StateData", return_value=crash_state), \
             patch.object(client, "_compute_yaw_error", return_value=0.0):
            client.on_run_step(iface, 0)
        # Should emit immediately even though window_tick=1 < 10
        self.assertFalse(client._state_queue.empty())
        step_state = client._state_queue.get_nowait()
        self.assertTrue(step_state.done)
        self.assertEqual(step_state.ticks_this_step, 1)

    def test_window_default_is_one(self):
        """Default action_window_ticks=1 means state emitted every tick."""
        client = _make_client(action_window_ticks=1)
        client._running = True
        client._finish_respawn_pending = False
        state_data = _make_state_data()

        iface = self._iface()
        client.set_action(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        with patch("games.tmnf.clients.rl_client.StateData", return_value=state_data), \
             patch.object(client, "_compute_yaw_error", return_value=0.0):
            client.on_run_step(iface, 0)
        self.assertFalse(client._state_queue.empty())


if __name__ == "__main__":
    unittest.main()
