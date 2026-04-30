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


def _make_client():
    """Instantiate RLClient with a mocked Centerline (no file I/O)."""
    with patch("games.tmnf.clients.rl_client.Centerline", return_value=MagicMock()):
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


def _make_windowed_client(action_window_ticks, decision_offset_pct=0.75):
    """Instantiate RLClient with windowing parameters and a mocked Centerline."""
    with patch("games.tmnf.clients.rl_client.Centerline", return_value=MagicMock()):
        return RLClient(
            centerline_file="fake.npy",
            speed=1.0,
            action_window_ticks=action_window_ticks,
            decision_offset_pct=decision_offset_pct,
        )


class TestActionWindow(unittest.TestCase):
    """Verify action-windowing behavior for action_window_ticks > 1."""

    def setUp(self):
        # window=4, decision_offset=0.75 → decision_idx=3.
        # Observation phase: ticks 0, 1, 2.
        # Transit phase: tick 3.
        self.client = _make_windowed_client(action_window_ticks=4, decision_offset_pct=0.75)
        self.client._running = True
        self.client._finish_respawn_pending = False
        self.client._simulation_finish_delivered = False
        # Start mid-window so we don't auto-commit on tick 0.
        self.client._force_commit_next_tick = False
        self.state_data = _make_state_data()

    def _iface(self):
        iface = MagicMock()
        iface.get_simulation_state.return_value = MagicMock()
        return iface

    def _run_step(self, iface, action=None, time_ms=1000, state_data=None):
        if action is not None:
            self.client.set_action(action)
        sd = state_data if state_data is not None else self.state_data
        with patch("games.tmnf.clients.rl_client.StateData", return_value=sd), \
             patch.object(self.client, "_compute_yaw_error", return_value=0.0):
            self.client.on_run_step(iface, time_ms)

    def test_decision_idx_computed(self):
        """decision_offset_pct=0.75 with window=4 → decision_idx=3."""
        self.assertEqual(self.client._decision_idx, 3)

    def test_first_tick_commits_pending_action(self):
        """Tick 0 of a window calls set_input_state once with the pending action."""
        iface = self._iface()
        # window_tick=0 at start of test → tick 0 of a window.
        self._run_step(iface, action=np.array([0.5, 1.0, 0.0], dtype=np.float32))
        iface.set_input_state.assert_called_once_with(
            accelerate=True, brake=False, steer=32768
        )

    def test_later_window_ticks_do_not_commit(self):
        """Ticks 1, 2, 3 of a window do NOT call set_input_state.

        Mid-window changes to the pending action are not applied to the game
        until the next window-start.
        """
        iface = self._iface()
        # Tick 0: commit with first action.
        self._run_step(iface, action=np.array([0.5, 1.0, 0.0], dtype=np.float32))
        iface.set_input_state.reset_mock()
        # Ticks 1, 2, 3: change pending action but no commit should fire.
        for _ in range(3):
            self._run_step(iface, action=np.array([-1.0, 0.0, 1.0], dtype=np.float32))
        iface.set_input_state.assert_not_called()

    def test_observation_phase_emits_states_transit_does_not(self):
        """Ticks 0, 1, 2 emit StepStates; tick 3 (transit) does not."""
        iface = self._iface()
        # Drain queue first.
        while not self.client._state_queue.empty():
            self.client._state_queue.get_nowait()
        # Ticks 0, 1, 2 — should each enqueue a StepState.
        for _ in range(3):
            self._run_step(iface)
            self.assertEqual(self.client._state_queue.qsize(), 1)
            self.client._state_queue.get_nowait()  # drain
        # Tick 3 — transit phase, no enqueue.
        self._run_step(iface)
        self.assertTrue(self.client._state_queue.empty())

    def test_transit_phase_ticks_accumulate_via_drain_and_put(self):
        """The next window's tick 0 StepState absorbs suppressed transit ticks."""
        iface = self._iface()
        # Run a full window: ticks 0, 1, 2 emit; tick 3 (transit) suppresses.
        for _ in range(4):
            self._run_step(iface)
        # Drain queue (whatever's left from observation phase).
        while not self.client._state_queue.empty():
            self.client._state_queue.get_nowait()
        # Next tick is tick 0 of a new window — should emit and absorb the
        # transit tick from the previous window. Exact count depends on
        # _drain_and_put state; this just verifies emission.
        self._run_step(iface)
        self.assertEqual(self.client._state_queue.qsize(), 1)

    def test_pending_action_locked_after_decision_tick(self):
        """Action set during transit phase does NOT reach the game until next window-start."""
        iface = self._iface()
        # Tick 0: commit action A.
        self._run_step(iface, action=np.array([0.5, 1.0, 0.0], dtype=np.float32))
        # Ticks 1, 2, 3: set new action B every tick. None should commit.
        for _ in range(3):
            self._run_step(iface, action=np.array([-1.0, 0.0, 1.0], dtype=np.float32))
        # Only the first tick's commit happened.
        self.assertEqual(iface.set_input_state.call_count, 1)
        # Tick 4 (next window's tick 0): commits action B.
        iface.set_input_state.reset_mock()
        self._run_step(iface)
        iface.set_input_state.assert_called_once_with(
            accelerate=False, brake=True, steer=-65536
        )

    def test_finish_forces_immediate_commit_and_emit(self):
        """A finish detected during transit phase triggers commit + StepState."""
        iface = self._iface()
        # Run through ticks 0, 1, 2 (observation) — drain.
        for _ in range(3):
            self._run_step(iface)
        while not self.client._state_queue.empty():
            self.client._state_queue.get_nowait()
        iface.set_input_state.reset_mock()
        # Tick 3: transit, but inject finish.
        finished_state = _make_state_data(track_progress=0.97)
        self._run_step(iface, action=np.array([1.0, 1.0, 0.0], dtype=np.float32),
                       state_data=finished_state)
        iface.set_input_state.assert_called_once()
        self.assertEqual(self.client._state_queue.qsize(), 1)
        emitted = self.client._state_queue.get_nowait()
        self.assertTrue(emitted.finished)

    def test_hard_crash_forces_commit(self):
        """A hard-crash detected during transit phase triggers commit + StepState."""
        from games.tmnf.clients.rl_client import _HARD_CRASH_THRESHOLD_M
        iface = self._iface()
        for _ in range(3):
            self._run_step(iface)
        while not self.client._state_queue.empty():
            self.client._state_queue.get_nowait()
        iface.set_input_state.reset_mock()
        crash_state = _make_state_data(lateral_offset=_HARD_CRASH_THRESHOLD_M + 1.0)
        self._run_step(iface, state_data=crash_state)
        iface.set_input_state.assert_called_once()
        self.assertEqual(self.client._state_queue.qsize(), 1)
        emitted = self.client._state_queue.get_nowait()
        self.assertTrue(emitted.done)

    def test_decision_idx_clamped_to_at_least_one(self):
        """decision_offset_pct very small still yields decision_idx >= 1."""
        client = _make_windowed_client(action_window_ticks=2, decision_offset_pct=0.01)
        self.assertEqual(client._decision_idx, 1)

    def test_decision_idx_clamped_to_at_most_window_minus_one(self):
        """decision_offset_pct=1.0 leaves at least one transit tick."""
        client = _make_windowed_client(action_window_ticks=4, decision_offset_pct=1.0)
        self.assertEqual(client._decision_idx, 3)

    def test_legacy_window_one_commits_every_tick(self):
        """action_window_ticks=1 (default): every tick commits and emits."""
        client = _make_windowed_client(action_window_ticks=1)
        client._running = True
        client._finish_respawn_pending = False
        client._simulation_finish_delivered = False
        self.assertEqual(client._decision_idx, 0)
        iface = MagicMock()
        iface.get_simulation_state.return_value = MagicMock()
        for i in range(5):
            with patch("games.tmnf.clients.rl_client.StateData", return_value=_make_state_data()), \
                 patch.object(client, "_compute_yaw_error", return_value=0.0):
                client.set_action(np.array([0.0, 1.0, 0.0], dtype=np.float32))
                client.on_run_step(iface, i)
        self.assertEqual(iface.set_input_state.call_count, 5)


class TestRLClientConstructorValidation(unittest.TestCase):
    def test_action_window_ticks_must_be_positive(self):
        with patch("games.tmnf.clients.rl_client.Centerline", return_value=MagicMock()):
            with self.assertRaises(AssertionError):
                RLClient(centerline_file="fake.npy", speed=1.0, action_window_ticks=0)

    def test_decision_offset_pct_bounds(self):
        with patch("games.tmnf.clients.rl_client.Centerline", return_value=MagicMock()):
            with self.assertRaises(AssertionError):
                RLClient(centerline_file="fake.npy", speed=1.0,
                         action_window_ticks=4, decision_offset_pct=0.0)
            with self.assertRaises(AssertionError):
                RLClient(centerline_file="fake.npy", speed=1.0,
                         action_window_ticks=4, decision_offset_pct=1.5)


if __name__ == "__main__":
    unittest.main()
