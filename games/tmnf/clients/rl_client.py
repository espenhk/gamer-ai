"""
RLClient — TMInterface client designed for reinforcement learning.

Threading model
---------------
TMInterface fires on_run_step() on its own internal thread.
The RL training loop runs on the main (or another) thread.
They communicate through:

  _action          : the RL thread writes, the game thread reads (lock-protected)
                     shape (3,) float32 — [steer ∈ [-1,1], accel ∈ {0,1}, brake ∈ {0,1}]
  _state_queue     : the game thread writes one StepState per tick;
                     the RL thread blocks on get() to receive it.
  _respawn_event   : RL thread sets → game thread calls iface.respawn()
  _episode_ready   : game thread sets → RL thread unblocks from reset()

Because the game runs at high speed (e.g. 10×) and the policy evaluation
takes some time, multiple game ticks may pass before the RL thread reads a
state. _state_queue has maxsize=1 with a drain-before-put strategy so the
RL thread always gets the *latest* state (no stale backlog).
"""

from __future__ import annotations

import math
import queue
import threading
from dataclasses import dataclass

import numpy as np
from tminterface.interface import TMInterface

from games.tmnf.clients.base import PhaseAwareClient
from games.tmnf.constants import N_ACTIONS, STEER_SCALE
from games.tmnf.steering import angle_diff
from games.tmnf.track import Centerline
from games.tmnf.state import StateData

import logging

logger = logging.getLogger(__name__)

VELOCITY_ZERO_THRESHOLD = 0.5  # m/s — wait for car to stop before starting episode

# Default action: coast straight — no steer, no accel, no brake.
_DEFAULT_ACTION = np.array([0.0, 0.0, 0.0], dtype=np.float32)

# Hard-limit lateral offset before the client itself declares the episode done.
# This is a safety net — the env's crash_threshold_m (default 10 m) triggers first.
_HARD_CRASH_THRESHOLD_M = 50.0

# Finish is detected when track_progress reaches this threshold.
# Slightly below 1.0 to catch cases where the game rounds or caps progress
# just before the exact finish line value.
_FINISH_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Discrete action table
# Index → (accelerate, brake, steer_percent, description)
# steer_percent is in [-100, 100]; converted to [-STEER_SCALE, STEER_SCALE] when applied.
# ---------------------------------------------------------------------------
ACTIONS: list[tuple[bool, bool, int, str]] = [
    (False,  True,  -100, "brake LEFT"),         # 0: brake  + full left
    (False,  True,     0, "brake"),              # 1: brake  + straight
    (False,  True,   100, "brake RIGHT"),        # 2: brake  + full right
    (False, False,  -100, "coast LEFT"),         # 3: coast  + full left
    (False, False,     0, "coast"),              # 4: coast  + straight
    (False, False,   100, "coast RIGHT"),        # 5: coast  + full right
    (True,  False,  -100, "accelerate LEFT"),    # 6: accel  + full left
    (True,  False,     0, "accelerate"),         # 7: accel  + straight   ← default
    (True,  False,   100, "accelerate right"),   # 8: accel  + full right
]
assert len(ACTIONS) == N_ACTIONS, (
    f"ACTIONS has {len(ACTIONS)} entries but N_ACTIONS={N_ACTIONS}; "
    "update games/tmnf/constants.py to match."
)


def get_action_description(idx: int) -> str:
    return ACTIONS[idx][3]

@dataclass
class StepState:
    """Everything the env needs to compute observations and rewards."""

    state_data: StateData
    yaw_error: float  # signed radians: track heading minus car heading, in [-π, π]
    done: bool  # True if game client detected a hard termination condition
    finished: bool = False  # True when car crossed the finish line
    ticks_this_step: int = (
        1  # game ticks covered by this RL step (≥1; >1 when events were skipped)
    )


class RLClient(PhaseAwareClient):
    """TMInterface client used by TMNFEnv during RL training."""

    def __init__(
        self,
        centerline_file: str,
        speed: float = 10.0,
        auto_respawn_on_finish: bool = False,
        action_window_ticks: int = 1,
        decision_offset_pct: float = 0.75,
    ) -> None:
        super().__init__()
        self.speed = speed
        self.centerline = Centerline(centerline_file)
        self._auto_respawn_on_finish = auto_respawn_on_finish

        # --- Action windowing (issue #65) ---------------------------------
        # The window has two phases:
        #   * Observation phase (tick 0 .. _decision_idx-1): game thread emits
        #     a StepState every tick; RL thread iteratively refines the pending
        #     action.
        #   * Transit phase (tick _decision_idx .. window-1): no StepStates,
        #     pending action is locked.
        # At every window boundary the pending action is committed to the game
        # via iface.set_input_state(...).
        # When action_window_ticks == 1, _decision_idx == 0 and every tick
        # commits and emits — preserving legacy behavior bit-for-bit.
        assert action_window_ticks >= 1, "action_window_ticks must be >= 1"
        assert 0.0 < decision_offset_pct <= 1.0, "decision_offset_pct must be in (0, 1]"
        self._action_window_ticks: int = action_window_ticks
        if action_window_ticks > 1:
            # Clamp to [1, window-1] so there's at least one observation tick
            # and at least one transit tick.
            self._decision_idx: int = max(
                1,
                min(action_window_ticks - 1,
                    int(action_window_ticks * decision_offset_pct)),
            )
        else:
            self._decision_idx = 0
        self._window_tick: int = 0
        self._force_commit_next_tick: bool = True

        # Shared state — written by RL thread, read by game thread.
        # _pending_action is what the RL thread is iteratively refining;
        # _committed_action is what the game is currently being driven with.
        # shape (3,): [steer ∈ [-1,1], accel ∈ {0,1}, brake ∈ {0,1}]
        self._pending_action: np.ndarray = _DEFAULT_ACTION.copy()
        self._committed_action: np.ndarray = _DEFAULT_ACTION.copy()
        self._action_lock = threading.Lock()

        # Shared state — written by game thread, read by RL thread
        self._state_queue: queue.Queue[StepState] = queue.Queue(maxsize=1)
        self._respawn_event = threading.Event()
        self._episode_ready = threading.Event()

        self._running = False  # False = braking to a stop, True = episode running
        self._registered_event = threading.Event()
        self._stop_event = threading.Event()

        # Set by game thread when a lap completes (auto-respawn path only).
        # The game thread acts on it the following tick, so the finish step
        # is delivered to the RL thread before the respawn is triggered.
        self._finish_respawn_pending: bool = False
        self._last_step_state: StepState | None = None

        # Guards against delivering multiple finish steps during replay
        # validation (on_simulation_step fires every tick during that phase).
        self._simulation_finish_delivered: bool = False

        # Debug: tick counter for periodic logging
        self._tick: int = 0
        # Cached nearest centerline index from the previous tick.  Passed as
        # hint_idx to project_with_forward() so it only searches a local window
        # (O(window)) instead of the full centerline (O(N)) every tick.
        # Reset to None on respawn so the first tick does a full scan.
        self._last_centerline_idx: int | None = None

    # ------------------------------------------------------------------
    # RL-thread API
    # ------------------------------------------------------------------

    def wait_registered(self, timeout: float = 15.0) -> bool:
        """Block until on_registered fires (or timeout). Returns True on success."""
        return self._registered_event.wait(timeout=timeout)

    def stop(self) -> None:
        """Unblock any waiting RL-thread calls so the process can exit cleanly."""
        self._stop_event.set()
        self._episode_ready.set()  # unblock wait_episode_ready

    def set_action(self, action: np.ndarray) -> None:
        """Set the next pending action. Thread-safe.

        Stored in _pending_action; copied to _committed_action and applied to
        the game at the next window boundary (or immediately when
        action_window_ticks == 1).

        action: shape (3,) float32 — [steer ∈ [-1,1], accel ∈ {0,1}, brake ∈ {0,1}]
        accel and brake are thresholded at 0.5 when applied to the game.
        """
        with self._action_lock:
            self._pending_action = action

    def get_step_state(self) -> StepState:
        """Block until the game thread delivers the next state."""
        while not self._stop_event.is_set():
            try:
                return self._state_queue.get(timeout=1.0)
            except queue.Empty:
                continue
        raise RuntimeError("RLClient stopped while waiting for step state")

    def request_respawn(self) -> None:
        """Signal the game thread to respawn the car. Call before wait_episode_ready()."""
        logger.debug(
            "[RLClient] request_respawn: setting _respawn_event "
            "(running=%s finish_pending=%s finish_delivered=%s queue_size=%d)",
            self._running, self._finish_respawn_pending,
            self._simulation_finish_delivered, self._state_queue.qsize(),
        )
        self._episode_ready.clear()
        self._simulation_finish_delivered = False
        self._respawn_event.set()

    def wait_episode_ready(self) -> StepState:
        """
        Block until the car has stopped after respawn (BRAKING_START → RUNNING).
        Returns the first state of the new episode.
        """
        logger.debug(
            "[RLClient] wait_episode_ready: blocking "
            "(running=%s finish_pending=%s finish_delivered=%s queue_size=%d)",
            self._running, self._finish_respawn_pending,
            self._simulation_finish_delivered, self._state_queue.qsize(),
        )
        wait_count = 0
        while not self._stop_event.is_set():
            if self._episode_ready.wait(timeout=1.0):
                break
            wait_count += 1
            logger.warning(
                "[RLClient] wait_episode_ready: still waiting after %ds "
                "(running=%s finish_pending=%s finish_delivered=%s respawn_set=%s queue_size=%d)",
                wait_count, self._running, self._finish_respawn_pending,
                self._simulation_finish_delivered, self._respawn_event.is_set(),
                self._state_queue.qsize(),
            )
        if self._stop_event.is_set():
            raise RuntimeError("RLClient stopped while waiting for episode ready")
        logger.debug("[RLClient] wait_episode_ready: _episode_ready fired — reading first state")
        self._episode_ready.clear()
        return self._state_queue.get(timeout=30.0)

    # ------------------------------------------------------------------
    # TMInterface callbacks — called by the game thread
    # ------------------------------------------------------------------

    def on_registered(self, iface: TMInterface) -> None:
        logger.info("Connected. RLClient running at %sx speed.", self.speed)
        iface.execute_command(f"set speed {self.speed}")
        self._registered_event.set()

    def on_simulation_begin(self, iface: TMInterface) -> None:
        """Fires ONCE when replay validation begins (car crossed finish line)."""
        logger.debug(
            "[RLClient] on_simulation_begin: replay validation started "
            "(running=%s finish_delivered=%s finish_pending=%s respawn_set=%s last_state=%s queue_size=%d)",
            self._running, self._simulation_finish_delivered,
            self._finish_respawn_pending, self._respawn_event.is_set(),
            "yes" if self._last_step_state else "NO",
            self._state_queue.qsize(),
        )

        if not self._running:
            logger.debug("[RLClient] on_simulation_begin: not RUNNING — ignoring")
            return
        if self._simulation_finish_delivered:
            logger.debug("[RLClient] on_simulation_begin: finish already delivered — skipping synthetic")
            return
        if self._last_step_state is None:
            logger.warning("[RLClient] on_simulation_begin: no last_step_state — cannot synthesize finish step!")
            return

        synthetic = StepState(
            state_data=self._last_step_state.state_data,
            yaw_error=self._last_step_state.yaw_error,
            done=False,
            finished=True,
            ticks_this_step=self._last_step_state.ticks_this_step,
        )
        self._drain_and_put(synthetic)
        self._simulation_finish_delivered = True
        logger.debug("[RLClient] on_simulation_begin: delivered synthetic finish step (done=False finished=True)")

        if self._auto_respawn_on_finish:
            self._finish_respawn_pending = True
            logger.debug("[RLClient] on_simulation_begin: auto-respawn pending set")


    def on_simulation_step(self, iface: TMInterface, _time: int) -> None:
        """Fires every tick during replay validation (after race finish).
        on_simulation_begin is the one-shot entry; this fires repeatedly for each tick."""
        if _time == 0:
            # First tick of replay validation — log full state.
            logger.debug(
                "[RLClient] on_simulation_step t=0 (first replay tick): "
                "running=%s finish_delivered=%s finish_pending=%s respawn_set=%s last_state=%s queue_size=%d",
                self._running, self._simulation_finish_delivered,
                self._finish_respawn_pending, self._respawn_event.is_set(),
                "yes" if self._last_step_state else "NO",
                self._state_queue.qsize(),
            )
        elif _time % 10_000 == 0:
            # Log every 10 s of replay time so we can see it's still running.
            logger.debug(
                "[RLClient] on_simulation_step t=%d still in replay validation "
                "(running=%s finish_pending=%s respawn_set=%s queue_size=%d)",
                _time, self._running, self._finish_respawn_pending,
                self._respawn_event.is_set(), self._state_queue.qsize(),
            )

        # on_run_step does not fire during replay validation, so any pending
        # respawn must be handled here instead.

        if self._respawn_event.is_set():
            # RL thread called request_respawn() (non-auto-respawn path).
            # Finish step was already delivered before on_run_step stopped, or by
            # on_simulation_begin above.  Safe to exit replay validation now.
            logger.debug(
                "[RLClient] on_simulation_step t=%d: _respawn_event set in replay validation "
                "→ give_up() here (finish_delivered=%s queue_size=%d)",
                _time, self._simulation_finish_delivered, self._state_queue.qsize(),
            )
            self._respawn_event.clear()
            self._simulation_finish_delivered = False
            self._last_centerline_idx = None
            self._running = False
            iface.give_up()
            return

        if self._finish_respawn_pending:
            # Auto-respawn path: finish step delivered, now exit replay validation.
            # on_run_step will resume after give_up() and handle BRAKING_START → RUNNING.
            logger.debug(
                "[RLClient] on_simulation_step t=%d: _finish_respawn_pending in replay validation "
                "→ give_up() here (finish_delivered=%s queue_size=%d)",
                _time, self._simulation_finish_delivered, self._state_queue.qsize(),
            )
            self._finish_respawn_pending = False
            self._episode_ready.clear()
            self._simulation_finish_delivered = False
            self._last_centerline_idx = None
            self._running = False
            iface.give_up()
            return

        if not self._running:
            return
        if self._simulation_finish_delivered:
            return  # already delivered; wait for respawn
        if self._last_step_state is None:
            logger.warning("[RLClient] on_simulation_step t=%d: no last_step_state — cannot synthesize!", _time)
            return

        synthetic = StepState(
            state_data=self._last_step_state.state_data,
            yaw_error=self._last_step_state.yaw_error,
            done=False,
            finished=True,
            ticks_this_step=self._last_step_state.ticks_this_step,
        )
        self._drain_and_put(synthetic)
        self._simulation_finish_delivered = True
        logger.debug("[RLClient] on_simulation_step t=%d: delivered synthetic finish step", _time)

        if self._auto_respawn_on_finish:
            self._finish_respawn_pending = True
            logger.debug("[RLClient] on_simulation_step t=%d: auto-respawn pending set", _time)

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
        self._tick += 1

        # Handle a pending respawn request from the RL thread.
        if self._respawn_event.is_set():
            self._respawn_event.clear()
            logger.debug(
                "[RLClient] on_run_step t=%d: _respawn_event set → give_up() "
                "(running=%s finish_pending=%s finish_delivered=%s)",
                _time, self._running, self._finish_respawn_pending,
                self._simulation_finish_delivered,
            )
            iface.give_up()
            self._last_centerline_idx = None  # full scan on next tick after respawn
            self._simulation_finish_delivered = False
            self._running = False
            self._window_tick = 0
            self._force_commit_next_tick = True
            return

        state = iface.get_simulation_state()
        data = StateData(
            state, centerline=self.centerline, hint_idx=self._last_centerline_idx
        )
        self._last_centerline_idx = data._centerline_idx
        speed_ms = data.velocity.magnitude()

        if self._tick % 100 == 0:
            logger.debug("[RLClient] tick=%d t=%d running=%s speed=%.2fm/s progress=%s",
                        self._tick, _time, self._running, speed_ms, data.track_progress)

        # Extra logging as car approaches finish.
        if data.track_progress is not None and data.track_progress >= 0.85:
            logger.debug(
                "[RLClient] on_run_step t=%d: approaching finish progress=%.4f "
                "(threshold=%.2f running=%s finish_pending=%s)",
                _time, data.track_progress, _FINISH_THRESHOLD,
                self._running, self._finish_respawn_pending,
            )

        if not self._running:
            iface.set_input_state(brake=True)
            if speed_ms < VELOCITY_ZERO_THRESHOLD:
                logger.debug("[RLClient] on_run_step t=%d: BRAKING_START → RUNNING (episode ready)", _time)
                self._running = True
                self._window_tick = 0
                self._force_commit_next_tick = True
                step_state = StepState(
                    state_data=data,
                    yaw_error=self._compute_yaw_error(data),
                    done=False,
                )
                self._drain_and_put(step_state)
                self._episode_ready.set()
        else:
            # Pending auto-respawn from the previous tick's lap completion:
            # act on it here so the finish step was already delivered first.
            if self._finish_respawn_pending:
                logger.debug(
                    "[RLClient] on_run_step t=%d: _finish_respawn_pending → give_up() "
                    "(finish_delivered=%s queue_size=%d)",
                    _time, self._simulation_finish_delivered, self._state_queue.qsize(),
                )
                self._finish_respawn_pending = False
                self._episode_ready.clear()
                iface.give_up()  # restart race from position zero
                self._simulation_finish_delivered = False
                self._running = False
                self._window_tick = 0
                self._force_commit_next_tick = True
                return

            finished  = data.track_progress is not None and data.track_progress >= _FINISH_THRESHOLD
            hard_crash = (
                data.lateral_offset is not None
                and abs(data.lateral_offset) > _HARD_CRASH_THRESHOLD_M
            )

            if finished:
                logger.info(
                    "[RLClient] on_run_step t=%d: FINISH DETECTED progress=%.4f >= %.2f "
                    "(auto_respawn=%s finish_pending=%s finish_delivered=%s queue_size=%d)",
                    _time, data.track_progress, _FINISH_THRESHOLD,
                    self._auto_respawn_on_finish, self._finish_respawn_pending,
                    self._simulation_finish_delivered, self._state_queue.qsize(),
                )

            # --- Commit boundary: copy pending → committed and apply to game.
            # Fires at the start of every window, on the very first running
            # tick, and whenever termination forces an immediate commit.
            is_window_start = (self._window_tick == 0) or self._force_commit_next_tick
            if is_window_start or finished or hard_crash:
                with self._action_lock:
                    self._committed_action = self._pending_action.copy()
                steer_norm = float(np.clip(self._committed_action[0], -1.0, 1.0))
                accel = bool(float(self._committed_action[1]) >= 0.5)
                brake = bool(float(self._committed_action[2]) >= 0.5)
                iface.set_input_state(
                    accelerate=accel,
                    brake=brake,
                    steer=int(steer_norm * STEER_SCALE),
                )
                self._force_commit_next_tick = False

            if finished and self._auto_respawn_on_finish:
                # Deliver the finish step with done=False; respawn next tick.
                self._finish_respawn_pending = True
                done = False
                logger.info("[RLClient] on_run_step t=%d: auto-respawn pending set", _time)
            else:
                done = finished or hard_crash

            # --- StepState gating: emit during the observation phase only.
            # Always emit on termination so the env sees the final state.
            in_observation_phase = self._window_tick < self._decision_idx or self._action_window_ticks == 1
            if in_observation_phase or finished or hard_crash:
                step_state = StepState(
                    state_data=data,
                    yaw_error=self._compute_yaw_error(data),
                    done=done,
                    finished=finished,
                )
                self._drain_and_put(step_state)

            # --- Advance window pointer.
            self._window_tick += 1
            if self._window_tick >= self._action_window_ticks:
                self._window_tick = 0

    def on_simulation_end(self, iface: TMInterface, result: int) -> None:
        """Fires once when replay validation finishes."""
        logger.info(
            "[RLClient] on_simulation_end result=%d "
            "(running=%s finish_pending=%s finish_delivered=%s respawn_set=%s queue_size=%d)",
            result, self._running, self._finish_respawn_pending,
            self._simulation_finish_delivered, self._respawn_event.is_set(),
            self._state_queue.qsize(),
        )

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int) -> None:
        logger.info(
            "[RLClient] on_checkpoint_count_changed %d/%d "
            "(running=%s finish_pending=%s)",
            current, target, self._running, self._finish_respawn_pending,
        )

    def on_laps_count_changed(self, iface: TMInterface, current: int) -> None:
        logger.info(
            "[RLClient] on_laps_count_changed laps=%d "
            "(running=%s finish_pending=%s finish_delivered=%s respawn_set=%s)",
            current, self._running, self._finish_respawn_pending,
            self._simulation_finish_delivered, self._respawn_event.is_set(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drain_and_put(self, step_state: StepState) -> None:
        """Replace any unread state in the queue with the newest one.

        If the RL thread hadn't yet read the previous state, carry its tick
        count forward so the new state's ticks_this_step reflects every game
        tick that fired since the last successful read.
        """
        self._last_step_state = step_state
        try:
            evicted = self._state_queue.get_nowait()
            step_state.ticks_this_step += evicted.ticks_this_step
        except queue.Empty:
            pass
        if step_state.done or step_state.finished or self._tick % 100 == 0:
            logger.debug("[RLClient] _drain_and_put: finished=%s done=%s progress=%s",
                        step_state.finished, step_state.done, step_state.state_data.track_progress)
        self._state_queue.put(step_state)

    def _compute_yaw_error(self, data: StateData) -> float:
        """Signed heading error: track yaw minus car yaw, wrapped to [-π, π]."""
        assert data.track_forward is not None  # always set when centerline is provided
        track_fwd = data.track_forward
        track_yaw = math.atan2(float(track_fwd[0]), float(track_fwd[2]))
        return angle_diff(track_yaw, data.rotation.yaw())
