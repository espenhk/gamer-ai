"""Unit tests for per-episode termination reason tracking in TMNFEnv.step()."""
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from games.tmnf.env import TMNFEnv
from games.tmnf.reward import RewardConfig, RewardCalculator
from games.tmnf.state import Vec3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_step_state(
    *,
    finished: bool = False,
    done: bool = False,
    track_progress: float = 0.5,
    lateral_offset: float = 0.0,
    ticks_this_step: int = 1,
) -> MagicMock:
    """Build a minimal StepState mock covering all fields read by step()."""
    step = MagicMock()
    step.finished = finished
    step.done = done
    step.ticks_this_step = ticks_this_step
    step.yaw_error = 0.0

    sd = MagicMock()
    sd.track_progress = track_progress
    sd.lateral_offset = lateral_offset
    sd.vertical_offset = 0.0
    sd.turning_rate = 0.0
    sd.velocity = Vec3(5.0, 0.0, 0.0)
    sd.angular_velocity = Vec3(0.0, 0.0, 0.0)
    sd.position = Vec3(0.0, 0.0, 0.0)
    sd.rotation = MagicMock()
    sd.rotation.pitch.return_value = 0.0
    sd.rotation.roll.return_value = 0.0
    sd.wheels = [MagicMock(contact=True) for _ in range(4)]
    sd.lookahead = [(0.0, 0.0)] * 3
    step.state_data = sd
    return step


def _make_env(
    crash_threshold_m: float = 10.0,
    max_episode_time_s: float = 60.0,
    auto_respawn_on_finish: bool = False,
) -> TMNFEnv:
    """Instantiate TMNFEnv bypassing __init__, wiring only what step() needs."""
    env = TMNFEnv.__new__(TMNFEnv)
    env._reward_config = RewardConfig(crash_threshold_m=crash_threshold_m)
    # Stub reward computation so MagicMock prev_state fields never reach arithmetic.
    env._reward_calc = MagicMock()
    env._reward_calc.compute.return_value = 0.0
    env._max_episode_time_s = max_episode_time_s
    env._auto_respawn_on_finish = auto_respawn_on_finish
    env._lidar = None
    env._prev_state = MagicMock()
    env._elapsed_s = 0.0
    # Place episode start 1 s in the past so elapsed_s is deterministically > 0.
    env._episode_start_s = time.monotonic() - 1.0
    env._laps_completed = 0
    env._ep_rl_steps = 0
    env._ep_total_ticks = 0
    env._ep_max_window_ticks = 0
    env._ep_max_overrun_ticks = 0
    env._action_window_ticks = 1
    env._total_rl_steps = 0
    env._client = MagicMock()
    env._log_skip_stats = MagicMock()
    return env


def _do_step(env: TMNFEnv, step_state: MagicMock) -> tuple:
    env._client.get_step_state.return_value = step_state
    action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
    dummy_obs = np.zeros(21, dtype=np.float32)
    with patch.object(env, "_build_obs", return_value=dummy_obs):
        return env.step(action)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTerminationReason(unittest.TestCase):

    def test_finish(self):
        env = _make_env()
        step = _make_step_state(finished=True, track_progress=1.0)
        _, _, terminated, truncated, info = _do_step(env, step)
        self.assertEqual(info["termination_reason"], "finish")
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_crash(self):
        env = _make_env(crash_threshold_m=10.0)
        step = _make_step_state(lateral_offset=15.0)
        _, _, terminated, truncated, info = _do_step(env, step)
        self.assertEqual(info["termination_reason"], "crash")
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_hard_crash(self):
        # done=True but lateral offset within threshold (not crashed) and not finished
        env = _make_env(crash_threshold_m=10.0)
        step = _make_step_state(done=True, lateral_offset=0.0)
        _, _, terminated, truncated, info = _do_step(env, step)
        self.assertEqual(info["termination_reason"], "hard_crash")
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_timeout(self):
        # max_episode_time_s=0.0 ensures time_over is True immediately
        env = _make_env(max_episode_time_s=0.0)
        step = _make_step_state()
        _, _, terminated, truncated, info = _do_step(env, step)
        self.assertEqual(info["termination_reason"], "timeout")
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_still_running(self):
        env = _make_env(max_episode_time_s=9999.0)
        step = _make_step_state(track_progress=0.3)
        _, _, terminated, truncated, info = _do_step(env, step)
        self.assertIsNone(info["termination_reason"])
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_finish_takes_priority_over_crash(self):
        env = _make_env(crash_threshold_m=10.0)
        step = _make_step_state(finished=True, track_progress=1.0, lateral_offset=20.0)
        _, _, _, _, info = _do_step(env, step)
        self.assertEqual(info["termination_reason"], "finish")

    def test_reason_key_always_present(self):
        # termination_reason must always be in info, not just on episode end
        env = _make_env(max_episode_time_s=9999.0)
        step = _make_step_state(track_progress=0.1)
        _, _, _, _, info = _do_step(env, step)
        self.assertIn("termination_reason", info)


if __name__ == "__main__":
    unittest.main()
