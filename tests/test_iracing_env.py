"""Tests for the iRacing env wrapper — telemetry parsing + episode logic.

The pyirsdk dependency (and a running iRacing client) are mocked, so these
run on the cross-platform unit suite.  The focus is the previously-untested
``IRacingEnv._read_telemetry`` mapping from pyirsdk telemetry keys into the
fixed-size observation vector, plus the reset/step/close lifecycle and the
finish / crash / timeout termination logic.
"""

from __future__ import annotations

import sys
import types
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Inject a fake ``irsdk`` module *before* importing games.iracing.env, whose
# module-level ``import irsdk`` would otherwise raise ImportError.
# ---------------------------------------------------------------------------


class _FakeIRSDK:
    """Stand-in for ``irsdk.IRSDK`` backed by a plain telemetry dict.

    Telemetry is read via ``self._ir[key]`` in the env; unknown keys return
    ``None`` exactly like pyirsdk does for an absent telemetry channel.
    """

    def __init__(self) -> None:
        self.telemetry: dict[str, float] = {}
        self.started = False
        self.shutdown_called = False

    def startup(self) -> bool:
        self.started = True
        return True

    def shutdown(self) -> None:
        self.shutdown_called = True

    def __getitem__(self, key: str):
        return self.telemetry.get(key)


def _install_fake_irsdk():
    fake = types.ModuleType("irsdk")
    fake.IRSDK = _FakeIRSDK  # type: ignore[attr-defined]
    sys.modules["irsdk"] = fake
    return fake


class _RecordingController:
    """Captures the last action the env injected (clipped steer/throttle/brake)."""

    def __init__(self) -> None:
        self.last = None
        self.reset_called = False
        self.close_called = False

    def send(self, *, steer: float, throttle: float, brake: float) -> None:
        self.last = (steer, throttle, brake)

    def reset(self) -> None:
        self.reset_called = True

    def close(self) -> None:
        self.close_called = True


# A canonical telemetry frame with a distinct value per channel so a wrong
# index mapping in _read_telemetry produces a wrong assertion.
_TELEMETRY = {
    "Speed": 42.0,
    "CarIdxLapDistPct": 0.3,
    "LapDistPct": 0.5,
    "YawRate": 0.11,
    "RPM": 7200.0,
    "Gear": 3.0,
    "FuelLevelPct": 0.85,
    "Throttle": 0.9,
    "Brake": 0.1,
    "SteeringWheelAngle": -0.25,
    # Shock deflection (metres) feeds the tyre-load slots; carcass temps feed
    # the tyre-temp slots — deliberately distinct values so a regression that
    # aliases the two blocks onto one channel is caught.
    "LFshockDefl": 0.011,
    "RFshockDefl": 0.012,
    "LRshockDefl": 0.013,
    "RRshockDefl": 0.014,
    "LFtempCL": 70.0,
    "RFtempCL": 71.0,
    "LRtempCL": 72.0,
    "RRtempCL": 73.0,
    "dcBrakeBias": 0.55,
    "LapCurrentLapTime": 88.4,
    "LapBestLapTime": 85.1,
}


class _IRacingEnvTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_irsdk = sys.modules.get("irsdk")
        _install_fake_irsdk()

    @classmethod
    def tearDownClass(cls):
        if cls._old_irsdk is None:
            sys.modules.pop("irsdk", None)
        else:
            sys.modules["irsdk"] = cls._old_irsdk

    def _make_env(self, **kwargs):
        from games.iracing.env import IRacingEnv

        controller = kwargs.pop("controller", None) or _RecordingController()
        env = IRacingEnv(controller=controller, **kwargs)
        env._ir.telemetry = dict(_TELEMETRY)
        return env, controller


class TestReadTelemetryMapping(_IRacingEnvTestBase):
    """_read_telemetry writes each pyirsdk channel into the correct obs slot."""

    def setUp(self):
        self.env, _ = self._make_env()
        self.obs = self.env._read_telemetry()

    def test_shape_and_dtype(self):
        from games.iracing.obs_spec import BASE_OBS_DIM

        self.assertEqual(self.obs.shape, (BASE_OBS_DIM,))
        self.assertEqual(self.obs.dtype, np.float32)

    def test_scalar_channels_map_to_expected_indices(self):
        self.assertAlmostEqual(self.obs[0], 42.0, places=4)  # speed_ms ← Speed
        self.assertAlmostEqual(self.obs[1], 0.3, places=4)  # lateral ← CarIdxLapDistPct
        self.assertAlmostEqual(self.obs[2], 0.5, places=4)  # progress ← LapDistPct
        self.assertAlmostEqual(self.obs[3], 0.11, places=4)  # yaw ← YawRate
        self.assertAlmostEqual(self.obs[4], 7200.0, places=4)  # rpm ← RPM
        self.assertAlmostEqual(self.obs[5], 3.0, places=4)  # gear ← Gear
        self.assertAlmostEqual(self.obs[6], 0.85, places=4)  # fuel ← FuelLevelPct
        self.assertAlmostEqual(self.obs[7], 0.9, places=4)  # throttle ← Throttle
        self.assertAlmostEqual(self.obs[8], 0.1, places=4)  # brake ← Brake
        self.assertAlmostEqual(self.obs[9], -0.25, places=4)  # steering ← SteeringWheelAngle
        self.assertAlmostEqual(self.obs[18], 0.55, places=4)  # brake_bias ← dcBrakeBias
        self.assertAlmostEqual(self.obs[19], 88.4, places=4)  # lap_time ← LapCurrentLapTime
        self.assertAlmostEqual(self.obs[20], 85.1, places=4)  # best_lap ← LapBestLapTime

    def test_tire_load_block_reads_shock_deflection(self):
        # Load slots (10–13) come from the per-corner shock-deflection channels.
        self.assertAlmostEqual(self.obs[10], 0.011, places=4)  # LFshockDefl
        self.assertAlmostEqual(self.obs[11], 0.012, places=4)  # RFshockDefl
        self.assertAlmostEqual(self.obs[12], 0.013, places=4)  # LRshockDefl
        self.assertAlmostEqual(self.obs[13], 0.014, places=4)  # RRshockDefl

    def test_tire_temp_block_reads_carcass_temp(self):
        # Temp slots (14–17) come from the per-corner carcass-temp channels.
        self.assertAlmostEqual(self.obs[14], 70.0, places=4)  # LFtempCL
        self.assertAlmostEqual(self.obs[15], 71.0, places=4)  # RFtempCL
        self.assertAlmostEqual(self.obs[16], 72.0, places=4)  # LRtempCL
        self.assertAlmostEqual(self.obs[17], 73.0, places=4)  # RRtempCL

    def test_tire_load_and_temp_blocks_are_distinct(self):
        # Regression guard for the load/temp aliasing fix (PR #479): the two
        # blocks must read different telemetry channels, not the same one.
        self.assertFalse(np.allclose(self.obs[10:14], self.obs[14:18]))

    def test_absent_channel_defaults_to_zero(self):
        env, _ = self._make_env()
        env._ir.telemetry = {"Speed": 12.0}  # everything else absent → None
        obs = env._read_telemetry()
        self.assertAlmostEqual(obs[0], 12.0, places=4)
        self.assertAlmostEqual(obs[2], 0.0, places=4)  # LapDistPct missing → 0.0
        self.assertFalse(np.any(np.isnan(obs)))


class TestEnvLifecycle(_IRacingEnvTestBase):
    def test_reset_connects_and_returns_obs(self):
        env, ctrl = self._make_env()
        obs, info = env.reset()
        self.assertTrue(env._ir.started)
        self.assertTrue(ctrl.reset_called)
        from games.iracing.obs_spec import BASE_OBS_DIM

        self.assertEqual(obs.shape, (BASE_OBS_DIM,))
        self.assertEqual(info, {})

    def test_step_returns_five_tuple(self):
        env, _ = self._make_env()
        env.reset()
        result = env.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        for key in ("speed_ms", "track_progress", "termination_reason"):
            self.assertIn(key, info)

    def test_step_clips_action_before_injection(self):
        env, ctrl = self._make_env()
        env.reset()
        env.step(np.array([5.0, 5.0, 5.0], dtype=np.float32))
        self.assertEqual(ctrl.last, (1.0, 1.0, 1.0))
        env.step(np.array([-5.0, -5.0, -5.0], dtype=np.float32))
        self.assertEqual(ctrl.last, (-1.0, 0.0, 0.0))

    def test_close_shuts_down_sdk_and_controller(self):
        env, ctrl = self._make_env()
        env.reset()
        env.close()
        self.assertTrue(env._ir.shutdown_called)
        self.assertTrue(ctrl.close_called)

    def test_episode_time_limit_round_trip(self):
        env, _ = self._make_env(max_episode_time_s=90.0)
        self.assertEqual(env.get_episode_time_limit(), 90.0)
        env.set_episode_time_limit(150.0)
        self.assertEqual(env.get_episode_time_limit(), 150.0)


class TestTerminationLogic(_IRacingEnvTestBase):
    def test_finish_when_progress_complete(self):
        env, _ = self._make_env()
        env.reset()
        env._ir.telemetry = {**_TELEMETRY, "LapDistPct": 1.0}
        _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "finish")

    def test_crash_when_lateral_exceeds_threshold(self):
        env, _ = self._make_env()
        env.reset()
        # CarIdxLapDistPct feeds the lateral-offset slot; push it past the
        # 25 m crash threshold while progress stays below 1.0.
        env._ir.telemetry = {**_TELEMETRY, "LapDistPct": 0.4, "CarIdxLapDistPct": 99.0}
        _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertEqual(info["termination_reason"], "crash")

    def test_timeout_truncates_without_terminating(self):
        env, _ = self._make_env(max_episode_time_s=0.0)
        env.reset()
        env._ir.telemetry = {**_TELEMETRY, "LapDistPct": 0.4, "CarIdxLapDistPct": 0.0}
        _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["termination_reason"], "timeout")


if __name__ == "__main__":
    unittest.main()
