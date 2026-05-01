"""Tests for the TORCS client observation/action mapping.

These tests validate the client's data transformation logic without
requiring TORCS to be installed.
"""
import unittest

import numpy as np

from games.torcs.client import TorcsClient
from games.torcs.obs_spec import BASE_OBS_DIM


class _FakeObs:
    """Mimics the observation object returned by gym_torcs."""
    def __init__(self, **kwargs):
        defaults = {
            "speedX": 0.5,          # normalised by 300 in gym_torcs
            "trackPos": 0.0,
            "angle": 0.1,
            "distRaced": 500.0,
            "trackLength": 5000.0,
            "rpm": 3000.0,
            "wheelSpinVel": np.array([10.0, 10.0, 10.0, 10.0]),
            "track": np.zeros(19),
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


class TestTorcsClientFlattenObs(unittest.TestCase):
    """Test the observation flattening logic."""

    def setUp(self):
        self.client = TorcsClient.__new__(TorcsClient)

    def test_output_shape(self):
        obs = self.client._flatten_obs(_FakeObs())
        self.assertEqual(obs.shape, (BASE_OBS_DIM,))
        self.assertEqual(obs.dtype, np.float32)

    def test_speed_conversion(self):
        obs = self.client._flatten_obs(_FakeObs(speedX=0.5))
        # speedX is normalised by 300 in gym_torcs → speed_ms = 0.5 * 300 = 150
        self.assertAlmostEqual(obs[0], 150.0, places=1)

    def test_lateral_offset(self):
        obs = self.client._flatten_obs(_FakeObs(trackPos=0.5))
        # trackPos ∈ [-1, 1], multiplied by 5 → lateral_offset = 2.5
        self.assertAlmostEqual(obs[1], 2.5, places=1)

    def test_angle(self):
        obs = self.client._flatten_obs(_FakeObs(angle=0.3))
        self.assertAlmostEqual(obs[2], 0.3, places=3)

    def test_progress(self):
        obs = self.client._flatten_obs(_FakeObs(distRaced=1000.0, trackLength=5000.0))
        self.assertAlmostEqual(obs[3], 0.2, places=3)

    def test_rpm(self):
        obs = self.client._flatten_obs(_FakeObs(rpm=5000.0))
        self.assertAlmostEqual(obs[4], 5000.0, places=1)

    def test_wheel_spin(self):
        ws = np.array([1.0, 2.0, 3.0, 4.0])
        obs = self.client._flatten_obs(_FakeObs(wheelSpinVel=ws))
        np.testing.assert_array_almost_equal(obs[5:9], ws)

    def test_track_position(self):
        obs = self.client._flatten_obs(_FakeObs(trackPos=-0.3))
        self.assertAlmostEqual(obs[18], -0.3, places=3)


class TestTorcsClientMapAction(unittest.TestCase):
    """Test the action mapping logic."""

    def test_full_throttle_straight(self):
        action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        result = TorcsClient._map_action(action)
        self.assertAlmostEqual(result[0], 0.0)   # steer
        self.assertAlmostEqual(result[1], 1.0)    # net accel

    def test_full_brake(self):
        action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        result = TorcsClient._map_action(action)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], -1.0)  # net accel = 0 - 1

    def test_steer_clipped(self):
        action = np.array([2.0, 0.5, 0.0], dtype=np.float32)
        result = TorcsClient._map_action(action)
        self.assertAlmostEqual(result[0], 1.0)    # clipped to 1

    def test_combined_accel_brake(self):
        action = np.array([0.0, 0.8, 0.3], dtype=np.float32)
        result = TorcsClient._map_action(action)
        self.assertAlmostEqual(result[1], 0.5)    # 0.8 - 0.3


if __name__ == "__main__":
    unittest.main()
