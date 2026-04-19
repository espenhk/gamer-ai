"""Tests for tmnf/utils.py — Vec3, Quat, StateData."""
import math
import os
import tempfile
import unittest

import numpy as np

from helpers import make_game_state
from games.tmnf.track import Centerline
from games.tmnf.state import Vec3, Quat, StateData


class TestVec3(unittest.TestCase):

    def test_magnitude_zero(self):
        self.assertEqual(Vec3(0, 0, 0).magnitude(), 0.0)

    def test_magnitude_unit(self):
        self.assertAlmostEqual(Vec3(1, 0, 0).magnitude(), 1.0)

    def test_magnitude_3d(self):
        self.assertAlmostEqual(Vec3(3, 4, 0).magnitude(), 5.0)

    def test_compute_speed_alias(self):
        v = Vec3(1, 2, 3)
        self.assertEqual(v.compute_speed(), v.magnitude())


class TestQuat(unittest.TestCase):

    def test_identity_yaw_zero(self):
        self.assertAlmostEqual(Quat(1, 0, 0, 0).yaw(), 0.0)

    def test_identity_pitch_zero(self):
        self.assertAlmostEqual(Quat(1, 0, 0, 0).pitch(), 0.0)

    def test_identity_roll_zero(self):
        self.assertAlmostEqual(Quat(1, 0, 0, 0).roll(), 0.0)

    def test_90deg_yaw(self):
        # 90° rotation around Y axis: w=cos(45°), y=sin(45°)
        half = math.sqrt(2) / 2
        q = Quat(half, 0, half, 0)
        self.assertAlmostEqual(q.yaw(), math.pi / 2, places=5)


class TestStateData(unittest.TestCase):

    def test_extracts_velocity(self):
        gs = make_game_state(linear_speed=(5.0, 2.0, 3.0))
        sd = StateData(gs)
        self.assertAlmostEqual(sd.velocity.x, 5.0)
        self.assertAlmostEqual(sd.velocity.y, 2.0)
        self.assertAlmostEqual(sd.velocity.z, 3.0)

    def test_extracts_wheels(self):
        gs = make_game_state(
            wheel_contacts=(True, False, True, False),
            wheel_sliding=(False, True, False, True),
        )
        sd = StateData(gs)
        self.assertTrue(sd.wheels[0].contact)
        self.assertFalse(sd.wheels[1].contact)
        self.assertTrue(sd.wheels[1].sliding)

    def test_with_centerline_sets_progress(self):
        points = np.array([[0, 0, i * 10.0] for i in range(11)], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, points)
            cl = Centerline(f.name)
        try:
            gs = make_game_state(position=(0.0, 0.0, 50.0))
            sd = StateData(gs, centerline=cl)
            self.assertIsNotNone(sd.track_progress)
            self.assertAlmostEqual(sd.track_progress, 0.5, places=2)
        finally:
            os.unlink(f.name)

    def test_lookahead_has_three_entries_when_centerline_present(self):
        points = np.array([[0, 0, i * 10.0] for i in range(11)], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, points)
            cl = Centerline(f.name)
        try:
            gs = make_game_state(position=(0.0, 0.0, 10.0))
            sd = StateData(gs, centerline=cl)
            self.assertEqual(len(sd.lookahead), 3)
            for lat, heading in sd.lookahead:
                self.assertTrue(math.isfinite(lat))
                self.assertTrue(math.isfinite(heading))
        finally:
            os.unlink(f.name)

    def test_lookahead_defaults_to_zeros_without_centerline(self):
        gs = make_game_state(position=(0.0, 0.0, 0.0))
        sd = StateData(gs)
        self.assertEqual(sd.lookahead, [(0.0, 0.0)] * 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
