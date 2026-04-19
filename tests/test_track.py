"""Tests for tmnf/track.py — Centerline projection."""
import os
import tempfile
import unittest

import numpy as np

from games.tmnf.track import Centerline
from games.tmnf.state import Vec3


class TestCenterline(unittest.TestCase):
    """Straight-line track along Z: 0 → 100 m (11 points, 10 m apart)."""

    @classmethod
    def setUpClass(cls):
        points = np.array([[0.0, 0.0, i * 10.0] for i in range(11)], dtype=np.float32)
        fd, cls._tmp_path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(cls._tmp_path, points)
        cls.cl = Centerline(cls._tmp_path)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls._tmp_path)

    def test_start_progress(self):
        progress, _, _ = self.cl.project(Vec3(0, 0, 0))
        self.assertAlmostEqual(progress, 0.0, places=3)

    def test_end_progress(self):
        progress, _, _ = self.cl.project(Vec3(0, 0, 100))
        self.assertAlmostEqual(progress, 1.0, places=3)

    def test_midpoint_progress(self):
        progress, _, _ = self.cl.project(Vec3(0, 0, 50))
        self.assertAlmostEqual(progress, 0.5, places=3)

    def test_lateral_offset_nonzero(self):
        # Point 2 m offset from centreline — lateral magnitude should be ~2
        _, lateral, _ = self.cl.project(Vec3(2.0, 0, 50))
        self.assertAlmostEqual(abs(lateral), 2.0, places=3)

    def test_on_centreline_zero_lateral(self):
        _, lateral, _ = self.cl.project(Vec3(0, 0, 30))
        self.assertAlmostEqual(lateral, 0.0, places=3)

    def test_forward_at_returns_unit_vector(self):
        fwd = self.cl.forward_at(Vec3(0, 0, 50))
        self.assertAlmostEqual(float(np.linalg.norm(fwd)), 1.0, places=5)


class TestProjectAhead(unittest.TestCase):
    """Straight-line track along Z: 0 → 100 m (11 points, 10 m apart)."""

    @classmethod
    def setUpClass(cls):
        points = np.array([[0.0, 0.0, i * 10.0] for i in range(11)], dtype=np.float32)
        fd, cls._tmp_path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(cls._tmp_path, points)
        cls.cl = Centerline(cls._tmp_path)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls._tmp_path)

    def test_returns_two_floats(self):
        result = self.cl.project_ahead(Vec3(0, 0, 0), nearest_idx=0, steps=5)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        lat, heading = result
        self.assertIsInstance(float(lat), float)
        self.assertIsInstance(float(heading), float)

    def test_straight_track_zero_heading_change(self):
        # Straight track: heading change between any two segments should be 0
        lat, heading = self.cl.project_ahead(Vec3(0, 0, 0), nearest_idx=0, steps=5)
        self.assertAlmostEqual(heading, 0.0, places=5)

    def test_straight_track_lateral_is_finite(self):
        lat, _ = self.cl.project_ahead(Vec3(0, 0, 0), nearest_idx=0, steps=5)
        self.assertTrue(np.isfinite(lat))

    def test_steps_clamped_at_track_end(self):
        # steps=9999 should not raise — clamped to last valid idx
        lat, heading = self.cl.project_ahead(Vec3(0, 0, 0), nearest_idx=0, steps=9999)
        self.assertTrue(np.isfinite(lat))
        self.assertTrue(np.isfinite(heading))

    def test_lateral_opposite_sign_when_car_moves_across_centreline(self):
        # Moving the car to opposite sides of the centreline should flip the sign
        # of the lateral offset, regardless of the axis convention.
        lat_pos, _ = self.cl.project_ahead(Vec3(3.0, 0, 0), nearest_idx=0, steps=5)
        lat_neg, _ = self.cl.project_ahead(Vec3(-3.0, 0, 0), nearest_idx=0, steps=5)
        # Signs must be strictly opposite
        self.assertGreater(lat_pos * lat_neg, -float('inf'))  # both finite
        self.assertNotEqual(lat_pos, 0.0)
        self.assertNotEqual(lat_neg, 0.0)
        # Opposite sides → opposite signs
        self.assertLess(lat_pos * lat_neg, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
