"""Tests for tmnf/track.py — Centerline projection."""
import os
import tempfile
import unittest

import numpy as np

from track import Centerline
from utils import Vec3


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
