"""Tests for the TORCS observation spec."""
import unittest

import numpy as np

from games.torcs.obs_spec import (
    TORCS_OBS_SPEC,
    BASE_OBS_DIM,
    OBS_NAMES,
    OBS_SCALES,
    OBS_SPEC,
)
from framework.obs_spec import ObsDim, ObsSpec


class TestTorcsObsSpec(unittest.TestCase):
    """Validate TORCS_OBS_SPEC structure and derived constants."""

    def test_is_obs_spec_instance(self):
        self.assertIsInstance(TORCS_OBS_SPEC, ObsSpec)

    def test_dim_matches_base(self):
        self.assertEqual(TORCS_OBS_SPEC.dim, BASE_OBS_DIM)

    def test_dim_is_19(self):
        self.assertEqual(BASE_OBS_DIM, 19)

    def test_names_length(self):
        self.assertEqual(len(OBS_NAMES), BASE_OBS_DIM)

    def test_scales_shape(self):
        self.assertEqual(OBS_SCALES.shape, (BASE_OBS_DIM,))
        self.assertEqual(OBS_SCALES.dtype, np.float32)

    def test_scales_positive(self):
        self.assertTrue(np.all(OBS_SCALES > 0))

    def test_obs_spec_list_matches(self):
        self.assertEqual(len(OBS_SPEC), BASE_OBS_DIM)
        for dim in OBS_SPEC:
            self.assertIsInstance(dim, ObsDim)

    def test_names_unique(self):
        self.assertEqual(len(OBS_NAMES), len(set(OBS_NAMES)))

    def test_first_feature_is_speed(self):
        self.assertEqual(OBS_NAMES[0], "speed_ms")

    def test_last_feature_is_track_position(self):
        self.assertEqual(OBS_NAMES[-1], "track_position")

    def test_track_edge_features_present(self):
        edge_names = [n for n in OBS_NAMES if n.startswith("track_edge_")]
        self.assertEqual(len(edge_names), 9)


class TestTorcsObsSpecWithLidar(unittest.TestCase):
    """Ensure the with_lidar extension works with TORCS spec."""

    def test_with_zero_lidar_returns_same(self):
        spec = TORCS_OBS_SPEC.with_lidar(0)
        self.assertIs(spec, TORCS_OBS_SPEC)

    def test_with_lidar_extends_dim(self):
        spec = TORCS_OBS_SPEC.with_lidar(5)
        self.assertEqual(spec.dim, BASE_OBS_DIM + 5)

    def test_with_lidar_names(self):
        spec = TORCS_OBS_SPEC.with_lidar(3)
        self.assertEqual(spec.names[-3:], ["lidar_0", "lidar_1", "lidar_2"])


if __name__ == "__main__":
    unittest.main()
