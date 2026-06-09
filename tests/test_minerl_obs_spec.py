"""Tests for the MineRL observation spec (vector obs, Phase 1)."""

from __future__ import annotations

import unittest

import numpy as np

from games.minerl.obs_spec import (
    BASE_OBS_DIM,
    MINERL_OBS_SPEC,
    OBS_NAMES,
    OBS_SCALES,
    OBS_SPEC,
)


class TestMineRLObsSpec(unittest.TestCase):
    def test_dim_is_3(self):
        self.assertEqual(BASE_OBS_DIM, 3)
        self.assertEqual(MINERL_OBS_SPEC.dim, 3)

    def test_obs_names_match_dim(self):
        self.assertEqual(len(OBS_NAMES), BASE_OBS_DIM)

    def test_obs_names_are_unique(self):
        self.assertEqual(len(set(OBS_NAMES)), len(OBS_NAMES))

    def test_obs_names_expected(self):
        self.assertIn("compass_angle", OBS_NAMES)
        self.assertIn("inventory_dirt", OBS_NAMES)
        self.assertIn("inventory_log", OBS_NAMES)

    def test_scales_shape_and_dtype(self):
        self.assertEqual(OBS_SCALES.shape, (BASE_OBS_DIM,))
        self.assertEqual(OBS_SCALES.dtype, np.float32)

    def test_compass_scale_is_180(self):
        compass_idx = OBS_NAMES.index("compass_angle")
        self.assertEqual(OBS_SCALES[compass_idx], 180.0)

    def test_inventory_scales_are_64(self):
        for name in ("inventory_dirt", "inventory_log"):
            idx = OBS_NAMES.index(name)
            self.assertEqual(OBS_SCALES[idx], 64.0)

    def test_dims_carry_descriptions(self):
        for dim in OBS_SPEC:
            self.assertTrue(dim.description)

    def test_obs_spec_list_length(self):
        self.assertEqual(len(OBS_SPEC), BASE_OBS_DIM)


if __name__ == "__main__":
    unittest.main()
