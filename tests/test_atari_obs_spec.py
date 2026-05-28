"""Tests for the Atari observation spec (128-byte RAM)."""

from __future__ import annotations

import unittest

import numpy as np

from games.atari.obs_spec import (
    ATARI_OBS_SPEC,
    BASE_OBS_DIM,
    OBS_NAMES,
    OBS_SCALES,
    RAM_SIZE,
)


class TestAtariObsSpec(unittest.TestCase):
    def test_dim_is_128(self):
        self.assertEqual(BASE_OBS_DIM, 128)
        self.assertEqual(ATARI_OBS_SPEC.dim, 128)
        self.assertEqual(RAM_SIZE, 128)

    def test_obs_names_match_dim(self):
        self.assertEqual(len(OBS_NAMES), BASE_OBS_DIM)

    def test_obs_names_are_unique(self):
        self.assertEqual(len(set(OBS_NAMES)), len(OBS_NAMES))

    def test_obs_names_are_ordered(self):
        for i, name in enumerate(OBS_NAMES):
            self.assertEqual(name, f"ram_{i:03d}")

    def test_scales_shape_and_dtype(self):
        self.assertEqual(OBS_SCALES.shape, (BASE_OBS_DIM,))
        self.assertEqual(OBS_SCALES.dtype, np.float32)

    def test_all_scales_are_255(self):
        np.testing.assert_array_equal(OBS_SCALES, np.full(BASE_OBS_DIM, 255.0, dtype=np.float32))

    def test_dims_carry_descriptions(self):
        for dim in ATARI_OBS_SPEC.dims:
            self.assertTrue(dim.description)


if __name__ == "__main__":
    unittest.main()
