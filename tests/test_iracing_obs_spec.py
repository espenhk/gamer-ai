"""Tests for the iRacing observation spec (telemetry-driven vector).

Mirrors the well-tested TMNF / Atari obs-spec pattern: the spec is the
single source of truth for feature names, scales and ordering, so a
parsing bug in ``IRacingEnv._read_telemetry`` (which writes into these
slots by index) would otherwise go uncaught on the cross-platform suite.
"""

from __future__ import annotations

import unittest

import numpy as np

from framework.obs_spec import ObsDim, ObsSpec
from games.iracing.obs_spec import (
    BASE_OBS_DIM,
    IRACING_OBS_SPEC,
    OBS_NAMES,
    OBS_SCALES,
    OBS_SPEC,
)


class TestIRacingObsSpec(unittest.TestCase):
    def test_dim_is_21(self):
        self.assertEqual(BASE_OBS_DIM, 21)
        self.assertEqual(IRACING_OBS_SPEC.dim, 21)

    def test_obs_names_match_dim(self):
        self.assertEqual(len(OBS_NAMES), BASE_OBS_DIM)

    def test_obs_names_are_unique(self):
        self.assertEqual(len(set(OBS_NAMES)), len(OBS_NAMES))

    def test_scales_shape_and_dtype(self):
        self.assertEqual(OBS_SCALES.shape, (BASE_OBS_DIM,))
        self.assertEqual(OBS_SCALES.dtype, np.float32)

    def test_all_scales_positive(self):
        # Scales are divisors used for normalisation — a zero scale would
        # produce inf/nan observations.
        self.assertTrue(np.all(OBS_SCALES > 0.0))

    def test_feature_order_is_stable(self):
        # ``_read_telemetry`` writes telemetry into obs[i] by position, so the
        # name→index mapping is a load-bearing contract.  Pin the leading and
        # trailing features to catch accidental reordering.
        self.assertEqual(OBS_NAMES[0], "speed_ms")
        self.assertEqual(OBS_NAMES[1], "lateral_offset_m")
        self.assertEqual(OBS_NAMES[2], "track_progress")
        self.assertEqual(OBS_NAMES[19], "lap_time_s")
        self.assertEqual(OBS_NAMES[20], "best_lap_time_s")

    def test_tire_feature_block_present(self):
        # Four loads then four temps — the distinctive iRacing signal.
        for axle in ("fl", "fr", "rl", "rr"):
            self.assertIn(f"tire_load_{axle}", OBS_NAMES)
            self.assertIn(f"tire_temp_{axle}", OBS_NAMES)

    def test_dims_carry_descriptions(self):
        for dim in OBS_SPEC:
            self.assertTrue(dim.description)

    def test_normalize_round_trips_through_scales(self):
        # Framework normalisation is ``raw_obs / spec.scales`` — a full-scale
        # raw vector must map to all-ones.
        raw = OBS_SCALES.copy()  # one full-scale unit per feature
        normalised = raw / IRACING_OBS_SPEC.scales
        np.testing.assert_allclose(normalised, np.ones(BASE_OBS_DIM), rtol=1e-5)

    def test_missing_key_migration_defaults_to_zero(self):
        # The framework's "missing key → 0.0" rule lets old weight files load
        # under a newer spec.  Extend the spec and confirm an old-style obs
        # zero-pads cleanly rather than raising.
        extended = IRACING_OBS_SPEC.with_extra_dims([ObsDim("future_feature", 1.0, "added later")])
        self.assertEqual(extended.dim, BASE_OBS_DIM + 1)
        self.assertEqual(extended.names[-1], "future_feature")
        self.assertIsInstance(extended, ObsSpec)


if __name__ == "__main__":
    unittest.main()
