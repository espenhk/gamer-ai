"""Tests for the SC2 observation spec presets."""
import unittest

import numpy as np

from games.sc2.obs_spec import (
    BASE_OBS_DIM,
    LADDER_OBS_DIM,
    MINIGAME_NAMES,
    OBS_NAMES,
    RICH_OBS_DIM,
    SC2_LADDER_OBS_SPEC,
    SC2_MINIGAME_OBS_SPEC,
    SC2_OBS_SPEC,
    SC2_RICH_OBS_SPEC,
    get_spec,
)


class TestSC2ObsSpec(unittest.TestCase):

    def test_minigame_spec_dim(self):
        self.assertEqual(BASE_OBS_DIM, 13)
        self.assertEqual(SC2_MINIGAME_OBS_SPEC.dim, 13)

    def test_ladder_spec_dim_grows_post_126(self):
        """Issue #126: ladder spec extended with economy / minimap-camera /
        score / HP / top-K-counts blocks (~40 dims, exact value asserted)."""
        self.assertEqual(LADDER_OBS_DIM, 43)
        self.assertEqual(SC2_LADDER_OBS_SPEC.dim, 43)

    def test_rich_spec_dim(self):
        self.assertEqual(RICH_OBS_DIM, SC2_RICH_OBS_SPEC.dim)
        self.assertGreater(RICH_OBS_DIM, LADDER_OBS_DIM)

    def test_ladder_extends_minigame(self):
        """Ladder spec must contain all minigame names as a prefix."""
        for i, name in enumerate(SC2_MINIGAME_OBS_SPEC.names):
            self.assertEqual(SC2_LADDER_OBS_SPEC.names[i], name)

    def test_rich_extends_ladder(self):
        """Rich spec must contain all ladder names as a prefix."""
        for i, name in enumerate(SC2_LADDER_OBS_SPEC.names):
            self.assertEqual(SC2_RICH_OBS_SPEC.names[i], name)

    def test_default_spec_is_minigame(self):
        self.assertIs(SC2_OBS_SPEC, SC2_MINIGAME_OBS_SPEC)

    def test_get_spec_for_minigame(self):
        for name in MINIGAME_NAMES:
            self.assertIs(get_spec(name), SC2_MINIGAME_OBS_SPEC)

    def test_get_spec_for_ladder_map(self):
        self.assertIs(get_spec("Simple64"), SC2_LADDER_OBS_SPEC)
        self.assertIs(get_spec("AbyssalReef"), SC2_LADDER_OBS_SPEC)

    def test_get_spec_explicit_preset_overrides_map(self):
        """obs_spec_preset='rich' upgrades a minigame map to the full spec."""
        self.assertIs(get_spec("MoveToBeacon", preset="rich"), SC2_RICH_OBS_SPEC)
        self.assertIs(get_spec("Simple64",     preset="minigame"), SC2_MINIGAME_OBS_SPEC)

    def test_get_spec_unknown_preset_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_spec("Simple64", preset="unknown")
        self.assertIn("unknown", str(ctx.exception))

    def test_minigame_count(self):
        self.assertEqual(len(MINIGAME_NAMES), 7)

    def test_obs_names_match_dims(self):
        self.assertEqual(len(OBS_NAMES), BASE_OBS_DIM)

    def test_all_preset_names_are_unique(self):
        for spec in (SC2_MINIGAME_OBS_SPEC, SC2_LADDER_OBS_SPEC, SC2_RICH_OBS_SPEC):
            self.assertEqual(len(spec.names), len(set(spec.names)))

    def test_all_preset_scales_finite(self):
        for spec in (SC2_MINIGAME_OBS_SPEC, SC2_LADDER_OBS_SPEC, SC2_RICH_OBS_SPEC):
            self.assertTrue(np.all(np.isfinite(spec.scales)))

    def test_all_preset_dims_have_descriptions(self):
        for spec in (SC2_MINIGAME_OBS_SPEC, SC2_LADDER_OBS_SPEC, SC2_RICH_OBS_SPEC):
            for d in spec.dims:
                self.assertGreater(len(d.description), 0,
                                   f"{d.name} has no description")


if __name__ == "__main__":
    unittest.main()
