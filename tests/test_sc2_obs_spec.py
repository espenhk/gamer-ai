"""Tests for the SC2 observation spec."""
import unittest

from games.sc2.obs_spec import (
    BASE_OBS_DIM,
    LADDER_OBS_DIM,
    MINIGAME_NAMES,
    OBS_NAMES,
    SC2_LADDER_OBS_SPEC,
    SC2_MINIGAME_OBS_SPEC,
    SC2_OBS_SPEC,
    get_spec,
)


class TestSC2ObsSpec(unittest.TestCase):

    def test_minigame_spec_dim(self):
        self.assertEqual(BASE_OBS_DIM, 13)
        self.assertEqual(SC2_MINIGAME_OBS_SPEC.dim, 13)

    def test_ladder_spec_dim(self):
        self.assertEqual(LADDER_OBS_DIM, 21)
        self.assertEqual(SC2_LADDER_OBS_SPEC.dim, 21)

    def test_ladder_extends_minigame(self):
        """Ladder spec must contain all minigame names as a prefix."""
        for i, name in enumerate(SC2_MINIGAME_OBS_SPEC.names):
            self.assertEqual(SC2_LADDER_OBS_SPEC.names[i], name)

    def test_default_spec_is_minigame(self):
        self.assertIs(SC2_OBS_SPEC, SC2_MINIGAME_OBS_SPEC)

    def test_get_spec_for_minigame(self):
        for name in MINIGAME_NAMES:
            self.assertIs(get_spec(name), SC2_MINIGAME_OBS_SPEC)

    def test_get_spec_for_ladder_map(self):
        self.assertIs(get_spec("Simple64"), SC2_LADDER_OBS_SPEC)
        self.assertIs(get_spec("AbyssalReef"), SC2_LADDER_OBS_SPEC)

    def test_minigame_count(self):
        self.assertEqual(len(MINIGAME_NAMES), 7)

    def test_obs_names_match_dims(self):
        self.assertEqual(len(OBS_NAMES), BASE_OBS_DIM)


if __name__ == "__main__":
    unittest.main()
