"""Tests for the Rocket League observation spec."""
import unittest

import numpy as np

from games.rocket_league.obs_spec import (
    ROCKET_LEAGUE_OBS_SPEC,
    BASE_OBS_DIM,
    OBS_NAMES,
    OBS_SCALES,
    OBS_SPEC,
)
from framework.obs_spec import ObsDim, ObsSpec


class TestRocketLeagueObsSpec(unittest.TestCase):
    """Validate ROCKET_LEAGUE_OBS_SPEC structure and derived constants."""

    def test_is_obs_spec_instance(self):
        self.assertIsInstance(ROCKET_LEAGUE_OBS_SPEC, ObsSpec)

    def test_dim_matches_base(self):
        self.assertEqual(ROCKET_LEAGUE_OBS_SPEC.dim, BASE_OBS_DIM)

    def test_dim_is_142(self):
        self.assertEqual(BASE_OBS_DIM, 142)

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

    def test_first_feature_is_car_pos_x(self):
        self.assertEqual(OBS_NAMES[0], "car_pos_x")

    def test_boost_amount_present(self):
        self.assertIn("boost_amount", OBS_NAMES)

    def test_ball_features_present(self):
        for name in ("ball_pos_x", "ball_pos_y", "ball_pos_z",
                     "ball_vel_x", "ball_vel_y", "ball_vel_z"):
            self.assertIn(name, OBS_NAMES)

    def test_opponent_features_present(self):
        for name in ("opp1_pos_x", "opp2_pos_x", "opp3_pos_x"):
            self.assertIn(name, OBS_NAMES)

    def test_friendly_features_present(self):
        for name in ("mate1_pos_x", "mate2_pos_x"):
            self.assertIn(name, OBS_NAMES)

    def test_relative_features_present(self):
        for name in ("rel_ball_pos_x", "dist_to_ball", "vel_towards_ball",
                     "ball_to_opp_goal_dist", "car_to_own_goal_dist"):
            self.assertIn(name, OBS_NAMES)

    def test_boost_pad_features_present(self):
        pad_names = [n for n in OBS_NAMES if n.startswith("boost_pad_")]
        self.assertEqual(len(pad_names), 10)

    def test_self_car_features_are_indices_0_to_17(self):
        """Car state (self) occupies indices 0–17."""
        self.assertEqual(OBS_NAMES[0], "car_pos_x")
        self.assertEqual(OBS_NAMES[17], "boost_amount")

    def test_ball_features_are_indices_18_to_26(self):
        self.assertEqual(OBS_NAMES[18], "ball_pos_x")
        self.assertEqual(OBS_NAMES[26], "ball_ang_vel_z")

    def test_teammate_features_are_indices_27_to_62(self):
        self.assertEqual(OBS_NAMES[27], "mate1_pos_x")
        self.assertEqual(OBS_NAMES[62], "mate2_boost")

    def test_opponent_features_are_indices_63_to_116(self):
        self.assertEqual(OBS_NAMES[63], "opp1_pos_x")
        self.assertEqual(OBS_NAMES[116], "opp3_boost")

    def test_boost_pad_features_are_last_10(self):
        self.assertEqual(OBS_NAMES[132], "boost_pad_0")
        self.assertEqual(OBS_NAMES[141], "boost_pad_9")

    def test_with_zero_lidar_returns_same(self):
        spec = ROCKET_LEAGUE_OBS_SPEC.with_lidar(0)
        self.assertIs(spec, ROCKET_LEAGUE_OBS_SPEC)

    def test_with_lidar_extends_dim(self):
        spec = ROCKET_LEAGUE_OBS_SPEC.with_lidar(5)
        self.assertEqual(spec.dim, BASE_OBS_DIM + 5)


if __name__ == "__main__":
    unittest.main()
