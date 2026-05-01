"""Tests for framework/info_gain.py — RegionStalenessTracker module."""
import unittest

import numpy as np

from framework.info_gain import RegionStalenessTracker


class TestRegionStalenessNeverObserved(unittest.TestCase):
    """Never-observed slots should report maximum staleness."""

    def setUp(self):
        self.tracker = RegionStalenessTracker(
            n_rows=2, n_cols=2,
            scout_horizon_s=60.0,
            stale_threshold=0.5,
            scout_drive_weight=0.1,
            never_seen_bonus=2.0,
        )

    def test_initial_staleness_all_ones(self):
        s = self.tracker.staleness()
        np.testing.assert_array_equal(s, np.ones(4, dtype=np.float64))

    def test_never_observed_is_maximum(self):
        """Never-observed slots produce staleness 1.0 (the maximum)."""
        s = self.tracker.staleness()
        self.assertEqual(s.max(), 1.0)
        self.assertTrue(np.all(s == 1.0))


class TestRegionStalenessRecentlyObserved(unittest.TestCase):
    """Recently-observed slots should report near-zero staleness."""

    def setUp(self):
        self.tracker = RegionStalenessTracker(
            n_rows=2, n_cols=2,
            scout_horizon_s=60.0,
            stale_threshold=0.5,
            scout_drive_weight=0.1,
        )

    def test_just_observed_near_zero(self):
        visible = np.array([True, False, False, False])
        self.tracker.update(np.zeros(4), {"time_s": 0.0, "visible_slots": visible})
        s = self.tracker.staleness()
        self.assertAlmostEqual(s[0], 0.0)
        # Unvisited slots remain at 1.0
        self.assertAlmostEqual(s[1], 1.0)
        self.assertAlmostEqual(s[2], 1.0)
        self.assertAlmostEqual(s[3], 1.0)

    def test_staleness_grows_linearly(self):
        visible = np.array([True, False, False, False])
        self.tracker.update(np.zeros(4), {"time_s": 0.0, "visible_slots": visible})
        # Advance time to half the horizon
        self.tracker.update(np.zeros(4), {"time_s": 30.0, "visible_slots": np.zeros(4, dtype=bool)})
        s = self.tracker.staleness()
        self.assertAlmostEqual(s[0], 0.5, places=5)


class TestIntrinsicRewardFires(unittest.TestCase):
    """Intrinsic reward should fire on stale→fresh transitions."""

    def setUp(self):
        self.tracker = RegionStalenessTracker(
            n_rows=2, n_cols=2,
            scout_horizon_s=60.0,
            stale_threshold=0.5,
            scout_drive_weight=1.0,  # weight=1 for easy assertion
            never_seen_bonus=2.0,
        )

    def test_reward_fires_on_stale_to_fresh(self):
        # First observation at t=0, slot 0 visible
        visible_0 = np.array([True, False, False, False])
        self.tracker.update(np.zeros(4), {"time_s": 0.0, "visible_slots": visible_0})
        _ = self.tracker.intrinsic_reward()  # consume any reward from initial

        # Advance time so slot 0 becomes stale (> threshold)
        self.tracker.update(np.zeros(4), {"time_s": 40.0, "visible_slots": np.zeros(4, dtype=bool)})
        self.assertAlmostEqual(self.tracker.intrinsic_reward(), 0.0)

        # Re-observe slot 0 → stale→fresh transition
        self.tracker.update(np.zeros(4), {"time_s": 40.0, "visible_slots": visible_0})
        r = self.tracker.intrinsic_reward()
        self.assertGreater(r, 0.0)

    def test_reward_zero_when_weight_zero(self):
        tracker = RegionStalenessTracker(
            n_rows=2, n_cols=2,
            scout_drive_weight=0.0,
        )
        visible = np.array([True, False, False, False])
        tracker.update(np.zeros(4), {"time_s": 0.0, "visible_slots": visible})
        self.assertAlmostEqual(tracker.intrinsic_reward(), 0.0)


class TestRegionStalenessReset(unittest.TestCase):
    """Reset should clear all state."""

    def test_reset_restores_initial_staleness(self):
        tracker = RegionStalenessTracker(n_rows=2, n_cols=2)
        visible = np.ones(4, dtype=bool)
        tracker.update(np.zeros(4), {"time_s": 0.0, "visible_slots": visible})
        tracker.reset()
        s = tracker.staleness()
        np.testing.assert_array_equal(s, np.ones(4, dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
