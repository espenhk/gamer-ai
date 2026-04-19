"""Tests for _discretize_obs helper in tmnf/policies.py."""
import unittest

import numpy as np

from games.tmnf.obs_spec import BASE_OBS_DIM
from policies import WeightedLinearPolicy, _discretize_obs

_N = BASE_OBS_DIM


class TestDiscretizeObs(unittest.TestCase):

    def _scales(self) -> np.ndarray:
        return WeightedLinearPolicy.OBS_SCALES

    def test_zero_obs_maps_to_middle_bin(self):
        # Zero normalised → middle of [-3, 3] range → bin 1 with n_bins=3
        bins = _discretize_obs(np.zeros(_N, dtype=np.float32), self._scales(), n_bins=3)
        self.assertTrue(all(b == 1 for b in bins))

    def test_clipped_high_obs_maps_to_max_bin(self):
        bins = _discretize_obs(np.full(_N, 1e9, dtype=np.float32), self._scales(), n_bins=3)
        self.assertTrue(all(b == 2 for b in bins))

    def test_clipped_low_obs_maps_to_min_bin(self):
        bins = _discretize_obs(np.full(_N, -1e9, dtype=np.float32), self._scales(), n_bins=3)
        self.assertTrue(all(b == 0 for b in bins))

    def test_symmetry(self):
        # +scale → norm=+1, -scale → norm=-1; should be symmetric around the middle bin
        scales = self._scales()
        pos_bins = _discretize_obs( scales.copy(), scales, n_bins=7)
        neg_bins = _discretize_obs(-scales.copy(), scales, n_bins=7)
        mid = 3
        for p, n in zip(pos_bins, neg_bins):
            self.assertEqual(p + n, 2 * mid)

    def test_returns_tuple_of_ints(self):
        bins = _discretize_obs(np.zeros(_N, dtype=np.float32), self._scales(), n_bins=3)
        self.assertIsInstance(bins, tuple)
        self.assertTrue(all(isinstance(b, int) for b in bins))

    def test_length_matches_obs_dim(self):
        obs = np.zeros(_N, dtype=np.float32)
        bins = _discretize_obs(obs, self._scales(), n_bins=3)
        self.assertEqual(len(bins), _N)


if __name__ == "__main__":
    unittest.main(verbosity=2)
