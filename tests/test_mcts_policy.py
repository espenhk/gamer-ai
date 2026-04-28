"""Tests for MCTSPolicy in tmnf/policies.py."""
import unittest

import numpy as np

from games.tmnf.obs_spec import BASE_OBS_DIM
from games.tmnf.policies import MCTSPolicy, WeightedLinearPolicy, _action_to_idx, _discretize_obs


def _zero_obs() -> np.ndarray:
    return np.zeros(BASE_OBS_DIM, dtype=np.float32)


def _state_key(obs: np.ndarray) -> tuple:
    return _discretize_obs(obs, WeightedLinearPolicy.OBS_SCALES, n_bins=3)


class TestMCTSPolicy(unittest.TestCase):

    def test_action_in_range(self):
        p = MCTSPolicy()
        self.assertIn(_action_to_idx(p(_zero_obs())), range(9))

    def test_unseen_state_uses_random(self):
        # With an empty table every call is random — expect multiple distinct actions over many calls.
        # Probability of all 30 calls picking the same action: 9^{-29} ≈ 0.
        p = MCTSPolicy()
        actions = {_action_to_idx(p(np.random.randn(BASE_OBS_DIM).astype(np.float32))) for _ in range(30)}
        self.assertGreater(len(actions), 1)

    def test_update_increments_visit_count(self):
        p = MCTSPolicy()
        obs = _zero_obs()
        p.update(obs, action=4, reward=1.0, next_obs=_zero_obs(), done=False)
        key = _state_key(obs)
        self.assertEqual(p._n_sa[key][4], 1.0)
        self.assertEqual(p._n_s[key], 1)

    def test_update_changes_q_value(self):
        p = MCTSPolicy(alpha=1.0, gamma=0.0)
        obs = _zero_obs()
        p.update(obs, action=2, reward=7.0, next_obs=_zero_obs(), done=True)
        self.assertAlmostEqual(float(p._q_table[_state_key(obs)][2]), 7.0, places=4)

    def test_exploitation_prefers_high_q_action(self):
        # c=0 disables exploration bonus → pure exploitation via Q-values
        p = MCTSPolicy(c=0.0)
        obs = _zero_obs()
        next_obs = _zero_obs()
        for _ in range(5):
            p.update(obs, action=3, reward= 100.0, next_obs=next_obs, done=True)
            p.update(obs, action=0, reward=-100.0, next_obs=next_obs, done=True)
        self.assertEqual(_action_to_idx(p(obs)), 3)

    def test_multiple_updates_accumulate_visits(self):
        p = MCTSPolicy()
        obs = _zero_obs()
        for _ in range(4):
            p.update(obs, action=1, reward=1.0, next_obs=_zero_obs(), done=False)
        key = _state_key(obs)
        self.assertEqual(p._n_sa[key][1], 4.0)
        self.assertEqual(p._n_s[key], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
