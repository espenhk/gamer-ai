"""Tests for EpsilonGreedyPolicy in tmnf/policies.py."""
import unittest

import numpy as np

from games.tmnf.obs_spec import BASE_OBS_DIM
from games.tmnf.policies import EpsilonGreedyPolicy, WeightedLinearPolicy, _action_to_idx, _discretize_obs


def _zero_obs() -> np.ndarray:
    return np.zeros(BASE_OBS_DIM, dtype=np.float32)


def _state_key(obs: np.ndarray) -> tuple:
    return _discretize_obs(obs, WeightedLinearPolicy.OBS_SCALES, n_bins=3)


class TestEpsilonGreedyPolicy(unittest.TestCase):

    def test_action_in_range(self):
        p = EpsilonGreedyPolicy(epsilon=1.0)
        self.assertIn(_action_to_idx(p(_zero_obs())), range(9))

    def test_greedy_picks_best_q_action(self):
        p = EpsilonGreedyPolicy(epsilon=0.0)
        obs = _zero_obs()
        key = _state_key(obs)
        p._q_table[key] = np.zeros(9, dtype=np.float32)
        p._q_table[key][1] = 10.0
        for _ in range(10):
            self.assertEqual(_action_to_idx(p(obs)), 1)

    def test_update_increases_q_for_positive_reward(self):
        p = EpsilonGreedyPolicy(epsilon=0.0, alpha=0.5, gamma=0.0)
        obs = _zero_obs()
        p.update(obs, action=3, reward=10.0, next_obs=_zero_obs(), done=True)
        self.assertGreater(p._q_table[_state_key(obs)][3], 0.0)

    def test_update_decreases_q_for_negative_reward(self):
        p = EpsilonGreedyPolicy(epsilon=0.0, alpha=0.5, gamma=0.0)
        obs = _zero_obs()
        p.update(obs, action=5, reward=-10.0, next_obs=_zero_obs(), done=True)
        self.assertLess(p._q_table[_state_key(obs)][5], 0.0)

    def test_bellman_backup(self):
        alpha, gamma = 0.5, 0.9
        p = EpsilonGreedyPolicy(epsilon=0.0, alpha=alpha, gamma=gamma)
        obs      = _zero_obs()
        next_obs = np.ones(BASE_OBS_DIM, dtype=np.float32)
        s  = _state_key(obs)
        s_ = _state_key(next_obs)
        # Seed Q(s', 2) = 5 → max_Q(s') = 5
        p._q_table[s_] = np.zeros(9, dtype=np.float32)
        p._q_table[s_][2] = 5.0
        p.update(obs, action=0, reward=2.0, next_obs=next_obs, done=False)
        # Expected: Q(s,0) += 0.5 * (2 + 0.9*5 - 0) = 0.5 * 6.5 = 3.25
        self.assertAlmostEqual(float(p._q_table[s][0]), 3.25, places=4)

    def test_epsilon_decays_on_episode_end(self):
        p = EpsilonGreedyPolicy(epsilon=1.0, epsilon_decay=0.5, epsilon_min=0.0)
        p.on_episode_end()
        self.assertAlmostEqual(p._epsilon, 0.5)

    def test_epsilon_floored_at_minimum(self):
        p = EpsilonGreedyPolicy(epsilon=0.01, epsilon_decay=0.01, epsilon_min=0.05)
        p.on_episode_end()
        self.assertGreaterEqual(p._epsilon, 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
