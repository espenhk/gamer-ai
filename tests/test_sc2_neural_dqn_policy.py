"""Tests for SC2NeuralDQNPolicy — available-actions masking and helper utilities.

All tests run without PySC2 installed; observations are fabricated numpy arrays
and available_fn_ids are set directly as FUNCTION_IDS key sets (0-5).
"""
from __future__ import annotations

import unittest

import numpy as np

from games.sc2.actions import (
    DISCRETE_ACTIONS,
    FUNCTION_IDS,
    build_available_actions_mask,
    discrete_action_to_fn_id,
)

from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC
from games.sc2.sc2_policies import SC2NeuralDQNPolicy

_OBS_SPEC = SC2_MINIGAME_OBS_SPEC
_OBS_DIM = SC2_MINIGAME_OBS_SPEC.dim
_N = len(DISCRETE_ACTIONS)


def _make_policy(**kw) -> SC2NeuralDQNPolicy:
    defaults = dict(
        obs_spec=SC2_MINIGAME_OBS_SPEC,
        hidden_sizes=[16, 16],
        replay_buffer_size=500,
        batch_size=16,
        min_replay_size=32,
        target_update_freq=20,
        learning_rate=0.01,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        gamma=0.99,
    )
    defaults.update(kw)
    return SC2NeuralDQNPolicy(**defaults)


def _zero_obs() -> np.ndarray:
    return np.zeros(_OBS_DIM, dtype=np.float32)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestActionMaskingHelpers(unittest.TestCase):

    def test_discrete_action_to_fn_id_row1_is_select_army(self):
        # Row 1 is select_army (fn_idx=1) in the new layout
        self.assertEqual(discrete_action_to_fn_id(1), 1)

    def test_discrete_action_to_fn_id_row0_is_no_op(self):
        # Row 0 is no_op (fn_idx=0)
        self.assertEqual(discrete_action_to_fn_id(0), 0)

    def test_discrete_action_to_fn_id_other_cells_cover_discrete_actions(self):
        # Each row maps to a valid fn_idx from FUNCTION_IDS
        for i in range(_N):
            fn_id = discrete_action_to_fn_id(i)
            self.assertIn(fn_id, FUNCTION_IDS)

    def test_build_mask_all_fn_ids_available(self):
        # All fn_ids available → all rows legal
        all_fn_ids = set(FUNCTION_IDS.keys())
        mask = build_available_actions_mask(all_fn_ids)
        self.assertEqual(mask.shape, (_N,))
        self.assertTrue(mask.all())

    def test_build_mask_only_specific_fn_available(self):
        # Only fn_idx=0 (no_op) available: row 0 must be True, rows with
        # different fn_ids must be False.
        mask = build_available_actions_mask({0})
        self.assertTrue(mask[0], "no_op (row 0) should be available when fn_idx=0 available")
        # At least some rows should be masked out
        self.assertFalse(mask.all(), "Not all rows should be legal with only one fn_id")

    def test_build_mask_empty_set_all_false(self):
        mask = build_available_actions_mask(set())
        self.assertFalse(mask.any())


# ---------------------------------------------------------------------------
# Masked Q-values → illegal action never chosen
# ---------------------------------------------------------------------------

class TestMaskedActionSelection(unittest.TestCase):

    def test_greedy_never_selects_masked_action(self):
        """With only fn_idx=2 available, only fn_idx=2 actions should be selected."""
        policy = _make_policy(epsilon_start=0.0, epsilon_end=0.0)
        # Only fn_idx=2 available — mask out everything else
        policy._cached_mask = build_available_actions_mask({2})
        obs = _zero_obs()
        for _ in range(50):
            action = policy(obs)
            fn_idx = int(action[0])
            self.assertEqual(fn_idx, 2,
                "only fn_idx=2 should be selected when others are masked")

    def test_random_never_selects_masked_action(self):
        """ε=1 random exploration must also respect the mask."""
        policy = _make_policy(epsilon_start=1.0, epsilon_end=1.0)
        policy._cached_mask = build_available_actions_mask({2})  # only fn_idx=2
        obs = _zero_obs()
        for _ in range(100):
            action = policy(obs)
            self.assertEqual(int(action[0]), 2,
                "random exploration must only pick fn_idx=2 when others are masked")

    def test_no_mask_selects_any_action(self):
        """Without a mask (all-True) multiple fn_idx values can be selected."""
        policy = _make_policy(epsilon_start=1.0, epsilon_end=1.0, seed=0)
        policy._cached_mask = np.ones(_N, dtype=bool)
        seen_fn_ids = set()
        obs = _zero_obs()
        for _ in range(1000):
            action = policy(obs)
            seen_fn_ids.add(int(action[0]))
        # Should see at least 3 distinct fn_idx values (from 118 total)
        self.assertGreaterEqual(len(seen_fn_ids), 3)

    def test_on_episode_start_without_info_resets_to_all_true(self):
        """on_episode_start() without info should reset to an all-True mask."""
        policy = _make_policy()
        policy._cached_mask = build_available_actions_mask({2})  # partially masked
        policy.on_episode_start()
        self.assertTrue(policy._cached_mask.all())

    def test_on_episode_start_with_info_primes_mask(self):
        """on_episode_start(info=...) should prime cached mask from available_fn_ids."""
        policy = _make_policy()
        policy.on_episode_start(info={"available_fn_ids": {2}})
        np.testing.assert_array_equal(policy._cached_mask, build_available_actions_mask({2}))


# ---------------------------------------------------------------------------
# available_fn_ids stored from update() info kwarg
# ---------------------------------------------------------------------------

class TestUpdateStoresAvailableFnIds(unittest.TestCase):

    def test_initial_mask_is_all_true(self):
        """Freshly constructed policy must start with an all-True mask."""
        policy = _make_policy()
        self.assertTrue(policy._cached_mask.all())

    def test_update_sets_cached_mask(self):
        """update() with info['available_fn_ids'] must update _cached_mask."""
        policy = _make_policy()
        obs = _zero_obs()
        policy.update(obs, 0, 1.0, obs, False, info={"available_fn_ids": {1, 2}})
        expected = build_available_actions_mask({1, 2})
        np.testing.assert_array_equal(policy._cached_mask, expected)

    def test_update_without_info_resets_to_all_true_mask(self):
        """update() without info kwarg should reset mask to all-True."""
        policy = _make_policy()
        partial_mask = build_available_actions_mask({2})
        policy._cached_mask = partial_mask.copy()
        obs = _zero_obs()
        policy.update(obs, 0, 1.0, obs, False)  # no info kwarg
        self.assertTrue(policy._cached_mask.all())

    def test_update_with_empty_info_resets_to_all_true_mask(self):
        """update() with empty info dict should reset mask to all-True."""
        policy = _make_policy()
        partial_mask = build_available_actions_mask({2})
        policy._cached_mask = partial_mask.copy()
        obs = _zero_obs()
        policy.update(obs, 0, 1.0, obs, False, info={})
        self.assertTrue(policy._cached_mask.all())

    def test_update_overwrites_mask(self):
        """Consecutive update() calls with different fn_ids update the mask each time."""
        policy = _make_policy()
        obs = _zero_obs()
        policy.update(obs, 0, 1.0, obs, False, info={"available_fn_ids": {1}})
        expected_first = build_available_actions_mask({1})
        np.testing.assert_array_equal(policy._cached_mask, expected_first)

        policy.update(obs, 0, 1.0, obs, False, info={"available_fn_ids": {2}})
        expected_second = build_available_actions_mask({2})
        np.testing.assert_array_equal(policy._cached_mask, expected_second)


# ---------------------------------------------------------------------------
# Gradient does not flow through masked logits
# ---------------------------------------------------------------------------

class TestMaskedGradientStep(unittest.TestCase):

    def test_masked_action_q_value_not_maximised(self):
        """Train where cell 2 (first Move_screen) gives +5 but fn_idx=0,1 are masked.
        After training with only fn_idx=2 available, the greedy action among
        legal cells must be the rewarded Move_screen cell."""
        np.random.seed(42)
        policy = SC2NeuralDQNPolicy(
            obs_spec=SC2_MINIGAME_OBS_SPEC,
            hidden_sizes=[32, 32],
            replay_buffer_size=5000,
            batch_size=32,
            min_replay_size=128,
            target_update_freq=25,
            learning_rate=0.005,
            epsilon_start=1.0,
            epsilon_end=1.0,  # pure replay; no online exploration
            epsilon_decay_steps=1,
            gamma=0.0,
        )

        obs = _zero_obs()
        next_obs = _zero_obs()
        BEST_LEGAL = 2  # first Move_screen cell → fn_idx=2 (legal when {2} available)
        for step in range(8000):
            action_idx = step % _N
            reward = 5.0 if action_idx == BEST_LEGAL else -0.1
            policy.update(obs, action_idx, reward, next_obs, done=True,
                          info={"available_fn_ids": {2}})

        policy._eps = 0.0
        obs_norm = (obs / policy._scales).astype(np.float32)
        q = policy._q_values(policy._online, obs_norm).copy()
        q[~build_available_actions_mask({2})] = -np.inf
        greedy = int(np.argmax(q))
        self.assertEqual(greedy, BEST_LEGAL,
            f"Expected greedy={BEST_LEGAL}, got {greedy}. Q={q.tolist()}")

    def test_policy_type_in_cfg(self):
        policy = _make_policy()
        self.assertEqual(policy.to_cfg()["policy_type"], "sc2_neural_dqn")


if __name__ == "__main__":
    unittest.main(verbosity=2)
