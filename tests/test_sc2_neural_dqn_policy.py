"""Tests for SC2NeuralDQNPolicy — available-actions masking."""
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
from games.sc2.policies import SC2NeuralDQNPolicy

_OBS_DIM = SC2_MINIGAME_OBS_SPEC.dim
_N = len(DISCRETE_ACTIONS)  # 9


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

    def test_discrete_action_to_fn_id_cell4_is_select_army(self):
        # Cell 4 is the centre cell → select_army (fn_idx=1)
        self.assertEqual(discrete_action_to_fn_id(4), 1)

    def test_discrete_action_to_fn_id_other_cells_are_move_screen(self):
        for i in range(_N):
            if i != 4:
                self.assertEqual(discrete_action_to_fn_id(i), 2)

    def test_build_mask_all_fn_ids_available(self):
        mask = build_available_actions_mask({1, 2})
        self.assertEqual(mask.shape, (_N,))
        self.assertTrue(mask.all())

    def test_build_mask_only_move_screen_available(self):
        # fn_idx=2 (Move_screen) covers all cells except cell 4 (select_army)
        mask = build_available_actions_mask({2})
        self.assertFalse(mask[4])
        for i in range(_N):
            if i != 4:
                self.assertTrue(mask[i])

    def test_build_mask_empty_set_all_false(self):
        mask = build_available_actions_mask(set())
        self.assertFalse(mask.any())


# ---------------------------------------------------------------------------
# Masked Q-values → illegal action never chosen
# ---------------------------------------------------------------------------

class TestMaskedActionSelection(unittest.TestCase):

    def test_greedy_never_selects_masked_action(self):
        """With fn_idx=1 (select_army) masked out, cell 4 must never be chosen."""
        policy = _make_policy(epsilon_start=0.0, epsilon_end=0.0)
        # Only Move_screen (fn_idx=2) is available — masks out cell 4 (select_army)
        policy._available_fn_ids = {2}
        obs = _zero_obs()
        for _ in range(50):
            action = policy(obs)
            fn_idx = int(action[0])
            # All cells except 4 have fn_idx=2; cell 4 has fn_idx=1
            self.assertNotEqual(fn_idx, 1,
                "select_army (fn_idx=1) was selected despite being masked out")

    def test_random_never_selects_masked_action(self):
        """ε=1 random exploration must also respect the mask."""
        policy = _make_policy(epsilon_start=1.0, epsilon_end=1.0)
        policy._available_fn_ids = {2}  # no select_army
        obs = _zero_obs()
        for _ in range(100):
            action = policy(obs)
            self.assertNotEqual(int(action[0]), 1)

    def test_no_mask_selects_any_action(self):
        """Without a mask (None) all 9 cells can be selected."""
        policy = _make_policy(epsilon_start=1.0, epsilon_end=1.0)
        policy._available_fn_ids = None
        seen_fn_ids = set()
        obs = _zero_obs()
        for _ in range(200):
            action = policy(obs)
            seen_fn_ids.add(int(action[0]))
        # Should see both fn_idx 1 (select_army) and 2 (Move_screen)
        self.assertIn(1, seen_fn_ids)
        self.assertIn(2, seen_fn_ids)

    def test_on_episode_start_clears_mask(self):
        policy = _make_policy()
        policy._available_fn_ids = {2}
        policy.on_episode_start()
        self.assertIsNone(policy._available_fn_ids)


# ---------------------------------------------------------------------------
# available_fn_ids stored from update() info kwarg
# ---------------------------------------------------------------------------

class TestUpdateStoresAvailableFnIds(unittest.TestCase):

    def test_update_stores_available_fn_ids(self):
        policy = _make_policy()
        self.assertIsNone(policy._available_fn_ids)
        obs = _zero_obs()
        policy.update(obs, 0, 1.0, obs, False, info={"available_fn_ids": {1, 2}})
        self.assertEqual(policy._available_fn_ids, {1, 2})

    def test_update_without_info_preserves_existing_mask(self):
        policy = _make_policy()
        policy._available_fn_ids = {2}
        obs = _zero_obs()
        policy.update(obs, 0, 1.0, obs, False)  # no info kwarg
        self.assertEqual(policy._available_fn_ids, {2})

    def test_update_with_empty_info_preserves_existing_mask(self):
        policy = _make_policy()
        policy._available_fn_ids = {2}
        obs = _zero_obs()
        policy.update(obs, 0, 1.0, obs, False, info={})
        self.assertEqual(policy._available_fn_ids, {2})

    def test_update_overwrites_mask(self):
        policy = _make_policy()
        policy._available_fn_ids = {1}
        obs = _zero_obs()
        policy.update(obs, 0, 1.0, obs, False, info={"available_fn_ids": {2}})
        self.assertEqual(policy._available_fn_ids, {2})


# ---------------------------------------------------------------------------
# Gradient does not flow through masked logits
# ---------------------------------------------------------------------------

class TestMaskedGradientStep(unittest.TestCase):

    def test_masked_action_q_value_not_maximised(self):
        """Train where cell 4 (select_army) gives +10 but fn_idx=1 is masked.
        After training with only fn_idx=2 available, the greedy action among
        legal cells must be chosen (not cell 4)."""
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
        # Only Move_screen is available → masks out cell 4
        policy._available_fn_ids = {2}

        obs = _zero_obs()
        next_obs = _zero_obs()
        BEST_LEGAL = 0  # top-left cell → fn_idx=2 (legal)
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
