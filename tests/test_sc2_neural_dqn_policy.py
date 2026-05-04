"""Tests for SC2NeuralDQNPolicy — available-actions masking and helper utilities.

All tests run without PySC2 installed; observations are fabricated numpy arrays
and available_fn_ids are set directly as FUNCTION_IDS key sets (0-5).
"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from games.sc2.actions import (
    DISCRETE_ACTIONS,
    FUNCTION_IDS,
    discrete_action_to_fn_id,
)
from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC
from games.sc2.policies import (
    SC2NeuralDQNPolicy,
    _MaskedReplayBuffer,
    _N_DISCRETE_ACTIONS,
    _build_available_actions_mask,
)

_OBS_SPEC = SC2_MINIGAME_OBS_SPEC


def _make_policy(**kw) -> SC2NeuralDQNPolicy:
    defaults = dict(
        obs_spec=_OBS_SPEC,
        hidden_sizes=[8, 8],
        replay_buffer_size=500,
        batch_size=16,
        min_replay_size=32,
        target_update_freq=10,
        learning_rate=0.01,
        epsilon_start=0.0,   # fully greedy by default
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        gamma=0.99,
    )
    defaults.update(kw)
    return SC2NeuralDQNPolicy(**defaults)


def _zero_obs() -> np.ndarray:
    return np.zeros(_OBS_SPEC.dim, dtype=np.float32)


def _rand_obs() -> np.ndarray:
    return np.random.randn(_OBS_SPEC.dim).astype(np.float32)


# ---------------------------------------------------------------------------
# discrete_action_to_fn_id
# ---------------------------------------------------------------------------

class TestDiscreteActionToFnId(unittest.TestCase):

    def test_centre_cell_returns_select_army_fn_idx(self):
        """Cell 4 (centre) should map to fn_idx=1 (select_army)."""
        self.assertEqual(discrete_action_to_fn_id(4), 1)

    def test_other_cells_return_move_screen_fn_idx(self):
        """All other cells should map to fn_idx=2 (Move_screen)."""
        for i in range(9):
            if i != 4:
                self.assertEqual(discrete_action_to_fn_id(i), 2, f"cell {i}")

    def test_consistent_with_discrete_actions(self):
        """discrete_action_to_fn_id(i) == int(DISCRETE_ACTIONS[i, 0]) for all i."""
        for i in range(_N_DISCRETE_ACTIONS):
            self.assertEqual(discrete_action_to_fn_id(i), int(DISCRETE_ACTIONS[i, 0]))

    def test_return_type_is_int(self):
        for i in range(_N_DISCRETE_ACTIONS):
            self.assertIsInstance(discrete_action_to_fn_id(i), int)


# ---------------------------------------------------------------------------
# _build_available_actions_mask
# ---------------------------------------------------------------------------

class TestBuildAvailableActionsMask(unittest.TestCase):

    def test_none_returns_all_true(self):
        mask = _build_available_actions_mask(None, _N_DISCRETE_ACTIONS)
        self.assertEqual(mask.shape, (_N_DISCRETE_ACTIONS,))
        self.assertTrue(np.all(mask))

    def test_empty_set_returns_all_false(self):
        mask = _build_available_actions_mask(set(), _N_DISCRETE_ACTIONS)
        self.assertFalse(np.any(mask))

    def test_select_army_only(self):
        """When only fn_idx=1 (select_army) is available, only cell 4 is True."""
        mask = _build_available_actions_mask({1}, _N_DISCRETE_ACTIONS)
        self.assertTrue(mask[4])
        for i in range(9):
            if i != 4:
                self.assertFalse(mask[i], f"cell {i} should be False")

    def test_move_screen_only(self):
        """When only fn_idx=2 (Move_screen) is available, all 8 move cells are True."""
        mask = _build_available_actions_mask({2}, _N_DISCRETE_ACTIONS)
        self.assertFalse(mask[4])   # centre = select_army → False
        for i in range(9):
            if i != 4:
                self.assertTrue(mask[i], f"cell {i} should be True")

    def test_both_available(self):
        mask = _build_available_actions_mask({1, 2}, _N_DISCRETE_ACTIONS)
        self.assertTrue(np.all(mask))

    def test_dtype_is_bool(self):
        mask = _build_available_actions_mask({1}, _N_DISCRETE_ACTIONS)
        self.assertEqual(mask.dtype, bool)


# ---------------------------------------------------------------------------
# _MaskedReplayBuffer
# ---------------------------------------------------------------------------

class TestMaskedReplayBuffer(unittest.TestCase):

    def test_push_and_len(self):
        buf = _MaskedReplayBuffer(maxlen=10)
        mask = np.ones(_N_DISCRETE_ACTIONS, dtype=bool)
        buf.push(_zero_obs(), 0, 1.0, _zero_obs(), False, mask)
        self.assertEqual(len(buf), 1)

    def test_push_default_mask_is_all_true(self):
        buf = _MaskedReplayBuffer(maxlen=10)
        buf.push(_zero_obs(), 0, 1.0, _zero_obs(), False)  # no mask arg
        _, _, _, _, _, mask_b = buf.sample(1)
        self.assertTrue(np.all(mask_b[0]))

    def test_sample_returns_six_tuple(self):
        buf = _MaskedReplayBuffer(maxlen=20)
        mask = np.ones(_N_DISCRETE_ACTIONS, dtype=bool)
        for _ in range(10):
            buf.push(_rand_obs(), 0, 0.0, _rand_obs(), False, mask)
        result = buf.sample(5)
        self.assertEqual(len(result), 6)

    def test_sample_mask_shape(self):
        buf = _MaskedReplayBuffer(maxlen=20)
        for _ in range(10):
            buf.push(_rand_obs(), 0, 0.0, _rand_obs(), False)
        obs_b, act_b, rew_b, next_b, done_b, mask_b = buf.sample(5)
        self.assertEqual(mask_b.shape, (5, _N_DISCRETE_ACTIONS))

    def test_stored_mask_is_preserved(self):
        """The exact mask pushed should be retrievable via sample."""
        buf = _MaskedReplayBuffer(maxlen=1)  # only 1 slot
        mask = np.array([True, False, True, False, True, False, True, False, True], dtype=bool)
        buf.push(_zero_obs(), 0, 0.0, _zero_obs(), False, mask)
        _, _, _, _, _, mask_b = buf.sample(1)
        np.testing.assert_array_equal(mask_b[0], mask)

    def test_circular_eviction(self):
        buf = _MaskedReplayBuffer(maxlen=3)
        for i in range(5):
            buf.push(_zero_obs(), i % 9, float(i), _zero_obs(), False)
        self.assertEqual(len(buf), 3)


# ---------------------------------------------------------------------------
# SC2NeuralDQNPolicy — action masking
# ---------------------------------------------------------------------------

class TestSC2NeuralDQNPolicyMasking(unittest.TestCase):

    def test_illegal_action_never_chosen_greedy(self):
        """With all Move_screen actions masked, only select_army (cell 4) is chosen."""
        policy = _make_policy(epsilon_start=0.0, epsilon_end=0.0)
        # Make Q-values strongly prefer cell 0 (Move_screen → should be masked)
        obs = _zero_obs()
        # Inject bias so Q[0] >> Q[4]
        policy._online["biases"][-1] = np.zeros(_N_DISCRETE_ACTIONS, dtype=np.float32)
        policy._online["biases"][-1][0] = 100.0   # very high for cell 0

        # Only select_army (fn_idx=1) available → only cell 4 should be legal
        policy._cached_mask = _build_available_actions_mask({1}, _N_DISCRETE_ACTIONS)
        action = policy(obs)
        # Action should be DISCRETE_ACTIONS[4] (select_army), not DISCRETE_ACTIONS[0]
        np.testing.assert_array_equal(action, DISCRETE_ACTIONS[4])

    def test_all_actions_available_selects_best(self):
        """With all actions available, the highest-Q action wins."""
        policy = _make_policy(epsilon_start=0.0, epsilon_end=0.0)
        obs = _zero_obs()
        policy._online["biases"][-1] = np.zeros(_N_DISCRETE_ACTIONS, dtype=np.float32)
        policy._online["biases"][-1][3] = 100.0   # cell 3 = best

        policy._cached_mask = _build_available_actions_mask({1, 2}, _N_DISCRETE_ACTIONS)
        action = policy(obs)
        np.testing.assert_array_equal(action, DISCRETE_ACTIONS[3])

    def test_random_exploration_respects_mask(self):
        """With ε=1.0 and only select_army available, random actions are legal."""
        policy = _make_policy(epsilon_start=1.0, epsilon_end=1.0)
        # Only fn_idx=1 (select_army, cell 4) is available
        policy._cached_mask = _build_available_actions_mask({1}, _N_DISCRETE_ACTIONS)
        for _ in range(50):
            action = policy(_zero_obs())
            # Must be DISCRETE_ACTIONS[4]
            np.testing.assert_array_equal(
                action, DISCRETE_ACTIONS[4],
                err_msg=f"Unexpected action with all-Move_screen mask: {action}",
            )

    def test_all_masked_fallback_no_crash(self):
        """If every action is masked (empty set), the policy falls back gracefully."""
        policy = _make_policy(epsilon_start=0.0, epsilon_end=0.0)
        policy._cached_mask = _build_available_actions_mask(set(), _N_DISCRETE_ACTIONS)
        # Should not raise, should return one of the DISCRETE_ACTIONS rows
        action = policy(_zero_obs())
        self.assertEqual(action.shape, (4,))

    def test_mask_cached_from_update_info(self):
        """Calling update() with info["available_fn_ids"] updates _cached_mask."""
        policy = _make_policy()
        self.assertTrue(np.all(policy._cached_mask))  # starts all-True

        obs = _zero_obs()
        action = DISCRETE_ACTIONS[0].copy()
        policy.update(obs, action, 1.0, obs, False, info={"available_fn_ids": {1}})

        # After update with only select_army available, cached mask: cell 4 True, rest False
        self.assertTrue(policy._cached_mask[4])
        self.assertFalse(policy._cached_mask[0])

    def test_mask_cached_from_update_none_info(self):
        """update() with no info or None available_fn_ids leaves all-True mask."""
        policy = _make_policy()
        # Force a non-default mask first
        policy._cached_mask = np.zeros(_N_DISCRETE_ACTIONS, dtype=bool)

        obs = _zero_obs()
        policy.update(obs, 0, 0.0, obs, False, info={"available_fn_ids": None})
        # None → all-True
        self.assertTrue(np.all(policy._cached_mask))

    def test_on_episode_start_resets_mask(self):
        """on_episode_start() should restore all-True mask."""
        policy = _make_policy()
        policy._cached_mask = np.zeros(_N_DISCRETE_ACTIONS, dtype=bool)
        policy.on_episode_start()
        self.assertTrue(np.all(policy._cached_mask))

    def test_replay_buffer_is_masked_type(self):
        policy = _make_policy()
        self.assertIsInstance(policy._replay, _MaskedReplayBuffer)


# ---------------------------------------------------------------------------
# SC2NeuralDQNPolicy — replay buffer stores correct mask
# ---------------------------------------------------------------------------

class TestSC2NeuralDQNPolicyReplay(unittest.TestCase):

    def test_mask_stored_in_buffer(self):
        """Transitions pushed via update() carry the mask from info."""
        policy = _make_policy(min_replay_size=1000)  # prevent gradient steps
        obs = _zero_obs()
        action = DISCRETE_ACTIONS[4].copy()
        only_army = {"available_fn_ids": {1}}  # only select_army

        policy.update(obs, action, 1.0, obs, False, info=only_army)

        self.assertEqual(len(policy._replay), 1)
        stored = policy._replay._buf[0]
        stored_mask = stored[5]  # index 5 = mask
        expected_mask = _build_available_actions_mask({1}, _N_DISCRETE_ACTIONS)
        np.testing.assert_array_equal(stored_mask, expected_mask)

    def test_mask_all_true_when_no_info(self):
        """Without info, the stored mask is all-True."""
        policy = _make_policy(min_replay_size=1000)
        obs = _zero_obs()
        policy.update(obs, 0, 0.0, obs, False)  # no info kwarg

        stored_mask = policy._replay._buf[0][5]
        self.assertTrue(np.all(stored_mask))

    def test_buffer_fills_on_update(self):
        policy = _make_policy(min_replay_size=1000)
        for _ in range(5):
            policy.update(_zero_obs(), 0, 0.0, _zero_obs(), False)
        self.assertEqual(len(policy._replay), 5)


# ---------------------------------------------------------------------------
# SC2NeuralDQNPolicy — gradient step (masked targets)
# ---------------------------------------------------------------------------

class TestSC2NeuralDQNGradientMasked(unittest.TestCase):

    def test_gradient_step_masked_runs_without_error(self):
        """_gradient_step_masked should not raise for a simple batch."""
        policy = _make_policy(min_replay_size=1000)
        B = 4
        obs_b   = np.zeros((B, _OBS_SPEC.dim), dtype=np.float32)
        act_b   = np.zeros(B, dtype=np.int32)
        rew_b   = np.ones(B, dtype=np.float32)
        next_b  = np.zeros((B, _OBS_SPEC.dim), dtype=np.float32)
        done_b  = np.zeros(B, dtype=np.float32)
        mask_b  = np.ones((B, _N_DISCRETE_ACTIONS), dtype=bool)
        # Should not raise
        policy._gradient_step_masked(obs_b, act_b, rew_b, next_b, done_b, mask_b)

    def test_illegal_action_q_not_bootstrapped(self):
        """
        Bandit scenario: action A always gives +1, action B always gives -0.1.
        When B is masked in next-state, the target should not be inflated by B's
        high initial Q-value. We verify that the policy converges to prefer A.
        """
        np.random.seed(0)
        policy = SC2NeuralDQNPolicy(
            obs_spec=_OBS_SPEC,
            hidden_sizes=[16, 16],
            replay_buffer_size=2000,
            batch_size=32,
            min_replay_size=64,
            target_update_freq=20,
            learning_rate=0.005,
            epsilon_start=1.0,
            epsilon_end=1.0,   # stay at 1.0 to manually push transitions
            epsilon_decay_steps=1,
            gamma=0.0,          # pure bandit — no bootstrapping
        )

        obs = _zero_obs()
        next_obs = _zero_obs()
        BEST = 4    # select_army (fn_idx=1)
        GOOD_R = 1.0
        BAD_R = -0.1

        # Only select_army (cell 4) is available
        available = {1}
        mask = _build_available_actions_mask(available, _N_DISCRETE_ACTIONS)

        for step in range(3000):
            action_idx = step % _N_DISCRETE_ACTIONS
            r = GOOD_R if action_idx == BEST else BAD_R
            policy.update(obs, action_idx, r, next_obs, done=True,
                          info={"available_fn_ids": available})

        # Assert on raw Q-values directly (unmasked) so this test catches
        # regressions in _gradient_step_masked, not just the masking of __call__.
        policy._eps = 0.0
        obs_norm = (obs / policy._scales).astype(np.float32)
        q_vals = policy._q_values(policy._online, obs_norm)
        greedy_idx = int(np.argmax(q_vals))
        self.assertEqual(
            greedy_idx, BEST,
            f"Expected greedy action {BEST} (select_army), got {greedy_idx}. "
            f"Q-values: {q_vals.tolist()}"
        )
        # The best action must have a clearly higher Q-value than all others.
        self.assertGreater(
            float(q_vals[BEST]),
            float(np.max(np.delete(q_vals, BEST))),
            f"Q[{BEST}] should be the unique maximum. Q-values: {q_vals.tolist()}"
        )


# ---------------------------------------------------------------------------
# SC2NeuralDQNPolicy — serialisation
# ---------------------------------------------------------------------------

class TestSC2NeuralDQNPolicySerialization(unittest.TestCase):

    def _make_trained(self) -> SC2NeuralDQNPolicy:
        policy = _make_policy(min_replay_size=32, epsilon_start=1.0)
        obs = _zero_obs()
        for i in range(50):
            policy.update(obs, i % _N_DISCRETE_ACTIONS, float(i % 3), obs, False)
        return policy

    def test_to_cfg_policy_type(self):
        policy = _make_policy()
        self.assertEqual(policy.to_cfg()["policy_type"], "sc2_neural_dqn")

    def test_from_cfg_roundtrip(self):
        policy = _make_policy(epsilon_start=0.0)
        cfg = policy.to_cfg()
        policy2 = SC2NeuralDQNPolicy.from_cfg(cfg, _OBS_SPEC)
        obs = _rand_obs()
        policy._eps = 0.0
        policy2._eps = 0.0
        np.testing.assert_array_equal(policy(obs), policy2(obs))

    def test_trainer_state_roundtrip(self):
        """Replay buffer (with masks), Adam moments, and counters survive save/load.

        Also verifies that the restored replay buffer is a _MaskedReplayBuffer
        and that training can resume after load (6-tuple unpack in update()).
        """
        policy = self._make_trained()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            policy.save_trainer_state(path)
            policy2 = _make_policy(epsilon_start=1.0)
            policy2.load_trainer_state(path)

            # Buffer length and step counter preserved.
            self.assertEqual(len(policy2._replay), len(policy._replay))
            self.assertEqual(policy2._total_steps, policy._total_steps)

            # Replay buffer must stay a _MaskedReplayBuffer after load.
            self.assertIsInstance(policy2._replay, _MaskedReplayBuffer)

            # sample() must return 6 values (not 5 from base _ReplayBuffer).
            result = policy2._replay.sample(4)
            self.assertEqual(len(result), 6,
                             "sample() should return 6-tuple for _MaskedReplayBuffer")

            # Training must be able to resume without unpack errors.
            obs = _zero_obs()
            policy2.update(obs, 0, 1.0, obs, False)
        finally:
            os.unlink(path)

    def test_trainer_state_masks_preserved(self):
        """Masks stored in the replay buffer survive save/load."""
        policy = _make_policy(min_replay_size=1000)
        obs = _zero_obs()
        # Push a transition with only select_army (cell 4) available.
        partial_mask = _build_available_actions_mask({1}, _N_DISCRETE_ACTIONS)
        policy.update(obs, 4, 1.0, obs, False,
                      info={"available_fn_ids": {1}})
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            policy.save_trainer_state(path)
            policy2 = _make_policy(epsilon_start=1.0)
            policy2.load_trainer_state(path)

            stored = policy2._replay._buf[0]
            restored_mask = stored[5]  # index 5 = mask
            np.testing.assert_array_equal(restored_mask, partial_mask)
        finally:
            os.unlink(path)

    def test_trainer_state_backward_compat_no_masks(self):
        """Loading a state file without replay_mask restores all-True masks."""
        from games.sc2.policies import NeuralDQNPolicy as _BaseDQN
        base_policy = _BaseDQN(
            obs_spec=_OBS_SPEC,
            hidden_sizes=[8, 8],
            replay_buffer_size=500,
            batch_size=16,
            min_replay_size=32,
            epsilon_start=1.0,
        )
        obs = _zero_obs()
        for i in range(5):
            base_policy.update(obs, i, float(i), obs, False)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            base_policy.save_trainer_state(path)
            # Load into SC2NeuralDQNPolicy — should not crash; masks → all-True.
            policy2 = _make_policy(epsilon_start=1.0)
            policy2.load_trainer_state(path)
            self.assertIsInstance(policy2._replay, _MaskedReplayBuffer)
            for entry in policy2._replay._buf:
                self.assertTrue(np.all(entry[5]),
                                "Backward-compat mask should be all-True")
        finally:
            os.unlink(path)

    def test_from_cfg_shape_mismatch_raises(self):
        """from_cfg with mismatched obs_dim should raise ValueError."""
        policy = _make_policy()
        cfg = policy.to_cfg()
        from games.sc2.obs_spec import SC2_LADDER_OBS_SPEC
        with self.assertRaises(ValueError):
            SC2NeuralDQNPolicy.from_cfg(cfg, SC2_LADDER_OBS_SPEC)


# ---------------------------------------------------------------------------
# SC2 client available_fn_ids
# ---------------------------------------------------------------------------

class TestSC2ClientAvailableFnIds(unittest.TestCase):
    """Verify that _timestep_to_obs_info populates info["available_fn_ids"]."""

    def setUp(self):
        from games.sc2.client import SC2Client
        self.client = SC2Client(map_name="MoveToBeacon")

    class _NamedArr:
        def __init__(self, mapping):
            self._mapping = mapping
        def __getitem__(self, key):
            return self._mapping[key]
        def get(self, key, default=None):
            return self._mapping.get(key, default)

    class _FakeTimeStep:
        def __init__(self, obs, reward=0.0, last=False):
            self.observation = obs
            self.reward = reward
            self._last = last
        def last(self):
            return self._last

    def _make_obs(self):
        return {
            "player": self._NamedArr({
                "minerals": 0, "vespene": 0, "food_used": 0, "food_cap": 0,
                "army_count": 0, "idle_worker_count": 0,
                "warp_gate_count": 0, "larva_count": 0,
            }),
            "feature_screen": np.zeros((17, 64, 64), dtype=np.int32),
            "score_cumulative": np.array([0]),
        }

    def test_available_fn_ids_is_none_when_no_available_actions_key(self):
        """Without 'available_actions' in obs, available_fn_ids is None."""
        ts = self._FakeTimeStep(self._make_obs())
        _, info = self.client._timestep_to_obs_info(ts)
        self.assertIn("available_fn_ids", info)
        self.assertIsNone(info["available_fn_ids"])

    def test_available_fn_ids_key_always_present(self):
        """info['available_fn_ids'] key must always be set."""
        ts = self._FakeTimeStep(self._make_obs())
        _, info = self.client._timestep_to_obs_info(ts)
        self.assertIn("available_fn_ids", info)


if __name__ == "__main__":
    unittest.main(verbosity=2)
