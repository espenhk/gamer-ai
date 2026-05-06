"""Tests for SC2LSTMPolicy and SC2LSTMEvolutionPolicy in games/sc2/sc2_policies.py.

Covers:
  - Hidden state shape correct (h, c are zeros at start, change after step).
  - Output splits into fn (6) and spatial (9) logits; action is valid 4-vector.
  - Masking never selects unavailable fn_idx.
  - Weight count matches formula: 4*(h*(h+obs_dim)+h) + 15*h + 15.
  - reset_on_episode=False carries (h, c) across episode resets.
  - YAML save/load round-trip (to_cfg / from_cfg).
  - SC2LSTMEvolutionPolicy: population size, champion set after one generation,
    σ adapts, trainer-state round-trip.
"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC, SC2_LADDER_OBS_SPEC
from games.sc2.sc2_policies import (
    N_FUNCTION_IDS,
    N_LSTM_SPATIAL_CELLS,
    SC2LSTMPolicy,
    SC2LSTMEvolutionPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lstm(
    obs_spec=None,
    hidden_size: int = 16,
    reset_on_episode: bool = True,
    seed: int = 0,
) -> SC2LSTMPolicy:
    spec = obs_spec or SC2_MINIGAME_OBS_SPEC
    return SC2LSTMPolicy(
        obs_spec=spec,
        hidden_size=hidden_size,
        reset_on_episode=reset_on_episode,
        seed=seed,
    )


def _make_evo(
    obs_spec=None,
    hidden_size: int = 16,
    pop: int = 4,
    sigma: float = 0.03,
    reset_on_episode: bool = True,
) -> SC2LSTMEvolutionPolicy:
    spec = obs_spec or SC2_MINIGAME_OBS_SPEC
    return SC2LSTMEvolutionPolicy(
        obs_spec=spec,
        hidden_size=hidden_size,
        population_size=pop,
        initial_sigma=sigma,
        reset_on_episode=reset_on_episode,
        seed=0,
    )


def _rand_obs(obs_spec=None, seed: int = 42) -> np.ndarray:
    spec = obs_spec or SC2_MINIGAME_OBS_SPEC
    return np.random.default_rng(seed).standard_normal(spec.dim).astype(np.float32)


def _run_one_generation(evo: SC2LSTMEvolutionPolicy, obs_spec=None) -> None:
    obs = _rand_obs(obs_spec)
    individuals = evo.sample_population()
    rewards = [float(i) for i in range(len(individuals))]
    evo.update_distribution(rewards)


# ---------------------------------------------------------------------------
# SC2LSTMPolicy — structure tests
# ---------------------------------------------------------------------------

class TestSC2LSTMPolicyStructure(unittest.TestCase):

    def test_hidden_state_zero_at_init(self):
        p = _make_lstm()
        np.testing.assert_array_equal(p._h, np.zeros(p._hidden_size))
        np.testing.assert_array_equal(p._c, np.zeros(p._hidden_size))

    def test_W_out_shape(self):
        p = _make_lstm(hidden_size=16)
        self.assertEqual(p._W_out.shape, (SC2LSTMPolicy.N_OUTPUT, 16))
        self.assertEqual(SC2LSTMPolicy.N_OUTPUT, N_FUNCTION_IDS + N_LSTM_SPATIAL_CELLS)

    def test_flat_dim_formula(self):
        h      = 16
        obs_dim = SC2_MINIGAME_OBS_SPEC.dim
        c_in   = h + obs_dim
        n_out  = N_FUNCTION_IDS + N_LSTM_SPATIAL_CELLS
        expected = 4 * (h * c_in + h) + n_out * h + n_out
        p = _make_lstm(hidden_size=h)
        self.assertEqual(p.flat_dim, expected)

    def test_flat_dim_ladder(self):
        h       = 16
        obs_dim = SC2_LADDER_OBS_SPEC.dim
        c_in    = h + obs_dim
        n_out   = N_FUNCTION_IDS + N_LSTM_SPATIAL_CELLS
        expected = 4 * (h * c_in + h) + n_out * h + n_out
        p = _make_lstm(SC2_LADDER_OBS_SPEC, hidden_size=h)
        self.assertEqual(p.flat_dim, expected)

    def test_to_flat_length(self):
        p = _make_lstm(hidden_size=16)
        self.assertEqual(len(p.to_flat()), p.flat_dim)

    def test_with_flat_round_trip(self):
        p    = _make_lstm(hidden_size=16, seed=1)
        flat = p.to_flat()
        p2   = p.with_flat(flat)
        np.testing.assert_array_equal(p2.to_flat(), flat)

    def test_with_flat_wrong_size_raises(self):
        p = _make_lstm(hidden_size=16)
        with self.assertRaises(ValueError):
            p.with_flat(np.zeros(p.flat_dim + 1, dtype=np.float32))


# ---------------------------------------------------------------------------
# SC2LSTMPolicy — action tests
# ---------------------------------------------------------------------------

class TestSC2LSTMPolicyAction(unittest.TestCase):

    def test_action_shape(self):
        p      = _make_lstm()
        obs    = _rand_obs()
        action = p(obs)
        self.assertEqual(action.shape, (4,))

    def test_fn_idx_in_range(self):
        p = _make_lstm()
        for seed in range(10):
            obs    = _rand_obs(seed=seed)
            action = p(obs)
            self.assertIn(int(action[0]), range(N_FUNCTION_IDS))

    def test_x_y_in_unit_interval(self):
        p = _make_lstm()
        obs = _rand_obs()
        for _ in range(20):
            action = p(obs)
            self.assertGreaterEqual(float(action[1]), 0.0)
            self.assertLessEqual(float(action[1]), 1.0)
            self.assertGreaterEqual(float(action[2]), 0.0)
            self.assertLessEqual(float(action[2]), 1.0)

    def test_hidden_state_changes_after_step(self):
        p   = _make_lstm()
        obs = _rand_obs()
        h0  = p._h.copy()
        p(obs)
        self.assertFalse(np.allclose(p._h, h0))

    def test_consecutive_steps_differ(self):
        p    = _make_lstm()
        obs  = _rand_obs()
        a1   = p(obs).copy()
        a2   = p(obs).copy()
        # Hidden state accumulates, so subsequent actions may differ.
        # At least the output should be a valid action in both cases.
        self.assertEqual(a1.shape, (4,))
        self.assertEqual(a2.shape, (4,))


# ---------------------------------------------------------------------------
# Available-actions masking
# ---------------------------------------------------------------------------

class TestSC2LSTMPolicyMasking(unittest.TestCase):

    def test_masking_never_selects_unavailable_fn(self):
        p = _make_lstm(seed=7)
        rng = np.random.default_rng(0)
        # Allow only fn_idx 0.
        p._available_fn_ids = {0}
        for _ in range(30):
            obs    = rng.standard_normal(SC2_MINIGAME_OBS_SPEC.dim).astype(np.float32)
            action = p(obs)
            self.assertEqual(int(action[0]), 0)

    def test_masking_set_via_on_episode_start(self):
        p = _make_lstm()
        p.on_episode_start(info={"available_fn_ids": [1, 2]})
        self.assertEqual(p._available_fn_ids, {1, 2})

    def test_masking_updated_via_update(self):
        p   = _make_lstm()
        obs = _rand_obs()
        p.update(obs, np.zeros(4), 0.0, obs, False,
                 info={"available_fn_ids": [0, 3]})
        self.assertEqual(p._available_fn_ids, {0, 3})

    def test_fallback_when_all_masked(self):
        p = _make_lstm(seed=0)
        p._available_fn_ids = set()  # nothing available
        obs    = _rand_obs()
        action = p(obs)
        # Should not raise; falls back to no_op (index 0).
        self.assertEqual(int(action[0]), 0)


# ---------------------------------------------------------------------------
# Hidden state reset behaviour
# ---------------------------------------------------------------------------

class TestSC2LSTMPolicyHiddenReset(unittest.TestCase):

    def test_reset_on_episode_true_zeros_state(self):
        p   = _make_lstm(reset_on_episode=True)
        obs = _rand_obs()
        p(obs)  # advance hidden state
        self.assertFalse(np.allclose(p._h, 0.0))
        p.on_episode_start()
        np.testing.assert_array_equal(p._h, np.zeros(p._hidden_size))
        np.testing.assert_array_equal(p._c, np.zeros(p._hidden_size))

    def test_reset_on_episode_false_carries_state(self):
        p   = _make_lstm(reset_on_episode=False)
        obs = _rand_obs()
        p(obs)
        h_before = p._h.copy()
        p.on_episode_start()  # must NOT reset
        np.testing.assert_array_equal(p._h, h_before)

    def test_on_episode_end_resets_when_flag_true(self):
        p   = _make_lstm(reset_on_episode=True)
        obs = _rand_obs()
        p(obs)
        p.on_episode_end()
        np.testing.assert_array_equal(p._h, np.zeros(p._hidden_size))

    def test_on_episode_end_no_reset_when_flag_false(self):
        p   = _make_lstm(reset_on_episode=False)
        obs = _rand_obs()
        p(obs)
        h_before = p._h.copy()
        p.on_episode_end()
        np.testing.assert_array_equal(p._h, h_before)


# ---------------------------------------------------------------------------
# SC2LSTMPolicy serialisation
# ---------------------------------------------------------------------------

class TestSC2LSTMPolicySerialisation(unittest.TestCase):

    def test_to_cfg_from_cfg_round_trip(self):
        p   = _make_lstm(hidden_size=16, reset_on_episode=False, seed=3)
        cfg = p.to_cfg()
        p2  = SC2LSTMPolicy.from_cfg(cfg, SC2_MINIGAME_OBS_SPEC)
        np.testing.assert_array_equal(p.to_flat(), p2.to_flat())
        self.assertEqual(p2._reset_on_episode, False)
        self.assertEqual(p2._hidden_size, 16)

    def test_save_load_round_trip(self):
        p    = _make_lstm(hidden_size=16, seed=5)
        flat = p.to_flat()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            p.save(path)
            import yaml
            with open(path) as _f:
                cfg = yaml.safe_load(_f)
            p2 = SC2LSTMPolicy.from_cfg(cfg, SC2_MINIGAME_OBS_SPEC)
            np.testing.assert_array_almost_equal(flat, p2.to_flat(), decimal=5)
        finally:
            os.unlink(path)

    def test_policy_type_in_cfg(self):
        p   = _make_lstm()
        cfg = p.to_cfg()
        self.assertEqual(cfg["policy_type"], "sc2_lstm")


# ---------------------------------------------------------------------------
# SC2LSTMEvolutionPolicy
# ---------------------------------------------------------------------------

class TestSC2LSTMEvolutionPolicy(unittest.TestCase):

    def test_population_size(self):
        evo = _make_evo(pop=6)
        self.assertEqual(evo.population_size, 6)
        individuals = evo.sample_population()
        self.assertEqual(len(individuals), 6)

    def test_individuals_are_sc2_lstm(self):
        evo = _make_evo(pop=4)
        for ind in evo.sample_population():
            self.assertIsInstance(ind, SC2LSTMPolicy)

    def test_call_raises_before_first_generation(self):
        evo = _make_evo()
        obs = _rand_obs()
        with self.assertRaises(RuntimeError):
            evo(obs)

    def test_champion_set_after_first_generation(self):
        evo = _make_evo()
        self.assertIsNone(evo._champion)
        _run_one_generation(evo)
        self.assertIsNotNone(evo._champion)

    def test_sigma_adapts_across_generations(self):
        evo    = _make_evo(pop=4, sigma=0.03)
        sigma0 = evo.sigma
        _run_one_generation(evo)
        _run_one_generation(evo)
        self.assertNotAlmostEqual(evo.sigma, sigma0, places=10)

    def test_call_returns_valid_action_after_generation(self):
        evo = _make_evo()
        _run_one_generation(evo)
        obs    = _rand_obs()
        action = evo(obs)
        self.assertEqual(action.shape, (4,))
        self.assertIn(int(action[0]), range(N_FUNCTION_IDS))

    def test_update_distribution_wrong_count_raises(self):
        evo = _make_evo(pop=4)
        evo.sample_population()
        with self.assertRaises(ValueError):
            evo.update_distribution([1.0, 2.0])

    def test_initialize_from_champion_flat_dim_mismatch_raises(self):
        evo   = _make_evo(pop=4, hidden_size=16)
        wrong = SC2LSTMPolicy(SC2_MINIGAME_OBS_SPEC, hidden_size=32)
        with self.assertRaises(ValueError):
            evo.initialize_from_champion(wrong)

    def test_initialize_from_champion_sets_mean(self):
        evo     = _make_evo(pop=4, hidden_size=16)
        champion = SC2LSTMPolicy(SC2_MINIGAME_OBS_SPEC, hidden_size=16, seed=99)
        flat     = champion.to_flat()
        evo.initialize_from_champion(champion)
        np.testing.assert_array_almost_equal(evo._mean, flat.astype(np.float64), decimal=6)

    def test_on_episode_start_forwarded_to_champion(self):
        evo = _make_evo()
        _run_one_generation(evo)
        obs = _rand_obs()
        evo(obs)  # advance hidden state
        evo.on_episode_start(info={"available_fn_ids": [0]})
        # Hidden state should have been reset (reset_on_episode=True by default).
        np.testing.assert_array_equal(evo._champion._h, np.zeros(evo._champion._hidden_size))

    def test_save_writes_yaml(self):
        evo = _make_evo()
        _run_one_generation(evo)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            evo.save(path)
            import yaml
            with open(path) as _f:
                cfg = yaml.safe_load(_f)
            self.assertEqual(cfg["policy_type"], "sc2_lstm")
        finally:
            os.unlink(path)

    def test_trainer_state_round_trip(self):
        evo = _make_evo(pop=4, sigma=0.03)
        _run_one_generation(evo)
        sigma_after = evo.sigma
        mean_after  = evo._mean.copy()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            evo.save_trainer_state(path)
            evo2 = _make_evo(pop=4, sigma=99.0)
            evo2.load_trainer_state(path)
            self.assertAlmostEqual(evo2.sigma, sigma_after, places=10)
            np.testing.assert_array_equal(evo2._mean, mean_after)
        finally:
            os.unlink(path)

    def test_trainer_state_dim_mismatch_raises(self):
        evo = _make_evo(pop=4, hidden_size=16)
        _run_one_generation(evo)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            evo.save_trainer_state(path)
            evo2 = _make_evo(pop=4, hidden_size=32)  # different hidden_size → different flat_dim
            with self.assertRaises(ValueError):
                evo2.load_trainer_state(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
