"""Tests for SC2CMAESPolicy in games/sc2/sc2_policies.py.

Covers:
  - θ dimension = (N_FUNCTION_IDS + N_SPATIAL_ROWS) × obs_dim.
  - Champion YAML round-trips losslessly (save → load → to_flat identical).
  - Available-actions masking produces valid fn_idx.
  - CMA-ES σ adapts (changes) across generations.
  - Trainer-state npz round-trip restores mean and sigma exactly.
  - initialize_from_champion seeds the mean correctly.
"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC, SC2_LADDER_OBS_SPEC
from games.sc2.sc2_policies import (
    N_FUNCTION_IDS,
    N_SPATIAL_ROWS,
    SC2CMAESPolicy,
    SC2MultiHeadLinearPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_policy(obs_spec=None, pop=4, sigma=0.5, eval_episodes=1) -> SC2CMAESPolicy:
    spec = obs_spec or SC2_MINIGAME_OBS_SPEC
    return SC2CMAESPolicy(
        obs_spec=spec,
        population_size=pop,
        initial_sigma=sigma,
        eval_episodes=eval_episodes,
        seed=0,
    )


def _rand_obs(obs_spec=None, seed: int = 42) -> np.ndarray:
    spec = obs_spec or SC2_MINIGAME_OBS_SPEC
    return np.random.default_rng(seed).standard_normal(spec.dim).astype(np.float32)


def _run_one_generation(policy: SC2CMAESPolicy, obs_spec=None) -> None:
    obs = _rand_obs(obs_spec)
    individuals = policy.sample_population()
    rewards = [float(i) for i in range(len(individuals))]
    policy.update_distribution(rewards)


# ---------------------------------------------------------------------------
# Dimension tests
# ---------------------------------------------------------------------------

class TestSC2CMAESPolicyDimension(unittest.TestCase):

    def test_theta_dim_minigame(self):
        p = _make_policy()
        expected = (N_FUNCTION_IDS + N_SPATIAL_ROWS) * SC2_MINIGAME_OBS_SPEC.dim
        self.assertEqual(p._n, expected)

    def test_theta_dim_ladder(self):
        p = _make_policy(SC2_LADDER_OBS_SPEC)
        expected = (N_FUNCTION_IDS + N_SPATIAL_ROWS) * SC2_LADDER_OBS_SPEC.dim
        self.assertEqual(p._n, expected)

    def test_sample_population_count(self):
        p = _make_policy(pop=6)
        individuals = p.sample_population()
        self.assertEqual(len(individuals), 6)

    def test_individuals_are_multihead(self):
        p = _make_policy()
        for ind in p.sample_population():
            self.assertIsInstance(ind, SC2MultiHeadLinearPolicy)


# ---------------------------------------------------------------------------
# CMA-ES mechanics
# ---------------------------------------------------------------------------

class TestSC2CMAESPolicyMechanics(unittest.TestCase):

    def test_update_distribution_requires_sample_first(self):
        p = _make_policy()
        with self.assertRaises(RuntimeError):
            p.update_distribution([1.0, 2.0, 3.0, 4.0])

    def test_sigma_adapts_across_generations(self):
        p = _make_policy(pop=4, sigma=0.5)
        sigma_before = p.sigma
        _run_one_generation(p)
        _run_one_generation(p)
        # sigma must change (either up or down from 1/5 rule is fine)
        self.assertNotAlmostEqual(p.sigma, sigma_before, places=10)

    def test_champion_set_after_first_generation(self):
        p = _make_policy()
        self.assertIsNone(p._champion)
        _run_one_generation(p)
        self.assertIsNotNone(p._champion)

    def test_champion_reward_monotonically_increases(self):
        p = _make_policy(pop=4)
        _run_one_generation(p)
        r1 = p.champion_reward
        _run_one_generation(p)
        self.assertGreaterEqual(p.champion_reward, r1)

    def test_call_raises_before_first_generation(self):
        p = _make_policy()
        obs = _rand_obs()
        with self.assertRaises(RuntimeError):
            p(obs)

    def test_call_returns_valid_action_after_generation(self):
        p = _make_policy()
        _run_one_generation(p)
        obs    = _rand_obs()
        action = p(obs)
        self.assertEqual(action.shape, (4,))
        self.assertIn(int(action[0]), range(N_FUNCTION_IDS))
        self.assertGreaterEqual(float(action[1]), 0.0)
        self.assertLessEqual(float(action[1]), 1.0)

    def test_update_distribution_wrong_count_raises(self):
        p = _make_policy(pop=4)
        p.sample_population()
        with self.assertRaises(ValueError):
            p.update_distribution([1.0, 2.0])


# ---------------------------------------------------------------------------
# Available-actions masking
# ---------------------------------------------------------------------------

class TestSC2CMAESPolicyMasking(unittest.TestCase):

    def test_masking_never_selects_unavailable_fn(self):
        p = SC2CMAESPolicy(
            obs_spec=SC2_MINIGAME_OBS_SPEC, population_size=4,
            initial_sigma=0.5, seed=0,
        )
        _run_one_generation(p)
        rng = np.random.default_rng(1)
        obs = rng.standard_normal(SC2_MINIGAME_OBS_SPEC.dim).astype(np.float32)

        # Allow only fn_idx 0 (no_op).
        p.on_episode_start(info={"available_fn_ids": [0]})
        for _ in range(20):
            action = p(obs)
            self.assertEqual(int(action[0]), 0)

    def test_masking_fallback_when_all_unavailable(self):
        p = SC2CMAESPolicy(
            obs_spec=SC2_MINIGAME_OBS_SPEC, population_size=4,
            initial_sigma=0.5, seed=0
        )
        _run_one_generation(p)
        obs = _rand_obs()
        # Empty available set — should fall back to no_op (index 0).
        p._available_fn_ids = set()
        action = p(obs)
        self.assertEqual(int(action[0]), 0)

    def test_masking_updated_via_update_kwargs(self):
        p = SC2CMAESPolicy(
            obs_spec=SC2_MINIGAME_OBS_SPEC, population_size=4,
            initial_sigma=0.5, seed=0
        )
        _run_one_generation(p)
        obs = _rand_obs()
        p.update(obs, np.zeros(4), 0.0, obs, False,
                 info={"available_fn_ids": [0, 1]})
        self.assertEqual(p._available_fn_ids, {0, 1})

    def test_no_masking_when_available_fn_ids_is_none(self):
        p = SC2CMAESPolicy(
            obs_spec=SC2_MINIGAME_OBS_SPEC, population_size=4,
            initial_sigma=0.5, seed=0
        )
        _run_one_generation(p)
        p._available_fn_ids = None
        obs = _rand_obs()
        # Should not raise.
        action = p(obs)
        self.assertEqual(action.shape, (4,))


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSC2CMAESPolicySerialisation(unittest.TestCase):

    def test_champion_yaml_round_trip(self):
        p = _make_policy(pop=4)
        _run_one_generation(p)
        flat_before = p._champion.to_flat()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            p.save(path)
            loaded = SC2MultiHeadLinearPolicy.load(path, SC2_MINIGAME_OBS_SPEC)
            np.testing.assert_array_almost_equal(flat_before, loaded.to_flat(), decimal=5)
        finally:
            os.unlink(path)

    def test_trainer_state_round_trip(self):
        p = _make_policy(pop=4, sigma=0.5)
        _run_one_generation(p)
        sigma_after = p.sigma
        mean_after  = p._mean.copy()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            p.save_trainer_state(path)
            p2 = _make_policy(pop=4, sigma=99.0)  # different sigma
            p2.load_trainer_state(path)
            self.assertAlmostEqual(p2.sigma, sigma_after, places=10)
            np.testing.assert_array_equal(p2._mean, mean_after)
        finally:
            os.unlink(path)

    def test_trainer_state_dim_mismatch_raises(self):
        p = _make_policy(pop=4)
        _run_one_generation(p)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            p.save_trainer_state(path)
            p_ladder = _make_policy(SC2_LADDER_OBS_SPEC, pop=4)
            with self.assertRaises(ValueError):
                p_ladder.load_trainer_state(path)
        finally:
            os.unlink(path)

    def test_initialize_from_champion_sets_mean(self):
        champ = SC2MultiHeadLinearPolicy(SC2_MINIGAME_OBS_SPEC)
        flat  = champ.to_flat()
        p     = _make_policy(pop=4)
        p.initialize_from_champion(champ)
        np.testing.assert_array_almost_equal(p._mean, flat.astype(np.float64), decimal=6)

    def test_initialize_random_zeros_mean(self):
        p = _make_policy(pop=4)
        p.initialize_random()
        np.testing.assert_array_equal(p._mean, np.zeros(p._n))


if __name__ == "__main__":
    unittest.main()
