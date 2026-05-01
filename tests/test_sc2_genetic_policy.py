"""Tests for SC2MultiHeadLinearPolicy and SC2GeneticPolicy."""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC, SC2_LADDER_OBS_SPEC
from games.sc2.sc2_policies import (
    N_FUNCTION_IDS,
    N_GRID_CELLS,
    SC2GeneticPolicy,
    SC2MultiHeadLinearPolicy,
    _ALL_ROW_NAMES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_policy(obs_spec=None) -> SC2MultiHeadLinearPolicy:
    spec = obs_spec or SC2_MINIGAME_OBS_SPEC
    return SC2MultiHeadLinearPolicy(spec)


def _make_genetic(pop=6, elite=2, eval_episodes=1,
                  obs_spec=None) -> SC2GeneticPolicy:
    spec = obs_spec or SC2_MINIGAME_OBS_SPEC
    return SC2GeneticPolicy(
        obs_spec=spec,
        population_size=pop,
        elite_k=elite,
        mutation_scale=0.1,
        mutation_share=0.3,
        eval_episodes=eval_episodes,
    )


# ---------------------------------------------------------------------------
# SC2MultiHeadLinearPolicy tests
# ---------------------------------------------------------------------------

class TestSC2MultiHeadLinearPolicyInit(unittest.TestCase):

    def test_fn_weights_shape_minigame(self):
        p = _make_policy()
        self.assertEqual(p._fn_weights.shape,
                         (N_FUNCTION_IDS, SC2_MINIGAME_OBS_SPEC.dim))

    def test_spatial_weights_shape_minigame(self):
        p = _make_policy()
        self.assertEqual(p._sp_weights.shape,
                         (N_GRID_CELLS, SC2_MINIGAME_OBS_SPEC.dim))

    def test_fn_weights_shape_ladder(self):
        p = _make_policy(SC2_LADDER_OBS_SPEC)
        self.assertEqual(p._fn_weights.shape,
                         (N_FUNCTION_IDS, SC2_LADDER_OBS_SPEC.dim))

    def test_spatial_weights_shape_ladder(self):
        p = _make_policy(SC2_LADDER_OBS_SPEC)
        self.assertEqual(p._sp_weights.shape,
                         (N_GRID_CELLS, SC2_LADDER_OBS_SPEC.dim))

    def test_total_flat_dim_minigame(self):
        """θ dimension = (N_FUNCTION_IDS + N_GRID_CELLS) × obs_dim."""
        p = _make_policy()
        obs_dim = SC2_MINIGAME_OBS_SPEC.dim
        expected = (N_FUNCTION_IDS + N_GRID_CELLS) * obs_dim
        self.assertEqual(len(p.to_flat()), expected)

    def test_total_flat_dim_ladder(self):
        p = _make_policy(SC2_LADDER_OBS_SPEC)
        obs_dim = SC2_LADDER_OBS_SPEC.dim
        expected = (N_FUNCTION_IDS + N_GRID_CELLS) * obs_dim
        self.assertEqual(len(p.to_flat()), expected)

    def test_explicit_weights_stored(self):
        obs_dim = SC2_MINIGAME_OBS_SPEC.dim
        fn_w = np.ones((N_FUNCTION_IDS, obs_dim), dtype=np.float32)
        sp_w = np.full((N_GRID_CELLS, obs_dim), 2.0, dtype=np.float32)
        p = SC2MultiHeadLinearPolicy(SC2_MINIGAME_OBS_SPEC, fn_w, sp_w)
        np.testing.assert_array_equal(p._fn_weights, fn_w)
        np.testing.assert_array_equal(p._sp_weights, sp_w)


class TestSC2MultiHeadLinearPolicyCall(unittest.TestCase):

    def test_call_returns_4_vector(self):
        p   = _make_policy()
        obs = np.ones(SC2_MINIGAME_OBS_SPEC.dim, dtype=np.float32)
        act = p(obs)
        self.assertEqual(act.shape, (4,))

    def test_call_fn_idx_in_valid_range(self):
        p   = _make_policy()
        obs = np.random.default_rng().standard_normal(SC2_MINIGAME_OBS_SPEC.dim)
        act = p(obs.astype(np.float32))
        self.assertGreaterEqual(int(act[0]), 0)
        self.assertLess(int(act[0]), N_FUNCTION_IDS)

    def test_call_spatial_coords_in_unit_range(self):
        p   = _make_policy()
        obs = np.random.default_rng().standard_normal(SC2_MINIGAME_OBS_SPEC.dim)
        act = p(obs.astype(np.float32))
        self.assertGreaterEqual(float(act[1]), 0.0)
        self.assertLessEqual(float(act[1]), 1.0)
        self.assertGreaterEqual(float(act[2]), 0.0)
        self.assertLessEqual(float(act[2]), 1.0)

    def test_call_queue_is_zero(self):
        p   = _make_policy()
        obs = np.ones(SC2_MINIGAME_OBS_SPEC.dim, dtype=np.float32)
        act = p(obs)
        self.assertEqual(float(act[3]), 0.0)

    def test_call_selects_max_fn_score(self):
        """Policy with all-zero fn weights except one should select that function."""
        obs_dim = SC2_MINIGAME_OBS_SPEC.dim
        fn_w = np.zeros((N_FUNCTION_IDS, obs_dim), dtype=np.float32)
        fn_w[3, :] = 1.0   # function ID 3 gets all positive weights
        sp_w = np.zeros((N_GRID_CELLS, obs_dim), dtype=np.float32)
        p    = SC2MultiHeadLinearPolicy(SC2_MINIGAME_OBS_SPEC, fn_w, sp_w)
        obs  = np.ones(obs_dim, dtype=np.float32)
        act  = p(obs)
        self.assertEqual(int(act[0]), 3)

    def test_call_selects_max_spatial_cell(self):
        """Policy with all-zero spatial weights except cell 7 selects cell 7."""
        obs_dim = SC2_MINIGAME_OBS_SPEC.dim
        from games.sc2.sc2_policies import _GRID_XY
        fn_w = np.zeros((N_FUNCTION_IDS, obs_dim), dtype=np.float32)
        sp_w = np.zeros((N_GRID_CELLS, obs_dim), dtype=np.float32)
        sp_w[7, :] = 1.0
        p = SC2MultiHeadLinearPolicy(SC2_MINIGAME_OBS_SPEC, fn_w, sp_w)
        obs = np.ones(obs_dim, dtype=np.float32)
        act = p(obs)
        expected_x, expected_y = _GRID_XY[7]
        self.assertAlmostEqual(float(act[1]), expected_x)
        self.assertAlmostEqual(float(act[2]), expected_y)


class TestSC2MultiHeadLinearPolicySerialization(unittest.TestCase):

    def test_to_cfg_has_correct_keys(self):
        p   = _make_policy()
        cfg = p.to_cfg()
        # 6 fn_idx rows + 9 spatial rows = 15 keys
        self.assertEqual(len(cfg), N_FUNCTION_IDS + N_GRID_CELLS)
        for name in _ALL_ROW_NAMES:
            self.assertIn(f"{name}_weights", cfg)

    def test_all_cfg_keys_end_with_weights(self):
        """GeneticPolicy._crossover relies on this suffix."""
        p   = _make_policy()
        cfg = p.to_cfg()
        for key in cfg:
            self.assertTrue(key.endswith("_weights"), f"Key {key!r} missing _weights suffix")

    def test_cfg_values_are_obs_name_dicts(self):
        p     = _make_policy()
        cfg   = p.to_cfg()
        names = SC2_MINIGAME_OBS_SPEC.names
        for key, row_cfg in cfg.items():
            self.assertIsInstance(row_cfg, dict)
            for n in names:
                self.assertIn(n, row_cfg)

    def test_from_cfg_round_trips_weights(self):
        p1  = _make_policy()
        cfg = p1.to_cfg()
        p2  = SC2MultiHeadLinearPolicy.from_cfg(cfg, SC2_MINIGAME_OBS_SPEC)
        np.testing.assert_array_almost_equal(p1._fn_weights, p2._fn_weights)
        np.testing.assert_array_almost_equal(p1._sp_weights, p2._sp_weights)

    def test_yaml_round_trip_lossless(self):
        """Champion YAML round-trips losslessly (issue requirement)."""
        p1 = _make_policy()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            p1.save(path)
            p2 = SC2MultiHeadLinearPolicy.load(path, SC2_MINIGAME_OBS_SPEC)
        finally:
            os.unlink(path)
        np.testing.assert_array_almost_equal(p1._fn_weights, p2._fn_weights, decimal=6)
        np.testing.assert_array_almost_equal(p1._sp_weights, p2._sp_weights, decimal=6)

    def test_from_cfg_missing_features_default_to_zero(self):
        """Graceful migration when obs_spec grows."""
        p   = SC2MultiHeadLinearPolicy.load.__func__  # just use from_cfg directly
        cfg = {}   # empty → all zeros
        p2  = SC2MultiHeadLinearPolicy.from_cfg(cfg, SC2_MINIGAME_OBS_SPEC)
        np.testing.assert_array_equal(p2._fn_weights, 0.0)
        np.testing.assert_array_equal(p2._sp_weights, 0.0)


class TestSC2MultiHeadLinearPolicyFlat(unittest.TestCase):

    def test_to_flat_length(self):
        p       = _make_policy()
        obs_dim = SC2_MINIGAME_OBS_SPEC.dim
        self.assertEqual(len(p.to_flat()), (N_FUNCTION_IDS + N_GRID_CELLS) * obs_dim)

    def test_with_flat_round_trip(self):
        p1   = _make_policy()
        flat = p1.to_flat()
        p2   = p1.with_flat(flat)
        np.testing.assert_array_equal(p1._fn_weights, p2._fn_weights)
        np.testing.assert_array_equal(p1._sp_weights, p2._sp_weights)


class TestSC2MultiHeadLinearPolicyMutation(unittest.TestCase):

    def test_mutated_returns_different_object(self):
        p  = _make_policy()
        pm = p.mutated(scale=0.1, share=1.0)
        self.assertIsNot(p, pm)

    def test_mutated_preserves_obs_spec(self):
        p  = _make_policy()
        pm = p.mutated()
        self.assertIs(pm._obs_spec, p._obs_spec)

    def test_mutated_weights_differ(self):
        p  = _make_policy()
        pm = p.mutated(scale=1.0, share=1.0)
        # Very unlikely to be identical with scale=1.0 and all weights mutated
        self.assertFalse(np.allclose(p._fn_weights, pm._fn_weights))

    def test_mutated_share_zero_leaves_weights_unchanged(self):
        obs_dim = SC2_MINIGAME_OBS_SPEC.dim
        fn_w = np.zeros((N_FUNCTION_IDS, obs_dim), dtype=np.float32)
        sp_w = np.zeros((N_GRID_CELLS,   obs_dim), dtype=np.float32)
        p    = SC2MultiHeadLinearPolicy(SC2_MINIGAME_OBS_SPEC, fn_w, sp_w)
        pm   = p.mutated(scale=1.0, share=0.0)
        np.testing.assert_array_equal(pm._fn_weights, fn_w)
        np.testing.assert_array_equal(pm._sp_weights, sp_w)


# ---------------------------------------------------------------------------
# SC2GeneticPolicy tests
# ---------------------------------------------------------------------------

class TestSC2GeneticPolicyInit(unittest.TestCase):

    def test_population_size_stored(self):
        gp = _make_genetic(pop=8)
        self.assertEqual(gp._pop_size, 8)

    def test_elite_k_stored(self):
        gp = _make_genetic(elite=3)
        self.assertEqual(gp._elite_k, 3)

    def test_eval_episodes_stored(self):
        gp = _make_genetic(eval_episodes=4)
        self.assertEqual(gp._eval_episodes, 4)

    def test_head_names_cover_all_rows(self):
        """head_names must encode all 15 weight-matrix rows."""
        gp = _make_genetic()
        self.assertEqual(len(gp._head_names), N_FUNCTION_IDS + N_GRID_CELLS)


class TestSC2GeneticPolicyPopulation(unittest.TestCase):

    def test_initialize_random_population_size(self):
        gp = _make_genetic(pop=6)
        gp.initialize_random()
        self.assertEqual(len(gp.population), 6)

    def test_initialize_random_members_are_sc2_policy(self):
        gp = _make_genetic(pop=4)
        gp.initialize_random()
        for ind in gp.population:
            self.assertIsInstance(ind, SC2MultiHeadLinearPolicy)

    def test_initialize_random_sets_champion(self):
        gp = _make_genetic()
        gp.initialize_random()
        self.assertIsNotNone(gp._champion)

    def test_population_theta_dimension(self):
        """Each individual must have the correct total weight count."""
        gp      = _make_genetic(pop=4)
        gp.initialize_random()
        obs_dim = SC2_MINIGAME_OBS_SPEC.dim
        expected_flat_len = (N_FUNCTION_IDS + N_GRID_CELLS) * obs_dim
        for ind in gp.population:
            self.assertEqual(len(ind.to_flat()), expected_flat_len)

    def test_initialize_from_champion_population_size(self):
        gp       = _make_genetic(pop=5)
        champion = SC2MultiHeadLinearPolicy(SC2_MINIGAME_OBS_SPEC)
        gp.initialize_from_champion(champion)
        self.assertEqual(len(gp.population), 5)

    def test_initialize_from_champion_sets_champion(self):
        gp       = _make_genetic()
        champion = SC2MultiHeadLinearPolicy(SC2_MINIGAME_OBS_SPEC)
        gp.initialize_from_champion(champion)
        self.assertIs(gp._champion, champion)

    def test_initialize_from_champion_members_are_sc2_policy(self):
        gp       = _make_genetic(pop=3)
        champion = SC2MultiHeadLinearPolicy(SC2_MINIGAME_OBS_SPEC)
        gp.initialize_from_champion(champion)
        for ind in gp.population:
            self.assertIsInstance(ind, SC2MultiHeadLinearPolicy)


class TestSC2GeneticPolicyCrossover(unittest.TestCase):

    def _two_extreme_cfgs(self):
        """Two configs with all +1 / -1 weights for easy crossover verification."""
        names = SC2_MINIGAME_OBS_SPEC.names
        cfg1 = {f"{h}_weights": {n:  1.0 for n in names} for h in _ALL_ROW_NAMES}
        cfg2 = {f"{h}_weights": {n: -1.0 for n in names} for h in _ALL_ROW_NAMES}
        return cfg1, cfg2

    def test_crossover_produces_valid_sc2_policy(self):
        cfg1, cfg2 = self._two_extreme_cfgs()
        child_cfg  = SC2GeneticPolicy._crossover(cfg1, cfg2)
        child      = SC2MultiHeadLinearPolicy.from_cfg(child_cfg, SC2_MINIGAME_OBS_SPEC)
        self.assertIsInstance(child, SC2MultiHeadLinearPolicy)

    def test_crossover_draws_from_both_parents(self):
        cfg1, cfg2 = self._two_extreme_cfgs()
        child_cfg  = SC2GeneticPolicy._crossover(cfg1, cfg2)
        flat_child = SC2MultiHeadLinearPolicy.from_cfg(
            child_cfg, SC2_MINIGAME_OBS_SPEC
        ).to_flat()
        # With 195 weights and ~50% crossover rate, both +1 and -1 should appear
        self.assertIn( 1.0, flat_child.tolist())
        self.assertIn(-1.0, flat_child.tolist())

    def test_crossover_child_has_correct_theta_dimension(self):
        cfg1, cfg2    = self._two_extreme_cfgs()
        child_cfg     = SC2GeneticPolicy._crossover(cfg1, cfg2)
        child         = SC2MultiHeadLinearPolicy.from_cfg(child_cfg, SC2_MINIGAME_OBS_SPEC)
        obs_dim       = SC2_MINIGAME_OBS_SPEC.dim
        expected_len  = (N_FUNCTION_IDS + N_GRID_CELLS) * obs_dim
        self.assertEqual(len(child.to_flat()), expected_len)

    def test_crossover_child_weights_only_from_parents(self):
        """Each weight in the child must equal the corresponding weight from one parent."""
        cfg1, cfg2 = self._two_extreme_cfgs()
        child_cfg  = SC2GeneticPolicy._crossover(cfg1, cfg2)
        for key in child_cfg:
            for k, v in child_cfg[key].items():
                self.assertIn(v, (1.0, -1.0), f"Unexpected value {v} for {key}/{k}")


class TestSC2GeneticPolicyEliteSelection(unittest.TestCase):

    def test_evaluate_and_evolve_updates_champion_reward(self):
        gp = _make_genetic(pop=4, elite=2)
        gp.initialize_random()
        gp.evaluate_and_evolve([10.0, 30.0, 5.0, 1.0])
        self.assertAlmostEqual(gp._champion_reward, 30.0)

    def test_evaluate_and_evolve_returns_true_when_improved(self):
        gp = _make_genetic(pop=4)
        gp.initialize_random()
        improved = gp.evaluate_and_evolve([10.0, 20.0, 5.0, 1.0])
        self.assertTrue(improved)

    def test_evaluate_and_evolve_returns_false_when_no_improvement(self):
        gp = _make_genetic(pop=4)
        gp.initialize_random()
        gp.evaluate_and_evolve([100.0, 90.0, 80.0, 70.0])
        improved = gp.evaluate_and_evolve([50.0, 40.0, 30.0, 20.0])
        self.assertFalse(improved)

    def test_elite_count_preserved(self):
        """After evolution the population still has pop_size members."""
        gp = _make_genetic(pop=6, elite=2)
        gp.initialize_random()
        gp.evaluate_and_evolve([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        self.assertEqual(len(gp.population), 6)

    def test_new_members_are_sc2_policy(self):
        gp = _make_genetic(pop=6, elite=2)
        gp.initialize_random()
        gp.evaluate_and_evolve([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        for ind in gp.population:
            self.assertIsInstance(ind, SC2MultiHeadLinearPolicy)


class TestSC2GeneticPolicySerialization(unittest.TestCase):

    def test_save_writes_yaml(self):
        gp = _make_genetic(pop=4)
        gp.initialize_random()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            gp.save(path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)

    def test_champion_yaml_round_trips_losslessly(self):
        """Issue requirement: champion YAML round-trips losslessly."""
        gp = _make_genetic(pop=4)
        gp.initialize_random()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            gp.save(path)
            loaded = SC2MultiHeadLinearPolicy.load(path, SC2_MINIGAME_OBS_SPEC)
        finally:
            os.unlink(path)
        np.testing.assert_array_almost_equal(
            gp._champion._fn_weights, loaded._fn_weights, decimal=6
        )
        np.testing.assert_array_almost_equal(
            gp._champion._sp_weights, loaded._sp_weights, decimal=6
        )

    def test_to_cfg_policy_type_is_sc2_genetic(self):
        gp = _make_genetic()
        gp.initialize_random()
        cfg = gp.to_cfg()
        self.assertEqual(cfg["policy_type"], "sc2_genetic")

    def test_from_cfg_round_trip_params(self):
        gp1 = SC2GeneticPolicy(
            obs_spec=SC2_MINIGAME_OBS_SPEC,
            population_size=12,
            elite_k=4,
            mutation_scale=0.2,
            mutation_share=0.5,
            eval_episodes=3,
        )
        gp1.initialize_random()
        cfg = gp1.to_cfg()
        gp2 = SC2GeneticPolicy.from_cfg(cfg, SC2_MINIGAME_OBS_SPEC)
        self.assertEqual(gp2._pop_size,        12)
        self.assertEqual(gp2._elite_k,          4)
        self.assertAlmostEqual(gp2._mutation_scale, 0.2)
        self.assertAlmostEqual(gp2._mutation_share, 0.5)
        self.assertEqual(gp2._eval_episodes,     3)


class TestSC2GeneticPolicyCall(unittest.TestCase):

    def test_call_returns_4_vector_after_initialize(self):
        gp  = _make_genetic()
        gp.initialize_random()
        obs = np.ones(SC2_MINIGAME_OBS_SPEC.dim, dtype=np.float32)
        act = gp(obs)
        self.assertEqual(act.shape, (4,))

    def test_call_before_initialize_raises(self):
        gp  = _make_genetic()
        obs = np.ones(SC2_MINIGAME_OBS_SPEC.dim, dtype=np.float32)
        with self.assertRaises(AssertionError):
            gp(obs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
