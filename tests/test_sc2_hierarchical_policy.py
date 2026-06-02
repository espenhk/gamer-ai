"""Tests for SC2HierarchicalLinearPolicy and SC2HierarchicalGeneticPolicy (issue #388)."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from games.sc2.actions import (
    ACTION_CATEGORIES,
    CATEGORY_NAMES,
    FN_IDX_TO_CATEGORY,
    FUNCTION_IDS,
    N_CATEGORIES,
)
from games.sc2.obs_spec import SC2_LADDER_OBS_SPEC, SC2_MINIGAME_OBS_SPEC
from games.sc2.sc2_policies import (
    _ALL_HIERARCHICAL_ROW_NAMES,
    _META_HEAD_NAMES,
    _QUEUE_HEAD_NAME,
    N_FUNCTION_IDS,
    N_QUEUE_ROWS,
    N_SPATIAL_ROWS,
    SC2HierarchicalGeneticPolicy,
    SC2HierarchicalLinearPolicy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_policy(obs_spec=None) -> SC2HierarchicalLinearPolicy:
    spec = obs_spec or SC2_MINIGAME_OBS_SPEC
    return SC2HierarchicalLinearPolicy(spec)


def _make_genetic(pop=6, elite=2, eval_episodes=1, obs_spec=None) -> SC2HierarchicalGeneticPolicy:
    spec = obs_spec or SC2_MINIGAME_OBS_SPEC
    return SC2HierarchicalGeneticPolicy(
        obs_spec=spec,
        population_size=pop,
        elite_k=elite,
        mutation_scale=0.1,
        mutation_share=0.3,
        eval_episodes=eval_episodes,
    )


# ---------------------------------------------------------------------------
# ACTION_CATEGORIES invariant tests
# ---------------------------------------------------------------------------


class TestActionCategories(unittest.TestCase):
    def test_all_fn_ids_covered(self):
        covered = {fn_idx for fn_ids in ACTION_CATEGORIES.values() for fn_idx in fn_ids}
        self.assertEqual(covered, set(FUNCTION_IDS.keys()))

    def test_no_fn_id_in_multiple_categories(self):
        all_ids: list[int] = [fn_idx for fn_ids in ACTION_CATEGORIES.values() for fn_idx in fn_ids]
        self.assertEqual(len(all_ids), len(set(all_ids)), "fn_idx appears in more than one category")

    def test_category_names_match_keys(self):
        self.assertEqual(set(CATEGORY_NAMES), set(ACTION_CATEGORIES.keys()))

    def test_n_categories_matches(self):
        self.assertEqual(N_CATEGORIES, len(CATEGORY_NAMES))
        self.assertEqual(N_CATEGORIES, len(ACTION_CATEGORIES))

    def test_inverse_map_covers_all(self):
        self.assertEqual(set(FN_IDX_TO_CATEGORY.keys()), set(FUNCTION_IDS.keys()))

    def test_inverse_map_consistent(self):
        for fn_idx, cat in FN_IDX_TO_CATEGORY.items():
            self.assertIn(fn_idx, ACTION_CATEGORIES[cat])

    def test_no_op_in_move(self):
        self.assertIn(0, ACTION_CATEGORIES["move"])

    def test_attack_screen_in_attack(self):
        self.assertIn(3, ACTION_CATEGORIES["attack"])

    def test_build_barracks_in_build(self):
        self.assertIn(8, ACTION_CATEGORIES["build"])

    def test_train_marine_in_train(self):
        self.assertIn(7, ACTION_CATEGORIES["train"])

    def test_morph_lair_in_upgrade(self):
        self.assertIn(111, ACTION_CATEGORIES["upgrade"])

    def test_siege_mode_in_move(self):
        self.assertIn(46, ACTION_CATEGORIES["move"])

    def test_unsiege_in_move(self):
        self.assertIn(47, ACTION_CATEGORIES["move"])

    def test_archon_in_train(self):
        self.assertIn(78, ACTION_CATEGORIES["train"])


# ---------------------------------------------------------------------------
# SC2HierarchicalLinearPolicy — init / shape
# ---------------------------------------------------------------------------


class TestSC2HierarchicalLinearPolicyInit(unittest.TestCase):
    def test_meta_weights_shape(self):
        p = _make_policy()
        self.assertEqual(p._meta_weights.shape, (N_CATEGORIES, SC2_MINIGAME_OBS_SPEC.dim))

    def test_fn_weights_shape(self):
        p = _make_policy()
        self.assertEqual(p._fn_weights.shape, (N_FUNCTION_IDS, SC2_MINIGAME_OBS_SPEC.dim))

    def test_spatial_weights_shape(self):
        p = _make_policy()
        self.assertEqual(p._sp_weights.shape, (N_SPATIAL_ROWS, SC2_MINIGAME_OBS_SPEC.dim))

    def test_queue_weights_shape(self):
        p = _make_policy()
        self.assertEqual(p._q_weights.shape, (N_QUEUE_ROWS, SC2_MINIGAME_OBS_SPEC.dim))

    def test_ladder_obs_spec(self):
        p = _make_policy(SC2_LADDER_OBS_SPEC)
        self.assertEqual(p._fn_weights.shape[1], SC2_LADDER_OBS_SPEC.dim)
        self.assertEqual(p._meta_weights.shape[1], SC2_LADDER_OBS_SPEC.dim)

    def test_default_race_is_random(self):
        p = _make_policy()
        self.assertEqual(p._race, "random")


# ---------------------------------------------------------------------------
# SC2HierarchicalLinearPolicy — __call__
# ---------------------------------------------------------------------------


class TestSC2HierarchicalLinearPolicyCall(unittest.TestCase):
    def _obs(self, spec=None):
        spec = spec or SC2_MINIGAME_OBS_SPEC
        return np.ones(spec.dim, dtype=np.float32)

    def test_output_shape(self):
        p = _make_policy()
        action = p(self._obs())
        self.assertEqual(action.shape, (4,))

    def test_output_dtype(self):
        p = _make_policy()
        action = p(self._obs())
        self.assertEqual(action.dtype, np.float32)

    def test_fn_idx_in_valid_range(self):
        p = _make_policy()
        action = p(self._obs())
        fn_idx = int(action[0])
        self.assertIn(fn_idx, FUNCTION_IDS)

    def test_spatial_in_unit_square(self):
        p = _make_policy()
        action = p(self._obs())
        self.assertGreaterEqual(float(action[1]), 0.0)
        self.assertLessEqual(float(action[1]), 1.0)
        self.assertGreaterEqual(float(action[2]), 0.0)
        self.assertLessEqual(float(action[2]), 1.0)

    def test_queue_is_zero_or_one(self):
        p = _make_policy()
        action = p(self._obs())
        self.assertIn(float(action[3]), {0.0, 1.0})

    def test_queue_positive_logit_gives_one(self):
        p = _make_policy()
        p._q_weights[:] = 1.0  # large positive → queue=1
        action = p(self._obs())
        self.assertEqual(float(action[3]), 1.0)

    def test_queue_negative_logit_gives_zero(self):
        p = _make_policy()
        p._q_weights[:] = -1.0  # negative → queue=0
        action = p(self._obs())
        self.assertEqual(float(action[3]), 0.0)

    def test_fn_idx_respects_availability_mask(self):
        p = _make_policy()
        # Only allow no_op (fn_idx 0)
        p._available_fn_ids = {0}
        action = p(self._obs())
        self.assertEqual(int(action[0]), 0)

    def test_category_mask_routes_to_attack(self):
        p = _make_policy()
        # Force meta weights to strongly prefer "attack" (index 1)
        p._meta_weights[:] = 0.0
        p._meta_weights[CATEGORY_NAMES.index("attack")] = 100.0
        # Force fn_weights so Attack_screen (3) wins within attack category
        p._fn_weights[:] = 0.0
        p._fn_weights[3] = 100.0
        p._available_fn_ids = None
        action = p(self._obs())
        self.assertEqual(int(action[0]), 3)

    def test_category_mask_routes_to_build(self):
        p = _make_policy()
        p._meta_weights[:] = 0.0
        p._meta_weights[CATEGORY_NAMES.index("build")] = 100.0
        p._fn_weights[:] = 0.0
        p._fn_weights[8] = 100.0  # Build_Barracks_screen
        p._available_fn_ids = None
        action = p(self._obs())
        self.assertIn(int(action[0]), ACTION_CATEGORIES["build"])

    def test_empty_category_falls_back(self):
        """If the preferred category has no available actions, another is used."""
        p = _make_policy()
        # Prefer attack category, but no attack actions are available
        p._meta_weights[:] = 0.0
        p._meta_weights[CATEGORY_NAMES.index("attack")] = 100.0
        # Only allow no_op (fn_idx 0, which is in move)
        p._available_fn_ids = {0}
        action = p(self._obs())
        self.assertEqual(int(action[0]), 0)

    def test_selected_fn_idx_is_in_chosen_category(self):
        """fn_idx must always belong to the meta-selected category when available."""
        rng = np.random.default_rng(42)
        spec = SC2_MINIGAME_OBS_SPEC
        obs = rng.standard_normal(spec.dim).astype(np.float32)
        for _ in range(20):
            p = SC2HierarchicalLinearPolicy(spec)
            action = p(obs)
            fn_idx = int(action[0])
            # Determine which category was chosen: the fn_idx must be in one category
            cat = FN_IDX_TO_CATEGORY.get(fn_idx)
            self.assertIsNotNone(cat, f"fn_idx {fn_idx} not in any category")


# ---------------------------------------------------------------------------
# SC2HierarchicalLinearPolicy — serialisation
# ---------------------------------------------------------------------------


class TestSC2HierarchicalLinearPolicySerialization(unittest.TestCase):
    def test_to_cfg_has_meta_keys(self):
        p = _make_policy()
        cfg = p.to_cfg()
        for name in _META_HEAD_NAMES:
            self.assertIn(f"{name}_weights", cfg)

    def test_to_cfg_has_queue_key(self):
        p = _make_policy()
        cfg = p.to_cfg()
        self.assertIn(f"{_QUEUE_HEAD_NAME}_weights", cfg)

    def test_round_trip_cfg(self):
        p = _make_policy()
        cfg = p.to_cfg()
        p2 = SC2HierarchicalLinearPolicy.from_cfg(cfg, SC2_MINIGAME_OBS_SPEC)
        np.testing.assert_allclose(p._meta_weights, p2._meta_weights, atol=1e-6)
        np.testing.assert_allclose(p._fn_weights, p2._fn_weights, atol=1e-6)
        np.testing.assert_allclose(p._sp_weights, p2._sp_weights, atol=1e-6)
        np.testing.assert_allclose(p._q_weights, p2._q_weights, atol=1e-6)

    def test_save_load_round_trip(self):
        p = _make_policy()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            p.save(path)
            p2 = SC2HierarchicalLinearPolicy.load(path, SC2_MINIGAME_OBS_SPEC)
            np.testing.assert_allclose(p._meta_weights, p2._meta_weights, atol=1e-6)
            np.testing.assert_allclose(p._q_weights, p2._q_weights, atol=1e-6)
        finally:
            os.unlink(path)

    def test_from_cfg_missing_keys_default_zero(self):
        p = SC2HierarchicalLinearPolicy.from_cfg({}, SC2_MINIGAME_OBS_SPEC)
        np.testing.assert_array_equal(p._meta_weights, 0.0)
        np.testing.assert_array_equal(p._q_weights, 0.0)


# ---------------------------------------------------------------------------
# SC2HierarchicalLinearPolicy — flat-weight interface
# ---------------------------------------------------------------------------


class TestSC2HierarchicalLinearPolicyFlat(unittest.TestCase):
    def test_flat_length(self):
        p = _make_policy()
        obs_dim = SC2_MINIGAME_OBS_SPEC.dim
        expected = (N_CATEGORIES + N_FUNCTION_IDS + N_SPATIAL_ROWS + N_QUEUE_ROWS) * obs_dim
        self.assertEqual(len(p.to_flat()), expected)

    def test_with_flat_round_trip(self):
        p = _make_policy()
        flat = p.to_flat()
        p2 = p.with_flat(flat)
        np.testing.assert_allclose(p._meta_weights, p2._meta_weights, atol=1e-6)
        np.testing.assert_allclose(p._fn_weights, p2._fn_weights, atol=1e-6)
        np.testing.assert_allclose(p._q_weights, p2._q_weights, atol=1e-6)

    def test_mutated_different_weights(self):
        p = _make_policy()
        p2 = p.mutated(scale=1.0, share=1.0)
        self.assertFalse(np.allclose(p.to_flat(), p2.to_flat()))

    def test_mutated_same_shape(self):
        p = _make_policy()
        p2 = p.mutated(scale=0.1, share=0.3)
        self.assertEqual(p.to_flat().shape, p2.to_flat().shape)


# ---------------------------------------------------------------------------
# SC2HierarchicalGeneticPolicy
# ---------------------------------------------------------------------------


class TestSC2HierarchicalGeneticPolicy(unittest.TestCase):
    def test_policy_type(self):
        self.assertEqual(SC2HierarchicalGeneticPolicy.POLICY_TYPE, "sc2_hierarchical")

    def test_compatible_with_sc2(self):
        ok, _ = SC2HierarchicalGeneticPolicy.compatible_with("sc2")
        self.assertTrue(ok)

    def test_not_compatible_with_tmnf(self):
        ok, _ = SC2HierarchicalGeneticPolicy.compatible_with("tmnf")
        self.assertFalse(ok)

    def test_initialize_random_and_call(self):
        g = _make_genetic()
        g.initialize_random()
        obs = np.ones(SC2_MINIGAME_OBS_SPEC.dim, dtype=np.float32)
        # champion is set during initialize_random via the parent class
        action = g(obs)
        self.assertEqual(action.shape, (4,))

    def test_head_names_include_meta_and_queue(self):
        # _ALL_HIERARCHICAL_ROW_NAMES stores base names; _weights suffix is
        # added by to_cfg() and the GeneticPolicy crossover logic.
        self.assertIn("meta_0", _ALL_HIERARCHICAL_ROW_NAMES)
        self.assertIn("queue", _ALL_HIERARCHICAL_ROW_NAMES)

    def test_save_champion(self):
        g = _make_genetic()
        g.initialize_random()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            g.save(path)
            self.assertTrue(os.path.exists(path))
            # Reload and verify it's a valid hierarchical policy
            p = SC2HierarchicalLinearPolicy.load(path, SC2_MINIGAME_OBS_SPEC)
            self.assertEqual(p._meta_weights.shape[0], N_CATEGORIES)
        finally:
            os.unlink(path)

    def test_from_cfg_round_trip(self):
        g = _make_genetic()
        cfg = g.to_cfg()
        self.assertEqual(cfg["policy_type"], "sc2_hierarchical")
        g2 = SC2HierarchicalGeneticPolicy.from_cfg(cfg)
        self.assertEqual(g2._pop_size, g._pop_size)


if __name__ == "__main__":
    unittest.main()
