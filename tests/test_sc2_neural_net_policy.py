"""Tests for SC2NeuralNetPolicy in games/sc2/sc2_policies.py."""
from __future__ import annotations

import unittest

import numpy as np

from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC
from games.sc2.sc2_policies import N_FUNCTION_IDS, SC2NeuralNetPolicy


def _make_policy(hidden_sizes: list[int] | None = None) -> SC2NeuralNetPolicy:
    return SC2NeuralNetPolicy(
        obs_spec=SC2_MINIGAME_OBS_SPEC,
        hidden_sizes=hidden_sizes or [16, 8],
    )


class TestSC2NeuralNetPolicy(unittest.TestCase):

    def test_action_shape_and_ranges(self):
        p = _make_policy([8])
        obs = np.random.default_rng(0).standard_normal(SC2_MINIGAME_OBS_SPEC.dim).astype(np.float32)
        act = p(obs)
        self.assertEqual(act.shape, (4,))
        self.assertGreaterEqual(int(act[0]), 0)
        self.assertLess(int(act[0]), N_FUNCTION_IDS)
        self.assertGreaterEqual(float(act[1]), 0.0)
        self.assertLessEqual(float(act[1]), 1.0)
        self.assertGreaterEqual(float(act[2]), 0.0)
        self.assertLessEqual(float(act[2]), 1.0)
        self.assertIn(float(act[3]), {0.0, 1.0})

    def test_deterministic_for_same_observation(self):
        p = _make_policy([12, 6])
        obs = np.random.default_rng(1).standard_normal(SC2_MINIGAME_OBS_SPEC.dim).astype(np.float32)
        np.testing.assert_array_equal(p(obs), p(obs))

    def test_from_cfg_roundtrip(self):
        p1 = _make_policy([32, 16, 8])
        obs = np.random.default_rng(2).standard_normal(SC2_MINIGAME_OBS_SPEC.dim).astype(np.float32)
        p2 = SC2NeuralNetPolicy.from_cfg(p1.to_cfg(), SC2_MINIGAME_OBS_SPEC)
        np.testing.assert_allclose(p1(obs), p2(obs))

    def test_hidden_sizes_preserved_in_cfg(self):
        p = _make_policy([16, 64, 64, 16])
        self.assertEqual(p.to_cfg()["hidden_sizes"], [16, 64, 64, 16])

    def test_weight_shapes_follow_hidden_layers(self):
        p = _make_policy([16, 8])
        cfg = p.to_cfg()
        weights = cfg["weights"]
        obs_dim = SC2_MINIGAME_OBS_SPEC.dim
        self.assertEqual(np.array(weights[0]).shape, (16, obs_dim))
        self.assertEqual(np.array(weights[1]).shape, (8, 16))
        self.assertEqual(np.array(weights[2]).shape, (4, 8))

    def test_mutated_changes_weights(self):
        p = _make_policy([8])
        m = p.mutated(scale=1.0)
        self.assertFalse(np.allclose(p.to_cfg()["weights"][0], m.to_cfg()["weights"][0]))

    def test_masks_unavailable_fn_idx(self):
        p = _make_policy([4])
        p._available_fn_ids = {1}  # noqa: SLF001 - white-box test
        obs = np.zeros(SC2_MINIGAME_OBS_SPEC.dim, dtype=np.float32)
        act = p(obs)
        self.assertEqual(int(act[0]), 1)

    def test_update_caches_available_fn_ids(self):
        p = _make_policy([4])
        obs = np.zeros(SC2_MINIGAME_OBS_SPEC.dim, dtype=np.float32)
        p.update(obs, np.zeros(4, dtype=np.float32), 0.0, obs, False, info={"available_fn_ids": {2, 4}})
        self.assertEqual(p._available_fn_ids, {2, 4})  # noqa: SLF001 - white-box test

    def test_update_missing_available_fn_ids_clears_stale_mask(self):
        p = _make_policy([4])
        p._available_fn_ids = {2, 4}  # noqa: SLF001 - white-box test
        obs = np.zeros(SC2_MINIGAME_OBS_SPEC.dim, dtype=np.float32)
        p.update(obs, np.zeros(4, dtype=np.float32), 0.0, obs, False, info={})
        self.assertIsNone(p._available_fn_ids)  # noqa: SLF001 - white-box test


# ---------------------------------------------------------------------------
# Race mask tests
# ---------------------------------------------------------------------------

class TestSC2NeuralNetPolicyRaceMask(unittest.TestCase):
    """Permanent race mask — wrong-race actions must never be projected."""

    def test_default_race_allows_all_fn_ids(self):
        from games.sc2.actions import fn_ids_for_race
        p = _make_policy()
        self.assertEqual(p._race_fn_ids, fn_ids_for_race("random"))  # noqa: SLF001

    def test_terran_race_fn_ids_excludes_zerg(self):
        from games.sc2.actions import fn_ids_for_race
        p = SC2NeuralNetPolicy(obs_spec=SC2_MINIGAME_OBS_SPEC, race="terran")
        self.assertEqual(p._race_fn_ids, fn_ids_for_race("terran"))  # noqa: SLF001
        # Zerg-only fn_ids (82-117) must NOT appear in Terran race set.
        for i in range(82, 118):
            self.assertNotIn(i, p._race_fn_ids)  # noqa: SLF001

    def test_project_fn_idx_terran_blocks_zerg(self):
        """_project_fn_idx snaps to nearest Terran fn_idx when target is Zerg-only."""
        p = SC2NeuralNetPolicy(obs_spec=SC2_MINIGAME_OBS_SPEC, race="terran")
        # sigmoid(10.0) ≈ 1.0 → fn_raw ≈ 117 (Zerg-only range).
        # The nearest allowed Terran fn_idx is 49 (highest in _TERRAN_FN_IDS).
        fn_idx = p._project_fn_idx(10.0)  # noqa: SLF001 - white-box test
        self.assertNotIn(fn_idx, range(82, 118))

    def test_project_fn_idx_intersection_with_available(self):
        """Race mask AND per-step mask are both applied."""
        p = SC2NeuralNetPolicy(obs_spec=SC2_MINIGAME_OBS_SPEC, race="terran")
        # Allow only fn_idx 0 (no_op) — which is in both Terran race and available.
        p._available_fn_ids = {0}  # noqa: SLF001
        fn_idx = p._project_fn_idx(10.0)
        self.assertEqual(fn_idx, 0)

    def test_to_cfg_includes_race(self):
        p = SC2NeuralNetPolicy(obs_spec=SC2_MINIGAME_OBS_SPEC, race="terran")
        self.assertEqual(p.to_cfg()["race"], "terran")

    def test_from_cfg_restores_race(self):
        p = SC2NeuralNetPolicy(obs_spec=SC2_MINIGAME_OBS_SPEC, race="terran")
        p2 = SC2NeuralNetPolicy.from_cfg(p.to_cfg(), SC2_MINIGAME_OBS_SPEC)
        self.assertEqual(p2._race, "terran")  # noqa: SLF001
        self.assertEqual(p2._race_fn_ids, p._race_fn_ids)  # noqa: SLF001

    def test_mutated_preserves_race(self):
        p = SC2NeuralNetPolicy(obs_spec=SC2_MINIGAME_OBS_SPEC, race="protoss")
        m = p.mutated(scale=0.1)
        self.assertEqual(m._race, "protoss")  # noqa: SLF001
        self.assertEqual(m._race_fn_ids, p._race_fn_ids)  # noqa: SLF001

    def test_construct_or_resume_reads_agent_race(self):
        p = SC2NeuralNetPolicy._construct_or_resume(
            obs_spec=SC2_MINIGAME_OBS_SPEC,
            head_names=["fn_idx", "x", "y", "queue"],
            discrete_actions=None,
            weights_file="/nonexistent/path.yaml",
            policy_params={"_agent_race": "terran"},
            re_initialize=True,
        )
        self.assertEqual(p._race, "terran")  # noqa: SLF001



if __name__ == "__main__":
    unittest.main(verbosity=2)
