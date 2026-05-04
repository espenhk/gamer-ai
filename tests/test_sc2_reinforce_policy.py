"""Tests for SC2REINFORCEPolicy in games/sc2/sc2_policies.py.

Covers:
  - Two-head forward pass shapes (fn head / spatial head).
  - Masked fn sampling never selects an unavailable function.
  - Gradients flow to both heads and the trunk after on_episode_end().
  - Entropy term produces different updates than entropy_coeff=0.
  - Serialisation round-trip (to_cfg / from_cfg, save / load, trainer state).
  - available_fn_ids caching via update() kwargs.
"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC, SC2_LADDER_OBS_SPEC
from games.sc2.sc2_policies import (
    N_FUNCTION_IDS,
    N_GRID_CELLS,
    SC2REINFORCEPolicy,
    _GRID_XY,
    _GradEntry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_policy(
    obs_spec=None,
    hidden_sizes: list[int] | None = None,
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    entropy_coeff: float = 0.01,
    baseline: str = "running_mean",
    seed: int = 0,
) -> SC2REINFORCEPolicy:
    spec = obs_spec or SC2_MINIGAME_OBS_SPEC
    return SC2REINFORCEPolicy(
        obs_spec=spec,
        hidden_sizes=hidden_sizes or [16, 8],
        learning_rate=learning_rate,
        gamma=gamma,
        entropy_coeff=entropy_coeff,
        baseline=baseline,
        seed=seed,
    )


def _rand_obs(obs_spec=None, seed: int = 42) -> np.ndarray:
    spec = obs_spec or SC2_MINIGAME_OBS_SPEC
    return np.random.default_rng(seed).standard_normal(spec.dim).astype(np.float32)


# ---------------------------------------------------------------------------
# Forward pass shape tests
# ---------------------------------------------------------------------------

class TestSC2REINFORCEForwardShapes(unittest.TestCase):

    def setUp(self):
        self.policy = _make_policy(seed=0)

    def test_call_returns_4_vector(self):
        obs = _rand_obs()
        act = self.policy(obs)
        self.assertEqual(act.shape, (4,))

    def test_fn_idx_in_valid_range(self):
        obs = _rand_obs()
        for _ in range(30):
            act = self.policy(obs)
            self.assertGreaterEqual(int(act[0]), 0)
            self.assertLess(int(act[0]), N_FUNCTION_IDS)

    def test_spatial_coords_in_unit_range(self):
        obs = _rand_obs()
        for _ in range(30):
            act = self.policy(obs)
            self.assertGreaterEqual(float(act[1]), 0.0)
            self.assertLessEqual(float(act[1]), 1.0)
            self.assertGreaterEqual(float(act[2]), 0.0)
            self.assertLessEqual(float(act[2]), 1.0)

    def test_queue_is_zero(self):
        obs = _rand_obs()
        for _ in range(10):
            act = self.policy(obs)
            self.assertEqual(float(act[3]), 0.0)

    def test_spatial_coords_match_grid_xy(self):
        """Returned (x, y) must correspond to one of the 9 grid cells."""
        obs = _rand_obs()
        for _ in range(20):
            act = self.policy(obs)
            x, y = float(act[1]), float(act[2])
            match = any(
                abs(x - gx) < 1e-5 and abs(y - gy) < 1e-5
                for gx, gy in _GRID_XY
            )
            self.assertTrue(match, f"({x}, {y}) not in grid cells")

    def test_ladder_obs_spec(self):
        p   = _make_policy(obs_spec=SC2_LADDER_OBS_SPEC)
        obs = _rand_obs(obs_spec=SC2_LADDER_OBS_SPEC)
        act = p(obs)
        self.assertEqual(act.shape, (4,))

    def test_trunk_weights_shape(self):
        p = _make_policy(hidden_sizes=[32, 16])
        obs_dim = SC2_MINIGAME_OBS_SPEC.dim
        # trunk layers: obs_dim→32, 32→16
        self.assertEqual(p._trunk_w[0].shape, (32, obs_dim))
        self.assertEqual(p._trunk_w[1].shape, (16, 32))

    def test_fn_head_weight_shape(self):
        p = _make_policy(hidden_sizes=[16])
        self.assertEqual(p._fn_w.shape, (N_FUNCTION_IDS, 16))

    def test_spatial_head_weight_shape(self):
        p = _make_policy(hidden_sizes=[16])
        self.assertEqual(p._sp_w.shape, (N_GRID_CELLS, 16))

    def test_empty_hidden_sizes_uses_obs_dim_directly(self):
        """With hidden_sizes=[], heads connect directly to obs."""
        obs_dim = SC2_MINIGAME_OBS_SPEC.dim
        p = SC2REINFORCEPolicy(obs_spec=SC2_MINIGAME_OBS_SPEC, hidden_sizes=[])
        self.assertEqual(p._fn_w.shape, (N_FUNCTION_IDS, obs_dim))
        self.assertEqual(p._sp_w.shape, (N_GRID_CELLS, obs_dim))
        obs = _rand_obs()
        act = p(obs)
        self.assertEqual(act.shape, (4,))


# ---------------------------------------------------------------------------
# Available-actions masking tests
# ---------------------------------------------------------------------------

class TestSC2REINFORCEMasking(unittest.TestCase):

    def test_masking_never_selects_unavailable_fn(self):
        """When only fn_idx=2 is available, policy must always select it."""
        policy = _make_policy(seed=1)
        obs    = _rand_obs(seed=5)
        # Manually cache available_fn_ids = {2}
        policy._available_fn_ids = {2}
        for _ in range(50):
            act = policy(obs)
            self.assertEqual(int(act[0]), 2,
                             f"Expected fn_idx=2 but got {int(act[0])}")

    def test_masking_with_subset_of_fns(self):
        """With only {0, 1} available, fn_idx must be in {0, 1}."""
        policy = _make_policy(seed=2)
        obs    = _rand_obs(seed=9)
        policy._available_fn_ids = {0, 1}
        for _ in range(50):
            act = policy(obs)
            self.assertIn(int(act[0]), {0, 1})

    def test_no_masking_when_available_fn_ids_none(self):
        """With no masking, all fn_idx values are possible over many samples."""
        policy = _make_policy(seed=3)
        obs    = _rand_obs(seed=11)
        policy._available_fn_ids = None
        seen = set()
        for _ in range(300):
            act = policy(obs)
            seen.add(int(act[0]))
        # With random weights and 300 samples, at least 2 distinct fn_idx expected.
        self.assertGreater(len(seen), 1)

    def test_fallback_to_no_op_when_all_masked(self):
        """If the available set is empty, policy falls back to fn_idx=0 (no_op)."""
        policy = _make_policy(seed=4)
        obs    = _rand_obs(seed=7)
        policy._available_fn_ids = set()  # empty → _build_fn_mask forces mask[0]=True
        for _ in range(20):
            act = policy(obs)
            self.assertEqual(int(act[0]), 0)

    def test_update_caches_available_fn_ids(self):
        """update() with info dict should cache available_fn_ids for next call."""
        policy = _make_policy()
        obs    = _rand_obs()
        policy(obs)
        policy.update(obs, np.zeros(4), 1.0, obs, False,
                      info={"available_fn_ids": {1, 3}})
        self.assertEqual(policy._available_fn_ids, {1, 3})

    def test_update_without_info_leaves_cache_unchanged(self):
        """update() without info kwarg must not alter existing cache."""
        policy = _make_policy()
        policy._available_fn_ids = {0, 2}
        obs = _rand_obs()
        policy(obs)
        policy.update(obs, np.zeros(4), 1.0, obs, False)
        self.assertEqual(policy._available_fn_ids, {0, 2})

    def test_update_with_none_available_fn_ids_does_not_cache(self):
        """info["available_fn_ids"] = None should not overwrite existing cache."""
        policy = _make_policy()
        policy._available_fn_ids = {1}
        obs = _rand_obs()
        policy(obs)
        policy.update(obs, np.zeros(4), 1.0, obs, False,
                      info={"available_fn_ids": None})
        # available is None in info → condition `if available is not None` fails
        self.assertEqual(policy._available_fn_ids, {1})


# ---------------------------------------------------------------------------
# Episode buffer tests
# ---------------------------------------------------------------------------

class TestSC2REINFORCEBuffers(unittest.TestCase):

    def test_episode_buffers_fill_on_call(self):
        policy = _make_policy()
        obs    = _rand_obs()
        for _ in range(5):
            policy(obs)
            policy.update(obs, np.zeros(4), 1.0, obs, False)
        self.assertEqual(len(policy._ep_grads),   5)
        self.assertEqual(len(policy._ep_rewards), 5)

    def test_episode_buffers_clear_after_on_episode_end(self):
        policy = _make_policy()
        obs    = _rand_obs()
        policy(obs)
        policy.update(obs, np.zeros(4), 1.0, obs, True)
        policy.on_episode_end()
        self.assertEqual(len(policy._ep_grads),   0)
        self.assertEqual(len(policy._ep_rewards), 0)

    def test_on_episode_end_empty_buffer_is_noop(self):
        policy = _make_policy()
        policy.on_episode_end()  # should not raise

    def test_on_episode_start_clears_buffers(self):
        policy = _make_policy()
        obs    = _rand_obs()
        policy(obs)
        policy.update(obs, np.zeros(4), 0.5, obs, False)
        policy.on_episode_start()
        self.assertEqual(len(policy._ep_grads),   0)
        self.assertEqual(len(policy._ep_rewards), 0)


# ---------------------------------------------------------------------------
# Gradient / weight update tests
# ---------------------------------------------------------------------------

class TestSC2REINFORCEGradients(unittest.TestCase):

    def _run_episode(self, policy, obs, reward=1.0) -> SC2REINFORCEPolicy:
        policy(obs)
        policy.update(obs, np.zeros(4), reward, obs, True)
        policy.on_episode_end()
        return policy

    def test_trunk_weights_change_after_update(self):
        policy   = _make_policy(learning_rate=1.0, entropy_coeff=0.0, seed=10)
        obs      = _rand_obs(seed=20)
        w_before = [w.copy() for w in policy._trunk_w]
        self._run_episode(policy, obs, reward=10.0)
        changed = any(
            not np.allclose(wb, wa)
            for wb, wa in zip(w_before, policy._trunk_w)
        )
        self.assertTrue(changed, "Trunk weights unchanged after gradient step")

    def test_fn_head_weights_change_after_update(self):
        policy   = _make_policy(learning_rate=1.0, entropy_coeff=0.0, seed=11)
        obs      = _rand_obs(seed=21)
        fn_before = policy._fn_w.copy()
        self._run_episode(policy, obs, reward=10.0)
        self.assertFalse(np.allclose(fn_before, policy._fn_w),
                         "fn_head weights unchanged after gradient step")

    def test_spatial_head_weights_change_after_update(self):
        policy   = _make_policy(learning_rate=1.0, entropy_coeff=0.0, seed=12)
        obs      = _rand_obs(seed=22)
        sp_before = policy._sp_w.copy()
        self._run_episode(policy, obs, reward=10.0)
        self.assertFalse(np.allclose(sp_before, policy._sp_w),
                         "spatial_head weights unchanged after gradient step")

    def test_entropy_coeff_changes_gradient(self):
        """Entropy term should produce different weight updates than entropy=0."""
        obs = _rand_obs(seed=30)

        def _weights_after(entropy_coeff):
            p = _make_policy(learning_rate=0.5, entropy_coeff=entropy_coeff, seed=5)
            p(obs)
            p.update(obs, np.zeros(4), 1.0, obs, True)
            p.on_episode_end()
            return np.concatenate([
                w.ravel() for w in p._trunk_w
            ] + [p._fn_w.ravel(), p._sp_w.ravel()])

        w_no_ent   = _weights_after(0.0)
        w_with_ent = _weights_after(1.0)
        self.assertFalse(np.allclose(w_no_ent, w_with_ent),
                         "Entropy coeff should change gradient direction")

    def test_masking_excludes_unavailable_from_gradient(self):
        """With only fn_idx=0 available, fn_head gradient for idx≥1 must be 0."""
        policy = _make_policy(learning_rate=1.0, entropy_coeff=0.0, seed=7)
        policy._available_fn_ids = {0}
        obs      = _rand_obs(seed=8)
        fn_before = policy._fn_w.copy()
        policy(obs)
        policy.update(obs, np.zeros(4), 5.0, obs, True)
        policy.on_episode_end()
        # Rows 1–5 should not have changed (zero gradient for unavailable).
        np.testing.assert_array_equal(
            fn_before[1:], policy._fn_w[1:],
            err_msg="Unavailable fn rows changed after masked gradient step",
        )

    def test_gradient_direction_increases_selected_fn_prob(self):
        """After training with reward, the selected fn_idx prob should increase."""
        np.random.seed(99)
        policy = _make_policy(
            hidden_sizes=[32], learning_rate=0.5, gamma=1.0,
            entropy_coeff=0.0, baseline="none", seed=0,
        )
        obs = _rand_obs(seed=5)

        # Force fn_idx=1 always by overriding grads after each call.
        target_fn = 1

        def _prob_fn1():
            obs_norm = obs / policy._scales
            h_last, _, _ = policy._trunk_forward(obs_norm)
            fn_logits = policy._fn_w @ h_last + policy._fn_b
            probs = policy._softmax(fn_logits)
            return float(probs[target_fn])

        prob_before = _prob_fn1()

        for _ in range(50):
            policy(obs)
            # Override the cached fn_idx to always be target_fn.
            entry = policy._ep_grads[-1]
            policy._ep_grads[-1] = entry._replace(fn_idx=target_fn)
            policy.update(obs, np.zeros(4), 10.0, obs, True)
            policy.on_episode_end()

        prob_after = _prob_fn1()
        self.assertGreater(prob_after, prob_before,
                           "REINFORCE did not increase prob of rewarded fn_idx")


# ---------------------------------------------------------------------------
# Serialisation tests
# ---------------------------------------------------------------------------

class TestSC2REINFORCESerialization(unittest.TestCase):

    def test_to_cfg_contains_required_keys(self):
        policy = _make_policy()
        cfg    = policy.to_cfg()
        for key in (
            "policy_type", "hidden_sizes", "learning_rate",
            "gamma", "entropy_coeff", "baseline",
            "trunk_weights", "trunk_biases",
            "fn_weights", "fn_biases",
            "sp_weights", "sp_biases",
        ):
            self.assertIn(key, cfg)

    def test_policy_type_string(self):
        policy = _make_policy()
        self.assertEqual(policy.to_cfg()["policy_type"], "sc2_reinforce")

    def test_from_cfg_restores_trunk_weights(self):
        p1  = _make_policy(seed=1)
        cfg = p1.to_cfg()
        p2  = SC2REINFORCEPolicy.from_cfg(cfg, SC2_MINIGAME_OBS_SPEC)
        for w1, w2 in zip(p1._trunk_w, p2._trunk_w):
            np.testing.assert_array_equal(w1, w2)
        for b1, b2 in zip(p1._trunk_b, p2._trunk_b):
            np.testing.assert_array_equal(b1, b2)

    def test_from_cfg_restores_fn_head(self):
        p1  = _make_policy(seed=2)
        cfg = p1.to_cfg()
        p2  = SC2REINFORCEPolicy.from_cfg(cfg, SC2_MINIGAME_OBS_SPEC)
        np.testing.assert_array_equal(p1._fn_w, p2._fn_w)
        np.testing.assert_array_equal(p1._fn_b, p2._fn_b)

    def test_from_cfg_restores_spatial_head(self):
        p1  = _make_policy(seed=3)
        cfg = p1.to_cfg()
        p2  = SC2REINFORCEPolicy.from_cfg(cfg, SC2_MINIGAME_OBS_SPEC)
        np.testing.assert_array_equal(p1._sp_w, p2._sp_w)
        np.testing.assert_array_equal(p1._sp_b, p2._sp_b)

    def test_from_cfg_restores_hyperparams(self):
        p1  = _make_policy(
            hidden_sizes=[32], learning_rate=0.0005,
            gamma=0.98, entropy_coeff=0.1, baseline="none",
        )
        cfg = p1.to_cfg()
        p2  = SC2REINFORCEPolicy.from_cfg(cfg, SC2_MINIGAME_OBS_SPEC)
        self.assertEqual(p2._hidden, [32])
        self.assertAlmostEqual(p2._lr, 0.0005)
        self.assertAlmostEqual(p2._gamma, 0.98)
        self.assertAlmostEqual(p2._entropy_coeff, 0.1)
        self.assertEqual(p2._baseline_type, "none")

    def test_yaml_round_trip(self):
        p1 = _make_policy(seed=4)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            p1.save(path)
            import yaml
            with open(path) as f:
                cfg = yaml.safe_load(f)
            p2 = SC2REINFORCEPolicy.from_cfg(cfg, SC2_MINIGAME_OBS_SPEC)
        finally:
            os.unlink(path)
        for w1, w2 in zip(p1._trunk_w, p2._trunk_w):
            np.testing.assert_array_almost_equal(w1, w2, decimal=5)
        np.testing.assert_array_almost_equal(p1._fn_w, p2._fn_w, decimal=5)
        np.testing.assert_array_almost_equal(p1._sp_w, p2._sp_w, decimal=5)

    def test_trainer_state_round_trip(self):
        policy = _make_policy()
        obs    = _rand_obs()
        # Run a few episodes to shift the baseline.
        for _ in range(3):
            policy(obs)
            policy.update(obs, np.zeros(4), 5.0, obs, True)
            policy.on_episode_end()
        self.assertNotEqual(policy._baseline_val, 0.0,
                            "Expected baseline_val to change after training")
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            policy.save_trainer_state(path)
            p2 = _make_policy()
            p2.load_trainer_state(path)
            self.assertAlmostEqual(policy._baseline_val, p2._baseline_val, places=10)
        finally:
            os.unlink(path)

    def test_trainer_state_obs_dim_mismatch_raises(self):
        p1 = _make_policy(obs_spec=SC2_MINIGAME_OBS_SPEC)
        p2 = _make_policy(obs_spec=SC2_LADDER_OBS_SPEC)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            p1.save_trainer_state(path)
            with self.assertRaises(ValueError):
                p2.load_trainer_state(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
