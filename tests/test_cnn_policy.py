"""Tests for framework/cnn_policy.py — shared CNN backbone and generic policy.

Covers:
- _conv2d_valid_relu / _adaptive_avg_pool helpers (same code as SC2 tests; kept
  here so the framework module is tested independently of games/).
- CNNBackbone: shape, param_dim, extract(), to_flat()/with_flat() roundtrip.
- CNNModel: flat_dim formula, forward/call shapes, with_flat roundtrip,
  tanh output range, dict-obs requirement.
- CNNEvolutionPolicy: ES loop, sample_population, update_distribution,
  save/load champion, save/load trainer state.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from framework.cnn_policy import (
    CNNBackbone,
    CNNEvolutionPolicy,
    CNNModel,
    _adaptive_avg_pool,
    _conv2d_valid_relu,
)
from framework.obs_spec import ObsDim, ObsSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 8  # small obs_dim for speed


def _make_obs_spec(dim: int = _DIM) -> ObsSpec:
    dims = [ObsDim(name=f"f{i}", scale=1.0, description="") for i in range(dim)]
    return ObsSpec(dims)


def _make_backbone(n_channels: int = 2, obs_spec: ObsSpec | None = None) -> CNNBackbone:
    spec = obs_spec or _make_obs_spec()
    return CNNBackbone(
        n_channels=n_channels,
        obs_spec=spec,
        conv1_out=4,
        conv2_out=8,
        pool_h=2,
        pool_w=2,
        kernel=3,
        fc_dim=16,
        seed=0,
    )


def _make_model(n_channels: int = 2, n_outputs: int = 3) -> CNNModel:
    spec = _make_obs_spec()
    return CNNModel(
        n_channels=n_channels,
        obs_spec=spec,
        n_outputs=n_outputs,
        conv1_out=4,
        conv2_out=8,
        pool_h=2,
        pool_w=2,
        kernel=3,
        fc_dim=16,
        seed=0,
    )


def _dict_obs(n_channels: int = 2, h: int = 16, w: int = 16) -> dict:
    spec = _make_obs_spec()
    return {
        "flat": np.zeros(spec.dim, dtype=np.float32),
        "spatial": np.random.default_rng(7).random((n_channels, h, w)).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class TestConv2dValidRelu(unittest.TestCase):
    def test_output_shape(self):
        x = np.ones((2, 8, 8), dtype=np.float32)
        W = np.ones((4, 2, 3, 3), dtype=np.float32) * 0.01
        b = np.zeros(4, dtype=np.float32)
        out = _conv2d_valid_relu(x, W, b)
        self.assertEqual(out.shape, (4, 6, 6))

    def test_relu_zeros_negative(self):
        x = np.ones((1, 4, 4), dtype=np.float32)
        W = -np.ones((1, 1, 3, 3), dtype=np.float32)
        b = np.zeros(1, dtype=np.float32)
        out = _conv2d_valid_relu(x, W, b)
        np.testing.assert_array_equal(out, 0.0)


class TestAdaptiveAvgPool(unittest.TestCase):
    def test_output_shape(self):
        x = np.ones((8, 60, 60), dtype=np.float32)
        out = _adaptive_avg_pool(x, 4, 4)
        self.assertEqual(out.shape, (8, 4, 4))

    def test_uniform_input_preserved(self):
        x = np.full((4, 60, 60), 3.0, dtype=np.float32)
        out = _adaptive_avg_pool(x, 4, 4)
        np.testing.assert_allclose(out, 3.0, atol=1e-5)


# ---------------------------------------------------------------------------
# CNNBackbone
# ---------------------------------------------------------------------------


class TestCNNBackbone(unittest.TestCase):
    def test_extract_shape(self):
        bb = _make_backbone(n_channels=2)
        obs = _dict_obs(n_channels=2, h=16, w=16)
        h = bb.extract(obs["spatial"], obs["flat"])
        self.assertEqual(h.shape, (16,))  # fc_dim=16

    def test_param_dim_matches_to_flat(self):
        bb = _make_backbone(n_channels=2)
        self.assertEqual(bb.to_flat().shape[0], bb.param_dim)

    def test_param_dim_varies_with_channels(self):
        bb1 = _make_backbone(n_channels=1)
        bb2 = _make_backbone(n_channels=4)
        self.assertLess(bb1.param_dim, bb2.param_dim)

    def test_with_flat_roundtrip(self):
        bb = _make_backbone()
        flat = bb.to_flat()
        bb2 = bb.with_flat(flat)
        np.testing.assert_array_equal(bb2.to_flat(), flat)

    def test_with_flat_produces_same_output(self):
        bb = _make_backbone()
        obs = _dict_obs()
        flat = bb.to_flat()
        bb2 = bb.with_flat(flat)
        h1 = bb.extract(obs["spatial"], obs["flat"])
        h2 = bb2.extract(obs["spatial"], obs["flat"])
        np.testing.assert_array_almost_equal(h1, h2)


# ---------------------------------------------------------------------------
# CNNModel
# ---------------------------------------------------------------------------


class TestCNNModel(unittest.TestCase):
    def test_flat_dim_matches_to_flat(self):
        m = _make_model(n_channels=2, n_outputs=3)
        self.assertEqual(m.to_flat().shape[0], m.flat_dim)

    def test_flat_dim_varies_with_outputs(self):
        m1 = _make_model(n_outputs=2)
        m2 = _make_model(n_outputs=6)
        self.assertLess(m1.flat_dim, m2.flat_dim)

    def test_forward_shape(self):
        m = _make_model(n_outputs=3)
        obs = _dict_obs()
        logits = m.forward(obs["spatial"], obs["flat"])
        self.assertEqual(logits.shape, (3,))

    def test_call_returns_tanh_range(self):
        m = _make_model(n_outputs=3)
        obs = _dict_obs()
        action = m(obs)
        self.assertEqual(action.shape, (3,))
        self.assertTrue(np.all(action > -1.0))
        self.assertTrue(np.all(action < 1.0))

    def test_with_flat_roundtrip(self):
        m = _make_model()
        flat = m.to_flat()
        m2 = m.with_flat(flat)
        np.testing.assert_array_equal(m2.to_flat(), flat)

    def test_with_flat_wrong_size_raises(self):
        m = _make_model()
        with self.assertRaises(ValueError):
            m.with_flat(np.zeros(5, dtype=np.float32))

    def test_non_dict_obs_raises(self):
        m = _make_model()
        with self.assertRaises(TypeError):
            m(np.zeros(_DIM, dtype=np.float32))

    def test_with_flat_same_output(self):
        m = _make_model()
        obs = _dict_obs()
        flat = m.to_flat()
        m2 = m.with_flat(flat)
        np.testing.assert_array_almost_equal(m(obs), m2(obs))

    def test_episode_hooks_no_error(self):
        m = _make_model()
        obs = _dict_obs()
        m.on_episode_start()
        m.update(obs, np.zeros(3), 1.0, obs, False)
        m.on_episode_end()


# ---------------------------------------------------------------------------
# CNNEvolutionPolicy
# ---------------------------------------------------------------------------


class TestCNNEvolutionPolicy(unittest.TestCase):
    def setUp(self):
        self.spec = _make_obs_spec()
        self.policy = CNNEvolutionPolicy(
            n_channels=2,
            obs_spec=self.spec,
            n_outputs=3,
            population_size=4,
            initial_sigma=0.05,
            seed=42,
            conv1_out=4,
            conv2_out=8,
            pool_h=2,
            pool_w=2,
            kernel=3,
            fc_dim=16,
        )

    def test_population_size_property(self):
        self.assertEqual(self.policy.population_size, 4)

    def test_sample_population_count(self):
        pop = self.policy.sample_population()
        self.assertEqual(len(pop), 4)
        for ind in pop:
            self.assertIsInstance(ind, CNNModel)

    def test_individuals_callable(self):
        pop = self.policy.sample_population()
        obs = _dict_obs()
        for ind in pop:
            action = ind(obs)
            self.assertEqual(action.shape, (3,))

    def test_update_distribution_returns_bool(self):
        self.policy.sample_population()
        improved = self.policy.update_distribution([1.0, 2.0, 0.5, 3.0])
        self.assertIsInstance(improved, bool)

    def test_champion_set_after_update(self):
        self.policy.sample_population()
        self.policy.update_distribution([10.0, 1.0, 2.0, 3.0])
        self.assertGreater(self.policy.champion_reward, float("-inf"))

    def test_policy_callable_after_update(self):
        self.policy.sample_population()
        self.policy.update_distribution([1.0, 2.0, 3.0, 4.0])
        action = self.policy(_dict_obs())
        self.assertEqual(action.shape, (3,))

    def test_sigma_adapts(self):
        sigma0 = self.policy.sigma
        self.policy.sample_population()
        self.policy.update_distribution([100.0, 100.0, 100.0, 100.0])
        self.assertNotEqual(self.policy.sigma, sigma0)

    def test_update_wrong_count_raises(self):
        self.policy.sample_population()
        with self.assertRaises(ValueError):
            self.policy.update_distribution([1.0, 2.0])

    def test_update_without_sample_raises(self):
        with self.assertRaises(RuntimeError):
            self.policy.update_distribution([1.0, 2.0, 3.0, 4.0])

    def test_call_before_update_raises(self):
        with self.assertRaises(RuntimeError):
            self.policy(_dict_obs())

    def test_trainer_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            state_path = os.path.join(d, "trainer_state.npz")
            self.policy.sample_population()
            self.policy.update_distribution([1.0, 2.0, 3.0, 4.0])
            self.policy.save_trainer_state(state_path)

            policy2 = CNNEvolutionPolicy(
                n_channels=2,
                obs_spec=self.spec,
                n_outputs=3,
                population_size=4,
                initial_sigma=0.05,
                seed=0,
                conv1_out=4,
                conv2_out=8,
                pool_h=2,
                pool_w=2,
                kernel=3,
                fc_dim=16,
            )
            policy2.load_trainer_state(state_path)
            np.testing.assert_array_almost_equal(policy2._mean, self.policy._mean)

    def test_save_load_champion(self):
        with tempfile.TemporaryDirectory() as d:
            champ_path = os.path.join(d, "champion.npz")
            self.policy.sample_population()
            self.policy.update_distribution([1.0, 2.0, 3.0, 4.0])
            self.policy.save(champ_path)

            policy2 = CNNEvolutionPolicy(
                n_channels=2,
                obs_spec=self.spec,
                n_outputs=3,
                population_size=4,
                initial_sigma=0.05,
                seed=0,
                conv1_out=4,
                conv2_out=8,
                pool_h=2,
                pool_w=2,
                kernel=3,
                fc_dim=16,
            )
            policy2.load_champion(champ_path)
            obs = _dict_obs()
            np.testing.assert_array_equal(policy2(obs), self.policy(obs))

    def test_compatible_with_non_sc2(self):
        ok, msg = CNNEvolutionPolicy.compatible_with("car_racing")
        self.assertTrue(ok)
        self.assertIsNone(msg)

    def test_not_compatible_with_sc2(self):
        ok, _ = CNNEvolutionPolicy.compatible_with("sc2")
        self.assertFalse(ok)

    def test_policy_type_registered(self):
        from framework.policies import POLICY_REGISTRY

        self.assertIn("cnn", POLICY_REGISTRY)
        self.assertIs(POLICY_REGISTRY["cnn"], CNNEvolutionPolicy)


if __name__ == "__main__":
    unittest.main()
