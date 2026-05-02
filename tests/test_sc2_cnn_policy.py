"""Tests for the SC2 CNN policy and spatial observation pipeline.

All tests mock PySC2 / the SC2Client so they run without a game binary.
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from games.sc2.cnn_policy import (
    SC2CNNModel,
    SC2CNNEvolutionPolicy,
    _conv2d_valid_relu,
    _adaptive_avg_pool,
)
from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(n_channels: int = 2, obs_dim: int | None = None) -> SC2CNNModel:
    spec = SC2_MINIGAME_OBS_SPEC
    return SC2CNNModel(n_channels=n_channels, obs_spec=spec, seed=0)


def _dict_obs(n_channels: int = 2, h: int = 64, w: int = 64) -> dict:
    spec = SC2_MINIGAME_OBS_SPEC
    return {
        "flat":    np.zeros(spec.dim, dtype=np.float32),
        "spatial": np.random.default_rng(0).random((n_channels, h, w)).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Conv2d + pool helpers
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
        # All pre-ReLU values are negative → output should be all zeros.
        np.testing.assert_array_equal(out, 0.0)


class TestAdaptiveAvgPool(unittest.TestCase):

    def test_output_shape(self):
        x = np.ones((32, 60, 60), dtype=np.float32)
        out = _adaptive_avg_pool(x, 4, 4)
        self.assertEqual(out.shape, (32, 4, 4))

    def test_uniform_input_preserved(self):
        x = np.full((8, 60, 60), 3.0, dtype=np.float32)
        out = _adaptive_avg_pool(x, 4, 4)
        np.testing.assert_allclose(out, 3.0, atol=1e-5)


# ---------------------------------------------------------------------------
# SC2CNNModel
# ---------------------------------------------------------------------------

class TestSC2CNNModelShape(unittest.TestCase):

    def test_flat_dim_formula(self):
        model = _make_model(n_channels=2)
        self.assertEqual(model.to_flat().shape[0], model.flat_dim)

    def test_flat_dim_varies_with_channels(self):
        m1 = _make_model(n_channels=1)
        m2 = _make_model(n_channels=4)
        self.assertLess(m1.flat_dim, m2.flat_dim)

    def test_forward_output_shapes(self):
        model  = _make_model(n_channels=2)
        obs    = _dict_obs(n_channels=2)
        fn_sc, sp_sc = model.forward(obs["spatial"], obs["flat"])
        self.assertEqual(fn_sc.shape, (6,))
        self.assertEqual(sp_sc.shape, (9,))

    def test_callable_returns_4vector(self):
        model  = _make_model(n_channels=2)
        obs    = _dict_obs(n_channels=2)
        action = model(obs)
        self.assertEqual(action.shape, (4,))
        self.assertIsInstance(float(action[0]), float)

    def test_with_flat_roundtrip(self):
        model    = _make_model(n_channels=2)
        flat     = model.to_flat()
        restored = model.with_flat(flat)
        np.testing.assert_array_equal(restored.to_flat(), flat)

    def test_with_flat_wrong_size_raises(self):
        model = _make_model(n_channels=2)
        with self.assertRaises(ValueError):
            model.with_flat(np.zeros(10, dtype=np.float32))

    def test_non_dict_obs_raises(self):
        model = _make_model(n_channels=2)
        with self.assertRaises(TypeError):
            model(np.zeros(13, dtype=np.float32))

    def test_flat_concat_dimension(self):
        """Flat obs and spatial features must be correctly concatenated."""
        model = _make_model(n_channels=2)
        obs   = _dict_obs(n_channels=2)
        # Just check no shape error occurs in forward.
        fn_sc, sp_sc = model.forward(obs["spatial"], obs["flat"])
        self.assertEqual(fn_sc.shape[0], 6)
        self.assertEqual(sp_sc.shape[0], 9)


# ---------------------------------------------------------------------------
# SC2CNNEvolutionPolicy
# ---------------------------------------------------------------------------

class TestSC2CNNEvolutionPolicy(unittest.TestCase):

    def setUp(self):
        self.obs_spec = SC2_MINIGAME_OBS_SPEC
        self.policy = SC2CNNEvolutionPolicy(
            n_channels=2,
            obs_spec=self.obs_spec,
            population_size=4,
            initial_sigma=0.01,
            seed=42,
        )

    def test_population_size(self):
        self.assertEqual(self.policy.population_size, 4)

    def test_sample_population_returns_correct_count(self):
        pop = self.policy.sample_population()
        self.assertEqual(len(pop), 4)
        for ind in pop:
            self.assertIsInstance(ind, SC2CNNModel)

    def test_individuals_are_callable(self):
        pop = self.policy.sample_population()
        obs = _dict_obs(n_channels=2)
        for ind in pop:
            action = ind(obs)
            self.assertEqual(action.shape, (4,))

    def test_update_distribution_returns_bool(self):
        self.policy.sample_population()
        rewards = [1.0, 2.0, 0.5, 3.0]
        improved = self.policy.update_distribution(rewards)
        self.assertIsInstance(improved, bool)

    def test_champion_improves_after_good_reward(self):
        self.policy.sample_population()
        self.policy.update_distribution([10.0, 1.0, 2.0, 3.0])
        self.assertGreater(self.policy.champion_reward, float("-inf"))

    def test_champion_callable_after_update(self):
        self.policy.sample_population()
        self.policy.update_distribution([1.0, 2.0, 3.0, 4.0])
        obs    = _dict_obs(n_channels=2)
        action = self.policy(obs)
        self.assertEqual(action.shape, (4,))

    def test_update_wrong_rewards_count_raises(self):
        self.policy.sample_population()
        with self.assertRaises(ValueError):
            self.policy.update_distribution([1.0, 2.0])

    def test_update_without_sample_raises(self):
        with self.assertRaises(RuntimeError):
            self.policy.update_distribution([1.0, 2.0, 3.0, 4.0])

    def test_sigma_adapts(self):
        sigma_before = self.policy.sigma
        self.policy.sample_population()
        self.policy.update_distribution([100.0, 100.0, 100.0, 100.0])
        self.assertNotEqual(self.policy.sigma, sigma_before)

    def test_trainer_state_roundtrip(self, tmp_path: str | None = None):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            state_path = os.path.join(d, "trainer_state.npz")
            self.policy.sample_population()
            self.policy.update_distribution([1.0, 2.0, 3.0, 4.0])
            self.policy.save_trainer_state(state_path)

            policy2 = SC2CNNEvolutionPolicy(
                n_channels=2, obs_spec=self.obs_spec,
                population_size=4, initial_sigma=0.01, seed=0,
            )
            policy2.load_trainer_state(state_path)
            np.testing.assert_array_almost_equal(
                policy2._mean, self.policy._mean
            )

    def test_save_load_champion(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            champ_path = os.path.join(d, "champion.npz")
            self.policy.sample_population()
            self.policy.update_distribution([1.0, 2.0, 3.0, 4.0])
            self.policy.save(champ_path)

            policy2 = SC2CNNEvolutionPolicy(
                n_channels=2, obs_spec=self.obs_spec,
                population_size=4, initial_sigma=0.01, seed=0,
            )
            policy2.load_champion(champ_path)
            obs = _dict_obs(n_channels=2)
            np.testing.assert_array_equal(
                policy2(obs), self.policy(obs)
            )


# ---------------------------------------------------------------------------
# SC2Env dict observation space
# ---------------------------------------------------------------------------

class TestSC2EnvDictObsSpace(unittest.TestCase):

    def _make_env(self, screen_layers, minimap_layers=None):
        from games.sc2.env import SC2Env
        with patch("games.sc2.env.SC2Client"):
            return SC2Env(
                map_name="MoveToBeacon",
                screen_layers=screen_layers,
                minimap_layers=minimap_layers or [],
            )

    def test_flat_obs_space_when_no_layers(self):
        from gymnasium import spaces
        env = self._make_env([])
        self.assertIsInstance(env.observation_space, spaces.Box)

    def test_dict_obs_space_when_layers_given(self):
        from gymnasium import spaces
        env = self._make_env(["player_relative", "selected"])
        self.assertIsInstance(env.observation_space, spaces.Dict)
        self.assertIn("flat", env.observation_space.spaces)
        self.assertIn("spatial", env.observation_space.spaces)

    def test_spatial_shape_matches_channels(self):
        env = self._make_env(["player_relative", "selected"])
        spatial_space = env.observation_space["spatial"]
        self.assertEqual(spatial_space.shape[0], 2)   # 2 channels
        self.assertEqual(spatial_space.shape[1], 64)  # screen H
        self.assertEqual(spatial_space.shape[2], 64)  # screen W

    def test_reset_returns_dict_obs(self):
        from games.sc2.env import SC2Env
        from games.sc2.obs_spec import BASE_OBS_DIM

        with patch("games.sc2.env.SC2Client") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.reset.return_value = (
                np.zeros(BASE_OBS_DIM, dtype=np.float32),
                {
                    "score": 0.0, "minerals": 0.0, "vespene": 0.0,
                    "food_used": 0.0, "food_cap": 0.0, "army_count": 0.0,
                    "spatial_obs": np.zeros((2, 64, 64), dtype=np.float32),
                },
            )
            env = SC2Env(
                map_name="MoveToBeacon",
                screen_layers=["player_relative", "selected"],
            )
            obs, info = env.reset()

        self.assertIsInstance(obs, dict)
        self.assertIn("flat", obs)
        self.assertIn("spatial", obs)
        self.assertEqual(obs["flat"].shape, (BASE_OBS_DIM,))
        self.assertEqual(obs["spatial"].shape, (2, 64, 64))

    def test_reset_fills_zeros_when_no_spatial_in_info(self):
        from games.sc2.env import SC2Env
        from games.sc2.obs_spec import BASE_OBS_DIM

        with patch("games.sc2.env.SC2Client") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.reset.return_value = (
                np.zeros(BASE_OBS_DIM, dtype=np.float32),
                {
                    "score": 0.0, "minerals": 0.0, "vespene": 0.0,
                    "food_used": 0.0, "food_cap": 0.0, "army_count": 0.0,
                    # no "spatial_obs" key
                },
            )
            env = SC2Env(
                map_name="MoveToBeacon",
                screen_layers=["player_relative", "selected"],
            )
            obs, _ = env.reset()

        self.assertIsInstance(obs, dict)
        np.testing.assert_array_equal(obs["spatial"], 0.0)

    def test_step_returns_dict_obs(self):
        from games.sc2.env import SC2Env
        from games.sc2.obs_spec import BASE_OBS_DIM

        with patch("games.sc2.env.SC2Client") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.reset.return_value = (
                np.zeros(BASE_OBS_DIM, dtype=np.float32),
                {"score": 0.0, "minerals": 0.0, "vespene": 0.0,
                 "food_used": 0.0, "food_cap": 0.0, "army_count": 0.0,
                 "spatial_obs": np.zeros((2, 64, 64), dtype=np.float32)},
            )
            mock_client.step.return_value = (
                np.zeros(BASE_OBS_DIM, dtype=np.float32),
                0.0,
                False,
                {"score": 1.0, "minerals": 0.0, "vespene": 0.0,
                 "food_used": 0.0, "food_cap": 0.0, "army_count": 0.0,
                 "spatial_obs": np.ones((2, 64, 64), dtype=np.float32)},
            )
            env = SC2Env(
                map_name="MoveToBeacon",
                screen_layers=["player_relative", "selected"],
            )
            env.reset()
            obs, reward, terminated, truncated, info = env.step(
                np.zeros(4, dtype=np.float32)
            )

        self.assertIsInstance(obs, dict)
        self.assertIn("spatial", obs)


# ---------------------------------------------------------------------------
# SC2Client spatial obs extraction
# ---------------------------------------------------------------------------

class TestSC2ClientSpatialObs(unittest.TestCase):

    @staticmethod
    def _fake_feature_screen(layer_data: dict) -> np.ndarray:
        """Return an ndarray subclass that also supports string indexing.

        ``_safe_array`` in SC2Client passes the result through
        ``np.asarray()``, so the object must pass ``isinstance(x, np.ndarray)``
        — hence the ndarray subclass approach.
        """
        h, w = next(iter(layer_data.values())).shape

        class _FakeScreen(np.ndarray):
            def __new__(cls, d):
                # Backing storage is a zero float array — we only care about
                # string-keyed access.
                obj = np.zeros((len(d), h, w), dtype=np.float32).view(cls)
                obj._layer_data = d
                return obj

            def __array_finalize__(self, obj):
                self._layer_data = getattr(obj, "_layer_data", {})

            def __getitem__(self, key):
                if isinstance(key, str):
                    return self._layer_data.get(key, np.zeros((h, w), dtype=np.float32))
                return super().__getitem__(key)

        return _FakeScreen(layer_data)

    def _make_timestep(self, player_relative: np.ndarray) -> MagicMock:
        """Build a minimal mock PySC2 TimeStep with a feature_screen layer."""
        zeros = np.zeros_like(player_relative)
        feat_screen = self._fake_feature_screen({
            "player_relative": player_relative,
            "selected":        zeros,
        })

        player_data = {
            "minerals": 0, "vespene": 0, "food_used": 0, "food_cap": 10,
            "army_count": 0, "idle_worker_count": 0,
            "warp_gate_count": 0, "larva_count": 0,
        }
        player_mock = MagicMock()
        player_mock.__getitem__.side_effect = player_data.__getitem__
        player_mock.get.side_effect = player_data.get

        ob = MagicMock()
        ob_data = {
            "feature_screen":    feat_screen,
            "player":            player_mock,
            "available_actions": np.array([0]),
            "score_cumulative":  np.array([0.0]),
            "single_select":     np.array([]),
            "multi_select":      np.array([]),
        }
        ob.__getitem__.side_effect = ob_data.__getitem__
        ob.get.side_effect = lambda k, d=None: ob_data.get(k, d)

        ts = MagicMock()
        ts.observation = ob
        ts.last.return_value = False
        ts.reward = 0.0
        return ts

    def test_spatial_obs_in_info(self):
        from games.sc2.client import SC2Client
        client = SC2Client(
            map_name="MoveToBeacon",
            screen_layers=["player_relative"],
        )
        pr = np.ones((64, 64), dtype=np.float32) * 2.0
        ts = self._make_timestep(pr)
        _, info = client._timestep_to_obs_info(ts)
        self.assertIn("spatial_obs", info)
        self.assertEqual(info["spatial_obs"].shape, (1, 64, 64))

    def test_spatial_obs_normalised(self):
        from games.sc2.client import SC2Client
        client = SC2Client(
            map_name="MoveToBeacon",
            screen_layers=["player_relative"],
        )
        pr = np.full((64, 64), 4.0, dtype=np.float32)  # max value for player_relative
        ts = self._make_timestep(pr)
        _, info = client._timestep_to_obs_info(ts)
        np.testing.assert_allclose(info["spatial_obs"], 1.0, atol=1e-5)

    def test_no_spatial_obs_when_no_layers(self):
        from games.sc2.client import SC2Client
        client = SC2Client(map_name="MoveToBeacon")
        pr = np.zeros((64, 64), dtype=np.float32)
        ts = self._make_timestep(pr)
        _, info = client._timestep_to_obs_info(ts)
        self.assertNotIn("spatial_obs", info)


if __name__ == "__main__":
    unittest.main()
