"""Tests for the Atari game adapter."""

from __future__ import annotations

import sys
import unittest

from framework.game_adapter import GAME_ADAPTERS


class TestAtariAdapter(unittest.TestCase):
    def _adapter(self):
        return GAME_ADAPTERS["atari"]()

    def test_adapter_registered(self):
        self.assertIn("atari", GAME_ADAPTERS)

    def test_adapter_name_and_config_dir(self):
        a = self._adapter()
        self.assertEqual(a.name, "atari")
        self.assertEqual(a.config_dir, "games/atari/config")

    def test_experiment_dir_embeds_game_policy_and_map(self):
        a = self._adapter()
        d = a.experiment_dir(
            "myrun",
            {"map_name": "Pong-v5", "policy_type": "genetic"},
            None,
        )
        self.assertIn("atari", d)
        self.assertIn("genetic", d)
        self.assertIn("Pong-v5", d)
        self.assertIn("myrun", d)

    def test_track_override_replaces_map_name(self):
        a = self._adapter()
        d = a.experiment_dir(
            "myrun",
            {"map_name": "Pong-v5", "policy_type": "genetic"},
            "Breakout-v5",
        )
        self.assertIn("Breakout-v5", d)
        self.assertNotIn("Pong-v5", d)

    def test_track_label_default(self):
        a = self._adapter()
        self.assertEqual(a.track_label({"map_name": "Pong-v5"}, None), "Pong-v5")

    def test_track_label_sanitizes_slash(self):
        """``ALE/Pong-v5`` is a valid map_name but contains a slash that
        must not leak into the experiment directory path."""
        a = self._adapter()
        label = a.track_label({"map_name": "ALE/Pong-v5"}, None)
        self.assertNotIn("/", label)
        self.assertEqual(label, "ALE_Pong-v5")

    def test_track_label_falls_back_to_default(self):
        a = self._adapter()
        self.assertEqual(a.track_label({}, None), "Pong-v5")

    def test_build_probe_returns_none(self):
        a = self._adapter()
        self.assertIsNone(a.build_probe({}))

    def test_build_warmup_returns_none(self):
        a = self._adapter()
        self.assertIsNone(a.build_warmup({}))

    def test_decorate_reward_cfg_is_noop(self):
        """Atari adapter has no game-specific reward decoration."""
        a = self._adapter()
        cfg = {"native_reward_scale": 1.0}
        a.decorate_reward_cfg(cfg, {"map_name": "Pong-v5"}, None)
        self.assertEqual(cfg, {"native_reward_scale": 1.0})

    def test_experiment_dir_root(self):
        a = self._adapter()
        root = a.experiment_dir_root({"map_name": "Pong-v5", "policy_type": "genetic"}, None)
        self.assertIn("atari", root)
        self.assertIn("Pong-v5", root)


class TestAtariGameSpec(unittest.TestCase):
    def _build_spec(self, policy_type="genetic"):
        return GAME_ADAPTERS["atari"]().build_game_spec(
            experiment_name="myrun",
            experiment_dir=f"experiments/atari/{policy_type}/Pong-v5/myrun",
            weights_file=f"experiments/atari/{policy_type}/Pong-v5/myrun/policy_weights.yaml",
            reward_cfg_file=f"experiments/atari/{policy_type}/Pong-v5/myrun/reward_config.yaml",
            training_params={
                "map_name": "Pong-v5",
                "in_game_episode_s": 60.0,
                "policy_type": policy_type,
            },
            track_override=None,
        )

    def test_build_game_spec_returns_atari_obs_spec_and_actions(self):
        from games.atari.obs_spec import ATARI_OBS_SPEC

        spec = self._build_spec()
        self.assertEqual(spec.game_name, "atari")
        self.assertEqual(spec.head_names, ["action"])
        self.assertIs(spec.obs_spec, ATARI_OBS_SPEC)
        self.assertEqual(spec.discrete_actions.shape, (18, 1))
        self.assertTrue(callable(spec.make_env_fn))
        self.assertTrue(callable(spec.save_results_fn))

    def test_build_game_spec_render_mode_forwarded(self):
        """render_mode from training_params is forwarded to make_env()."""
        from unittest.mock import MagicMock, patch

        a = GAME_ADAPTERS["atari"]()
        with patch("games.atari.adapter.make_env") as mock_make_env:
            mock_make_env.return_value = MagicMock()
            spec = a.build_game_spec(
                experiment_name="myrun",
                experiment_dir="experiments/atari/genetic/Pong-v5/myrun",
                weights_file="experiments/atari/genetic/Pong-v5/myrun/policy_weights.yaml",
                reward_cfg_file="experiments/atari/genetic/Pong-v5/myrun/reward_config.yaml",
                training_params={
                    "map_name": "Pong-v5",
                    "in_game_episode_s": 60.0,
                    "policy_type": "genetic",
                    "render_mode": "human",
                },
                track_override=None,
            )
            spec.make_env_fn()
            mock_make_env.assert_called_once()
            _, kwargs = mock_make_env.call_args
            self.assertEqual(kwargs.get("render_mode"), "human")

    def test_build_game_spec_registers_atari_policies(self):
        """build_game_spec must trigger the side-effect import of games.atari.policies.

        Force a fresh import by temporarily removing the module from sys.modules
        so the test is not order-dependent on other test files that may have
        already imported games.atari.policies as a side effect.
        """
        from framework.policies import POLICY_REGISTRY

        # Remove so the next build_game_spec() call re-imports it.
        sys.modules.pop("games.atari.policies", None)
        self._build_spec()
        self.assertIn("games.atari.policies", sys.modules, "build_game_spec() must import games.atari.policies")
        for policy_type in ("neural_dqn", "reinforce", "lstm"):
            self.assertIn(
                policy_type,
                POLICY_REGISTRY,
                f"Expected '{policy_type}' to be registered after build_game_spec()",
            )


if __name__ == "__main__":
    unittest.main()
