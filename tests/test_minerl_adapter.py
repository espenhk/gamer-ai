"""Tests for the MineRL game adapter."""

from __future__ import annotations

import unittest

from framework.game_adapter import GAME_ADAPTERS


class TestMineRLAdapter(unittest.TestCase):
    def _adapter(self):
        return GAME_ADAPTERS["minerl"]()

    def test_adapter_registered(self):
        self.assertIn("minerl", GAME_ADAPTERS)

    def test_adapter_name_and_config_dir(self):
        a = self._adapter()
        self.assertEqual(a.name, "minerl")
        self.assertEqual(a.config_dir, "games/minerl/config")

    def test_experiment_dir_embeds_game_policy_and_map(self):
        a = self._adapter()
        d = a.experiment_dir(
            "myrun",
            {"map_name": "MineRLNavigateDense-v0", "policy_type": "genetic"},
            None,
        )
        self.assertIn("minerl", d)
        self.assertIn("genetic", d)
        self.assertIn("MineRLNavigateDense-v0", d)
        self.assertIn("myrun", d)

    def test_track_override_replaces_map_name(self):
        a = self._adapter()
        d = a.experiment_dir(
            "myrun",
            {"map_name": "MineRLNavigateDense-v0", "policy_type": "genetic"},
            "MineRLTreechop-v0",
        )
        self.assertIn("MineRLTreechop-v0", d)
        self.assertNotIn("MineRLNavigateDense-v0", d)

    def test_track_label_default(self):
        a = self._adapter()
        self.assertEqual(
            a.track_label({"map_name": "MineRLNavigateDense-v0"}, None),
            "MineRLNavigateDense-v0",
        )

    def test_track_label_falls_back_when_no_map_name(self):
        a = self._adapter()
        self.assertEqual(a.track_label({}, None), "MineRLNavigateDense-v0")

    def test_track_label_sanitizes_slash(self):
        a = self._adapter()
        label = a.track_label({"map_name": "MineRL/Navigate-v0"}, None)
        self.assertNotIn("/", label)
        self.assertEqual(label, "MineRL_Navigate-v0")

    def test_build_probe_returns_none(self):
        a = self._adapter()
        self.assertIsNone(a.build_probe({}))

    def test_build_warmup_returns_none(self):
        a = self._adapter()
        self.assertIsNone(a.build_warmup({}))

    def test_decorate_reward_cfg_is_noop(self):
        a = self._adapter()
        cfg = {"native_reward_scale": 1.0}
        a.decorate_reward_cfg(cfg, {"map_name": "MineRLNavigateDense-v0"}, None)
        self.assertEqual(cfg, {"native_reward_scale": 1.0})

    def test_experiment_dir_root(self):
        a = self._adapter()
        root = a.experiment_dir_root(
            {"map_name": "MineRLNavigateDense-v0", "policy_type": "genetic"},
            None,
        )
        self.assertIn("minerl", root)
        self.assertIn("MineRLNavigateDense-v0", root)

    def test_build_game_spec_structure(self):
        from games.minerl.obs_spec import MINERL_OBS_SPEC

        a = self._adapter()
        spec = a.build_game_spec(
            experiment_name="myrun",
            experiment_dir="experiments/minerl/genetic/MineRLNavigateDense-v0/myrun",
            weights_file="experiments/minerl/genetic/MineRLNavigateDense-v0/myrun/policy_weights.yaml",
            reward_cfg_file="experiments/minerl/genetic/MineRLNavigateDense-v0/myrun/reward_config.yaml",
            training_params={
                "map_name": "MineRLNavigateDense-v0",
                "in_game_episode_s": 120.0,
                "policy_type": "genetic",
            },
            track_override=None,
        )
        self.assertEqual(spec.game_name, "minerl")
        self.assertEqual(spec.head_names, ["action"])
        self.assertIs(spec.obs_spec, MINERL_OBS_SPEC)
        self.assertEqual(spec.discrete_actions.shape, (9, 1))
        self.assertTrue(callable(spec.make_env_fn))
        self.assertTrue(callable(spec.save_results_fn))


if __name__ == "__main__":
    unittest.main()
