"""Tests for the Atari reward calculator."""

from __future__ import annotations

import os
import tempfile
import unittest

import yaml

from games.atari.reward import AtariRewardCalculator, AtariRewardConfig


class TestAtariRewardConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = AtariRewardConfig()
        self.assertEqual(cfg.native_reward_scale, 1.0)
        self.assertFalse(cfg.clip_sign)
        self.assertEqual(cfg.step_penalty, 0.0)

    def test_from_yaml_loads_known_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "reward.yaml")
            with open(path, "w") as f:
                yaml.dump(
                    {
                        "native_reward_scale": 2.0,
                        "clip_sign": True,
                        "step_penalty": -0.01,
                    },
                    f,
                )
            cfg = AtariRewardConfig.from_yaml(path)
            self.assertEqual(cfg.native_reward_scale, 2.0)
            self.assertTrue(cfg.clip_sign)
            self.assertEqual(cfg.step_penalty, -0.01)

    def test_from_yaml_ignores_unknown_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "reward.yaml")
            with open(path, "w") as f:
                yaml.dump({"native_reward_scale": 1.5, "bogus_field": 42}, f)
            cfg = AtariRewardConfig.from_yaml(path)
            self.assertEqual(cfg.native_reward_scale, 1.5)


class TestAtariRewardCalculator(unittest.TestCase):
    def test_native_reward_passthrough(self):
        calc = AtariRewardCalculator(AtariRewardConfig())
        r = calc.compute(None, None, False, 0.0, {"native_reward": 3.0})
        self.assertEqual(r, 3.0)

    def test_native_reward_scaled(self):
        calc = AtariRewardCalculator(AtariRewardConfig(native_reward_scale=0.5))
        r = calc.compute(None, None, False, 0.0, {"native_reward": 4.0})
        self.assertEqual(r, 2.0)

    def test_step_penalty_applied(self):
        calc = AtariRewardCalculator(AtariRewardConfig(step_penalty=-0.1))
        r = calc.compute(None, None, False, 0.0, {"native_reward": 1.0})
        self.assertAlmostEqual(r, 0.9)

    def test_clip_sign_positive(self):
        calc = AtariRewardCalculator(AtariRewardConfig(clip_sign=True))
        r = calc.compute(None, None, False, 0.0, {"native_reward": 50.0})
        self.assertEqual(r, 1.0)

    def test_clip_sign_negative(self):
        calc = AtariRewardCalculator(AtariRewardConfig(clip_sign=True))
        r = calc.compute(None, None, False, 0.0, {"native_reward": -50.0})
        self.assertEqual(r, -1.0)

    def test_clip_sign_zero_stays_zero(self):
        calc = AtariRewardCalculator(AtariRewardConfig(clip_sign=True))
        r = calc.compute(None, None, False, 0.0, {"native_reward": 0.0})
        self.assertEqual(r, 0.0)

    def test_missing_native_reward_defaults_to_zero(self):
        calc = AtariRewardCalculator(AtariRewardConfig())
        r = calc.compute(None, None, False, 0.0, {})
        self.assertEqual(r, 0.0)


if __name__ == "__main__":
    unittest.main()
