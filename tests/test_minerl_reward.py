"""Tests for the MineRL reward configuration and calculator."""

from __future__ import annotations

import tempfile
import unittest

import yaml

from games.minerl.reward import MineRLRewardCalculator, MineRLRewardConfig


class TestMineRLRewardConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = MineRLRewardConfig()
        self.assertEqual(cfg.native_reward_scale, 1.0)
        self.assertEqual(cfg.step_penalty, -0.001)
        self.assertEqual(cfg.finish_bonus, 100.0)

    def test_from_yaml(self):
        data = {
            "native_reward_scale": 2.0,
            "step_penalty": -0.01,
            "finish_bonus": 200.0,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name

        cfg = MineRLRewardConfig.from_yaml(path)
        self.assertEqual(cfg.native_reward_scale, 2.0)
        self.assertEqual(cfg.step_penalty, -0.01)
        self.assertEqual(cfg.finish_bonus, 200.0)

    def test_from_yaml_ignores_unknown_keys(self):
        data = {"native_reward_scale": 0.5, "unknown_key": 99}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name

        cfg = MineRLRewardConfig.from_yaml(path)
        self.assertEqual(cfg.native_reward_scale, 0.5)
        # Defaults preserved for keys not in the file.
        self.assertEqual(cfg.step_penalty, -0.001)


class TestMineRLRewardCalculator(unittest.TestCase):
    def _calc(self, **kwargs) -> MineRLRewardCalculator:
        return MineRLRewardCalculator(MineRLRewardConfig(**kwargs))

    def test_native_reward_scaled(self):
        calc = self._calc(native_reward_scale=2.0, step_penalty=0.0, finish_bonus=0.0)
        reward = calc.compute(None, None, False, 0.0, {"native_reward": 3.0})
        self.assertAlmostEqual(reward, 6.0)

    def test_step_penalty_applied_every_step(self):
        calc = self._calc(native_reward_scale=0.0, step_penalty=-0.5, finish_bonus=0.0)
        reward = calc.compute(None, None, False, 0.0, {"native_reward": 0.0})
        self.assertAlmostEqual(reward, -0.5)

    def test_finish_bonus_added_on_termination(self):
        calc = self._calc(native_reward_scale=0.0, step_penalty=0.0, finish_bonus=50.0)
        reward = calc.compute(None, None, True, 1.0, {"native_reward": 0.0})
        self.assertAlmostEqual(reward, 50.0)

    def test_no_finish_bonus_when_not_terminated(self):
        calc = self._calc(native_reward_scale=0.0, step_penalty=0.0, finish_bonus=50.0)
        reward = calc.compute(None, None, False, 1.0, {"native_reward": 0.0})
        self.assertAlmostEqual(reward, 0.0)

    def test_missing_native_reward_defaults_to_zero(self):
        calc = self._calc(native_reward_scale=1.0, step_penalty=0.0, finish_bonus=0.0)
        reward = calc.compute(None, None, False, 0.0, {})
        self.assertAlmostEqual(reward, 0.0)

    def test_reset_clears_episode_state(self):
        calc = self._calc(native_reward_scale=1.0, step_penalty=0.0, finish_bonus=0.0)
        calc.compute(None, None, False, 0.0, {"native_reward": 10.0})
        self.assertAlmostEqual(calc._total_native_reward, 10.0)
        calc.reset()
        self.assertAlmostEqual(calc._total_native_reward, 0.0)


if __name__ == "__main__":
    unittest.main()
