"""Tests for the Rocket League reward calculator."""

import os
import tempfile
import unittest

from games.rocket_league.reward import RocketLeagueRewardConfig, RocketLeagueRewardCalculator


def _write_yaml(content: str) -> str:
    """Write content to a temp YAML file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(content)
    f.close()
    return f.name


class TestRocketLeagueRewardConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = RocketLeagueRewardConfig()
        self.assertGreater(cfg.goal_weight, 0.0)
        self.assertGreater(cfg.concede_penalty, 0.0)
        self.assertLessEqual(cfg.step_penalty, 0.0)
        self.assertGreaterEqual(cfg.touch_bonus, 0.0)

    def test_custom_values(self):
        cfg = RocketLeagueRewardConfig(goal_weight=20.0, step_penalty=-0.01)
        self.assertEqual(cfg.goal_weight, 20.0)
        self.assertEqual(cfg.step_penalty, -0.01)

    def test_from_yaml(self):
        path = _write_yaml("goal_weight: 50.0\nstep_penalty: -0.05\n")
        try:
            cfg = RocketLeagueRewardConfig.from_yaml(path)
            self.assertEqual(cfg.goal_weight, 50.0)
            self.assertEqual(cfg.step_penalty, -0.05)
            # Other fields keep defaults
            self.assertGreater(cfg.touch_bonus, 0.0)
        finally:
            os.unlink(path)

    def test_from_yaml_unknown_key_raises(self):
        path = _write_yaml("unknown_key: 1.0\n")
        try:
            with self.assertRaises(ValueError) as ctx:
                RocketLeagueRewardConfig.from_yaml(path)
            self.assertIn("unknown_key", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_from_yaml_loads_bundled_config(self):
        """Ensure the bundled Rocket League reward config loads without error."""
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..",
            "games", "rocket_league", "config", "reward_config.yaml",
        )
        cfg = RocketLeagueRewardConfig.from_yaml(cfg_path)
        self.assertIsInstance(cfg.goal_weight, float)


class TestRocketLeagueRewardCalculator(unittest.TestCase):

    def _make_calc(self, **kwargs) -> RocketLeagueRewardCalculator:
        return RocketLeagueRewardCalculator(RocketLeagueRewardConfig(**kwargs))

    def test_step_penalty(self):
        calc = self._make_calc(
            vel_to_ball_weight=0.0, boost_weight=0.0, touch_bonus=0.0,
            goal_weight=0.0, concede_penalty=0.0, step_penalty=-0.5,
        )
        r = calc.compute(None, None, False, 1.0, {})
        self.assertAlmostEqual(r, -0.5)

    def test_vel_to_ball_reward(self):
        calc = self._make_calc(
            vel_to_ball_weight=0.01, boost_weight=0.0, touch_bonus=0.0,
            goal_weight=0.0, concede_penalty=0.0, step_penalty=0.0,
        )
        r = calc.compute(None, None, False, 1.0, {"vel_towards_ball": 100.0})
        self.assertAlmostEqual(r, 1.0)  # 0.01 * 100

    def test_touch_bonus_fires_once(self):
        calc = self._make_calc(
            vel_to_ball_weight=0.0, boost_weight=0.0, touch_bonus=5.0,
            goal_weight=0.0, concede_penalty=0.0, step_penalty=0.0,
        )
        r1 = calc.compute(None, None, False, 1.0, {"ball_touched": True})
        r2 = calc.compute(None, None, False, 2.0, {"ball_touched": True})
        self.assertAlmostEqual(r1, 5.0)
        self.assertAlmostEqual(r2, 0.0)  # second touch in same ep: no bonus

    def test_touch_bonus_resets_each_episode(self):
        calc = self._make_calc(
            vel_to_ball_weight=0.0, boost_weight=0.0, touch_bonus=5.0,
            goal_weight=0.0, concede_penalty=0.0, step_penalty=0.0,
        )
        calc.compute(None, None, False, 1.0, {"ball_touched": True})  # ep 1
        calc.reset()
        r = calc.compute(None, None, False, 1.0, {"ball_touched": True})  # ep 2
        self.assertAlmostEqual(r, 5.0)

    def test_goal_scored_reward(self):
        calc = self._make_calc(
            vel_to_ball_weight=0.0, boost_weight=0.0, touch_bonus=0.0,
            goal_weight=10.0, concede_penalty=0.0, step_penalty=0.0,
        )
        r = calc.compute(None, None, True, 10.0, {"goal_scored": True})
        self.assertAlmostEqual(r, 10.0)

    def test_goal_conceded_penalty(self):
        calc = self._make_calc(
            vel_to_ball_weight=0.0, boost_weight=0.0, touch_bonus=0.0,
            goal_weight=0.0, concede_penalty=5.0, step_penalty=0.0,
        )
        r = calc.compute(None, None, True, 10.0, {"goal_conceded": True})
        self.assertAlmostEqual(r, -5.0)

    def test_boost_weight(self):
        calc = self._make_calc(
            vel_to_ball_weight=0.0, boost_weight=2.0, touch_bonus=0.0,
            goal_weight=0.0, concede_penalty=0.0, step_penalty=0.0,
        )
        r = calc.compute(None, None, False, 1.0, {"boosting": True})
        self.assertAlmostEqual(r, 2.0)

    def test_combined_rewards(self):
        """Verify additive nature of all reward components."""
        calc = self._make_calc(
            vel_to_ball_weight=0.01, boost_weight=1.0, touch_bonus=3.0,
            goal_weight=10.0, concede_penalty=0.0, step_penalty=-0.1,
        )
        r = calc.compute(None, None, True, 5.0, {
            "vel_towards_ball": 200.0,
            "boosting": True,
            "ball_touched": True,
            "goal_scored": True,
        })
        # 0.01*200 + 1.0 + 3.0 + 10.0 - 0.1 = 15.9
        self.assertAlmostEqual(r, 15.9)


if __name__ == "__main__":
    unittest.main()
