"""Tests for the SC2 reward calculator."""
import os
import tempfile
import unittest

from games.sc2.reward import SC2RewardCalculator, SC2RewardConfig


def _write_yaml(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(content)
    f.close()
    return f.name


class TestSC2RewardConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.score_weight, 1.0)
        self.assertEqual(cfg.win_bonus, 100.0)
        self.assertEqual(cfg.loss_penalty, -100.0)
        self.assertLess(cfg.step_penalty, 0.0)

    def test_from_yaml(self):
        path = _write_yaml("score_weight: 0.5\nwin_bonus: 50.0\n")
        try:
            cfg = SC2RewardConfig.from_yaml(path)
            self.assertEqual(cfg.score_weight, 0.5)
            self.assertEqual(cfg.win_bonus, 50.0)
            # Untouched fields keep defaults.
            self.assertEqual(cfg.loss_penalty, -100.0)
        finally:
            os.unlink(path)

    def test_from_yaml_unknown_key_raises(self):
        path = _write_yaml("unknown_key: 1.0\n")
        try:
            with self.assertRaises(ValueError) as ctx:
                SC2RewardConfig.from_yaml(path)
            self.assertIn("unknown_key", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_from_yaml_loads_bundled_config(self):
        cfg_path = os.path.join(
            os.path.dirname(__file__),
            "..", "games", "sc2", "config", "reward_config.yaml",
        )
        cfg = SC2RewardConfig.from_yaml(cfg_path)
        self.assertIsInstance(cfg.score_weight, float)


class TestSC2RewardCalculator(unittest.TestCase):

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        return SC2RewardCalculator(SC2RewardConfig(**kwargs))

    def test_score_delta_reward(self):
        calc = self._make_calc(
            score_weight=2.0, step_penalty=0.0,
            win_bonus=0.0, loss_penalty=0.0, economy_weight=0.0,
        )
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info={"prev_score": 5.0, "score": 8.0},
        )
        self.assertAlmostEqual(r, 6.0)  # (8 - 5) * 2.0

    def test_step_penalty_only(self):
        calc = self._make_calc(
            score_weight=0.0, step_penalty=-0.5, economy_weight=0.0,
        )
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0},
        )
        self.assertAlmostEqual(r, -0.5)

    def test_step_penalty_scales_with_n_ticks(self):
        calc = self._make_calc(
            score_weight=0.0, step_penalty=-0.5, economy_weight=0.0,
        )
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0},
            n_ticks=4,
        )
        self.assertAlmostEqual(r, -2.0)

    def test_win_bonus(self):
        calc = self._make_calc(
            score_weight=0.0, step_penalty=0.0,
            win_bonus=200.0, loss_penalty=-200.0, economy_weight=0.0,
        )
        r = calc.compute(
            prev_state=None, curr_state=None, finished=True,
            elapsed_s=60.0,
            info={"prev_score": 0.0, "score": 0.0, "player_outcome": 1.0},
        )
        self.assertAlmostEqual(r, 200.0)

    def test_loss_penalty(self):
        calc = self._make_calc(
            score_weight=0.0, step_penalty=0.0,
            win_bonus=200.0, loss_penalty=-200.0, economy_weight=0.0,
        )
        r = calc.compute(
            prev_state=None, curr_state=None, finished=True,
            elapsed_s=60.0,
            info={"prev_score": 0.0, "score": 0.0, "player_outcome": -1.0},
        )
        self.assertAlmostEqual(r, -200.0)

    def test_no_outcome_no_bonus(self):
        """Game ends without explicit outcome (e.g. minigame timeout) → no win/loss."""
        calc = self._make_calc(
            score_weight=0.0, step_penalty=0.0,
            win_bonus=200.0, loss_penalty=-200.0, economy_weight=0.0,
        )
        r = calc.compute(
            prev_state=None, curr_state=None, finished=True,
            elapsed_s=60.0,
            info={"prev_score": 0.0, "score": 0.0, "player_outcome": None},
        )
        self.assertAlmostEqual(r, 0.0)

    def test_economy_weight(self):
        calc = self._make_calc(
            score_weight=0.0, step_penalty=0.0, economy_weight=0.01,
        )
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info={
                "prev_score": 0.0, "score": 0.0,
                "prev_minerals": 100.0, "minerals": 200.0,
                "prev_vespene": 0.0, "vespene": 50.0,
            },
        )
        # delta = (200-100) + (50-0) = 150; reward = 0.01 * 150 = 1.5
        self.assertAlmostEqual(r, 1.5)

    def test_idle_penalty_when_idle(self):
        calc = self._make_calc(
            score_weight=0.0, step_penalty=0.0,
            idle_penalty=-1.0, economy_weight=0.0,
        )
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info={
                "prev_score": 0.0, "score": 0.0,
                "army_count": 0, "food_used": 5, "food_cap": 10,
            },
        )
        self.assertAlmostEqual(r, -1.0)

    def test_idle_penalty_not_applied_when_busy(self):
        calc = self._make_calc(
            score_weight=0.0, step_penalty=0.0,
            idle_penalty=-1.0, economy_weight=0.0,
        )
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info={
                "prev_score": 0.0, "score": 0.0,
                "army_count": 5, "food_used": 5, "food_cap": 10,
            },
        )
        self.assertAlmostEqual(r, 0.0)


class TestSC2IdleBonus(unittest.TestCase):
    """Tests for the idle_bonus reward (issue #127)."""

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg_kwargs = {"score_weight": 0.0, "step_penalty": 0.0,
                      "win_bonus": 0.0, "loss_penalty": 0.0,
                      "economy_weight": 0.0}
        cfg_kwargs.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg_kwargs))

    def _combat_info(self, fn_idx: int, dist: float = 5.0) -> dict:
        """Info dict with a friendly unit at (10, 10) and enemy near it."""
        return {
            "prev_score": 0.0, "score": 0.0,
            "action_fn_idx": fn_idx,
            "screen_self_count":  1.0,
            "screen_enemy_count": 1.0,
            "screen_self_cx":  10.0,
            "screen_self_cy":  10.0,
            "screen_enemy_cx": 10.0 + dist,
            "screen_enemy_cy": 10.0,
        }

    def test_idle_bonus_fires_on_no_op_in_combat_range(self):
        calc = self._make_calc(idle_bonus=2.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info=self._combat_info(fn_idx=0, dist=5.0),
        )
        self.assertAlmostEqual(r, 2.0)

    def test_idle_bonus_skipped_when_action_is_not_no_op(self):
        calc = self._make_calc(idle_bonus=2.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info=self._combat_info(fn_idx=2, dist=5.0),  # Move_screen
        )
        self.assertAlmostEqual(r, 0.0)

    def test_idle_bonus_skipped_when_enemy_out_of_range(self):
        calc = self._make_calc(idle_bonus=2.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info=self._combat_info(fn_idx=0, dist=60.0),  # far away
        )
        self.assertAlmostEqual(r, 0.0)

    def test_idle_bonus_skipped_when_no_enemy_present(self):
        calc = self._make_calc(idle_bonus=2.0)
        info = self._combat_info(fn_idx=0)
        info["screen_enemy_count"] = 0.0
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info=info,
        )
        self.assertAlmostEqual(r, 0.0)

    def test_idle_bonus_skipped_when_no_self_present(self):
        calc = self._make_calc(idle_bonus=2.0)
        info = self._combat_info(fn_idx=0)
        info["screen_self_count"] = 0.0
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info=info,
        )
        self.assertAlmostEqual(r, 0.0)

    def test_idle_bonus_disabled_by_default(self):
        """idle_bonus default is 0.0 — existing experiments unaffected."""
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.idle_bonus, 0.0)

    def test_idle_bonus_scales_with_n_ticks(self):
        calc = self._make_calc(idle_bonus=1.5)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info=self._combat_info(fn_idx=0, dist=5.0),
            n_ticks=4,
        )
        self.assertAlmostEqual(r, 6.0)


if __name__ == "__main__":
    unittest.main()
