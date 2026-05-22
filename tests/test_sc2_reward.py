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
        self.assertEqual(cfg.small_selection_bonus, 0.0)
        self.assertEqual(cfg.early_random_action_bonus, 0.0)
        self.assertLess(cfg.step_penalty, 0.0)
        self.assertGreater(cfg.move_exploration_bonus, 0.0)
        self.assertLess(cfg.move_repeat_penalty, 0.0)
        self.assertLess(cfg.move_self_penalty, 0.0)

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
                      "economy_weight": 0.0,
                      "move_exploration_bonus": 0.0, "move_repeat_penalty": 0.0}
        cfg_kwargs.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg_kwargs))

    def _combat_info(
        self,
        fn_idx: int,
        dist: float = 5.0,
        self_attack_range_px: float | None = None,
    ) -> dict:
        """Info dict with a friendly unit at (10, 10) and enemy near it."""
        out = {
            "prev_score": 0.0, "score": 0.0,
            "action_fn_idx": fn_idx,
            "screen_self_count":  1.0,
            "screen_enemy_count": 1.0,
            "screen_self_cx":  10.0,
            "screen_self_cy":  10.0,
            "screen_enemy_cx": 10.0 + dist,
            "screen_enemy_cy": 10.0,
        }
        if self_attack_range_px is not None:
            out["self_attack_range_px"] = self_attack_range_px
        return out

    def test_idle_bonus_fires_on_no_op_in_combat_range(self):
        calc = self._make_calc(idle_bonus=2.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info=self._combat_info(fn_idx=0, dist=5.0),
        )
        self.assertAlmostEqual(r, 2.0)

    def test_idle_bonus_fires_when_inside_unit_range_margin(self):
        calc = self._make_calc(idle_bonus=2.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info=self._combat_info(
                fn_idx=0,
                dist=19.0,
                self_attack_range_px=20.0,
            ),
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

    def test_idle_bonus_skipped_at_unit_max_range_due_to_inside_margin(self):
        calc = self._make_calc(idle_bonus=2.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info=self._combat_info(
                fn_idx=0,
                dist=20.0,
                self_attack_range_px=20.0,
            ),
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


class TestSC2MoveShaping(unittest.TestCase):
    """Tests for anti-hyperfixation movement shaping."""

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg_kwargs = {
            "score_weight": 0.0,
            "step_penalty": 0.0,
            "win_bonus": 0.0,
            "loss_penalty": 0.0,
            "economy_weight": 0.0,
            "idle_bonus": 0.0,
            "idle_penalty": 0.0,
            "move_exploration_bonus": 1.0,
            "move_repeat_penalty": -2.0,
            "move_self_penalty": -3.0,
        }
        cfg_kwargs.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg_kwargs))

    def _move_info(
        self,
        *,
        x: float,
        y: float,
        prev_x: float | None = None,
        prev_y: float | None = None,
        self_cx: float = 40.0,
        self_cy: float = 40.0,
        self_count: float = 1.0,
    ) -> dict:
        return {
            "prev_score": 0.0,
            "score": 0.0,
            "action_fn_idx": 2,  # Move_screen
            "action_target_x": x,
            "action_target_y": y,
            "prev_move_target_x": prev_x,
            "prev_move_target_y": prev_y,
            "screen_size": 64.0,
            "screen_self_count": self_count,
            "screen_self_cx": self_cx,
            "screen_self_cy": self_cy,
        }

    def test_move_exploration_bonus_first_visit(self):
        """First Move_screen with visible units awards the bonus (new cell)."""
        calc = self._make_calc(move_repeat_penalty=0.0, move_self_penalty=0.0)
        # centroid at (40, 40) → cell (5, 5) in an 8×8 64-px grid; never seen before
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info=self._move_info(x=0.9, y=0.1, self_cx=40.0, self_cy=40.0),
        )
        self.assertAlmostEqual(r, 1.0)

    def test_move_exploration_bonus_second_visit_same_cell(self):
        """Issuing a second Move_screen while units remain in the same cell yields no bonus."""
        calc = self._make_calc(move_repeat_penalty=0.0, move_self_penalty=0.0)
        info = self._move_info(x=0.9, y=0.1, self_cx=40.0, self_cy=40.0)
        calc.compute(prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info)
        r = calc.compute(prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info)
        self.assertAlmostEqual(r, 0.0)

    def test_move_exploration_bonus_new_cell_after_first(self):
        """Moving the unit centroid to a different cell earns a second bonus."""
        calc = self._make_calc(move_repeat_penalty=0.0, move_self_penalty=0.0)
        # first visit — cell (5, 5)
        calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._move_info(x=0.5, y=0.5, self_cx=40.0, self_cy=40.0),
        )
        # second visit — centroid moved to cell (0, 0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._move_info(x=0.5, y=0.5, self_cx=4.0, self_cy=4.0),
        )
        self.assertAlmostEqual(r, 1.0)

    def test_move_exploration_bonus_skips_cells_visited_on_non_move_steps(self):
        """Cells visited during non-move steps are not bonus-eligible later."""
        calc = self._make_calc(move_repeat_penalty=0.0, move_self_penalty=0.0)
        info = self._move_info(x=0.8, y=0.2, self_cx=40.0, self_cy=40.0)
        info["action_fn_idx"] = 0  # no_op
        r_non_move = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(r_non_move, 0.0)

        info["action_fn_idx"] = 2  # Move_screen
        r_move = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(r_move, 0.0)

    def test_exploit_fixed_spam_commands_no_unit_movement(self):
        """Spamming move commands to far-apart targets earns at most one bonus when units don't move."""
        calc = self._make_calc(move_repeat_penalty=0.0, move_self_penalty=0.0)
        # Units stay at centroid (32, 32) — cell (4, 4).
        # Commands alternate to opposite corners of the screen.
        targets = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]
        rewards = []
        prev_x, prev_y = None, None
        for tx, ty in targets:
            rewards.append(
                calc.compute(
                    prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
                    info=self._move_info(
                        x=tx, y=ty, prev_x=prev_x, prev_y=prev_y,
                        self_cx=32.0, self_cy=32.0,
                    ),
                )
            )
            prev_x, prev_y = tx, ty
        # First command visits the cell and earns the bonus; all subsequent ones earn nothing.
        self.assertAlmostEqual(rewards[0], 1.0)
        for r in rewards[1:]:
            self.assertAlmostEqual(r, 0.0)

    def test_move_exploration_bonus_no_units_no_bonus(self):
        """No bonus when no friendly units are visible (screen_self_count == 0)."""
        calc = self._make_calc(move_repeat_penalty=0.0, move_self_penalty=0.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info=self._move_info(x=0.5, y=0.5, self_cx=40.0, self_cy=40.0, self_count=0.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def _move_step(self, calc, *, self_cx, self_cy, fn_idx=2):
        """Run one step at a given centroid; return the move_exploration term."""
        info = self._move_info(x=0.5, y=0.5, self_cx=self_cx, self_cy=self_cy)
        info["action_fn_idx"] = fn_idx
        _, comps = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0, info=info,
        )
        return comps["move_exploration"]

    def test_move_exploration_decay_rerewards_after_stale(self):
        """A cell vacated for longer than the decay window is rewarded again on return."""
        calc = self._make_calc(
            move_repeat_penalty=0.0, move_self_penalty=0.0,
            move_exploration_decay_steps=5,
        )
        # First visit to cell (0, 0) — bonus.
        self.assertAlmostEqual(self._move_step(calc, self_cx=4.0, self_cy=4.0), 1.0)
        # Wander to cell (7, 7) for 6 steps (> decay) so (0, 0) goes stale.
        for _ in range(6):
            self._move_step(calc, self_cx=60.0, self_cy=60.0)
        # Return to (0, 0): it expired, so the bonus fires again.
        self.assertAlmostEqual(self._move_step(calc, self_cx=4.0, self_cy=4.0), 1.0)

    def test_move_exploration_decay_stationary_never_rerewards(self):
        """A centroid that never leaves its cell is rewarded once, even past the decay window."""
        calc = self._make_calc(
            move_repeat_penalty=0.0, move_self_penalty=0.0,
            move_exploration_decay_steps=5,
        )
        rewards = [self._move_step(calc, self_cx=32.0, self_cy=32.0) for _ in range(20)]
        self.assertAlmostEqual(rewards[0], 1.0)
        for r in rewards[1:]:
            self.assertAlmostEqual(r, 0.0)

    def test_move_exploration_decay_zero_is_permanent(self):
        """decay_steps == 0 keeps the once-per-episode behaviour (no re-reward)."""
        calc = self._make_calc(
            move_repeat_penalty=0.0, move_self_penalty=0.0,
            move_exploration_decay_steps=0,
        )
        self.assertAlmostEqual(self._move_step(calc, self_cx=4.0, self_cy=4.0), 1.0)
        for _ in range(50):
            self._move_step(calc, self_cx=60.0, self_cy=60.0)
        # (0, 0) was visited and never expires → no second bonus.
        self.assertAlmostEqual(self._move_step(calc, self_cx=4.0, self_cy=4.0), 0.0)

    def test_move_exploration_grid_size_controls_cell_granularity(self):
        """A coarser grid merges centroids that an 8×8 grid would separate."""
        # On a 2×2 grid (32-px cells) the centroids (4, 4) and (20, 20) share
        # cell (0, 0), so the second move earns no bonus.
        calc = self._make_calc(
            move_repeat_penalty=0.0, move_self_penalty=0.0,
            move_exploration_grid_size=2, move_exploration_decay_steps=0,
        )
        self.assertAlmostEqual(self._move_step(calc, self_cx=4.0, self_cy=4.0), 1.0)
        self.assertAlmostEqual(self._move_step(calc, self_cx=20.0, self_cy=20.0), 0.0)
        # On the default 8×8 grid those same points are cells (0, 0) and (2, 2).
        calc8 = self._make_calc(
            move_repeat_penalty=0.0, move_self_penalty=0.0,
            move_exploration_grid_size=8, move_exploration_decay_steps=0,
        )
        self.assertAlmostEqual(self._move_step(calc8, self_cx=4.0, self_cy=4.0), 1.0)
        self.assertAlmostEqual(self._move_step(calc8, self_cx=20.0, self_cy=20.0), 1.0)

    def test_move_repeat_penalty_for_same_target(self):
        calc = self._make_calc(move_exploration_bonus=0.0, move_self_penalty=0.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info=self._move_info(x=0.5, y=0.5, prev_x=0.5, prev_y=0.5),
        )
        self.assertAlmostEqual(r, -2.0)

    def test_stutter_step_below_threshold_gets_repeat_penalty(self):
        """A tiny non-zero move (below threshold) triggers the repeat penalty."""
        calc = self._make_calc(move_exploration_bonus=0.0, move_self_penalty=0.0)
        # dist = 2/64 ≈ 0.03125 — below threshold, so repeat penalty must fire
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info=self._move_info(x=0.5 + 2.0 / 64.0, y=0.5, prev_x=0.5, prev_y=0.5),
        )
        self.assertAlmostEqual(r, -2.0)

    def test_meaningful_move_at_threshold_no_repeat_penalty(self):
        """A command move at or above the threshold must NOT trigger the repeat penalty."""
        calc = self._make_calc(move_exploration_bonus=0.0, move_self_penalty=0.0)
        threshold = 6.0 / 64.0
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info=self._move_info(x=0.5 + threshold, y=0.5, prev_x=0.5, prev_y=0.5),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_move_self_penalty_when_targeting_friendly_centroid(self):
        calc = self._make_calc(move_exploration_bonus=0.0, move_repeat_penalty=0.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info=self._move_info(x=40.0 / 64.0, y=40.0 / 64.0),
        )
        self.assertAlmostEqual(r, -3.0)

    def test_move_self_penalty_not_applied_without_visible_friendlies(self):
        calc = self._make_calc(move_exploration_bonus=0.0, move_repeat_penalty=0.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info=self._move_info(x=40.0 / 64.0, y=40.0 / 64.0, self_count=0.0),
        )
        self.assertAlmostEqual(r, 0.0)


class TestSC2RewardComponents(unittest.TestCase):
    """Issue #128/2b: per-component reward breakdown."""

    def _calc(self, **kwargs) -> SC2RewardCalculator:
        return SC2RewardCalculator(SC2RewardConfig(**kwargs))

    def test_components_keys_present(self):
        calc = self._calc(score_weight=1.0, economy_weight=0.001,
                          step_penalty=-0.001)
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0,
                  "prev_minerals": 0.0, "minerals": 0.0,
                  "prev_vespene": 0.0, "vespene": 0.0},
        )
        for key in ("score", "economy", "idle_penalty", "idle_bonus",
                    "move_exploration", "move_repeat_penalty", "move_self_penalty",
                    "attack_move_bonus", "click_attack_bonus",
                    "attack_friendly_penalty", "early_random_action", "small_selection",
                    "step_penalty", "terminal"):
            self.assertIn(key, comp)

    def test_components_sum_equals_total(self):
        calc = self._calc(score_weight=2.0, economy_weight=0.01,
                          step_penalty=-0.5, win_bonus=200.0)
        info = {
            "prev_score": 5.0, "score": 8.0,
            "prev_minerals": 100.0, "minerals": 200.0,
            "prev_vespene": 0.0, "vespene": 0.0,
            "player_outcome": 1.0,
        }
        total, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=True, elapsed_s=1.0,
            info=info, n_ticks=2,
        )
        self.assertAlmostEqual(total, sum(comp.values()), places=5)
        # Spot-check individual contributions.
        self.assertAlmostEqual(comp["score"],        6.0)    # 2.0 * (8 - 5)
        self.assertAlmostEqual(comp["economy"],      1.0)    # 0.01 * 100
        self.assertAlmostEqual(comp["step_penalty"], -1.0)   # -0.5 * 2
        self.assertAlmostEqual(comp["terminal"],     200.0)  # win_bonus

    def test_compute_default_delegates_to_with_components(self):
        calc = self._calc(score_weight=3.0, step_penalty=0.0)
        info = {"prev_score": 1.0, "score": 4.0}
        r = calc.compute(prev_state=None, curr_state=None, finished=False,
                         elapsed_s=1.0, info=info)
        r2, _ = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0, info=info,
        )
        self.assertAlmostEqual(r, r2)


class TestSC2AttackMoveBonusAndClickAttackBonus(unittest.TestCase):
    """Tests for the attack-move and click-to-attack reward split."""

    # Screen size used in all helpers below.
    _SS = 64.0

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg_kwargs = {
            "score_weight": 0.0, "step_penalty": 0.0,
            "win_bonus": 0.0, "loss_penalty": 0.0, "economy_weight": 0.0,
        }
        cfg_kwargs.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg_kwargs))

    def _info(
        self,
        fn_idx: int,
        enemy_count: float = 1.0,
        enemy_cx: float = 32.0,
        enemy_cy: float = 32.0,
        target_x_norm: float = 0.5,   # normalised [0,1]
        target_y_norm: float = 0.5,
    ) -> dict:
        return {
            "prev_score": 0.0, "score": 0.0,
            "action_fn_idx": fn_idx,
            "screen_enemy_count": enemy_count,
            "screen_enemy_cx": enemy_cx,
            "screen_enemy_cy": enemy_cy,
            "action_target_x": target_x_norm,
            "action_target_y": target_y_norm,
            "screen_size": self._SS,
        }

    # --- attack_move_bonus ---

    def test_attack_move_bonus_fires_when_target_not_on_enemy(self):
        """Attack_screen to empty ground while enemies visible → attack_move."""
        calc = self._make_calc(attack_move_bonus=1.0)
        # enemy at (32,32), target at (0,0) — far from enemy
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0,
                            target_x_norm=0.0, target_y_norm=0.0),
        )
        self.assertAlmostEqual(r, 1.0)

    def test_attack_move_bonus_skipped_when_no_enemy_on_screen(self):
        calc = self._make_calc(attack_move_bonus=1.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_count=0.0,
                            target_x_norm=0.0, target_y_norm=0.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_attack_move_bonus_skipped_for_move_screen(self):
        """Plain Move_screen does not trigger attack_move_bonus."""
        calc = self._make_calc(attack_move_bonus=1.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=2, target_x_norm=0.0, target_y_norm=0.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_attack_move_bonus_disabled_by_default(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.attack_move_bonus, 0.0)

    def test_attack_move_bonus_scales_with_n_ticks(self):
        calc = self._make_calc(attack_move_bonus=1.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=3, target_x_norm=0.0, target_y_norm=0.0),
            n_ticks=3,
        )
        self.assertAlmostEqual(r, 3.0)

    # --- click_attack_bonus ---

    def test_click_attack_bonus_fires_when_target_on_enemy(self):
        """Attack_screen with target near enemy centroid → click_attack."""
        calc = self._make_calc(click_attack_bonus=2.0)
        # enemy centroid at (32,32), target at norm (0.5,0.5) = pixel (32,32)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0,
                            target_x_norm=0.5, target_y_norm=0.5),
        )
        self.assertAlmostEqual(r, 2.0)

    def test_click_attack_bonus_skipped_when_target_far_from_enemy(self):
        """Target far from enemy centroid → not a click-to-attack → 0."""
        calc = self._make_calc(click_attack_bonus=2.0)
        # enemy at (32,32), target at (0,0) — distance >> click radius
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0,
                            target_x_norm=0.0, target_y_norm=0.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_click_attack_bonus_skipped_when_no_enemy(self):
        calc = self._make_calc(click_attack_bonus=2.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_count=0.0,
                            target_x_norm=0.5, target_y_norm=0.5),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_click_attack_bonus_disabled_by_default(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.click_attack_bonus, 0.0)

    def test_click_attack_bonus_scales_with_n_ticks(self):
        calc = self._make_calc(click_attack_bonus=2.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0,
                            target_x_norm=0.5, target_y_norm=0.5),
            n_ticks=3,
        )
        self.assertAlmostEqual(r, 6.0)

    def test_attack_move_bonus_carries_on_following_no_op_steps(self):
        calc = self._make_calc(attack_move_bonus=1.0, idle_bonus=2.0)
        calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0,
                            target_x_norm=0.0, target_y_norm=0.0),
        )
        r, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=0, enemy_cx=32.0, enemy_cy=32.0),
        )
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(comp["attack_move_bonus"], 1.0)
        self.assertAlmostEqual(comp["idle_bonus"], 0.0)

    def test_click_attack_bonus_carries_on_following_no_op_steps(self):
        calc = self._make_calc(click_attack_bonus=2.0)
        calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0,
                            target_x_norm=0.5, target_y_norm=0.5),
        )
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=0, enemy_cx=32.0, enemy_cy=32.0),
        )
        self.assertAlmostEqual(r, 2.0)

    def test_attack_bonus_carry_stops_on_non_no_op_action(self):
        calc = self._make_calc(attack_move_bonus=1.0)
        calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0,
                            target_x_norm=0.0, target_y_norm=0.0),
        )
        calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=2, target_x_norm=0.2, target_y_norm=0.8),
        )
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=0, enemy_cx=32.0, enemy_cy=32.0),
        )
        self.assertAlmostEqual(r, 0.0)

    # --- cooldown (rapid target switching) ---

    def test_cooldown_default_is_eight(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.click_attack_cooldown_steps, 8)

    def test_same_target_always_fires(self):
        """Clicking the same enemy unit repeatedly is always rewarded."""
        calc = self._make_calc(click_attack_bonus=1.0, click_attack_cooldown_steps=10)
        info = self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0,
                          target_x_norm=0.5, target_y_norm=0.5)
        for _ in range(5):
            r = calc.compute(prev_state=None, curr_state=None, finished=False,
                             elapsed_s=1.0, info=info)
            self.assertAlmostEqual(r, 1.0)

    def test_rapid_switch_withholds_bonus(self):
        """Switching to a new target within cooldown window gets 0."""
        calc = self._make_calc(click_attack_bonus=1.0, click_attack_cooldown_steps=5)
        # First click at centre
        calc.compute(prev_state=None, curr_state=None, finished=False,
                     elapsed_s=1.0,
                     info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0,
                                     target_x_norm=0.5, target_y_norm=0.5))
        # Immediately switch to a very different target (far enemy)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=4.0, enemy_cy=4.0,
                            target_x_norm=0.0625, target_y_norm=0.0625),  # px≈4
        )
        self.assertAlmostEqual(r, 0.0)

    def test_bonus_fires_after_cooldown_elapsed(self):
        """After cooldown_steps of non-attack-screen actions, bonus fires again."""
        cooldown = 4
        calc = self._make_calc(click_attack_bonus=1.0,
                               click_attack_cooldown_steps=cooldown)
        # First click (centre target)
        calc.compute(prev_state=None, curr_state=None, finished=False,
                     elapsed_s=1.0,
                     info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0,
                                     target_x_norm=0.5, target_y_norm=0.5))
        # Advance step count with non-attack actions
        no_attack_info = {
            "prev_score": 0.0, "score": 0.0, "action_fn_idx": 0,
            "screen_enemy_count": 1.0, "screen_size": self._SS,
        }
        for _ in range(cooldown):
            calc.compute(prev_state=None, curr_state=None, finished=False,
                         elapsed_s=1.0, info=no_attack_info)
        # Now click a different enemy target — cooldown expired
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=4.0, enemy_cy=4.0,
                            target_x_norm=0.0625, target_y_norm=0.0625),
        )
        self.assertAlmostEqual(r, 1.0)

    def test_reset_clears_cooldown_state(self):
        """After reset(), the cooldown state is cleared for a new episode."""
        calc = self._make_calc(click_attack_bonus=1.0, click_attack_cooldown_steps=100)
        # Click at centre to prime cooldown
        calc.compute(prev_state=None, curr_state=None, finished=False,
                     elapsed_s=1.0,
                     info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0,
                                     target_x_norm=0.5, target_y_norm=0.5))
        # Reset — starts a fresh episode
        calc.reset()
        # Click a different target immediately after reset — should fire
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=4.0, enemy_cy=4.0,
                            target_x_norm=0.0625, target_y_norm=0.0625),
        )
        self.assertAlmostEqual(r, 1.0)

    # --- attack_friendly_penalty ---

    def _info_with_self(
        self,
        fn_idx: int,
        self_count: float = 1.0,
        self_cx: float = 32.0,
        self_cy: float = 32.0,
        enemy_count: float = 0.0,
        enemy_cx: float = 0.0,
        enemy_cy: float = 0.0,
        target_x_norm: float = 0.5,
        target_y_norm: float = 0.5,
    ) -> dict:
        return {
            "prev_score": 0.0, "score": 0.0,
            "action_fn_idx": fn_idx,
            "screen_self_count": self_count,
            "screen_self_cx": self_cx,
            "screen_self_cy": self_cy,
            "screen_enemy_count": enemy_count,
            "screen_enemy_cx": enemy_cx,
            "screen_enemy_cy": enemy_cy,
            "action_target_x": target_x_norm,
            "action_target_y": target_y_norm,
            "screen_size": self._SS,
        }

    def test_attack_friendly_penalty_default_is_negative(self):
        cfg = SC2RewardConfig()
        self.assertAlmostEqual(cfg.attack_friendly_penalty, -5.0)

    def test_attack_friendly_penalty_fires_when_target_on_friendly(self):
        """Attack_screen aimed at friendly centroid → penalty fires."""
        calc = self._make_calc(attack_friendly_penalty=-5.0)
        # friendly centroid at (32,32), target at norm (0.5,0.5) = pixel (32,32)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info_with_self(fn_idx=3, self_count=1.0,
                                      self_cx=32.0, self_cy=32.0,
                                      target_x_norm=0.5, target_y_norm=0.5),
        )
        self.assertAlmostEqual(r, -5.0)

    def test_attack_friendly_penalty_skipped_when_target_far_from_friendly(self):
        """Attack_screen aimed at empty ground far from friendlies → no penalty."""
        calc = self._make_calc(attack_friendly_penalty=-5.0)
        # friendly at (32,32), target at (0,0) — far away
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info_with_self(fn_idx=3, self_count=1.0,
                                      self_cx=32.0, self_cy=32.0,
                                      target_x_norm=0.0, target_y_norm=0.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_attack_friendly_penalty_skipped_when_no_friendly_on_screen(self):
        """No friendly units visible → no penalty even if target is at centroid."""
        calc = self._make_calc(attack_friendly_penalty=-5.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info_with_self(fn_idx=3, self_count=0.0,
                                      target_x_norm=0.5, target_y_norm=0.5),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_attack_friendly_penalty_skipped_for_move_screen(self):
        """Move_screen (fn_idx 2) does not trigger the friendly-fire penalty."""
        calc = self._make_calc(attack_friendly_penalty=-5.0, move_self_penalty=0.0,
                               move_exploration_bonus=0.0, move_repeat_penalty=0.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info_with_self(fn_idx=2, self_count=1.0,
                                      self_cx=32.0, self_cy=32.0,
                                      target_x_norm=0.5, target_y_norm=0.5),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_attack_friendly_penalty_disabled_when_zero(self):
        """Setting attack_friendly_penalty=0.0 disables the check entirely."""
        calc = self._make_calc(attack_friendly_penalty=0.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info_with_self(fn_idx=3, self_count=1.0,
                                      self_cx=32.0, self_cy=32.0,
                                      target_x_norm=0.5, target_y_norm=0.5),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_attack_friendly_penalty_scales_with_n_ticks(self):
        calc = self._make_calc(attack_friendly_penalty=-5.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info_with_self(fn_idx=3, self_count=1.0,
                                      self_cx=32.0, self_cy=32.0,
                                      target_x_norm=0.5, target_y_norm=0.5),
            n_ticks=3,
        )
        self.assertAlmostEqual(r, -15.0)

    def test_attack_friendly_penalty_in_components(self):
        """attack_friendly_penalty appears as a separate component."""
        calc = self._make_calc(attack_friendly_penalty=-5.0)
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info_with_self(fn_idx=3, self_count=1.0,
                                      self_cx=32.0, self_cy=32.0,
                                      target_x_norm=0.5, target_y_norm=0.5),
        )
        self.assertAlmostEqual(comp["attack_friendly_penalty"], -5.0)

    def test_both_bonuses_exclusive(self):
        """Ground target → attack_move_bonus; on-enemy target → click_attack_bonus."""
        calc = self._make_calc(attack_move_bonus=1.0, click_attack_bonus=2.0)
        # Ground target (far from enemy centroid)
        r_move, comp_move = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0,
                            target_x_norm=0.0, target_y_norm=0.0),
        )
        self.assertAlmostEqual(comp_move["attack_move_bonus"],  1.0)
        self.assertAlmostEqual(comp_move["click_attack_bonus"], 0.0)

        # Click on enemy centroid
        calc.reset()
        r_click, comp_click = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0,
                            target_x_norm=0.5, target_y_norm=0.5),
        )
        self.assertAlmostEqual(comp_click["attack_move_bonus"],  0.0)
        self.assertAlmostEqual(comp_click["click_attack_bonus"], 2.0)


class TestSC2EarlyRandomActionBonus(unittest.TestCase):
    """Tests for the early-random-action exploration bonus."""

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg = {
            "score_weight": 0.0, "step_penalty": 0.0,
            "win_bonus": 0.0, "loss_penalty": 0.0, "economy_weight": 0.0,
        }
        cfg.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg))

    def _info(self, fn_idx: int) -> dict:
        return {"prev_score": 0.0, "score": 0.0, "action_fn_idx": fn_idx}

    def test_fires_for_unseen_non_noop_action_in_window(self):
        calc = self._make_calc(
            early_random_action_bonus=3.0,
            early_random_action_window_steps=10,
        )
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=2),
        )
        self.assertAlmostEqual(comp["early_random_action"], 3.0)

    def test_skips_repeated_action(self):
        calc = self._make_calc(
            early_random_action_bonus=3.0,
            early_random_action_window_steps=10,
        )
        calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=2),
        )
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=2),
        )
        self.assertAlmostEqual(comp["early_random_action"], 0.0)

    def test_skips_actions_after_window(self):
        calc = self._make_calc(
            early_random_action_bonus=3.0,
            early_random_action_window_steps=1,
        )
        calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=0),
        )
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=3),
        )
        self.assertAlmostEqual(comp["early_random_action"], 0.0)


class TestSC2UnitLossPenalty(unittest.TestCase):
    """Tests for the unit_loss_penalty reward term."""

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg = {"score_weight": 0.0, "step_penalty": 0.0,
                "win_bonus": 0.0, "loss_penalty": 0.0, "economy_weight": 0.0}
        cfg.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg))

    def test_penalty_fires_when_units_die(self):
        calc = self._make_calc(unit_loss_penalty=-5.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0,
                  "prev_army_count": 4.0, "army_count": 2.0},
        )
        self.assertAlmostEqual(r, -10.0)  # 2 units lost × -5.0

    def test_penalty_zero_when_no_units_lost(self):
        calc = self._make_calc(unit_loss_penalty=-5.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0,
                  "prev_army_count": 4.0, "army_count": 4.0},
        )
        self.assertAlmostEqual(r, 0.0)

    def test_no_penalty_when_army_grows(self):
        """Producing new units should not yield a penalty."""
        calc = self._make_calc(unit_loss_penalty=-5.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0,
                  "prev_army_count": 2.0, "army_count": 4.0},
        )
        self.assertAlmostEqual(r, 0.0)

    def test_disabled_by_default(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.unit_loss_penalty, 0.0)

    def test_in_components_dict(self):
        calc = self._make_calc(unit_loss_penalty=-5.0)
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0,
                  "prev_army_count": 3.0, "army_count": 2.0},
        )
        self.assertAlmostEqual(comp["unit_loss"], -5.0)


class TestSC2DamageTakenPenalty(unittest.TestCase):
    """Tests for the damage_taken_penalty reward term."""

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg = {"score_weight": 0.0, "step_penalty": 0.0,
                "win_bonus": 0.0, "loss_penalty": 0.0, "economy_weight": 0.0}
        cfg.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg))

    def test_penalty_fires_on_hp_loss(self):
        calc = self._make_calc(damage_taken_penalty=-0.1)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0,
                  "prev_total_self_hp": 100.0, "total_self_hp": 60.0},
        )
        self.assertAlmostEqual(r, -4.0)  # 40 HP lost × -0.1

    def test_no_penalty_when_hp_unchanged(self):
        calc = self._make_calc(damage_taken_penalty=-0.1)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0,
                  "prev_total_self_hp": 100.0, "total_self_hp": 100.0},
        )
        self.assertAlmostEqual(r, 0.0)

    def test_no_penalty_when_hp_increases(self):
        """Healing or new units appearing on-screen should not penalise."""
        calc = self._make_calc(damage_taken_penalty=-0.1)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0,
                  "prev_total_self_hp": 50.0, "total_self_hp": 100.0},
        )
        self.assertAlmostEqual(r, 0.0)

    def test_disabled_by_default(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.damage_taken_penalty, 0.0)

    def test_in_components_dict(self):
        calc = self._make_calc(damage_taken_penalty=-0.5)
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0,
                  "prev_total_self_hp": 80.0, "total_self_hp": 60.0},
        )
        self.assertAlmostEqual(comp["damage_taken"], -10.0)  # 20 × -0.5

    def test_zero_when_info_keys_absent(self):
        """Missing prev/curr HP keys → no penalty (safe default)."""
        calc = self._make_calc(damage_taken_penalty=-0.5)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0},
        )
        self.assertAlmostEqual(r, 0.0)


class TestSC2PassiveUnderFirePenalty(unittest.TestCase):
    """Tests for passive_under_fire_penalty."""

    _SS = 64.0

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg = {
            "score_weight": 0.0, "step_penalty": 0.0,
            "win_bonus": 0.0, "loss_penalty": 0.0, "economy_weight": 0.0,
            # Silence other shaping terms so only passive_under_fire is active.
            "move_exploration_bonus": 0.0, "move_repeat_penalty": 0.0,
            "move_self_penalty": 0.0, "attack_friendly_penalty": 0.0,
            "attack_move_bonus": 0.0, "click_attack_bonus": 0.0,
        }
        cfg.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg))

    def _info(self, fn_idx: int, dist: float = 10.0,
              self_attack_range_px: float | None = None) -> dict:
        out = {
            "prev_score": 0.0, "score": 0.0,
            "action_fn_idx": fn_idx,
            "screen_self_count": 1.0,
            "screen_enemy_count": 1.0,
            "screen_self_cx": 32.0,
            "screen_self_cy": 32.0,
            "screen_enemy_cx": 32.0 + dist,
            "screen_enemy_cy": 32.0,
            "screen_size": self._SS,
        }
        if self_attack_range_px is not None:
            out["self_attack_range_px"] = self_attack_range_px
        return out

    def test_fires_on_no_op_when_enemy_in_range(self):
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=0, dist=10.0),  # no_op, enemy close
        )
        self.assertAlmostEqual(r, -2.0)

    def test_fires_on_move_screen_when_enemy_in_range(self):
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=2, dist=10.0),  # Move_screen, enemy close
        )
        self.assertAlmostEqual(r, -2.0)

    def test_skipped_when_attack_screen_issued(self):
        """Attack_screen (fn_idx 3) suppresses the penalty."""
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=3, dist=10.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_skipped_when_enemy_out_of_range(self):
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=0, dist=60.0),  # enemy far away
        )
        self.assertAlmostEqual(r, 0.0)

    def test_skipped_when_no_enemy_on_screen(self):
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        info = self._info(fn_idx=0, dist=10.0)
        info["screen_enemy_count"] = 0.0
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=info,
        )
        self.assertAlmostEqual(r, 0.0)

    def test_skipped_when_no_self_on_screen(self):
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        info = self._info(fn_idx=0, dist=10.0)
        info["screen_self_count"] = 0.0
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=info,
        )
        self.assertAlmostEqual(r, 0.0)

    def test_uses_self_attack_range_px_when_provided(self):
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        # Enemy at dist=25 px; default range (~20) would miss but explicit 30 catches it.
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=0, dist=25.0, self_attack_range_px=30.0),
        )
        self.assertAlmostEqual(r, -2.0)

    def test_skipped_beyond_explicit_range(self):
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        # Enemy at dist=35 px, explicit range=30 → outside range.
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=0, dist=35.0, self_attack_range_px=30.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_scales_with_n_ticks(self):
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=0, dist=10.0),
            n_ticks=3,
        )
        self.assertAlmostEqual(r, -6.0)

    def test_disabled_by_default(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.passive_under_fire_penalty, 0.0)

    def test_in_components_dict(self):
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=0, dist=10.0),
        )
        self.assertAlmostEqual(comp["passive_under_fire"], -2.0)


class TestSC2SmallSelectionBonus(unittest.TestCase):
    """Tests for the small_selection_bonus reward term."""

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg = {
            "score_weight": 0.0, "step_penalty": 0.0,
            "win_bonus": 0.0, "loss_penalty": 0.0, "economy_weight": 0.0,
        }
        cfg.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg))

    def _info(
        self,
        fn_idx: int = 2,
        selected_count: float = 1.0,
        visible_self_unit_count: float = 4.0,
    ) -> dict:
        return {
            "prev_score": 0.0,
            "score": 0.0,
            "action_fn_idx": fn_idx,
            "selected_count": selected_count,
            "visible_self_unit_count": visible_self_unit_count,
        }

    def test_fires_for_single_selected_unit(self):
        calc = self._make_calc(small_selection_bonus=1.5)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(selected_count=1.0, visible_self_unit_count=6.0),
        )
        self.assertAlmostEqual(r, 1.5)

    def test_fires_when_selection_is_under_half_visible_units(self):
        calc = self._make_calc(small_selection_bonus=1.5)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(selected_count=2.0, visible_self_unit_count=6.0),
        )
        self.assertAlmostEqual(r, 1.5)

    def test_skips_at_exactly_half_selected_units(self):
        calc = self._make_calc(small_selection_bonus=1.5)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(selected_count=2.0, visible_self_unit_count=4.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_skips_for_non_unit_targeted_actions(self):
        calc = self._make_calc(small_selection_bonus=1.5)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(fn_idx=0, selected_count=1.0, visible_self_unit_count=4.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_in_components_dict(self):
        calc = self._make_calc(small_selection_bonus=1.5)
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=self._info(selected_count=1.0, visible_self_unit_count=4.0),
        )
        self.assertAlmostEqual(comp["small_selection"], 1.5)


class TestSC2RewardComponentsExtended(unittest.TestCase):
    """Verify the three new component keys appear in compute_with_components."""

    def test_new_component_keys_present(self):
        calc = SC2RewardCalculator(SC2RewardConfig(
            score_weight=1.0, economy_weight=0.001, step_penalty=-0.001,
            unit_loss_penalty=-1.0, damage_taken_penalty=-0.1,
            passive_under_fire_penalty=-1.0,
        ))
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0,
                  "prev_minerals": 0.0, "minerals": 0.0,
                  "prev_vespene": 0.0, "vespene": 0.0},
        )
        for key in ("unit_loss", "damage_taken", "passive_under_fire"):
            self.assertIn(key, comp)

    def test_components_sum_equals_total_with_new_terms(self):
        calc = SC2RewardCalculator(SC2RewardConfig(
            score_weight=1.0, step_penalty=0.0, economy_weight=0.0,
            unit_loss_penalty=-5.0, damage_taken_penalty=-0.1,
            passive_under_fire_penalty=-2.0,
            win_bonus=0.0, loss_penalty=0.0,
        ))
        info = {
            "prev_score": 0.0, "score": 10.0,
            "prev_army_count": 4.0, "army_count": 3.0,
            "prev_total_self_hp": 100.0, "total_self_hp": 70.0,
            "action_fn_idx": 0,
            "screen_self_count": 1.0, "screen_enemy_count": 1.0,
            "screen_self_cx": 32.0, "screen_self_cy": 32.0,
            "screen_enemy_cx": 42.0, "screen_enemy_cy": 32.0,
            "screen_size": 64.0,
        }
        total, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0,
            info=info,
        )
        self.assertAlmostEqual(total, sum(comp.values()), places=5)


if __name__ == "__main__":
    unittest.main()
