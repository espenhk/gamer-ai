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
            "..",
            "games",
            "sc2",
            "config",
            "reward_config.yaml",
        )
        cfg = SC2RewardConfig.from_yaml(cfg_path)
        self.assertIsInstance(cfg.score_weight, float)


class TestSC2RewardCalculator(unittest.TestCase):
    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        return SC2RewardCalculator(SC2RewardConfig(**kwargs))

    def test_score_delta_reward(self):
        calc = self._make_calc(
            score_weight=2.0,
            step_penalty=0.0,
            win_bonus=0.0,
            loss_penalty=0.0,
            economy_weight=0.0,
        )
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 5.0, "score": 8.0},
        )
        self.assertAlmostEqual(r, 6.0)  # (8 - 5) * 2.0

    def test_step_penalty_only(self):
        calc = self._make_calc(
            score_weight=0.0,
            step_penalty=-0.5,
            economy_weight=0.0,
        )
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0},
        )
        self.assertAlmostEqual(r, -0.5)

    def test_step_penalty_scales_with_n_ticks(self):
        calc = self._make_calc(
            score_weight=0.0,
            step_penalty=-0.5,
            economy_weight=0.0,
        )
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0},
            n_ticks=4,
        )
        self.assertAlmostEqual(r, -2.0)

    def test_win_bonus(self):
        calc = self._make_calc(
            score_weight=0.0,
            step_penalty=0.0,
            win_bonus=200.0,
            loss_penalty=-200.0,
            economy_weight=0.0,
        )
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=True,
            elapsed_s=60.0,
            info={"prev_score": 0.0, "score": 0.0, "player_outcome": 1.0},
        )
        self.assertAlmostEqual(r, 200.0)

    def test_loss_penalty(self):
        calc = self._make_calc(
            score_weight=0.0,
            step_penalty=0.0,
            win_bonus=200.0,
            loss_penalty=-200.0,
            economy_weight=0.0,
        )
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=True,
            elapsed_s=60.0,
            info={"prev_score": 0.0, "score": 0.0, "player_outcome": -1.0},
        )
        self.assertAlmostEqual(r, -200.0)

    def test_no_outcome_no_bonus(self):
        """Game ends without explicit outcome (e.g. minigame timeout) → no win/loss."""
        calc = self._make_calc(
            score_weight=0.0,
            step_penalty=0.0,
            win_bonus=200.0,
            loss_penalty=-200.0,
            economy_weight=0.0,
        )
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=True,
            elapsed_s=60.0,
            info={"prev_score": 0.0, "score": 0.0, "player_outcome": None},
        )
        self.assertAlmostEqual(r, 0.0)

    def test_economy_weight(self):
        calc = self._make_calc(
            score_weight=0.0,
            step_penalty=0.0,
            economy_weight=0.01,
        )
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={
                "prev_score": 0.0,
                "score": 0.0,
                "prev_minerals": 100.0,
                "minerals": 200.0,
                "prev_vespene": 0.0,
                "vespene": 50.0,
            },
        )
        # delta = (200-100) + (50-0) = 150; reward = 0.01 * 150 = 1.5
        self.assertAlmostEqual(r, 1.5)

    def test_idle_penalty_when_idle(self):
        calc = self._make_calc(
            score_weight=0.0,
            step_penalty=0.0,
            idle_penalty=-1.0,
            economy_weight=0.0,
        )
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={
                "prev_score": 0.0,
                "score": 0.0,
                "army_count": 0,
                "food_used": 5,
                "food_cap": 10,
            },
        )
        self.assertAlmostEqual(r, -1.0)

    def test_idle_penalty_not_applied_when_busy(self):
        calc = self._make_calc(
            score_weight=0.0,
            step_penalty=0.0,
            idle_penalty=-1.0,
            economy_weight=0.0,
        )
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={
                "prev_score": 0.0,
                "score": 0.0,
                "army_count": 5,
                "food_used": 5,
                "food_cap": 10,
            },
        )
        self.assertAlmostEqual(r, 0.0)


class TestSC2IdleBonus(unittest.TestCase):
    """Tests for the idle_bonus reward (issue #127)."""

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg_kwargs = {
            "score_weight": 0.0,
            "step_penalty": 0.0,
            "win_bonus": 0.0,
            "loss_penalty": 0.0,
            "economy_weight": 0.0,
            "move_exploration_bonus": 0.0,
            "move_repeat_penalty": 0.0,
        }
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
            "prev_score": 0.0,
            "score": 0.0,
            "action_fn_idx": fn_idx,
            "screen_self_count": 1.0,
            "screen_enemy_count": 1.0,
            "screen_self_cx": 10.0,
            "screen_self_cy": 10.0,
            "screen_enemy_cx": 10.0 + dist,
            "screen_enemy_cy": 10.0,
        }
        if self_attack_range_px is not None:
            out["self_attack_range_px"] = self_attack_range_px
        return out

    def test_idle_bonus_fires_on_no_op_in_combat_range(self):
        calc = self._make_calc(idle_bonus=2.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._combat_info(fn_idx=0, dist=5.0),
        )
        self.assertAlmostEqual(r, 2.0)

    def test_idle_bonus_fires_when_inside_unit_range_margin(self):
        calc = self._make_calc(idle_bonus=2.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
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
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._combat_info(fn_idx=2, dist=5.0),  # Move_screen
        )
        self.assertAlmostEqual(r, 0.0)

    def test_idle_bonus_skipped_at_unit_max_range_due_to_inside_margin(self):
        calc = self._make_calc(idle_bonus=2.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
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
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._combat_info(fn_idx=0, dist=60.0),  # far away
        )
        self.assertAlmostEqual(r, 0.0)

    def test_idle_bonus_skipped_when_no_enemy_present(self):
        calc = self._make_calc(idle_bonus=2.0)
        info = self._combat_info(fn_idx=0)
        info["screen_enemy_count"] = 0.0
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=info,
        )
        self.assertAlmostEqual(r, 0.0)

    def test_idle_bonus_skipped_when_no_self_present(self):
        calc = self._make_calc(idle_bonus=2.0)
        info = self._combat_info(fn_idx=0)
        info["screen_self_count"] = 0.0
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
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
            prev_state=None,
            curr_state=None,
            finished=False,
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
            prev_state=None,
            curr_state=None,
            finished=False,
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
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._move_info(x=0.5, y=0.5, self_cx=40.0, self_cy=40.0),
        )
        # second visit — centroid moved to cell (0, 0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._move_info(x=0.5, y=0.5, self_cx=4.0, self_cy=4.0),
        )
        self.assertAlmostEqual(r, 1.0)

    def test_move_exploration_bonus_skips_cells_visited_on_non_move_steps(self):
        """Cells visited during non-move steps are not bonus-eligible later."""
        calc = self._make_calc(move_repeat_penalty=0.0, move_self_penalty=0.0)
        info = self._move_info(x=0.8, y=0.2, self_cx=40.0, self_cy=40.0)
        info["action_fn_idx"] = 0  # no_op
        r_non_move = calc.compute(prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info)
        self.assertAlmostEqual(r_non_move, 0.0)

        info["action_fn_idx"] = 2  # Move_screen
        r_move = calc.compute(prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info)
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
                    prev_state=None,
                    curr_state=None,
                    finished=False,
                    elapsed_s=1.0,
                    info=self._move_info(
                        x=tx,
                        y=ty,
                        prev_x=prev_x,
                        prev_y=prev_y,
                        self_cx=32.0,
                        self_cy=32.0,
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
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._move_info(x=0.5, y=0.5, self_cx=40.0, self_cy=40.0, self_count=0.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def _move_step(self, calc, *, self_cx, self_cy, fn_idx=2):
        """Run one step at a given centroid; return the move_exploration term."""
        info = self._move_info(x=0.5, y=0.5, self_cx=self_cx, self_cy=self_cy)
        info["action_fn_idx"] = fn_idx
        _, comps = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=info,
        )
        return comps["move_exploration"]

    def test_move_exploration_decay_rerewards_after_stale(self):
        """A cell vacated for longer than the decay window is rewarded again on return."""
        calc = self._make_calc(
            move_repeat_penalty=0.0,
            move_self_penalty=0.0,
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
            move_repeat_penalty=0.0,
            move_self_penalty=0.0,
            move_exploration_decay_steps=5,
        )
        rewards = [self._move_step(calc, self_cx=32.0, self_cy=32.0) for _ in range(20)]
        self.assertAlmostEqual(rewards[0], 1.0)
        for r in rewards[1:]:
            self.assertAlmostEqual(r, 0.0)

    def test_move_exploration_decay_zero_is_permanent(self):
        """decay_steps == 0 keeps the once-per-episode behaviour (no re-reward)."""
        calc = self._make_calc(
            move_repeat_penalty=0.0,
            move_self_penalty=0.0,
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
            move_repeat_penalty=0.0,
            move_self_penalty=0.0,
            move_exploration_grid_size=2,
            move_exploration_decay_steps=0,
        )
        self.assertAlmostEqual(self._move_step(calc, self_cx=4.0, self_cy=4.0), 1.0)
        self.assertAlmostEqual(self._move_step(calc, self_cx=20.0, self_cy=20.0), 0.0)
        # On the default 8×8 grid those same points are cells (0, 0) and (2, 2).
        calc8 = self._make_calc(
            move_repeat_penalty=0.0,
            move_self_penalty=0.0,
            move_exploration_grid_size=8,
            move_exploration_decay_steps=0,
        )
        self.assertAlmostEqual(self._move_step(calc8, self_cx=4.0, self_cy=4.0), 1.0)
        self.assertAlmostEqual(self._move_step(calc8, self_cx=20.0, self_cy=20.0), 1.0)

    def test_move_repeat_penalty_for_same_target(self):
        calc = self._make_calc(move_exploration_bonus=0.0, move_self_penalty=0.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._move_info(x=0.5, y=0.5, prev_x=0.5, prev_y=0.5),
        )
        self.assertAlmostEqual(r, -2.0)

    def test_stutter_step_below_threshold_gets_repeat_penalty(self):
        """A tiny non-zero move (below threshold) triggers the repeat penalty."""
        calc = self._make_calc(move_exploration_bonus=0.0, move_self_penalty=0.0)
        # dist = 2/64 ≈ 0.03125 — below threshold, so repeat penalty must fire
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._move_info(x=0.5 + 2.0 / 64.0, y=0.5, prev_x=0.5, prev_y=0.5),
        )
        self.assertAlmostEqual(r, -2.0)

    def test_meaningful_move_at_threshold_no_repeat_penalty(self):
        """A command move at or above the threshold must NOT trigger the repeat penalty."""
        calc = self._make_calc(move_exploration_bonus=0.0, move_self_penalty=0.0)
        threshold = 6.0 / 64.0
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._move_info(x=0.5 + threshold, y=0.5, prev_x=0.5, prev_y=0.5),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_move_self_penalty_when_targeting_friendly_centroid(self):
        calc = self._make_calc(move_exploration_bonus=0.0, move_repeat_penalty=0.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._move_info(x=40.0 / 64.0, y=40.0 / 64.0),
        )
        self.assertAlmostEqual(r, -3.0)

    def test_move_self_penalty_not_applied_without_visible_friendlies(self):
        calc = self._make_calc(move_exploration_bonus=0.0, move_repeat_penalty=0.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._move_info(x=40.0 / 64.0, y=40.0 / 64.0, self_count=0.0),
        )
        self.assertAlmostEqual(r, 0.0)


class TestSC2RewardComponents(unittest.TestCase):
    """Issue #128/2b: per-component reward breakdown."""

    def _calc(self, **kwargs) -> SC2RewardCalculator:
        return SC2RewardCalculator(SC2RewardConfig(**kwargs))

    def test_components_keys_present(self):
        calc = self._calc(score_weight=1.0, economy_weight=0.001, step_penalty=-0.001)
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={
                "prev_score": 0.0,
                "score": 0.0,
                "prev_minerals": 0.0,
                "minerals": 0.0,
                "prev_vespene": 0.0,
                "vespene": 0.0,
            },
        )
        for key in (
            "score",
            "economy",
            "idle_penalty",
            "idle_worker_penalty",
            "idle_bonus",
            "move_exploration",
            "move_repeat_penalty",
            "move_self_penalty",
            "attack_move_bonus",
            "click_attack_bonus",
            "attack_friendly_penalty",
            "early_random_action",
            "small_selection",
            "step_penalty",
            "terminal",
        ):
            self.assertIn(key, comp)

    def test_components_sum_equals_total(self):
        calc = self._calc(score_weight=2.0, economy_weight=0.01, step_penalty=-0.5, win_bonus=200.0)
        info = {
            "prev_score": 5.0,
            "score": 8.0,
            "prev_minerals": 100.0,
            "minerals": 200.0,
            "prev_vespene": 0.0,
            "vespene": 0.0,
            "player_outcome": 1.0,
        }
        total, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=True,
            elapsed_s=1.0,
            info=info,
            n_ticks=2,
        )
        self.assertAlmostEqual(total, sum(comp.values()), places=5)
        # Spot-check individual contributions.
        self.assertAlmostEqual(comp["score"], 6.0)  # 2.0 * (8 - 5)
        self.assertAlmostEqual(comp["economy"], 1.0)  # 0.01 * 100
        self.assertAlmostEqual(comp["step_penalty"], -1.0)  # -0.5 * 2
        self.assertAlmostEqual(comp["terminal"], 200.0)  # win_bonus

    def test_compute_default_delegates_to_with_components(self):
        calc = self._calc(score_weight=3.0, step_penalty=0.0)
        info = {"prev_score": 1.0, "score": 4.0}
        r = calc.compute(prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info)
        r2, _ = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=info,
        )
        self.assertAlmostEqual(r, r2)


class TestSC2AttackMoveBonusAndClickAttackBonus(unittest.TestCase):
    """Tests for the attack-move and click-to-attack reward split."""

    # Screen size used in all helpers below.
    _SS = 64.0

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg_kwargs = {
            "score_weight": 0.0,
            "step_penalty": 0.0,
            "win_bonus": 0.0,
            "loss_penalty": 0.0,
            "economy_weight": 0.0,
        }
        cfg_kwargs.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg_kwargs))

    def _info(
        self,
        fn_idx: int,
        enemy_count: float = 1.0,
        enemy_cx: float = 32.0,
        enemy_cy: float = 32.0,
        target_x_norm: float = 0.5,  # normalised [0,1]
        target_y_norm: float = 0.5,
    ) -> dict:
        return {
            "prev_score": 0.0,
            "score": 0.0,
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
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0, target_x_norm=0.0, target_y_norm=0.0),
        )
        self.assertAlmostEqual(r, 1.0)

    def test_attack_move_bonus_skipped_when_no_enemy_on_screen(self):
        calc = self._make_calc(attack_move_bonus=1.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_count=0.0, target_x_norm=0.0, target_y_norm=0.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_attack_move_bonus_skipped_for_move_screen(self):
        """Plain Move_screen does not trigger attack_move_bonus."""
        calc = self._make_calc(attack_move_bonus=1.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=2, target_x_norm=0.0, target_y_norm=0.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_attack_move_bonus_disabled_by_default(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.attack_move_bonus, 0.0)

    def test_attack_move_bonus_scales_with_n_ticks(self):
        calc = self._make_calc(attack_move_bonus=1.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
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
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0, target_x_norm=0.5, target_y_norm=0.5),
        )
        self.assertAlmostEqual(r, 2.0)

    def test_click_attack_bonus_skipped_when_target_far_from_enemy(self):
        """Target far from enemy centroid → not a click-to-attack → 0."""
        calc = self._make_calc(click_attack_bonus=2.0)
        # enemy at (32,32), target at (0,0) — distance >> click radius
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0, target_x_norm=0.0, target_y_norm=0.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_click_attack_bonus_skipped_when_no_enemy(self):
        calc = self._make_calc(click_attack_bonus=2.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_count=0.0, target_x_norm=0.5, target_y_norm=0.5),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_click_attack_bonus_disabled_by_default(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.click_attack_bonus, 0.0)

    def test_click_attack_bonus_scales_with_n_ticks(self):
        calc = self._make_calc(click_attack_bonus=2.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0, target_x_norm=0.5, target_y_norm=0.5),
            n_ticks=3,
        )
        self.assertAlmostEqual(r, 6.0)

    def test_attack_move_bonus_carries_on_following_no_op_steps(self):
        calc = self._make_calc(attack_move_bonus=1.0, idle_bonus=2.0)
        calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0, target_x_norm=0.0, target_y_norm=0.0),
        )
        r, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=0, enemy_cx=32.0, enemy_cy=32.0),
        )
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(comp["attack_move_bonus"], 1.0)
        self.assertAlmostEqual(comp["idle_bonus"], 0.0)

    def test_click_attack_bonus_carries_on_following_no_op_steps(self):
        calc = self._make_calc(click_attack_bonus=2.0)
        calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0, target_x_norm=0.5, target_y_norm=0.5),
        )
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=0, enemy_cx=32.0, enemy_cy=32.0),
        )
        self.assertAlmostEqual(r, 2.0)

    def test_attack_bonus_carry_stops_on_non_no_op_action(self):
        calc = self._make_calc(attack_move_bonus=1.0)
        calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0, target_x_norm=0.0, target_y_norm=0.0),
        )
        calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=2, target_x_norm=0.2, target_y_norm=0.8),
        )
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
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
        info = self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0, target_x_norm=0.5, target_y_norm=0.5)
        for _ in range(5):
            r = calc.compute(prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info)
            self.assertAlmostEqual(r, 1.0)

    def test_rapid_switch_withholds_bonus(self):
        """Switching to a new target within cooldown window gets 0."""
        calc = self._make_calc(click_attack_bonus=1.0, click_attack_cooldown_steps=5)
        # First click at centre
        calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0, target_x_norm=0.5, target_y_norm=0.5),
        )
        # Immediately switch to a very different target (far enemy)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=4.0, enemy_cy=4.0, target_x_norm=0.0625, target_y_norm=0.0625),  # px≈4
        )
        self.assertAlmostEqual(r, 0.0)

    def test_bonus_fires_after_cooldown_elapsed(self):
        """After cooldown_steps of non-attack-screen actions, bonus fires again."""
        cooldown = 4
        calc = self._make_calc(click_attack_bonus=1.0, click_attack_cooldown_steps=cooldown)
        # First click (centre target)
        calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0, target_x_norm=0.5, target_y_norm=0.5),
        )
        # Advance step count with non-attack actions
        no_attack_info = {
            "prev_score": 0.0,
            "score": 0.0,
            "action_fn_idx": 0,
            "screen_enemy_count": 1.0,
            "screen_size": self._SS,
        }
        for _ in range(cooldown):
            calc.compute(prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=no_attack_info)
        # Now click a different enemy target — cooldown expired
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=4.0, enemy_cy=4.0, target_x_norm=0.0625, target_y_norm=0.0625),
        )
        self.assertAlmostEqual(r, 1.0)

    def test_reset_clears_cooldown_state(self):
        """After reset(), the cooldown state is cleared for a new episode."""
        calc = self._make_calc(click_attack_bonus=1.0, click_attack_cooldown_steps=100)
        # Click at centre to prime cooldown
        calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0, target_x_norm=0.5, target_y_norm=0.5),
        )
        # Reset — starts a fresh episode
        calc.reset()
        # Click a different target immediately after reset — should fire
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=4.0, enemy_cy=4.0, target_x_norm=0.0625, target_y_norm=0.0625),
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
            "prev_score": 0.0,
            "score": 0.0,
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
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info_with_self(
                fn_idx=3, self_count=1.0, self_cx=32.0, self_cy=32.0, target_x_norm=0.5, target_y_norm=0.5
            ),
        )
        self.assertAlmostEqual(r, -5.0)

    def test_attack_friendly_penalty_skipped_when_target_far_from_friendly(self):
        """Attack_screen aimed at empty ground far from friendlies → no penalty."""
        calc = self._make_calc(attack_friendly_penalty=-5.0)
        # friendly at (32,32), target at (0,0) — far away
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info_with_self(
                fn_idx=3, self_count=1.0, self_cx=32.0, self_cy=32.0, target_x_norm=0.0, target_y_norm=0.0
            ),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_attack_friendly_penalty_skipped_when_no_friendly_on_screen(self):
        """No friendly units visible → no penalty even if target is at centroid."""
        calc = self._make_calc(attack_friendly_penalty=-5.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info_with_self(fn_idx=3, self_count=0.0, target_x_norm=0.5, target_y_norm=0.5),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_attack_friendly_penalty_skipped_for_move_screen(self):
        """Move_screen (fn_idx 2) does not trigger the friendly-fire penalty."""
        calc = self._make_calc(
            attack_friendly_penalty=-5.0, move_self_penalty=0.0, move_exploration_bonus=0.0, move_repeat_penalty=0.0
        )
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info_with_self(
                fn_idx=2, self_count=1.0, self_cx=32.0, self_cy=32.0, target_x_norm=0.5, target_y_norm=0.5
            ),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_attack_friendly_penalty_disabled_when_zero(self):
        """Setting attack_friendly_penalty=0.0 disables the check entirely."""
        calc = self._make_calc(attack_friendly_penalty=0.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info_with_self(
                fn_idx=3, self_count=1.0, self_cx=32.0, self_cy=32.0, target_x_norm=0.5, target_y_norm=0.5
            ),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_attack_friendly_penalty_scales_with_n_ticks(self):
        calc = self._make_calc(attack_friendly_penalty=-5.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info_with_self(
                fn_idx=3, self_count=1.0, self_cx=32.0, self_cy=32.0, target_x_norm=0.5, target_y_norm=0.5
            ),
            n_ticks=3,
        )
        self.assertAlmostEqual(r, -15.0)

    def test_attack_friendly_penalty_in_components(self):
        """attack_friendly_penalty appears as a separate component."""
        calc = self._make_calc(attack_friendly_penalty=-5.0)
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info_with_self(
                fn_idx=3, self_count=1.0, self_cx=32.0, self_cy=32.0, target_x_norm=0.5, target_y_norm=0.5
            ),
        )
        self.assertAlmostEqual(comp["attack_friendly_penalty"], -5.0)

    def test_both_bonuses_exclusive(self):
        """Ground target → attack_move_bonus; on-enemy target → click_attack_bonus."""
        calc = self._make_calc(attack_move_bonus=1.0, click_attack_bonus=2.0)
        # Ground target (far from enemy centroid)
        r_move, comp_move = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0, target_x_norm=0.0, target_y_norm=0.0),
        )
        self.assertAlmostEqual(comp_move["attack_move_bonus"], 1.0)
        self.assertAlmostEqual(comp_move["click_attack_bonus"], 0.0)

        # Click on enemy centroid
        calc.reset()
        r_click, comp_click = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, enemy_cx=32.0, enemy_cy=32.0, target_x_norm=0.5, target_y_norm=0.5),
        )
        self.assertAlmostEqual(comp_click["attack_move_bonus"], 0.0)
        self.assertAlmostEqual(comp_click["click_attack_bonus"], 2.0)


class TestSC2EarlyRandomActionBonus(unittest.TestCase):
    """Tests for the early-random-action exploration bonus."""

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg = {
            "score_weight": 0.0,
            "step_penalty": 0.0,
            "win_bonus": 0.0,
            "loss_penalty": 0.0,
            "economy_weight": 0.0,
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
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=2),
        )
        self.assertAlmostEqual(comp["early_random_action"], 3.0)

    def test_skips_repeated_action(self):
        calc = self._make_calc(
            early_random_action_bonus=3.0,
            early_random_action_window_steps=10,
        )
        calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=2),
        )
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=2),
        )
        self.assertAlmostEqual(comp["early_random_action"], 0.0)

    def test_skips_actions_after_window(self):
        calc = self._make_calc(
            early_random_action_bonus=3.0,
            early_random_action_window_steps=1,
        )
        calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=0),
        )
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3),
        )
        self.assertAlmostEqual(comp["early_random_action"], 0.0)


class TestSC2UnitLossPenalty(unittest.TestCase):
    """Tests for the unit_loss_penalty reward term."""

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg = {"score_weight": 0.0, "step_penalty": 0.0, "win_bonus": 0.0, "loss_penalty": 0.0, "economy_weight": 0.0}
        cfg.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg))

    def test_penalty_fires_when_units_die(self):
        calc = self._make_calc(unit_loss_penalty=-5.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "prev_army_count": 4.0, "army_count": 2.0},
        )
        self.assertAlmostEqual(r, -10.0)  # 2 units lost × -5.0

    def test_penalty_zero_when_no_units_lost(self):
        calc = self._make_calc(unit_loss_penalty=-5.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "prev_army_count": 4.0, "army_count": 4.0},
        )
        self.assertAlmostEqual(r, 0.0)

    def test_no_penalty_when_army_grows(self):
        """Producing new units should not yield a penalty."""
        calc = self._make_calc(unit_loss_penalty=-5.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "prev_army_count": 2.0, "army_count": 4.0},
        )
        self.assertAlmostEqual(r, 0.0)

    def test_disabled_by_default(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.unit_loss_penalty, 0.0)

    def test_in_components_dict(self):
        calc = self._make_calc(unit_loss_penalty=-5.0)
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "prev_army_count": 3.0, "army_count": 2.0},
        )
        self.assertAlmostEqual(comp["unit_loss"], -5.0)


class TestSC2DamageTakenPenalty(unittest.TestCase):
    """Tests for the damage_taken_penalty reward term."""

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg = {"score_weight": 0.0, "step_penalty": 0.0, "win_bonus": 0.0, "loss_penalty": 0.0, "economy_weight": 0.0}
        cfg.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg))

    def test_penalty_fires_on_hp_loss(self):
        calc = self._make_calc(damage_taken_penalty=-0.1)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "prev_total_self_hp": 100.0, "total_self_hp": 60.0},
        )
        self.assertAlmostEqual(r, -4.0)  # 40 HP lost × -0.1

    def test_no_penalty_when_hp_unchanged(self):
        calc = self._make_calc(damage_taken_penalty=-0.1)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "prev_total_self_hp": 100.0, "total_self_hp": 100.0},
        )
        self.assertAlmostEqual(r, 0.0)

    def test_no_penalty_when_hp_increases(self):
        """Healing or new units appearing on-screen should not penalise."""
        calc = self._make_calc(damage_taken_penalty=-0.1)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "prev_total_self_hp": 50.0, "total_self_hp": 100.0},
        )
        self.assertAlmostEqual(r, 0.0)

    def test_disabled_by_default(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.damage_taken_penalty, 0.0)

    def test_in_components_dict(self):
        calc = self._make_calc(damage_taken_penalty=-0.5)
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "prev_total_self_hp": 80.0, "total_self_hp": 60.0},
        )
        self.assertAlmostEqual(comp["damage_taken"], -10.0)  # 20 × -0.5

    def test_zero_when_info_keys_absent(self):
        """Missing prev/curr HP keys → no penalty (safe default)."""
        calc = self._make_calc(damage_taken_penalty=-0.5)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0},
        )
        self.assertAlmostEqual(r, 0.0)


class TestSC2PassiveUnderFirePenalty(unittest.TestCase):
    """Tests for passive_under_fire_penalty."""

    _SS = 64.0

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg = {
            "score_weight": 0.0,
            "step_penalty": 0.0,
            "win_bonus": 0.0,
            "loss_penalty": 0.0,
            "economy_weight": 0.0,
            # Silence other shaping terms so only passive_under_fire is active.
            "move_exploration_bonus": 0.0,
            "move_repeat_penalty": 0.0,
            "move_self_penalty": 0.0,
            "attack_friendly_penalty": 0.0,
            "attack_move_bonus": 0.0,
            "click_attack_bonus": 0.0,
        }
        cfg.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg))

    def _info(self, fn_idx: int, dist: float = 10.0, self_attack_range_px: float | None = None) -> dict:
        out = {
            "prev_score": 0.0,
            "score": 0.0,
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
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=0, dist=10.0),  # no_op, enemy close
        )
        self.assertAlmostEqual(r, -2.0)

    def test_fires_on_move_screen_when_enemy_in_range(self):
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=2, dist=10.0),  # Move_screen, enemy close
        )
        self.assertAlmostEqual(r, -2.0)

    def test_skipped_when_attack_screen_issued(self):
        """Attack_screen (fn_idx 3) suppresses the penalty."""
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=3, dist=10.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_skipped_when_enemy_out_of_range(self):
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=0, dist=60.0),  # enemy far away
        )
        self.assertAlmostEqual(r, 0.0)

    def test_skipped_when_no_enemy_on_screen(self):
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        info = self._info(fn_idx=0, dist=10.0)
        info["screen_enemy_count"] = 0.0
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=info,
        )
        self.assertAlmostEqual(r, 0.0)

    def test_skipped_when_no_self_on_screen(self):
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        info = self._info(fn_idx=0, dist=10.0)
        info["screen_self_count"] = 0.0
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=info,
        )
        self.assertAlmostEqual(r, 0.0)

    def test_uses_self_attack_range_px_when_provided(self):
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        # Enemy at dist=25 px; default range (~20) would miss but explicit 30 catches it.
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=0, dist=25.0, self_attack_range_px=30.0),
        )
        self.assertAlmostEqual(r, -2.0)

    def test_skipped_beyond_explicit_range(self):
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        # Enemy at dist=35 px, explicit range=30 → outside range.
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=0, dist=35.0, self_attack_range_px=30.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_scales_with_n_ticks(self):
        calc = self._make_calc(passive_under_fire_penalty=-2.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
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
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=0, dist=10.0),
        )
        self.assertAlmostEqual(comp["passive_under_fire"], -2.0)


class TestSC2SmallSelectionBonus(unittest.TestCase):
    """Tests for the small_selection_bonus reward term."""

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg = {
            "score_weight": 0.0,
            "step_penalty": 0.0,
            "win_bonus": 0.0,
            "loss_penalty": 0.0,
            "economy_weight": 0.0,
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
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(selected_count=1.0, visible_self_unit_count=6.0),
        )
        self.assertAlmostEqual(r, 1.5)

    def test_fires_when_selection_is_under_half_visible_units(self):
        calc = self._make_calc(small_selection_bonus=1.5)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(selected_count=2.0, visible_self_unit_count=6.0),
        )
        self.assertAlmostEqual(r, 1.5)

    def test_skips_at_exactly_half_selected_units(self):
        calc = self._make_calc(small_selection_bonus=1.5)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(selected_count=2.0, visible_self_unit_count=4.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_skips_for_non_unit_targeted_actions(self):
        calc = self._make_calc(small_selection_bonus=1.5)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(fn_idx=0, selected_count=1.0, visible_self_unit_count=4.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_in_components_dict(self):
        calc = self._make_calc(small_selection_bonus=1.5)
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(selected_count=1.0, visible_self_unit_count=4.0),
        )
        self.assertAlmostEqual(comp["small_selection"], 1.5)


class TestSC2RewardComponentsExtended(unittest.TestCase):
    """Verify the three new component keys appear in compute_with_components."""

    def test_new_component_keys_present(self):
        calc = SC2RewardCalculator(
            SC2RewardConfig(
                score_weight=1.0,
                economy_weight=0.001,
                step_penalty=-0.001,
                unit_loss_penalty=-1.0,
                damage_taken_penalty=-0.1,
                passive_under_fire_penalty=-1.0,
            )
        )
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={
                "prev_score": 0.0,
                "score": 0.0,
                "prev_minerals": 0.0,
                "minerals": 0.0,
                "prev_vespene": 0.0,
                "vespene": 0.0,
            },
        )
        for key in ("unit_loss", "damage_taken", "passive_under_fire"):
            self.assertIn(key, comp)

    def test_components_sum_equals_total_with_new_terms(self):
        calc = SC2RewardCalculator(
            SC2RewardConfig(
                score_weight=1.0,
                step_penalty=0.0,
                economy_weight=0.0,
                unit_loss_penalty=-5.0,
                damage_taken_penalty=-0.1,
                passive_under_fire_penalty=-2.0,
                win_bonus=0.0,
                loss_penalty=0.0,
            )
        )
        info = {
            "prev_score": 0.0,
            "score": 10.0,
            "prev_army_count": 4.0,
            "army_count": 3.0,
            "prev_total_self_hp": 100.0,
            "total_self_hp": 70.0,
            "action_fn_idx": 0,
            "screen_self_count": 1.0,
            "screen_enemy_count": 1.0,
            "screen_self_cx": 32.0,
            "screen_self_cy": 32.0,
            "screen_enemy_cx": 42.0,
            "screen_enemy_cy": 32.0,
            "screen_size": 64.0,
        }
        total, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=info,
        )
        self.assertAlmostEqual(total, sum(comp.values()), places=5)


class TestSC2IdleWorkerPenalty(unittest.TestCase):
    """Tests for the idle_worker_penalty reward (issue #358)."""

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg_kwargs = {
            "score_weight": 0.0,
            "step_penalty": 0.0,
            "win_bonus": 0.0,
            "loss_penalty": 0.0,
            "economy_weight": 0.0,
        }
        cfg_kwargs.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg_kwargs))

    def test_penalty_fires_for_each_idle_worker(self):
        calc = self._make_calc(idle_worker_penalty=-1.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "idle_worker_count": 3},
        )
        self.assertAlmostEqual(r, -3.0)

    def test_penalty_zero_when_no_idle_workers(self):
        calc = self._make_calc(idle_worker_penalty=-1.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "idle_worker_count": 0},
        )
        self.assertAlmostEqual(r, 0.0)

    def test_penalty_absent_when_key_missing(self):
        calc = self._make_calc(idle_worker_penalty=-1.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0},
        )
        self.assertAlmostEqual(r, 0.0)

    def test_penalty_disabled_by_default(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.idle_worker_penalty, 0.0)

    def test_penalty_scales_with_n_ticks(self):
        calc = self._make_calc(idle_worker_penalty=-1.0)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "idle_worker_count": 2},
            n_ticks=3,
        )
        self.assertAlmostEqual(r, -6.0)  # -1.0 * 2 workers * 3 ticks

    def test_penalty_appears_in_components(self):
        calc = self._make_calc(idle_worker_penalty=-2.0)
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "idle_worker_count": 4},
        )
        self.assertIn("idle_worker_penalty", comp)
        self.assertAlmostEqual(comp["idle_worker_penalty"], -8.0)


class TestNewActionUnlockBonus(unittest.TestCase):
    """Tests for SC2RewardCalculator.new_action_unlock_bonus (issue #360)."""

    def _base_info(self) -> dict:
        return {
            "prev_score": 0.0,
            "score": 0.0,
            "prev_minerals": 0.0,
            "minerals": 0.0,
            "prev_vespene": 0.0,
            "vespene": 0.0,
            "action_fn_idx": 0,
        }

    def _make_calc(self, bonus: float) -> SC2RewardCalculator:
        return SC2RewardCalculator(
            SC2RewardConfig(
                new_action_unlock_bonus=bonus,
                score_weight=0.0,
                step_penalty=0.0,
                economy_weight=0.0,
                win_bonus=0.0,
                loss_penalty=0.0,
                move_exploration_bonus=0.0,
            )
        )

    def test_default_is_zero(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.new_action_unlock_bonus, 0.0)

    def test_tech_gated_fn_ids_nonempty(self):
        # Verify the class-level precomputed set has at least some entries.
        # fn_idx 8 = Build_Barracks_screen (requires SupplyDepot) must be included.
        self.assertIn(8, SC2RewardCalculator._TECH_GATED_FN_IDS)

    def test_non_gated_fn_ids_excluded(self):
        # no_op (0), select_army (1), Move_screen (2) have no required_buildings.
        for fn_idx in (0, 1, 2):
            self.assertNotIn(fn_idx, SC2RewardCalculator._TECH_GATED_FN_IDS)

    def test_train_marine_is_tech_gated(self):
        # fn_idx 7 = Train_Marine_quick: selection_target={"Barracks"};
        # Barracks has non-empty BUILDING_PREREQS (requires SupplyDepot).
        # Must be included so new_action_unlock_bonus fires when Barracks is built.
        self.assertIn(7, SC2RewardCalculator._TECH_GATED_FN_IDS)

    def test_train_hellion_is_tech_gated(self):
        # fn_idx 35 = Train_Hellion_quick: selection_target={"Factory"};
        # Factory requires Barracks.
        self.assertIn(35, SC2RewardCalculator._TECH_GATED_FN_IDS)

    def test_train_zealot_is_tech_gated(self):
        # fn_idx 64 = Train_Zealot_quick: selection_target={"Gateway","WarpGate"};
        # Gateway requires Pylon.
        self.assertIn(64, SC2RewardCalculator._TECH_GATED_FN_IDS)

    def test_train_scv_not_tech_gated(self):
        # fn_idx 10 = Train_SCV_quick: selection_target={"CommandCenter",...};
        # CommandCenter has no BUILDING_PREREQS — it's the starting structure.
        self.assertNotIn(10, SC2RewardCalculator._TECH_GATED_FN_IDS)

    def test_train_drone_not_tech_gated(self):
        # fn_idx 95 = Train_Drone_quick: selection_target={"Larva"};
        # Larva is not in BUILDING_PREREQS at all.
        self.assertNotIn(95, SC2RewardCalculator._TECH_GATED_FN_IDS)

    def test_unlock_bonus_fires_for_train_marine(self):
        # Verifies end-to-end: unlock bonus fires when Train_Marine_quick (fn_idx 7)
        # appears in available_fn_ids for the first time.
        calc = self._make_calc(bonus=10.0)
        info = {**self._base_info(), "available_fn_ids": {7}}
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp["new_action_unlock"], 10.0)

    def test_unlock_bonus_silent_for_train_scv(self):
        # Train_SCV_quick must never trigger the unlock bonus.
        calc = self._make_calc(bonus=10.0)
        info = {**self._base_info(), "available_fn_ids": {10}}
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp["new_action_unlock"], 0.0)

    def test_train_probe_not_tech_gated(self):
        # fn_idx 63 = Train_Probe_quick: selection_target={"Nexus"};
        # Nexus has no BUILDING_PREREQS (it is the Protoss starting structure).
        self.assertNotIn(63, SC2RewardCalculator._TECH_GATED_FN_IDS)

    def test_unlock_bonus_silent_for_train_probe(self):
        # End-to-end: unlock bonus must not fire when only Train_Probe_quick
        # appears in available_fn_ids (Nexus is always present — no tech gate).
        calc = self._make_calc(bonus=10.0)
        info = {**self._base_info(), "available_fn_ids": {63}}
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp["new_action_unlock"], 0.0)

    def test_bonus_fires_on_first_appearance(self):
        calc = self._make_calc(bonus=5.0)
        info = {**self._base_info(), "available_fn_ids": {8}}  # Build_Barracks_screen
        total, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp["new_action_unlock"], 5.0)
        self.assertAlmostEqual(total, 5.0)

    def test_bonus_does_not_fire_again_same_episode(self):
        calc = self._make_calc(bonus=5.0)
        info = {**self._base_info(), "available_fn_ids": {8}}
        calc.compute_with_components(prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info)
        # Second step — same fn_idx still available.
        _, comp2 = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp2["new_action_unlock"], 0.0)

    def test_bonus_fires_again_after_reset(self):
        calc = self._make_calc(bonus=5.0)
        info = {**self._base_info(), "available_fn_ids": {8}}
        calc.compute_with_components(prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info)
        calc.reset()
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp["new_action_unlock"], 5.0)

    def test_bonus_scales_with_count(self):
        # Two tech-gated fn_ids unlocked simultaneously → 2× bonus.
        calc = self._make_calc(bonus=3.0)
        # fn_idx 8 = Build_Barracks_screen, fn_idx 26 = Build_Factory_screen
        # (both are building-gated per PRECONDITIONS)
        tech_gated = SC2RewardCalculator._TECH_GATED_FN_IDS
        two_gated = set(list(tech_gated)[:2])
        info = {**self._base_info(), "available_fn_ids": two_gated}
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp["new_action_unlock"], 3.0 * 2)

    def test_non_gated_actions_do_not_trigger(self):
        calc = self._make_calc(bonus=5.0)
        # Only non-gated fn_ids in available_fn_ids.
        info = {**self._base_info(), "available_fn_ids": {0, 1, 2, 3}}
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp["new_action_unlock"], 0.0)

    def test_disabled_when_bonus_is_zero(self):
        calc = self._make_calc(bonus=0.0)
        info = {**self._base_info(), "available_fn_ids": {8, 26}}
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp["new_action_unlock"], 0.0)

    def test_missing_available_fn_ids_key_no_crash(self):
        calc = self._make_calc(bonus=5.0)
        info = self._base_info()  # no "available_fn_ids" key
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp["new_action_unlock"], 0.0)

    def test_components_sum_equals_total(self):
        calc = self._make_calc(bonus=5.0)
        info = {**self._base_info(), "available_fn_ids": {8}}
        total, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(total, sum(comp.values()), places=5)


class TestNewActionUsageBonus(unittest.TestCase):
    """Tests for SC2RewardCalculator.new_action_usage_bonus (issue #400)."""

    def _base_info(self) -> dict:
        return {
            "prev_score": 0.0,
            "score": 0.0,
            "prev_minerals": 0.0,
            "minerals": 0.0,
            "prev_vespene": 0.0,
            "vespene": 0.0,
            "action_fn_idx": 0,
        }

    def _make_calc(self, usage_bonus: float, max_uses: int = 50, unlock_bonus: float = 0.0) -> SC2RewardCalculator:
        return SC2RewardCalculator(
            SC2RewardConfig(
                new_action_usage_bonus=usage_bonus,
                new_action_usage_max_uses=max_uses,
                new_action_unlock_bonus=unlock_bonus,
                score_weight=0.0,
                step_penalty=0.0,
                economy_weight=0.0,
                win_bonus=0.0,
                loss_penalty=0.0,
                move_exploration_bonus=0.0,
            )
        )

    def _tech_fn_idx(self) -> int:
        return next(iter(SC2RewardCalculator._TECH_GATED_FN_IDS))

    def test_default_is_zero(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.new_action_usage_bonus, 0.0)

    def test_default_max_uses(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.new_action_usage_max_uses, 50)

    def test_disabled_when_bonus_is_zero(self):
        fn_idx = self._tech_fn_idx()
        calc = self._make_calc(usage_bonus=0.0)
        info = {**self._base_info(), "available_fn_ids": {fn_idx}, "action_fn_idx": fn_idx}
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp["new_action_usage"], 0.0)

    def test_no_bonus_before_unlock(self):
        # Issuing a tech fn_idx before it appears in available_fn_ids yields no bonus.
        fn_idx = self._tech_fn_idx()
        calc = self._make_calc(usage_bonus=1.0)
        info = {**self._base_info(), "action_fn_idx": fn_idx}  # no available_fn_ids
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp["new_action_usage"], 0.0)

    def test_bonus_fires_when_action_used_after_unlock(self):
        fn_idx = self._tech_fn_idx()
        calc = self._make_calc(usage_bonus=2.0)
        # Step 1: unlock (action_fn_idx=0 so no usage this step).
        calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={**self._base_info(), "available_fn_ids": {fn_idx}},
        )
        # Step 2: issue the action.
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={**self._base_info(), "available_fn_ids": {fn_idx}, "action_fn_idx": fn_idx},
        )
        self.assertAlmostEqual(comp["new_action_usage"], 2.0)

    def test_bonus_fires_on_same_step_as_unlock(self):
        # If the action appears in available_fn_ids AND is issued in the same step,
        # the unlock updates _unlocked_tech_fn_ids first, so the usage bonus also fires.
        fn_idx = self._tech_fn_idx()
        calc = self._make_calc(usage_bonus=1.5)
        info = {**self._base_info(), "available_fn_ids": {fn_idx}, "action_fn_idx": fn_idx}
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp["new_action_usage"], 1.5)

    def test_bonus_fires_repeatedly_up_to_max_uses(self):
        fn_idx = self._tech_fn_idx()
        max_uses = 3
        calc = self._make_calc(usage_bonus=1.0, max_uses=max_uses)
        # Unlock first.
        calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={**self._base_info(), "available_fn_ids": {fn_idx}},
        )
        usage_info = {**self._base_info(), "available_fn_ids": {fn_idx}, "action_fn_idx": fn_idx}
        bonuses = []
        for _ in range(max_uses + 2):
            _, comp = calc.compute_with_components(
                prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=usage_info
            )
            bonuses.append(comp["new_action_usage"])
        self.assertEqual(bonuses[:max_uses], [1.0] * max_uses)
        self.assertEqual(bonuses[max_uses:], [0.0, 0.0])

    def test_no_bonus_after_max_uses_reached(self):
        fn_idx = self._tech_fn_idx()
        max_uses = 2
        calc = self._make_calc(usage_bonus=1.0, max_uses=max_uses)
        info = {**self._base_info(), "available_fn_ids": {fn_idx}, "action_fn_idx": fn_idx}
        for _ in range(max_uses):
            calc.compute_with_components(prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info)
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp["new_action_usage"], 0.0)

    def test_counts_reset_after_episode_reset(self):
        fn_idx = self._tech_fn_idx()
        calc = self._make_calc(usage_bonus=1.0, max_uses=1)
        info = {**self._base_info(), "available_fn_ids": {fn_idx}, "action_fn_idx": fn_idx}
        calc.compute_with_components(prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info)
        # Should be exhausted — no bonus.
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp["new_action_usage"], 0.0)
        # After reset, count resets.
        calc.reset()
        _, comp2 = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp2["new_action_usage"], 1.0)

    def test_noop_does_not_trigger_bonus(self):
        fn_idx = self._tech_fn_idx()
        calc = self._make_calc(usage_bonus=1.0)
        # Unlock the action.
        calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={**self._base_info(), "available_fn_ids": {fn_idx}},
        )
        # Issue no_op (fn_idx 0).
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={**self._base_info(), "available_fn_ids": {fn_idx}, "action_fn_idx": 0},
        )
        self.assertAlmostEqual(comp["new_action_usage"], 0.0)

    def test_non_tech_action_does_not_trigger_bonus(self):
        # fn_idx 2 = Move_screen — not tech-gated.
        fn_idx = self._tech_fn_idx()
        calc = self._make_calc(usage_bonus=1.0)
        # "Unlock" a tech action to populate _unlocked_tech_fn_ids.
        calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={**self._base_info(), "available_fn_ids": {fn_idx}},
        )
        # Issue Move_screen (fn_idx 2) — not tech-gated.
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={**self._base_info(), "available_fn_ids": {fn_idx}, "action_fn_idx": 2},
        )
        self.assertAlmostEqual(comp["new_action_usage"], 0.0)

    def test_independent_counts_per_fn_idx(self):
        # Two different tech-gated fn_ids have independent use counts.
        tech_gated = SC2RewardCalculator._TECH_GATED_FN_IDS
        fn_a, fn_b = list(tech_gated)[:2]
        max_uses = 2
        calc = self._make_calc(usage_bonus=1.0, max_uses=max_uses)
        # Unlock both.
        calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={**self._base_info(), "available_fn_ids": {fn_a, fn_b}},
        )
        # Exhaust fn_a.
        for _ in range(max_uses):
            calc.compute_with_components(
                prev_state=None,
                curr_state=None,
                finished=False,
                elapsed_s=1.0,
                info={**self._base_info(), "available_fn_ids": {fn_a, fn_b}, "action_fn_idx": fn_a},
            )
        # fn_a exhausted, fn_b still has budget.
        _, comp_a = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={**self._base_info(), "available_fn_ids": {fn_a, fn_b}, "action_fn_idx": fn_a},
        )
        _, comp_b = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={**self._base_info(), "available_fn_ids": {fn_a, fn_b}, "action_fn_idx": fn_b},
        )
        self.assertAlmostEqual(comp_a["new_action_usage"], 0.0)
        self.assertAlmostEqual(comp_b["new_action_usage"], 1.0)

    def test_works_without_unlock_bonus(self):
        # Usage bonus works independently even when unlock_bonus is disabled.
        fn_idx = self._tech_fn_idx()
        calc = self._make_calc(usage_bonus=3.0, unlock_bonus=0.0)
        info = {**self._base_info(), "available_fn_ids": {fn_idx}, "action_fn_idx": fn_idx}
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp["new_action_usage"], 3.0)
        self.assertAlmostEqual(comp["new_action_unlock"], 0.0)

    def test_both_bonuses_fire_independently(self):
        # When both bonuses are enabled, unlock fires once and usage fires per use.
        fn_idx = self._tech_fn_idx()
        calc = self._make_calc(usage_bonus=1.0, unlock_bonus=5.0)
        info = {**self._base_info(), "available_fn_ids": {fn_idx}, "action_fn_idx": fn_idx}
        # First step: both fire.
        _, comp1 = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp1["new_action_unlock"], 5.0)
        self.assertAlmostEqual(comp1["new_action_usage"], 1.0)
        # Second step: only usage fires (unlock already consumed).
        _, comp2 = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(comp2["new_action_unlock"], 0.0)
        self.assertAlmostEqual(comp2["new_action_usage"], 1.0)

    def test_components_sum_equals_total(self):
        fn_idx = self._tech_fn_idx()
        calc = self._make_calc(usage_bonus=2.0, unlock_bonus=5.0)
        info = {**self._base_info(), "available_fn_ids": {fn_idx}, "action_fn_idx": fn_idx}
        total, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertAlmostEqual(total, sum(comp.values()), places=5)

    def test_component_present_when_disabled(self):
        calc = self._make_calc(usage_bonus=0.0)
        info = self._base_info()
        _, comp = calc.compute_with_components(
            prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=info
        )
        self.assertIn("new_action_usage", comp)
        self.assertAlmostEqual(comp["new_action_usage"], 0.0)


class TestSC2ResourceBankingPenalty(unittest.TestCase):
    """Tests for the resource_banking_penalty reward term (issue #372)."""

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg = {
            "score_weight": 0.0,
            "step_penalty": 0.0,
            "win_bonus": 0.0,
            "loss_penalty": 0.0,
            "economy_weight": 0.0,
        }
        cfg.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg))

    def _info(self, minerals: float = 0.0, vespene: float = 0.0) -> dict:
        return {"prev_score": 0.0, "score": 0.0, "minerals": minerals, "vespene": vespene}

    def test_disabled_by_default(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.resource_banking_penalty, 0.0)

    def test_default_mineral_threshold(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.mineral_banking_threshold, 300.0)

    def test_default_gas_threshold(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.gas_banking_threshold, 200.0)

    def test_no_penalty_below_thresholds(self):
        calc = self._make_calc(resource_banking_penalty=-0.001)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(minerals=299.0, vespene=199.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_no_penalty_at_exact_threshold(self):
        calc = self._make_calc(resource_banking_penalty=-0.001)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(minerals=300.0, vespene=200.0),
        )
        self.assertAlmostEqual(r, 0.0)

    def test_penalty_fires_for_excess_minerals(self):
        calc = self._make_calc(resource_banking_penalty=-0.001)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(minerals=400.0, vespene=0.0),
        )
        # excess minerals = 100; penalty = -0.001 * 100 = -0.1
        self.assertAlmostEqual(r, -0.1)

    def test_penalty_fires_for_excess_gas(self):
        calc = self._make_calc(resource_banking_penalty=-0.001)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(minerals=0.0, vespene=300.0),
        )
        # excess gas = 100; penalty = -0.001 * 100 = -0.1
        self.assertAlmostEqual(r, -0.1)

    def test_penalty_accumulates_both_resources(self):
        calc = self._make_calc(resource_banking_penalty=-0.001)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(minerals=500.0, vespene=400.0),
        )
        # excess minerals = 200, excess gas = 200, total = 400
        # penalty = -0.001 * 400 = -0.4
        self.assertAlmostEqual(r, -0.4)

    def test_penalty_scales_with_n_ticks(self):
        calc = self._make_calc(resource_banking_penalty=-0.001)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(minerals=400.0, vespene=0.0),
            n_ticks=4,
        )
        # excess = 100; penalty = -0.001 * 100 * 4 = -0.4
        self.assertAlmostEqual(r, -0.4)

    def test_penalty_zero_when_keys_absent(self):
        """Missing minerals/vespene keys → treat as 0 (below threshold, no penalty)."""
        calc = self._make_calc(resource_banking_penalty=-0.001)
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0},
        )
        self.assertAlmostEqual(r, 0.0)

    def test_in_components_dict(self):
        calc = self._make_calc(resource_banking_penalty=-0.001)
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(minerals=400.0, vespene=0.0),
        )
        self.assertIn("resource_banking", comp)
        self.assertAlmostEqual(comp["resource_banking"], -0.1)

    def test_component_zero_when_disabled(self):
        calc = self._make_calc(resource_banking_penalty=0.0)
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(minerals=1000.0, vespene=1000.0),
        )
        self.assertAlmostEqual(comp["resource_banking"], 0.0)

    def test_components_sum_equals_total(self):
        calc = self._make_calc(resource_banking_penalty=-0.001)
        total, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(minerals=500.0, vespene=300.0),
        )
        self.assertAlmostEqual(total, sum(comp.values()), places=5)

    def test_custom_thresholds(self):
        calc = self._make_calc(
            resource_banking_penalty=-0.01,
            mineral_banking_threshold=100.0,
            gas_banking_threshold=50.0,
        )
        r = calc.compute(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info=self._info(minerals=200.0, vespene=100.0),
        )
        # excess minerals = 100, excess gas = 50, total = 150
        # penalty = -0.01 * 150 = -1.5
        self.assertAlmostEqual(r, -1.5)


class TestMacroProgressionRewards(unittest.TestCase):
    """Supply / worker / army growth, supply-block, tech-building, expansion, scout."""

    def _quiet(self, **kwargs) -> SC2RewardCalculator:
        base = dict(
            score_weight=0.0,
            step_penalty=0.0,
            economy_weight=0.0,
            win_bonus=0.0,
            loss_penalty=0.0,
            move_exploration_bonus=0.0,
        )
        base.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**base))

    def _step(self, calc, **info):
        base = {"prev_score": 0.0, "score": 0.0, "action_fn_idx": 0}
        base.update(info)
        return calc.compute_with_components(prev_state=None, curr_state=None, finished=False, elapsed_s=1.0, info=base)

    # --- supply-block ---
    def test_supply_block_fires_when_capped(self):
        calc = self._quiet(supply_block_penalty=-0.1)
        _, comp = self._step(calc, food_used=15.0, food_cap=15.0)
        self.assertAlmostEqual(comp["supply_block"], -0.1)

    def test_supply_block_not_at_hard_cap(self):
        calc = self._quiet(supply_block_penalty=-0.1)
        _, comp = self._step(calc, food_used=200.0, food_cap=200.0)
        self.assertAlmostEqual(comp["supply_block"], 0.0)

    def test_supply_block_not_when_room(self):
        calc = self._quiet(supply_block_penalty=-0.1)
        _, comp = self._step(calc, food_used=10.0, food_cap=15.0)
        self.assertAlmostEqual(comp["supply_block"], 0.0)

    # --- supply / worker / army growth (delta-based, first step is baseline) ---
    def test_supply_growth_rewards_cap_increase(self):
        calc = self._quiet(supply_growth_bonus=1.0)
        self._step(calc, food_cap=15.0)  # baseline
        _, comp = self._step(calc, food_cap=23.0)
        self.assertAlmostEqual(comp["supply_growth"], 8.0)

    def test_supply_growth_no_baseline_reward(self):
        calc = self._quiet(supply_growth_bonus=1.0)
        _, comp = self._step(calc, food_cap=15.0)
        self.assertAlmostEqual(comp["supply_growth"], 0.0)

    def test_supply_growth_ignores_decrease(self):
        calc = self._quiet(supply_growth_bonus=1.0)
        self._step(calc, food_cap=23.0)
        _, comp = self._step(calc, food_cap=15.0)
        self.assertAlmostEqual(comp["supply_growth"], 0.0)

    def test_worker_growth_rewards_increase(self):
        calc = self._quiet(worker_growth_bonus=1.0)
        self._step(calc, food_workers=12.0)
        _, comp = self._step(calc, food_workers=14.0)
        self.assertAlmostEqual(comp["worker_growth"], 2.0)

    def test_army_growth_rewards_increase(self):
        calc = self._quiet(army_growth_bonus=2.0)
        self._step(calc, food_army=0.0)
        _, comp = self._step(calc, food_army=3.0)
        self.assertAlmostEqual(comp["army_growth"], 6.0)

    # --- tech-building (one-shot per new structure type) ---
    def test_tech_building_fires_on_new_type(self):
        calc = self._quiet(tech_building_bonus=5.0)
        _, comp = self._step(calc, owned_building_names={"CommandCenter", "Barracks"})
        self.assertAlmostEqual(comp["tech_building"], 10.0)

    def test_tech_building_only_new_types(self):
        calc = self._quiet(tech_building_bonus=5.0)
        self._step(calc, owned_building_names={"CommandCenter"})
        _, comp = self._step(calc, owned_building_names={"CommandCenter", "Barracks"})
        self.assertAlmostEqual(comp["tech_building"], 5.0)

    # --- expansion (running-max town-hall count) ---
    def test_expansion_fires_on_new_max(self):
        calc = self._quiet(expansion_bonus=15.0)
        self._step(calc, townhall_count=1)  # starting base — baseline
        _, comp = self._step(calc, townhall_count=2)
        self.assertAlmostEqual(comp["expansion"], 15.0)

    def test_expansion_no_reward_for_starting_base(self):
        calc = self._quiet(expansion_bonus=15.0)
        _, comp = self._step(calc, townhall_count=1)
        self.assertAlmostEqual(comp["expansion"], 0.0)

    def test_expansion_not_rewarded_twice_for_same_count(self):
        calc = self._quiet(expansion_bonus=15.0)
        self._step(calc, townhall_count=1)
        self._step(calc, townhall_count=2)
        _, comp = self._step(calc, townhall_count=2)
        self.assertAlmostEqual(comp["expansion"], 0.0)

    def test_expansion_ignores_temporary_drop(self):
        calc = self._quiet(expansion_bonus=15.0)
        self._step(calc, townhall_count=2)  # baseline at 2
        _, comp = self._step(calc, townhall_count=1)  # lost vision / base
        self.assertAlmostEqual(comp["expansion"], 0.0)

    # --- scout_explore (explored-fraction delta) ---
    def test_scout_rewards_exploration(self):
        calc = self._quiet(scout_bonus=20.0)
        self._step(calc, minimap_explored_frac=0.10)
        _, comp = self._step(calc, minimap_explored_frac=0.15)
        self.assertAlmostEqual(comp["scout_explore"], 1.0)  # 20 * 0.05

    def test_scout_ignores_decrease(self):
        calc = self._quiet(scout_bonus=20.0)
        self._step(calc, minimap_explored_frac=0.20)
        _, comp = self._step(calc, minimap_explored_frac=0.10)
        self.assertAlmostEqual(comp["scout_explore"], 0.0)

    # --- safety: missing keys, components sum, all present when disabled ---
    def test_new_components_present_when_disabled(self):
        calc = self._quiet()
        _, comp = self._step(calc)
        for key in (
            "new_action_usage",
            "supply_block",
            "supply_growth",
            "worker_growth",
            "army_growth",
            "tech_building",
            "expansion",
            "scout_explore",
        ):
            self.assertIn(key, comp)
            self.assertEqual(comp[key], 0.0)

    def test_components_sum_equals_total(self):
        calc = self._quiet(
            supply_growth_bonus=1.0,
            worker_growth_bonus=1.0,
            army_growth_bonus=1.0,
            tech_building_bonus=5.0,
            expansion_bonus=15.0,
            scout_bonus=20.0,
            supply_block_penalty=-0.1,
        )
        self._step(calc, food_cap=15.0, food_workers=12.0, food_army=0.0, townhall_count=1, minimap_explored_frac=0.1)
        total, comp = self._step(
            calc,
            food_cap=23.0,
            food_workers=14.0,
            food_army=3.0,
            food_used=23.0,
            townhall_count=2,
            minimap_explored_frac=0.15,
            owned_building_names={"CommandCenter", "Barracks"},
        )
        self.assertAlmostEqual(total, sum(comp.values()))


class TestSC2BuildBonus(unittest.TestCase):
    """Tests for the build_bonus reward term (issue #416)."""

    # fn_idx 8 = Build_Barracks_screen (Terran build)
    # fn_idx 7 = Train_Marine_quick (Terran train)
    # fn_idx 2 = Move_screen (move category — should NOT trigger)
    # fn_idx 0 = no_op (move category — should NOT trigger)
    _BUILD_FN_IDX = 8
    _TRAIN_FN_IDX = 7
    _MOVE_FN_IDX = 2
    _NOOP_FN_IDX = 0

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg = {"score_weight": 0.0, "step_penalty": 0.0, "win_bonus": 0.0, "loss_penalty": 0.0, "economy_weight": 0.0}
        cfg.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg))

    def _step(self, calc: SC2RewardCalculator, fn_idx: int = 0) -> tuple[float, dict]:
        return calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "action_fn_idx": fn_idx},
        )

    def test_disabled_by_default(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.build_bonus, 0.0)
        self.assertEqual(cfg.train_bonus, 0.0)

    def test_build_bonus_fires_on_build_action(self):
        calc = self._make_calc(build_bonus=1.0)
        _, comp = self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self.assertAlmostEqual(comp["build_bonus"], 1.0)

    def test_build_bonus_does_not_fire_on_train_action(self):
        calc = self._make_calc(build_bonus=1.0)
        _, comp = self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        self.assertAlmostEqual(comp["build_bonus"], 0.0)

    def test_train_bonus_fires_on_train_action(self):
        calc = self._make_calc(train_bonus=1.0)
        _, comp = self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        self.assertAlmostEqual(comp["train_bonus"], 1.0)

    def test_train_bonus_does_not_fire_on_build_action(self):
        calc = self._make_calc(train_bonus=1.0)
        _, comp = self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self.assertAlmostEqual(comp["train_bonus"], 0.0)

    def test_no_bonus_on_move_action(self):
        calc = self._make_calc(build_bonus=1.0, train_bonus=1.0)
        _, comp = self._step(calc, fn_idx=self._MOVE_FN_IDX)
        self.assertAlmostEqual(comp["build_bonus"], 0.0)
        self.assertAlmostEqual(comp["train_bonus"], 0.0)

    def test_no_bonus_on_noop(self):
        calc = self._make_calc(build_bonus=1.0, train_bonus=1.0)
        _, comp = self._step(calc, fn_idx=self._NOOP_FN_IDX)
        self.assertAlmostEqual(comp["build_bonus"], 0.0)
        self.assertAlmostEqual(comp["train_bonus"], 0.0)

    def test_build_bonus_zero_when_disabled(self):
        calc = self._make_calc(build_bonus=0.0)
        _, comp = self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self.assertAlmostEqual(comp["build_bonus"], 0.0)

    def test_train_bonus_scales_with_value(self):
        calc = self._make_calc(train_bonus=2.5)
        _, comp = self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        self.assertAlmostEqual(comp["train_bonus"], 2.5)

    def test_train_bonus_is_four_times_build_bonus(self):
        """Canonical 4× relationship: train_bonus == 4 × build_bonus."""
        calc = self._make_calc(build_bonus=0.5, train_bonus=2.0)
        _, comp_build = self._step(calc, fn_idx=self._BUILD_FN_IDX)
        calc.reset()
        _, comp_train = self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        self.assertAlmostEqual(comp_train["train_bonus"], 4 * comp_build["build_bonus"])

    def test_n_ticks_scaling_build(self):
        calc = self._make_calc(build_bonus=1.0)
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "action_fn_idx": self._BUILD_FN_IDX},
            n_ticks=4,
        )
        self.assertAlmostEqual(comp["build_bonus"], 4.0)

    def test_n_ticks_scaling_train(self):
        calc = self._make_calc(train_bonus=1.0)
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "action_fn_idx": self._TRAIN_FN_IDX},
            n_ticks=4,
        )
        self.assertAlmostEqual(comp["train_bonus"], 4.0)

    def test_present_in_components_when_disabled(self):
        calc = self._make_calc()
        _, comp = self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self.assertIn("build_bonus", comp)
        self.assertIn("train_bonus", comp)
        self.assertAlmostEqual(comp["build_bonus"], 0.0)
        self.assertAlmostEqual(comp["train_bonus"], 0.0)

    def test_components_sum_equals_total(self):
        calc = self._make_calc(build_bonus=0.5, train_bonus=2.0)
        total, comp = self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self.assertAlmostEqual(total, sum(comp.values()))


class TestSC2BuildRepeatPenalty(unittest.TestCase):
    """Tests for the build_repeat_penalty reward term."""

    # fn_idx 8 = Build_Barracks_screen (build category)
    # fn_idx 9 = Build_SupplyDepot_screen (build category)
    # fn_idx 7 = Train_Marine_quick (train category — NOT build, resets counter)
    # fn_idx 2 = Move_screen (move category — resets counter)
    _BUILD_FN_IDX = 8
    _BUILD_FN_IDX_2 = 9
    _TRAIN_FN_IDX = 7
    _MOVE_FN_IDX = 2

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg = {"score_weight": 0.0, "step_penalty": 0.0, "win_bonus": 0.0, "loss_penalty": 0.0, "economy_weight": 0.0}
        cfg.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg))

    def _step(self, calc: SC2RewardCalculator, fn_idx: int = 0) -> tuple[float, dict]:
        return calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "action_fn_idx": fn_idx},
        )

    def test_disabled_by_default(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.build_repeat_penalty, 0.0)

    def test_first_build_no_penalty(self):
        calc = self._make_calc(build_repeat_penalty=-1.0)
        _, comp = self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self.assertAlmostEqual(comp["build_repeat_penalty"], 0.0)

    def test_second_same_build_fires_penalty(self):
        calc = self._make_calc(build_repeat_penalty=-1.0)
        self._step(calc, fn_idx=self._BUILD_FN_IDX)
        _, comp = self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self.assertAlmostEqual(comp["build_repeat_penalty"], -1.0)

    def test_third_consecutive_same_build_fires_again(self):
        calc = self._make_calc(build_repeat_penalty=-1.0)
        self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self._step(calc, fn_idx=self._BUILD_FN_IDX)
        _, comp = self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self.assertAlmostEqual(comp["build_repeat_penalty"], -1.0)

    def test_different_build_fn_idx_does_not_fire(self):
        calc = self._make_calc(build_repeat_penalty=-1.0)
        self._step(calc, fn_idx=self._BUILD_FN_IDX)
        _, comp = self._step(calc, fn_idx=self._BUILD_FN_IDX_2)
        self.assertAlmostEqual(comp["build_repeat_penalty"], 0.0)

    def test_intervening_move_resets_counter(self):
        calc = self._make_calc(build_repeat_penalty=-1.0)
        self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self._step(calc, fn_idx=self._MOVE_FN_IDX)
        _, comp = self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self.assertAlmostEqual(comp["build_repeat_penalty"], 0.0)

    def test_intervening_train_resets_counter(self):
        calc = self._make_calc(build_repeat_penalty=-1.0)
        self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        _, comp = self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self.assertAlmostEqual(comp["build_repeat_penalty"], 0.0)

    def test_disabled_zero_weight_no_penalty(self):
        calc = self._make_calc(build_repeat_penalty=0.0)
        self._step(calc, fn_idx=self._BUILD_FN_IDX)
        _, comp = self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self.assertAlmostEqual(comp["build_repeat_penalty"], 0.0)

    def test_n_ticks_scaling(self):
        calc = self._make_calc(build_repeat_penalty=-1.0)
        self._step(calc, fn_idx=self._BUILD_FN_IDX)
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "action_fn_idx": self._BUILD_FN_IDX},
            n_ticks=4,
        )
        self.assertAlmostEqual(comp["build_repeat_penalty"], -4.0)

    def test_present_in_components_when_disabled(self):
        calc = self._make_calc()
        _, comp = self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self.assertIn("build_repeat_penalty", comp)

    def test_net_zero_with_matching_bonus(self):
        """Repeated build earns build_bonus + build_repeat_penalty = 0."""
        calc = self._make_calc(build_bonus=0.5, build_repeat_penalty=-0.5)
        self._step(calc, fn_idx=self._BUILD_FN_IDX)
        _, comp = self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self.assertAlmostEqual(comp["build_bonus"] + comp["build_repeat_penalty"], 0.0)

    def test_components_sum_equals_total(self):
        calc = self._make_calc(build_repeat_penalty=-0.5)
        self._step(calc, fn_idx=self._BUILD_FN_IDX)
        total, comp = self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self.assertAlmostEqual(total, sum(comp.values()))

    def test_reset_clears_last_build_fn_idx(self):
        calc = self._make_calc(build_repeat_penalty=-1.0)
        self._step(calc, fn_idx=self._BUILD_FN_IDX)
        calc.reset()
        _, comp = self._step(calc, fn_idx=self._BUILD_FN_IDX)
        self.assertAlmostEqual(comp["build_repeat_penalty"], 0.0)


class TestSC2TrainRepeatPenalty(unittest.TestCase):
    """Tests for the train_repeat_penalty reward term."""

    # fn_idx 7 = Train_Marine_quick (train category)
    # fn_idx 10 = Train_SCV_quick (train category)
    # fn_idx 8 = Build_Barracks_screen (build category — resets counter)
    # fn_idx 2 = Move_screen (move category — resets counter)
    _TRAIN_FN_IDX = 7
    _TRAIN_FN_IDX_2 = 10
    _BUILD_FN_IDX = 8
    _MOVE_FN_IDX = 2

    def _make_calc(self, **kwargs) -> SC2RewardCalculator:
        cfg = {"score_weight": 0.0, "step_penalty": 0.0, "win_bonus": 0.0, "loss_penalty": 0.0, "economy_weight": 0.0}
        cfg.update(kwargs)
        return SC2RewardCalculator(SC2RewardConfig(**cfg))

    def _step(self, calc: SC2RewardCalculator, fn_idx: int = 0) -> tuple[float, dict]:
        return calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "action_fn_idx": fn_idx},
        )

    def test_disabled_by_default(self):
        cfg = SC2RewardConfig()
        self.assertEqual(cfg.train_repeat_penalty, 0.0)

    def test_first_train_no_penalty(self):
        calc = self._make_calc(train_repeat_penalty=-1.0)
        _, comp = self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        self.assertAlmostEqual(comp["train_repeat_penalty"], 0.0)

    def test_second_same_train_fires_penalty(self):
        calc = self._make_calc(train_repeat_penalty=-1.0)
        self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        _, comp = self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        self.assertAlmostEqual(comp["train_repeat_penalty"], -1.0)

    def test_third_consecutive_same_train_fires_again(self):
        calc = self._make_calc(train_repeat_penalty=-1.0)
        self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        _, comp = self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        self.assertAlmostEqual(comp["train_repeat_penalty"], -1.0)

    def test_different_train_fn_idx_does_not_fire(self):
        calc = self._make_calc(train_repeat_penalty=-1.0)
        self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        _, comp = self._step(calc, fn_idx=self._TRAIN_FN_IDX_2)
        self.assertAlmostEqual(comp["train_repeat_penalty"], 0.0)

    def test_intervening_move_resets_counter(self):
        calc = self._make_calc(train_repeat_penalty=-1.0)
        self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        self._step(calc, fn_idx=self._MOVE_FN_IDX)
        _, comp = self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        self.assertAlmostEqual(comp["train_repeat_penalty"], 0.0)

    def test_intervening_build_resets_counter(self):
        calc = self._make_calc(train_repeat_penalty=-1.0)
        self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        self._step(calc, fn_idx=self._BUILD_FN_IDX)
        _, comp = self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        self.assertAlmostEqual(comp["train_repeat_penalty"], 0.0)

    def test_disabled_zero_weight_no_penalty(self):
        calc = self._make_calc(train_repeat_penalty=0.0)
        self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        _, comp = self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        self.assertAlmostEqual(comp["train_repeat_penalty"], 0.0)

    def test_n_ticks_scaling(self):
        calc = self._make_calc(train_repeat_penalty=-1.0)
        self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        _, comp = calc.compute_with_components(
            prev_state=None,
            curr_state=None,
            finished=False,
            elapsed_s=1.0,
            info={"prev_score": 0.0, "score": 0.0, "action_fn_idx": self._TRAIN_FN_IDX},
            n_ticks=4,
        )
        self.assertAlmostEqual(comp["train_repeat_penalty"], -4.0)

    def test_present_in_components_when_disabled(self):
        calc = self._make_calc()
        _, comp = self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        self.assertIn("train_repeat_penalty", comp)

    def test_net_zero_with_matching_bonus(self):
        """Repeated train earns train_bonus + train_repeat_penalty = 0."""
        calc = self._make_calc(train_bonus=2.0, train_repeat_penalty=-2.0)
        self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        _, comp = self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        self.assertAlmostEqual(comp["train_bonus"] + comp["train_repeat_penalty"], 0.0)

    def test_components_sum_equals_total(self):
        calc = self._make_calc(train_repeat_penalty=-2.0)
        self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        total, comp = self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        self.assertAlmostEqual(total, sum(comp.values()))

    def test_reset_clears_last_train_fn_idx(self):
        calc = self._make_calc(train_repeat_penalty=-1.0)
        self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        calc.reset()
        _, comp = self._step(calc, fn_idx=self._TRAIN_FN_IDX)
        self.assertAlmostEqual(comp["train_repeat_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
