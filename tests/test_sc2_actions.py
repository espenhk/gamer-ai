"""Tests for the SC2 action definitions."""
import unittest

import numpy as np

from games.sc2.actions import (
    DISCRETE_ACTIONS,
    FUNCTION_IDS,
    PROBE_ACTIONS,
    SCREEN_GRID_RESOLUTION,
    WARMUP_ACTION,
)


_EXPECTED_ROWS = 2 + SCREEN_GRID_RESOLUTION ** 2  # no_op + select_army + N×N grid


class TestSC2Actions(unittest.TestCase):

    def test_discrete_actions_shape(self):
        # 2 special rows (no_op, select_army) + N×N Move_screen grid.
        self.assertEqual(DISCRETE_ACTIONS.shape, (_EXPECTED_ROWS, 4))

    def test_discrete_actions_dtype(self):
        self.assertEqual(DISCRETE_ACTIONS.dtype, np.float32)

    def test_discrete_actions_x_y_in_unit_square(self):
        xs = DISCRETE_ACTIONS[:, 1]
        ys = DISCRETE_ACTIONS[:, 2]
        self.assertTrue(np.all((xs >= 0.0) & (xs <= 1.0)))
        self.assertTrue(np.all((ys >= 0.0) & (ys <= 1.0)))

    def test_row_zero_is_no_op(self):
        """Issue #127: row 0 must be no_op so tabular policies can idle."""
        self.assertEqual(int(DISCRETE_ACTIONS[0, 0]), 0)

    def test_row_one_is_select_army(self):
        """Row 1 is the select_army precondition action."""
        self.assertEqual(int(DISCRETE_ACTIONS[1, 0]), 1)

    def test_grid_rows_are_move_screen(self):
        """Rows 2..N-1 must all be Move_screen (fn_idx == 2)."""
        for i in range(2, _EXPECTED_ROWS):
            self.assertEqual(int(DISCRETE_ACTIONS[i, 0]), 2,
                             f"row {i} expected Move_screen")

    def test_move_screen_cells_span_unit_square(self):
        """The N×N grid must cover the screen — issue #122 fix."""
        move_xs = DISCRETE_ACTIONS[2:, 1]
        move_ys = DISCRETE_ACTIONS[2:, 2]
        # Cell centres at resolution 8 land at 0.0625 and 0.9375 (extremes).
        self.assertLessEqual(float(move_xs.min()), 0.1)
        self.assertGreaterEqual(float(move_xs.max()), 0.9)
        self.assertLessEqual(float(move_ys.min()), 0.1)
        self.assertGreaterEqual(float(move_ys.max()), 0.9)

    def test_move_screen_cells_are_unique(self):
        """Each (x, y) cell appears exactly once in the Move_screen rows."""
        coords = {(float(r[1]), float(r[2])) for r in DISCRETE_ACTIONS[2:]}
        self.assertEqual(len(coords), SCREEN_GRID_RESOLUTION ** 2)

    def test_probe_actions_count(self):
        self.assertEqual(len(PROBE_ACTIONS), 5)

    def test_probe_actions_shape(self):
        for action, name in PROBE_ACTIONS:
            self.assertEqual(action.shape, (4,))
            self.assertIsInstance(name, str)

    def test_probe_actions_include_no_op(self):
        """Probe coverage of no_op (issue #127)."""
        names = [name for _, name in PROBE_ACTIONS]
        self.assertIn("no_op", names)

    def test_warmup_action_shape(self):
        self.assertEqual(WARMUP_ACTION.shape, (4,))

    def test_warmup_action_is_select_army(self):
        self.assertEqual(int(WARMUP_ACTION[0]), 1)

    def test_function_ids_table_is_complete(self):
        # Indices used by DISCRETE_ACTIONS / PROBE_ACTIONS / WARMUP_ACTION must
        # exist in the FUNCTION_IDS lookup so the client can resolve them.
        used = set()
        for row in DISCRETE_ACTIONS:
            used.add(int(row[0]))
        for action, _ in PROBE_ACTIONS:
            used.add(int(action[0]))
        used.add(int(WARMUP_ACTION[0]))
        for fn_idx in used:
            self.assertIn(fn_idx, FUNCTION_IDS, f"missing fn_idx={fn_idx}")


if __name__ == "__main__":
    unittest.main()
