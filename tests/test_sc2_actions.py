"""Tests for the SC2 action definitions."""
import unittest

import numpy as np

from games.sc2.actions import (
    DISCRETE_ACTIONS,
    FUNCTION_IDS,
    PROBE_ACTIONS,
    WARMUP_ACTION,
)


class TestSC2Actions(unittest.TestCase):

    def test_discrete_actions_shape(self):
        # 9 cells × 4-dim action vector [fn_idx, x, y, queue].
        self.assertEqual(DISCRETE_ACTIONS.shape, (9, 4))

    def test_discrete_actions_dtype(self):
        self.assertEqual(DISCRETE_ACTIONS.dtype, np.float32)

    def test_discrete_actions_x_y_in_unit_square(self):
        xs = DISCRETE_ACTIONS[:, 1]
        ys = DISCRETE_ACTIONS[:, 2]
        self.assertTrue(np.all((xs >= 0.0) & (xs <= 1.0)))
        self.assertTrue(np.all((ys >= 0.0) & (ys <= 1.0)))

    def test_centre_cell_is_select_army(self):
        """Cell index 4 is the 3×3 grid centre and selects the army."""
        self.assertEqual(int(DISCRETE_ACTIONS[4, 0]), 1)  # select_army

    def test_other_cells_are_move_screen(self):
        for i in range(9):
            if i == 4:
                continue
            self.assertEqual(int(DISCRETE_ACTIONS[i, 0]), 2)  # Move_screen

    def test_probe_actions_count(self):
        self.assertEqual(len(PROBE_ACTIONS), 5)

    def test_probe_actions_shape(self):
        for action, name in PROBE_ACTIONS:
            self.assertEqual(action.shape, (4,))
            self.assertIsInstance(name, str)

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
