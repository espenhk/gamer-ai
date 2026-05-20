"""Tests for the SC2 action definitions."""
import unittest

import numpy as np

from games.sc2.actions import (
    DISCRETE_ACTIONS,
    FUNCTION_IDS,
    PROBE_ACTIONS,
    RACE_FUNCTION_IDS,
    SCREEN_GRID_RESOLUTION,
    SPATIAL_FN_IDS,
    WARMUP_ACTION,
    fn_ids_for_race,
)


_N = SCREEN_GRID_RESOLUTION
_N2 = _N ** 2
# Spatial fn_ids get N² rows; non-spatial get 1 row each.
_N_SPATIAL = len(SPATIAL_FN_IDS)
_N_NON_SPATIAL = len(FUNCTION_IDS) - _N_SPATIAL
_EXPECTED_ROWS = _N_SPATIAL * _N2 + _N_NON_SPATIAL


class TestSC2Actions(unittest.TestCase):

    def test_discrete_actions_shape(self):
        self.assertEqual(DISCRETE_ACTIONS.shape, (_EXPECTED_ROWS, 4))

    def test_discrete_actions_dtype(self):
        self.assertEqual(DISCRETE_ACTIONS.dtype, np.float32)

    def test_discrete_actions_x_y_in_unit_square(self):
        xs = DISCRETE_ACTIONS[:, 1]
        ys = DISCRETE_ACTIONS[:, 2]
        self.assertTrue(np.all((xs >= 0.0) & (xs <= 1.0)))
        self.assertTrue(np.all((ys >= 0.0) & (ys <= 1.0)))

    def test_row_zero_is_no_op(self):
        """Issue #127: row 0 must be no_op (fn_idx 0 is first in sorted order
        and non-spatial → single row at row 0)."""
        self.assertEqual(int(DISCRETE_ACTIONS[0, 0]), 0)

    def test_row_one_is_select_army(self):
        """fn_idx 1 (select_army) is non-spatial → single row at row 1."""
        self.assertEqual(int(DISCRETE_ACTIONS[1, 0]), 1)

    def test_fn_idx_row_layout(self):
        """Each fn_id has the correct number of DISCRETE_ACTIONS rows.

        Spatial fn_ids (names ending in _screen or _minimap) get N² rows;
        all others get 1 row.  fn_ids appear in ascending index order.
        """
        row = 0
        for fn_idx in sorted(FUNCTION_IDS.keys()):
            name = FUNCTION_IDS[fn_idx]
            if fn_idx in SPATIAL_FN_IDS:
                expected_rows = _N2
            else:
                expected_rows = 1
            for offset in range(expected_rows):
                self.assertEqual(
                    int(DISCRETE_ACTIONS[row + offset, 0]), fn_idx,
                    f"row {row + offset}: expected fn_idx={fn_idx} ({name}), "
                    f"offset {offset}/{expected_rows}",
                )
            row += expected_rows
        self.assertEqual(row, len(DISCRETE_ACTIONS),
                         "All rows accounted for")

    def test_spatial_actions_span_unit_square(self):
        """Spatial grid rows must cover the screen — derived from issue #122."""
        for fn_idx in sorted(SPATIAL_FN_IDS):
            # Find the first row for this fn_id.
            offset = sum(
                _N2 if i in SPATIAL_FN_IDS else 1
                for i in sorted(FUNCTION_IDS.keys()) if i < fn_idx
            )
            xs = DISCRETE_ACTIONS[offset:offset + _N2, 1]
            ys = DISCRETE_ACTIONS[offset:offset + _N2, 2]
            name = FUNCTION_IDS[fn_idx]
            self.assertLessEqual(float(xs.min()), 0.1,
                                 f"{name}: x grid doesn't reach near 0")
            self.assertGreaterEqual(float(xs.max()), 0.9,
                                    f"{name}: x grid doesn't reach near 1")
            self.assertLessEqual(float(ys.min()), 0.1,
                                 f"{name}: y grid doesn't reach near 0")
            self.assertGreaterEqual(float(ys.max()), 0.9,
                                    f"{name}: y grid doesn't reach near 1")

    def test_move_screen_cells_are_unique(self):
        """Each (x, y) cell appears exactly once in the Move_screen rows."""
        # Move_screen is fn_idx 2; fn_idx 0 and 1 are non-spatial (1 row each).
        coords = {(float(r[1]), float(r[2]))
                  for r in DISCRETE_ACTIONS[2:2 + _N2]}
        self.assertEqual(len(coords), _N2)

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
        """fn_idx values used in DISCRETE_ACTIONS / PROBE_ACTIONS / WARMUP_ACTION
        must exist in FUNCTION_IDS so the client can resolve them."""
        used = set()
        for row in DISCRETE_ACTIONS:
            used.add(int(row[0]))
        for action, _ in PROBE_ACTIONS:
            used.add(int(action[0]))
        used.add(int(WARMUP_ACTION[0]))
        for fn_idx in used:
            self.assertIn(fn_idx, FUNCTION_IDS, f"missing fn_idx={fn_idx}")

    def test_spatial_fn_ids_are_screen_or_minimap(self):
        """SPATIAL_FN_IDS contains exactly the fn_ids whose names end in
        _screen or _minimap."""
        expected = frozenset(
            fn_idx for fn_idx, name in FUNCTION_IDS.items()
            if name.endswith("_screen") or name.endswith("_minimap")
        )
        self.assertEqual(SPATIAL_FN_IDS, expected)


# ---------------------------------------------------------------------------
# Race gating tests
# ---------------------------------------------------------------------------

class TestRaceGating(unittest.TestCase):

    def test_race_keys_exist(self):
        for race in ("terran", "protoss", "zerg", "random"):
            self.assertIn(race, RACE_FUNCTION_IDS)

    def test_race_fn_ids_are_subsets_of_function_ids(self):
        all_ids = frozenset(FUNCTION_IDS.keys())
        for race, ids in RACE_FUNCTION_IDS.items():
            self.assertTrue(ids <= all_ids,
                            f"{race} has fn_ids outside FUNCTION_IDS: "
                            f"{ids - all_ids}")

    def test_random_race_includes_all(self):
        self.assertEqual(fn_ids_for_race("random"), frozenset(FUNCTION_IDS.keys()))

    def test_race_sets_are_disjoint_from_each_other_for_race_specific(self):
        """Race-specific (non-universal) fn_ids must not overlap between races."""
        from games.sc2.actions import (
            _TERRAN_FN_IDS, _PROTOSS_FN_IDS, _ZERG_FN_IDS,
        )
        self.assertFalse(_TERRAN_FN_IDS & _PROTOSS_FN_IDS,
                         "Terran and Protoss-specific fn_ids overlap")
        self.assertFalse(_TERRAN_FN_IDS & _ZERG_FN_IDS,
                         "Terran and Zerg-specific fn_ids overlap")
        self.assertFalse(_PROTOSS_FN_IDS & _ZERG_FN_IDS,
                         "Protoss and Zerg-specific fn_ids overlap")

    def test_fn_ids_for_race_unknown_falls_back_to_all(self):
        self.assertEqual(fn_ids_for_race("unknown_race"),
                         frozenset(FUNCTION_IDS.keys()))

    def test_terran_has_barracks_not_nexus(self):
        terran_ids = fn_ids_for_race("terran")
        # Build_Barracks_screen is fn_idx 8 (Terran)
        self.assertIn(8, terran_ids)
        # Build_Nexus_screen is fn_idx 50 (Protoss)
        self.assertNotIn(50, terran_ids)

    def test_protoss_has_nexus_not_barracks(self):
        protoss_ids = fn_ids_for_race("protoss")
        self.assertIn(50, protoss_ids)   # Build_Nexus_screen
        self.assertNotIn(8, protoss_ids)  # Build_Barracks_screen

    def test_zerg_has_hatchery_not_barracks(self):
        zerg_ids = fn_ids_for_race("zerg")
        self.assertIn(82, zerg_ids)   # Build_Hatchery_screen
        self.assertNotIn(8, zerg_ids)  # Build_Barracks_screen

    def test_all_races_include_move_screen(self):
        for race in ("terran", "protoss", "zerg"):
            self.assertIn(2, fn_ids_for_race(race),
                          f"{race} missing Move_screen")

    def test_all_races_include_no_op(self):
        for race in ("terran", "protoss", "zerg"):
            self.assertIn(0, fn_ids_for_race(race),
                          f"{race} missing no_op")


if __name__ == "__main__":
    unittest.main()
