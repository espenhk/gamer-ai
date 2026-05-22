"""Tests for the SC2 action definitions."""
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from games.sc2.actions import (
    BUILD_PREREQUISITES,
    DISCRETE_ACTIONS,
    FUNCTION_IDS,
    PREREQ_STRUCTURE_NAMES,
    PROBE_ACTIONS,
    RACE_FUNCTION_IDS,
    SCREEN_GRID_RESOLUTION,
    SPATIAL_FN_IDS,
    WARMUP_ACTION,
    action_to_function_call,
    fn_ids_blocked_by_prerequisites,
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
        _screen or _minimap, plus select_point and select_rect."""
        expected = frozenset(
            fn_idx for fn_idx, name in FUNCTION_IDS.items()
            if name.endswith("_screen") or name.endswith("_minimap")
            or name in ("select_point", "select_rect")
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


class _FakeFunctionCall:
    def __init__(self, function: int, arguments: list[list[int]]) -> None:
        self.function = function
        self.arguments = arguments


def _fake_pysc2_modules() -> dict[str, types.ModuleType]:
    pysc2_mod = types.ModuleType("pysc2")
    lib_mod = types.ModuleType("pysc2.lib")
    actions_mod = types.ModuleType("pysc2.lib.actions")
    actions_mod.FunctionCall = _FakeFunctionCall
    functions = types.SimpleNamespace()
    for fn_idx, name in FUNCTION_IDS.items():
        setattr(functions, name, types.SimpleNamespace(id=1000 + fn_idx))
    actions_mod.FUNCTIONS = functions
    return {
        "pysc2": pysc2_mod,
        "pysc2.lib": lib_mod,
        "pysc2.lib.actions": actions_mod,
    }


class TestActionToFunctionCall(unittest.TestCase):
    def test_quick_action_uses_queue_only(self):
        with patch.dict(sys.modules, _fake_pysc2_modules()):
            action = np.array([7, 0.2, 0.4, 1.0], dtype=np.float32)  # Train_Marine_quick
            call = action_to_function_call(action, screen_size=64)
        self.assertEqual(call.arguments, [[1]])

    def test_select_point_uses_screen_coords(self):
        with patch.dict(sys.modules, _fake_pysc2_modules()):
            action = np.array([6, 1.0, 0.0, 0.0], dtype=np.float32)  # select_point
            call = action_to_function_call(action, screen_size=64)
        self.assertEqual(call.arguments, [[0], [63, 0]])

    def test_select_rect_uses_degenerate_rect(self):
        with patch.dict(sys.modules, _fake_pysc2_modules()):
            action = np.array([17, 0.5, 0.5, 0.0], dtype=np.float32)  # select_rect
            call = action_to_function_call(action, screen_size=64)
        self.assertEqual(call.arguments, [[0], [31, 31], [31, 31]])

    def test_minimap_action_uses_minimap_size(self):
        with patch.dict(sys.modules, _fake_pysc2_modules()):
            action = np.array([11, 1.0, 1.0, 0.0], dtype=np.float32)  # Move_minimap
            call = action_to_function_call(action, screen_size=64, minimap_size=32)
        self.assertEqual(call.arguments, [[0], [31, 31]])

    def test_screen_action_uses_screen_size(self):
        with patch.dict(sys.modules, _fake_pysc2_modules()):
            action = np.array([2, 1.0, 1.0, 0.0], dtype=np.float32)  # Move_screen
            call = action_to_function_call(action, screen_size=64, minimap_size=32)
        self.assertEqual(call.arguments, [[0], [63, 63]])


# ---------------------------------------------------------------------------
# BUILD_PREREQUISITES and fn_ids_blocked_by_prerequisites
# ---------------------------------------------------------------------------

class TestBuildPrerequisites(unittest.TestCase):
    """Tests for BUILD_PREREQUISITES table and fn_ids_blocked_by_prerequisites."""

    def test_build_prerequisites_keys_are_valid_fn_ids(self):
        """Every fn_idx key in BUILD_PREREQUISITES must exist in FUNCTION_IDS."""
        for fn_idx in BUILD_PREREQUISITES:
            self.assertIn(fn_idx, FUNCTION_IDS,
                          f"fn_idx {fn_idx} is in BUILD_PREREQUISITES but not in FUNCTION_IDS")

    def test_prereq_structure_names_matches_prerequisites_union(self):
        """PREREQ_STRUCTURE_NAMES must equal the union of all prereq value sets."""
        expected = frozenset(
            name
            for names in BUILD_PREREQUISITES.values()
            for name in names
        )
        self.assertEqual(PREREQ_STRUCTURE_NAMES, expected)

    def test_fn_ids_blocked_empty_counts(self):
        """With no structures built, all actions that have prerequisites are blocked."""
        blocked = fn_ids_blocked_by_prerequisites({})
        for fn_idx in BUILD_PREREQUISITES:
            self.assertIn(fn_idx, blocked,
                          f"fn_idx {fn_idx} should be blocked when no structures exist")

    def test_fn_ids_blocked_returns_frozenset(self):
        """fn_ids_blocked_by_prerequisites always returns a frozenset."""
        result = fn_ids_blocked_by_prerequisites({})
        self.assertIsInstance(result, frozenset)

    def test_build_barracks_blocked_without_supply_depot(self):
        """Build_Barracks_screen (fn_idx 8) requires SupplyDepot."""
        self.assertIn(8, BUILD_PREREQUISITES)
        self.assertIn("SupplyDepot", BUILD_PREREQUISITES[8])
        blocked = fn_ids_blocked_by_prerequisites({})
        self.assertIn(8, blocked)
        unblocked = fn_ids_blocked_by_prerequisites({"SupplyDepot": 1})
        self.assertNotIn(8, unblocked)

    def test_train_marine_blocked_without_barracks(self):
        """Train_Marine_quick (fn_idx 7) requires Barracks."""
        self.assertIn(7, BUILD_PREREQUISITES)
        self.assertIn("Barracks", BUILD_PREREQUISITES[7])
        blocked = fn_ids_blocked_by_prerequisites({})
        self.assertIn(7, blocked)
        unblocked = fn_ids_blocked_by_prerequisites({"Barracks": 1})
        self.assertNotIn(7, unblocked)

    def test_train_thor_requires_factory_and_armory(self):
        """Train_Thor_quick (fn_idx 45) requires both Factory and Armory."""
        self.assertIn(45, BUILD_PREREQUISITES)
        self.assertIn("Factory", BUILD_PREREQUISITES[45])
        self.assertIn("Armory", BUILD_PREREQUISITES[45])
        # Only Factory: still blocked.
        self.assertIn(45, fn_ids_blocked_by_prerequisites({"Factory": 1}))
        # Only Armory: still blocked.
        self.assertIn(45, fn_ids_blocked_by_prerequisites({"Armory": 1}))
        # Both: unblocked.
        self.assertNotIn(45, fn_ids_blocked_by_prerequisites({"Factory": 1, "Armory": 1}))

    def test_actions_without_prerequisites_never_blocked(self):
        """Actions not in BUILD_PREREQUISITES are never returned as blocked."""
        blocked = fn_ids_blocked_by_prerequisites({})
        for fn_idx in FUNCTION_IDS:
            if fn_idx not in BUILD_PREREQUISITES:
                self.assertNotIn(fn_idx, blocked,
                                 f"fn_idx {fn_idx} has no prereqs but appears in blocked set")

    def test_all_prerequisites_are_in_prereq_structure_names(self):
        """Every prerequisite name referenced by any fn_idx must be in PREREQ_STRUCTURE_NAMES."""
        for fn_idx, names in BUILD_PREREQUISITES.items():
            for name in names:
                self.assertIn(name, PREREQ_STRUCTURE_NAMES,
                              f"Prerequisite '{name}' for fn_idx {fn_idx} not in PREREQ_STRUCTURE_NAMES")

    def test_zerg_zergling_requires_spawning_pool(self):
        """Train_Zergling_quick (fn_idx 99) requires SpawningPool."""
        self.assertIn(99, BUILD_PREREQUISITES)
        self.assertIn("SpawningPool", BUILD_PREREQUISITES[99])
        self.assertIn(99, fn_ids_blocked_by_prerequisites({}))
        self.assertNotIn(99, fn_ids_blocked_by_prerequisites({"SpawningPool": 1}))

    def test_protoss_gateway_requires_pylon(self):
        """Build_Gateway_screen (fn_idx 52) requires Pylon."""
        self.assertIn(52, BUILD_PREREQUISITES)
        self.assertIn("Pylon", BUILD_PREREQUISITES[52])
        self.assertIn(52, fn_ids_blocked_by_prerequisites({}))
        self.assertNotIn(52, fn_ids_blocked_by_prerequisites({"Pylon": 1}))


if __name__ == "__main__":
    unittest.main()
