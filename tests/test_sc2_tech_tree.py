"""Tests for the hardcoded SC2 tech-tree precondition logic (issue #346).

These tests verify the precondition table catches the recurring bug
that previous PRs (#307, #311, #315, #317, #322, #339) didn't fully
fix: actions like ``Build_FusionCore_screen`` being chosen when their
prerequisites aren't met.
"""

import unittest

from games.sc2.tech_tree import (
    PRECONDITIONS,
    WORKER_NAMES,
    SelectionReq,
    building_prereqs_met,
    fn_idx_satisfied,
)

_SCV = "SCV"
_PROBE = "Probe"
_DRONE = "Drone"


class TestBuildingPrereqsMet(unittest.TestCase):
    def test_no_prereq_building_always_buildable(self):
        # CommandCenter / Nexus / Hatchery have no prereqs.
        self.assertTrue(building_prereqs_met("CommandCenter", frozenset()))
        self.assertTrue(building_prereqs_met("Nexus", frozenset()))
        self.assertTrue(building_prereqs_met("Hatchery", frozenset()))

    def test_terran_chain(self):
        self.assertFalse(building_prereqs_met("Barracks", frozenset()))
        self.assertTrue(building_prereqs_met("Barracks", frozenset({"SupplyDepot"})))
        self.assertFalse(building_prereqs_met("FusionCore", frozenset({"Barracks"})))
        self.assertTrue(building_prereqs_met("FusionCore", frozenset({"Starport"})))

    def test_or_semantics_zerg_spire(self):
        # Build_Spire requires Lair OR Hive — either one suffices.
        self.assertFalse(building_prereqs_met("Spire", frozenset({"Hatchery"})))
        self.assertTrue(building_prereqs_met("Spire", frozenset({"Lair"})))
        self.assertTrue(building_prereqs_met("Spire", frozenset({"Hive"})))


class TestFnIdxSatisfiedTerran(unittest.TestCase):
    def test_build_fusion_core_requires_starport(self):
        """Issue #346 specific: Build_FusionCore_screen (fn_idx=32) must be
        unsatisfied with only CC + SCV visible, and satisfied with Starport."""
        # SCV selected, CommandCenter only.
        self.assertFalse(fn_idx_satisfied(32, frozenset({"CommandCenter"}), frozenset(), _SCV))
        # SCV selected, Starport exists.
        self.assertTrue(
            fn_idx_satisfied(
                32,
                frozenset({"CommandCenter", "Factory", "Starport"}),
                frozenset(),
                _SCV,
            )
        )

    def test_build_supply_depot_only_needs_worker(self):
        # No buildings required; just an SCV.
        self.assertTrue(fn_idx_satisfied(9, frozenset(), frozenset(), _SCV))
        # No selection → unsatisfied.
        self.assertFalse(fn_idx_satisfied(9, frozenset(), frozenset(), None))
        # Marine selected (wrong type) → unsatisfied.
        self.assertFalse(fn_idx_satisfied(9, frozenset(), frozenset(), "Marine"))

    def test_train_marine_needs_barracks_selected(self):
        # SCV selected → unsatisfied (Barracks needs to be selected).
        self.assertFalse(fn_idx_satisfied(7, frozenset({"Barracks"}), frozenset(), _SCV))
        # Barracks selected → satisfied.
        self.assertTrue(fn_idx_satisfied(7, frozenset({"Barracks"}), frozenset(), "Barracks"))

    def test_train_marauder_needs_tech_lab(self):
        # Barracks selected but no TechLab → unsatisfied.
        self.assertFalse(fn_idx_satisfied(35, frozenset({"Barracks"}), frozenset(), "Barracks"))
        # BarracksTechLab present → satisfied.
        self.assertTrue(
            fn_idx_satisfied(
                35,
                frozenset({"Barracks", "BarracksTechLab"}),
                frozenset(),
                "Barracks",
            )
        )

    def test_train_battlecruiser_chain(self):
        # fn_idx=43 (Battlecruiser) needs StarportTechLab + FusionCore + Starport selected.
        # Missing FusionCore.
        self.assertFalse(fn_idx_satisfied(43, frozenset({"Starport", "StarportTechLab"}), frozenset(), "Starport"))
        # Complete chain.
        self.assertTrue(
            fn_idx_satisfied(
                43,
                frozenset({"Starport", "StarportTechLab", "FusionCore"}),
                frozenset(),
                "Starport",
            )
        )

    def test_effect_stim_requires_research(self):
        # Marine selected but no Stim research → unsatisfied.
        self.assertFalse(fn_idx_satisfied(47, frozenset({"Barracks"}), frozenset(), "Marine"))
        # Stim research completed.
        self.assertTrue(fn_idx_satisfied(47, frozenset({"Barracks"}), frozenset({"Stimpack"}), "Marine"))
        # Stim research but SCV selected → unsatisfied (Marines/Marauders only).
        self.assertFalse(fn_idx_satisfied(47, frozenset({"Barracks"}), frozenset({"Stimpack"}), _SCV))


class TestFnIdxSatisfiedProtoss(unittest.TestCase):
    def test_carrier_needs_fleet_beacon(self):
        # fn_idx=73 (Carrier) needs Stargate selected + FleetBeacon exists.
        self.assertFalse(fn_idx_satisfied(73, frozenset({"Stargate"}), frozenset(), "Stargate"))
        self.assertTrue(fn_idx_satisfied(73, frozenset({"Stargate", "FleetBeacon"}), frozenset(), "Stargate"))

    def test_stalker_from_gateway_or_warpgate(self):
        # fn_idx=67 (Stalker) — accept either Gateway or WarpGate as selected.
        for selected in ("Gateway", "WarpGate"):
            with self.subTest(selected=selected):
                self.assertTrue(fn_idx_satisfied(67, frozenset({"CyberneticsCore"}), frozenset(), selected))


class TestFnIdxSatisfiedZerg(unittest.TestCase):
    def test_hive_needs_lair_and_infestation_pit(self):
        # fn_idx=114 (Morph_Hive_quick) needs Lair selected + InfestationPit exists.
        self.assertFalse(fn_idx_satisfied(114, frozenset(), frozenset(), "Lair"))
        self.assertTrue(fn_idx_satisfied(114, frozenset({"InfestationPit"}), frozenset(), "Lair"))

    def test_lurker_needs_hydra_den(self):
        # fn_idx=111 (Train_Lurker_quick) morphs Hydralisk + needs LurkerDenMP.
        self.assertFalse(fn_idx_satisfied(111, frozenset(), frozenset(), "Hydralisk"))
        self.assertTrue(fn_idx_satisfied(111, frozenset({"LurkerDenMP"}), frozenset(), "Hydralisk"))

    def test_baneling_morph_consumes_zergling(self):
        # fn_idx=100 (Train_Baneling) — needs Zergling selected + BanelingNest exists.
        self.assertFalse(fn_idx_satisfied(100, frozenset(), frozenset(), "Zergling"))
        self.assertFalse(fn_idx_satisfied(100, frozenset({"BanelingNest"}), frozenset(), "Roach"))
        self.assertTrue(fn_idx_satisfied(100, frozenset({"BanelingNest"}), frozenset(), "Zergling"))

    def test_overseer_needs_lair_or_hive(self):
        # fn_idx=115 (Morph_Overseer) — Overlord selected + Lair OR Hive exists.
        self.assertFalse(fn_idx_satisfied(115, frozenset({"Hatchery"}), frozenset(), "Overlord"))
        self.assertTrue(fn_idx_satisfied(115, frozenset({"Lair"}), frozenset(), "Overlord"))
        self.assertTrue(fn_idx_satisfied(115, frozenset({"Hive"}), frozenset(), "Overlord"))
        # Zergling selected (wrong type) → unsatisfied.
        self.assertFalse(fn_idx_satisfied(115, frozenset({"Lair"}), frozenset(), "Zergling"))

    def test_lair_morph_consumes_hatchery(self):
        # fn_idx=113 (Morph_Lair) — Hatchery selected + SpawningPool exists.
        self.assertFalse(fn_idx_satisfied(113, frozenset(), frozenset(), "Hatchery"))
        self.assertTrue(fn_idx_satisfied(113, frozenset({"SpawningPool"}), frozenset(), "Hatchery"))

    def test_brood_lord_needs_greater_spire(self):
        # fn_idx=108 (Train_BroodLord_quick) — Corruptor selected + GreaterSpire exists.
        self.assertFalse(fn_idx_satisfied(108, frozenset({"Spire"}), frozenset(), "Corruptor"))
        self.assertTrue(fn_idx_satisfied(108, frozenset({"GreaterSpire"}), frozenset(), "Corruptor"))


class TestMorphsFullyIntegrated(unittest.TestCase):
    """All morph fn_ids route through ``UNIT_PRODUCERS`` via ``_train``
    rather than the old ``_morph_inplace`` helper (the lone exception is
    Morph_Unsiege, where the result ``SiegeTank`` is already in
    UNIT_PRODUCERS as the Factory-built unit)."""

    def test_archon_accepts_high_or_dark_templar(self):
        # fn_idx=80 (Morph_Archon) — should accept either templar type.
        for selected in ("HighTemplar", "DarkTemplar"):
            with self.subTest(selected=selected):
                self.assertTrue(fn_idx_satisfied(80, frozenset({"TemplarArchive"}), frozenset(), selected))
        # Zealot is not a templar → unsatisfied.
        self.assertFalse(fn_idx_satisfied(80, frozenset({"TemplarArchive"}), frozenset(), "Zealot"))

    def test_siege_mode_consumes_unsieged_tank(self):
        # fn_idx=48 (Morph_SiegeMode) — SiegeTank selected + FactoryTechLab.
        self.assertFalse(fn_idx_satisfied(48, frozenset(), frozenset(), "SiegeTank"))
        self.assertTrue(fn_idx_satisfied(48, frozenset({"FactoryTechLab"}), frozenset(), "SiegeTank"))

    def test_unsiege_consumes_sieged_tank(self):
        # fn_idx=49 (Morph_Unsiege) — SiegeTankSieged selected, no building req.
        self.assertTrue(fn_idx_satisfied(49, frozenset(), frozenset(), "SiegeTankSieged"))
        self.assertFalse(fn_idx_satisfied(49, frozenset(), frozenset(), "SiegeTank"))


class TestUniversalActions(unittest.TestCase):
    def test_no_op_always_satisfied(self):
        self.assertTrue(fn_idx_satisfied(0, frozenset(), frozenset(), None))

    def test_select_army_always_satisfied(self):
        self.assertTrue(fn_idx_satisfied(1, frozenset(), frozenset(), None))

    def test_select_point_always_satisfied(self):
        self.assertTrue(fn_idx_satisfied(6, frozenset(), frozenset(), None))

    def test_move_screen_needs_any_unit(self):
        self.assertFalse(fn_idx_satisfied(2, frozenset(), frozenset(), None))
        self.assertTrue(fn_idx_satisfied(2, frozenset(), frozenset(), "Marine"))


class TestPreconditionsTableShape(unittest.TestCase):
    """Sanity checks on the PRECONDITIONS table itself."""

    def test_every_fn_idx_in_function_ids_has_preconditions(self):
        from games.sc2.actions import FUNCTION_IDS

        for fn_idx in FUNCTION_IDS:
            with self.subTest(fn_idx=fn_idx):
                self.assertIn(fn_idx, PRECONDITIONS)

    def test_universal_actions_have_no_selection_or_any_unit(self):
        # Universal actions (no_op, select_*, Move_*, Stop, …) should never
        # require a specific selected unit-type.
        universal = {0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22}
        for fn_idx in universal:
            with self.subTest(fn_idx=fn_idx):
                pre = PRECONDITIONS[fn_idx]
                self.assertIn(pre.required_selection, (SelectionReq.NONE, SelectionReq.ANY_UNIT))

    def test_build_actions_target_workers(self):
        # All Build_*_screen actions should require a worker selection.
        from games.sc2.actions import FUNCTION_IDS

        for fn_idx, name in FUNCTION_IDS.items():
            if not name.startswith("Build_") or not name.endswith("_screen"):
                continue
            if name == "Build_CreepTumor_screen":  # Queen ability, not worker build
                continue
            with self.subTest(fn_idx=fn_idx, name=name):
                pre = PRECONDITIONS[fn_idx]
                self.assertEqual(pre.required_selection, SelectionReq.OF_TYPE)
                self.assertTrue(
                    pre.selection_target & WORKER_NAMES,
                    f"{name} should accept worker selection",
                )


if __name__ == "__main__":
    unittest.main()
