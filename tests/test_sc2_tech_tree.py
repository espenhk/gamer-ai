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

# Selection-set shorthands.  ``fn_idx_satisfied`` takes a ``frozenset`` so
# that a single API can express "nothing selected" (empty), single-type
# selections (one element), and post-``select_army`` mixed-type
# selections (multiple elements).
_NONE: frozenset[str] = frozenset()
_SCV = frozenset({"SCV"})


def _sel(*names: str) -> frozenset[str]:
    """Build a selection-set from one or more unit-type names."""
    return frozenset(names)


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
        self.assertFalse(fn_idx_satisfied(9, frozenset(), frozenset(), _NONE))
        # Marine selected (wrong type) → unsatisfied.
        self.assertFalse(fn_idx_satisfied(9, frozenset(), frozenset(), _sel("Marine")))

    def test_train_marine_needs_barracks_selected(self):
        # SCV selected → unsatisfied (Barracks needs to be selected).
        self.assertFalse(fn_idx_satisfied(7, frozenset({"Barracks"}), frozenset(), _SCV))
        # Barracks selected → satisfied.
        self.assertTrue(fn_idx_satisfied(7, frozenset({"Barracks"}), frozenset(), _sel("Barracks")))

    def test_train_marauder_needs_tech_lab(self):
        # Barracks selected but no TechLab → unsatisfied.
        self.assertFalse(fn_idx_satisfied(35, frozenset({"Barracks"}), frozenset(), _sel("Barracks")))
        # BarracksTechLab present → satisfied.
        self.assertTrue(
            fn_idx_satisfied(
                35,
                frozenset({"Barracks", "BarracksTechLab"}),
                frozenset(),
                _sel("Barracks"),
            )
        )

    def test_train_battlecruiser_chain(self):
        # fn_idx=43 (Battlecruiser) needs StarportTechLab + FusionCore + Starport selected.
        # Missing FusionCore.
        self.assertFalse(
            fn_idx_satisfied(
                43,
                frozenset({"Starport", "StarportTechLab"}),
                frozenset(),
                _sel("Starport"),
            )
        )
        # Complete chain.
        self.assertTrue(
            fn_idx_satisfied(
                43,
                frozenset({"Starport", "StarportTechLab", "FusionCore"}),
                frozenset(),
                _sel("Starport"),
            )
        )

    def test_effect_stim_requires_research(self):
        # Marine selected but no Stim research → unsatisfied.
        self.assertFalse(fn_idx_satisfied(47, frozenset({"Barracks"}), frozenset(), _sel("Marine")))
        # Stim research completed.
        self.assertTrue(fn_idx_satisfied(47, frozenset({"Barracks"}), frozenset({"Stimpack"}), _sel("Marine")))
        # Stim research but SCV selected → unsatisfied (Marines/Marauders only).
        self.assertFalse(fn_idx_satisfied(47, frozenset({"Barracks"}), frozenset({"Stimpack"}), _SCV))


class TestFnIdxSatisfiedProtoss(unittest.TestCase):
    def test_carrier_needs_fleet_beacon(self):
        # fn_idx=73 (Carrier) needs Stargate selected + FleetBeacon exists.
        self.assertFalse(fn_idx_satisfied(73, frozenset({"Stargate"}), frozenset(), _sel("Stargate")))
        self.assertTrue(fn_idx_satisfied(73, frozenset({"Stargate", "FleetBeacon"}), frozenset(), _sel("Stargate")))

    def test_stalker_from_gateway_or_warpgate(self):
        # fn_idx=67 (Stalker) — accept either Gateway or WarpGate as selected.
        for selected in ("Gateway", "WarpGate"):
            with self.subTest(selected=selected):
                self.assertTrue(fn_idx_satisfied(67, frozenset({"CyberneticsCore"}), frozenset(), _sel(selected)))


class TestFnIdxSatisfiedZerg(unittest.TestCase):
    def test_hive_needs_lair_and_infestation_pit(self):
        # fn_idx=114 (Morph_Hive_quick) needs Lair selected + InfestationPit exists.
        self.assertFalse(fn_idx_satisfied(114, frozenset(), frozenset(), _sel("Lair")))
        self.assertTrue(fn_idx_satisfied(114, frozenset({"InfestationPit"}), frozenset(), _sel("Lair")))

    def test_lurker_needs_hydra_den(self):
        # fn_idx=111 (Train_Lurker_quick) morphs Hydralisk + needs LurkerDenMP.
        self.assertFalse(fn_idx_satisfied(111, frozenset(), frozenset(), _sel("Hydralisk")))
        self.assertTrue(fn_idx_satisfied(111, frozenset({"LurkerDenMP"}), frozenset(), _sel("Hydralisk")))

    def test_baneling_morph_consumes_zergling(self):
        # fn_idx=100 (Train_Baneling) — needs Zergling selected + BanelingNest exists.
        self.assertFalse(fn_idx_satisfied(100, frozenset(), frozenset(), _sel("Zergling")))
        self.assertFalse(fn_idx_satisfied(100, frozenset({"BanelingNest"}), frozenset(), _sel("Roach")))
        self.assertTrue(fn_idx_satisfied(100, frozenset({"BanelingNest"}), frozenset(), _sel("Zergling")))

    def test_overseer_needs_lair_or_hive(self):
        # fn_idx=115 (Morph_Overseer) — Overlord selected + Lair OR Hive exists.
        self.assertFalse(fn_idx_satisfied(115, frozenset({"Hatchery"}), frozenset(), _sel("Overlord")))
        self.assertTrue(fn_idx_satisfied(115, frozenset({"Lair"}), frozenset(), _sel("Overlord")))
        self.assertTrue(fn_idx_satisfied(115, frozenset({"Hive"}), frozenset(), _sel("Overlord")))
        # Zergling selected (wrong type) → unsatisfied.
        self.assertFalse(fn_idx_satisfied(115, frozenset({"Lair"}), frozenset(), _sel("Zergling")))

    def test_lair_morph_consumes_hatchery(self):
        # fn_idx=113 (Morph_Lair) — Hatchery selected + SpawningPool exists.
        self.assertFalse(fn_idx_satisfied(113, frozenset(), frozenset(), _sel("Hatchery")))
        self.assertTrue(fn_idx_satisfied(113, frozenset({"SpawningPool"}), frozenset(), _sel("Hatchery")))

    def test_brood_lord_needs_greater_spire(self):
        # fn_idx=108 (Train_BroodLord_quick) — Corruptor selected + GreaterSpire exists.
        self.assertFalse(fn_idx_satisfied(108, frozenset({"Spire"}), frozenset(), _sel("Corruptor")))
        self.assertTrue(fn_idx_satisfied(108, frozenset({"GreaterSpire"}), frozenset(), _sel("Corruptor")))


class TestMorphsFullyIntegrated(unittest.TestCase):
    """All morph fn_ids route through ``UNIT_PRODUCERS`` via ``_train``
    rather than the old ``_morph_inplace`` helper (the lone exception is
    Morph_Unsiege, where the result ``SiegeTank`` is already in
    UNIT_PRODUCERS as the Factory-built unit)."""

    def test_archon_accepts_high_or_dark_templar(self):
        # fn_idx=80 (Morph_Archon) — should accept either templar type.
        for selected in ("HighTemplar", "DarkTemplar"):
            with self.subTest(selected=selected):
                self.assertTrue(fn_idx_satisfied(80, frozenset({"TemplarArchive"}), frozenset(), _sel(selected)))
        # Zealot is not a templar → unsatisfied.
        self.assertFalse(fn_idx_satisfied(80, frozenset({"TemplarArchive"}), frozenset(), _sel("Zealot")))

    def test_siege_mode_consumes_unsieged_tank(self):
        # fn_idx=48 (Morph_SiegeMode) — SiegeTank selected + FactoryTechLab.
        self.assertFalse(fn_idx_satisfied(48, frozenset(), frozenset(), _sel("SiegeTank")))
        self.assertTrue(fn_idx_satisfied(48, frozenset({"FactoryTechLab"}), frozenset(), _sel("SiegeTank")))

    def test_unsiege_consumes_sieged_tank(self):
        # fn_idx=49 (Morph_Unsiege) — SiegeTankSieged selected, no building req.
        self.assertTrue(fn_idx_satisfied(49, frozenset(), frozenset(), _sel("SiegeTankSieged")))
        self.assertFalse(fn_idx_satisfied(49, frozenset(), frozenset(), _sel("SiegeTank")))


class TestUniversalActions(unittest.TestCase):
    def test_no_op_always_satisfied(self):
        self.assertTrue(fn_idx_satisfied(0, frozenset(), frozenset(), _NONE))

    def test_select_army_always_satisfied(self):
        self.assertTrue(fn_idx_satisfied(1, frozenset(), frozenset(), _NONE))

    def test_select_point_always_satisfied(self):
        self.assertTrue(fn_idx_satisfied(6, frozenset(), frozenset(), _NONE))

    def test_move_screen_needs_any_unit(self):
        self.assertFalse(fn_idx_satisfied(2, frozenset(), frozenset(), _NONE))
        self.assertTrue(fn_idx_satisfied(2, frozenset(), frozenset(), _sel("Marine")))


class TestMixedSelection(unittest.TestCase):
    """A mixed-type selection (post-``select_army``) must still satisfy
    ANY_UNIT and OF_TYPE actions whose target intersects the selection.
    Regression for the bug Copilot reviewer caught on PR #348."""

    def test_mixed_army_satisfies_move(self):
        # After select_army on a Marine + Marauder army, Move_screen must
        # still be available even though no single "type" describes the selection.
        self.assertTrue(fn_idx_satisfied(2, frozenset(), frozenset(), _sel("Marine", "Marauder")))

    def test_mixed_army_satisfies_stim(self):
        # Effect_Stim needs Marine or Marauder selected — a mixed selection
        # containing either should satisfy it (PySC2 applies Stim to both).
        self.assertTrue(
            fn_idx_satisfied(
                47,
                frozenset({"Barracks"}),
                frozenset({"Stimpack"}),
                _sel("Marine", "Marauder"),
            )
        )

    def test_mixed_army_without_target_type_still_fails_of_type(self):
        # Train_Marine needs Barracks selected; a mixed army selection
        # without any Barracks should still fail.
        self.assertFalse(fn_idx_satisfied(7, frozenset({"Barracks"}), frozenset(), _sel("Marine", "Marauder")))


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
