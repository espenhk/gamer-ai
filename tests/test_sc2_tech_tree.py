"""Tests for the hardcoded SC2 tech-tree precondition logic (issue #346).

These tests verify the precondition table catches the recurring bug
that previous PRs (#307, #311, #315, #317, #322, #339) didn't fully
fix: actions like ``Build_FusionCore_screen`` being chosen when their
prerequisites aren't met.
"""

import unittest

from games.sc2.tech_tree import (
    PRECONDITIONS,
    RESOURCE_COSTS,
    WORKER_NAMES,
    SelectionReq,
    building_prereqs_met,
    fn_idx_satisfied,
)

# Accessible-type shorthands used by fn_idx_satisfied.  The parameter now
# represents all unit/building types the agent *could* select (owned
# buildings + visible units), not just the current selection.
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
        """Issue #346 specific: Build_FusionCore_screen (fn_idx=30) must be
        unsatisfied with only CC + SCV visible, and satisfied with Starport."""
        # SCV selected, CommandCenter only.
        self.assertFalse(fn_idx_satisfied(30, frozenset({"CommandCenter"}), frozenset(), _SCV))
        # SCV selected, Starport exists.
        self.assertTrue(
            fn_idx_satisfied(
                30,
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

    def test_train_marine_needs_barracks_accessible(self):
        # No Barracks in accessible types → unsatisfied even with SCV accessible.
        self.assertFalse(fn_idx_satisfied(7, frozenset({"Barracks"}), frozenset(), _SCV))
        # Barracks accessible (e.g. owned + visible) → satisfied even when SCV
        # is also accessible (the resolver will auto-select the Barracks).
        self.assertTrue(fn_idx_satisfied(7, frozenset({"Barracks"}), frozenset(), _sel("Barracks")))
        self.assertTrue(fn_idx_satisfied(7, frozenset({"Barracks"}), frozenset(), _sel("SCV", "Barracks")))
        # No Barracks at all → unsatisfied.
        self.assertFalse(fn_idx_satisfied(7, frozenset(), frozenset(), _SCV))

    def test_train_marauder_needs_tech_lab(self):
        # Barracks selected but no TechLab → unsatisfied.
        self.assertFalse(fn_idx_satisfied(33, frozenset({"Barracks"}), frozenset(), _sel("Barracks")))
        # BarracksTechLab present → satisfied.
        self.assertTrue(
            fn_idx_satisfied(
                33,
                frozenset({"Barracks", "BarracksTechLab"}),
                frozenset(),
                _sel("Barracks"),
            )
        )

    def test_train_battlecruiser_chain(self):
        # fn_idx=41 (Battlecruiser) needs StarportTechLab + FusionCore + Starport selected.
        # Missing FusionCore.
        self.assertFalse(
            fn_idx_satisfied(
                41,
                frozenset({"Starport", "StarportTechLab"}),
                frozenset(),
                _sel("Starport"),
            )
        )
        # Complete chain.
        self.assertTrue(
            fn_idx_satisfied(
                41,
                frozenset({"Starport", "StarportTechLab", "FusionCore"}),
                frozenset(),
                _sel("Starport"),
            )
        )

    def test_effect_stim_requires_research(self):
        # Marine selected but no Stim research → unsatisfied.
        self.assertFalse(fn_idx_satisfied(45, frozenset({"Barracks"}), frozenset(), _sel("Marine")))
        # Stim research completed.
        self.assertTrue(fn_idx_satisfied(45, frozenset({"Barracks"}), frozenset({"Stimpack"}), _sel("Marine")))
        # Stim research but SCV selected → unsatisfied (Marines/Marauders only).
        self.assertFalse(fn_idx_satisfied(45, frozenset({"Barracks"}), frozenset({"Stimpack"}), _SCV))


class TestFnIdxSatisfiedProtoss(unittest.TestCase):
    def test_carrier_needs_fleet_beacon(self):
        # fn_idx=71 (Carrier) needs Stargate selected + FleetBeacon exists.
        self.assertFalse(fn_idx_satisfied(71, frozenset({"Stargate"}), frozenset(), _sel("Stargate")))
        self.assertTrue(fn_idx_satisfied(71, frozenset({"Stargate", "FleetBeacon"}), frozenset(), _sel("Stargate")))

    def test_stalker_from_gateway_or_warpgate(self):
        # fn_idx=65 (Stalker) — accept either Gateway or WarpGate as selected.
        for selected in ("Gateway", "WarpGate"):
            with self.subTest(selected=selected):
                self.assertTrue(fn_idx_satisfied(65, frozenset({"CyberneticsCore"}), frozenset(), _sel(selected)))


class TestFnIdxSatisfiedZerg(unittest.TestCase):
    def test_hive_needs_lair_and_infestation_pit(self):
        # fn_idx=112 (Morph_Hive_quick) needs Lair selected + InfestationPit exists.
        self.assertFalse(fn_idx_satisfied(112, frozenset(), frozenset(), _sel("Lair")))
        self.assertTrue(fn_idx_satisfied(112, frozenset({"InfestationPit"}), frozenset(), _sel("Lair")))

    def test_lurker_needs_hydra_den(self):
        # fn_idx=109 (Train_Lurker_quick) morphs Hydralisk + needs LurkerDenMP.
        self.assertFalse(fn_idx_satisfied(109, frozenset(), frozenset(), _sel("Hydralisk")))
        self.assertTrue(fn_idx_satisfied(109, frozenset({"LurkerDenMP"}), frozenset(), _sel("Hydralisk")))

    def test_baneling_morph_consumes_zergling(self):
        # fn_idx=98 (Train_Baneling) — needs Zergling selected + BanelingNest exists.
        self.assertFalse(fn_idx_satisfied(98, frozenset(), frozenset(), _sel("Zergling")))
        self.assertFalse(fn_idx_satisfied(98, frozenset({"BanelingNest"}), frozenset(), _sel("Roach")))
        self.assertTrue(fn_idx_satisfied(98, frozenset({"BanelingNest"}), frozenset(), _sel("Zergling")))

    def test_overseer_needs_lair_or_hive(self):
        # fn_idx=113 (Morph_Overseer) — Overlord selected + Lair OR Hive exists.
        self.assertFalse(fn_idx_satisfied(113, frozenset({"Hatchery"}), frozenset(), _sel("Overlord")))
        self.assertTrue(fn_idx_satisfied(113, frozenset({"Lair"}), frozenset(), _sel("Overlord")))
        self.assertTrue(fn_idx_satisfied(113, frozenset({"Hive"}), frozenset(), _sel("Overlord")))
        # Zergling selected (wrong type) → unsatisfied.
        self.assertFalse(fn_idx_satisfied(113, frozenset({"Lair"}), frozenset(), _sel("Zergling")))

    def test_lair_morph_consumes_hatchery(self):
        # fn_idx=111 (Morph_Lair) — Hatchery selected + SpawningPool exists.
        self.assertFalse(fn_idx_satisfied(111, frozenset(), frozenset(), _sel("Hatchery")))
        self.assertTrue(fn_idx_satisfied(111, frozenset({"SpawningPool"}), frozenset(), _sel("Hatchery")))

    def test_brood_lord_needs_greater_spire(self):
        # fn_idx=106 (Train_BroodLord_quick) — Corruptor selected + GreaterSpire exists.
        self.assertFalse(fn_idx_satisfied(106, frozenset({"Spire"}), frozenset(), _sel("Corruptor")))
        self.assertTrue(fn_idx_satisfied(106, frozenset({"GreaterSpire"}), frozenset(), _sel("Corruptor")))


class TestMorphsFullyIntegrated(unittest.TestCase):
    """All morph fn_ids route through ``UNIT_PRODUCERS`` via ``_train``
    rather than the old ``_morph_inplace`` helper (the lone exception is
    Morph_Unsiege, where the result ``SiegeTank`` is already in
    UNIT_PRODUCERS as the Factory-built unit)."""

    def test_archon_accepts_high_or_dark_templar(self):
        # fn_idx=78 (Morph_Archon) — should accept either templar type.
        for selected in ("HighTemplar", "DarkTemplar"):
            with self.subTest(selected=selected):
                self.assertTrue(fn_idx_satisfied(78, frozenset({"TemplarArchive"}), frozenset(), _sel(selected)))
        # Zealot is not a templar → unsatisfied.
        self.assertFalse(fn_idx_satisfied(78, frozenset({"TemplarArchive"}), frozenset(), _sel("Zealot")))

    def test_siege_mode_consumes_unsieged_tank(self):
        # fn_idx=46 (Morph_SiegeMode) — SiegeTank selected + FactoryTechLab.
        self.assertFalse(fn_idx_satisfied(46, frozenset(), frozenset(), _sel("SiegeTank")))
        self.assertTrue(fn_idx_satisfied(46, frozenset({"FactoryTechLab"}), frozenset(), _sel("SiegeTank")))

    def test_unsiege_consumes_sieged_tank(self):
        # fn_idx=47 (Morph_Unsiege) — SiegeTankSieged selected, no building req.
        self.assertTrue(fn_idx_satisfied(47, frozenset(), frozenset(), _sel("SiegeTankSieged")))
        self.assertFalse(fn_idx_satisfied(47, frozenset(), frozenset(), _sel("SiegeTank")))


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
                45,
                frozenset({"Barracks"}),
                frozenset({"Stimpack"}),
                _sel("Marine", "Marauder"),
            )
        )

    def test_mixed_army_without_barracks_accessible_fails_train_marine(self):
        # Train_Marine needs Barracks accessible.  A mixed army (Marine +
        # Marauder visible) without any Barracks accessible → unsatisfied.
        self.assertFalse(fn_idx_satisfied(7, frozenset(), frozenset(), _sel("Marine", "Marauder")))


class TestResourceCostFilter(unittest.TestCase):
    """Resource-cost gating (issue #357): actions whose mineral/vespene cost
    exceeds current supply must be excluded from the available mask."""

    def test_no_minerals_blocks_supply_depot(self):
        # fn_idx=9 (Build_SupplyDepot) costs 100 minerals.
        # Satisfied with enough minerals, blocked without.
        self.assertTrue(fn_idx_satisfied(9, frozenset(), frozenset(), _SCV, minerals=100))
        self.assertFalse(fn_idx_satisfied(9, frozenset(), frozenset(), _SCV, minerals=99))
        self.assertFalse(fn_idx_satisfied(9, frozenset(), frozenset(), _SCV, minerals=0))

    def test_exactly_at_cost_is_allowed(self):
        # Boundary: exactly 150 minerals for Barracks (fn_idx=8).
        # Barracks also requires a SupplyDepot as a building prerequisite.
        self.assertTrue(fn_idx_satisfied(8, frozenset({"SupplyDepot"}), frozenset(), _SCV, minerals=150))
        self.assertFalse(fn_idx_satisfied(8, frozenset({"SupplyDepot"}), frozenset(), _SCV, minerals=149))

    def test_vespene_cost_gating(self):
        # fn_idx=24 (Build_Factory) costs 150 minerals + 100 vespene.
        # Fail on insufficient vespene even if minerals are sufficient.
        self.assertFalse(fn_idx_satisfied(24, frozenset({"Barracks"}), frozenset(), _SCV, minerals=200, vespene=99))
        self.assertTrue(fn_idx_satisfied(24, frozenset({"Barracks"}), frozenset(), _SCV, minerals=150, vespene=100))

    def test_battlecruiser_double_cost(self):
        # fn_idx=41 (Train_Battlecruiser) costs 400 minerals + 300 vespene.
        self.assertFalse(
            fn_idx_satisfied(
                41,
                frozenset({"Starport", "StarportTechLab", "FusionCore"}),
                frozenset(),
                _sel("Starport"),
                minerals=399,
                vespene=300,
            )
        )
        self.assertFalse(
            fn_idx_satisfied(
                41,
                frozenset({"Starport", "StarportTechLab", "FusionCore"}),
                frozenset(),
                _sel("Starport"),
                minerals=400,
                vespene=299,
            )
        )
        self.assertTrue(
            fn_idx_satisfied(
                41,
                frozenset({"Starport", "StarportTechLab", "FusionCore"}),
                frozenset(),
                _sel("Starport"),
                minerals=400,
                vespene=300,
            )
        )

    def test_no_cost_actions_unaffected(self):
        # Universal actions with no resource cost must always pass the resource check.
        # fn_idx=0 (no_op), fn_idx=1 (select_army), fn_idx=2 (Move_screen).
        self.assertTrue(fn_idx_satisfied(0, frozenset(), frozenset(), _NONE, minerals=0, vespene=0))
        self.assertTrue(fn_idx_satisfied(1, frozenset(), frozenset(), _NONE, minerals=0, vespene=0))
        self.assertTrue(fn_idx_satisfied(2, frozenset(), frozenset(), _sel("Marine"), minerals=0, vespene=0))

    def test_morph_actions_with_zero_cost(self):
        # Morph_SiegeMode (fn_idx=46) and Morph_Unsiege (fn_idx=47) have no
        # resource cost (mode changes only).
        self.assertFalse(fn_idx_satisfied(46, frozenset(), frozenset(), _sel("SiegeTank"), minerals=0))
        # fails because missing FactoryTechLab prerequisite, not resource cost
        self.assertTrue(fn_idx_satisfied(46, frozenset({"FactoryTechLab"}), frozenset(), _sel("SiegeTank"), minerals=0))
        self.assertTrue(fn_idx_satisfied(47, frozenset(), frozenset(), _sel("SiegeTankSieged"), minerals=0))

    def test_backwards_compat_no_resource_args(self):
        # Calling fn_idx_satisfied without minerals/vespene defaults to inf
        # so all existing callers remain unaffected.
        self.assertTrue(fn_idx_satisfied(9, frozenset(), frozenset(), _SCV))
        self.assertFalse(fn_idx_satisfied(9, frozenset(), frozenset(), _NONE))

    def test_resource_costs_table_coverage(self):
        # Every fn_idx in RESOURCE_COSTS must be within the known fn_idx range
        # (0–115 as defined in FUNCTION_IDS) and must appear in PRECONDITIONS.
        # (We avoid importing games.sc2.actions here because it requires numpy.)
        for fn_idx in RESOURCE_COSTS:
            with self.subTest(fn_idx=fn_idx):
                self.assertIn(fn_idx, PRECONDITIONS, f"fn_idx {fn_idx} in RESOURCE_COSTS but not in PRECONDITIONS")

    def test_zerg_morph_cost(self):
        # fn_idx=111 (Morph_Lair) costs 150 minerals + 100 vespene.
        self.assertFalse(
            fn_idx_satisfied(111, frozenset({"SpawningPool"}), frozenset(), _sel("Hatchery"), minerals=100, vespene=100)
        )
        self.assertTrue(
            fn_idx_satisfied(111, frozenset({"SpawningPool"}), frozenset(), _sel("Hatchery"), minerals=150, vespene=100)
        )


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
        universal = {0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15, 17, 18, 19, 20}
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
