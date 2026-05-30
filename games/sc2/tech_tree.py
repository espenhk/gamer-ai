"""StarCraft 2 tech-tree preconditions for action gating (issue #346).

PySC2's ``available_actions`` mask only checks whether the *selected
unit* has the ability button — not whether the *building/upgrade
prerequisites* are met. With an SCV selected, PySC2 will happily report
``Build_FusionCore_screen`` as "available" even with no Starport on the
map. The race-only filter previously added in ``client.py`` does not
narrow this further.

This module hardcodes a tech tree (buildings + research) plus
selection requirements per fn_idx, so the client can produce an action
mask that is exactly the set of actions the agent can *execute* this
step.

Three lookup tables drive the per-fn_idx :class:`Preconditions` records:

- :data:`BUILDING_PREREQS` — building name → frozenset of buildings
  (OR-set) that must already exist to build it.
- :data:`UNIT_PRODUCERS` — unit name → frozenset of producing-building
  names (OR-set; e.g. Stalker comes from Gateway *or* WarpGate).
- :data:`UPGRADE_PREREQS` — research/upgrade name → frozenset of
  required buildings (typically the add-on, e.g. ``BarracksTechLab``
  for Stim).

Name conventions follow ``pysc2.lib.units`` enum member names so the
client can match these strings against ``feature_units`` unit-type
labels without any translation layer.

References
----------
- Liquipedia LotV tech trees (Terran/Protoss/Zerg).
- PySC2 issues #163 / #291 documenting ``available_actions`` semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class SelectionReq(Enum):
    """What kind of selection an action requires."""

    NONE = auto()  # selection ignored: no_op, select_*, etc.
    ANY_UNIT = auto()  # any non-building friendly unit selected
    OF_TYPE = auto()  # selected unit-type must be in selection_target


@dataclass(frozen=True)
class Preconditions:
    """Per-fn_idx preconditions for execution.

    ``required_buildings`` is a tuple of OR-sets (DNF form): each
    frozenset is satisfied when *any* of its members exists in
    ``owned_buildings``; the tuple is satisfied when *all* of its
    OR-sets are satisfied.  E.g. ``({"Lair", "Hive"},)`` means "Lair OR
    Hive must exist", while ``({"StarportTechLab"}, {"FusionCore"})``
    means "StarportTechLab AND FusionCore must both exist".
    """

    required_buildings: tuple[frozenset[str], ...] = ()
    required_upgrades: frozenset[str] = frozenset()
    required_selection: SelectionReq = SelectionReq.NONE
    # OR-set of acceptable selected unit-type names; only meaningful when
    # required_selection == OF_TYPE. e.g. {"SCV", "Probe", "Drone"} for
    # Build actions, {"Barracks"} for Train_Marine, {"Gateway", "WarpGate"}
    # for Train_Zealot.
    selection_target: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Worker unit-type names — one per race.
# ---------------------------------------------------------------------------
WORKER_NAMES: frozenset[str] = frozenset({"SCV", "Probe", "Drone"})

# ---------------------------------------------------------------------------
# Building prerequisites (per race).
# ---------------------------------------------------------------------------
# Each entry maps a building's pysc2.lib.units name to the set of other
# buildings that must already exist somewhere on the map for it to be
# buildable.  Entries with an empty set have no building prerequisites
# (mineral cost / supply etc. are not modeled).
#
# Source: Liquipedia LotV tech-tree pages.

_TERRAN_BUILDING_PREREQS: dict[str, frozenset[str]] = {
    "CommandCenter": frozenset(),
    "SupplyDepot": frozenset(),
    "Refinery": frozenset(),
    "Barracks": frozenset({"SupplyDepot"}),
    "EngineeringBay": frozenset({"CommandCenter"}),
    "Bunker": frozenset({"Barracks"}),
    "MissileTurret": frozenset({"EngineeringBay"}),
    "Factory": frozenset({"Barracks"}),
    "GhostAcademy": frozenset({"Barracks"}),
    "Armory": frozenset({"Factory"}),
    "Starport": frozenset({"Factory"}),
    "FusionCore": frozenset({"Starport"}),
}

_PROTOSS_BUILDING_PREREQS: dict[str, frozenset[str]] = {
    "Nexus": frozenset(),
    "Pylon": frozenset(),
    "Assimilator": frozenset(),
    "Gateway": frozenset({"Pylon"}),
    "Forge": frozenset({"Nexus"}),
    "CyberneticsCore": frozenset({"Gateway"}),
    "PhotonCannon": frozenset({"Forge"}),
    "ShieldBattery": frozenset({"CyberneticsCore"}),
    "RoboticsFacility": frozenset({"CyberneticsCore"}),
    "Stargate": frozenset({"CyberneticsCore"}),
    "TwilightCouncil": frozenset({"CyberneticsCore"}),
    "TemplarArchive": frozenset({"TwilightCouncil"}),
    "DarkShrine": frozenset({"TwilightCouncil"}),
    "RoboticsBay": frozenset({"RoboticsFacility"}),
    "FleetBeacon": frozenset({"Stargate"}),
}

_ZERG_BUILDING_PREREQS: dict[str, frozenset[str]] = {
    "Hatchery": frozenset(),
    "Extractor": frozenset(),
    "SpawningPool": frozenset({"Hatchery", "Lair", "Hive"}),
    "EvolutionChamber": frozenset({"Hatchery", "Lair", "Hive"}),
    "RoachWarren": frozenset({"SpawningPool"}),
    "BanelingNest": frozenset({"SpawningPool"}),
    "SpineCrawler": frozenset({"SpawningPool"}),
    "SporeCrawler": frozenset({"EvolutionChamber"}),
    "HydraliskDen": frozenset({"Lair", "Hive"}),
    "LurkerDenMP": frozenset({"HydraliskDen"}),
    "Spire": frozenset({"Lair", "Hive"}),
    "NydusNetwork": frozenset({"Lair", "Hive"}),
    "InfestationPit": frozenset({"Lair", "Hive"}),
    "UltraliskCavern": frozenset({"Hive"}),
}

BUILDING_PREREQS: dict[str, frozenset[str]] = {
    **_TERRAN_BUILDING_PREREQS,
    **_PROTOSS_BUILDING_PREREQS,
    **_ZERG_BUILDING_PREREQS,
}


# ---------------------------------------------------------------------------
# Unit producers (per race).
# ---------------------------------------------------------------------------
# Maps unit name → frozenset of producing-building names (OR-set: any one
# of those buildings selected suffices). Additional building / upgrade
# prerequisites (e.g. Marauder needs BarracksTechLab) live in the
# per-fn_idx Preconditions record below.

_TERRAN_UNIT_PRODUCERS: dict[str, frozenset[str]] = {
    "SCV": frozenset({"CommandCenter", "OrbitalCommand", "PlanetaryFortress"}),
    "Marine": frozenset({"Barracks"}),
    "Marauder": frozenset({"Barracks"}),
    "Reaper": frozenset({"Barracks"}),
    "Ghost": frozenset({"Barracks"}),
    "Hellion": frozenset({"Factory"}),
    "WidowMine": frozenset({"Factory"}),
    "SiegeTank": frozenset({"Factory"}),
    "Cyclone": frozenset({"Factory"}),
    "Thor": frozenset({"Factory"}),
    "Viking": frozenset({"Starport"}),
    "Medivac": frozenset({"Starport"}),
    "Liberator": frozenset({"Starport"}),
    "Raven": frozenset({"Starport"}),
    "Banshee": frozenset({"Starport"}),
    "Battlecruiser": frozenset({"Starport"}),
    # Unit-state morphs: SiegeMode produces a SiegeTankSieged from a
    # SiegeTank; Unsiege does the reverse. Modeling these in UNIT_PRODUCERS
    # lets _train() handle them uniformly with the rest of the table.
    "SiegeTankSieged": frozenset({"SiegeTank"}),
}

_PROTOSS_UNIT_PRODUCERS: dict[str, frozenset[str]] = {
    "Probe": frozenset({"Nexus"}),
    "Mothership": frozenset({"Nexus"}),
    "Zealot": frozenset({"Gateway", "WarpGate"}),
    "Stalker": frozenset({"Gateway", "WarpGate"}),
    "Adept": frozenset({"Gateway", "WarpGate"}),
    "Sentry": frozenset({"Gateway", "WarpGate"}),
    "HighTemplar": frozenset({"Gateway", "WarpGate"}),
    "DarkTemplar": frozenset({"Gateway", "WarpGate"}),
    "Immortal": frozenset({"RoboticsFacility"}),
    "Observer": frozenset({"RoboticsFacility"}),
    "WarpPrism": frozenset({"RoboticsFacility"}),
    "Colossus": frozenset({"RoboticsFacility"}),
    "Disruptor": frozenset({"RoboticsFacility"}),
    "Phoenix": frozenset({"Stargate"}),
    "VoidRay": frozenset({"Stargate"}),
    "Oracle": frozenset({"Stargate"}),
    "Tempest": frozenset({"Stargate"}),
    "Carrier": frozenset({"Stargate"}),
    # Templar merge: either HighTemplar or DarkTemplar can morph into an
    # Archon (two source units consumed; the selection-type filter only
    # checks that the right *type* is selected).
    "Archon": frozenset({"HighTemplar", "DarkTemplar"}),
}

_ZERG_UNIT_PRODUCERS: dict[str, frozenset[str]] = {
    "Drone": frozenset({"Larva"}),
    "Overlord": frozenset({"Larva"}),
    "Zergling": frozenset({"Larva"}),
    "Roach": frozenset({"Larva"}),
    "Hydralisk": frozenset({"Larva"}),
    "Infestor": frozenset({"Larva"}),
    "SwarmHost": frozenset({"Larva"}),
    "Mutalisk": frozenset({"Larva"}),
    "Corruptor": frozenset({"Larva"}),
    "Viper": frozenset({"Larva"}),
    "Ultralisk": frozenset({"Larva"}),
    "Queen": frozenset({"Hatchery", "Lair", "Hive"}),
    # Unit-level morphs (source unit consumed → target unit).  Modeling
    # these as "producers" lets the train/morph fn_idx entries share the
    # same _train() helper as every other unit, and the OR-set semantics
    # naturally handle Overlord→Overseer (Lair or Hive must exist).
    "Baneling": frozenset({"Zergling"}),
    "Ravager": frozenset({"Roach"}),
    "Lurker": frozenset({"Hydralisk"}),
    "BroodLord": frozenset({"Corruptor"}),
    "Overseer": frozenset({"Overlord"}),
    # Building morphs (Hatchery → Lair → Hive, Spire → GreaterSpire).
    "Lair": frozenset({"Hatchery"}),
    "Hive": frozenset({"Lair"}),
    "GreaterSpire": frozenset({"Spire"}),
}

UNIT_PRODUCERS: dict[str, frozenset[str]] = {
    **_TERRAN_UNIT_PRODUCERS,
    **_PROTOSS_UNIT_PRODUCERS,
    **_ZERG_UNIT_PRODUCERS,
}


# ---------------------------------------------------------------------------
# Upgrade / research prerequisites.
# ---------------------------------------------------------------------------
# Names follow pysc2.lib.upgrades members (UpgradeId enum). The mapped
# frozenset is the set of buildings that must exist for that research to
# complete — usually the add-on building (BarracksTechLab, etc.).

UPGRADE_PREREQS: dict[str, frozenset[str]] = {
    # Terran upgrades referenced by abilities we expose.
    "Stimpack": frozenset({"BarracksTechLab"}),
    "CombatShield": frozenset({"BarracksTechLab"}),
    "ConcussiveShells": frozenset({"BarracksTechLab"}),
    "SiegeTech": frozenset({"FactoryTechLab"}),
    "DrillingClaws": frozenset({"FactoryTechLab", "Armory"}),
    "BansheeCloak": frozenset({"StarportTechLab"}),
    "BansheeSpeed": frozenset({"StarportTechLab"}),
}


# ---------------------------------------------------------------------------
# Structure / building name set.
# ---------------------------------------------------------------------------
# Friendly unit-type names that count as structures for tech-tree purposes.
# ``SC2Client._compute_owned_buildings`` filters its scan of ``feature_units``
# down to this set so that SCVs / Marines / Probes don't pollute the
# building set (which would bloat the state-dump's "buildings" line and
# fool any future logic that scans owned_buildings).
#
# Covers every structure referenced by BUILDING_PREREQS, UNIT_PRODUCERS,
# and PRECONDITIONS.required_buildings, plus add-ons and morphed-tier
# buildings (Lair, Hive, OrbitalCommand, PlanetaryFortress, WarpGate,
# GreaterSpire) and the six Terran tech-lab / reactor add-ons.

STRUCTURE_NAMES: frozenset[str] = frozenset(
    {
        # Terran
        "CommandCenter",
        "OrbitalCommand",
        "PlanetaryFortress",
        "SupplyDepot",
        "Refinery",
        "Barracks",
        "EngineeringBay",
        "Bunker",
        "MissileTurret",
        "Factory",
        "GhostAcademy",
        "Armory",
        "Starport",
        "FusionCore",
        "BarracksTechLab",
        "BarracksReactor",
        "FactoryTechLab",
        "FactoryReactor",
        "StarportTechLab",
        "StarportReactor",
        # Protoss
        "Nexus",
        "Pylon",
        "Assimilator",
        "Gateway",
        "WarpGate",
        "Forge",
        "CyberneticsCore",
        "PhotonCannon",
        "ShieldBattery",
        "RoboticsFacility",
        "Stargate",
        "TwilightCouncil",
        "TemplarArchive",
        "DarkShrine",
        "RoboticsBay",
        "FleetBeacon",
        # Zerg
        "Hatchery",
        "Lair",
        "Hive",
        "Extractor",
        "SpawningPool",
        "EvolutionChamber",
        "RoachWarren",
        "BanelingNest",
        "HydraliskDen",
        "LurkerDenMP",
        "Spire",
        "GreaterSpire",
        "NydusNetwork",
        "InfestationPit",
        "UltraliskCavern",
        "SpineCrawler",
        "SporeCrawler",
        "CreepTumor",
        "CreepTumorBurrowed",
    }
)


# ---------------------------------------------------------------------------
# Per-fn_idx Preconditions table.
# ---------------------------------------------------------------------------
# One Preconditions record per fn_idx in games.sc2.actions.FUNCTION_IDS.
# Universal actions (no_op, select_*, Move/Attack, Stop, Patrol,
# HoldPosition, Rally_*_minimap, ...) have empty preconditions and
# require no specific selection — they're handled by the surrounding
# logic in SC2Client (warmup, selection presence, ...).
#
# Building / Train / Morph / Effect actions get the appropriate
# building/upgrade prerequisites plus a SelectionReq.OF_TYPE constraint
# with the right selection_target.

_BUILDER_PROBES = WORKER_NAMES  # any worker can issue Build_*
_ATTACK_INFANTRY = frozenset({"Marine", "Marauder"})
_SIEGE_TANK = frozenset({"SiegeTank"})
_SIEGED_TANK = frozenset({"SiegeTankSieged"})


def _and(*names: str | frozenset[str]) -> tuple[frozenset[str], ...]:
    """Build a DNF tuple from a mix of single names and OR-sets."""
    return tuple(frozenset({n}) if isinstance(n, str) else n for n in names)


def _build(prereq_name: str) -> Preconditions:
    """Build_X_screen helper: worker selected + building prereqs met."""
    prereqs = BUILDING_PREREQS.get(prereq_name, frozenset())
    return Preconditions(
        required_buildings=(prereqs,) if prereqs else (),
        required_selection=SelectionReq.OF_TYPE,
        selection_target=_BUILDER_PROBES,
    )


def _addon(parents: frozenset[str]) -> Preconditions:
    """Build_TechLab / Build_Reactor helper: one of `parents` selected."""
    return Preconditions(
        required_selection=SelectionReq.OF_TYPE,
        selection_target=parents,
    )


def _train(
    unit_name: str,
    extra_buildings: tuple[frozenset[str], ...] = (),
    extra_upgrades: frozenset[str] = frozenset(),
) -> Preconditions:
    """Train_X_quick helper: producing-building selected + extras met."""
    producers = UNIT_PRODUCERS.get(unit_name, frozenset())
    return Preconditions(
        required_buildings=extra_buildings,
        required_upgrades=extra_upgrades,
        required_selection=SelectionReq.OF_TYPE,
        selection_target=producers,
    )


def _morph_inplace(parent_name: str, extra_buildings: tuple[frozenset[str], ...] = ()) -> Preconditions:
    """In-place state morph (the unit transforms without consuming itself).

    Used for SiegeMode/Unsiege where the *parent* and the result are
    different states of the same unit type, and ``UNIT_PRODUCERS`` may
    not be the cleanest model.  For all consume-and-create morphs (Lair,
    Hive, Baneling, Archon, …) use ``_train()`` so the source unit is
    pulled from :data:`UNIT_PRODUCERS` like every other unit.
    """
    return Preconditions(
        required_buildings=extra_buildings,
        required_selection=SelectionReq.OF_TYPE,
        selection_target=frozenset({parent_name}),
    )


def _effect_on(units: frozenset[str], upgrades: frozenset[str] = frozenset()) -> Preconditions:
    """Effect_X_quick helper: one of `units` selected, optional research."""
    return Preconditions(
        required_upgrades=upgrades,
        required_selection=SelectionReq.OF_TYPE,
        selection_target=units,
    )


def _any() -> Preconditions:
    """Move/Attack/Stop/Patrol/Hold helper: any unit selected."""
    return Preconditions(required_selection=SelectionReq.ANY_UNIT)


def _none() -> Preconditions:
    """no_op, select_*, Rally_*_minimap helper: no selection constraint."""
    return Preconditions(required_selection=SelectionReq.NONE)


# fn_idx → Preconditions. Keys correspond to games.sc2.actions.FUNCTION_IDS.
PRECONDITIONS: dict[int, Preconditions] = {
    # core
    0: _none(),  # no_op
    1: _none(),  # select_army
    2: _any(),  # Move_screen
    3: _any(),  # Attack_screen
    4: _none(),  # select_idle_worker
    5: _effect_on(_BUILDER_PROBES),  # Harvest_Gather_screen
    6: _none(),  # select_point
    7: _train("Marine"),  # Train_Marine_quick
    8: _build("Barracks"),
    9: _build("SupplyDepot"),
    10: _train("SCV"),  # Train_SCV_quick
    # movement / combat
    11: _any(),  # Move_minimap
    12: _any(),  # Patrol_screen
    13: _any(),  # Patrol_minimap
    14: _any(),  # HoldPosition_quick
    15: _any(),  # Stop_quick
    16: _any(),  # Attack_minimap
    # selection / logistics
    17: _none(),  # select_rect
    18: _effect_on(_BUILDER_PROBES),  # Harvest_Return_quick
    19: _any(),  # Rally_Units_screen
    20: _any(),  # Rally_Workers_screen
    21: _any(),  # Rally_Units_minimap
    22: _any(),  # Rally_Workers_minimap
    # Terran buildings
    23: _build("CommandCenter"),
    24: _build("Refinery"),
    25: _build("EngineeringBay"),
    26: _build("Factory"),
    27: _build("Armory"),
    28: _build("Bunker"),
    29: _build("MissileTurret"),
    30: _build("Starport"),
    31: _build("GhostAcademy"),
    32: _build("FusionCore"),
    33: _addon(frozenset({"Barracks", "Factory", "Starport"})),  # Build_TechLab_quick
    34: _addon(frozenset({"Barracks", "Factory", "Starport"})),  # Build_Reactor_quick
    # Terran training
    35: _train("Marauder", extra_buildings=_and("BarracksTechLab")),
    36: _train("Ghost", extra_buildings=_and("BarracksTechLab", "GhostAcademy")),
    37: _train("Hellion"),
    38: _train("SiegeTank", extra_buildings=_and("FactoryTechLab")),
    39: _train("Medivac"),
    40: _train("Viking"),
    41: _train("Raven", extra_buildings=_and("StarportTechLab")),
    42: _train("Banshee", extra_buildings=_and("StarportTechLab")),
    43: _train("Battlecruiser", extra_buildings=_and("StarportTechLab", "FusionCore")),
    44: _train("Cyclone", extra_buildings=_and("FactoryTechLab")),
    45: _train("Thor", extra_buildings=_and("FactoryTechLab", "Armory")),
    46: _train("Liberator"),
    # Terran unit abilities
    47: _effect_on(_ATTACK_INFANTRY, upgrades=frozenset({"Stimpack"})),  # Effect_Stim_quick
    # Morph_SiegeMode_quick — SiegeTank → SiegeTankSieged (FactoryTechLab gates it).
    48: _train("SiegeTankSieged", extra_buildings=_and("FactoryTechLab")),
    # Morph_Unsiege_quick — in-place morph of an already-sieged tank.  Kept on
    # _morph_inplace because the "produced" SiegeTank entry in UNIT_PRODUCERS
    # is the Factory-built one, and we want the selection target to be
    # SiegeTankSieged here.
    49: _morph_inplace("SiegeTankSieged"),
    # Protoss buildings
    50: _build("Nexus"),
    51: _build("Pylon"),
    52: _build("Gateway"),
    53: _build("Assimilator"),
    54: _build("CyberneticsCore"),
    55: _build("Forge"),
    56: _build("PhotonCannon"),
    57: _build("RoboticsFacility"),
    58: _build("Stargate"),
    59: _build("TwilightCouncil"),
    60: _build("TemplarArchive"),
    61: _build("DarkShrine"),
    62: _build("RoboticsBay"),
    63: _build("FleetBeacon"),
    64: _build("ShieldBattery"),
    # Protoss training / morphs
    65: _train("Probe"),
    66: _train("Zealot"),
    67: _train("Stalker", extra_buildings=_and("CyberneticsCore")),
    68: _train("Adept", extra_buildings=_and("CyberneticsCore")),
    69: _train("HighTemplar", extra_buildings=_and("TemplarArchive")),
    70: _train("DarkTemplar", extra_buildings=_and("DarkShrine")),
    71: _train("Sentry", extra_buildings=_and("CyberneticsCore")),
    72: _train("Phoenix"),
    73: _train("Carrier", extra_buildings=_and("FleetBeacon")),
    74: _train("VoidRay"),
    75: _train("Oracle"),
    76: _train("Colossus", extra_buildings=_and("RoboticsBay")),
    77: _train("Immortal"),
    78: _train("Tempest", extra_buildings=_and("FleetBeacon")),
    79: _train("Disruptor", extra_buildings=_and("RoboticsBay")),
    # Morph_Archon_quick — needs two HighTemplars OR two DarkTemplars.  The
    # selection-type filter only checks the right *type* is selected; the
    # quantity check (≥ 2 templars) is enforced by PySC2 at execution time.
    80: _train("Archon", extra_buildings=_and("TemplarArchive")),
    81: _train("Mothership", extra_buildings=_and("FleetBeacon")),
    # Zerg buildings
    82: _build("Hatchery"),
    83: _build("SpawningPool"),
    84: _build("Extractor"),
    85: _build("EvolutionChamber"),
    86: _build("HydraliskDen"),
    87: _build("BanelingNest"),
    88: _build("RoachWarren"),
    89: _build("Spire"),
    90: _build("InfestationPit"),
    91: _build("UltraliskCavern"),
    92: Preconditions(  # Build_CreepTumor_screen — Queen ability, not Drone build.
        required_selection=SelectionReq.OF_TYPE,
        selection_target=frozenset({"Queen", "CreepTumorBurrowed"}),
    ),
    93: _build("SpineCrawler"),
    94: _build("SporeCrawler"),
    95: _build("NydusNetwork"),
    96: _build("LurkerDenMP"),
    # Zerg training / morphs
    97: _train("Drone"),
    98: _train("Overlord"),
    99: _train("Zergling", extra_buildings=_and("SpawningPool")),
    100: _train("Baneling", extra_buildings=_and("BanelingNest")),  # Train_Baneling_quick
    101: _train("Roach", extra_buildings=_and("RoachWarren")),
    102: _train("Ravager", extra_buildings=_and("RoachWarren")),  # Train_Ravager_quick
    103: _train("Hydralisk", extra_buildings=_and("HydraliskDen")),
    104: _train("Infestor", extra_buildings=_and("InfestationPit")),
    105: _train("SwarmHost", extra_buildings=_and("InfestationPit")),
    106: _train("Mutalisk", extra_buildings=_and("Spire")),
    107: _train("Corruptor", extra_buildings=_and("Spire")),
    108: _train("BroodLord", extra_buildings=_and("GreaterSpire")),  # Train_BroodLord_quick
    109: _train("Viper", extra_buildings=_and("Hive")),
    110: _train("Ultralisk", extra_buildings=_and("UltraliskCavern")),
    111: _train("Lurker", extra_buildings=_and("LurkerDenMP")),  # Train_Lurker_quick
    112: _train("Queen", extra_buildings=_and("SpawningPool")),
    113: _train("Lair", extra_buildings=_and("SpawningPool")),  # Morph_Lair
    114: _train("Hive", extra_buildings=_and("InfestationPit")),  # Morph_Hive
    # Morph_Overseer — Overlord → Overseer; requires Lair (or Hive).
    115: _train("Overseer", extra_buildings=(frozenset({"Lair", "Hive"}),)),
    116: _train("GreaterSpire", extra_buildings=_and("Hive")),  # Morph_GreaterSpire
    117: _train("BroodLord", extra_buildings=_and("GreaterSpire")),  # Morph_BroodLord
}


def fn_idx_satisfied(
    fn_idx: int,
    owned_buildings: frozenset[str],
    completed_upgrades: frozenset[str],
    selected_unit_types: frozenset[str],
) -> bool:
    """Return True if all preconditions for *fn_idx* are met.

    Selection presence is checked here: if ``required_selection`` is
    ``OF_TYPE`` or ``ANY_UNIT`` and *selected_unit_types* is empty,
    the action is reported as unsatisfied.  The client's
    deferred-action queue is responsible for *issuing* the right
    selection — this function only reports whether the action could
    execute *given the current state*.

    Parameters
    ----------
    fn_idx :
        Internal fn_idx (key into ``games.sc2.actions.FUNCTION_IDS``).
    owned_buildings :
        Set of structure names currently owned by the agent.
    completed_upgrades :
        Set of upgrade/research names already completed.
    selected_unit_types :
        Set of unit-type names currently in the selection.  Empty
        frozenset means "nothing selected".  A multi-type selection
        (e.g. ``{"Marine", "Marauder"}`` after ``select_army``) satisfies
        ``ANY_UNIT``, and satisfies ``OF_TYPE`` whenever any selected
        type is in :attr:`Preconditions.selection_target` — PySC2 will
        apply the issued command only to compatible units in the
        selection.
    """
    pre = PRECONDITIONS.get(fn_idx)
    if pre is None:
        # Unknown fn_idx: be conservative and allow it; the PySC2 mask
        # will still gate it if it isn't actually available.
        return True

    # required_buildings is a tuple of OR-sets (DNF): each frozenset is
    # satisfied when any of its members exists in owned_buildings; the
    # tuple is satisfied when all of its OR-sets are satisfied.
    for or_set in pre.required_buildings:
        if not any(b in owned_buildings for b in or_set):
            return False

    for upgrade in pre.required_upgrades:
        if upgrade not in completed_upgrades:
            return False

    if pre.required_selection == SelectionReq.NONE:
        return True
    if not selected_unit_types:
        return False
    if pre.required_selection == SelectionReq.ANY_UNIT:
        return True
    # OF_TYPE — at least one selected type must be in the target set.
    return bool(selected_unit_types & pre.selection_target)


def building_prereqs_met(building_name: str, owned_buildings: frozenset[str]) -> bool:
    """Return True if *building_name*'s building prerequisites are met.

    Uses :data:`BUILDING_PREREQS` with OR-semantics inside each record
    (e.g. ``Spire`` requires ``{"Lair", "Hive"}`` — Lair OR Hive is
    enough).  Used by the client when deciding whether a Build_X
    action is masked.
    """
    prereqs = BUILDING_PREREQS.get(building_name)
    if not prereqs:
        return True
    return any(b in owned_buildings for b in prereqs)


# ---------------------------------------------------------------------------
# Resource costs (minerals, vespene) per fn_idx (issue #357).
# ---------------------------------------------------------------------------
# Only build / train / morph actions that consume resources are listed.
# Actions absent from this dict are assumed free (no cost gate applied).
# Source: StarCraft II LotV unit/building costs (liquipedia.net/starcraft2).

RESOURCE_COSTS: dict[int, tuple[int, int]] = {
    # ------------------------------------------------------------------ Terran
    # Training
    7: (50, 0),    # Train_Marine_quick
    10: (50, 0),   # Train_SCV_quick
    35: (100, 25), # Train_Marauder_quick
    36: (200, 100),# Train_Ghost_quick
    37: (100, 0),  # Train_Hellion_quick
    38: (150, 125),# Train_SiegeTank_quick
    39: (100, 100),# Train_Medivac_quick
    40: (150, 75), # Train_Viking_quick
    41: (100, 200),# Train_Raven_quick
    42: (150, 100),# Train_Banshee_quick
    43: (400, 300),# Train_Battlecruiser_quick
    44: (150, 100),# Train_Cyclone_quick
    45: (300, 200),# Train_Thor_quick
    46: (150, 150),# Train_Liberator_quick
    # Buildings
    8: (150, 0),   # Build_Barracks_screen
    9: (100, 0),   # Build_SupplyDepot_screen
    23: (400, 0),  # Build_CommandCenter_screen
    24: (75, 0),   # Build_Refinery_screen
    25: (150, 0),  # Build_EngineeringBay_screen
    26: (150, 100),# Build_Factory_screen
    27: (150, 100),# Build_Armory_screen
    28: (100, 0),  # Build_Bunker_screen
    29: (100, 0),  # Build_MissileTurret_screen
    30: (150, 100),# Build_Starport_screen
    31: (150, 50), # Build_GhostAcademy_screen
    32: (150, 150),# Build_FusionCore_screen
    33: (50, 25),  # Build_TechLab_quick
    34: (50, 50),  # Build_Reactor_quick
    # ---------------------------------------------------------------- Protoss
    # Buildings
    50: (400, 0),  # Build_Nexus_screen
    51: (100, 0),  # Build_Pylon_screen
    52: (150, 0),  # Build_Gateway_screen
    53: (75, 0),   # Build_Assimilator_screen
    54: (150, 0),  # Build_CyberneticsCore_screen
    55: (150, 0),  # Build_Forge_screen
    56: (150, 0),  # Build_PhotonCannon_screen
    57: (200, 100),# Build_RoboticsFacility_screen
    58: (150, 150),# Build_Stargate_screen
    59: (150, 100),# Build_TwilightCouncil_screen
    60: (150, 200),# Build_TemplarArchive_screen
    61: (150, 150),# Build_DarkShrine_screen
    62: (150, 150),# Build_RoboticsBay_screen
    63: (300, 200),# Build_FleetBeacon_screen
    64: (100, 0),  # Build_ShieldBattery_screen
    # Training
    65: (50, 0),   # Train_Probe_quick
    66: (100, 0),  # Train_Zealot_quick
    67: (125, 50), # Train_Stalker_quick
    68: (100, 25), # Train_Adept_quick
    69: (50, 150), # Train_HighTemplar_quick
    70: (125, 125),# Train_DarkTemplar_quick
    71: (50, 100), # Train_Sentry_quick
    72: (150, 100),# Train_Phoenix_quick
    73: (350, 250),# Train_Carrier_quick
    74: (250, 150),# Train_VoidRay_quick
    75: (150, 150),# Train_Oracle_quick
    76: (300, 200),# Train_Colossus_quick
    77: (275, 100),# Train_Immortal_quick
    78: (250, 175),# Train_Tempest_quick
    79: (150, 150),# Train_Disruptor_quick
    81: (300, 300),# Train_Mothership_quick
    # ------------------------------------------------------------------ Zerg
    # Buildings
    82: (300, 0),  # Build_Hatchery_screen
    83: (200, 0),  # Build_SpawningPool_screen
    84: (25, 0),   # Build_Extractor_screen
    85: (75, 0),   # Build_EvolutionChamber_screen
    86: (100, 100),# Build_HydraliskDen_screen
    87: (100, 50), # Build_BanelingNest_screen
    88: (150, 0),  # Build_RoachWarren_screen
    89: (200, 200),# Build_Spire_screen
    90: (100, 100),# Build_InfestationPit_screen
    91: (150, 200),# Build_UltraliskCavern_screen
    93: (100, 0),  # Build_SpineCrawler_screen
    94: (75, 0),   # Build_SporeCrawler_screen
    95: (150, 200),# Build_NydusNetwork_screen
    96: (100, 150),# Build_LurkerDen_screen
    # Training / morphs
    97: (50, 0),   # Train_Drone_quick
    98: (100, 0),  # Train_Overlord_quick
    99: (50, 0),   # Train_Zergling_quick
    100: (50, 25), # Train_Baneling_quick (morph from Zergling)
    101: (75, 25), # Train_Roach_quick
    102: (25, 75), # Train_Ravager_quick (morph from Roach)
    103: (100, 50),# Train_Hydralisk_quick
    104: (100, 150),# Train_Infestor_quick
    105: (200, 100),# Train_SwarmHost_quick
    106: (100, 100),# Train_Mutalisk_quick
    107: (150, 100),# Train_Corruptor_quick
    108: (150, 150),# Train_BroodLord_quick (morph from Corruptor)
    109: (100, 200),# Train_Viper_quick
    110: (300, 200),# Train_Ultralisk_quick
    111: (50, 100), # Train_Lurker_quick (morph from Hydralisk)
    112: (150, 0),  # Train_Queen_quick
    113: (150, 100),# Morph_Lair
    114: (200, 150),# Morph_Hive
    115: (50, 100), # Morph_Overseer (from Overlord)
    116: (100, 150),# Morph_GreaterSpire
    117: (150, 150),# Morph_BroodLord (from Corruptor)
}


def fn_idx_affordable(fn_idx: int, minerals: float, vespene: float) -> bool:
    """Return True if the player can afford fn_idx's resource cost.

    Actions not listed in :data:`RESOURCE_COSTS` are considered free and
    always return True.  This is called once per candidate fn_idx in
    ``SC2Client._compute_available_fn_ids`` to gate build/train actions
    behind the current mineral and vespene balance (issue #357).
    """
    cost = RESOURCE_COSTS.get(fn_idx)
    if cost is None:
        return True
    min_cost, gas_cost = cost
    return minerals >= min_cost and vespene >= gas_cost
