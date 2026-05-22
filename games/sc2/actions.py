"""StarCraft 2 action definitions.

PySC2's full action space is parameterised: ``function_id`` + per-arg target
coordinates.  That doesn't fit the existing ``Discrete(N)`` policies cleanly,
so this MVP exposes a flat discrete subset for tabular / discrete-output
policies.  Continuous target coordinates are emitted directly by the
multi-head linear / CMA-ES / LSTM policies.

Action representation
---------------------
Each row in ``DISCRETE_ACTIONS`` is a 4-vector ``[fn_idx, x, y, queue]`` where:

  ``fn_idx``   — integer index into ``FUNCTION_IDS`` below.
  ``x, y``     — normalised screen target coords in ``[0, 1]``.  Ignored for
                 functions that don't take a screen-point arg.
  ``queue``    — 0 or 1, whether to queue the order.

The framework's discrete-action policies still see fixed-shape rows; the
client (``games.sc2.client``) is responsible for translating each row into
a real ``actions.FunctionCall`` at execution time.

Layout (issue #276)
--------------------
Function IDs are processed in ascending order.  Spatial fn_ids (names ending
in ``_screen`` or ``_minimap``) receive a full N×N grid of target rows; all
other fn_ids (quick-cast, select_army, no_op, …) receive a single centre row.
This gives a uniform **[command × location]** structure: all spatial commands
have identical positional coverage.

Function-ID encoding
--------------------
Names in ``FUNCTION_IDS`` must be valid ``pysc2.lib.actions.FUNCTIONS``
attribute names.  Unknown names resolve to ``no_op`` gracefully via
``getattr(..., no_op)`` — a wrong name is a silent no-op, never a crash.

Action-call encoding by fn_name pattern (see ``action_to_function_call``)
--------------------------------------------------------------------------
- ``no_op``                  → ``FunctionCall(fn_id, [])``
- ``select_army`` / ``select_idle_worker``
                             → ``FunctionCall(fn_id, [[0]])``
- ``select_point``            → ``FunctionCall(fn_id, [[0], [sx, sy]])``
- ``select_rect``             → ``FunctionCall(fn_id, [[0], [sx,sy],[sx,sy]])``
  (degenerate rect = single-point click)
- names ending in ``_quick`` (all train/morph/ability quick-casts)
                             → ``FunctionCall(fn_id, [[queue]])``
- everything else (``_screen`` / ``_minimap`` spatial)
                             → ``FunctionCall(fn_id, [[queue], [sx, sy]])``

Race gating
-----------
``fn_ids_for_race(race)`` returns the subset of ``FUNCTION_IDS`` keys
applicable to *race* (``"terran"``, ``"protoss"``, ``"zerg"``,
``"random"``).  Policies that know their race can permanently mask the
irrelevant fn_idx scores to ``-inf`` so the agent never wastes capacity
learning that a Terran cannot build a Hatchery.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Function-id table
# ---------------------------------------------------------------------------
# Indices are stable; the env client resolves them to PySC2 function ids at
# call time so we don't need pysc2 imported at framework level.
#
# Names must match pysc2.lib.actions.FUNCTIONS attributes exactly.
# An incorrect name degrades silently to no_op (never a crash).

FUNCTION_IDS = {
    # --- core ---
    0: "no_op",
    1: "select_army",
    2: "Move_screen",
    3: "Attack_screen",
    4: "select_idle_worker",
    5: "Harvest_Gather_screen",
    # --- issue #276 (initial 5): BuildMarines / Simple64 basics ---
    6: "select_point",
    7: "Train_Marine_quick",
    8: "Build_Barracks_screen",
    9: "Build_SupplyDepot_screen",
    10: "Train_SCV_quick",
    # --- issue #276 (expanded): full cross-race action coverage ---
    # movement / combat
    11: "Move_minimap",
    12: "Patrol_screen",
    13: "Patrol_minimap",
    14: "HoldPosition_quick",
    15: "Stop_quick",
    16: "Attack_minimap",
    # selection / logistics
    17: "select_rect",
    18: "Harvest_Return_quick",
    19: "Rally_Units_screen",
    20: "Rally_Workers_screen",
    21: "Rally_Units_minimap",
    22: "Rally_Workers_minimap",
    # === Terran buildings ===
    23: "Build_CommandCenter_screen",
    24: "Build_Refinery_screen",
    25: "Build_EngineeringBay_screen",
    26: "Build_Factory_screen",
    27: "Build_Armory_screen",
    28: "Build_Bunker_screen",
    29: "Build_MissileTurret_screen",
    30: "Build_Starport_screen",
    31: "Build_GhostAcademy_screen",
    32: "Build_FusionCore_screen",
    33: "Build_TechLab_quick",
    34: "Build_Reactor_quick",
    # === Terran training ===
    35: "Train_Marauder_quick",
    36: "Train_Ghost_quick",
    37: "Train_Hellion_quick",
    38: "Train_SiegeTank_quick",
    39: "Train_Medivac_quick",
    40: "Train_Viking_quick",
    41: "Train_Raven_quick",
    42: "Train_Banshee_quick",
    43: "Train_Battlecruiser_quick",
    44: "Train_Cyclone_quick",
    45: "Train_Thor_quick",
    46: "Train_Liberator_quick",
    # === Terran unit abilities ===
    47: "Effect_Stim_quick",
    48: "Morph_SiegeMode_quick",
    49: "Morph_Unsiege_quick",
    # === Protoss buildings ===
    50: "Build_Nexus_screen",
    51: "Build_Pylon_screen",
    52: "Build_Gateway_screen",
    53: "Build_Assimilator_screen",
    54: "Build_CyberneticsCore_screen",
    55: "Build_Forge_screen",
    56: "Build_PhotonCannon_screen",
    57: "Build_RoboticsFacility_screen",
    58: "Build_Stargate_screen",
    59: "Build_TwilightCouncil_screen",
    60: "Build_TemplarArchive_screen",
    61: "Build_DarkShrine_screen",
    62: "Build_RoboticsBay_screen",
    63: "Build_FleetBeacon_screen",
    64: "Build_ShieldBattery_screen",
    # === Protoss training / morphs ===
    65: "Train_Probe_quick",
    66: "Train_Zealot_quick",
    67: "Train_Stalker_quick",
    68: "Train_Adept_quick",
    69: "Train_HighTemplar_quick",
    70: "Train_DarkTemplar_quick",
    71: "Train_Sentry_quick",
    72: "Train_Phoenix_quick",
    73: "Train_Carrier_quick",
    74: "Train_VoidRay_quick",
    75: "Train_Oracle_quick",
    76: "Train_Colossus_quick",
    77: "Train_Immortal_quick",
    78: "Train_Tempest_quick",
    79: "Train_Disruptor_quick",
    80: "Morph_Archon_quick",
    81: "Train_Mothership_quick",
    # === Zerg buildings ===
    82: "Build_Hatchery_screen",
    83: "Build_SpawningPool_screen",
    84: "Build_Extractor_screen",
    85: "Build_EvolutionChamber_screen",
    86: "Build_HydraliskDen_screen",
    87: "Build_BanelingNest_screen",
    88: "Build_RoachWarren_screen",
    89: "Build_Spire_screen",
    90: "Build_InfestationPit_screen",
    91: "Build_UltraliskCavern_screen",
    92: "Build_CreepTumor_screen",
    93: "Build_SpineCrawler_screen",
    94: "Build_SporeCrawler_screen",
    95: "Build_NydusNetwork_screen",
    96: "Build_LurkerDen_screen",
    # === Zerg training / morphs ===
    97: "Train_Drone_quick",
    98: "Train_Overlord_quick",
    99: "Train_Zergling_quick",
    100: "Train_Baneling_quick",
    101: "Train_Roach_quick",
    102: "Train_Ravager_quick",
    103: "Train_Hydralisk_quick",
    104: "Train_Infestor_quick",
    105: "Train_SwarmHost_quick",
    106: "Train_Mutalisk_quick",
    107: "Train_Corruptor_quick",
    108: "Train_BroodLord_quick",
    109: "Train_Viper_quick",
    110: "Train_Ultralisk_quick",
    111: "Train_Lurker_quick",
    112: "Train_Queen_quick",
    113: "Morph_Lair_quick",
    114: "Morph_Hive_quick",
    115: "Morph_Overseer_quick",
    116: "Morph_GreaterSpire_quick",
    117: "Morph_BroodLord_quick",
}

# ---------------------------------------------------------------------------
# Spatial fn_id set — actions that take a screen or minimap point argument.
# These get a full N×N grid in DISCRETE_ACTIONS; all others get one row.
# ---------------------------------------------------------------------------

SPATIAL_FN_IDS: frozenset[int] = frozenset(
    fn_idx
    for fn_idx, name in FUNCTION_IDS.items()
    if name.endswith("_screen") or name.endswith("_minimap")
    or name in ("select_point", "select_rect")
)

# ---------------------------------------------------------------------------
# Race gating
# ---------------------------------------------------------------------------
# Maps race name → set of fn_idx values applicable to that race.
# Universal actions (movement, attack, harvesting, selection) are in every set.
# "random" includes every action (agent may be any race).

_UNIVERSAL_FN_IDS: frozenset[int] = frozenset({
    0,   # no_op
    1,   # select_army
    2,   # Move_screen
    3,   # Attack_screen
    4,   # select_idle_worker
    5,   # Harvest_Gather_screen
    6,   # select_point
    11,  # Move_minimap
    12,  # Patrol_screen
    13,  # Patrol_minimap
    14,  # HoldPosition_quick
    15,  # Stop_quick
    16,  # Attack_minimap
    17,  # select_rect
    18,  # Harvest_Return_quick
    19,  # Rally_Units_screen
    20,  # Rally_Workers_screen
    21,  # Rally_Units_minimap
    22,  # Rally_Workers_minimap
})

_TERRAN_FN_IDS: frozenset[int] = frozenset({
    7, 8, 9, 10,           # Train_Marine, Build_Barracks, Build_SupplyDepot, Train_SCV
    23, 24, 25, 26, 27,    # CommandCenter, Refinery, EngineeringBay, Factory, Armory
    28, 29, 30, 31, 32,    # Bunker, MissileTurret, Starport, GhostAcademy, FusionCore
    33, 34,                # TechLab_quick, Reactor_quick
    35, 36, 37, 38, 39,    # Marauder, Ghost, Hellion, SiegeTank, Medivac
    40, 41, 42, 43, 44,    # Viking, Raven, Banshee, Battlecruiser, Cyclone
    45, 46,                # Thor, Liberator
    47, 48, 49,            # Stim, SiegeMode, Unsiege
})

_PROTOSS_FN_IDS: frozenset[int] = frozenset({
    50, 51, 52, 53, 54,    # Nexus, Pylon, Gateway, Assimilator, CyberneticsCore
    55, 56, 57, 58, 59,    # Forge, PhotonCannon, RoboticsFacility, Stargate, TwilightCouncil
    60, 61, 62, 63, 64,    # TemplarArchive, DarkShrine, RoboticsBay, FleetBeacon, ShieldBattery
    65, 66, 67, 68, 69,    # Probe, Zealot, Stalker, Adept, HighTemplar
    70, 71, 72, 73, 74,    # DarkTemplar, Sentry, Phoenix, Carrier, VoidRay
    75, 76, 77, 78, 79,    # Oracle, Colossus, Immortal, Tempest, Disruptor
    80, 81,                # Morph_Archon, Mothership
})

_ZERG_FN_IDS: frozenset[int] = frozenset({
    82, 83, 84, 85, 86,    # Hatchery, SpawningPool, Extractor, EvolutionChamber, HydraliskDen
    87, 88, 89, 90, 91,    # BanelingNest, RoachWarren, Spire, InfestationPit, UltraliskCavern
    92, 93, 94, 95, 96,    # CreepTumor, SpineCrawler, SporeCrawler, NydusNetwork, LurkerDen
    97, 98, 99, 100, 101,  # Drone, Overlord, Zergling, Baneling, Roach
    102, 103, 104, 105,    # Ravager, Hydralisk, Infestor, SwarmHost
    106, 107, 108, 109,    # Mutalisk, Corruptor, BroodLord, Viper
    110, 111, 112,         # Ultralisk, Lurker, Queen
    113, 114, 115, 116, 117,  # Morph_Lair, Hive, Overseer, GreaterSpire, BroodLord
})

RACE_FUNCTION_IDS: dict[str, frozenset[int]] = {
    "terran":  _UNIVERSAL_FN_IDS | _TERRAN_FN_IDS,
    "protoss": _UNIVERSAL_FN_IDS | _PROTOSS_FN_IDS,
    "zerg":    _UNIVERSAL_FN_IDS | _ZERG_FN_IDS,
    "random":  frozenset(FUNCTION_IDS.keys()),
}


def fn_ids_for_race(race: str) -> frozenset[int]:
    """Return the fn_idx values applicable to *race*.

    Falls back to all fn_ids if *race* is not one of
    ``"terran"``, ``"protoss"``, ``"zerg"``, or ``"random"``.
    """
    return RACE_FUNCTION_IDS.get(race.lower(), frozenset(FUNCTION_IDS.keys()))


# ---------------------------------------------------------------------------
# Build-order prerequisites
# ---------------------------------------------------------------------------
# Maps fn_idx → frozenset of unit/structure names that must exist (count ≥ 1)
# before the action is allowed.  Only the *direct* prerequisite is listed
# (e.g. Build_Barracks just needs SupplyDepot; what SupplyDepot itself needs
# is captured separately by its own entry).  Names follow pysc2.lib.units
# enum member naming exactly so the client can look them up at runtime.
#
# Actions without an entry have no additional structural prerequisites beyond
# what PySC2's available_actions already enforces (e.g. having an SCV
# selected for building, or enough minerals).

BUILD_PREREQUISITES: dict[int, frozenset[str]] = {
    # --- Terran buildings ---
    8:  frozenset({"SupplyDepot"}),         # Build_Barracks_screen
    25: frozenset({"Barracks"}),            # Build_EngineeringBay_screen
    26: frozenset({"Barracks"}),            # Build_Factory_screen
    27: frozenset({"Factory"}),             # Build_Armory_screen
    28: frozenset({"Barracks"}),            # Build_Bunker_screen
    29: frozenset({"EngineeringBay"}),      # Build_MissileTurret_screen
    30: frozenset({"Factory"}),             # Build_Starport_screen
    31: frozenset({"Barracks"}),            # Build_GhostAcademy_screen
    32: frozenset({"Starport"}),            # Build_FusionCore_screen
    # --- Terran training ---
    7:  frozenset({"Barracks"}),            # Train_Marine_quick
    35: frozenset({"Barracks"}),            # Train_Marauder_quick
    36: frozenset({"GhostAcademy"}),        # Train_Ghost_quick
    37: frozenset({"Factory"}),             # Train_Hellion_quick
    38: frozenset({"Factory"}),             # Train_SiegeTank_quick
    39: frozenset({"Starport"}),            # Train_Medivac_quick
    40: frozenset({"Starport"}),            # Train_Viking_quick
    41: frozenset({"Starport"}),            # Train_Raven_quick
    42: frozenset({"Starport"}),            # Train_Banshee_quick
    43: frozenset({"FusionCore"}),          # Train_Battlecruiser_quick
    44: frozenset({"Factory"}),             # Train_Cyclone_quick
    45: frozenset({"Armory", "Factory"}),   # Train_Thor_quick
    46: frozenset({"Starport"}),            # Train_Liberator_quick
    # --- Protoss buildings ---
    52: frozenset({"Pylon"}),               # Build_Gateway_screen
    54: frozenset({"Gateway"}),             # Build_CyberneticsCore_screen
    55: frozenset({"Pylon"}),               # Build_Forge_screen
    56: frozenset({"Forge"}),               # Build_PhotonCannon_screen
    57: frozenset({"CyberneticsCore"}),     # Build_RoboticsFacility_screen
    58: frozenset({"CyberneticsCore"}),     # Build_Stargate_screen
    59: frozenset({"CyberneticsCore"}),     # Build_TwilightCouncil_screen
    60: frozenset({"TwilightCouncil"}),     # Build_TemplarArchive_screen
    61: frozenset({"TwilightCouncil"}),     # Build_DarkShrine_screen
    62: frozenset({"RoboticsFacility"}),    # Build_RoboticsBay_screen
    63: frozenset({"Stargate"}),            # Build_FleetBeacon_screen
    64: frozenset({"Gateway"}),             # Build_ShieldBattery_screen
    # --- Protoss training ---
    66: frozenset({"Gateway"}),             # Train_Zealot_quick
    67: frozenset({"CyberneticsCore"}),     # Train_Stalker_quick
    68: frozenset({"CyberneticsCore"}),     # Train_Adept_quick
    69: frozenset({"TemplarArchive"}),      # Train_HighTemplar_quick
    70: frozenset({"DarkShrine"}),          # Train_DarkTemplar_quick
    71: frozenset({"CyberneticsCore"}),     # Train_Sentry_quick
    72: frozenset({"Stargate"}),            # Train_Phoenix_quick
    73: frozenset({"FleetBeacon"}),         # Train_Carrier_quick
    74: frozenset({"Stargate"}),            # Train_VoidRay_quick
    75: frozenset({"Stargate"}),            # Train_Oracle_quick
    76: frozenset({"RoboticsFacility"}),    # Train_Colossus_quick
    77: frozenset({"RoboticsFacility"}),    # Train_Immortal_quick
    78: frozenset({"FleetBeacon"}),         # Train_Tempest_quick
    79: frozenset({"RoboticsBay"}),         # Train_Disruptor_quick
    81: frozenset({"FleetBeacon"}),         # Train_Mothership_quick
    # --- Zerg buildings ---
    86: frozenset({"SpawningPool"}),        # Build_HydraliskDen_screen
    87: frozenset({"SpawningPool"}),        # Build_BanelingNest_screen
    88: frozenset({"SpawningPool"}),        # Build_RoachWarren_screen
    89: frozenset({"Lair"}),               # Build_Spire_screen
    90: frozenset({"Lair"}),               # Build_InfestationPit_screen
    91: frozenset({"Hive"}),               # Build_UltraliskCavern_screen
    93: frozenset({"SpawningPool"}),        # Build_SpineCrawler_screen
    94: frozenset({"SpawningPool"}),        # Build_SporeCrawler_screen
    95: frozenset({"Lair"}),               # Build_NydusNetwork_screen
    96: frozenset({"HydraliskDen"}),        # Build_LurkerDen_screen
    # --- Zerg training ---
    99:  frozenset({"SpawningPool"}),       # Train_Zergling_quick
    100: frozenset({"BanelingNest"}),       # Train_Baneling_quick
    101: frozenset({"RoachWarren"}),        # Train_Roach_quick
    102: frozenset({"RoachWarren"}),        # Train_Ravager_quick
    103: frozenset({"HydraliskDen"}),       # Train_Hydralisk_quick
    104: frozenset({"InfestationPit"}),     # Train_Infestor_quick
    105: frozenset({"InfestationPit"}),     # Train_SwarmHost_quick
    106: frozenset({"Spire"}),             # Train_Mutalisk_quick
    107: frozenset({"Spire"}),             # Train_Corruptor_quick
    108: frozenset({"GreaterSpire"}),       # Train_BroodLord_quick
    109: frozenset({"InfestationPit"}),     # Train_Viper_quick
    110: frozenset({"UltraliskCavern"}),    # Train_Ultralisk_quick
    111: frozenset({"LurkerDenMP"}),        # Train_Lurker_quick (pysc2 names Lurker Den as LurkerDenMP)
    112: frozenset({"SpawningPool"}),       # Train_Queen_quick
    # --- Zerg morphs ---
    113: frozenset({"SpawningPool"}),       # Morph_Lair_quick
    114: frozenset({"Lair"}),              # Morph_Hive_quick
    115: frozenset({"Lair"}),              # Morph_Overseer_quick
    116: frozenset({"Spire"}),             # Morph_GreaterSpire_quick
    117: frozenset({"GreaterSpire"}),       # Morph_BroodLord_quick
}

#: All unique unit/structure names that appear in :data:`BUILD_PREREQUISITES`.
#: Used by the client to build a targeted lookup table.
PREREQ_STRUCTURE_NAMES: frozenset[str] = frozenset(
    name
    for names in BUILD_PREREQUISITES.values()
    for name in names
)


def fn_ids_blocked_by_prerequisites(unit_counts: dict[str, int]) -> frozenset[int]:
    """Return fn_idx values whose build-order prerequisites are *not* met.

    Parameters
    ----------
    unit_counts :
        Mapping of structure/unit name → count of that type observed during
        the current episode.  A count ≥ 1 means the prerequisite is satisfied.

    Returns
    -------
    frozenset[int]
        Set of fn_idx values that should be masked because at least one of
        their required structures does not yet exist (count == 0 or absent).

    Notes
    -----
    Only checks :data:`BUILD_PREREQUISITES`; actions without an entry are never
    blocked by this function.
    """
    blocked: set[int] = set()
    for fn_idx, required_names in BUILD_PREREQUISITES.items():
        for name in required_names:
            if unit_counts.get(name, 0) < 1:
                blocked.add(fn_idx)
                break
    return frozenset(blocked)


# ---------------------------------------------------------------------------
# Discrete action grid for minigame / tabular policies
# ---------------------------------------------------------------------------
# fn_ids are processed in ascending order.
# Spatial fn_ids (names ending in _screen or _minimap) get an N×N grid.
# Non-spatial fn_ids get a single centre row.

SCREEN_GRID_RESOLUTION: int = 8


def _grid_centres(resolution: int) -> list[tuple[float, float]]:
    """Return cell centres for an N×N grid over the unit square."""
    step = 1.0 / resolution
    centres = [(j * step + step / 2.0, i * step + step / 2.0)
               for i in range(resolution)
               for j in range(resolution)]
    return centres


def _build_discrete_actions(resolution: int) -> np.ndarray:
    """Construct the [fn_idx, x, y, queue] table.

    All spatial fn_ids get an N×N grid of target rows (uniform [command ×
    location] coverage).  Non-spatial fn_ids get a single centre row.
    fn_ids are placed in ascending order.
    """
    centres = _grid_centres(resolution)
    rows: list[list[float]] = []
    for fn_idx in sorted(FUNCTION_IDS.keys()):
        name = FUNCTION_IDS[fn_idx]
        if fn_idx in SPATIAL_FN_IDS:
            for x, y in centres:
                rows.append([fn_idx, x, y, 0])
        else:
            rows.append([fn_idx, 0.5, 0.5, 0])
    return np.array(rows, dtype=np.float32)


DISCRETE_ACTIONS: np.ndarray = _build_discrete_actions(SCREEN_GRID_RESOLUTION)


# ---------------------------------------------------------------------------
# Probe actions — fixed action vectors for cold-start evaluation
# ---------------------------------------------------------------------------
# Each entry is (action_array, description_string).  Probes establish a
# reward floor before random-restart hill-climbing kicks in.
# Issue #127: keep no_op as the first probe so the cold-start search learns
# whether idling is competitive on the chosen map.

PROBE_ACTIONS: list[tuple[np.ndarray, str]] = [
    (np.array([0, 0.5, 0.5, 0], dtype=np.float32), "no_op"),
    (np.array([1, 0.5, 0.5, 0], dtype=np.float32), "select_army"),
    (np.array([2, 0.5, 0.5, 0], dtype=np.float32), "move_centre"),
    (np.array([2, 0.2, 0.2, 0], dtype=np.float32), "move_top_left"),
    (np.array([2, 0.8, 0.8, 0], dtype=np.float32), "move_bottom_right"),
]


# ---------------------------------------------------------------------------
# Warmup action — forced for the first N steps of each episode
# ---------------------------------------------------------------------------
# select_army is a near-universal precondition; running it on step 0 means
# subsequent moves can target individual units without first re-selecting.

WARMUP_ACTION = np.array([1, 0.5, 0.5, 0], dtype=np.float32)


def discrete_action_to_fn_id(cell_idx: int) -> int:
    """Return the FUNCTION_IDS key for grid cell *cell_idx*."""
    return int(DISCRETE_ACTIONS[cell_idx, 0])


def pysc2_ids_to_internal_fn_idx(pysc2_available_ids: set[int]) -> set[int]:
    """Convert a set of raw PySC2 function IDs to internal fn_idx values.

    PySC2's ``ob["available_actions"]`` contains PySC2 function IDs (e.g. 331
    for Move_screen).  Our mask logic uses the repo-internal fn_idx keys that
    index into FUNCTION_IDS.  This converts between the two so that
    ``build_available_actions_mask()`` receives the correct values.

    Imports PySC2 lazily so callers without it installed (framework code,
    unit tests) can import this module freely.
    """
    try:
        from pysc2.lib import actions as _pysc2_actions  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        return set(FUNCTION_IDS.keys())
    result: set[int] = set()
    for fn_idx, name in FUNCTION_IDS.items():
        fn = getattr(_pysc2_actions.FUNCTIONS, name, None)
        if fn is not None and int(fn.id) in pysc2_available_ids:
            result.add(fn_idx)
    return result


def build_available_actions_mask(
    available_fn_ids: set[int], n_cells: int = len(DISCRETE_ACTIONS)
) -> np.ndarray:
    """Boolean mask of shape (n_cells,) — True where the action is legal."""
    return np.array(
        [discrete_action_to_fn_id(i) in available_fn_ids for i in range(n_cells)],
        dtype=bool,
    )


def action_to_function_call(
    action: np.ndarray, screen_size: int, minimap_size: int | None = None
):
    """Translate a 4-vector action row into a PySC2 ``FunctionCall``.

    Parameters
    ----------
    action :
        4-vector ``[fn_idx, x, y, queue]`` produced by a policy.
    screen_size :
        Size of the screen feature layer (e.g. 64).  Used to denormalise
        the coordinate args for ``*_screen`` actions.
    minimap_size :
        Size of the minimap feature layer.  Used to denormalise ``*_minimap``
        targets.  Defaults to ``screen_size`` for backwards compatibility.

    Returns
    -------
    pysc2.lib.actions.FunctionCall

    Notes
    -----
    Imports PySC2 lazily so that callers without PySC2 installed (unit
    tests, framework code) can import this module freely.

    Encoding rules (see module docstring for full table):
    - ``no_op``                → ``FunctionCall(fn_id, [])``
    - ``select_army`` / ``select_idle_worker``
                               → ``FunctionCall(fn_id, [[0]])``
    - ``select_point``          → ``FunctionCall(fn_id, [[0], [sx, sy]])``
    - ``select_rect``           → ``FunctionCall(fn_id, [[0], [sx,sy],[sx,sy]])``
    - names ending in ``_quick`` (train/morph/ability)
                               → ``FunctionCall(fn_id, [[queue]])``
    - all other spatial actions (``_screen`` / ``_minimap``)
                               → ``FunctionCall(fn_id, [[queue], [sx, sy]])``
    """
    from pysc2.lib import actions  # type: ignore[import-untyped]

    fn_idx = int(action[0])
    x_norm = float(np.clip(action[1], 0.0, 1.0))
    y_norm = float(np.clip(action[2], 0.0, 1.0))
    queue = int(np.clip(round(float(action[3])), 0, 1))
    name = FUNCTION_IDS.get(fn_idx, "no_op")
    functions = actions.FUNCTIONS
    if hasattr(functions, "__getitem__"):
        try:
            fn = functions[name]
        except KeyError:
            fn = functions.no_op
            name = "no_op"
    else:
        fn = getattr(functions, name, None)
        if fn is None:
            fn = functions.no_op
            name = "no_op"
    fn_id = int(fn.id)
    minimap = screen_size if minimap_size is None else minimap_size
    target_size = minimap if name.endswith("_minimap") else screen_size
    sx = int(x_norm * (target_size - 1))
    sy = int(y_norm * (target_size - 1))

    if name == "no_op":
        return actions.FunctionCall(fn_id, [])
    if name in ("select_army", "select_idle_worker"):
        return actions.FunctionCall(fn_id, [[0]])
    if name == "select_point":
        # select_point_act=0: single-unit click (not add/toggle).
        return actions.FunctionCall(fn_id, [[0], [sx, sy]])
    if name == "select_rect":
        # Degenerate rect (start == end) acts as a single-point click.
        return actions.FunctionCall(fn_id, [[0], [sx, sy], [sx, sy]])
    if name.endswith("_quick"):
        # All quick-cast actions (train, morph, abilities) take only a
        # queued flag — no spatial target.
        return actions.FunctionCall(fn_id, [[queue]])
    # Default: spatial actions with [queued, target_screen/minimap].
    # Covers Move_screen/minimap, Attack_screen/minimap, Patrol_screen/minimap,
    # Harvest_Gather_screen, all Build_*_screen, Rally_*_screen/minimap, etc.
    return actions.FunctionCall(fn_id, [[queue], [sx, sy]])
