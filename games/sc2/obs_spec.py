"""StarCraft 2 observation space definition.

The PySC2 API exposes a rich observation: structured ``player`` totals, two
spatial feature-layer stacks (minimap and screen), per-unit feature lists, and
score breakdowns.  The framework policies (linear, MLP, evolutionary) consume
flat fixed-size vectors, so the SC2 integration projects PySC2's observation
into one of three preset specs (issue #126):

``SC2_MINIGAME_OBS_SPEC``  (13 dims)
    Compact spec covering player totals plus a few spatial summary statistics.
    Designed for the 7 standard PySC2 minigames.  Unchanged from before #126
    so existing minigame champions keep loading.

``SC2_LADDER_OBS_SPEC``   (~40 dims)
    Extension adding economy breakdowns (collected / spent minerals + vespene,
    killed / total value, idle worker / production time), army-vs-worker supply
    split, screen unit-density / mean-HP summaries, top-K enemy features and
    minimap-camera position.  Default for 1v1 ladder play.

``SC2_RICH_OBS_SPEC``     (80 dims)
    Full superset for research experiments: everything in the ladder spec plus
    8 per-unit-type counts, per-quadrant screen counts of self / enemy,
    top-3 closest enemies (rel_x, rel_y, hp_ratio), available-actions binary
    mask and one-hot of last function id.

Map → preset selection is done by ``get_spec(map_name, preset=None)``.  Pass
an explicit ``preset`` (``"minigame"`` / ``"ladder"`` / ``"rich"``) to override
the default mapping.
"""

from __future__ import annotations

import numpy as np

from framework.obs_spec import ObsDim, ObsSpec
from games.sc2.actions import FUNCTION_IDS


# ---------------------------------------------------------------------------
# Per-block dim definitions
# ---------------------------------------------------------------------------
# Each block is an immutable list of ObsDim entries.  Presets concatenate
# blocks together so the resulting spec is the union of all chosen blocks.

# --- Player totals (PySC2 obs.observation["player"]) -------------------------
_PLAYER_BASE_DIMS: list[ObsDim] = [
    ObsDim("minerals",         1000.0,  "Current mineral count"),
    ObsDim("vespene",          1000.0,  "Current vespene count"),
    ObsDim("food_used",         200.0,  "Supply used"),
    ObsDim("food_cap",          200.0,  "Supply cap"),
    ObsDim("army_count",        100.0,  "Total army units"),
]

_SELECTED_DIMS: list[ObsDim] = [
    ObsDim("selected_count",     50.0,  "Number of units currently selected"),
    ObsDim("selected_avg_hp",   100.0,  "Mean HP of selected units"),
]

# --- Screen summary (player_relative) ----------------------------------------
_SCREEN_SUMMARY_DIMS: list[ObsDim] = [
    ObsDim("screen_self_count", 200.0,  "Friendly unit pixel count on screen"),
    ObsDim("screen_enemy_count",200.0,  "Enemy unit pixel count on screen"),
    ObsDim("screen_self_cx",     64.0,  "Friendly unit centroid x (screen)"),
    ObsDim("screen_self_cy",     64.0,  "Friendly unit centroid y (screen)"),
    ObsDim("screen_enemy_cx",    64.0,  "Enemy unit centroid x (screen)"),
    ObsDim("screen_enemy_cy",    64.0,  "Enemy unit centroid y (screen)"),
]

# --- Player extras (used by the ladder spec) ---------------------------------
_PLAYER_EXTRA_DIMS: list[ObsDim] = [
    ObsDim("idle_worker_count",  50.0,  "Idle worker count"),
    ObsDim("warp_gate_count",    20.0,  "Warp gate count"),
    ObsDim("larva_count",        20.0,  "Larva count"),
    ObsDim("food_workers",      200.0,  "Supply tied up in workers"),
    ObsDim("food_army",         200.0,  "Supply tied up in army units"),
]

# --- Minimap summary (ladder-only) -------------------------------------------
_MINIMAP_SUMMARY_DIMS: list[ObsDim] = [
    ObsDim("minimap_self_count", 200.0, "Friendly pixel count on minimap"),
    ObsDim("minimap_enemy_count",200.0, "Enemy pixel count on minimap (visible only)"),
    ObsDim("minimap_visible_frac", 1.0, "Fraction of minimap currently visible"),
    ObsDim("minimap_explored_frac",1.0, "Fraction of minimap ever explored"),
    ObsDim("minimap_camera_x",    64.0, "Camera centroid x on minimap"),
    ObsDim("minimap_camera_y",    64.0, "Camera centroid y on minimap"),
    ObsDim("game_loop",        20000.0, "Current game loop tick"),
]

# --- Score-cumulative breakdown (ladder + rich) ------------------------------
# PySC2's score_cumulative has 13 entries; we expose the most informative ones.
_SCORE_DIMS: list[ObsDim] = [
    ObsDim("score_total",                  10000.0, "Cumulative environment score"),
    ObsDim("idle_production_time",         10000.0, "Time structures spent idle (sum)"),
    ObsDim("idle_worker_time",             10000.0, "Time workers spent idle (sum)"),
    ObsDim("total_value_units",            10000.0, "Mineral+vespene value of all units built"),
    ObsDim("total_value_structures",       10000.0, "Mineral+vespene value of all structures built"),
    ObsDim("killed_value_units",           10000.0, "Mineral+vespene value of enemy units killed"),
    ObsDim("killed_value_structures",      10000.0, "Mineral+vespene value of enemy structures killed"),
    ObsDim("collected_minerals",           10000.0, "Cumulative minerals collected"),
    ObsDim("collected_vespene",            10000.0, "Cumulative vespene collected"),
    ObsDim("collection_rate_minerals",      2000.0, "Mineral collection rate (per minute)"),
    ObsDim("collection_rate_vespene",       2000.0, "Vespene collection rate (per minute)"),
    ObsDim("spent_minerals",               10000.0, "Cumulative minerals spent"),
    ObsDim("spent_vespene",                10000.0, "Cumulative vespene spent"),
]

# --- Screen unit-density / HP summaries (ladder + rich) ----------------------
_SCREEN_HP_DIMS: list[ObsDim] = [
    ObsDim("screen_unit_density_mean",     16.0,  "Mean unit density across screen"),
    ObsDim("screen_self_hp_mean",         100.0,  "Mean friendly unit HP on screen"),
    ObsDim("screen_enemy_hp_mean",        100.0,  "Mean enemy unit HP on screen"),
]

# --- Top-K enemy summary (ladder gives counts; rich gives positions) ---------
_TOPK_ENEMY_COUNTS_DIMS: list[ObsDim] = [
    ObsDim("topk_enemy_within_8",          50.0,  "Enemy units within 8 px of friendly centroid"),
    ObsDim("topk_enemy_within_24",         50.0,  "Enemy units within 24 px of friendly centroid"),
]

#: Top-3 closest enemies relative to friendly centroid (rich spec only).
_TOPK_ENEMY_FEATURES_DIMS: list[ObsDim] = [
    *(
        ObsDim(f"topk_enemy_{i}_rel_x", 64.0,
               f"Top-{i + 1} closest enemy: rel x to friendly centroid")
        for i in range(3)
    ),
    *(
        ObsDim(f"topk_enemy_{i}_rel_y", 64.0,
               f"Top-{i + 1} closest enemy: rel y to friendly centroid")
        for i in range(3)
    ),
    *(
        ObsDim(f"topk_enemy_{i}_hp_ratio", 1.0,
               f"Top-{i + 1} closest enemy: HP / max HP")
        for i in range(3)
    ),
]

# --- Per-unit-type counts (rich) ---------------------------------------------
# Static union of common unit-type names across the three races.  PySC2
# exposes feature_units with a unit_type id; the client maps the id to a
# friendly label using pysc2.lib.units.

_RICH_UNIT_TYPES: tuple[str, ...] = (
    "Marine", "SCV", "Zergling", "Drone", "Probe", "Stalker", "Roach", "Mutalisk",
)

_PER_UNIT_TYPE_DIMS: list[ObsDim] = [
    ObsDim(f"unit_count_{name}", 50.0, f"Friendly count of unit type {name}")
    for name in _RICH_UNIT_TYPES
]

# --- Per-quadrant screen counts (rich) ---------------------------------------
_QUADRANT_DIMS: list[ObsDim] = [
    ObsDim("screen_self_NE_count",   100.0, "Friendly count, NE screen quadrant"),
    ObsDim("screen_self_NW_count",   100.0, "Friendly count, NW screen quadrant"),
    ObsDim("screen_self_SE_count",   100.0, "Friendly count, SE screen quadrant"),
    ObsDim("screen_self_SW_count",   100.0, "Friendly count, SW screen quadrant"),
    ObsDim("screen_enemy_NE_count",  100.0, "Enemy count, NE screen quadrant"),
    ObsDim("screen_enemy_NW_count",  100.0, "Enemy count, NW screen quadrant"),
    ObsDim("screen_enemy_SE_count",  100.0, "Enemy count, SE screen quadrant"),
    ObsDim("screen_enemy_SW_count",  100.0, "Enemy count, SW screen quadrant"),
]

# --- Available-actions binary mask + last-action one-hot (rich) --------------
_AVAILABLE_ACTIONS_DIMS: list[ObsDim] = [
    ObsDim(f"available_fn_{i}", 1.0, f"1 if PySC2 function id index {i} is available")
    for i in range(len(FUNCTION_IDS))
]

_LAST_ACTION_DIMS: list[ObsDim] = [
    ObsDim(f"last_fn_{i}", 1.0, f"1 if last issued action had fn_idx == {i}")
    for i in range(len(FUNCTION_IDS))
]


# ---------------------------------------------------------------------------
# Preset assembly
# ---------------------------------------------------------------------------

# Minigame preset — kept exactly as before #126 for backward compatibility
# with existing minigame champion weight files.
_MINIGAME_DIMS: list[ObsDim] = (
    _PLAYER_BASE_DIMS
    + _SELECTED_DIMS
    + _SCREEN_SUMMARY_DIMS
)

#: 13-dim preset for PySC2 minigames.
SC2_MINIGAME_OBS_SPEC: ObsSpec = ObsSpec(_MINIGAME_DIMS)

# Ladder preset — minigame baseline plus economy / minimap / score / HP
# summaries and top-K enemy counts.
_LADDER_DIMS: list[ObsDim] = (
    _MINIGAME_DIMS
    + _PLAYER_EXTRA_DIMS
    + _MINIMAP_SUMMARY_DIMS
    + _SCORE_DIMS
    + _SCREEN_HP_DIMS
    + _TOPK_ENEMY_COUNTS_DIMS
)

#: ~40-dim preset for 1v1 ladder play.
SC2_LADDER_OBS_SPEC: ObsSpec = ObsSpec(_LADDER_DIMS)

# Rich preset — full superset for research / CNN policies.
_RICH_DIMS: list[ObsDim] = (
    _LADDER_DIMS
    + _PER_UNIT_TYPE_DIMS
    + _QUADRANT_DIMS
    + _TOPK_ENEMY_FEATURES_DIMS
    + _AVAILABLE_ACTIONS_DIMS
    + _LAST_ACTION_DIMS
)

#: ~70-dim research preset.
SC2_RICH_OBS_SPEC: ObsSpec = ObsSpec(_RICH_DIMS)


# ---------------------------------------------------------------------------
# Derived constants — mirror the style used by games.tmnf / games.torcs.
# ---------------------------------------------------------------------------

#: Default spec — minigames are the MVP.
SC2_OBS_SPEC: ObsSpec = SC2_MINIGAME_OBS_SPEC

BASE_OBS_DIM: int = SC2_OBS_SPEC.dim
OBS_NAMES: list[str] = SC2_OBS_SPEC.names
OBS_SCALES: np.ndarray = SC2_OBS_SPEC.scales
OBS_SPEC: list[ObsDim] = list(SC2_OBS_SPEC.dims)

LADDER_OBS_DIM: int = SC2_LADDER_OBS_SPEC.dim
LADDER_OBS_NAMES: list[str] = SC2_LADDER_OBS_SPEC.names

RICH_OBS_DIM: int = SC2_RICH_OBS_SPEC.dim
RICH_OBS_NAMES: list[str] = SC2_RICH_OBS_SPEC.names


#: The 7 standard PySC2 minigame map names.
MINIGAME_NAMES: tuple[str, ...] = (
    "MoveToBeacon",
    "CollectMineralShards",
    "FindAndDefeatZerglings",
    "DefeatRoaches",
    "DefeatZerglingsAndBanelings",
    "CollectMineralsAndGas",
    "BuildMarines",
)


_PRESETS: dict[str, ObsSpec] = {
    "minigame": SC2_MINIGAME_OBS_SPEC,
    "ladder":   SC2_LADDER_OBS_SPEC,
    "rich":     SC2_RICH_OBS_SPEC,
}


def get_spec(map_name: str, preset: str | None = None) -> ObsSpec:
    """Return the appropriate ObsSpec for *map_name* / *preset*.

    Parameters
    ----------
    map_name :
        PySC2 map name.  When *preset* is omitted the default mapping is:
        minigame names → ``minigame``; anything else → ``ladder``.
    preset :
        Optional explicit preset key (``"minigame"`` / ``"ladder"`` / ``"rich"``).
        When set, overrides the map-name default.  Lets users opt into the
        ``rich`` superset without renaming their map.

    Raises
    ------
    ValueError
        If *preset* is not one of the registered keys.
    """
    if preset is not None:
        if preset not in _PRESETS:
            raise ValueError(
                f"Unknown obs_spec_preset {preset!r}.  "
                f"Valid keys: {sorted(_PRESETS.keys())}"
            )
        return _PRESETS[preset]
    if map_name in MINIGAME_NAMES:
        return SC2_MINIGAME_OBS_SPEC
    return SC2_LADDER_OBS_SPEC
