"""StarCraft 2 observation space definition.

The PySC2 API exposes a rich observation: structured ``player`` totals, two
spatial feature-layer stacks (minimap and screen), and per-unit feature lists.
This game integration starts with a flat, fixed-size structured-only vector
so the existing framework policies (linear, MLP, evolutionary) can consume
SC2 observations without architectural changes.

Two specs are defined:

``SC2_MINIGAME_OBS_SPEC``
    Compact 13-dim spec covering player totals plus a few spatial summary
    statistics.  Designed for the 7 standard PySC2 minigames where the
    relevant signal is small-scale unit positioning.

``SC2_LADDER_OBS_SPEC``
    21-dim extension adding economy and supply features used by the 1v1
    ladder env stub.  Issue #91 (belief framework) will further extend this
    with per-region staleness / belief features.
"""

from __future__ import annotations

import numpy as np

from framework.obs_spec import ObsDim, ObsSpec


# ---------------------------------------------------------------------------
# Minigame spec — 13 dims
# ---------------------------------------------------------------------------
# The 7 standard PySC2 minigames give scalar score directly; the structured
# observation here exists so policies can condition on the immediate state.

_MINIGAME_DIMS: list[ObsDim] = [
    # Player totals (PySC2 obs.observation["player"] vector).
    ObsDim("minerals",         1000.0,  "Current mineral count"),
    ObsDim("vespene",          1000.0,  "Current vespene count"),
    ObsDim("food_used",         200.0,  "Supply used"),
    ObsDim("food_cap",          200.0,  "Supply cap"),
    ObsDim("army_count",        100.0,  "Total army units"),
    # Selected-unit summary (cached from feature_screen.selected when present).
    ObsDim("selected_count",     50.0,  "Number of units currently selected"),
    ObsDim("selected_avg_hp",   100.0,  "Mean HP of selected units"),
    # Spatial summary stats over the screen feature layers.
    ObsDim("screen_self_count", 200.0,  "Friendly unit pixel count on screen"),
    ObsDim("screen_enemy_count",200.0,  "Enemy unit pixel count on screen"),
    ObsDim("screen_self_cx",     64.0,  "Friendly unit centroid x (screen)"),
    ObsDim("screen_self_cy",     64.0,  "Friendly unit centroid y (screen)"),
    ObsDim("screen_enemy_cx",    64.0,  "Enemy unit centroid x (screen)"),
    ObsDim("screen_enemy_cy",    64.0,  "Enemy unit centroid y (screen)"),
]

#: The canonical observation spec for PySC2 minigames.
SC2_MINIGAME_OBS_SPEC: ObsSpec = ObsSpec(_MINIGAME_DIMS)


# ---------------------------------------------------------------------------
# Ladder game spec — extends minigame spec with economy + minimap summaries
# ---------------------------------------------------------------------------

_LADDER_EXTRA_DIMS: list[ObsDim] = [
    ObsDim("idle_worker_count",  50.0,  "Idle worker count"),
    ObsDim("warp_gate_count",    20.0,  "Warp gate count"),
    ObsDim("larva_count",        20.0,  "Larva count"),
    # Minimap-level summaries — distinct from screen stats above.
    ObsDim("minimap_self_count",200.0,  "Friendly pixel count on minimap"),
    ObsDim("minimap_enemy_count",200.0, "Enemy pixel count on minimap (visible only)"),
    ObsDim("minimap_visible_frac",1.0,  "Fraction of minimap currently visible"),
    ObsDim("minimap_explored_frac",1.0, "Fraction of minimap ever explored"),
    ObsDim("game_loop",        20000.0, "Current game loop tick"),
]

#: Observation spec for the 1v1 ladder game stub.
SC2_LADDER_OBS_SPEC: ObsSpec = ObsSpec(_MINIGAME_DIMS + _LADDER_EXTRA_DIMS)


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


def get_spec(map_name: str) -> ObsSpec:
    """Return the appropriate ObsSpec for the given map.

    Minigame names map to ``SC2_MINIGAME_OBS_SPEC``.  Anything else is
    treated as a ladder map and gets ``SC2_LADDER_OBS_SPEC``.
    """
    if map_name in MINIGAME_NAMES:
        return SC2_MINIGAME_OBS_SPEC
    return SC2_LADDER_OBS_SPEC


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
