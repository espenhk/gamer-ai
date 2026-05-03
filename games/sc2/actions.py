"""StarCraft 2 action definitions.

PySC2's full action space is parameterised: ``function_id`` + per-arg target
coordinates.  That doesn't fit the existing ``Discrete(N)`` policies cleanly,
so this MVP exposes a small flat discrete subset per minigame.  Continuous
target coordinates are emitted for spatial actions (move-screen, attack-screen).

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

The 9-action grid below is sufficient for ``MoveToBeacon`` and
``CollectMineralShards`` — both reduce to "select army, move to a screen
cell".  Other minigames use the same grid plus a small extension elsewhere.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Function-id table
# ---------------------------------------------------------------------------
# Indices are stable; the env client resolves them to PySC2 function ids at
# call time so we don't need pysc2 imported at framework level.

FUNCTION_IDS = {
    0: "no_op",                         # actions.FUNCTIONS.no_op
    1: "select_army",                   # actions.FUNCTIONS.select_army
    2: "Move_screen",                   # actions.FUNCTIONS.Move_screen
    3: "Attack_screen",                 # actions.FUNCTIONS.Attack_screen
    4: "select_idle_worker",            # actions.FUNCTIONS.select_idle_worker
    5: "Harvest_Gather_screen",         # actions.FUNCTIONS.Harvest_Gather_screen
}


# ---------------------------------------------------------------------------
# Discrete action grid for minigame policies — 9 cells
# ---------------------------------------------------------------------------
# fn_idx 1 = select_army (instant), fn_idx 2 = Move_screen at one of 8
# directional cells.  This is the minimum action set needed for MoveToBeacon
# and is reused for the other minigames.
#
# Cells are arranged as a 3×3 grid over the screen.  The centre cell (4)
# selects the army instead of moving — this gives the policy a way to issue
# the prerequisite selection.

_GRID = [
    (0.20, 0.20),  # 0: top-left
    (0.50, 0.20),  # 1: top
    (0.80, 0.20),  # 2: top-right
    (0.20, 0.50),  # 3: left
    (0.50, 0.50),  # 4: centre  (select_army)
    (0.80, 0.50),  # 5: right
    (0.20, 0.80),  # 6: bottom-left
    (0.50, 0.80),  # 7: bottom
    (0.80, 0.80),  # 8: bottom-right
]

DISCRETE_ACTIONS = np.array(
    [
        [1, _GRID[i][0], _GRID[i][1], 0] if i == 4
        else [2, _GRID[i][0], _GRID[i][1], 0]
        for i in range(9)
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Probe actions — fixed action vectors for cold-start evaluation
# ---------------------------------------------------------------------------
# Each entry is (action_array, description_string).  Probes establish a
# reward floor before random-restart hill-climbing kicks in.

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


def build_available_actions_mask(
    available_fn_ids: set[int], n_cells: int = len(DISCRETE_ACTIONS)
) -> np.ndarray:
    """Boolean mask of shape (n_cells,) — True where the action is legal."""
    return np.array(
        [discrete_action_to_fn_id(i) in available_fn_ids for i in range(n_cells)],
        dtype=bool,
    )


def action_to_function_call(action: np.ndarray, screen_size: int):
    """Translate a 4-vector action row into a PySC2 ``FunctionCall``.

    Parameters
    ----------
    action :
        4-vector ``[fn_idx, x, y, queue]`` produced by a policy.
    screen_size :
        Size of the screen feature layer (e.g. 64).  Used to denormalise
        the coordinate args.

    Returns
    -------
    pysc2.lib.actions.FunctionCall

    Notes
    -----
    Imports PySC2 lazily so that callers without PySC2 installed (unit
    tests, framework code) can import this module freely.
    """
    from pysc2.lib import actions  # type: ignore[import-untyped]

    fn_idx = int(action[0])
    x_norm = float(np.clip(action[1], 0.0, 1.0))
    y_norm = float(np.clip(action[2], 0.0, 1.0))
    queue = int(np.clip(round(float(action[3])), 0, 1))
    sx = int(x_norm * (screen_size - 1))
    sy = int(y_norm * (screen_size - 1))

    name = FUNCTION_IDS.get(fn_idx, "no_op")
    fn = getattr(actions.FUNCTIONS, name, actions.FUNCTIONS.no_op)
    fn_id = int(fn.id)

    if name == "no_op" or name == "select_army" or name == "select_idle_worker":
        # No spatial args; queue may not apply for instant actions.
        if name == "select_army" or name == "select_idle_worker":
            return actions.FunctionCall(fn_id, [[0]])
        return actions.FunctionCall(fn_id, [])
    # Spatial actions: [queued, target_screen]
    return actions.FunctionCall(fn_id, [[queue], [sx, sy]])
