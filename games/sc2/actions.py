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

Layout (issue #122 / #127)
--------------------------
- Row 0 is ``no_op`` (issue #127): tabular policies must be able to elect to
  do nothing — necessary so units can shoot while standing still, etc.
- Row 1 is ``select_army``: a near-universal precondition for issuing orders
  to the player's army.  Also the warmup action.
- Rows 2..N-1 are ``Move_screen`` calls at the centres of an N×N grid (where
  N == ``SCREEN_GRID_RESOLUTION``).  At resolution 8 this gives 64 unique
  cells covering the screen at one-cell-per-8-pixels granularity.
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
# Discrete action grid for minigame / tabular policies
# ---------------------------------------------------------------------------
# Resolution N gives an N×N grid of Move_screen targets at cell centres.
# Default 8×8 = 64 cells; combined with no_op + select_army the discrete
# action set has ``2 + N*N`` rows.

SCREEN_GRID_RESOLUTION: int = 8


def _grid_centres(resolution: int) -> list[tuple[float, float]]:
    """Return cell centres for an N×N grid over the unit square."""
    step = 1.0 / resolution
    centres = [(j * step + step / 2.0, i * step + step / 2.0)
               for i in range(resolution)
               for j in range(resolution)]
    return centres


def _build_discrete_actions(resolution: int) -> np.ndarray:
    """Construct the [fn_idx, x, y, queue] table for a given grid resolution.

    Layout: ``[no_op, select_army, Move_screen × resolution^2]``.
    """
    rows: list[list[float]] = []
    # Row 0 — no_op (issue #127).  x/y unused; carry centre coords for sanity.
    rows.append([0, 0.5, 0.5, 0])
    # Row 1 — select_army (warmup precondition).
    rows.append([1, 0.5, 0.5, 0])
    # Rows 2..N²+1 — Move_screen at each cell centre.
    for x, y in _grid_centres(resolution):
        rows.append([2, x, y, 0])
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
# Even with no_op now reachable post-warmup, select_army remains the right
# warmup so the first real policy step has units selected.

WARMUP_ACTION = np.array([1, 0.5, 0.5, 0], dtype=np.float32)


def discrete_action_to_fn_id(cell_idx: int) -> int:
    """Return the FUNCTION_IDS key for grid cell *cell_idx*."""
    return int(DISCRETE_ACTIONS[cell_idx, 0])


def pysc2_ids_to_internal_fn_idx(pysc2_available_ids: set[int]) -> set[int]:
    """Convert a set of raw PySC2 function IDs to internal fn_idx values (0-5).

    PySC2's ``ob["available_actions"]`` contains PySC2 function IDs (e.g. 331
    for Move_screen).  Our mask logic uses the repo-internal fn_idx keys that
    index into FUNCTION_IDS (0..5).  This converts between the two so that
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
