"""Backward-compatibility shim.

All symbols previously defined here have moved to games.tmnf.constants.
Import from that module directly in new TMNF code.
"""

from games.tmnf.constants import STEER_SCALE, UP_VECTOR, N_ACTIONS  # noqa: F401
