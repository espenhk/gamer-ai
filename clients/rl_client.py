"""Backward-compatibility shim. Symbols moved to games.tmnf.clients.rl_client."""
from games.tmnf.clients.rl_client import (  # noqa: F401
    ACTIONS,
    StepState,
    RLClient,
    _DEFAULT_ACTION,
    get_action_description,
    VELOCITY_ZERO_THRESHOLD,
)
from games.tmnf.track import Centerline  # noqa: F401  (tests patch clients.rl_client.Centerline)
