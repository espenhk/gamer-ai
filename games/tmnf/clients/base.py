"""PhaseAwareClient — base class for all TMNF TMInterface clients.

Provides the shared phase state machine fields and _transition() helper so
each concrete client only implements the phase-specific driving logic.
"""

import logging
from enum import Enum, auto

from tminterface.client import Client

logger = logging.getLogger(__name__)


class Phase(Enum):
    BRAKING_START = auto()
    PAUSE_START   = auto()
    RUNNING       = auto()
    BRAKING_END   = auto()
    PAUSE_END     = auto()
    DONE          = auto()


class PhaseAwareClient(Client):
    """Base class that owns _phase/_phase_start_ms and the _transition() helper.

    All three client types (InstructionClient, AdaptiveClient, RLClient) follow
    the same phase lifecycle.  Rather than duplicating the fields and the
    transition method in each class, they inherit from here.
    """

    def __init__(self) -> None:
        super().__init__()
        self._phase: Phase = Phase.BRAKING_START
        self._phase_start_ms: int = 0

    def _transition(self, phase: Phase, current_time_ms: int) -> None:
        """Log and apply a phase transition."""
        logger.debug("Phase: %s -> %s", self._phase.name, phase.name)
        self._phase = phase
        self._phase_start_ms = current_time_ms
