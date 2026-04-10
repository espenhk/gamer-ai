"""Shared constants for the TMNF project.

Centralising these prevents the same magic number appearing in multiple files
and ensures that changing one value (e.g. STEER_SCALE) propagates everywhere.
"""

import numpy as np

# TMInterface encodes steering as a signed integer in [-65536, 65536].
# Convert a [-100, 100] percentage: int(pct / 100 * STEER_SCALE).
STEER_SCALE: int = 65536

# World up-vector in TMNF's coordinate system (Y is up).
UP_VECTOR: np.ndarray = np.array([0.0, 1.0, 0.0])

# Number of discrete actions in the action space.
# Must equal len(ACTIONS) in clients/rl_client.py — asserted there at import time.
N_ACTIONS: int = 9
