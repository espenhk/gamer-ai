"""TMNF-specific constants.

These values are tightly coupled to the TMInterface API and the TMNF
game engine.  Non-TMNF code should never import from this module.
"""

import numpy as np

# TMInterface encodes steering as a signed integer in [-65536, 65536].
# Convert a [-100, 100] percentage: int(pct / 100 * STEER_SCALE).
STEER_SCALE: int = 65536

# World up-vector in TMNF's coordinate system (Y is up).
UP_VECTOR: np.ndarray = np.array([0.0, 1.0, 0.0])

# Number of discrete actions in the TMNF action space.
# Must stay in sync with len(ACTIONS) in games/tmnf/clients/rl_client.py.
N_ACTIONS: int = 9
