"""<GAME_TITLE> action definitions.

``DISCRETE_ACTIONS`` lists every discrete action a tabular policy can pick.
The simple CarRacing convention uses 9 actions (3 × 3 grid: {left, straight, right} ×
{brake, coast, accel}):

    DISCRETE_ACTIONS = np.array([
        [-1.0, 0.0, 1.0],   # brake left
        [ 0.0, 0.0, 1.0],   # brake straight
        [ 1.0, 0.0, 1.0],   # brake right
        [-1.0, 0.0, 0.0],   # coast left
        [ 0.0, 0.0, 0.0],   # coast (no-op)
        [ 1.0, 0.0, 0.0],   # coast right
        [-1.0, 1.0, 0.0],   # accel left
        [ 0.0, 1.0, 0.0],   # accel straight
        [ 1.0, 1.0, 0.0],   # accel right
    ], dtype=np.float32)

Replace with whatever action tuples make sense for your game.
"""

from __future__ import annotations

import numpy as np

# TODO: populate with your game's discrete action tuples.
DISCRETE_ACTIONS = np.zeros((0, 3), dtype=np.float32)
