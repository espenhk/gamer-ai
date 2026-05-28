"""Rocket League action definitions.

Action space
------------
``Box([-1, -1, -1, -1, -1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], shape=(8,))``

  [0] throttle  — forward/reverse throttle in [-1, 1]
  [1] steer     — left/right steering in [-1, 1]
  [2] pitch     — nose up/down in [-1, 1]
  [3] yaw       — left/right rotation (air only) in [-1, 1]
  [4] roll      — barrel-roll (air only) in [-1, 1]
  [5] jump      — jump button [0, 1]; thresholded at 0.5 → bool
  [6] boost     — boost button [0, 1]; thresholded at 0.5 → bool
  [7] handbrake — powerslide/handbrake button [0, 1]; thresholded at 0.5 → bool

DISCRETE_ACTIONS
----------------
A 9×8 array covering the most common control combinations:
  {throttle: -1/0/+1} × {steer: -1/0/+1} with boost/jump/handbrake off.
Tabular policies (EpsilonGreedy, UCBQ) index into this table.
"""

from __future__ import annotations

import numpy as np

from framework.run_config import ProbeAction

# ---------------------------------------------------------------------------
# Continuous action bounds
# ---------------------------------------------------------------------------

ACTION_LOW = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
ACTION_HIGH = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

# ---------------------------------------------------------------------------
# Discrete action set (for tabular / epsilon-greedy policies)
# ---------------------------------------------------------------------------
# Each row: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]

DISCRETE_ACTIONS = np.array(
    [
        # throttle -1 (reverse)
        [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #  0: reverse + steer left
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #  1: reverse + straight
        [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #  2: reverse + steer right
        # throttle 0 (coast)
        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #  3: coast + steer left
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #  4: coast + straight (no-op)
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #  5: coast + steer right
        # throttle +1 (full throttle)
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #  6: throttle + steer left
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #  7: throttle + straight
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #  8: throttle + steer right
        # With boost
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  #  9: boost + steer left
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # 10: boost + straight
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # 11: boost + steer right
        # Jump (single frame)
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # 12: throttle + jump
        # Handbrake (powerslide)
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 13: throttle + left + handbrake
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 14: throttle + right + handbrake
    ],
    dtype=np.float32,
)

# ---------------------------------------------------------------------------
# Probe actions — fixed-action episodes for cold-start evaluation.
# ---------------------------------------------------------------------------

PROBE_ACTIONS: list[ProbeAction] = [
    ProbeAction(np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), "reverse"),
    ProbeAction(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), "coast"),
    ProbeAction(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), "throttle"),
    ProbeAction(np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), "throttle left"),
    ProbeAction(np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), "throttle right"),
    ProbeAction(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32), "boost straight"),
]

# Warmup action: full throttle straight, no boost/jump.
WARMUP_ACTION = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
