"""TORCS-specific action definitions.

DISCRETE_ACTIONS  — 9×3 array covering {brake, coast, accel} × {left, straight, right}
PROBE_ACTIONS     — fixed-action episodes for cold-start evaluation
WARMUP_ACTION     — full-throttle straight used during episode warmup steps

These mirror the TMNF action definitions but are kept separate so TORCS can
diverge if needed (e.g. different steering range, gear control).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Discrete action set for Q-table policies (EpsilonGreedy, MCTS)
# ---------------------------------------------------------------------------
# Each row is a (3,) action: [steer, accel, brake]

DISCRETE_ACTIONS = np.array([
    [-1., 0., 1.],   #  0: brake + left
    [ 0., 0., 1.],   #  1: brake + straight
    [ 1., 0., 1.],   #  2: brake + right
    [-1., 0., 0.],   #  3: coast + left
    [ 0., 0., 0.],   #  4: coast + straight
    [ 1., 0., 0.],   #  5: coast + right
    [-1., 1., 0.],   #  6: accel + left
    [ 0., 1., 0.],   #  7: accel + straight
    [ 1., 1., 0.],   #  8: accel + right
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Probe actions — fixed action vectors for cold-start evaluation.
# Each entry is (action_array, description_string).
# ---------------------------------------------------------------------------

PROBE_ACTIONS: list[tuple[np.ndarray, str]] = [
    (np.array([-1., 0., 1.], dtype=np.float32), "brake left"),
    (np.array([ 0., 0., 1.], dtype=np.float32), "brake"),
    (np.array([ 1., 0., 1.], dtype=np.float32), "brake right"),
    (np.array([-1., 1., 0.], dtype=np.float32), "accel left"),
    (np.array([ 0., 1., 0.], dtype=np.float32), "accel"),
    (np.array([ 1., 1., 0.], dtype=np.float32), "accel right"),
]

# Forced action used during episode warmup (first N steps): full throttle, straight.
WARMUP_ACTION = np.array([0., 1., 0.], dtype=np.float32)
