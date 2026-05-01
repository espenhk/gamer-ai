"""Assetto Corsa action definitions.

The continuous action layout is identical to TMNF's so framework policies
(WeightedLinearPolicy, NeuralNetPolicy, CMAESPolicy, …) work unchanged:

    action[0] steer  ∈ [-1, 1]
    action[1] accel  ∈ [0, 1]   (thresholded at 0.5 for discrete output heads)
    action[2] brake  ∈ [0, 1]   (thresholded at 0.5)

For Q-table policies (epsilon_greedy, mcts) the same 9-action discrete grid
as TMNF is used.
"""

from __future__ import annotations

import numpy as np


# 9-action discrete grid: {brake, coast, accel} × {left, straight, right}
DISCRETE_ACTIONS = np.array([
    [-1., 0., 1.],   # 0: brake + left
    [ 0., 0., 1.],   # 1: brake + straight
    [ 1., 0., 1.],   # 2: brake + right
    [-1., 0., 0.],   # 3: coast + left
    [ 0., 0., 0.],   # 4: coast + straight
    [ 1., 0., 0.],   # 5: coast + right
    [-1., 1., 0.],   # 6: accel + left
    [ 0., 1., 0.],   # 7: accel + straight
    [ 1., 1., 0.],   # 8: accel + right
], dtype=np.float32)


# Probe phase fixed-action episodes — used only by the hill-climbing policy.
# The AC env defaults to the genetic policy, so an empty list is fine in
# normal operation; populating it lets users opt into hill-climbing later.
PROBE_ACTIONS: list[tuple[np.ndarray, str]] = [
    (np.array([-1., 1., 0.], dtype=np.float32), "accel left"),
    (np.array([ 0., 1., 0.], dtype=np.float32), "accel"),
    (np.array([ 1., 1., 0.], dtype=np.float32), "accel right"),
    (np.array([-1., 0., 1.], dtype=np.float32), "brake left"),
    (np.array([ 0., 0., 1.], dtype=np.float32), "brake"),
    (np.array([ 1., 0., 1.], dtype=np.float32), "brake right"),
]


# Forced action used during episode warmup (first N steps): full throttle, straight.
WARMUP_ACTION = np.array([0., 1., 0.], dtype=np.float32)
