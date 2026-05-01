"""CarRacing-specific action definitions."""
from __future__ import annotations

import numpy as np


DISCRETE_ACTIONS = np.array([
    [-1., 0., 1.],
    [ 0., 0., 1.],
    [ 1., 0., 1.],
    [-1., 0., 0.],
    [ 0., 0., 0.],
    [ 1., 0., 0.],
    [-1., 1., 0.],
    [ 0., 1., 0.],
    [ 1., 1., 0.],
], dtype=np.float32)

PROBE_ACTIONS: list[tuple[np.ndarray, str]] = [
    (np.array([-1., 0., 1.], dtype=np.float32), "brake left"),
    (np.array([ 0., 0., 1.], dtype=np.float32), "brake"),
    (np.array([ 1., 0., 1.], dtype=np.float32), "brake right"),
    (np.array([-1., 1., 0.], dtype=np.float32), "accel left"),
    (np.array([ 0., 1., 0.], dtype=np.float32), "accel"),
    (np.array([ 1., 1., 0.], dtype=np.float32), "accel right"),
]

WARMUP_ACTION = np.array([0., 1., 0.], dtype=np.float32)
