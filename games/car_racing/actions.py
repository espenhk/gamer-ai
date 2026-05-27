"""CarRacing-specific action definitions."""

from __future__ import annotations

import numpy as np

from framework.run_config import ProbeAction

DISCRETE_ACTIONS = np.array(
    [
        [-1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

PROBE_ACTIONS: list[ProbeAction] = [
    ProbeAction(np.array([-1.0, 0.0, 1.0], dtype=np.float32), "brake left"),
    ProbeAction(np.array([0.0, 0.0, 1.0], dtype=np.float32), "brake"),
    ProbeAction(np.array([1.0, 0.0, 1.0], dtype=np.float32), "brake right"),
    ProbeAction(np.array([-1.0, 1.0, 0.0], dtype=np.float32), "accel left"),
    ProbeAction(np.array([0.0, 1.0, 0.0], dtype=np.float32), "accel"),
    ProbeAction(np.array([1.0, 1.0, 0.0], dtype=np.float32), "accel right"),
]

WARMUP_ACTION = np.array([0.0, 1.0, 0.0], dtype=np.float32)
