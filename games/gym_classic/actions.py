"""Gymnasium classic-control action definitions.

Each supported env has a different number of discrete actions.  The
framework's ``DISCRETE_ACTIONS`` table follows the same convention as the
Atari adapter: each row holds the integer action index as a float so the
array is compatible with the framework's ``np.ndarray`` action contract.

The env wrapper maps the float action value back to a gym integer action
index using the same logic as ``games.atari.env``:
  - Integer-valued floats (e.g. 0.0, 1.0, 2.0) from tabular/DQN policies
    are used as-is (after rounding and clamping).
  - Continuous [-1, 1] values from evolutionary/gradient policies are
    linearly mapped to [0, n_actions - 1].

Pendulum-v1 uses a continuous ``Box([-2], [2])`` action space.  Its
``DISCRETE_ACTIONS`` table provides a fine grid of torque values for
tabular policies; continuous policies receive their raw output scaled by
``PENDULUM_TORQUE_SCALE``.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Discrete action counts per env
# ---------------------------------------------------------------------------

N_ACTIONS: dict[str, int] = {
    "CartPole-v1": 2,
    "MountainCar-v0": 3,
    "Acrobot-v1": 3,
    "LunarLander-v2": 4,
    # Pendulum is continuous; treat as 11-point torque grid for tabular policies.
    "Pendulum-v1": 11,
}

_DEFAULT_N_ACTIONS: int = 2

# Whether the native gym env uses a continuous Box action space.
IS_CONTINUOUS: dict[str, bool] = {
    "Pendulum-v1": True,
}

# Scale factor from framework [-1, 1] output to Pendulum torque [-2, 2].
PENDULUM_TORQUE_SCALE: float = 2.0


def get_n_actions(map_name: str) -> int:
    """Return the number of discrete actions for *map_name*."""
    return N_ACTIONS.get(map_name, _DEFAULT_N_ACTIONS)


def is_continuous(map_name: str) -> bool:
    """Return True if *map_name* uses a continuous gym action space."""
    return IS_CONTINUOUS.get(map_name, False)


def get_discrete_actions(map_name: str) -> np.ndarray:
    """Return a ``(n, 1)`` DISCRETE_ACTIONS table for *map_name*.

    For discrete envs: rows are integer action indices as floats (0.0, 1.0, ...).
    For Pendulum (continuous): rows are torque values on [-2, 2] as a 11-point grid.
    """
    n = get_n_actions(map_name)
    if IS_CONTINUOUS.get(map_name, False):
        # Pendulum: evenly spaced torque grid from -2 to +2.
        return np.linspace(-PENDULUM_TORQUE_SCALE, PENDULUM_TORQUE_SCALE, n, dtype=np.float32).reshape(-1, 1)
    return np.arange(n, dtype=np.float32).reshape(-1, 1)
