"""Atari action definitions.

Atari games expose a discrete action set of variable size (typically 4–18
depending on the title).  The largest valid set is the 18-action full
Atari space; per-game minimal sets are a subset of these indices.

The framework needs a fixed ``discrete_actions`` ndarray at adapter
build-time, before the env is created.  We use the full 18-action grid:
games with smaller legal sets clamp out-of-range indices to a NOOP in
``AtariEnv.step``.
"""

from __future__ import annotations

import numpy as np

#: Full Atari action space size (ALE base set).
N_ACTIONS_FULL: int = 18

#: One-dimensional discrete action grid, shape ``(N_ACTIONS_FULL, 1)``.
#: Each row holds the integer action index as a float so the array is
#: compatible with the framework's ``np.ndarray`` action contract.
DISCRETE_ACTIONS: np.ndarray = np.arange(N_ACTIONS_FULL, dtype=np.float32).reshape(-1, 1)
