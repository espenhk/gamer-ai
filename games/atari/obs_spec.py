"""Atari RAM observation space definition.

The Atari 2600 has 128 bytes of console RAM exposed by ALE.  Gymnasium
returns these as a uint8 vector when the env is built with
``obs_type="ram"``.  Each byte is scaled by 255.0 so normalised values land
in [0, 1] across all features.

A fixed-size flat vector lets all framework flat-observation policies
(``hill_climbing``, ``neural_net``, ``genetic``, ``cmaes``, ``epsilon_greedy``,
``mcts``, ``neural_dqn``, ``reinforce``, ``lstm``, ``ppo``) run on Atari
without any pixel-CNN machinery.
"""

from __future__ import annotations

import numpy as np

from framework.obs_spec import ObsDim, ObsSpec

RAM_SIZE: int = 128

_ATARI_RAM_DIMS: list[ObsDim] = [
    ObsDim(f"ram_{i:03d}", 255.0, f"Console RAM byte {i} (0–255)") for i in range(RAM_SIZE)
]

#: The canonical Atari RAM observation spec (128 dims).
ATARI_OBS_SPEC: ObsSpec = ObsSpec(_ATARI_RAM_DIMS)

#: Number of base observation features.
BASE_OBS_DIM: int = ATARI_OBS_SPEC.dim

#: Ordered list of feature names.
OBS_NAMES: list[str] = ATARI_OBS_SPEC.names

#: Float32 scale array, shape (BASE_OBS_DIM,).
OBS_SCALES: np.ndarray = ATARI_OBS_SPEC.scales

#: Plain list of ObsDim entries for callers that iterate over them.
OBS_SPEC: list[ObsDim] = _ATARI_RAM_DIMS
