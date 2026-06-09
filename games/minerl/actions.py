"""MineRL discrete action definitions.

Nine discrete actions cover the primary navigation and gathering behaviours.
The environment decodes an integer index (0–8) into a MineRL action dict;
see ``games.minerl.env.MineRLEnv._decode_action`` for the mapping.
"""

from __future__ import annotations

import numpy as np

#: Total number of discrete actions.
N_ACTIONS: int = 9

#: Discrete action set — integer indices in a (N_ACTIONS, 1) float32 array,
#: matching the Atari convention used by the framework's tabular policies.
DISCRETE_ACTIONS: np.ndarray = np.arange(N_ACTIONS, dtype=np.float32).reshape(-1, 1)

#: Human-readable labels for each action index (for debugging / analytics).
ACTION_LABELS: list[str] = [
    "noop",
    "forward",
    "forward+jump",
    "forward+attack",
    "attack",
    "left+forward",
    "right+forward",
    "back",
    "forward+sprint",
]

#: Per-action overrides applied on top of a no-op base dict.
#: Keys follow the MineRL action-space naming convention.
_ACTION_OVERRIDES: list[dict] = [
    {},  # 0: noop
    {"forward": 1},  # 1: forward
    {"forward": 1, "jump": 1},  # 2: forward+jump
    {"forward": 1, "attack": 1},  # 3: forward+attack
    {"attack": 1},  # 4: attack
    {"left": 1, "forward": 1},  # 5: left+forward
    {"right": 1, "forward": 1},  # 6: right+forward
    {"back": 1},  # 7: back
    {"forward": 1, "sprint": 1},  # 8: forward+sprint
]
