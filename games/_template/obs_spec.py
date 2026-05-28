"""<GAME_TITLE> observation space definition.

Each feature in the observation vector gets a name and a scale.  The
framework divides the raw value by the scale before handing it to a policy,
so aim for normalised values roughly in ``[-1, 1]``.

Copy this file into ``games/<name>/obs_spec.py`` and replace the example
row(s) with real features.  See ``docs/framework/obs_spec.md`` for details.
"""

from __future__ import annotations

import numpy as np

from framework.obs_spec import ObsDim, ObsSpec

# Replace these example dims with your game's actual features.
_TEMPLATE_DIMS: list[ObsDim] = [
    ObsDim("example_feature", 1.0, "Replace with a real feature"),
]

#: The canonical observation spec for this game.
TEMPLATE_OBS_SPEC: ObsSpec = ObsSpec(_TEMPLATE_DIMS)

#: Number of base observation features.
BASE_OBS_DIM: int = TEMPLATE_OBS_SPEC.dim

#: Ordered list of feature names.
OBS_NAMES: list[str] = TEMPLATE_OBS_SPEC.names

#: Float32 scale array, shape (BASE_OBS_DIM,).
OBS_SCALES: np.ndarray = TEMPLATE_OBS_SPEC.scales

#: Plain OBS_SPEC list for callers that iterate over ObsDim entries.
OBS_SPEC: list[ObsDim] = _TEMPLATE_DIMS
