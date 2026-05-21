"""SC2-specific compatibility policies kept for linear/genetic weight loading.

This module now intentionally only contains:
- SC2LinearPolicy
- SC2GeneticPolicy

Legacy bare-name SC2 algorithm classes were removed.
"""
from __future__ import annotations

import numpy as np

from framework.obs_spec import ObsSpec
from framework.policies import GeneticPolicy, WeightedLinearPolicy
from games.sc2.actions import FUNCTION_IDS

# Number of distinct SC2 function IDs exposed by the action mapping.
_N_FUNCS = len(FUNCTION_IDS)


class SC2LinearPolicy(WeightedLinearPolicy):
    """Linear policy with sigmoid-based output encoding for SC2 actions."""

    _N_FUNCS: int = _N_FUNCS

    @staticmethod
    def _sigmoid(score: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(score, -20.0, 20.0))))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        norm_obs = obs / self._obs_spec.scales
        fn_score = float(np.dot(self._weights["fn_idx"], norm_obs))
        x_score = float(np.dot(self._weights["x"], norm_obs))
        y_score = float(np.dot(self._weights["y"], norm_obs))
        q_score = float(np.dot(self._weights["queue"], norm_obs))

        fn_idx = self._sigmoid(fn_score) * (self._N_FUNCS - 1)
        x = self._sigmoid(x_score)
        y = self._sigmoid(y_score)
        queue = float(int(self._sigmoid(q_score) > 0.5))
        return np.array([fn_idx, x, y, queue], dtype=np.float32)


class SC2GeneticPolicy(GeneticPolicy):
    """GeneticPolicy variant that uses SC2LinearPolicy members."""

    def _make_member(self, cfg: dict) -> SC2LinearPolicy:
        return SC2LinearPolicy.from_cfg(cfg, self._obs_spec, self._head_names)
