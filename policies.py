"""Backward-compatibility shim — TMNF-flavoured policy classes.

All framework policy logic has moved to framework/policies.py.
This module re-exports all classes with TMNF defaults baked in so that
existing callers and tests continue to work without modification.

New game integrations should import directly from framework.policies and
pass their own ObsSpec / action arrays.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import yaml

from framework.policies import (  # noqa: F401
    BasePolicy,
    _discretize_obs,
    WeightedLinearPolicy as _FrameworkWLP,
    NeuralNetPolicy as _FrameworkNNP,
    QTablePolicy as _FrameworkQTP,
    EpsilonGreedyPolicy as _FrameworkEGP,
    MCTSPolicy as _FrameworkMCTSP,
    GeneticPolicy as _FrameworkGP,
)
from games.tmnf.actions import (
    DISCRETE_ACTIONS,
    _action_to_idx,
    _normalize_weight_cfg,
)
from games.tmnf.obs_spec import (
    TMNF_OBS_SPEC,
    OBS_NAMES,
    OBS_SCALES,
    obs_names_with_lidar,
    obs_scales_with_lidar,
)
from games.tmnf.simple_policy import SimplePolicy  # noqa: F401
from games.tmnf.policies import (  # noqa: F401
    ReplayBuffer,
    NeuralDQNPolicy,
    CMAESPolicy,
    REINFORCEPolicy,
    LSTMPolicy,
    LSTMEvolutionPolicy,
)

logger = logging.getLogger(__name__)

# Aliases expected by test_neural_dqn_policy.py (original code used _DISCRETE_ACTIONS)
_DISCRETE_ACTIONS = DISCRETE_ACTIONS

_HEAD_NAMES = ["steer", "accel", "brake"]


# ---------------------------------------------------------------------------
# WeightedLinearPolicy — TMNF variant (steer / accel / brake heads)
# ---------------------------------------------------------------------------

class WeightedLinearPolicy(_FrameworkWLP):
    """
    TMNF WeightedLinearPolicy.

    Drop-in replacement for the original: same constructor signature, same
    YAML format, same class attributes.  Internally backed by
    framework.policies.WeightedLinearPolicy with TMNF obs_spec injected.
    """

    # Class-level aliases kept for backward compatibility.
    OBS_NAMES  = OBS_NAMES
    OBS_SCALES = OBS_SCALES

    @classmethod
    def get_obs_names(cls, n_lidar_rays: int = 0) -> list[str]:
        return obs_names_with_lidar(n_lidar_rays)

    @classmethod
    def get_obs_scales(cls, n_lidar_rays: int = 0) -> np.ndarray:
        return obs_scales_with_lidar(n_lidar_rays)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, weights_file: str, n_lidar_rays: int = 0) -> None:
        self._obs_spec     = TMNF_OBS_SPEC.with_lidar(n_lidar_rays)
        self._head_names   = _HEAD_NAMES
        self._weights_file = weights_file
        cfg = self._tmnf_load_or_init()
        self._apply_cfg(cfg)
        logger.info("[WeightedLinearPolicy] loaded weights from %s", weights_file)

    @classmethod
    def from_cfg(cls, cfg: dict, n_lidar_rays: int = 0) -> "WeightedLinearPolicy":
        """Create a policy from a weights dict (not backed by a file)."""
        obs_spec = TMNF_OBS_SPEC.with_lidar(n_lidar_rays)
        names    = obs_spec.names
        normalized = _normalize_weight_cfg(cfg, names)
        obj = object.__new__(cls)
        obj._weights_file = None
        obj._obs_spec     = obs_spec
        obj._head_names   = _HEAD_NAMES
        obj._apply_cfg(normalized)
        return obj

    # ------------------------------------------------------------------
    # TMNF-specific file I/O (with throttle→accel/brake migration)
    # ------------------------------------------------------------------

    def _tmnf_load_or_init(self) -> dict:
        names = self._obs_spec.names
        if os.path.exists(self._weights_file):
            with open(self._weights_file) as f:
                cfg = yaml.safe_load(f)
            loaded_dim = len(cfg.get("steer_weights", {}))
            if loaded_dim != len(names):
                logger.warning(
                    "[WeightedLinearPolicy] loaded weights dim=%d doesn't match obs_dim=%d; "
                    "new features initialised to 0.0 → %s",
                    loaded_dim, len(names), self._weights_file,
                )
            normalized = _normalize_weight_cfg(cfg, names)
            if normalized != cfg:
                with open(self._weights_file, "w") as f:
                    yaml.dump(normalized, f, default_flow_style=False, sort_keys=False)
                logger.info("[WeightedLinearPolicy] migrated weights file → %s",
                            self._weights_file)
            return normalized

        rng = np.random.default_rng()
        cfg = {
            "steer_weights": {n: float(rng.standard_normal()) for n in names},
            "accel_weights": {n: float(rng.standard_normal()) for n in names},
            "brake_weights": {n: float(rng.standard_normal()) for n in names},
        }
        with open(self._weights_file, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        logger.info("[WeightedLinearPolicy] initialised random weights → %s",
                    self._weights_file)
        return cfg


# ---------------------------------------------------------------------------
# NeuralNetPolicy — TMNF variant
# ---------------------------------------------------------------------------

class NeuralNetPolicy(_FrameworkNNP):
    """TMNF NeuralNetPolicy — obs_spec and action_dim=3 baked in."""

    _OUTPUT_DIM = 3  # backward-compat class attribute

    def __init__(self, hidden_sizes: list[int] | None = None, n_lidar_rays: int = 0) -> None:
        super().__init__(
            obs_spec     = TMNF_OBS_SPEC.with_lidar(n_lidar_rays),
            action_dim   = 3,
            hidden_sizes = hidden_sizes,
        )

    @classmethod
    def from_cfg(cls, cfg: dict, n_lidar_rays: int = 0) -> "NeuralNetPolicy":
        obj             = object.__new__(cls)
        obj._obs_spec   = TMNF_OBS_SPEC.with_lidar(n_lidar_rays)
        obj._action_dim = 3
        obj._hidden     = cfg["hidden_sizes"]
        obj._weights    = [np.array(w, dtype=np.float32) for w in cfg["weights"]]
        obj._biases     = [np.array(b, dtype=np.float32) for b in cfg["biases"]]
        return obj

    def to_cfg(self) -> dict:
        return {
            "policy_type":  "neural_net",
            "hidden_sizes": self._hidden,
            "n_lidar_rays": 0,   # kept for file compatibility
            "weights":      [w.tolist() for w in self._weights],
            "biases":       [b.tolist() for b in self._biases],
        }


# ---------------------------------------------------------------------------
# QTablePolicy — TMNF variant (exposed for tests / isinstance checks)
# ---------------------------------------------------------------------------

class QTablePolicy(_FrameworkQTP):
    """TMNF QTablePolicy — DISCRETE_ACTIONS baked in."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        n_bins: int = 3,
        n_lidar_rays: int = 0,
    ) -> None:
        super().__init__(
            obs_spec        = TMNF_OBS_SPEC.with_lidar(n_lidar_rays),
            discrete_actions = DISCRETE_ACTIONS,
            alpha           = alpha,
            gamma           = gamma,
            n_bins          = n_bins,
        )

    def _select_action(self, s: tuple) -> int:  # pragma: no cover
        return int(np.argmax(self._q(s)))

    def to_cfg(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# EpsilonGreedyPolicy — TMNF variant
# ---------------------------------------------------------------------------

class EpsilonGreedyPolicy(_FrameworkEGP):
    """TMNF EpsilonGreedyPolicy — DISCRETE_ACTIONS and obs_spec baked in."""

    N_ACTIONS = len(DISCRETE_ACTIONS)

    def __init__(
        self,
        n_bins: int = 3,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        alpha: float = 0.1,
        gamma: float = 0.99,
        n_lidar_rays: int = 0,
    ) -> None:
        super().__init__(
            obs_spec        = TMNF_OBS_SPEC.with_lidar(n_lidar_rays),
            discrete_actions = DISCRETE_ACTIONS,
            n_bins          = n_bins,
            epsilon         = epsilon,
            epsilon_decay   = epsilon_decay,
            epsilon_min     = epsilon_min,
            alpha           = alpha,
            gamma           = gamma,
        )

    @classmethod
    def from_cfg(cls, cfg: dict, n_lidar_rays: int = 0) -> "EpsilonGreedyPolicy":
        return cls(
            n_bins        = cfg.get("n_bins",        3),
            epsilon       = cfg.get("epsilon",       1.0),
            epsilon_decay = cfg.get("epsilon_decay", 0.995),
            epsilon_min   = cfg.get("epsilon_min",   0.05),
            alpha         = cfg.get("alpha",         0.1),
            gamma         = cfg.get("gamma",         0.99),
            n_lidar_rays  = n_lidar_rays,
        )


# ---------------------------------------------------------------------------
# MCTSPolicy — TMNF variant
# ---------------------------------------------------------------------------

class MCTSPolicy(_FrameworkMCTSP):
    """TMNF MCTSPolicy — DISCRETE_ACTIONS and obs_spec baked in."""

    N_ACTIONS = len(DISCRETE_ACTIONS)

    def __init__(
        self,
        c: float = 1.41,
        alpha: float = 0.1,
        gamma: float = 0.99,
        n_bins: int = 3,
        n_lidar_rays: int = 0,
    ) -> None:
        super().__init__(
            obs_spec        = TMNF_OBS_SPEC.with_lidar(n_lidar_rays),
            discrete_actions = DISCRETE_ACTIONS,
            c               = c,
            alpha           = alpha,
            gamma           = gamma,
            n_bins          = n_bins,
        )

    @classmethod
    def from_cfg(cls, cfg: dict, n_lidar_rays: int = 0) -> "MCTSPolicy":
        return cls(
            c            = cfg.get("c",     1.41),
            alpha        = cfg.get("alpha", 0.1),
            gamma        = cfg.get("gamma", 0.99),
            n_bins       = cfg.get("n_bins", 3),
            n_lidar_rays = n_lidar_rays,
        )


# ---------------------------------------------------------------------------
# GeneticPolicy — TMNF variant
# ---------------------------------------------------------------------------

class GeneticPolicy(_FrameworkGP):
    """TMNF GeneticPolicy — obs_spec and head_names baked in."""

    def __init__(
        self,
        population_size: int = 10,
        elite_k: int = 3,
        mutation_scale: float = 0.1,
        mutation_share: float = 1.0,
        n_lidar_rays: int = 0,
    ) -> None:
        self._n_lidar_rays = n_lidar_rays
        super().__init__(
            obs_spec       = TMNF_OBS_SPEC.with_lidar(n_lidar_rays),
            head_names     = _HEAD_NAMES,
            population_size = population_size,
            elite_k        = elite_k,
            mutation_scale = mutation_scale,
            mutation_share = mutation_share,
        )

    def _make_member(self, cfg: dict) -> WeightedLinearPolicy:
        return WeightedLinearPolicy.from_cfg(cfg, self._n_lidar_rays)

    @classmethod
    def from_cfg(cls, cfg: dict, n_lidar_rays: int = 0) -> "GeneticPolicy":
        obj = cls(
            population_size = cfg.get("population_size", 10),
            elite_k         = cfg.get("elite_k", 3),
            mutation_scale  = cfg.get("mutation_scale", 0.1),
            mutation_share  = cfg.get("mutation_share", 1.0),
            n_lidar_rays    = n_lidar_rays,
        )
        champion_w = cfg.get("champion_weights")
        if champion_w:
            obj._champion        = WeightedLinearPolicy.from_cfg(champion_w, n_lidar_rays)
            obj._champion_reward = float(cfg.get("champion_reward", float("-inf")))
        return obj
