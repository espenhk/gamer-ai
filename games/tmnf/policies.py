"""TMNF-specific policies: thin registered subclasses of framework algorithms.

Each class pins the TMNF observation spec, action set, and head names into
the game-agnostic framework algorithms.  Algorithm code lives in framework/;
this module is a thin shell (~15 lines per policy).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import yaml

from framework.cmaes import CMAESPolicy as _FrameworkCMAES
from framework.dqn import DQNPolicy as _FrameworkDQN
from framework.lstm import LSTMCore as _FrameworkLSTMCore, LSTMEvolutionPolicy as _FrameworkLSTMEvo
from framework.policies import (
    BasePolicy,
    _discretize_obs,
    WeightedLinearPolicy as _FrameworkWLP,
    NeuralNetPolicy as _FrameworkNNP,
    QTablePolicy as _FrameworkQTP,
    EpsilonGreedyPolicy as _FrameworkEGP,
    MCTSPolicy as _FrameworkMCTSP,
    GeneticPolicy as _FrameworkGP,
    register_policy,
    trainer_state_path,
)
from framework.reinforce import REINFORCEPolicy as _FrameworkREINFORCE
from games.tmnf.actions import DISCRETE_ACTIONS as _DISCRETE_ACTIONS, _action_to_idx, _normalize_weight_cfg
from games.tmnf.obs_spec import BASE_OBS_DIM, TMNF_OBS_SPEC, OBS_NAMES, OBS_SCALES, obs_names_with_lidar, obs_scales_with_lidar

logger = logging.getLogger(__name__)

_N_DISCRETE_ACTIONS = len(_DISCRETE_ACTIONS)
_HEAD_NAMES = ["steer", "accel", "brake"]


# ---------------------------------------------------------------------------
# WeightedLinearPolicy — TMNF variant (steer / accel / brake heads)
# ---------------------------------------------------------------------------

class WeightedLinearPolicy(_FrameworkWLP):
    """TMNF WeightedLinearPolicy with steer/accel/brake heads and TMNF obs_spec."""

    OBS_NAMES  = OBS_NAMES
    OBS_SCALES = OBS_SCALES

    @classmethod
    def get_obs_names(cls, n_lidar_rays: int = 0) -> list[str]:
        return obs_names_with_lidar(n_lidar_rays)

    @classmethod
    def get_obs_scales(cls, n_lidar_rays: int = 0) -> np.ndarray:
        return obs_scales_with_lidar(n_lidar_rays)

    def __init__(self, weights_file: str, n_lidar_rays: int = 0) -> None:
        self._obs_spec     = TMNF_OBS_SPEC.with_lidar(n_lidar_rays)
        self._head_names   = _HEAD_NAMES
        self._weights_file = weights_file
        cfg = self._tmnf_load_or_init()
        self._apply_cfg(cfg)
        logger.info("[WeightedLinearPolicy] loaded weights from %s", weights_file)

    @classmethod
    def from_cfg(cls, cfg: dict, n_lidar_rays: int = 0, head_names: list[str] | None = None) -> "WeightedLinearPolicy":
        """Create a policy from a weights dict (not backed by a file)."""
        obs_spec   = TMNF_OBS_SPEC.with_lidar(n_lidar_rays)
        normalized = _normalize_weight_cfg(cfg, obs_spec.names)
        obj = object.__new__(cls)
        obj._weights_file = None
        obj._obs_spec     = obs_spec
        obj._head_names   = head_names if head_names is not None else _HEAD_NAMES
        obj._apply_cfg(normalized)
        return obj

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

    _OUTPUT_DIM = 3

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
            "n_lidar_rays": 0,
            "weights":      [w.tolist() for w in self._weights],
            "biases":       [b.tolist() for b in self._biases],
        }


# ---------------------------------------------------------------------------
# QTablePolicy — TMNF variant
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
            obs_spec         = TMNF_OBS_SPEC.with_lidar(n_lidar_rays),
            discrete_actions = _DISCRETE_ACTIONS,
            alpha            = alpha,
            gamma            = gamma,
            n_bins           = n_bins,
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

    N_ACTIONS = len(_DISCRETE_ACTIONS)

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
            obs_spec         = TMNF_OBS_SPEC.with_lidar(n_lidar_rays),
            discrete_actions = _DISCRETE_ACTIONS,
            n_bins           = n_bins,
            epsilon          = epsilon,
            epsilon_decay    = epsilon_decay,
            epsilon_min      = epsilon_min,
            alpha            = alpha,
            gamma            = gamma,
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

    N_ACTIONS = len(_DISCRETE_ACTIONS)

    def __init__(
        self,
        c: float = 1.41,
        alpha: float = 0.1,
        gamma: float = 0.99,
        n_bins: int = 3,
        n_lidar_rays: int = 0,
    ) -> None:
        super().__init__(
            obs_spec         = TMNF_OBS_SPEC.with_lidar(n_lidar_rays),
            discrete_actions = _DISCRETE_ACTIONS,
            c                = c,
            alpha            = alpha,
            gamma            = gamma,
            n_bins           = n_bins,
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
        eval_episodes: int = 1,
    ) -> None:
        self._n_lidar_rays = n_lidar_rays
        super().__init__(
            obs_spec        = TMNF_OBS_SPEC.with_lidar(n_lidar_rays),
            head_names      = _HEAD_NAMES,
            population_size = population_size,
            elite_k         = elite_k,
            mutation_scale  = mutation_scale,
            mutation_share  = mutation_share,
            eval_episodes   = eval_episodes,
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
            eval_episodes   = cfg.get("eval_episodes", 1),
        )
        champion_w = cfg.get("champion_weights")
        if champion_w:
            obj._champion        = WeightedLinearPolicy.from_cfg(champion_w, n_lidar_rays)
            obj._champion_reward = float(cfg.get("champion_reward", float("-inf")))
        return obj



# ---------------------------------------------------------------------------
# NeuralDQNPolicy — TMNF thin subclass of framework DQNPolicy
# ---------------------------------------------------------------------------

@register_policy
class NeuralDQNPolicy(_FrameworkDQN):
    """TMNF Deep Q-Network: 25-action discrete set, TMNF obs_spec."""

    POLICY_TYPE = "neural_dqn"
    LOOP_TYPE   = "q_learning"
    VALID_POLICY_PARAMS = frozenset({
        "hidden_sizes", "replay_buffer_size", "batch_size", "min_replay_size",
        "target_update_freq", "learning_rate", "epsilon_start", "epsilon_end",
        "epsilon_decay_steps", "gamma",
    })

    def to_cfg(self) -> dict:
        cfg = super().to_cfg()
        cfg["policy_type"] = "neural_dqn"
        return cfg

    @classmethod
    def _construct_or_resume(cls, *, obs_spec, head_names, discrete_actions,
                             weights_file, policy_params, re_initialize):
        pp = policy_params
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as _f:
                cfg = yaml.safe_load(_f)
            if isinstance(cfg, dict) and cfg.get("policy_type") == "neural_dqn":
                policy = cls.from_cfg(cfg, obs_spec, discrete_actions)
                ts = trainer_state_path(weights_file)
                if os.path.exists(ts):
                    try:
                        policy.load_trainer_state(ts)
                        logger.info("[NeuralDQNPolicy] loaded trainer state from %s", ts)
                    except (ValueError, KeyError) as exc:
                        logger.warning("[NeuralDQNPolicy] could not load trainer state — %s; "                                       "continuing with default state.", exc)
                return policy
        return cls(
            obs_spec            = obs_spec,
            discrete_actions    = discrete_actions,
            hidden_sizes        = pp.get("hidden_sizes",        [64, 64]),
            replay_buffer_size  = pp.get("replay_buffer_size",  10_000),
            batch_size          = pp.get("batch_size",          64),
            min_replay_size     = pp.get("min_replay_size",     500),
            target_update_freq  = pp.get("target_update_freq",  200),
            learning_rate       = pp.get("learning_rate",       0.001),
            epsilon_start       = pp.get("epsilon_start",       1.0),
            epsilon_end         = pp.get("epsilon_end",         0.05),
            epsilon_decay_steps = pp.get("epsilon_decay_steps", 5_000),
            gamma               = pp.get("gamma",               0.99),
        )


# ---------------------------------------------------------------------------
# CMAESPolicy — TMNF registrar; constructs framework CMAESPolicy
# ---------------------------------------------------------------------------

@register_policy
class CMAESPolicy(BasePolicy):
    """Registers 'cmaes' policy_type for TMNF; _construct_or_resume returns
    a framework.CMAESPolicy wrapping a WeightedLinearPolicy factory."""

    POLICY_TYPE = "cmaes"
    LOOP_TYPE   = "cmaes"
    VALID_POLICY_PARAMS = frozenset({"population_size", "initial_sigma", "eval_episodes"})

    def __call__(self, obs: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def to_cfg(self) -> dict:  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def _construct_or_resume(cls, *, obs_spec, head_names, discrete_actions,
                             weights_file, policy_params, re_initialize):
        pp      = policy_params
        _heads  = list(head_names)

        def _factory(flat: np.ndarray, spec) -> _FrameworkWLP:
            return _FrameworkWLP.from_cfg({}, spec, _heads).with_flat(flat)

        template = _FrameworkWLP.from_cfg({}, obs_spec, _heads)
        n_params = len(template.to_flat())

        policy = _FrameworkCMAES(
            obs_spec,
            _factory,
            n_params,
            population_size = pp.get("population_size", 20),
            initial_sigma   = pp.get("initial_sigma",   0.3),
            eval_episodes   = pp.get("eval_episodes",   1),
        )

        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as _f:
                cfg = yaml.safe_load(_f) or {}
            champion = _FrameworkWLP.from_cfg(cfg, obs_spec, _heads)
            policy.initialize_from_champion(champion)
            ts = trainer_state_path(weights_file)
            if os.path.exists(ts):
                try:
                    policy.load_trainer_state(ts)
                    logger.info("[CMAESPolicy] loaded trainer state from %s", ts)
                except (ValueError, KeyError) as exc:
                    logger.warning("[CMAESPolicy] could not load trainer state from %s — %s; "                                   "continuing with champion weights and default distribution.",
                                   ts, exc)
        else:
            policy.initialize_random()

        return policy


# ---------------------------------------------------------------------------
# REINFORCEPolicy — TMNF thin subclass of framework REINFORCEPolicy
# ---------------------------------------------------------------------------

@register_policy
class REINFORCEPolicy(_FrameworkREINFORCE):
    """TMNF REINFORCE: 25-action discrete set, TMNF obs_spec."""

    POLICY_TYPE = "reinforce"
    LOOP_TYPE   = "q_learning"
    VALID_POLICY_PARAMS = frozenset({
        "hidden_sizes", "learning_rate", "gamma", "entropy_coeff", "baseline",
    })

    @classmethod
    def _construct_or_resume(cls, *, obs_spec, head_names, discrete_actions,
                             weights_file, policy_params, re_initialize):
        pp             = policy_params
        action_decoder = lambda i: discrete_actions[i]  # noqa: E731

        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as _f:
                cfg = yaml.safe_load(_f) or {}
            if isinstance(cfg, dict) and cfg.get("policy_type") == "reinforce":
                policy = cls.from_cfg(cfg, obs_spec, action_decoder)
                ts = trainer_state_path(weights_file)
                if os.path.exists(ts):
                    try:
                        policy.load_trainer_state(ts)
                        logger.info("[REINFORCEPolicy] loaded trainer state from %s", ts)
                    except (ValueError, KeyError) as exc:
                        logger.warning("[REINFORCEPolicy] could not load trainer state — %s; "                                       "continuing with default state.", exc)
                return policy

        return cls(
            obs_spec       = obs_spec,
            action_decoder = action_decoder,
            output_dim     = len(discrete_actions),
            hidden_sizes   = pp.get("hidden_sizes",   [64, 64]),
            learning_rate  = pp.get("learning_rate",  0.001),
            gamma          = pp.get("gamma",          0.99),
            entropy_coeff  = pp.get("entropy_coeff",  0.01),
            baseline       = pp.get("baseline",       "running_mean"),
        )


# ---------------------------------------------------------------------------
# LSTMEvolutionPolicy — TMNF thin subclass of framework LSTMEvolutionPolicy
# ---------------------------------------------------------------------------

@register_policy
class LSTMEvolutionPolicy(_FrameworkLSTMEvo):
    """TMNF LSTM evolution policy: TMNF obs_spec, LSTMCore inner individual."""

    POLICY_TYPE = "lstm"
    LOOP_TYPE   = "cmaes"
    VALID_POLICY_PARAMS = frozenset({"hidden_size", "population_size", "initial_sigma"})

    @classmethod
    def _construct_or_resume(cls, *, obs_spec, head_names, discrete_actions,
                             weights_file, policy_params, re_initialize):
        pp          = policy_params
        hidden_size = pp.get("hidden_size", 32)

        policy = cls(
            obs_spec        = obs_spec,
            hidden_size     = hidden_size,
            population_size = pp.get("population_size", 20),
            initial_sigma   = pp.get("initial_sigma",   0.05),
        )

        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as _f:
                cfg = yaml.safe_load(_f) or {}
            if isinstance(cfg, dict) and cfg.get("policy_type") == "lstm":
                saved_hidden  = cfg.get("hidden_size")
                saved_obs_dim = cfg.get("obs_dim")
                if saved_hidden is not None and saved_hidden != hidden_size:
                    raise ValueError(
                        "Saved LSTM champion hidden_size does not match current run: "
                        f"saved={saved_hidden}, current={hidden_size}"
                    )
                if saved_obs_dim is not None and saved_obs_dim != obs_spec.dim:
                    raise ValueError(
                        "Saved LSTM champion obs_dim does not match current run: "
                        f"saved={saved_obs_dim}, current={obs_spec.dim}"
                    )
                champion = _FrameworkLSTMCore.from_cfg(cfg, obs_spec)
                policy.initialize_from_champion(champion)
                ts = trainer_state_path(weights_file)
                if os.path.exists(ts):
                    try:
                        policy.load_trainer_state(ts)
                        logger.info("[LSTMEvolutionPolicy] loaded trainer state from %s", ts)
                    except (ValueError, KeyError) as exc:
                        logger.warning(
                            "[LSTMEvolutionPolicy] could not load trainer state from %s — %s; "                            "continuing with champion weights and default distribution.",
                            ts, exc,
                        )

        return policy
