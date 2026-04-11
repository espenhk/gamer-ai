"""
Driving policies for TMNF.

All policies return a (3,) float32 action array: [steer, accel, brake]
  steer ∈ [-1.0, 1.0]  — maps to [-65536, 65536] in-game
  accel ∈ {0.0, 1.0}   — thresholded at 0.5 by the game client
  brake ∈ {0.0, 1.0}   — thresholded at 0.5; independent of accel

BasePolicy           — abstract base class for all policies
WeightedLinearPolicy — trainable linear policy; weights stored in YAML
NeuralNetPolicy      — small MLP policy; trained via hill-climbing
QTablePolicy         — shared base for tabular Q-learning policies
EpsilonGreedyPolicy  — Q-table with epsilon-greedy exploration
MCTSPolicy           — Q-table with UCB1 (UCT-style) action selection
GeneticPolicy        — population of WeightedLinearPolicy, evolutionary training
CMAESPolicy          — CMA-ES over flat WeightedLinearPolicy weights (Hansen 2016)
"""

from __future__ import annotations

import logging
import math
import os
from collections import deque
from abc import ABC, abstractmethod

import numpy as np
import yaml

logger = logging.getLogger(__name__)

from constants import N_ACTIONS
from obs_spec import OBS_NAMES, OBS_SCALES, obs_names_with_lidar, obs_scales_with_lidar
from steering import PDHeadingController


# ---------------------------------------------------------------------------
# Discrete action set for Q-table policies (EpsilonGreedy, MCTS)
# ---------------------------------------------------------------------------
# Each row is a (3,) action: [steer, accel, brake]
# Includes brake+accel combinations since both can be pressed simultaneously.

_DISCRETE_ACTIONS = np.array([
    [-1., 0., 1.],   #  0: brake + left
    [ 0., 0., 1.],   #  1: brake + straight
    [ 1., 0., 1.],   #  2: brake + right
    [-1., 0., 0.],   #  3: coast + left
    [ 0., 0., 0.],   #  4: coast + straight
    [ 1., 0., 0.],   #  5: coast + right
    [-1., 1., 0.],   #  6: accel + left
    [ 0., 1., 0.],   #  7: accel + straight
    [ 1., 1., 0.],   #  8: accel + right
], dtype=np.float32)
_N_DISCRETE_ACTIONS = len(_DISCRETE_ACTIONS)


def _action_to_idx(action: np.ndarray) -> int:
    """Map an action array back to its nearest index in _DISCRETE_ACTIONS."""
    diffs = np.abs(_DISCRETE_ACTIONS - action[np.newaxis, :]).sum(axis=1)
    return int(np.argmin(diffs))


def _normalize_weight_cfg(cfg: dict, names: list[str]) -> dict:
    """Return a weight config in the current steer/accel/brake format.

    Older configs may still use a single throttle head. Those are mapped to
    accel=throttle and brake=-throttle so existing saved policies and tests
    continue to load.
    """
    normalized = {
        k: v for k, v in cfg.items()
        if k not in {"steer_weights", "accel_weights", "brake_weights", "throttle_weights"}
    }

    steer_weights = dict(cfg.get("steer_weights", {}))
    if "accel_weights" in cfg or "brake_weights" in cfg:
        accel_weights = dict(cfg.get("accel_weights", {}))
        brake_weights = dict(cfg.get("brake_weights", {}))
    else:
        throttle_weights = dict(cfg.get("throttle_weights", {}))
        accel_weights = {name: float(throttle_weights.get(name, 0.0)) for name in names}
        brake_weights = {name: float(-throttle_weights.get(name, 0.0)) for name in names}

    for name in names:
        steer_weights.setdefault(name, 0.0)
        accel_weights.setdefault(name, 0.0)
        brake_weights.setdefault(name, 0.0)

    normalized["steer_weights"] = {name: float(steer_weights[name]) for name in names}
    normalized["accel_weights"] = {name: float(accel_weights[name]) for name in names}
    normalized["brake_weights"] = {name: float(brake_weights[name]) for name in names}
    return normalized


# ---------------------------------------------------------------------------
# BasePolicy
# ---------------------------------------------------------------------------

class BasePolicy(ABC):
    """Abstract base class for all driving policies."""

    @abstractmethod
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Select action given observation array.

        Returns a (3,) float32 array: [steer ∈ [-1,1], accel ∈ {0,1}, brake ∈ {0,1}]
        """

    @abstractmethod
    def to_cfg(self) -> dict:
        """Return a YAML-serializable dict representing this policy's state."""

    def update(self, obs: np.ndarray, action: np.ndarray, reward: float,
               next_obs: np.ndarray, done: bool) -> None:
        """Per-step feedback from the environment. No-op for non-online policies."""

    def on_episode_end(self) -> None:
        """Called once at the end of each episode. No-op by default."""

    def save(self, path: str) -> None:
        """Write to_cfg() to YAML at path."""
        with open(path, "w") as f:
            yaml.dump(self.to_cfg(), f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# SimplePolicy
# ---------------------------------------------------------------------------

class SimplePolicy:
    """
    Hardcoded PD+heading policy mirroring AdaptiveClient's steering formula.

    Maps obs[1] (lateral offset) and obs[3] (yaw error) to a continuous
    steering value; always accelerates without braking.

    The D term approximates lateral velocity as Δlateral_offset per tick.
    """

    LATERAL_GAIN    = 16.0   # P: steer per metre off-centre (normalised)
    DERIVATIVE_GAIN =  8.0   # D: steer per m/tick of lateral drift
    HEADING_GAIN    =  5.0   # steer per radian of heading error

    def __init__(self) -> None:
        self._prev_lateral = 0.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        lateral = obs[1]
        yaw     = obs[3]

        lateral_vel        = lateral - self._prev_lateral
        self._prev_lateral = lateral

        # Compute a signed steer value in units of "steer%" ÷ 100
        # (LATERAL_GAIN is tuned for the old steer_pct range, so divide by 100)
        steer_pct = (
            -lateral     * self.LATERAL_GAIN
            - lateral_vel  * self.DERIVATIVE_GAIN
            + yaw          * self.HEADING_GAIN
        )
        steer = float(np.clip(steer_pct / 100.0, -1.0, 1.0))
        return np.array([steer, 1.0, 0.0], dtype=np.float32)  # always accelerate


# ---------------------------------------------------------------------------
# WeightedLinearPolicy
# ---------------------------------------------------------------------------

class WeightedLinearPolicy(BasePolicy):
    """
    Linear policy with three independent output heads:

        steer_score    = dot(steer_weights,  norm_obs)  →  clipped to [-1, 1]
        accel_score    = dot(accel_weights,  norm_obs)  →  accel = 1 if score > 0
        brake_score    = dot(brake_weights,  norm_obs)  →  brake = 1 if score > 0

    accel and brake are independent — both can fire simultaneously.

    Weights are loaded from / saved to a YAML file for observability.
    Create via WeightedLinearPolicy(file) or WeightedLinearPolicy.from_cfg(dict).
    """

    # Observation names and scales are imported from obs_spec.py — one source of truth.
    OBS_NAMES  = OBS_NAMES
    OBS_SCALES = OBS_SCALES

    @classmethod
    def get_obs_names(cls, n_lidar_rays: int = 0) -> list[str]:
        return obs_names_with_lidar(n_lidar_rays)

    @classmethod
    def get_obs_scales(cls, n_lidar_rays: int = 0) -> np.ndarray:
        return obs_scales_with_lidar(n_lidar_rays)

    def __init__(self, weights_file: str, n_lidar_rays: int = 0) -> None:
        self._weights_file = weights_file
        self._n_lidar_rays = n_lidar_rays
        cfg = self._load_or_init()
        self._apply_cfg(cfg)
        logger.info("[WeightedLinearPolicy] loaded weights from %s", weights_file)

    @classmethod
    def from_cfg(cls, cfg: dict, n_lidar_rays: int = 0) -> WeightedLinearPolicy:
        """Create a policy from a weights dict (not backed by a file)."""
        obj = object.__new__(cls)
        obj._weights_file = None
        obj._n_lidar_rays = n_lidar_rays
        obj._apply_cfg(cfg)
        return obj

    # ------------------------------------------------------------------
    # Weights dict I/O
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        names = self.get_obs_names(self._n_lidar_rays)
        return {
            "steer_weights": {n: float(self._steer_w[i]) for i, n in enumerate(names)},
            "accel_weights": {n: float(self._accel_w[i]) for i, n in enumerate(names)},
            "brake_weights": {n: float(self._brake_w[i]) for i, n in enumerate(names)},
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_cfg(), f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def to_flat(self) -> np.ndarray:
        """Return [steer_weights | accel_weights | brake_weights] as one float32 vector."""
        return np.concatenate([self._steer_w, self._accel_w, self._brake_w])

    def with_flat(self, flat: np.ndarray) -> "WeightedLinearPolicy":
        """Return a new policy with weights replaced by a flat vector."""
        from obs_spec import BASE_OBS_DIM
        n = BASE_OBS_DIM + self._n_lidar_rays
        names = self.get_obs_names(self._n_lidar_rays)
        cfg = {
            "steer_weights": {names[i]: float(flat[i]) for i in range(n)},
            "accel_weights": {names[i]: float(flat[n + i]) for i in range(n)},
            "brake_weights": {names[i]: float(flat[2 * n + i]) for i in range(n)},
        }
        return WeightedLinearPolicy.from_cfg(cfg, n_lidar_rays=self._n_lidar_rays)

    def mutated(self, scale: float = 0.1, share: float = 1.0) -> "WeightedLinearPolicy":
        """Return a new policy with Gaussian perturbation applied to a random subset of weights.

        share: probability [0, 1] that each individual weight is perturbed.
               1.0 = all weights mutated (original behaviour).
        """
        rng = np.random.default_rng()
        cfg = self.to_cfg()
        for group in ("steer_weights", "accel_weights", "brake_weights"):
            for k in cfg[group]:
                if share >= 1.0 or rng.random() < share:
                    cfg[group][k] += float(rng.normal(0, scale))
        return WeightedLinearPolicy.from_cfg(cfg, n_lidar_rays=self._n_lidar_rays)

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        norm_obs    = obs / self.get_obs_scales(self._n_lidar_rays)
        steer       = float(np.clip(np.dot(self._steer_w, norm_obs), -1.0, 1.0))
        accel       = 1.0 if float(np.dot(self._accel_w, norm_obs)) > 0.0 else 0.0
        brake       = 1.0 if float(np.dot(self._brake_w, norm_obs)) > 0.0 else 0.0
        return np.array([steer, accel, brake], dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_cfg(self, cfg: dict) -> None:
        names = self.get_obs_names(self._n_lidar_rays)
        normalized = _normalize_weight_cfg(cfg, names)
        self._steer_w = np.array([normalized["steer_weights"][n] for n in names], dtype=np.float32)
        self._accel_w = np.array([normalized["accel_weights"][n] for n in names], dtype=np.float32)
        self._brake_w = np.array([normalized["brake_weights"][n] for n in names], dtype=np.float32)

    def _load_or_init(self) -> dict:
        names = self.get_obs_names(self._n_lidar_rays)
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
                logger.info("[WeightedLinearPolicy] migrated weights file → %s", self._weights_file)
            return normalized

        rng = np.random.default_rng()
        cfg = {
            "steer_weights": {n: float(rng.standard_normal()) for n in names},
            "accel_weights": {n: float(rng.standard_normal()) for n in names},
            "brake_weights": {n: float(rng.standard_normal()) for n in names},
        }
        with open(self._weights_file, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        logger.info("[WeightedLinearPolicy] initialised random weights → %s", self._weights_file)
        return cfg


# ---------------------------------------------------------------------------
# NeuralNetPolicy
# ---------------------------------------------------------------------------

class NeuralNetPolicy(BasePolicy):
    """
    Small MLP policy trained via hill-climbing (same loop as WeightedLinearPolicy).

    Architecture: obs → Linear → ReLU → ... → Linear(3) → [tanh, step, step]
    Output: [steer = tanh(out[0]), accel = 1 if out[1] > 0 else 0,
             brake = 1 if out[2] > 0 else 0]
    Pure numpy, no external ML framework required.
    Weights serialized to YAML as nested lists.
    """

    _OUTPUT_DIM = 3  # [steer, accel, brake]

    def __init__(self, hidden_sizes: list[int] | None = None, n_lidar_rays: int = 0) -> None:
        from obs_spec import BASE_OBS_DIM
        self._hidden = list(hidden_sizes or [16, 16])
        self._n_lidar_rays = n_lidar_rays
        obs_dim = BASE_OBS_DIM + n_lidar_rays
        layer_dims = [obs_dim] + self._hidden + [self._OUTPUT_DIM]
        rng = np.random.default_rng()
        self._weights: list[np.ndarray] = []
        self._biases:  list[np.ndarray] = []
        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            w = rng.standard_normal((layer_dims[i + 1], fan_in)).astype(np.float32)
            w *= np.sqrt(2.0 / fan_in)   # He init
            b = np.zeros(layer_dims[i + 1], dtype=np.float32)
            self._weights.append(w)
            self._biases.append(b)

    @classmethod
    def from_cfg(cls, cfg: dict, n_lidar_rays: int = 0) -> NeuralNetPolicy:
        obj = cls.__new__(cls)
        obj._hidden = cfg["hidden_sizes"]
        obj._n_lidar_rays = n_lidar_rays
        obj._weights = [np.array(w, dtype=np.float32) for w in cfg["weights"]]
        obj._biases  = [np.array(b, dtype=np.float32) for b in cfg["biases"]]
        return obj

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        scales = WeightedLinearPolicy.get_obs_scales(self._n_lidar_rays)
        x = (obs / scales).astype(np.float32)
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = w @ x + b
            if i < len(self._weights) - 1:
                x = np.maximum(0.0, x)   # ReLU on all but output layer
        # Output layer: [steer via tanh, accel via step, brake via step]
        steer = float(np.tanh(x[0]))
        accel = 1.0 if float(x[1]) > 0.0 else 0.0
        brake = 1.0 if float(x[2]) > 0.0 else 0.0
        return np.array([steer, accel, brake], dtype=np.float32)

    def mutated(self, scale: float = 0.1) -> NeuralNetPolicy:
        """Return a new policy with Gaussian noise added to all weights and biases."""
        rng = np.random.default_rng()
        obj = NeuralNetPolicy.__new__(NeuralNetPolicy)
        obj._hidden = self._hidden
        obj._n_lidar_rays = self._n_lidar_rays
        obj._weights = [w + rng.normal(0.0, scale, w.shape).astype(np.float32)
                        for w in self._weights]
        obj._biases  = [b + rng.normal(0.0, scale, b.shape).astype(np.float32)
                        for b in self._biases]
        return obj

    def to_cfg(self) -> dict:
        return {
            "policy_type":  "neural_net",
            "hidden_sizes": self._hidden,
            "n_lidar_rays": self._n_lidar_rays,
            "weights": [w.tolist() for w in self._weights],
            "biases":  [b.tolist() for b in self._biases],
        }


# ---------------------------------------------------------------------------
# QTablePolicy — shared base for tabular Q-learning policies
# ---------------------------------------------------------------------------

def _discretize_obs(obs: np.ndarray, scales: np.ndarray, n_bins: int) -> tuple[int, ...]:
    """
    Map a continuous observation vector to a discrete state key.

    Each feature is normalised by *scales*, clipped to [-3, 3], then
    mapped to one of *n_bins* integer buckets.  Returns a hashable tuple.
    """
    norm    = obs / scales
    clipped = np.clip(norm, -3.0, 3.0)
    bins    = ((clipped + 3.0) / 6.0 * (n_bins - 1)).astype(np.int32)
    return tuple(bins.tolist())


class QTablePolicy(BasePolicy):
    """
    Shared base for tabular Q-learning policies (EpsilonGreedy and MCTS).

    Manages the Q-table, visit counts, discretization, and Bellman updates.
    Subclasses override _select_action() to implement their exploration strategy.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        n_bins: int = 3,
        n_lidar_rays: int = 0,
    ) -> None:
        self._alpha        = alpha
        self._gamma        = gamma
        self._n_bins       = n_bins
        self._n_lidar_rays = n_lidar_rays
        self._scales       = obs_scales_with_lidar(n_lidar_rays)
        self._q_table: dict[tuple, np.ndarray] = {}
        self._n_sa:    dict[tuple, np.ndarray] = {}   # N(s, a) — visit counts
        self._n_s:     dict[tuple, int]        = {}   # N(s) = Σ N(s, a)
        self._last_obs    = None
        self._last_action = None

    def _q(self, s: tuple) -> np.ndarray:
        if s not in self._q_table:
            self._q_table[s] = np.zeros(N_ACTIONS, dtype=np.float32)
        return self._q_table[s]

    def _n(self, s: tuple) -> np.ndarray:
        if s not in self._n_sa:
            self._n_sa[s] = np.zeros(N_ACTIONS, dtype=np.float32)
        return self._n_sa[s]

    @abstractmethod
    def _select_action(self, s: tuple) -> int:
        """Choose an action for state key s. Implemented by subclasses."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        s = _discretize_obs(obs, self._scales, self._n_bins)
        self._last_obs = obs
        action_idx = self._select_action(s)
        self._last_action = action_idx
        return _DISCRETE_ACTIONS[action_idx].copy()

    def update(self, obs: np.ndarray, action: int, reward: float,
               next_obs: np.ndarray, done: bool) -> None:
        s  = _discretize_obs(obs,      self._scales, self._n_bins)
        s_ = _discretize_obs(next_obs, self._scales, self._n_bins)
        q_next = 0.0 if done else float(np.max(self._q(s_)))
        td = reward + self._gamma * q_next - self._q(s)[action]
        self._q(s)[action] += self._alpha * td
        self._n(s)[action] += 1.0
        self._n_s[s] = self._n_s.get(s, 0) + 1

    def on_episode_end(self) -> None:
        self._last_obs    = None
        self._last_action = None

    @property
    def n_states_visited(self) -> int:
        return len(self._q_table)


# ---------------------------------------------------------------------------
# EpsilonGreedyPolicy
# ---------------------------------------------------------------------------

class EpsilonGreedyPolicy(QTablePolicy):
    """
    Tabular Q-learning with epsilon-greedy exploration.

    State space is the observation vector discretized into n_bins buckets per
    feature (default 3).  Q-values are updated online via the Bellman equation
    after every environment step.  Epsilon decays each episode.

    Actions are selected from _DISCRETE_ACTIONS (9 entries).

    Note: the Q-table is NOT persisted to policy_weights.yaml (it can be very
    large).  to_cfg() records only hyperparameters; the table lives in memory
    for the duration of a training run.
    """

    N_ACTIONS = _N_DISCRETE_ACTIONS

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
        super().__init__(alpha=alpha, gamma=gamma, n_bins=n_bins, n_lidar_rays=n_lidar_rays)
        self._epsilon       = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min   = epsilon_min
        self._alpha         = alpha
        self._gamma         = gamma
        self._n_lidar_rays  = n_lidar_rays
        self._scales        = WeightedLinearPolicy.get_obs_scales(n_lidar_rays)
        self._q_table: dict[tuple, np.ndarray] = {}

    @classmethod
    def from_cfg(cls, cfg: dict, n_lidar_rays: int = 0) -> EpsilonGreedyPolicy:
        return cls(
            n_bins        = cfg.get("n_bins",        3),
            epsilon       = cfg.get("epsilon",       1.0),
            epsilon_decay = cfg.get("epsilon_decay", 0.995),
            epsilon_min   = cfg.get("epsilon_min",   0.05),
            alpha         = cfg.get("alpha",         0.1),
            gamma         = cfg.get("gamma",         0.99),
            n_lidar_rays  = n_lidar_rays,
        )

    def _q(self, state_key: tuple) -> np.ndarray:
        if state_key not in self._q_table:
            self._q_table[state_key] = np.zeros(self.N_ACTIONS, dtype=np.float32)
        return self._q_table[state_key]

    def _select_action(self, state_key: tuple) -> int:
        if np.random.random() < self._epsilon:
            return int(np.random.randint(self.N_ACTIONS))
        return int(np.argmax(self._q(state_key)))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        state_key = _discretize_obs(obs, self._scales, self._n_bins)
        self._last_obs = obs
        action_idx = self._select_action(state_key)
        self._last_action = action_idx
        return _DISCRETE_ACTIONS[action_idx].copy()

    def update(self, obs: np.ndarray, action: np.ndarray | int, reward: float,
               next_obs: np.ndarray, done: bool) -> None:
        action_idx = int(action) if np.isscalar(action) else _action_to_idx(action)
        s  = _discretize_obs(obs, self._scales, self._n_bins)
        s_ = _discretize_obs(next_obs, self._scales, self._n_bins)
        q_next  = 0.0 if done else float(np.max(self._q(s_)))
        td      = reward + self._gamma * q_next - self._q(s)[action_idx]
        self._q(s)[action_idx] += self._alpha * td

    def on_episode_end(self) -> None:
        super().on_episode_end()
        self._epsilon = max(self._epsilon_min,
                            self._epsilon * self._epsilon_decay)

    def to_cfg(self) -> dict:
        return {
            "policy_type":      "epsilon_greedy",
            "n_bins":           self._n_bins,
            "epsilon":          float(self._epsilon),
            "epsilon_decay":    float(self._epsilon_decay),
            "epsilon_min":      float(self._epsilon_min),
            "alpha":            float(self._alpha),
            "gamma":            float(self._gamma),
            "n_states_visited": self.n_states_visited,
        }


# ---------------------------------------------------------------------------
# MCTSPolicy  (UCT-style online learner)
# ---------------------------------------------------------------------------

class MCTSPolicy(QTablePolicy):
    """
    UCT-inspired online Q-learner.

    Action selection uses the UCB1 formula:
        score(s, a) = Q(s, a) + c * sqrt(ln(N(s) + 1) / (N(s, a) + 1e-8))

    where N(s, a) is the visit count for (state, action) and N(s) = Σ_a N(s, a).

    Actions are selected from _DISCRETE_ACTIONS (9 entries).

    NOTE: True Monte Carlo Tree Search requires cloning the environment state,
    which is not possible with TMInterface.  This is a UCT-style approximation
    that builds value/count tables incrementally over real episodes.
    """

    N_ACTIONS = _N_DISCRETE_ACTIONS

    def __init__(
        self,
        c: float = 1.41,
        alpha: float = 0.1,
        gamma: float = 0.99,
        n_bins: int = 3,
        n_lidar_rays: int = 0,
    ) -> None:
        super().__init__(alpha=alpha, gamma=gamma, n_bins=n_bins, n_lidar_rays=n_lidar_rays)
        self._c = c

    @classmethod
    def from_cfg(cls, cfg: dict, n_lidar_rays: int = 0) -> MCTSPolicy:
        return cls(
            c            = cfg.get("c",     1.41),
            alpha        = cfg.get("alpha", 0.1),
            gamma        = cfg.get("gamma", 0.99),
            n_bins       = cfg.get("n_bins", 3),
            n_lidar_rays = n_lidar_rays,
        )

    def _q(self, s: tuple) -> np.ndarray:
        if s not in self._q_table:
            self._q_table[s] = np.zeros(self.N_ACTIONS, dtype=np.float32)
        return self._q_table[s]

    def _n(self, s: tuple) -> np.ndarray:
        if s not in self._n_sa:
            self._n_sa[s] = np.zeros(self.N_ACTIONS, dtype=np.float32)
        return self._n_sa[s]

    def _select_action(self, s: tuple) -> int:
        n_s = self._n_s.get(s, 0)
        if n_s == 0:
            return int(np.random.randint(self.N_ACTIONS))
        ucb = self._q(s) + self._c * np.sqrt(
            math.log(n_s + 1) / (self._n(s) + 1e-8)
        )
        return int(np.argmax(ucb))

    def update(self, obs: np.ndarray, action: np.ndarray | int, reward: float,
               next_obs: np.ndarray, done: bool) -> None:
        action_idx = int(action) if np.isscalar(action) else _action_to_idx(action)
        s  = _discretize_obs(obs, self._scales, self._n_bins)
        s_ = _discretize_obs(next_obs, self._scales, self._n_bins)
        q_next  = 0.0 if done else float(np.max(self._q(s_)))
        td      = reward + self._gamma * q_next - self._q(s)[action_idx]
        self._q(s)[action_idx]  += self._alpha * td
        self._n(s)[action_idx]  += 1.0
        self._n_s[s]             = self._n_s.get(s, 0) + 1

    def on_episode_end(self) -> None:
        pass

    def to_cfg(self) -> dict:
        return {
            "policy_type":      "mcts",
            "c":                float(self._c),
            "alpha":            float(self._alpha),
            "gamma":            float(self._gamma),
            "n_bins":           self._n_bins,
            "n_states_visited": self.n_states_visited,
        }


# ---------------------------------------------------------------------------
# GeneticPolicy
# ---------------------------------------------------------------------------

class GeneticPolicy(BasePolicy):
    """
    Evolutionary policy: a population of WeightedLinearPolicy instances.

    Each training generation:
      1. Evaluate all population members (one episode each).
      2. Select the top `elite_k` individuals (elites survive unchanged).
      3. Fill the rest via uniform crossover between two random elites + mutation.
      4. Update the champion if any individual beats the previous best.

    Inference always uses the champion (best individual seen so far).
    `save()` writes the champion in WeightedLinearPolicy YAML format so
    existing analytics (weight heatmaps, etc.) work without changes.
    """

    def __init__(
        self,
        population_size: int = 10,
        elite_k: int = 3,
        mutation_scale: float = 0.1,
        mutation_share: float = 1.0,
        n_lidar_rays: int = 0,
    ) -> None:
        self._pop_size       = population_size
        self._elite_k        = min(elite_k, population_size)
        self._mutation_scale = mutation_scale
        self._mutation_share = mutation_share
        self._n_lidar_rays   = n_lidar_rays
        self._population: list[WeightedLinearPolicy] = []
        self._champion: WeightedLinearPolicy | None = None
        self._champion_reward: float = float("-inf")

    @classmethod
    def from_cfg(cls, cfg: dict, n_lidar_rays: int = 0) -> GeneticPolicy:
        obj = cls(
            population_size = cfg.get("population_size", 10),
            elite_k         = cfg.get("elite_k", 3),
            mutation_scale  = cfg.get("mutation_scale", 0.1),
            mutation_share  = cfg.get("mutation_share", 1.0),
            n_lidar_rays    = n_lidar_rays,
        )
        champion_w = cfg.get("champion_weights")
        if champion_w:
            obj._champion = WeightedLinearPolicy.from_cfg(champion_w, n_lidar_rays)
            obj._champion_reward = float(cfg.get("champion_reward", float("-inf")))
        return obj

    @property
    def population(self) -> list[WeightedLinearPolicy]:
        return self._population

    @property
    def champion_reward(self) -> float:
        return self._champion_reward

    def initialize_random(self) -> None:
        """Build a fresh random population."""
        rng = np.random.default_rng()
        names = obs_names_with_lidar(self._n_lidar_rays)
        pop = []
        for _ in range(self._pop_size):
            cfg = {
                "steer_weights": {n: float(rng.standard_normal()) for n in names},
                "accel_weights": {n: float(rng.standard_normal()) for n in names},
                "brake_weights": {n: float(rng.standard_normal()) for n in names},
            }
            pop.append(WeightedLinearPolicy.from_cfg(cfg, self._n_lidar_rays))
        self._population = pop
        if self._champion is None:
            self._champion = pop[0]

    def initialize_from_champion(self, champion: WeightedLinearPolicy) -> None:
        """Seed the population by mutating the given champion."""
        self._champion = champion
        self._population = [
            champion.mutated(self._mutation_scale, self._mutation_share)
            for _ in range(self._pop_size)
        ]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        assert self._champion is not None, "GeneticPolicy: champion not set — call initialize_*() first"
        return self._champion(obs)

    def evaluate_and_evolve(self, rewards: list[float]) -> bool:
        """
        Update population based on episode rewards.

        Args:
            rewards: episode reward[i] for self._population[i].

        Returns:
            True if the champion was updated this generation.
        """
        assert len(rewards) == len(self._population)
        ranked  = sorted(zip(rewards, self._population), key=lambda x: -x[0])
        improved = False

        if ranked[0][0] > self._champion_reward:
            self._champion_reward = ranked[0][0]
            self._champion        = ranked[0][1]
            improved              = True

        elites  = [ind for _, ind in ranked[:self._elite_k]]
        new_pop = list(elites)
        rng_idx = np.random.default_rng()

        while len(new_pop) < self._pop_size:
            i1 = int(rng_idx.integers(self._elite_k))
            i2 = int(rng_idx.integers(self._elite_k))
            child_cfg = self._crossover(elites[i1].to_cfg(), elites[i2].to_cfg())
            child     = WeightedLinearPolicy.from_cfg(child_cfg, self._n_lidar_rays)
            new_pop.append(child.mutated(self._mutation_scale, self._mutation_share))

        self._population = new_pop
        return improved

    @staticmethod
    def _crossover(cfg1: dict, cfg2: dict) -> dict:
        """Uniform weight crossover: each weight is randomly drawn from parent 1 or 2."""
        names = list(cfg1.get("steer_weights", {}).keys())
        cfg1 = _normalize_weight_cfg(cfg1, names)
        cfg2 = _normalize_weight_cfg(cfg2, names)
        result = {
            "steer_weights": {},
            "accel_weights": {},
            "brake_weights": {},
        }
        for group in ("steer_weights", "accel_weights", "brake_weights"):
            for k in cfg1[group]:
                result[group][k] = (cfg1[group][k] if np.random.random() < 0.5
                                    else cfg2[group][k])
        return result

    def to_cfg(self) -> dict:
        return {
            "policy_type":      "genetic",
            "population_size":  self._pop_size,
            "elite_k":          self._elite_k,
            "mutation_scale":   float(self._mutation_scale),
            "mutation_share":   float(self._mutation_share),
            "champion_reward":  float(self._champion_reward),
            "champion_weights": self._champion.to_cfg() if self._champion else {},
        }

    def save(self, path: str) -> None:
        """Save champion in WeightedLinearPolicy YAML format for analytics compatibility."""
        if self._champion is not None:
            self._champion.save(path)


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size circular buffer of (obs, action_idx, reward, next_obs, done) tuples."""

    def __init__(self, maxlen: int) -> None:
        self._buf: deque = deque(maxlen=maxlen)

    def push(self, obs: np.ndarray, action_idx: int, reward: float,
             next_obs: np.ndarray, done: bool) -> None:
        self._buf.append((obs.copy(), int(action_idx), float(reward), next_obs.copy(), bool(done)))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        replace = batch_size > len(self._buf)
        idxs  = np.random.choice(len(self._buf), size=batch_size, replace=replace)
        batch = [self._buf[i] for i in idxs]
        obs_b  = np.stack([t[0] for t in batch]).astype(np.float32)
        act_b  = np.array([t[1] for t in batch], dtype=np.int32)
        rew_b  = np.array([t[2] for t in batch], dtype=np.float32)
        next_b = np.stack([t[3] for t in batch]).astype(np.float32)
        done_b = np.array([t[4] for t in batch], dtype=np.float32)
        return obs_b, act_b, rew_b, next_b, done_b

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# NeuralDQNPolicy
# ---------------------------------------------------------------------------

class NeuralDQNPolicy(BasePolicy):
    """
    DQN over the 9-action discrete action set.

    Online network:  Q(s, a; θ_online)
    Target network:  Q(s, a; θ_target)  — synced every target_update_freq gradient steps
    Replay buffer:   circular buffer of (s, a_idx, r, s', done)

    Architecture: obs → Linear → ReLU → ... → Linear(9)
    Pure numpy with Adam optimiser — no external ML framework required.
    Epsilon decays linearly from epsilon_start → epsilon_end over epsilon_decay_steps steps.
# CMAESPolicy
# ---------------------------------------------------------------------------

class CMAESPolicy(BasePolicy):
    """
    CMA-ES over the flat weight vector of a WeightedLinearPolicy.
    Uses the (μ/μ_w, λ)-CMA-ES algorithm (Hansen 2016).

    Each generation:
      1. sample_population() — draw λ offspring from N(mean, σ²·C)
      2. Evaluate each offspring for one episode
      3. update_distribution(rewards) — update mean, σ, C, and evolution paths

    Inference always uses the champion (best individual seen so far).
    save() writes the champion in WeightedLinearPolicy YAML format so
    existing analytics (weight heatmaps, etc.) work without changes.
    """

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
        min_replay_size: int = 500,
        target_update_freq: int = 200,
        learning_rate: float = 0.001,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5000,
        gamma: float = 0.99,
        n_lidar_rays: int = 0,
    ) -> None:
        from obs_spec import BASE_OBS_DIM
        self._hidden       = list(hidden_sizes or [64, 64])
        self._buf_maxlen   = int(replay_buffer_size)
        self._batch_size   = int(batch_size)
        self._min_replay   = int(min_replay_size)
        self._target_freq  = int(target_update_freq)
        self._lr           = float(learning_rate)
        self._eps_start    = float(epsilon_start)
        self._eps          = float(epsilon_start)
        self._eps_end      = float(epsilon_end)
        self._eps_steps    = int(epsilon_decay_steps)
        self._eps_delta    = (float(epsilon_start) - float(epsilon_end)) / max(1, int(epsilon_decay_steps))
        self._gamma        = float(gamma)
        self._n_lidar_rays = n_lidar_rays
        self._obs_dim      = BASE_OBS_DIM + n_lidar_rays
        self._scales       = obs_scales_with_lidar(n_lidar_rays)

        self._replay      = ReplayBuffer(replay_buffer_size)
        self._total_steps = 0   # transitions pushed (drives epsilon schedule)
        self._grad_steps  = 0   # gradient updates (drives target sync)

        self._online = self._build_net()
        self._target = self._build_net()
        self._sync_target()

        # Adam first/second moments — one entry per layer
        self._m_w = [np.zeros_like(w) for w in self._online["weights"]]
        self._m_b = [np.zeros_like(b) for b in self._online["biases"]]
        self._v_w = [np.zeros_like(w) for w in self._online["weights"]]
        self._v_b = [np.zeros_like(b) for b in self._online["biases"]]
        self._adam_t = 0

    @classmethod
    def from_cfg(cls, cfg: dict, n_lidar_rays: int = 0) -> NeuralDQNPolicy:
        obj = cls(
            hidden_sizes        = cfg.get("hidden_sizes",        [64, 64]),
            replay_buffer_size  = cfg.get("replay_buffer_size",  10000),
            batch_size          = cfg.get("batch_size",          64),
            min_replay_size     = cfg.get("min_replay_size",     500),
            target_update_freq  = cfg.get("target_update_freq",  200),
            learning_rate       = cfg.get("learning_rate",       0.001),
            epsilon_start       = cfg.get("epsilon_start",       1.0),
            epsilon_end         = cfg.get("epsilon_end",         0.05),
            epsilon_decay_steps = cfg.get("epsilon_decay_steps", 5000),
            gamma               = cfg.get("gamma",               0.99),
            n_lidar_rays        = n_lidar_rays,
        )
        if "online_weights" in cfg:
            required = ["online_weights", "online_biases", "target_weights", "target_biases"]
            missing  = [k for k in required if k not in cfg]
            if missing:
                raise KeyError(f"NeuralDQNPolicy.from_cfg: missing keys {missing}")

            loaded_w = [np.array(w, dtype=np.float32) for w in cfg["online_weights"]]
            # Validate that the first layer's input dim matches obs_dim
            if loaded_w[0].shape[1] != obj._obs_dim:
                raise ValueError(
                    f"NeuralDQNPolicy.from_cfg: weight shape mismatch — "
                    f"first layer expects input dim {loaded_w[0].shape[1]} "
                    f"but obs_dim is {obj._obs_dim} (n_lidar_rays={n_lidar_rays})"
                )

            obj._online["weights"] = loaded_w
            obj._online["biases"]  = [np.array(b, dtype=np.float32) for b in cfg["online_biases"]]
            obj._target["weights"] = [np.array(w, dtype=np.float32) for w in cfg["target_weights"]]
            obj._target["biases"]  = [np.array(b, dtype=np.float32) for b in cfg["target_biases"]]
            obj._eps         = float(cfg.get("epsilon",      obj._eps_end))
            obj._total_steps = int(cfg.get("total_steps",   0))
            obj._grad_steps  = int(cfg.get("grad_steps",    0))
            # Re-init Adam moments for the restored weights
            obj._m_w = [np.zeros_like(w) for w in obj._online["weights"]]
            obj._m_b = [np.zeros_like(b) for b in obj._online["biases"]]
            obj._v_w = [np.zeros_like(w) for w in obj._online["weights"]]
            obj._v_b = [np.zeros_like(b) for b in obj._online["biases"]]
            logger.info("[NeuralDQNPolicy] loaded weights from cfg (eps=%.4f, steps=%d)",
                        obj._eps, obj._total_steps)
        return obj

    # ------------------------------------------------------------------
    # Network construction and sync
    # ------------------------------------------------------------------

    def _build_net(self) -> dict:
        """He-initialised MLP: weights and biases as lists of float32 arrays."""
        rng  = np.random.default_rng()
        dims = [self._obs_dim] + self._hidden + [_N_DISCRETE_ACTIONS]
        weights, biases = [], []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            w = rng.standard_normal((dims[i + 1], fan_in)).astype(np.float32)
            w *= np.sqrt(2.0 / fan_in)
            b = np.zeros(dims[i + 1], dtype=np.float32)
            weights.append(w)
            biases.append(b)
        return {"weights": weights, "biases": biases}

    def _sync_target(self) -> None:
        """Copy online network weights → target network (hard update)."""
        self._target["weights"] = [w.copy() for w in self._online["weights"]]
        self._target["biases"]  = [b.copy() for b in self._online["biases"]]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(
        self, net: dict, x: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Forward pass through *net*.

        x: (B, obs_dim) or (obs_dim,)
        Returns:
          q        — (B, n_actions) or (n_actions,) Q-values
          inputs   — layer_inputs[i] is the (post-ReLU) input fed into layer i
          pre_relu — pre_relu[i] is the linear output of hidden layer i (before ReLU)
        """
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]

        h: np.ndarray = x.astype(np.float32)
        layer_inputs: list[np.ndarray] = []
        pre_relu:     list[np.ndarray] = []

        for i, (w, b) in enumerate(zip(net["weights"], net["biases"])):
            layer_inputs.append(h)
            z = h @ w.T + b
            if i < len(net["weights"]) - 1:   # hidden layer
                pre_relu.append(z)
                h = np.maximum(0.0, z)
            else:                              # output layer — linear
                h = z

        return (h[0] if single else h), layer_inputs, pre_relu

    def _q_values(self, net: dict, obs_norm: np.ndarray) -> np.ndarray:
        """Normalised obs → Q-value array (no bookkeeping)."""
        q, _, _ = self._forward(net, obs_norm)
        return q

    # ------------------------------------------------------------------
    # Gradient update
    # ------------------------------------------------------------------

    def _gradient_step(
        self,
        obs_b: np.ndarray, act_b: np.ndarray,
        rew_b: np.ndarray, next_b: np.ndarray, done_b: np.ndarray,
    ) -> None:
        """One Adam step on the online network using a sampled minibatch."""
        obs_norm  = obs_b  / self._scales
        next_norm = next_b / self._scales
        B = len(act_b)

        # DQN targets: y = r + γ * max_a Q_target(s') * (1 - done)
        q_next   = self._q_values(self._target, next_norm)        # (B, 9)
        targets  = rew_b + self._gamma * np.max(q_next, axis=1) * (1.0 - done_b)  # (B,)

        # Online forward (save intermediate values for backprop)
        q_all, layer_inputs, pre_relu = self._forward(self._online, obs_norm)  # (B, 9)

        # Loss gradient: 2*(Q(s,a) - y) / B, only for the taken action
        grad_out = np.zeros_like(q_all)                           # (B, 9)
        grad_out[np.arange(B), act_b] = (
            2.0 * (q_all[np.arange(B), act_b] - targets) / B
        )

        # Backprop through layers (reverse order)
        g = grad_out
        grad_params: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(self._online["weights"]) - 1, -1, -1):
            a_in = layer_inputs[i]                 # (B, in_dim) — post-ReLU input
            dW   = g.T @ a_in                      # (out_dim, in_dim)
            db   = g.sum(axis=0)                   # (out_dim,)
            grad_params.append((dW, db))
            if i > 0:
                # Propagate gradient through weights and apply ReLU mask
                g = (g @ self._online["weights"][i]) * (pre_relu[i - 1] > 0)
        grad_params.reverse()

        # Adam update
        self._adam_t += 1
        t          = self._adam_t
        b1, b2     = 0.9, 0.999
        eps_adam   = 1e-8

        for i, (dW, db) in enumerate(grad_params):
            self._m_w[i] = b1 * self._m_w[i] + (1.0 - b1) * dW
            self._v_w[i] = b2 * self._v_w[i] + (1.0 - b2) * dW ** 2
            mw_hat = self._m_w[i] / (1.0 - b1 ** t)
            vw_hat = self._v_w[i] / (1.0 - b2 ** t)
            self._online["weights"][i] -= self._lr * mw_hat / (np.sqrt(vw_hat) + eps_adam)

            self._m_b[i] = b1 * self._m_b[i] + (1.0 - b1) * db
            self._v_b[i] = b2 * self._v_b[i] + (1.0 - b2) * db ** 2
            mb_hat = self._m_b[i] / (1.0 - b1 ** t)
            vb_hat = self._v_b[i] / (1.0 - b2 ** t)
            self._online["biases"][i] -= self._lr * mb_hat / (np.sqrt(vb_hat) + eps_adam)

        self._grad_steps += 1
        if self._grad_steps % self._target_freq == 0:
            self._sync_target()
            logger.debug("[NeuralDQNPolicy] target network synced at grad_step %d", self._grad_steps)
        population_size: int = 20,
        initial_sigma: float = 0.3,
        n_lidar_rays: int = 0,
        seed: int | None = None,
    ) -> None:
        from obs_spec import BASE_OBS_DIM
        self._lam          = population_size
        self._n_lidar_rays = n_lidar_rays
        n                  = (BASE_OBS_DIM + n_lidar_rays) * 3   # steer + accel + brake heads
        self._n            = n

        # Recombination weights (elite half, log-based, normalised)
        mu            = self._lam // 2
        self._mu      = mu
        raw_w         = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)],
                                 dtype=np.float64)
        self._weights = raw_w / raw_w.sum()
        self._mu_eff  = 1.0 / float(np.sum(self._weights ** 2))

        # Step-size adaptation constants (Hansen 2016, §3)
        self._cs   = (self._mu_eff + 2) / (n + self._mu_eff + 5)
        self._ds   = (1 + 2 * max(0.0, float(np.sqrt((self._mu_eff - 1) / (n + 1))) - 1)
                      + self._cs)
        self._chin = float(np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2)))

        # Covariance adaptation constants (Hansen 2016, §3)
        self._cc  = (4 + self._mu_eff / n) / (n + 4 + 2 * self._mu_eff / n)
        self._c1  = 2.0 / ((n + 1.3) ** 2 + self._mu_eff)
        self._cmu = min(
            1.0 - self._c1,
            2.0 * (self._mu_eff - 2 + 1.0 / self._mu_eff) / ((n + 2) ** 2 + self._mu_eff),
        )

        # Shared RNG — seeding makes sampling reproducible (useful for tests)
        self._rng = np.random.default_rng(seed)

        # Distribution state (float64 for numerical stability)
        self._mean      = self._rng.standard_normal(n).astype(np.float64)
        self._sigma     = float(initial_sigma)
        self._ps        = np.zeros(n, dtype=np.float64)   # step-size evolution path
        self._pc        = np.zeros(n, dtype=np.float64)   # covariance evolution path
        self._C         = np.eye(n, dtype=np.float64)     # covariance matrix
        self._B         = np.eye(n, dtype=np.float64)     # eigenvectors of C
        self._D         = np.ones(n, dtype=np.float64)    # sqrt(eigenvalues) of C
        self._invsqrtC  = np.eye(n, dtype=np.float64)     # C^{-1/2}
        self._eigengen  = 0    # generation of last eigendecomposition
        self._gen       = 0    # current generation counter

        # Sampling buffer (filled by sample_population, consumed by update_distribution)
        self._pop_xs: list[np.ndarray] = []
        self._pop_ys: list[np.ndarray] = []

        # Champion
        self._champion: WeightedLinearPolicy | None = None
        self._champion_reward: float = float("-inf")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def champion_reward(self) -> float:
        return self._champion_reward

    @property
    def population_size(self) -> int:
        """Number of offspring sampled each generation (λ)."""
        return self._lam

    @property
    def sigma(self) -> float:
        """Current step size σ (adapts each generation)."""
        return self._sigma

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_random(self) -> None:
        """Initialise the search mean at zero; CMA-ES adapts scale via σ."""
        self._mean = np.zeros(self._n, dtype=np.float64)
        logger.info("[CMAESPolicy] initialised with zero mean, σ=%.3f", self._sigma)

    def initialize_from_champion(self, champion: WeightedLinearPolicy) -> None:
        """Seed the search mean from an existing champion's flat weight vector."""
        self._champion = champion

        seeded_reward = None
        for attr_name in ("champion_reward", "reward"):
            reward_value = getattr(champion, attr_name, None)
            if reward_value is not None:
                try:
                    seeded_reward = float(reward_value)
                except (TypeError, ValueError):
                    seeded_reward = None
                else:
                    if math.isfinite(seeded_reward):
                        break
                    seeded_reward = None

        if seeded_reward is None and math.isfinite(self._champion_reward):
            seeded_reward = float(self._champion_reward)

        self._champion_reward = seeded_reward if seeded_reward is not None else float("-inf")
        self._mean = champion.to_flat().astype(np.float64)
        logger.info(
            "[CMAESPolicy] seeded mean from champion%s",
            "" if seeded_reward is None else f" (baseline reward={self._champion_reward:.6f})",
        )
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flat_to_policy(self, flat: np.ndarray) -> WeightedLinearPolicy:
        """Build a WeightedLinearPolicy from a flat [steer|accel|brake] weight vector."""
        names  = obs_names_with_lidar(self._n_lidar_rays)
        obs_n  = len(names)
        cfg = {
            "steer_weights": {names[i]: float(flat[i])             for i in range(obs_n)},
            "accel_weights": {names[i]: float(flat[obs_n + i])     for i in range(obs_n)},
            "brake_weights": {names[i]: float(flat[2 * obs_n + i]) for i in range(obs_n)},
        }
        return WeightedLinearPolicy.from_cfg(cfg, n_lidar_rays=self._n_lidar_rays)

    def _update_eigen(self) -> None:
        """Eigendecompose C and refresh B, D, invsqrtC."""
        self._C = np.triu(self._C) + np.triu(self._C, 1).T     # enforce symmetry
        eigvals, self._B = np.linalg.eigh(self._C)
        eigvals          = np.maximum(eigvals, 1e-20)           # clamp negatives
        self._D          = np.sqrt(eigvals)
        self._invsqrtC   = self._B @ np.diag(1.0 / self._D) @ self._B.T
        self._eigengen   = self._gen

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    def sample_population(self) -> list[WeightedLinearPolicy]:
        """Sample λ offspring from N(mean, σ²·C).

        Returns a list of WeightedLinearPolicy instances ready for evaluation.
        Must be followed by update_distribution(rewards) with the same ordering.
        """
        n = self._n

        # Refresh eigendecomposition every generation (λ/(10n) < 1 for typical dims)
        if self._gen - self._eigengen >= max(1, self._lam // max(1, 10 * n)):
            self._update_eigen()

        self._pop_xs = []
        self._pop_ys = []
        for _ in range(self._lam):
            z = self._rng.standard_normal(n)
            y = self._B @ (self._D * z)        # sample from N(0, C)
            x = self._mean + self._sigma * y
            self._pop_xs.append(x)
            self._pop_ys.append(y)

        return [self._flat_to_policy(x) for x in self._pop_xs]

    def update_distribution(self, rewards: list[float]) -> bool:
        """Apply (μ/μ_w, λ)-CMA-ES update given per-offspring episode rewards.

        Updates mean, σ, p_σ, p_c, and C in place.

        Returns:
            True if the all-time champion was improved this generation.
        """
        if len(rewards) != self._lam:
            raise ValueError(f"Expected {self._lam} rewards, got {len(rewards)}")
        if len(self._pop_xs) != self._lam or len(self._pop_ys) != self._lam:
            raise RuntimeError(
                "update_distribution() called before a matching sample_population(). "
                f"Expected {self._lam} samples in _pop_xs/_pop_ys, "
                f"got {len(self._pop_xs)}/{len(self._pop_ys)}. "
                "Call sample_population() first."
            )
        n = self._n

        # Rank offspring descending by reward
        order = np.argsort(rewards)[::-1]

        # Update champion
        improved = False
        best_r   = rewards[order[0]]
        if best_r > self._champion_reward:
            self._champion_reward = best_r
            self._champion        = self._flat_to_policy(self._pop_xs[order[0]])
            improved              = True

        # Weighted average of elite steps y_i = (x_i − mean) / σ
        elite_ys = np.stack([self._pop_ys[order[i]] for i in range(self._mu)])  # (mu, n)
        step     = np.einsum("i,ij->j", self._weights, elite_ys)                # (n,)

        # Mean update
        self._mean = self._mean + self._sigma * step

        # Step-size evolution path p_σ  (Hansen 2016, eq. 43)
        ps_scale  = float(np.sqrt(self._cs * (2 - self._cs) * self._mu_eff))
        self._ps  = (1 - self._cs) * self._ps + ps_scale * (self._invsqrtC @ step)

        # Step-size update σ  (Hansen 2016, eq. 44)
        ps_norm     = float(np.linalg.norm(self._ps))
        self._sigma = float(np.clip(
            self._sigma * np.exp((self._cs / self._ds) * (ps_norm / self._chin - 1)),
            1e-10, 1e6,
        ))

        # h_σ stall indicator
        ps_norm_normed = ps_norm / float(np.sqrt(1 - (1 - self._cs) ** (2 * (self._gen + 1))))
        h_sigma = 1.0 if ps_norm_normed < (1.4 + 2.0 / (n + 1)) * self._chin else 0.0

        # Covariance evolution path p_c  (Hansen 2016, eq. 45)
        pc_scale  = float(np.sqrt(self._cc * (2 - self._cc) * self._mu_eff))
        self._pc  = (1 - self._cc) * self._pc + h_sigma * pc_scale * step

        # Covariance update  (Hansen 2016, eq. 47)
        delta_h = (1 - h_sigma) * self._cc * (2 - self._cc)
        rank1   = np.outer(self._pc, self._pc)
        rank_mu = np.einsum("i,ij,ik->jk", self._weights, elite_ys, elite_ys)
        self._C = (
            (1 - self._c1 - self._cmu) * self._C
            + self._c1 * (rank1 + delta_h * self._C)
            + self._cmu * rank_mu
        )

        self._gen += 1
        return improved

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if np.random.random() < self._eps:
            return _DISCRETE_ACTIONS[np.random.randint(_N_DISCRETE_ACTIONS)].copy()
        obs_norm = (obs / self._scales).astype(np.float32)
        q        = self._q_values(self._online, obs_norm)
        return _DISCRETE_ACTIONS[int(np.argmax(q))].copy()

    def update(self, obs: np.ndarray, action: np.ndarray | int, reward: float,
               next_obs: np.ndarray, done: bool) -> None:
        action_idx = int(action) if np.isscalar(action) else _action_to_idx(action)
        self._replay.push(obs, action_idx, reward, next_obs, done)
        self._total_steps += 1
        # Linear epsilon decay per step
        self._eps = max(self._eps_end, self._eps - self._eps_delta)

        if len(self._replay) >= self._min_replay:
            obs_b, act_b, rew_b, next_b, done_b = self._replay.sample(self._batch_size)
            self._gradient_step(obs_b, act_b, rew_b, next_b, done_b)

    def on_episode_end(self) -> None:
        pass   # Epsilon decays per step, not per episode

    def to_cfg(self) -> dict:
        return {
            "policy_type":        "neural_dqn",
            "hidden_sizes":       self._hidden,
            "replay_buffer_size": self._buf_maxlen,
            "batch_size":         self._batch_size,
            "min_replay_size":    self._min_replay,
            "target_update_freq": self._target_freq,
            "learning_rate":      float(self._lr),
            "epsilon_start":      float(self._eps_start),
            "epsilon_end":        float(self._eps_end),
            "epsilon_decay_steps": self._eps_steps,
            "gamma":              float(self._gamma),
            "n_lidar_rays":       self._n_lidar_rays,
            "epsilon":            float(self._eps),
            "total_steps":        self._total_steps,
            "grad_steps":         self._grad_steps,
            "online_weights":     [w.tolist() for w in self._online["weights"]],
            "online_biases":      [b.tolist() for b in self._online["biases"]],
            "target_weights":     [w.tolist() for w in self._target["weights"]],
            "target_biases":      [b.tolist() for b in self._target["biases"]],
        }
        if self._champion is None:
            raise RuntimeError(
                "CMAESPolicy: no champion yet — call sample_population() "
                "and update_distribution() for at least one generation first."
            )
        return self._champion(obs)

    def save(self, path: str) -> None:
        """Save champion in WeightedLinearPolicy YAML format for analytics compatibility."""
        if self._champion is not None:
            self._champion.save(path)
