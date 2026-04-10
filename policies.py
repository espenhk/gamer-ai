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
"""

from __future__ import annotations

import logging
import math
import os
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
