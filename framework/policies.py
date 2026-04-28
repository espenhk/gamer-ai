"""Generic control policies, parameterised by observation spec and action space.

Game integrations inject an ObsSpec and action definitions at construction time
so the framework layer never imports from games/.

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
import pickle
from abc import ABC, abstractmethod

import numpy as np
import yaml

from framework.obs_spec import ObsSpec

logger = logging.getLogger(__name__)


_MAX_QTABLE_ENTRIES = 500_000


def _qtable_pkl_path(yaml_path: str) -> str:
    base, _ = os.path.splitext(yaml_path)
    return base + "_qtable.pkl"


# ---------------------------------------------------------------------------
# BasePolicy
# ---------------------------------------------------------------------------

class BasePolicy(ABC):
    """Abstract base class for all driving policies."""

    @abstractmethod
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Select action given observation array."""

    @abstractmethod
    def to_cfg(self) -> dict:
        """Return a YAML-serializable dict representing this policy's state."""

    def update(self, obs: np.ndarray, action: np.ndarray, reward: float,
               next_obs: np.ndarray, done: bool) -> None:
        """Per-step feedback from the environment.  No-op for non-online policies."""

    def on_episode_end(self) -> None:
        """Called once at the end of each episode.  No-op by default."""

    def save(self, path: str) -> None:
        """Write to_cfg() to YAML at path."""
        with open(path, "w") as f:
            yaml.dump(self.to_cfg(), f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# WeightedLinearPolicy
# ---------------------------------------------------------------------------

class WeightedLinearPolicy(BasePolicy):
    """
    Linear policy with one output head per entry in head_names.

    Output convention:
        head_names[0]  → clipped to [-1, 1]   (continuous, e.g. steering)
        head_names[1:] → thresholded at 0      (binary on/off, e.g. accel, brake)

    Weights are stored as {head}_weights in YAML; existing TMNF files
    (steer_weights / accel_weights / brake_weights) are byte-compatible.

    Construct via:
        WeightedLinearPolicy(obs_spec, head_names, weights_file)
        WeightedLinearPolicy.from_cfg(cfg, obs_spec, head_names)
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        head_names: list[str],
        weights_file: str | None = None,
    ) -> None:
        self._obs_spec     = obs_spec
        self._head_names   = list(head_names)
        self._weights_file = weights_file
        if weights_file is not None:
            cfg = self._load_or_init()
            logger.info("[WeightedLinearPolicy] loaded weights from %s", weights_file)
        else:
            cfg = self._init_random()
        self._apply_cfg(cfg)

    @classmethod
    def from_cfg(
        cls,
        cfg: dict,
        obs_spec: ObsSpec,
        head_names: list[str],
    ) -> "WeightedLinearPolicy":
        """Create a policy from a weights dict (not backed by a file)."""
        obj = object.__new__(cls)
        obj._weights_file = None
        obj._obs_spec     = obs_spec
        obj._head_names   = list(head_names)
        obj._apply_cfg(cfg)
        return obj

    # ------------------------------------------------------------------
    # Weights dict I/O
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        names = self._obs_spec.names
        return {
            f"{h}_weights": {n: float(self._weights[h][i]) for i, n in enumerate(names)}
            for h in self._head_names
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_cfg(), f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Mutation / flat-weight interface
    # ------------------------------------------------------------------

    def to_flat(self) -> np.ndarray:
        """Return all head weights concatenated as a single float32 vector."""
        return np.concatenate([self._weights[h] for h in self._head_names])

    def with_flat(self, flat: np.ndarray) -> "WeightedLinearPolicy":
        """Return a new policy with weights replaced by a flat vector."""
        n = self._obs_spec.dim
        obj = object.__new__(type(self))
        obj._obs_spec     = self._obs_spec
        obj._head_names   = self._head_names
        obj._weights_file = self._weights_file
        obj._weights      = {}
        for i, head in enumerate(self._head_names):
            offset = i * n
            obj._weights[head] = flat[offset: offset + n].astype(np.float32).copy()
        return obj

    def mutated(self, scale: float = 0.1, share: float = 1.0) -> "WeightedLinearPolicy":
        """Return a new policy with Gaussian perturbation applied to a random subset of weights.

        share: probability [0, 1] that each individual weight is perturbed.
               1.0 = all weights mutated (original behaviour).
        """
        rng = np.random.default_rng()
        obj = object.__new__(type(self))
        obj._obs_spec     = self._obs_spec
        obj._head_names   = self._head_names
        obj._weights_file = self._weights_file
        obj._weights      = {}
        for head, w in self._weights.items():
            if share >= 1.0:
                mask = np.ones(len(w), dtype=bool)
            else:
                mask = rng.random(len(w)) < share
            noise = rng.normal(0.0, scale, len(w)).astype(np.float32)
            obj._weights[head] = w + noise * mask
        return obj

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        norm_obs = obs / self._obs_spec.scales
        outputs  = []
        for i, head in enumerate(self._head_names):
            score = float(np.dot(self._weights[head], norm_obs))
            if i == 0:
                outputs.append(float(np.clip(score, -1.0, 1.0)))  # continuous head
            else:
                outputs.append(1.0 if score > 0.0 else 0.0)       # binary head
        return np.array(outputs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_cfg(self, cfg: dict) -> None:
        names = self._obs_spec.names
        self._weights = {}
        for head in self._head_names:
            key      = f"{head}_weights"
            head_cfg = cfg.get(key, {})
            self._weights[head] = np.array(
                [float(head_cfg.get(n, 0.0)) for n in names],
                dtype=np.float32,
            )

    def _load_or_init(self) -> dict:
        names = self._obs_spec.names
        if os.path.exists(self._weights_file):
            with open(self._weights_file) as f:
                cfg = yaml.safe_load(f) or {}
            if not isinstance(cfg, dict):
                logger.warning(
                    "[WeightedLinearPolicy] invalid weights config in %s; expected a mapping, got %s. "
                    "Using empty config.",
                    self._weights_file,
                    type(cfg).__name__,
                )
                cfg = {}
            first_key   = f"{self._head_names[0]}_weights"
            loaded_dim  = len(cfg.get(first_key, {}))
            if loaded_dim != len(names):
                logger.warning(
                    "[WeightedLinearPolicy] loaded weights dim=%d doesn't match obs_dim=%d; "
                    "new features initialised to 0.0 → %s",
                    loaded_dim, len(names), self._weights_file,
                )
            return cfg
        return self._init_random()

    def _init_random(self) -> dict:
        names = self._obs_spec.names
        rng   = np.random.default_rng()
        cfg   = {
            f"{h}_weights": {n: float(rng.standard_normal()) for n in names}
            for h in self._head_names
        }
        if self._weights_file is not None:
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

    Architecture: obs → Linear → ReLU → ... → Linear(action_dim)
    Output: [out[0] via tanh, out[1:] via step (1 if > 0 else 0)]
    Pure numpy, no external ML framework required.
    Weights serialized to YAML as nested lists.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        action_dim: int = 3,
        hidden_sizes: list[int] | None = None,
    ) -> None:
        self._obs_spec   = obs_spec
        self._action_dim = action_dim
        self._hidden     = list(hidden_sizes or [16, 16])
        layer_dims       = [obs_spec.dim] + self._hidden + [action_dim]
        rng              = np.random.default_rng()
        self._weights: list[np.ndarray] = []
        self._biases:  list[np.ndarray] = []
        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            w = rng.standard_normal((layer_dims[i + 1], fan_in)).astype(np.float32)
            w *= np.sqrt(2.0 / fan_in)  # He init
            b = np.zeros(layer_dims[i + 1], dtype=np.float32)
            self._weights.append(w)
            self._biases.append(b)

    @classmethod
    def from_cfg(cls, cfg: dict, obs_spec: ObsSpec) -> "NeuralNetPolicy":
        obj              = object.__new__(cls)
        obj._obs_spec    = obs_spec
        obj._action_dim  = len(cfg["biases"][-1]) if cfg.get("biases") else 3
        obj._hidden      = cfg["hidden_sizes"]
        obj._weights     = [np.array(w, dtype=np.float32) for w in cfg["weights"]]
        obj._biases      = [np.array(b, dtype=np.float32) for b in cfg["biases"]]
        return obj

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs / self._obs_spec.scales).astype(np.float32)
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = w @ x + b
            if i < len(self._weights) - 1:
                x = np.maximum(0.0, x)  # ReLU on all but output layer
        # Output: first dim tanh (continuous), rest step (binary)
        outputs = [float(np.tanh(x[0]))]
        for j in range(1, self._action_dim):
            outputs.append(1.0 if float(x[j]) > 0.0 else 0.0)
        return np.array(outputs, dtype=np.float32)

    def mutated(self, scale: float = 0.1, **_) -> "NeuralNetPolicy":
        """Return a new policy with Gaussian noise added to all weights and biases."""
        rng = np.random.default_rng()
        obj             = object.__new__(type(self))
        obj._obs_spec   = self._obs_spec
        obj._action_dim = self._action_dim
        obj._hidden     = self._hidden
        obj._weights    = [w + rng.normal(0.0, scale, w.shape).astype(np.float32)
                           for w in self._weights]
        obj._biases     = [b + rng.normal(0.0, scale, b.shape).astype(np.float32)
                           for b in self._biases]
        return obj

    def to_cfg(self) -> dict:
        return {
            "policy_type":  "neural_net",
            "hidden_sizes": self._hidden,
            "action_dim":   self._action_dim,
            "weights":      [w.tolist() for w in self._weights],
            "biases":       [b.tolist() for b in self._biases],
        }


# ---------------------------------------------------------------------------
# Observation discretization helper
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


# ---------------------------------------------------------------------------
# QTablePolicy — shared base for tabular Q-learning policies
# ---------------------------------------------------------------------------

class QTablePolicy(BasePolicy):
    """
    Shared base for tabular Q-learning policies (EpsilonGreedy and MCTS).

    Manages the Q-table, visit counts, discretization, and Bellman updates.
    Subclasses override _select_action() to implement their exploration strategy.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        discrete_actions: np.ndarray,
        alpha: float = 0.1,
        gamma: float = 0.99,
        n_bins: int = 3,
    ) -> None:
        self._obs_spec        = obs_spec
        self._discrete_actions = discrete_actions
        self._n_actions       = len(discrete_actions)
        self._alpha           = alpha
        self._gamma           = gamma
        self._n_bins          = n_bins
        self._scales          = obs_spec.scales
        self._q_table: dict[tuple, np.ndarray] = {}
        self._n_sa:    dict[tuple, np.ndarray] = {}
        self._n_s:     dict[tuple, int]        = {}
        self._last_obs    = None
        self._last_action = None

    def _q(self, s: tuple) -> np.ndarray:
        if s not in self._q_table:
            self._q_table[s] = np.zeros(self._n_actions, dtype=np.float32)
        return self._q_table[s]

    def _n(self, s: tuple) -> np.ndarray:
        if s not in self._n_sa:
            self._n_sa[s] = np.zeros(self._n_actions, dtype=np.float32)
        return self._n_sa[s]

    @abstractmethod
    def _select_action(self, state_key: tuple) -> int:
        """Choose an action for state key s.  Implemented by subclasses."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        s = _discretize_obs(obs, self._scales, self._n_bins)
        self._last_obs    = obs
        action_idx        = self._select_action(s)
        self._last_action = action_idx
        return self._discrete_actions[action_idx].copy()

    def update(self, obs: np.ndarray, action, reward: float,
               next_obs: np.ndarray, done: bool) -> None:
        action_idx = int(action) if np.isscalar(action) else self._nearest_action(action)
        s  = _discretize_obs(obs,      self._scales, self._n_bins)
        s_ = _discretize_obs(next_obs, self._scales, self._n_bins)
        q_next = 0.0 if done else float(np.max(self._q(s_)))
        td = reward + self._gamma * q_next - self._q(s)[action_idx]
        self._q(s)[action_idx] += self._alpha * td
        self._n(s)[action_idx] += 1.0
        self._n_s[s] = self._n_s.get(s, 0) + 1

    def _nearest_action(self, action: np.ndarray) -> int:
        diffs = np.abs(self._discrete_actions - action[np.newaxis, :]).sum(axis=1)
        return int(np.argmin(diffs))

    def on_episode_end(self) -> None:
        self._last_obs    = None
        self._last_action = None

    @property
    def n_states_visited(self) -> int:
        return len(self._q_table)

    def save(self, path: str) -> None:
        super().save(path)
        if len(self._q_table) > _MAX_QTABLE_ENTRIES:
            logger.warning(
                "Q-table has %d entries (>%d), skipping pickle.",
                len(self._q_table), _MAX_QTABLE_ENTRIES,
            )
            return
        pkl_path = _qtable_pkl_path(path)
        with open(pkl_path, "wb") as f:
            pickle.dump((self._q_table, self._n_sa, self._n_s), f)
        logger.info("Q-table saved: %d states → %s", len(self._q_table), pkl_path)

    def _load_table(self, path: str) -> None:
        pkl_path = _qtable_pkl_path(path)
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                q_table, n_sa, n_s = pickle.load(f)
            self._q_table = q_table
            self._n_sa    = n_sa
            self._n_s     = n_s
            logger.info("Q-table loaded: %d states from %s", len(q_table), pkl_path)


# ---------------------------------------------------------------------------
# EpsilonGreedyPolicy
# ---------------------------------------------------------------------------

class EpsilonGreedyPolicy(QTablePolicy):
    """
    Tabular Q-learning with epsilon-greedy exploration.

    Actions are selected from discrete_actions (injected at construction time).
    Q-values are updated online via the Bellman equation after every env step.
    Epsilon decays each episode.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        discrete_actions: np.ndarray,
        n_bins: int = 3,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        alpha: float = 0.1,
        gamma: float = 0.99,
    ) -> None:
        super().__init__(obs_spec, discrete_actions, alpha=alpha, gamma=gamma, n_bins=n_bins)
        self._epsilon       = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min   = epsilon_min

    def _select_action(self, state_key: tuple) -> int:
        if np.random.random() < self._epsilon:
            return int(np.random.randint(self._n_actions))
        return int(np.argmax(self._q(state_key)))

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

    NOTE: True MCTS requires env cloning.  This is a UCT-style approximation
    that builds value/count tables incrementally over real episodes.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        discrete_actions: np.ndarray,
        c: float = 1.41,
        alpha: float = 0.1,
        gamma: float = 0.99,
        n_bins: int = 3,
    ) -> None:
        super().__init__(obs_spec, discrete_actions, alpha=alpha, gamma=gamma, n_bins=n_bins)
        self._c = c

    def _select_action(self, state_key: tuple) -> int:
        n_s = self._n_s.get(state_key, 0)
        if n_s == 0:
            return int(np.random.randint(self._n_actions))
        ucb = self._q(state_key) + self._c * np.sqrt(
            math.log(n_s + 1) / (self._n(state_key) + 1e-8)
        )
        return int(np.argmax(ucb))

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
    `save()` writes the champion in WeightedLinearPolicy YAML format.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        head_names: list[str],
        population_size: int = 10,
        elite_k: int = 3,
        mutation_scale: float = 0.1,
        mutation_share: float = 1.0,
    ) -> None:
        self._obs_spec       = obs_spec
        self._head_names     = list(head_names)
        self._pop_size       = population_size
        self._elite_k        = min(elite_k, population_size)
        self._mutation_scale = mutation_scale
        self._mutation_share = mutation_share
        self._population: list[WeightedLinearPolicy] = []
        self._champion: WeightedLinearPolicy | None  = None
        self._champion_reward: float                  = float("-inf")

    def _make_member(self, cfg: dict) -> WeightedLinearPolicy:
        """Factory for population members.  Subclasses can override."""
        return WeightedLinearPolicy.from_cfg(cfg, self._obs_spec, self._head_names)

    @property
    def population(self) -> list[WeightedLinearPolicy]:
        return self._population

    @property
    def champion_reward(self) -> float:
        return self._champion_reward

    def initialize_random(self) -> None:
        """Build a fresh random population."""
        rng   = np.random.default_rng()
        names = self._obs_spec.names
        pop   = []
        for _ in range(self._pop_size):
            cfg = {
                f"{h}_weights": {n: float(rng.standard_normal()) for n in names}
                for h in self._head_names
            }
            pop.append(self._make_member(cfg))
        self._population = pop
        if self._champion is None:
            self._champion = pop[0]

    def initialize_from_champion(self, champion: WeightedLinearPolicy) -> None:
        """Seed the population by mutating the given champion."""
        self._champion    = champion
        self._population  = [
            champion.mutated(self._mutation_scale, self._mutation_share)
            for _ in range(self._pop_size)
        ]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        assert self._champion is not None, \
            "GeneticPolicy: champion not set — call initialize_*() first"
        return self._champion(obs)

    def evaluate_and_evolve(self, rewards: list[float]) -> bool:
        """
        Update population based on episode rewards.

        Returns True if the champion was updated this generation.
        """
        assert len(rewards) == len(self._population)
        ranked   = sorted(zip(rewards, self._population), key=lambda x: -x[0])
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
            child     = self._make_member(child_cfg)
            new_pop.append(child.mutated(self._mutation_scale, self._mutation_share))

        self._population = new_pop
        return improved

    @staticmethod
    def _crossover(cfg1: dict, cfg2: dict) -> dict:
        """Uniform weight crossover: each weight is randomly drawn from parent 1 or 2."""
        result = {}
        for key in cfg1:
            if not key.endswith("_weights"):
                continue
            result[key] = {}
            for k in cfg1[key]:
                result[key][k] = (cfg1[key][k] if np.random.random() < 0.5
                                  else cfg2[key].get(k, cfg1[key][k]))
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
