"""SC2-specific policy classes.

SC2MultiHeadLinearPolicy
    Multi-output linear policy for SC2.  Two weight matrices are maintained:

    * **fn_idx head** — shape ``(N_FUNCTION_IDS, obs_dim)`` → 6 scores, one per
      available function ID.  ``argmax`` gives the selected function.
    * **spatial head** — shape ``(2, obs_dim)`` → two scalar scores fed
      through a sigmoid to produce continuous ``(x, y) ∈ [0, 1]²`` screen
      coordinates (issue #122).  Continuous output lets the policy target
      arbitrary screen pixels rather than collapsing onto a fixed grid.

    The resulting action is a 4-vector ``[fn_idx, x, y, 0]`` compatible with
    :data:`games.sc2.actions.DISCRETE_ACTIONS` and the ``SC2Env`` action space.

    YAML serialisation uses one ``{head}_{row}_weights`` key per matrix row so
    that the base-class ``GeneticPolicy._crossover`` works without modification
    (every key ends with ``_weights`` and maps to a ``{obs_name: float}`` dict).

SC2GeneticPolicy
    Thin subclass of ``framework.policies.GeneticPolicy`` that substitutes
    ``SC2MultiHeadLinearPolicy`` as the individual type.  All evolutionary
    mechanics (crossover, mutation, elite selection) are unchanged.

    Register as ``sc2_genetic`` in the training factory.
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple

import numpy as np
import yaml

from framework.cmaes import CMAESPolicy as _FrameworkCMAES
from framework.dqn import DQNPolicy as _FrameworkDQN
from framework.obs_spec import ObsSpec
from framework.policies import BasePolicy, GeneticPolicy, register_policy, trainer_state_path
from games.sc2.actions import DISCRETE_ACTIONS, FUNCTION_IDS, build_available_actions_mask
from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants derived from the action definitions
# ---------------------------------------------------------------------------

#: Number of function IDs exposed by the SC2 action set.
N_FUNCTION_IDS: int = len(FUNCTION_IDS)   # 6

#: Number of rows in the spatial (x, y) sigmoid head.
N_SPATIAL_ROWS: int = 2

#: Backwards-compat alias — pre-#122 code referred to the old 9-cell grid as
#: ``N_GRID_CELLS``.  After issue #122 the spatial head emits continuous
#: ``(x, y)`` coordinates instead, so this is now the size of the new
#: 2-row spatial weight matrix.  Kept under the historic name so external
#: imports continue to work.
N_GRID_CELLS: int = N_SPATIAL_ROWS

#: Head-name prefixes — one row per output neuron stored as a separate YAML key.
_FN_HEAD_NAMES: list[str]      = [f"fn_idx_{i}" for i in range(N_FUNCTION_IDS)]
_SPATIAL_HEAD_NAMES: list[str] = ["x", "y"]
_ALL_ROW_NAMES: list[str]      = _FN_HEAD_NAMES + _SPATIAL_HEAD_NAMES


def _sigmoid(x: float) -> float:
    """Stable scalar sigmoid."""
    return 1.0 / (1.0 + float(np.exp(-np.clip(x, -20.0, 20.0))))


# ---------------------------------------------------------------------------
# SC2MultiHeadLinearPolicy
# ---------------------------------------------------------------------------

class SC2MultiHeadLinearPolicy:
    """Multi-head linear policy for StarCraft 2.

    Parameters
    ----------
    obs_spec :
        Observation spec describing feature names and normalisation scales.
    fn_weights :
        Weight matrix of shape ``(N_FUNCTION_IDS, obs_dim)``.  If *None* a
        random initialisation is used.
    spatial_weights :
        Weight matrix of shape ``(N_SPATIAL_ROWS, obs_dim)`` whose rows are
        the linear-projection coefficients for ``x`` and ``y`` respectively.
        If *None* a random initialisation is used.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        fn_weights: np.ndarray | None = None,
        spatial_weights: np.ndarray | None = None,
    ) -> None:
        self._obs_spec = obs_spec
        obs_dim = obs_spec.dim
        rng = np.random.default_rng()

        self._fn_weights: np.ndarray = (
            fn_weights.astype(np.float32)
            if fn_weights is not None
            else rng.standard_normal((N_FUNCTION_IDS, obs_dim)).astype(np.float32)
        )
        self._sp_weights: np.ndarray = (
            spatial_weights.astype(np.float32)
            if spatial_weights is not None
            else rng.standard_normal((N_SPATIAL_ROWS, obs_dim)).astype(np.float32)
        )

        # Cache of available fn_ids from the most recent on_episode_start() /
        # update() call.  None means all functions are available (no masking).
        self._available_fn_ids: set[int] | None = None

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Select action given observation.

        Unavailable function IDs (cached from the previous ``update()`` call
        or from ``on_episode_start()``) are masked to ``-inf`` before taking
        the ``argmax``, so the policy never emits an action that the SC2
        client would have to substitute with ``no_op`` or ``select_army``.

        Returns
        -------
        np.ndarray
            4-vector ``[fn_idx, x, y, 0]`` compatible with ``SC2Env``.  ``x``
            and ``y`` are continuous in ``[0, 1]`` (sigmoid-encoded), so the
            policy can target arbitrary screen pixels.
        """
        norm_obs  = obs / self._obs_spec.scales
        fn_scores = self._fn_weights @ norm_obs   # (N_FUNCTION_IDS,)
        sp_scores = self._sp_weights @ norm_obs   # (2,) — raw x and y logits

        # Mask unavailable function IDs so argmax never selects them.
        if self._available_fn_ids is not None:
            for i in range(N_FUNCTION_IDS):
                if i not in self._available_fn_ids:
                    fn_scores[i] = -np.inf
            # If all scored to -inf (empty set), fall back to no_op (idx 0).
            if not np.any(np.isfinite(fn_scores)):
                fn_scores[0] = 0.0

        fn_idx = int(np.argmax(fn_scores))
        x      = _sigmoid(float(sp_scores[0]))
        y      = _sigmoid(float(sp_scores[1]))
        return np.array([fn_idx, x, y, 0.0], dtype=np.float32)

    # ------------------------------------------------------------------
    # Serialisation — row-per-head YAML format
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        """Return a YAML-serialisable dict.

        Format::

            fn_idx_0_weights: {obs_name_0: float, ...}
            ...
            fn_idx_5_weights: {obs_name_0: float, ...}
            x_weights:        {obs_name_0: float, ...}
            y_weights:        {obs_name_0: float, ...}

        Every key ends with ``_weights``, so
        :meth:`framework.policies.GeneticPolicy._crossover` works without
        modification.
        """
        names = self._obs_spec.names
        cfg: dict = {}
        for i, row_name in enumerate(_FN_HEAD_NAMES):
            cfg[f"{row_name}_weights"] = {
                n: float(self._fn_weights[i, j]) for j, n in enumerate(names)
            }
        for i, row_name in enumerate(_SPATIAL_HEAD_NAMES):
            cfg[f"{row_name}_weights"] = {
                n: float(self._sp_weights[i, j]) for j, n in enumerate(names)
            }
        return cfg

    def save(self, path: str) -> None:
        """Write ``to_cfg()`` to YAML at *path*."""
        with open(path, "w") as f:
            yaml.dump(self.to_cfg(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_cfg(cls, cfg: dict, obs_spec: ObsSpec) -> "SC2MultiHeadLinearPolicy":
        """Reconstruct a policy from a ``to_cfg()`` dict.

        Unknown keys and missing observation features default to 0.0 so that
        the policy can load configs created with a different obs_spec dimension
        (same migration behaviour as ``WeightedLinearPolicy``).  Pre-#122
        weight files (with ``spatial_{0..8}_weights`` keys) silently migrate
        to all-zero ``x_weights`` / ``y_weights`` because the old 9-cell
        argmax encoding has no meaningful projection onto the new continuous
        head.
        """
        names   = obs_spec.names
        obs_dim = obs_spec.dim

        fn_weights = np.zeros((N_FUNCTION_IDS, obs_dim), dtype=np.float32)
        sp_weights = np.zeros((N_SPATIAL_ROWS,   obs_dim), dtype=np.float32)

        for i, row_name in enumerate(_FN_HEAD_NAMES):
            row_cfg = cfg.get(f"{row_name}_weights", {})
            for j, n in enumerate(names):
                fn_weights[i, j] = float(row_cfg.get(n, 0.0))

        for i, row_name in enumerate(_SPATIAL_HEAD_NAMES):
            row_cfg = cfg.get(f"{row_name}_weights", {})
            for j, n in enumerate(names):
                sp_weights[i, j] = float(row_cfg.get(n, 0.0))

        return cls(obs_spec, fn_weights=fn_weights, spatial_weights=sp_weights)

    @classmethod
    def load(cls, path: str, obs_spec: ObsSpec) -> "SC2MultiHeadLinearPolicy":
        """Load from a YAML file written by :meth:`save`."""
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        return cls.from_cfg(cfg, obs_spec)

    # ------------------------------------------------------------------
    # Flat-weight interface (for CMA-ES interoperability)
    # ------------------------------------------------------------------

    def to_flat(self) -> np.ndarray:
        """Return all weights as a single ``float32`` vector.

        Layout: ``[fn_row_0 | … | fn_row_5 | x_row | y_row]``.
        Total length: ``(N_FUNCTION_IDS + N_SPATIAL_ROWS) × obs_dim``.
        """
        return np.concatenate(
            [self._fn_weights.ravel(), self._sp_weights.ravel()]
        ).astype(np.float32)

    def with_flat(self, flat: np.ndarray) -> "SC2MultiHeadLinearPolicy":
        """Return a new policy whose weights are set from a flat vector."""
        obs_dim    = self._obs_spec.dim
        fn_size    = N_FUNCTION_IDS * obs_dim
        fn_weights = flat[:fn_size].reshape(N_FUNCTION_IDS, obs_dim).astype(np.float32)
        sp_weights = flat[fn_size:].reshape(N_SPATIAL_ROWS, obs_dim).astype(np.float32)
        return SC2MultiHeadLinearPolicy(self._obs_spec, fn_weights, sp_weights)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mutated(self, scale: float = 0.1, share: float = 1.0) -> "SC2MultiHeadLinearPolicy":
        """Return a new policy with Gaussian perturbation on a random subset of weights.

        Parameters
        ----------
        scale :
            Standard deviation of the Gaussian noise.
        share :
            Probability ``[0, 1]`` that each individual weight is perturbed.
            ``1.0`` mutates every weight.
        """
        rng      = np.random.default_rng()
        flat     = self.to_flat()
        new_flat = flat.copy()
        if share >= 1.0:
            new_flat += rng.normal(0.0, scale, len(flat)).astype(np.float32)
        else:
            mask  = rng.random(len(flat)) < share
            idx   = np.where(mask)[0]
            if len(idx) > 0:
                noise = rng.normal(0.0, scale, len(idx)).astype(np.float32)
                new_flat[idx] += noise
        return self.with_flat(new_flat)

    # ------------------------------------------------------------------
    # Framework compatibility shims
    # ------------------------------------------------------------------

    def on_episode_start(self, **kwargs) -> None:
        """Cache available_fn_ids from the reset info for the first step.

        The ``info`` kwarg carries the dict returned by ``env.reset()``.
        If it contains ``"available_fn_ids"``, it is used to prime the
        available-actions mask so the very first action of the episode is
        chosen from the correct set.  If the key is absent the mask is
        cleared (all functions enabled).
        """
        info = kwargs.get("info") or {}
        available = info.get("available_fn_ids")
        if available is not None:
            self._available_fn_ids = set(available)
        else:
            self._available_fn_ids = None

    def update(self, obs: np.ndarray, action: np.ndarray, reward: float,
               next_obs: np.ndarray, done: bool, **kwargs) -> None:
        """Cache available_fn_ids for the next __call__; no weight update.

        Evolutionary policies update weights between episodes (via the genetic
        loop), not per-step.  This method only caches the available function
        IDs so that the next ``__call__`` can mask unavailable actions.
        """
        info = kwargs.get("info") or {}
        available = info.get("available_fn_ids")
        if available is not None:
            self._available_fn_ids = set(available)


# ---------------------------------------------------------------------------
# SC2GeneticPolicy
# ---------------------------------------------------------------------------

@register_policy
class SC2GeneticPolicy(GeneticPolicy):
    """SC2 variant of :class:`framework.policies.GeneticPolicy`.

    Uses :class:`SC2MultiHeadLinearPolicy` as the individual type.  The
    evolutionary mechanics (uniform crossover, mutation, elite selection,
    champion tracking) are inherited unchanged from the framework base class.

    The YAML format of each individual matches ``SC2MultiHeadLinearPolicy``
    (``fn_idx_{i}_weights`` + ``x_weights`` + ``y_weights`` keys), so
    ``GeneticPolicy._crossover`` works without any modification.

    Parameters
    ----------
    obs_spec :
        Observation spec.  Defaults to ``SC2_MINIGAME_OBS_SPEC`` (13 dims).
    population_size :
        Number of individuals per generation (λ).
    elite_k :
        Number of top individuals preserved unchanged each generation.
    mutation_scale :
        Standard deviation of Gaussian noise applied to mutated weights.
    mutation_share :
        Fraction of weights perturbed per mutation (sparse mutation).
    eval_episodes :
        Episodes per individual per generation; fitness is the average reward.
    """

    POLICY_TYPE = "sc2_genetic"
    LOOP_TYPE   = "genetic"
    VALID_POLICY_PARAMS = frozenset({
        "population_size", "elite_k", "mutation_scale", "mutation_share", "eval_episodes",
    })

    @classmethod
    def compatible_with(cls, game_name: str) -> tuple[bool, str | None]:
        # Cancel the parent GeneticPolicy's blanket SC2 rejection while still
        # rejecting non-SC2 games: this SC2-native variant emits the SC2
        # [fn_idx, x, y, queue] action encoding.
        if game_name != "sc2":
            return False, "This policy is SC2-specific; use game='sc2'."
        return True, None

    def __init__(
        self,
        obs_spec: ObsSpec = SC2_MINIGAME_OBS_SPEC,
        population_size: int = 30,
        elite_k: int = 5,
        mutation_scale: float = 0.1,
        mutation_share: float = 0.3,
        eval_episodes: int = 2,
    ) -> None:
        # Pass the flat row-names as head_names so the parent's
        # initialize_random() builds the correct {row_name}_weights keys.
        super().__init__(
            obs_spec        = obs_spec,
            head_names      = _ALL_ROW_NAMES,
            population_size = population_size,
            elite_k         = elite_k,
            mutation_scale  = mutation_scale,
            mutation_share  = mutation_share,
            eval_episodes   = eval_episodes,
        )

    # ------------------------------------------------------------------
    # Individual factory — override to use SC2MultiHeadLinearPolicy
    # ------------------------------------------------------------------

    def _make_member(self, cfg: dict) -> SC2MultiHeadLinearPolicy:  # type: ignore[override]
        """Build an SC2MultiHeadLinearPolicy from a ``to_cfg()`` dict."""
        return SC2MultiHeadLinearPolicy.from_cfg(cfg, self._obs_spec)

    # ------------------------------------------------------------------
    # Population seed from a saved champion file
    # ------------------------------------------------------------------

    def initialize_from_file(self, path: str) -> None:
        """Load champion from YAML and seed the population by mutation."""
        champion = SC2MultiHeadLinearPolicy.load(path, self._obs_spec)
        self.initialize_from_champion(champion)
        logger.info("[SC2GeneticPolicy] seeded population from champion at %s", path)

    # ------------------------------------------------------------------
    # to_cfg / save — propagate sc2_genetic policy_type label
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        cfg = super().to_cfg()
        cfg["policy_type"] = "sc2_genetic"
        return cfg

    def save(self, path: str) -> None:
        """Save champion in ``SC2MultiHeadLinearPolicy`` YAML format."""
        if self._champion is not None:
            self._champion.save(path)

    # ------------------------------------------------------------------
    # from_cfg convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_cfg(
        cls,
        cfg: dict,
        obs_spec: ObsSpec = SC2_MINIGAME_OBS_SPEC,
    ) -> "SC2GeneticPolicy":
        """Reconstruct policy from a ``to_cfg()`` dict.

        Restores hyperparameters and, when ``champion_weights`` is present,
        also restores the champion individual and ``champion_reward`` so that a
        full round-trip through ``to_cfg()`` / ``from_cfg()`` is lossless.
        """
        policy = cls(
            obs_spec        = obs_spec,
            population_size = cfg.get("population_size", 30),
            elite_k         = cfg.get("elite_k", 5),
            mutation_scale  = float(cfg.get("mutation_scale", 0.1)),
            mutation_share  = float(cfg.get("mutation_share", 0.3)),
            eval_episodes   = int(cfg.get("eval_episodes", 2)),
        )
        champion_cfg = cfg.get("champion_weights")
        if champion_cfg and isinstance(champion_cfg, dict):
            policy._champion = SC2MultiHeadLinearPolicy.from_cfg(
                champion_cfg, obs_spec
            )
            policy._champion_reward = float(cfg.get("champion_reward", float("-inf")))
        return policy

    @classmethod
    def _construct_or_resume(cls, *, obs_spec, head_names, discrete_actions,
                             weights_file, policy_params, re_initialize):
        pp = policy_params
        policy = cls(
            obs_spec        = obs_spec,
            population_size = pp.get("population_size", 30),
            elite_k         = pp.get("elite_k",         5),
            mutation_scale  = float(pp.get("mutation_scale", 0.1)),
            mutation_share  = float(pp.get("mutation_share", 0.3)),
            eval_episodes   = int(pp.get("eval_episodes",    2)),
        )
        if os.path.exists(weights_file) and not re_initialize:
            policy.initialize_from_file(weights_file)
        else:
            policy.initialize_random()
            logger.info("[SC2GeneticPolicy] random population of %d", policy._lam)
        return policy


# ---------------------------------------------------------------------------
# SC2NeuralNetPolicy — TMNF-style MLP with SC2 action encoding
# ---------------------------------------------------------------------------

@register_policy
class SC2NeuralNetPolicy(BasePolicy):
    """Small MLP policy trained via hill-climbing (same loop as TMNF neural_net).

    Architecture: obs → Linear → ReLU → ... → Linear(4)

    Output encoding for SC2 action vector ``[fn_idx, x, y, queue]``:
        fn_idx: sigmoid(out[0]) scaled to [0, N_FUNCTION_IDS-1], then snapped
                to an available integer function ID.
        x, y  : sigmoid(out[1]), sigmoid(out[2]) ∈ [0, 1]
        queue : step(out[3]) ∈ {0, 1}
    """

    POLICY_TYPE = "sc2_neural_net"
    LOOP_TYPE   = "hill_climbing"
    VALID_POLICY_PARAMS = frozenset({"hidden_sizes"})

    @classmethod
    def compatible_with(cls, game_name: str) -> tuple[bool, str | None]:
        if game_name != "sc2":
            return False, "This policy is SC2-specific; use game='sc2'."
        return True, None

    def __init__(
        self,
        obs_spec: ObsSpec,
        hidden_sizes: list[int] | None = None,
    ) -> None:
        self._obs_spec = obs_spec
        self._action_dim = 4
        self._hidden = list(hidden_sizes or [16, 16])
        layer_dims = [obs_spec.dim] + self._hidden + [self._action_dim]
        rng = np.random.default_rng()
        self._weights: list[np.ndarray] = []
        self._biases: list[np.ndarray] = []
        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            w = rng.standard_normal((layer_dims[i + 1], fan_in)).astype(np.float32)
            w *= np.sqrt(2.0 / fan_in)  # He init
            b = np.zeros(layer_dims[i + 1], dtype=np.float32)
            self._weights.append(w)
            self._biases.append(b)

        self._available_fn_ids: set[int] | None = None

    @classmethod
    def from_cfg(cls, cfg: dict, obs_spec: ObsSpec) -> "SC2NeuralNetPolicy":
        obj = object.__new__(cls)
        obj._obs_spec = obs_spec
        obj._action_dim = 4
        obj._hidden = cfg.get("hidden_sizes", [16, 16])
        obj._weights = [np.array(w, dtype=np.float32) for w in cfg["weights"]]
        obj._biases = [np.array(b, dtype=np.float32) for b in cfg["biases"]]
        obj._available_fn_ids = None
        return obj

    def _project_fn_idx(self, fn_scalar: float) -> int:
        fn_raw = _sigmoid(fn_scalar) * (N_FUNCTION_IDS - 1)
        fn_idx = int(np.clip(fn_raw, 0.0, float(N_FUNCTION_IDS - 1)))
        if self._available_fn_ids is None:
            return fn_idx
        available = sorted(i for i in self._available_fn_ids if 0 <= i < N_FUNCTION_IDS)
        if not available:
            return 0
        if fn_idx in self._available_fn_ids:
            return fn_idx
        return min(available, key=lambda i: abs(i - fn_raw))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs / self._obs_spec.scales).astype(np.float32)
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = w @ x + b
            if i < len(self._weights) - 1:
                x = np.maximum(0.0, x)

        fn_idx = self._project_fn_idx(float(x[0]))
        act_x = _sigmoid(float(x[1]))
        act_y = _sigmoid(float(x[2]))
        queue = float(int(float(x[3]) > 0.0))
        return np.array([fn_idx, act_x, act_y, queue], dtype=np.float32)

    def mutated(self, scale: float = 0.1, **_) -> "SC2NeuralNetPolicy":
        rng = np.random.default_rng()
        obj = object.__new__(type(self))
        obj._obs_spec = self._obs_spec
        obj._action_dim = self._action_dim
        obj._hidden = list(self._hidden)
        obj._weights = [
            w + rng.normal(0.0, scale, w.shape).astype(np.float32)
            for w in self._weights
        ]
        obj._biases = [
            b + rng.normal(0.0, scale, b.shape).astype(np.float32)
            for b in self._biases
        ]
        obj._available_fn_ids = (
            set(self._available_fn_ids) if self._available_fn_ids is not None else None
        )
        return obj

    def to_cfg(self) -> dict:
        return {
            "policy_type": "sc2_neural_net",
            "hidden_sizes": self._hidden,
            "action_dim": self._action_dim,
            "weights": [w.tolist() for w in self._weights],
            "biases": [b.tolist() for b in self._biases],
        }

    def on_episode_start(self, **kwargs) -> None:
        info = kwargs.get("info") or {}
        available = info.get("available_fn_ids")
        if available is not None:
            self._available_fn_ids = set(available)
        else:
            self._available_fn_ids = None

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray | int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        **kwargs,
    ) -> None:
        info = kwargs.get("info") or {}
        available = info.get("available_fn_ids")
        if available is not None:
            self._available_fn_ids = set(available)

    @classmethod
    def _construct_or_resume(cls, *, obs_spec, head_names, discrete_actions,
                             weights_file, policy_params, re_initialize):
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as _f:
                cfg = yaml.safe_load(_f) or {}
            if isinstance(cfg, dict) and cfg.get("policy_type") == "sc2_neural_net":
                return cls.from_cfg(cfg, obs_spec)
        return cls(
            obs_spec=obs_spec,
            hidden_sizes=policy_params.get("hidden_sizes", [16, 16]),
        )


# ---------------------------------------------------------------------------
# SC2NeuralDQNPolicy — masked DQN over SC2 discrete action rows
# ---------------------------------------------------------------------------

def _sc2_available_actions_mask(info: dict) -> np.ndarray:
    available = info.get("available_fn_ids")
    if available is None:
        return np.ones(len(DISCRETE_ACTIONS), dtype=bool)
    return build_available_actions_mask(set(available), len(DISCRETE_ACTIONS))


def _sc2_available_actions_mask_for_n_actions(n_actions: int):
    def _mask_fn(info: dict) -> np.ndarray:
        available = info.get("available_fn_ids")
        if available is None:
            return np.ones(n_actions, dtype=bool)
        return build_available_actions_mask(set(available), n_actions)

    return _mask_fn


@register_policy
class SC2NeuralDQNPolicy(_FrameworkDQN):
    """SC2 DQN wrapper with available-actions masking and registry metadata."""

    POLICY_TYPE = "sc2_neural_dqn"
    LOOP_TYPE = "q_learning"
    VALID_POLICY_PARAMS = frozenset({
        "hidden_sizes", "replay_buffer_size", "batch_size", "min_replay_size",
        "target_update_freq", "learning_rate", "epsilon_start", "epsilon_end",
        "epsilon_decay_steps", "gamma",
    })

    @classmethod
    def compatible_with(cls, game_name: str) -> tuple[bool, str | None]:
        if game_name != "sc2":
            return False, "This policy is SC2-specific; use game='sc2'."
        return True, None

    def __init__(
        self,
        obs_spec: ObsSpec,
        discrete_actions: np.ndarray | None = None,
        hidden_sizes: list[int] | None = None,
        replay_buffer_size: int = 50_000,
        batch_size: int = 64,
        min_replay_size: int = 2_000,
        target_update_freq: int = 200,
        learning_rate: float = 0.001,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 20_000,
        gamma: float = 0.995,
        available_actions_fn=None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            obs_spec=obs_spec,
            discrete_actions=DISCRETE_ACTIONS if discrete_actions is None else discrete_actions,
            hidden_sizes=hidden_sizes,
            replay_buffer_size=replay_buffer_size,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            target_update_freq=target_update_freq,
            learning_rate=learning_rate,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            gamma=gamma,
            available_actions_fn=(
                _sc2_available_actions_mask_for_n_actions(
                    len(DISCRETE_ACTIONS if discrete_actions is None else discrete_actions)
                )
                if available_actions_fn is None
                else available_actions_fn
            ),
            seed=seed,
        )

    def to_cfg(self) -> dict:
        cfg = super().to_cfg()
        cfg["policy_type"] = "sc2_neural_dqn"
        return cfg

    @classmethod
    def from_cfg(cls, cfg: dict, obs_spec: ObsSpec) -> "SC2NeuralDQNPolicy":
        return super().from_cfg(
            cfg,
            obs_spec=obs_spec,
            discrete_actions=DISCRETE_ACTIONS,
            available_actions_fn=_sc2_available_actions_mask,
        )

    def on_episode_start(self, **kwargs) -> None:
        info = kwargs.get("info") or {}
        if self._masked:
            self._cached_mask = np.asarray(
                self._avail_fn(info),
                dtype=bool,
            )

    @classmethod
    def _construct_or_resume(cls, *, obs_spec, head_names, discrete_actions,
                             weights_file, policy_params, re_initialize):
        pp = policy_params
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as _f:
                cfg = yaml.safe_load(_f) or {}
            if isinstance(cfg, dict) and cfg.get("policy_type") == "sc2_neural_dqn":
                policy = cls.from_cfg(cfg, obs_spec)
                ts = trainer_state_path(weights_file)
                if os.path.exists(ts):
                    try:
                        policy.load_trainer_state(ts)
                        logger.info("[SC2NeuralDQNPolicy] loaded trainer state from %s", ts)
                    except (ValueError, KeyError) as exc:
                        logger.warning(
                            "[SC2NeuralDQNPolicy] could not load trainer state — %s; "
                            "continuing with default state.",
                            exc,
                        )
                return policy
        return cls(
            obs_spec=obs_spec,
            hidden_sizes=pp.get("hidden_sizes", [64, 64]),
            replay_buffer_size=pp.get("replay_buffer_size", 50_000),
            batch_size=pp.get("batch_size", 64),
            min_replay_size=pp.get("min_replay_size", 2_000),
            target_update_freq=pp.get("target_update_freq", 200),
            learning_rate=pp.get("learning_rate", 0.001),
            epsilon_start=pp.get("epsilon_start", 1.0),
            epsilon_end=pp.get("epsilon_end", 0.05),
            epsilon_decay_steps=pp.get("epsilon_decay_steps", 20_000),
            gamma=pp.get("gamma", 0.995),
        )


# ---------------------------------------------------------------------------
# SC2REINFORCEPolicy — two-head REINFORCE for StarCraft 2
# ---------------------------------------------------------------------------

class _GradEntry(NamedTuple):
    """Per-step trajectory entry stored by SC2REINFORCEPolicy during an episode."""
    trunk_layer_inputs: list       # input to each trunk layer for backprop
    trunk_pre_relu:     list       # pre-activation values in trunk (for ReLU mask)
    h_last:             np.ndarray # shared trunk output (h_dim,)
    fn_probs:           np.ndarray # softmax fn probabilities after masking (N_FUNCTION_IDS,)
    fn_idx:             int        # sampled function index
    sp_sig:             np.ndarray # (2,) = sigmoid(sp_logits) = [x, y] ∈ [0,1]²
    fn_mask:            np.ndarray # bool mask: True = available fn (N_FUNCTION_IDS,)


@register_policy
class SC2REINFORCEPolicy(BasePolicy):
    """REINFORCE (Monte Carlo Policy Gradient) with a two-head MLP for SC2.

    The network has a **shared trunk** (hidden FC layers + ReLU) followed by
    two independent linear output heads:

    * **fn_head** — 6 logits, softmax → ``fn_idx ∈ {0…5}``; unavailable
      function IDs are masked to ``-∞`` before sampling.
    * **spatial_head** — 2 logits, sigmoid → continuous (x, y) ∈ [0, 1]²
      screen coordinates (issue #122).

    Both heads are trained jointly via REINFORCE:

    .. code-block:: text

        loss_t = -log π_fn(fn_idx | obs) × G_t
               - advantage × σ'(sp_logits) × G_t  (deterministic spatial gradient)

    Dispatched via ``_greedy_loop_q_learning`` (``update()`` per step,
    ``on_episode_end()`` per episode).

    Parameters
    ----------
    obs_spec :
        Observation spec describing feature names and normalisation scales.
    hidden_sizes :
        Shared trunk hidden-layer widths (default ``[128, 64]``).
    learning_rate :
        Gradient-ascent step size (default ``0.0003``).
    gamma :
        Discount factor (default ``0.995``).
    entropy_coeff :
        Entropy regularisation weight for both heads (default ``0.05``).
    baseline :
        ``"running_mean"`` (EMA of episode returns) or ``"none"``.
    seed :
        Optional RNG seed for reproducibility.
    """

    POLICY_TYPE = "sc2_reinforce"
    LOOP_TYPE   = "q_learning"
    VALID_POLICY_PARAMS = frozenset({
        "hidden_sizes", "learning_rate", "gamma", "entropy_coeff", "baseline",
    })

    def __init__(
        self,
        obs_spec: ObsSpec,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 0.0003,
        gamma: float = 0.995,
        entropy_coeff: float = 0.05,
        baseline: str = "running_mean",
        seed: int | None = None,
    ) -> None:
        self._obs_spec      = obs_spec
        self._obs_dim       = obs_spec.dim
        self._scales        = obs_spec.scales
        self._hidden        = list(hidden_sizes) if hidden_sizes is not None else [128, 64]
        self._lr            = float(learning_rate)
        self._gamma         = float(gamma)
        self._entropy_coeff = float(entropy_coeff)
        self._baseline_type = baseline

        # Dedicated RNG for sampling — seeded so two instances with the same
        # seed produce the same action sequence under the same weights.
        self._rng = np.random.default_rng(seed)

        (
            self._trunk_w,
            self._trunk_b,
            self._fn_w,
            self._fn_b,
            self._sp_w,
            self._sp_b,
        ) = self._build_net(seed)

        # Per-episode trajectory storage.
        # Each entry: (trunk_layer_inputs, trunk_pre_relu, h_last,
        #              fn_probs, fn_idx, sp_probs, cell_idx, fn_available_mask)
        self._ep_grads: list[tuple] = []
        self._ep_rewards: list[float] = []

        # Running-mean baseline.
        self._baseline_val   = 0.0
        self._baseline_alpha = 0.05

        # Cache of available fn_ids from the most recent update() call.
        # None means all functions are available (no masking).
        self._available_fn_ids: set[int] | None = None

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_net(
        self, seed: int | None
    ) -> tuple[
        list[np.ndarray],
        list[np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Build shared trunk + fn head + spatial head weights.

        Returns ``(trunk_w, trunk_b, fn_w, fn_b, sp_w, sp_b)``
        where each ``*_w`` / ``*_b`` list / array is He-initialised.
        """
        rng = np.random.default_rng(seed)

        # Shared trunk: obs_dim → h0 → … → h_last  (all hidden, with ReLU)
        trunk_w: list[np.ndarray] = []
        trunk_b: list[np.ndarray] = []
        dims = [self._obs_dim] + self._hidden
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            w = rng.standard_normal((dims[i + 1], fan_in)).astype(np.float32)
            w *= np.sqrt(2.0 / fan_in)
            b = np.zeros(dims[i + 1], dtype=np.float32)
            trunk_w.append(w)
            trunk_b.append(b)

        h_dim = self._hidden[-1] if self._hidden else self._obs_dim

        # fn head: h_last → N_FUNCTION_IDS logits
        fn_w = rng.standard_normal((N_FUNCTION_IDS, h_dim)).astype(np.float32)
        fn_w *= np.sqrt(2.0 / h_dim)
        fn_b = np.zeros(N_FUNCTION_IDS, dtype=np.float32)

        # spatial head: h_last → N_GRID_CELLS logits
        sp_w = rng.standard_normal((N_GRID_CELLS, h_dim)).astype(np.float32)
        sp_w *= np.sqrt(2.0 / h_dim)
        sp_b = np.zeros(N_GRID_CELLS, dtype=np.float32)

        return trunk_w, trunk_b, fn_w, fn_b, sp_w, sp_b

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_s = z - z.max()
        e   = np.exp(z_s)
        return e / e.sum()

    def _trunk_forward(
        self, obs_norm: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Forward pass through the shared trunk.

        Returns ``(h_last, layer_inputs, pre_relu)`` for backprop.
        """
        x: np.ndarray           = obs_norm.astype(np.float32)
        layer_inputs: list      = []
        pre_relu: list          = []
        for w, b in zip(self._trunk_w, self._trunk_b):
            layer_inputs.append(x.copy())
            z = w @ x + b
            pre_relu.append(z.copy())
            x = np.maximum(0.0, z)
        return x, layer_inputs, pre_relu

    def _build_fn_mask(self, available_fn_ids: set[int] | None) -> np.ndarray:
        """Return a boolean mask (True = available) over all N_FUNCTION_IDS."""
        mask = np.ones(N_FUNCTION_IDS, dtype=bool)
        if available_fn_ids is not None:
            for i in range(N_FUNCTION_IDS):
                mask[i] = i in available_fn_ids
        # Ensure at least one action is available (fall back to no_op = idx 0).
        if not mask.any():
            mask[0] = True
        return mask

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def on_episode_start(self, **kwargs) -> None:
        """Reset episode buffers and apply available_fn_ids from reset info.

        The ``info`` kwarg carries the dict returned by ``env.reset()``.
        If it contains ``"available_fn_ids"``, it is used to prime the
        available-actions mask so the very first action of the episode is
        sampled correctly rather than against the previous episode's mask.
        If the key is absent the mask is cleared (all functions enabled),
        which is safe because the SC2 client always masks at execution time.
        """
        self._ep_grads.clear()
        self._ep_rewards.clear()
        info = kwargs.get("info") or {}
        available = info.get("available_fn_ids")
        if available is not None:
            self._available_fn_ids = set(available)
        else:
            # Clear stale terminal-state mask so first step is unmasked.
            self._available_fn_ids = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Sample an action from the two-head policy.

        Uses the cached ``_available_fn_ids`` (populated by the most recent
        ``update()`` call) to mask unavailable function logits before sampling.
        """
        obs_norm = obs / self._scales
        h_last, l_in, pre_r = self._trunk_forward(obs_norm)

        # --- fn head ---
        fn_logits = self._fn_w @ h_last + self._fn_b  # (N_FUNCTION_IDS,)
        fn_mask   = self._build_fn_mask(self._available_fn_ids)
        fn_logits_masked = fn_logits.copy()
        fn_logits_masked[~fn_mask] = -np.inf
        fn_probs  = self._softmax(fn_logits_masked)
        fn_idx    = int(self._rng.choice(N_FUNCTION_IDS, p=fn_probs))

        # --- spatial head ---
        # Issue #122: continuous (x, y) ∈ [0, 1]² via sigmoid instead of
        # softmax over a fixed grid.
        sp_logits = self._sp_w @ h_last + self._sp_b  # (2,) = [x_logit, y_logit]
        sp_sig    = np.array(
            [_sigmoid(float(sp_logits[0])), _sigmoid(float(sp_logits[1]))],
            dtype=np.float32,
        )

        self._ep_grads.append(
            _GradEntry(
                trunk_layer_inputs=l_in,
                trunk_pre_relu=pre_r,
                h_last=h_last.copy(),
                fn_probs=fn_probs.copy(),
                fn_idx=fn_idx,
                sp_sig=sp_sig.copy(),
                fn_mask=fn_mask.copy(),
            )
        )

        x, y = float(sp_sig[0]), float(sp_sig[1])
        return np.array([fn_idx, x, y, 0.0], dtype=np.float32)

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray | int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        **kwargs,
    ) -> None:
        """Accumulate per-step reward and cache available_fn_ids for next call."""
        self._ep_rewards.append(float(reward))
        info = kwargs.get("info") or {}
        available = info.get("available_fn_ids")
        if available is not None:
            self._available_fn_ids = set(available)

    def on_episode_end(self) -> None:
        """Run the REINFORCE gradient update over the accumulated episode."""
        T = min(len(self._ep_grads), len(self._ep_rewards))
        if T == 0:
            self._ep_grads.clear()
            self._ep_rewards.clear()
            return

        # Discounted returns G_t.
        G = np.zeros(T, dtype=np.float64)
        running = 0.0
        for t in reversed(range(T)):
            running = self._ep_rewards[t] + self._gamma * running
            G[t]    = running

        baseline_for_advantages = self._baseline_val
        if self._baseline_type == "running_mean":
            self._baseline_val = (
                (1 - self._baseline_alpha) * self._baseline_val
                + self._baseline_alpha * float(G[0])
            )

        G_std = float(G.std())
        if G_std > 1e-6:
            G_norm = (G - G.mean()) / (G_std + 1e-8)
        else:
            G_norm = G - baseline_for_advantages

        # Gradient accumulators for trunk, fn head, spatial head.
        dW_trunk = [np.zeros_like(w, dtype=np.float64) for w in self._trunk_w]
        dB_trunk = [np.zeros_like(b, dtype=np.float64) for b in self._trunk_b]
        dW_fn    = np.zeros_like(self._fn_w, dtype=np.float64)
        dB_fn    = np.zeros_like(self._fn_b, dtype=np.float64)
        dW_sp    = np.zeros_like(self._sp_w, dtype=np.float64)
        dB_sp    = np.zeros_like(self._sp_b, dtype=np.float64)

        for t in range(T):
            entry     = self._ep_grads[t]
            advantage = float(G_norm[t])

            l_in     = entry.trunk_layer_inputs
            pre_r    = entry.trunk_pre_relu
            h_last   = entry.h_last
            fn_probs = entry.fn_probs
            fn_idx   = entry.fn_idx
            sp_sig   = entry.sp_sig.astype(np.float64)  # (2,) = [x, y]
            fn_mask  = entry.fn_mask

            # --- fn head gradient ---
            # Only include log-probs for available actions in the policy gradient.
            delta_fn = -fn_probs.copy().astype(np.float64)
            delta_fn[fn_idx] += 1.0
            # Zero out gradient for masked (unavailable) actions.
            delta_fn[~fn_mask] = 0.0
            delta_fn *= advantage

            if self._entropy_coeff > 0.0:
                # Entropy only over available actions (probs already 0 for others).
                log_p_fn  = np.log(fn_probs.astype(np.float64) + 1e-8)
                H_fn      = -float(np.dot(fn_probs[fn_mask], log_p_fn[fn_mask]))
                ent_grad_fn = np.zeros(N_FUNCTION_IDS, dtype=np.float64)
                ent_grad_fn[fn_mask] = -(
                    fn_probs[fn_mask].astype(np.float64) * (log_p_fn[fn_mask] + H_fn)
                )
                delta_fn += self._entropy_coeff * ent_grad_fn

            # --- spatial head gradient (issue #122: sigmoid continuous head) ---
            # Deterministic policy gradient: δ = advantage × σ'(logit) where
            # σ'(logit) = σ(logit) × (1 − σ(logit)) = sp_sig × (1 − sp_sig).
            # Positive advantage pushes x/y toward 1; negative advantage pushes
            # toward 0.  Entropy bonus is omitted for the spatial head since
            # there is no discrete distribution to regularise.
            delta_sp = advantage * (sp_sig * (1.0 - sp_sig))  # (2,)

            # Update head weight gradients.
            h_last_d = h_last.astype(np.float64)
            dW_fn += np.outer(delta_fn, h_last_d)
            dB_fn += delta_fn
            dW_sp += np.outer(delta_sp, h_last_d)
            dB_sp += delta_sp

            # Gradient flowing into the trunk from both heads.
            if self._trunk_w:
                g_trunk = (
                    self._fn_w.T.astype(np.float64) @ delta_fn
                    + self._sp_w.T.astype(np.float64) @ delta_sp
                )  # shape (h_dim,)

                # Backprop through trunk layers (reversed).
                n_trunk = len(self._trunk_w)
                for i in range(n_trunk - 1, -1, -1):
                    # Apply ReLU gradient (pre_relu[i] is the pre-activation value).
                    g_trunk = g_trunk * (pre_r[i] > 0).astype(np.float64)
                    dW_trunk[i] += np.outer(g_trunk, l_in[i].astype(np.float64))
                    dB_trunk[i] += g_trunk
                    if i > 0:
                        g_trunk = self._trunk_w[i].T.astype(np.float64) @ g_trunk

        lr_t = self._lr / T
        for i in range(len(self._trunk_w)):
            self._trunk_w[i] += (lr_t * dW_trunk[i]).astype(np.float32)
            self._trunk_b[i] += (lr_t * dB_trunk[i]).astype(np.float32)
        self._fn_w += (lr_t * dW_fn).astype(np.float32)
        self._fn_b += (lr_t * dB_fn).astype(np.float32)
        self._sp_w += (lr_t * dW_sp).astype(np.float32)
        self._sp_b += (lr_t * dB_sp).astype(np.float32)

        self._ep_grads.clear()
        self._ep_rewards.clear()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        return {
            "policy_type":    "sc2_reinforce",
            "hidden_sizes":   self._hidden,
            "learning_rate":  float(self._lr),
            "gamma":          float(self._gamma),
            "entropy_coeff":  float(self._entropy_coeff),
            "baseline":       self._baseline_type,
            "obs_dim":        self._obs_dim,
            "baseline_value": float(self._baseline_val),
            "trunk_weights":  [w.tolist() for w in self._trunk_w],
            "trunk_biases":   [b.tolist() for b in self._trunk_b],
            "fn_weights":     self._fn_w.tolist(),
            "fn_biases":      self._fn_b.tolist(),
            "sp_weights":     self._sp_w.tolist(),
            "sp_biases":      self._sp_b.tolist(),
        }

    def save(self, path: str) -> None:
        """Write ``to_cfg()`` to YAML at *path*."""
        import yaml as _yaml
        with open(path, "w") as f:
            _yaml.dump(self.to_cfg(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_cfg(cls, cfg: dict, obs_spec: ObsSpec) -> "SC2REINFORCEPolicy":
        obj = cls(
            obs_spec      = obs_spec,
            hidden_sizes  = cfg.get("hidden_sizes",  [128, 64]),
            learning_rate = cfg.get("learning_rate", 0.0003),
            gamma         = cfg.get("gamma",         0.995),
            entropy_coeff = cfg.get("entropy_coeff", 0.05),
            baseline      = cfg.get("baseline",      "running_mean"),
        )
        if "trunk_weights" in cfg:
            obj._trunk_w = [np.array(w, dtype=np.float32) for w in cfg["trunk_weights"]]
            obj._trunk_b = [np.array(b, dtype=np.float32) for b in cfg["trunk_biases"]]
        if "fn_weights" in cfg:
            obj._fn_w = np.array(cfg["fn_weights"], dtype=np.float32)
            obj._fn_b = np.array(cfg["fn_biases"],  dtype=np.float32)
        if "sp_weights" in cfg:
            obj._sp_w = np.array(cfg["sp_weights"], dtype=np.float32)
            obj._sp_b = np.array(cfg["sp_biases"],  dtype=np.float32)
        if "baseline_value" in cfg:
            obj._baseline_val = float(cfg["baseline_value"])
        return obj

    def save_trainer_state(self, path: str) -> None:
        """Persist baseline value and obs_dim to an .npz file."""
        np.savez(path,
                 baseline_val=np.float64(self._baseline_val),
                 obs_dim=np.int64(self._obs_dim))

    def load_trainer_state(self, path: str) -> None:
        """Restore baseline value from an .npz file."""
        with np.load(path) as data:
            saved_obs_dim = int(data["obs_dim"])
            if saved_obs_dim != self._obs_dim:
                raise ValueError(
                    f"SC2REINFORCEPolicy: trainer state obs_dim mismatch — "
                    f"saved={saved_obs_dim}, current={self._obs_dim}. "
                    f"Use --re-initialize to restart from scratch."
                )
            self._baseline_val = float(data["baseline_val"])
        logger.info("[SC2REINFORCEPolicy] trainer state loaded from %s", path)

    @classmethod
    def _construct_or_resume(cls, *, obs_spec, head_names, discrete_actions,
                             weights_file, policy_params, re_initialize):
        pp = policy_params
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as _f:
                cfg = yaml.safe_load(_f) or {}
            if isinstance(cfg, dict) and cfg.get("policy_type") == "sc2_reinforce":
                policy = cls.from_cfg(cfg, obs_spec)
                ts = trainer_state_path(weights_file)
                if os.path.exists(ts):
                    try:
                        policy.load_trainer_state(ts)
                        logger.info("[SC2REINFORCEPolicy] loaded trainer state from %s", ts)
                    except (ValueError, KeyError) as exc:
                        logger.warning("[SC2REINFORCEPolicy] could not load trainer state — %s", exc)
                return policy
        return cls(
            obs_spec      = obs_spec,
            hidden_sizes  = pp.get("hidden_sizes",  [128, 64]),
            learning_rate = pp.get("learning_rate", 0.0003),
            gamma         = pp.get("gamma",         0.995),
            entropy_coeff = pp.get("entropy_coeff", 0.05),
            baseline      = pp.get("baseline",      "running_mean"),
        )


# ---------------------------------------------------------------------------
# SC2CMAESPolicy — CMA-ES over SC2MultiHeadLinearPolicy weight vectors
# ---------------------------------------------------------------------------

#: Number of spatial logits for SC2LSTMPolicy (3×3 grid).
N_LSTM_SPATIAL_CELLS: int = 9

#: 3×3 spatial grid: cell_idx → (x, y) ∈ [0, 1]².
_SPATIAL_GRID: np.ndarray = np.array(
    [(col / 2.0, row / 2.0) for row in range(3) for col in range(3)],
    dtype=np.float32,
)  # shape (9, 2)


@register_policy
class SC2CMAESPolicy(_FrameworkCMAES):
    """(μ/μ_w, λ)-CMA-ES over SC2MultiHeadLinearPolicy — thin framework subclass.

    Delegates all CMA-ES mechanics to :class:`framework.cmaes.CMAESPolicy`.
    Adds available-actions masking during inference.
    """

    POLICY_TYPE = "sc2_cmaes"
    LOOP_TYPE   = "cmaes"
    VALID_POLICY_PARAMS = frozenset({"population_size", "initial_sigma", "eval_episodes"})

    def __init__(
        self,
        obs_spec: ObsSpec = SC2_MINIGAME_OBS_SPEC,
        population_size: int = 30,
        initial_sigma: float = 0.5,
        eval_episodes: int = 2,
        seed: int | None = None,
    ) -> None:
        _tpl = SC2MultiHeadLinearPolicy(obs_spec)
        n_params = len(_tpl.to_flat())

        def _factory(flat: np.ndarray, spec: ObsSpec) -> SC2MultiHeadLinearPolicy:
            return SC2MultiHeadLinearPolicy(spec).with_flat(flat.astype(np.float32))

        super().__init__(
            obs_spec, _factory, n_params,
            population_size=population_size,
            initial_sigma=initial_sigma,
            eval_episodes=eval_episodes,
            seed=seed,
        )
        self._available_fn_ids: set[int] | None = None

    def _build_fn_mask(self, available_fn_ids: set[int] | None) -> np.ndarray:
        mask = np.ones(N_FUNCTION_IDS, dtype=bool)
        if available_fn_ids is not None:
            for i in range(N_FUNCTION_IDS):
                mask[i] = i in available_fn_ids
        if not mask.any():
            mask[0] = True
        return mask

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self._champion is None:
            raise RuntimeError(
                "SC2CMAESPolicy: no champion yet — run at least one generation first."
            )
        norm_obs  = obs / self._obs_spec.scales
        fn_scores = self._champion._fn_weights @ norm_obs
        fn_mask   = self._build_fn_mask(self._available_fn_ids)
        masked    = fn_scores.copy().astype(np.float64)
        masked[~fn_mask] = -np.inf
        fn_idx    = int(np.argmax(masked))
        sp_scores = self._champion._sp_weights @ norm_obs
        x         = _sigmoid(float(sp_scores[0]))
        y         = _sigmoid(float(sp_scores[1]))
        return np.array([fn_idx, x, y, 0.0], dtype=np.float32)

    def on_episode_start(self, **kwargs) -> None:
        info = kwargs.get("info") or {}
        available = info.get("available_fn_ids")
        self._available_fn_ids = set(available) if available is not None else None

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        **kwargs,
    ) -> None:
        info = kwargs.get("info") or {}
        available = info.get("available_fn_ids")
        if available is not None:
            self._available_fn_ids = set(available)

    def on_episode_end(self) -> None:
        pass

    def to_cfg(self) -> dict:
        return {
            "policy_type":     "sc2_cmaes",
            "population_size": self._lam,
            "sigma":           float(self.sigma),
            "obs_dim":         self._obs_spec.dim,
            "eval_episodes":   self._eval_episodes,
            "champion_reward": float(self._champion_reward),
        }

    @classmethod
    def _construct_or_resume(cls, *, obs_spec, head_names, discrete_actions,
                             weights_file, policy_params, re_initialize):
        pp = policy_params
        policy = cls(
            obs_spec        = obs_spec,
            population_size = pp.get("population_size", 30),
            initial_sigma   = pp.get("initial_sigma",   0.5),
            eval_episodes   = pp.get("eval_episodes",   2),
        )
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as _f:
                cfg = yaml.safe_load(_f) or {}
            if isinstance(cfg, dict):
                champion = SC2MultiHeadLinearPolicy.from_cfg(cfg, obs_spec)
                policy.initialize_from_champion(champion)
            ts = trainer_state_path(weights_file)
            if os.path.exists(ts):
                try:
                    policy.load_trainer_state(ts)
                    logger.info("[SC2CMAESPolicy] loaded trainer state from %s", ts)
                except (ValueError, KeyError) as exc:
                    logger.warning(
                        "[SC2CMAESPolicy] could not load trainer state — %s; continuing.",
                        exc,
                    )
        else:
            policy.initialize_random()
        return policy


# ---------------------------------------------------------------------------
# SC2LSTMPolicy — LSTM with two-head output for SC2
# ---------------------------------------------------------------------------

class SC2LSTMPolicy:
    """Single-layer LSTM policy with two-head output for StarCraft 2.

    The output layer maps hidden state → 15 values:
    - First 6 = fn logits  → softmax + available-actions masking → fn_idx
    - Last  9 = spatial logits → softmax → 3×3 grid cell_idx → (x, y)

    Parameters
    ----------
    obs_spec :
        Observation spec describing feature names and normalisation scales.
    hidden_size :
        LSTM hidden state dimensionality (default 64).
    reset_on_episode :
        If True (default), reset ``(h, c)`` to zeros at each episode start.
        Set to False to carry hidden state across truncated resets within the
        same match (useful for long ladder episodes).
    seed :
        Optional RNG seed.
    """

    #: Total output dimension: fn_head (6) + spatial_head (9).
    N_OUTPUT: int = N_FUNCTION_IDS + N_LSTM_SPATIAL_CELLS

    def __init__(
        self,
        obs_spec: ObsSpec,
        hidden_size: int = 64,
        reset_on_episode: bool = True,
        seed: int | None = None,
    ) -> None:
        self._obs_spec        = obs_spec
        self._hidden_size     = hidden_size
        self._obs_dim         = obs_spec.dim
        self._scales          = obs_spec.scales
        self._reset_on_episode = reset_on_episode

        h    = hidden_size
        c_in = h + self._obs_dim
        rng  = np.random.default_rng(seed)
        gain = np.sqrt(2.0 / c_in)

        # LSTM gates: forget, input, cell, output
        self._W_f = rng.standard_normal((h, c_in)).astype(np.float32) * gain
        self._b_f = np.zeros(h, dtype=np.float32)
        self._W_i = rng.standard_normal((h, c_in)).astype(np.float32) * gain
        self._b_i = np.zeros(h, dtype=np.float32)
        self._W_g = rng.standard_normal((h, c_in)).astype(np.float32) * gain
        self._b_g = np.zeros(h, dtype=np.float32)
        self._W_o = rng.standard_normal((h, c_in)).astype(np.float32) * gain
        self._b_o = np.zeros(h, dtype=np.float32)

        # Output head: hidden_size → 15 (6 fn logits + 9 spatial logits)
        self._W_out = (
            rng.standard_normal((self.N_OUTPUT, h)).astype(np.float32)
            * np.sqrt(2.0 / h)
        )
        self._b_out = np.zeros(self.N_OUTPUT, dtype=np.float32)

        self._h = np.zeros(h, dtype=np.float32)
        self._c = np.zeros(h, dtype=np.float32)

        self._rng = np.random.default_rng(seed)

        # Available fn_ids cache (set by on_episode_start / update callers).
        self._available_fn_ids: set[int] | None = None

    # ------------------------------------------------------------------
    # Flat weight interface (for CMA-ES)
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def flat_dim(self) -> int:
        h    = self._hidden_size
        c_in = h + self._obs_dim
        # 4 LSTM gates × (h×c_in weights + h biases) + output (N_OUTPUT×h + N_OUTPUT)
        return 4 * (h * c_in + h) + self.N_OUTPUT * h + self.N_OUTPUT

    def to_flat(self) -> np.ndarray:
        return np.concatenate([
            self._W_f.ravel(), self._b_f,
            self._W_i.ravel(), self._b_i,
            self._W_g.ravel(), self._b_g,
            self._W_o.ravel(), self._b_o,
            self._W_out.ravel(), self._b_out,
        ]).astype(np.float32)

    def with_flat(self, flat: np.ndarray) -> "SC2LSTMPolicy":
        flat = np.asarray(flat, dtype=np.float32)
        if flat.shape[0] != self.flat_dim:
            raise ValueError(
                f"SC2LSTMPolicy.with_flat: expected {self.flat_dim}, got {flat.shape[0]}"
            )
        obj = object.__new__(SC2LSTMPolicy)
        obj._obs_spec         = self._obs_spec
        obj._hidden_size      = self._hidden_size
        obj._obs_dim          = self._obs_dim
        obj._scales           = self._scales
        obj._reset_on_episode = self._reset_on_episode
        obj._rng              = np.random.default_rng()
        obj._available_fn_ids = None

        h    = self._hidden_size
        c_in = h + self._obs_dim
        off  = 0

        def _take(shape: tuple) -> np.ndarray:
            nonlocal off
            n   = int(np.prod(shape))
            out = flat[off: off + n].reshape(shape).copy()
            off += n
            return out

        obj._W_f   = _take((h, c_in))
        obj._b_f   = _take((h,))
        obj._W_i   = _take((h, c_in))
        obj._b_i   = _take((h,))
        obj._W_g   = _take((h, c_in))
        obj._b_g   = _take((h,))
        obj._W_o   = _take((h, c_in))
        obj._b_o   = _take((h,))
        obj._W_out = _take((SC2LSTMPolicy.N_OUTPUT, h))
        obj._b_out = _take((SC2LSTMPolicy.N_OUTPUT,))
        obj._h     = np.zeros(h, dtype=np.float32)
        obj._c     = np.zeros(h, dtype=np.float32)
        return obj

    # ------------------------------------------------------------------
    # LSTM step
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_s = z - z.max()
        e   = np.exp(z_s)
        return e / e.sum()

    def _reset_hidden_state(self) -> None:
        self._h = np.zeros(self._hidden_size, dtype=np.float32)
        self._c = np.zeros(self._hidden_size, dtype=np.float32)

    def on_episode_start(self, **kwargs) -> None:
        if self._reset_on_episode:
            self._reset_hidden_state()
        info = kwargs.get("info") or {}
        available = info.get("available_fn_ids")
        self._available_fn_ids = set(available) if available is not None else None

    def on_episode_end(self) -> None:
        if self._reset_on_episode:
            self._reset_hidden_state()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x  = (obs / self._scales).astype(np.float32)
        hx = np.concatenate([self._h, x])

        _vsigmoid = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
        f = _vsigmoid(self._W_f @ hx + self._b_f)
        i = _vsigmoid(self._W_i @ hx + self._b_i)
        g = np.tanh(self._W_g  @ hx + self._b_g)
        o = _vsigmoid(self._W_o @ hx + self._b_o)

        self._c = f * self._c + i * g
        self._h = o * np.tanh(self._c)

        out         = self._W_out @ self._h + self._b_out
        fn_logits   = out[:N_FUNCTION_IDS].copy().astype(np.float64)
        sp_logits   = out[N_FUNCTION_IDS:].astype(np.float64)

        # Available-actions masking on fn head.
        if self._available_fn_ids is not None:
            for k in range(N_FUNCTION_IDS):
                if k not in self._available_fn_ids:
                    fn_logits[k] = -np.inf
        if not np.isfinite(fn_logits).any():
            fn_logits[0] = 0.0  # fallback to no_op

        fn_probs  = self._softmax(fn_logits)
        fn_idx    = int(self._rng.choice(N_FUNCTION_IDS, p=fn_probs))
        cell_idx  = int(self._rng.choice(N_LSTM_SPATIAL_CELLS,
                                         p=self._softmax(sp_logits)))
        x_out, y_out = float(_SPATIAL_GRID[cell_idx, 0]), float(_SPATIAL_GRID[cell_idx, 1])
        return np.array([fn_idx, x_out, y_out, 0.0], dtype=np.float32)

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        **kwargs,
    ) -> None:
        info = kwargs.get("info") or {}
        available = info.get("available_fn_ids")
        if available is not None:
            self._available_fn_ids = set(available)

    # ------------------------------------------------------------------
    # Serialisation — YAML (to_cfg / from_cfg / save / load)
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        return {
            "policy_type":      "sc2_lstm",
            "hidden_size":      self._hidden_size,
            "reset_on_episode": self._reset_on_episode,
            "obs_dim":          self._obs_dim,
            "W_f":  self._W_f.tolist(),  "b_f":  self._b_f.tolist(),
            "W_i":  self._W_i.tolist(),  "b_i":  self._b_i.tolist(),
            "W_g":  self._W_g.tolist(),  "b_g":  self._b_g.tolist(),
            "W_o":  self._W_o.tolist(),  "b_o":  self._b_o.tolist(),
            "W_out": self._W_out.tolist(), "b_out": self._b_out.tolist(),
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_cfg(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_cfg(cls, cfg: dict, obs_spec: ObsSpec) -> "SC2LSTMPolicy":
        obj = object.__new__(cls)
        obj._obs_spec         = obs_spec
        obj._hidden_size      = int(cfg["hidden_size"])
        obj._obs_dim          = obs_spec.dim
        obj._scales           = obs_spec.scales
        obj._reset_on_episode = bool(cfg.get("reset_on_episode", True))
        obj._available_fn_ids = None
        obj._W_f   = np.array(cfg["W_f"],   dtype=np.float32)
        obj._b_f   = np.array(cfg["b_f"],   dtype=np.float32)
        obj._W_i   = np.array(cfg["W_i"],   dtype=np.float32)
        obj._b_i   = np.array(cfg["b_i"],   dtype=np.float32)
        obj._W_g   = np.array(cfg["W_g"],   dtype=np.float32)
        obj._b_g   = np.array(cfg["b_g"],   dtype=np.float32)
        obj._W_o   = np.array(cfg["W_o"],   dtype=np.float32)
        obj._b_o   = np.array(cfg["b_o"],   dtype=np.float32)
        obj._W_out = np.array(cfg["W_out"], dtype=np.float32)
        obj._b_out = np.array(cfg["b_out"], dtype=np.float32)
        h          = obj._hidden_size
        obj._h     = np.zeros(h, dtype=np.float32)
        obj._c     = np.zeros(h, dtype=np.float32)
        obj._rng   = np.random.default_rng()
        return obj


# ---------------------------------------------------------------------------
# SC2LSTMEvolutionPolicy — CMA-ES outer loop over SC2LSTMPolicy weights
# ---------------------------------------------------------------------------

@register_policy
class SC2LSTMEvolutionPolicy(BasePolicy):
    """(μ/μ_w, λ)-isotropic ES wrapping :class:`SC2LSTMPolicy`.

    Uses the ``_greedy_loop_cmaes`` interface: :meth:`sample_population` /
    :meth:`update_distribution`.  Step size is adapted via the 1/5 success rule.

    Parameters
    ----------
    obs_spec :
        Observation spec for the target environment.
    hidden_size :
        LSTM hidden state dimensionality (default 64).
    population_size :
        λ — offspring evaluated per generation (default 20).
    initial_sigma :
        Starting perturbation scale (default 0.03).
    reset_on_episode :
        Forwarded to each :class:`SC2LSTMPolicy` individual.
    seed :
        Optional RNG seed.
    """

    POLICY_TYPE = "sc2_lstm"
    LOOP_TYPE   = "cmaes"
    VALID_POLICY_PARAMS = frozenset({
        "hidden_size", "population_size", "initial_sigma", "reset_on_episode",
    })

    def __init__(
        self,
        obs_spec: ObsSpec,
        hidden_size: int = 64,
        population_size: int = 20,
        initial_sigma: float = 0.03,
        reset_on_episode: bool = True,
        seed: int | None = None,
    ) -> None:
        self._lam             = int(population_size)
        self._sigma           = float(initial_sigma)
        self._obs_spec        = obs_spec
        self._reset_on_episode = reset_on_episode
        self._rng             = np.random.default_rng(seed)

        self._template  = SC2LSTMPolicy(
            obs_spec         = obs_spec,
            hidden_size      = hidden_size,
            reset_on_episode = reset_on_episode,
        )
        self._flat_dim  = self._template.flat_dim
        self._mean      = self._template.to_flat().astype(np.float64)

        mu             = self._lam // 2
        self._mu       = mu
        raw_w          = np.array(
            [np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)], dtype=np.float64
        )
        self._recomb_w = raw_w / raw_w.sum()

        self._pop: list[np.ndarray]       = []
        self._champion: SC2LSTMPolicy | None  = None
        self._champion_reward: float          = float("-inf")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def population_size(self) -> int:
        return self._lam

    @property
    def champion_reward(self) -> float:
        return self._champion_reward

    @property
    def sigma(self) -> float:
        return self._sigma

    # ------------------------------------------------------------------
    # Champion seeding
    # ------------------------------------------------------------------

    def initialize_from_champion(self, champion: SC2LSTMPolicy) -> None:
        if champion.flat_dim != self._flat_dim:
            raise ValueError(
                f"SC2LSTMEvolutionPolicy: flat_dim mismatch — "
                f"expected {self._flat_dim}, got {champion.flat_dim}. "
                f"Use --re-initialize to restart from scratch."
            )
        self._champion = champion
        self._mean     = champion.to_flat().astype(np.float64)
        logger.info("[SC2LSTMEvolutionPolicy] seeded mean from champion")

    # ------------------------------------------------------------------
    # CMA-ES loop interface
    # ------------------------------------------------------------------

    def sample_population(self) -> list[SC2LSTMPolicy]:
        self._pop = []
        for _ in range(self._lam):
            z = self._rng.standard_normal(self._flat_dim)
            self._pop.append(self._mean + self._sigma * z)
        return [self._template.with_flat(x) for x in self._pop]

    def update_distribution(self, rewards: list[float]) -> bool:
        if len(rewards) != self._lam:
            raise ValueError(f"Expected {self._lam} rewards, got {len(rewards)}")
        if len(self._pop) != self._lam:
            raise RuntimeError("update_distribution() called before sample_population().")

        order     = np.argsort(rewards)[::-1]
        prev_best = self._champion_reward
        improved  = False

        best_r = rewards[order[0]]
        if best_r > self._champion_reward:
            self._champion_reward = best_r
            self._champion        = self._template.with_flat(
                np.array(self._pop[order[0]], dtype=np.float32)
            )
            improved = True

        elite_xs   = np.stack([self._pop[order[i]] for i in range(self._mu)])
        self._mean = np.einsum("i,ij->j", self._recomb_w, elite_xs)

        n_success    = sum(1 for r in rewards if r > prev_best)
        success_rate = n_success / self._lam
        self._sigma  = float(np.clip(
            self._sigma * (1.2 if success_rate > 0.2 else 0.85),
            1e-6, 1e2,
        ))
        return improved

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self._champion is None:
            raise RuntimeError(
                "SC2LSTMEvolutionPolicy: no champion yet — "
                "run at least one generation first."
            )
        return self._champion(obs)

    def on_episode_start(self, **kwargs) -> None:
        if self._champion is not None:
            self._champion.on_episode_start(**kwargs)

    def on_episode_end(self) -> None:
        if self._champion is not None:
            self._champion.on_episode_end()

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        **kwargs,
    ) -> None:
        if self._champion is not None:
            self._champion.update(obs, action, reward, next_obs, done, **kwargs)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        return {
            "policy_type":      "sc2_lstm",
            "hidden_size":      self._template._hidden_size,
            "reset_on_episode": self._reset_on_episode,
            "obs_dim":          self._obs_spec.dim,
            "sigma":            self._sigma,
            "champion_reward":  float(self._champion_reward),
        }

    def save(self, path: str) -> None:
        if self._champion is not None:
            self._champion.save(path)

    def save_trainer_state(self, path: str) -> None:
        np.savez(
            path,
            mean      = self._mean,
            sigma     = np.float64(self._sigma),
            flat_dim  = np.int64(self._flat_dim),
        )

    def load_trainer_state(self, path: str) -> None:
        with np.load(path) as data:
            saved_dim = int(data["flat_dim"])
            if saved_dim != self._flat_dim:
                raise ValueError(
                    f"SC2LSTMEvolutionPolicy: flat_dim mismatch — "
                    f"saved={saved_dim}, current={self._flat_dim}. "
                    f"Use --re-initialize to restart from scratch."
                )
            self._mean  = data["mean"].astype(np.float64)
            self._sigma = float(data["sigma"])
        logger.info("[SC2LSTMEvolutionPolicy] trainer state loaded from %s", path)

    @classmethod
    def _construct_or_resume(cls, *, obs_spec, head_names, discrete_actions,
                             weights_file, policy_params, re_initialize):
        pp = policy_params
        policy = cls(
            obs_spec         = obs_spec,
            hidden_size      = pp.get("hidden_size",      64),
            population_size  = pp.get("population_size",  20),
            initial_sigma    = pp.get("initial_sigma",    0.03),
            reset_on_episode = pp.get("reset_on_episode", True),
        )
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as _f:
                cfg = yaml.safe_load(_f) or {}
            if isinstance(cfg, dict) and cfg.get("policy_type") == "sc2_lstm":
                champion = SC2LSTMPolicy.from_cfg(cfg, obs_spec)
                policy.initialize_from_champion(champion)
            ts = trainer_state_path(weights_file)
            if os.path.exists(ts):
                try:
                    policy.load_trainer_state(ts)
                    logger.info("[SC2LSTMEvolutionPolicy] loaded trainer state from %s", ts)
                except (ValueError, KeyError) as exc:
                    logger.warning(
                        "[SC2LSTMEvolutionPolicy] could not load trainer state — %s; continuing.",
                        exc,
                    )
        return policy
