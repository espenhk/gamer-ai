"""SC2-specific policy classes.

SC2MultiHeadLinearPolicy
    Multi-output linear policy for SC2.  Two weight matrices are maintained:

    * **fn_idx head** — shape ``(N_FUNCTION_IDS, obs_dim)`` → one score per
      function ID.  ``argmax`` gives the selected function.
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

import numpy as np
import yaml

from framework.cmaes import CMAESPolicy as _FrameworkCMAES
from framework.dqn import DQNPolicy as _FrameworkDQN
from framework.lstm import LSTMEvolutionPolicy as _FrameworkLSTMEvo
from framework.obs_spec import ObsSpec
from framework.policies import BasePolicy, GeneticPolicy, register_policy, trainer_state_path
from framework.reinforce import TwoHeadREINFORCEPolicy as _FrameworkTwoHeadREINFORCE, _GradEntry  # noqa: F401 — _GradEntry re-exported for test compatibility
from games.sc2.actions import DISCRETE_ACTIONS, FUNCTION_IDS, build_available_actions_mask, fn_ids_for_race
from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants derived from the action definitions
# ---------------------------------------------------------------------------

#: Number of function IDs exposed by the SC2 action set.
N_FUNCTION_IDS: int = len(FUNCTION_IDS)

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
    race :
        Agent race (``"terran"``, ``"protoss"``, ``"zerg"``, or
        ``"random"``).  fn_ids outside the race's applicable set are
        permanently masked to ``-inf`` so the policy never selects them.
        Defaults to ``"random"`` (all fn_ids enabled).
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        fn_weights: np.ndarray | None = None,
        spatial_weights: np.ndarray | None = None,
        race: str = "random",
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

        # Permanent race mask — fn_ids outside the race's applicable set are
        # always masked, regardless of what PySC2 reports as available.
        self._race: str = race
        self._race_fn_ids: frozenset[int] = fn_ids_for_race(race)

        # Per-step cache of fn_ids from the most recent on_episode_start() /
        # update() call.  None means all functions are available.
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

        # Apply permanent race mask and per-step availability mask.
        for i in range(N_FUNCTION_IDS):
            if i not in self._race_fn_ids:
                fn_scores[i] = -np.inf
            elif self._available_fn_ids is not None and i not in self._available_fn_ids:
                fn_scores[i] = -np.inf
        # If all masked (e.g. race set is empty or nothing available), fall
        # back to no_op (idx 0) which is always a valid PySC2 action.
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
    def from_cfg(
        cls,
        cfg: dict,
        obs_spec: ObsSpec,
        race: str = "random",
    ) -> "SC2MultiHeadLinearPolicy":
        """Reconstruct a policy from a ``to_cfg()`` dict.

        Unknown keys and missing observation features default to 0.0 so that
        the policy can load configs created with a different obs_spec dimension
        (same migration behaviour as ``WeightedLinearPolicy``).  Pre-#122
        weight files (with ``spatial_{0..8}_weights`` keys) silently migrate
        to all-zero ``x_weights`` / ``y_weights`` because the old 9-cell
        argmax encoding has no meaningful projection onto the new continuous
        head.

        The *race* parameter gates which fn_ids are active at inference time;
        it does not affect which weights are loaded (all rows are always
        restored from the file so the champion can be evaluated under any race
        without losing weights).
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

        return cls(obs_spec, fn_weights=fn_weights, spatial_weights=sp_weights,
                   race=race)

    @classmethod
    def load(
        cls,
        path: str,
        obs_spec: ObsSpec,
        race: str = "random",
    ) -> "SC2MultiHeadLinearPolicy":
        """Load from a YAML file written by :meth:`save`."""
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        return cls.from_cfg(cfg, obs_spec, race=race)

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
        return SC2MultiHeadLinearPolicy(self._obs_spec, fn_weights, sp_weights,
                                        race=self._race)

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
        race: str = "random",
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
        self._race = race

    # ------------------------------------------------------------------
    # Individual factory — override to use SC2MultiHeadLinearPolicy
    # ------------------------------------------------------------------

    def _make_member(self, cfg: dict) -> SC2MultiHeadLinearPolicy:  # type: ignore[override]
        """Build an SC2MultiHeadLinearPolicy from a ``to_cfg()`` dict."""
        return SC2MultiHeadLinearPolicy.from_cfg(cfg, self._obs_spec, race=self._race)

    # ------------------------------------------------------------------
    # Population seed from a saved champion file
    # ------------------------------------------------------------------

    def initialize_from_file(self, path: str) -> None:
        """Load champion from YAML and seed the population by mutation."""
        champion = SC2MultiHeadLinearPolicy.load(path, self._obs_spec, race=self._race)
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
            race            = cfg.get("race", "random"),
        )
        champion_cfg = cfg.get("champion_weights")
        if champion_cfg and isinstance(champion_cfg, dict):
            policy._champion = SC2MultiHeadLinearPolicy.from_cfg(
                champion_cfg, obs_spec, race=policy._race
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
# SC2REINFORCEPolicy — thin subclass of framework TwoHeadREINFORCEPolicy
# ---------------------------------------------------------------------------

def _sc2_action_fn(fn_idx: int, sp_sig: np.ndarray) -> np.ndarray:
    """Convert sampled fn_idx and spatial sigmoid outputs to SC2 action vector."""
    return np.array([fn_idx, float(sp_sig[0]), float(sp_sig[1]), 0.0], dtype=np.float32)


def _sc2_available_fn_ids_fn(info: dict) -> "set[int] | None":
    """Extract available fn_ids set from environment step info dict."""
    return info.get("available_fn_ids") if info else None


@register_policy
class SC2REINFORCEPolicy(_FrameworkTwoHeadREINFORCE):
    """REINFORCE (Monte Carlo Policy Gradient) with a two-head MLP for SC2.

    Thin subclass of :class:`framework.reinforce.TwoHeadREINFORCEPolicy` that
    pins the SC2 action encoding (``fn_idx`` softmax + sigmoid ``(x, y)``
    spatial head) and the SC2-specific available-actions hook.

    All gradient math is inherited unchanged from the framework base class.
    Use ``POLICY_TYPE="sc2_reinforce"`` in ``training_params.yaml``.

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
        Entropy regularisation weight for the fn head (default ``0.05``).
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

    @classmethod
    def compatible_with(cls, game_name: str) -> tuple[bool, str | None]:
        if game_name != "sc2":
            return False, "This policy is SC2-specific; use game='sc2'."
        return True, None

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
        super().__init__(
            obs_spec             = obs_spec,
            n_fn_ids             = N_FUNCTION_IDS,
            n_spatial            = N_SPATIAL_ROWS,
            action_fn            = _sc2_action_fn,
            hidden_sizes         = hidden_sizes,
            learning_rate        = learning_rate,
            gamma                = gamma,
            entropy_coeff        = entropy_coeff,
            baseline             = baseline,
            available_fn_ids_fn  = _sc2_available_fn_ids_fn,
            seed                 = seed,
        )

    def to_cfg(self) -> dict:
        cfg = super().to_cfg()
        cfg["policy_type"] = "sc2_reinforce"
        return cfg

    @classmethod
    def from_cfg(cls, cfg: dict, obs_spec: ObsSpec) -> "SC2REINFORCEPolicy":  # type: ignore[override]
        """Reconstruct from a ``to_cfg()`` dict without requiring caller-supplied hooks."""
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
        race: str = "random",
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
        self._race: str = race
        self._race_fn_ids: frozenset[int] = fn_ids_for_race(race)
        self._available_fn_ids: set[int] | None = None


    def _build_fn_mask(self, available_fn_ids: set[int] | None) -> np.ndarray:
        mask = np.ones(N_FUNCTION_IDS, dtype=bool)
        for i in range(N_FUNCTION_IDS):
            if i not in self._race_fn_ids:
                mask[i] = False
            elif available_fn_ids is not None and i not in available_fn_ids:
                mask[i] = False
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
        race: str = "random",
    ) -> None:
        self._obs_spec        = obs_spec
        self._hidden_size     = hidden_size
        self._obs_dim         = obs_spec.dim
        self._scales          = obs_spec.scales
        self._reset_on_episode = reset_on_episode
        self._race: str                  = race
        self._race_fn_ids: frozenset[int] = fn_ids_for_race(race)

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
        obj._race             = self._race
        obj._race_fn_ids      = self._race_fn_ids
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

        # Permanent race mask + per-step available-actions mask.
        for k in range(N_FUNCTION_IDS):
            if k not in self._race_fn_ids:
                fn_logits[k] = -np.inf
            elif self._available_fn_ids is not None and k not in self._available_fn_ids:
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
    def from_cfg(cls, cfg: dict, obs_spec: ObsSpec, race: str = "random") -> "SC2LSTMPolicy":
        obj = object.__new__(cls)
        obj._obs_spec         = obs_spec
        obj._hidden_size      = int(cfg["hidden_size"])
        obj._obs_dim          = obs_spec.dim
        obj._scales           = obs_spec.scales
        obj._reset_on_episode = bool(cfg.get("reset_on_episode", True))
        obj._race             = race
        obj._race_fn_ids      = fn_ids_for_race(obj._race)
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
# SC2LSTMEvolutionPolicy — thin subclass of framework LSTMEvolutionPolicy
# ---------------------------------------------------------------------------

@register_policy
class SC2LSTMEvolutionPolicy(_FrameworkLSTMEvo):
    """(μ/μ_w, λ)-isotropic ES wrapping :class:`SC2LSTMPolicy`.

    Thin subclass of :class:`framework.lstm.LSTMEvolutionPolicy` that uses
    :class:`SC2LSTMPolicy` as the inner individual instead of
    :class:`framework.lstm.LSTMCore`.  All evolutionary mechanics
    (``sample_population``, ``update_distribution``, step-size adaptation,
    trainer-state serialisation) are inherited from the framework class.

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

    @classmethod
    def compatible_with(cls, game_name: str) -> tuple[bool, str | None]:
        if game_name != "sc2":
            return False, "This policy is SC2-specific; use game='sc2'."
        return True, None

    def __init__(
        self,
        obs_spec: ObsSpec,
        hidden_size: int = 64,
        population_size: int = 20,
        initial_sigma: float = 0.03,
        reset_on_episode: bool = True,
        seed: int | None = None,
        race: str = "random",
    ) -> None:
        template = SC2LSTMPolicy(
            obs_spec         = obs_spec,
            hidden_size      = hidden_size,
            reset_on_episode = reset_on_episode,
            race             = race,
        )
        super().__init__(
            obs_spec         = obs_spec,
            hidden_size      = hidden_size,
            population_size  = population_size,
            initial_sigma    = initial_sigma,
            seed             = seed,
            _template        = template,
        )
        self._reset_on_episode = reset_on_episode

    def to_cfg(self) -> dict:
        cfg = super().to_cfg()
        cfg["policy_type"]      = "sc2_lstm"
        cfg["reset_on_episode"] = self._reset_on_episode
        return cfg

    def save(self, path: str) -> None:
        """Save champion in SC2LSTMPolicy YAML format."""
        if self._champion is not None:
            self._champion.save(path)

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

