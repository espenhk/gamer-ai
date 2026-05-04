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

import numpy as np
import yaml

from framework.obs_spec import ObsSpec
from framework.policies import GeneticPolicy
from games.sc2.actions import FUNCTION_IDS
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

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Select action given observation.

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
        fn_idx    = int(np.argmax(fn_scores))
        x         = _sigmoid(float(sp_scores[0]))
        y         = _sigmoid(float(sp_scores[1]))
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

    def on_episode_start(self) -> None:
        """No-op — required by training loop interface."""

    def update(self, obs: np.ndarray, action: np.ndarray, reward: float,
               next_obs: np.ndarray, done: bool) -> None:
        """No-op — evolutionary policy; updated between episodes."""


# ---------------------------------------------------------------------------
# SC2GeneticPolicy
# ---------------------------------------------------------------------------

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
