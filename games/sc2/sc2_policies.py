"""SC2-specific policy classes.

SC2MultiHeadLinearPolicy
    Multi-output linear policy for SC2.  Two weight matrices are maintained:

    * **fn_idx head** — shape ``(N_FUNCTION_IDS, obs_dim)`` → 6 scores, one per
      available function ID.  ``argmax`` gives the selected function.
    * **spatial head** — shape ``(N_GRID_CELLS, obs_dim)`` → 9 scores, one per
      cell in the 3×3 action grid.  ``argmax`` gives the target screen cell.

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
from typing import NamedTuple

import numpy as np
import yaml

from framework.obs_spec import ObsSpec
from framework.policies import BasePolicy, GeneticPolicy
from games.sc2.actions import DISCRETE_ACTIONS, FUNCTION_IDS
from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants derived from the action definitions
# ---------------------------------------------------------------------------

#: Number of function IDs exposed by the SC2 action set.
N_FUNCTION_IDS: int = len(FUNCTION_IDS)   # 6

#: Number of spatial grid cells in the 3×3 action grid.
N_GRID_CELLS: int = len(DISCRETE_ACTIONS)  # 9

#: Head-name prefixes — one row per output neuron stored as a separate YAML key.
_FN_HEAD_NAMES: list[str]      = [f"fn_idx_{i}" for i in range(N_FUNCTION_IDS)]
_SPATIAL_HEAD_NAMES: list[str] = [f"spatial_{i}" for i in range(N_GRID_CELLS)]
_ALL_ROW_NAMES: list[str]      = _FN_HEAD_NAMES + _SPATIAL_HEAD_NAMES

# Pre-build the (x, y) coordinate for each grid cell from DISCRETE_ACTIONS.
_GRID_XY: list[tuple[float, float]] = [
    (float(DISCRETE_ACTIONS[i, 1]), float(DISCRETE_ACTIONS[i, 2]))
    for i in range(N_GRID_CELLS)
]


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
        Weight matrix of shape ``(N_GRID_CELLS, obs_dim)``.  If *None* a
        random initialisation is used.
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
            else rng.standard_normal((N_GRID_CELLS, obs_dim)).astype(np.float32)
        )

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Select action given observation.

        Returns
        -------
        np.ndarray
            4-vector ``[fn_idx, x, y, 0]`` compatible with ``SC2Env``.
        """
        norm_obs  = obs / self._obs_spec.scales
        fn_scores = self._fn_weights  @ norm_obs   # (N_FUNCTION_IDS,)
        sp_scores = self._sp_weights  @ norm_obs   # (N_GRID_CELLS,)
        fn_idx    = int(np.argmax(fn_scores))
        cell_idx  = int(np.argmax(sp_scores))
        x, y      = _GRID_XY[cell_idx]
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
            spatial_0_weights: {obs_name_0: float, ...}
            ...
            spatial_8_weights: {obs_name_0: float, ...}

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
        (same migration behaviour as ``WeightedLinearPolicy``).
        """
        names   = obs_spec.names
        obs_dim = obs_spec.dim

        fn_weights = np.zeros((N_FUNCTION_IDS, obs_dim), dtype=np.float32)
        sp_weights = np.zeros((N_GRID_CELLS,   obs_dim), dtype=np.float32)

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

        Layout: ``[fn_row_0 | fn_row_1 | … | fn_row_5 | sp_row_0 | … | sp_row_8]``
        Total length: ``(N_FUNCTION_IDS + N_GRID_CELLS) × obs_dim``.
        """
        return np.concatenate(
            [self._fn_weights.ravel(), self._sp_weights.ravel()]
        ).astype(np.float32)

    def with_flat(self, flat: np.ndarray) -> "SC2MultiHeadLinearPolicy":
        """Return a new policy whose weights are set from a flat vector."""
        obs_dim    = self._obs_spec.dim
        fn_size    = N_FUNCTION_IDS * obs_dim
        fn_weights = flat[:fn_size].reshape(N_FUNCTION_IDS, obs_dim).astype(np.float32)
        sp_weights = flat[fn_size:].reshape(N_GRID_CELLS,   obs_dim).astype(np.float32)
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
               next_obs: np.ndarray, done: bool, **kwargs) -> None:
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
    (``fn_idx_{i}_weights`` + ``spatial_{i}_weights`` keys), so
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
    sp_probs:           np.ndarray # softmax spatial probabilities (N_GRID_CELLS,)
    cell_idx:           int        # sampled spatial cell index
    fn_mask:            np.ndarray # bool mask: True = available fn (N_FUNCTION_IDS,)


class SC2REINFORCEPolicy(BasePolicy):
    """REINFORCE (Monte Carlo Policy Gradient) with a two-head MLP for SC2.

    The network has a **shared trunk** (hidden FC layers + ReLU) followed by
    two independent linear output heads:

    * **fn_head** — 6 logits, softmax → ``fn_idx ∈ {0…5}``; unavailable
      function IDs are masked to ``-∞`` before sampling.
    * **spatial_head** — 9 logits, softmax → grid-cell index → (x, y).

    Both heads are trained jointly via REINFORCE:

    .. code-block:: text

        loss_t = -log π_fn(fn_idx | obs) × G_t
               - log π_sp(cell  | obs) × G_t

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

    def on_episode_start(self) -> None:
        """Reset episode buffers and (optionally) clear the cached available fns."""
        self._ep_grads.clear()
        self._ep_rewards.clear()

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
        fn_idx    = int(np.random.choice(N_FUNCTION_IDS, p=fn_probs))

        # --- spatial head ---
        sp_logits = self._sp_w @ h_last + self._sp_b  # (N_GRID_CELLS,)
        sp_probs  = self._softmax(sp_logits)
        cell_idx  = int(np.random.choice(N_GRID_CELLS, p=sp_probs))

        self._ep_grads.append(
            _GradEntry(
                trunk_layer_inputs=l_in,
                trunk_pre_relu=pre_r,
                h_last=h_last.copy(),
                fn_probs=fn_probs.copy(),
                fn_idx=fn_idx,
                sp_probs=sp_probs.copy(),
                cell_idx=cell_idx,
                fn_mask=fn_mask.copy(),
            )
        )

        x, y = _GRID_XY[cell_idx]
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
            sp_probs = entry.sp_probs
            cell_idx = entry.cell_idx
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

            # --- spatial head gradient ---
            delta_sp = -sp_probs.copy().astype(np.float64)
            delta_sp[cell_idx] += 1.0
            delta_sp *= advantage

            if self._entropy_coeff > 0.0:
                log_p_sp  = np.log(sp_probs.astype(np.float64) + 1e-8)
                H_sp      = -float(np.dot(sp_probs, log_p_sp))
                ent_grad_sp = -(sp_probs.astype(np.float64) * (log_p_sp + H_sp))
                delta_sp += self._entropy_coeff * ent_grad_sp

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

