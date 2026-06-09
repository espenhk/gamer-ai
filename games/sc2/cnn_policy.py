"""SC2 CNN policy evolved by isotropic evolutionary strategy.

Architecture (per the issue spec)::

    spatial (C, 64, 64)
        │
    Conv2d(C → 32, 3×3, relu)
    Conv2d(32 → 64, 3×3, relu)
    AdaptiveAvgPool2d(4×4)
    Flatten → (1024,)
        │
    Concat with flat obs (obs_dim,)
        │
    FC(1024 + obs_dim → 256, relu)
        │
      ┌───┴────┐
    fn_head   spatial_head
      (6,)       (9,)

Weights are evolved by the same isotropic ES used by
:class:`games.sc2.sc2_policies.SC2LSTMEvolutionPolicy` (sample_population /
update_distribution interface, 1/5 success-rule sigma adaptation).
No backprop — purely evolutionary.

Usage with ``policy_type: sc2_cnn`` in training_params.yaml.  Requires
``screen_layers`` (non-empty) so that ``SC2Env`` returns dict observations.

The conv math helpers and ES outer loop live in
:mod:`framework.cnn_policy`; only SC2-specific action heads and race masking
are defined here.
"""

from __future__ import annotations

import logging

import numpy as np

from framework.cnn_policy import (
    CNNBackbone,
    _CNNESBase,
)
from framework.obs_spec import ObsSpec
from framework.policies import register_policy
from games.sc2.actions import DISCRETE_ACTIONS, FUNCTION_IDS, fn_ids_for_race

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_FUNCS = len(FUNCTION_IDS)  # 6
_N_SPATIAL_CELLS = len(DISCRETE_ACTIONS)  # 9

# Pre-build (x, y) coordinates for each grid cell from DISCRETE_ACTIONS.
_GRID_XY: list[tuple[float, float]] = [
    (float(DISCRETE_ACTIONS[i, 1]), float(DISCRETE_ACTIONS[i, 2])) for i in range(_N_SPATIAL_CELLS)
]


# ---------------------------------------------------------------------------
# SC2CNNModel — the individual network evaluated each episode
# ---------------------------------------------------------------------------


class SC2CNNModel:
    """CNN model that maps dict obs → SC2 action.

    Callable as a policy individual during ES evaluation.  Uses
    :class:`~framework.cnn_policy.CNNBackbone` for feature extraction and
    adds SC2-specific fn_idx / spatial output heads with race masking.

    Parameters
    ----------
    n_channels :
        Number of spatial input channels (len(screen_layers) +
        len(minimap_layers)).
    obs_spec :
        Flat observation spec — used for normalisation.
    seed :
        RNG seed for weight initialisation.
    race :
        SC2 race string for the permanent race-action mask.
    """

    # Architecture hyperparams (fixed; matching the issue spec).
    _CONV1_OUT = 32
    _CONV2_OUT = 64
    _POOL_H = 4
    _POOL_W = 4
    _KERNEL = 3
    _FC_DIM = 256

    def __init__(
        self,
        n_channels: int,
        obs_spec: ObsSpec,
        seed: int | None = None,
        race: str = "random",
    ) -> None:
        self._n_channels = n_channels
        self._obs_spec = obs_spec
        self._race: str = race
        self._race_fn_ids: frozenset[int] = fn_ids_for_race(race)

        self._backbone = CNNBackbone(
            n_channels=n_channels,
            obs_spec=obs_spec,
            conv1_out=self._CONV1_OUT,
            conv2_out=self._CONV2_OUT,
            pool_h=self._POOL_H,
            pool_w=self._POOL_W,
            kernel=self._KERNEL,
            fc_dim=self._FC_DIM,
            seed=seed,
        )

        rng = np.random.default_rng(seed)
        fc_dim = self._FC_DIM

        def _he(shape: tuple) -> np.ndarray:
            fan_in = int(np.prod(shape[1:]))
            return rng.standard_normal(shape).astype(np.float32) * np.sqrt(2.0 / fan_in)

        self.W_fn = _he((_N_FUNCS, fc_dim))
        self.b_fn = np.zeros(_N_FUNCS, dtype=np.float32)
        self.W_sp = _he((_N_SPATIAL_CELLS, fc_dim))
        self.b_sp = np.zeros(_N_SPATIAL_CELLS, dtype=np.float32)
        self._available_fn_ids: set[int] | None = None

    # Pass-through properties so callers (including tests) can access backbone
    # weights directly — e.g. ``model.W1.fill(0.0)`` for white-box test setup.
    @property
    def W1(self) -> np.ndarray:
        return self._backbone.W1

    @property
    def b1(self) -> np.ndarray:
        return self._backbone.b1

    @property
    def W2(self) -> np.ndarray:
        return self._backbone.W2

    @property
    def b2(self) -> np.ndarray:
        return self._backbone.b2

    @property
    def W3(self) -> np.ndarray:
        return self._backbone.W3

    @property
    def b3(self) -> np.ndarray:
        return self._backbone.b3

    @property
    def flat_dim(self) -> int:
        fc_dim = self._FC_DIM
        return self._backbone.param_dim + _N_FUNCS * fc_dim + _N_FUNCS + _N_SPATIAL_CELLS * fc_dim + _N_SPATIAL_CELLS

    def to_flat(self) -> np.ndarray:
        return np.concatenate(
            [
                self._backbone.to_flat(),
                self.W_fn.ravel(),
                self.b_fn,
                self.W_sp.ravel(),
                self.b_sp,
            ]
        ).astype(np.float32)

    def with_flat(self, flat: np.ndarray) -> "SC2CNNModel":
        flat = np.asarray(flat, dtype=np.float32)
        if flat.shape[0] != self.flat_dim:
            raise ValueError(f"SC2CNNModel.with_flat: expected {self.flat_dim} params, got {flat.shape[0]}")
        obj = object.__new__(SC2CNNModel)
        obj._n_channels = self._n_channels
        obj._obs_spec = self._obs_spec
        obj._race = self._race
        obj._race_fn_ids = self._race_fn_ids

        n_bb = self._backbone.param_dim
        obj._backbone = self._backbone.with_flat(flat[:n_bb])

        off = n_bb
        fc_dim = self._FC_DIM

        def _take(shape: tuple) -> np.ndarray:
            nonlocal off
            n = int(np.prod(shape))
            out = flat[off : off + n].reshape(shape).copy()
            off += n
            return out

        obj.W_fn = _take((_N_FUNCS, fc_dim))
        obj.b_fn = _take((_N_FUNCS,))
        obj.W_sp = _take((_N_SPATIAL_CELLS, fc_dim))
        obj.b_sp = _take((_N_SPATIAL_CELLS,))
        obj._available_fn_ids = set(self._available_fn_ids) if self._available_fn_ids is not None else None
        return obj

    def forward(
        self,
        spatial: np.ndarray,
        flat_obs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass.

        Parameters
        ----------
        spatial : (C, H, W) float32  — already normalised to ~[0, 1]
        flat_obs : (obs_dim,) float32 — raw (un-normalised) flat observation

        Returns
        -------
        fn_scores : (N_FUNCS,) logits
        sp_scores : (N_SPATIAL_CELLS,) logits
        """
        h = self._backbone.extract(spatial, flat_obs)
        fn_scores = self.W_fn @ h + self.b_fn
        sp_scores = self.W_sp @ h + self.b_sp
        return fn_scores, sp_scores

    # ------------------------------------------------------------------
    # Policy interface (used when this model is an ES individual)
    # ------------------------------------------------------------------

    def __call__(self, obs: dict | np.ndarray) -> np.ndarray:
        if isinstance(obs, dict):
            flat_obs = obs["flat"]
            spatial = obs["spatial"]
        else:
            raise TypeError(
                "SC2CNNModel expects a dict observation with keys 'flat' and 'spatial'.  Got: " + type(obs).__name__
            )
        fn_scores, sp_scores = self.forward(spatial, flat_obs)
        # Apply permanent race mask, then per-step availability mask.
        for i in range(_N_FUNCS):
            if i not in self._race_fn_ids:
                fn_scores[i] = -np.inf
            elif self._available_fn_ids is not None and i not in self._available_fn_ids:
                fn_scores[i] = -np.inf
        if not np.isfinite(fn_scores).any():
            fn_scores[0] = 0.0
        fn_idx = int(np.argmax(fn_scores))
        cell_idx = int(np.argmax(sp_scores))
        x, y = _GRID_XY[cell_idx]
        return np.array([fn_idx, x, y, 0.0], dtype=np.float32)

    def on_episode_start(self, **kwargs) -> None:
        info = kwargs.get("info") or {}
        available = info.get("available_fn_ids")
        self._available_fn_ids = set(available) if available is not None else None

    def on_episode_end(self) -> None:
        pass

    def update(
        self,
        obs: dict | np.ndarray,
        action: np.ndarray | int,
        reward: float,
        next_obs: dict | np.ndarray,
        done: bool,
        **kwargs,
    ) -> None:
        info = kwargs.get("info") or {}
        available = info.get("available_fn_ids")
        self._available_fn_ids = set(available) if available is not None else None


# ---------------------------------------------------------------------------
# SC2CNNEvolutionPolicy — isotropic ES outer optimiser wrapping SC2CNNModel
# ---------------------------------------------------------------------------


@register_policy
class SC2CNNEvolutionPolicy(_CNNESBase):
    """Isotropic-ES outer optimiser for :class:`SC2CNNModel`.

    Inherits the shared ES loop from
    :class:`~framework.cnn_policy._CNNESBase`; only SC2-specific construction
    and persistence are overridden here.

    Parameters
    ----------
    n_channels :
        Number of spatial input channels.
    obs_spec :
        Flat observation spec.
    population_size :
        λ — offspring evaluated per generation (default 20).
    initial_sigma :
        Starting perturbation scale (default 0.01; CNN weight space is
        large, so a smaller sigma than the LSTM policy is appropriate).
    eval_episodes :
        Episodes per individual per generation (averaged for fitness).
    seed :
        RNG seed.
    race :
        SC2 agent race string for the permanent race-action mask.
    """

    POLICY_TYPE = "sc2_cnn"
    LOOP_TYPE = "cmaes"
    VALID_POLICY_PARAMS = frozenset({"population_size", "initial_sigma", "eval_episodes"})

    @classmethod
    def compatible_with(cls, game_name: str) -> tuple[bool, str | None]:
        if game_name != "sc2":
            return False, "This policy is SC2-specific; use game='sc2'."
        return True, None

    def __init__(
        self,
        n_channels: int,
        obs_spec: ObsSpec,
        population_size: int = 20,
        initial_sigma: float = 0.01,
        eval_episodes: int = 1,
        seed: int | None = None,
        race: str = "random",
    ) -> None:
        self._obs_spec = obs_spec
        template = SC2CNNModel(n_channels=n_channels, obs_spec=obs_spec, seed=seed, race=race)
        self._init_es(
            template=template,
            population_size=population_size,
            initial_sigma=initial_sigma,
            eval_episodes=eval_episodes,
            rng=np.random.default_rng(seed),
        )
        logger.info(
            "[SC2CNNEvolutionPolicy] n_channels=%d  obs_dim=%d  flat_dim=%d  pop=%d  sigma=%.4f",
            n_channels,
            obs_spec.dim,
            self._flat_dim,
            self._lam,
            self._sigma,
        )

    def to_cfg(self) -> dict:
        return {
            "policy_type": "sc2_cnn",
            "n_channels": self._template._n_channels,
            "obs_dim": self._obs_spec.dim,
            "population_size": self._lam,
            "sigma": float(self._sigma),
            "eval_episodes": self._eval_episodes,
            "champion_reward": float(self._champion_reward),
        }

    def save(self, path: str) -> None:
        """Save champion weights as a numpy .npz file."""
        if self._champion is not None:
            flat = self._champion.to_flat()
            np.savez(
                path.replace(".yaml", ".npz") if path.endswith(".yaml") else path,
                flat=flat,
                n_channels=np.int64(self._template._n_channels),
                obs_dim=np.int64(self._obs_spec.dim),
                flat_dim=np.int64(self._flat_dim),
            )

    @classmethod
    def _construct_or_resume(
        cls, *, obs_spec, head_names, discrete_actions, weights_file, policy_params, re_initialize
    ):
        # n_channels (number of spatial layers) is injected by the SC2 adapter
        # via the private ``_n_channels`` policy_param — see games/sc2/adapter.py.
        n_channels = int(policy_params.get("_n_channels", 0))
        if n_channels == 0:
            raise ValueError("sc2_cnn requires at least one spatial layer.  Set screen_layers in training_params.yaml.")
        policy = cls(
            n_channels=n_channels,
            obs_spec=obs_spec,
            population_size=policy_params.get("population_size", 20),
            initial_sigma=policy_params.get("initial_sigma", 0.01),
            eval_episodes=policy_params.get("eval_episodes", 1),
            race=policy_params.get("_agent_race", "random"),
        )
        policy._load_if_exists(weights_file, re_initialize)
        return policy
