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
:class:`games.sc2.policies.LSTMEvolutionPolicy` (sample_population /
update_distribution interface, 1/5 success-rule sigma adaptation).
No backprop — purely evolutionary.

Usage with ``policy_type: sc2_cnn`` in training_params.yaml.  Requires
``screen_layers`` (non-empty) so that ``SC2Env`` returns dict observations.
"""

from __future__ import annotations

import logging

import numpy as np

from framework.obs_spec import ObsSpec
from framework.policies import BasePolicy
from games.sc2.actions import DISCRETE_ACTIONS, FUNCTION_IDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_FUNCS           = len(FUNCTION_IDS)   # 6
_N_SPATIAL_CELLS   = len(DISCRETE_ACTIONS)  # 9

# Pre-build (x, y) coordinates for each grid cell from DISCRETE_ACTIONS.
_GRID_XY: list[tuple[float, float]] = [
    (float(DISCRETE_ACTIONS[i, 1]), float(DISCRETE_ACTIONS[i, 2]))
    for i in range(_N_SPATIAL_CELLS)
]


# ---------------------------------------------------------------------------
# Pure-numpy conv2d + adaptive avg pool helpers
# ---------------------------------------------------------------------------

def _conv2d_valid_relu(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Valid 2-D convolution followed by ReLU — pure numpy.

    Parameters
    ----------
    x : (C_in, H, W) float32
    W : (C_out, C_in, k, k) float32
    b : (C_out,) float32

    Returns
    -------
    (C_out, H-k+1, W-k+1) float32
    """
    x = x.astype(np.float32)
    C_in, H, W_in = x.shape
    C_out, _, k, _ = W.shape
    H_out = H - k + 1
    W_out = W_in - k + 1
    # im2col via stride tricks — avoids Python loops over spatial positions.
    xc = np.lib.stride_tricks.as_strided(
        x,
        shape=(C_in, k, k, H_out, W_out),
        strides=(x.strides[0], x.strides[1], x.strides[2],
                 x.strides[1], x.strides[2]),
    ).reshape(C_in * k * k, H_out * W_out)
    out = (W.reshape(C_out, -1) @ xc + b[:, None]).reshape(C_out, H_out, W_out)
    np.maximum(out, 0.0, out=out)  # in-place ReLU
    return out


def _adaptive_avg_pool(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Adaptive average pooling from (C, H, W) to (C, out_h, out_w)."""
    C, H, W = x.shape
    result = np.empty((C, out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        h0 = int(i * H / out_h)
        h1 = int((i + 1) * H / out_h)
        for j in range(out_w):
            w0 = int(j * W / out_w)
            w1 = int((j + 1) * W / out_w)
            result[:, i, j] = x[:, h0:h1, w0:w1].mean(axis=(1, 2))
    return result


# ---------------------------------------------------------------------------
# SC2CNNModel — the individual network evaluated each episode
# ---------------------------------------------------------------------------

class SC2CNNModel:
    """CNN model that maps dict obs → SC2 action.

    Callable as a policy individual during ES evaluation.  Holds all network
    weights in plain numpy arrays so ``to_flat()`` / ``with_flat()`` give a
    flat parameter vector for evolutionary perturbation.

    Parameters
    ----------
    n_channels :
        Number of spatial input channels (len(screen_layers) +
        len(minimap_layers)).
    obs_spec :
        Flat observation spec — used for normalisation.
    seed :
        RNG seed for weight initialisation.
    """

    # Architecture hyperparams (fixed; matching the issue spec).
    _CONV1_OUT = 32
    _CONV2_OUT = 64
    _POOL_H    = 4
    _POOL_W    = 4
    _KERNEL    = 3
    _FC_DIM    = 256

    def __init__(
        self,
        n_channels: int,
        obs_spec: ObsSpec,
        seed: int | None = None,
    ) -> None:
        self._n_channels = n_channels
        self._obs_spec   = obs_spec
        self._obs_dim    = obs_spec.dim
        self._scales     = obs_spec.scales

        self._pool_flat = self._CONV2_OUT * self._POOL_H * self._POOL_W  # 1024
        fc_in           = self._pool_flat + self._obs_dim

        rng = np.random.default_rng(seed)

        def _he(shape: tuple) -> np.ndarray:
            fan_in = int(np.prod(shape[1:]))
            return rng.standard_normal(shape).astype(np.float32) * np.sqrt(2.0 / fan_in)

        C = n_channels
        k = self._KERNEL
        self.W1 = _he((self._CONV1_OUT, C, k, k))
        self.b1 = np.zeros(self._CONV1_OUT, dtype=np.float32)
        self.W2 = _he((self._CONV2_OUT, self._CONV1_OUT, k, k))
        self.b2 = np.zeros(self._CONV2_OUT, dtype=np.float32)
        self.W3 = _he((self._FC_DIM, fc_in))
        self.b3 = np.zeros(self._FC_DIM, dtype=np.float32)
        self.W_fn = _he((_N_FUNCS, self._FC_DIM))
        self.b_fn = np.zeros(_N_FUNCS, dtype=np.float32)
        self.W_sp = _he((_N_SPATIAL_CELLS, self._FC_DIM))
        self.b_sp = np.zeros(_N_SPATIAL_CELLS, dtype=np.float32)

    @property
    def flat_dim(self) -> int:
        C = self._n_channels
        k = self._KERNEL
        fc_in = self._pool_flat + self._obs_dim
        return (
            self._CONV1_OUT * C * k * k + self._CONV1_OUT
            + self._CONV2_OUT * self._CONV1_OUT * k * k + self._CONV2_OUT
            + self._FC_DIM * fc_in + self._FC_DIM
            + _N_FUNCS * self._FC_DIM + _N_FUNCS
            + _N_SPATIAL_CELLS * self._FC_DIM + _N_SPATIAL_CELLS
        )

    def to_flat(self) -> np.ndarray:
        return np.concatenate([
            self.W1.ravel(), self.b1,
            self.W2.ravel(), self.b2,
            self.W3.ravel(), self.b3,
            self.W_fn.ravel(), self.b_fn,
            self.W_sp.ravel(), self.b_sp,
        ]).astype(np.float32)

    def with_flat(self, flat: np.ndarray) -> "SC2CNNModel":
        flat = np.asarray(flat, dtype=np.float32)
        if flat.shape[0] != self.flat_dim:
            raise ValueError(
                f"SC2CNNModel.with_flat: expected {self.flat_dim} params, "
                f"got {flat.shape[0]}"
            )
        obj = object.__new__(SC2CNNModel)
        obj._n_channels = self._n_channels
        obj._obs_spec   = self._obs_spec
        obj._obs_dim    = self._obs_dim
        obj._scales     = self._scales
        obj._pool_flat  = self._pool_flat

        C, k = self._n_channels, self._KERNEL
        fc_in = self._pool_flat + self._obs_dim
        off = 0

        def _take(shape: tuple) -> np.ndarray:
            nonlocal off
            n = int(np.prod(shape))
            out = flat[off: off + n].reshape(shape).copy()
            off += n
            return out

        obj.W1   = _take((self._CONV1_OUT, C, k, k))
        obj.b1   = _take((self._CONV1_OUT,))
        obj.W2   = _take((self._CONV2_OUT, self._CONV1_OUT, k, k))
        obj.b2   = _take((self._CONV2_OUT,))
        obj.W3   = _take((self._FC_DIM, fc_in))
        obj.b3   = _take((self._FC_DIM,))
        obj.W_fn = _take((_N_FUNCS, self._FC_DIM))
        obj.b_fn = _take((_N_FUNCS,))
        obj.W_sp = _take((_N_SPATIAL_CELLS, self._FC_DIM))
        obj.b_sp = _take((_N_SPATIAL_CELLS,))
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
        x = _conv2d_valid_relu(spatial.astype(np.float32), self.W1, self.b1)
        x = _conv2d_valid_relu(x, self.W2, self.b2)
        x = _adaptive_avg_pool(x, self._POOL_H, self._POOL_W)
        cnn_feat = x.ravel()                               # (pool_flat,)

        norm_flat = (flat_obs.astype(np.float32) / self._scales)
        combined  = np.concatenate([cnn_feat, norm_flat])  # (pool_flat + obs_dim,)
        h         = np.maximum(0.0, self.W3 @ combined + self.b3)

        fn_scores = self.W_fn @ h + self.b_fn
        sp_scores = self.W_sp @ h + self.b_sp
        return fn_scores, sp_scores

    # ------------------------------------------------------------------
    # Policy interface (used when this model is an ES individual)
    # ------------------------------------------------------------------

    def __call__(self, obs: dict | np.ndarray) -> np.ndarray:
        if isinstance(obs, dict):
            flat_obs   = obs["flat"]
            spatial    = obs["spatial"]
        else:
            raise TypeError(
                "SC2CNNModel expects a dict observation with keys "
                "'flat' and 'spatial'.  Got: " + type(obs).__name__
            )
        fn_scores, sp_scores = self.forward(spatial, flat_obs)
        fn_idx   = int(np.argmax(fn_scores))
        cell_idx = int(np.argmax(sp_scores))
        x, y     = _GRID_XY[cell_idx]
        return np.array([fn_idx, x, y, 0.0], dtype=np.float32)

    def on_episode_start(self) -> None:
        pass

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
        pass  # trained via outer evolutionary optimiser


# ---------------------------------------------------------------------------
# SC2CNNEvolutionPolicy — isotropic ES outer optimiser wrapping SC2CNNModel
# ---------------------------------------------------------------------------

class SC2CNNEvolutionPolicy(BasePolicy):
    """Isotropic-ES outer optimiser for :class:`SC2CNNModel`.

    Uses the ``_greedy_loop_cmaes`` interface:
    ``sample_population()`` / ``update_distribution()``.
    Step size is adapted via the 1/5 success rule (same as
    :class:`games.sc2.policies.LSTMEvolutionPolicy`).

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
    """

    def __init__(
        self,
        n_channels: int,
        obs_spec: ObsSpec,
        population_size: int = 20,
        initial_sigma: float = 0.01,
        eval_episodes: int = 1,
        seed: int | None = None,
    ) -> None:
        self._lam           = int(population_size)
        self._sigma         = float(initial_sigma)
        self._eval_episodes = max(1, int(eval_episodes))
        self._obs_spec      = obs_spec
        self._rng           = np.random.default_rng(seed)

        self._template = SC2CNNModel(n_channels=n_channels, obs_spec=obs_spec, seed=seed)
        self._flat_dim = self._template.flat_dim
        self._mean     = self._template.to_flat().astype(np.float64)

        mu = self._lam // 2
        self._mu = mu
        raw_w = np.array(
            [np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)],
            dtype=np.float64,
        )
        self._recomb_w = raw_w / raw_w.sum()

        self._pop: list[np.ndarray] = []
        self._champion: SC2CNNModel | None        = None
        self._champion_reward: float              = float("-inf")

        logger.info(
            "[SC2CNNEvolutionPolicy] n_channels=%d  obs_dim=%d  "
            "flat_dim=%d  pop=%d  sigma=%.4f",
            n_channels, obs_spec.dim, self._flat_dim,
            self._lam, self._sigma,
        )

    # ------------------------------------------------------------------
    # Properties expected by _greedy_loop_cmaes
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
    # ES interface
    # ------------------------------------------------------------------

    def sample_population(self) -> list[SC2CNNModel]:
        self._pop = []
        for _ in range(self._lam):
            z = self._rng.standard_normal(self._flat_dim)
            self._pop.append(self._mean + self._sigma * z)
        return [self._template.with_flat(x.astype(np.float32)) for x in self._pop]

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

        # Weighted recombination of top-μ.
        elite_xs   = np.stack([self._pop[order[i]] for i in range(self._mu)])
        self._mean = np.einsum("i,ij->j", self._recomb_w, elite_xs)

        # 1/5 success-rule sigma adaptation.
        n_success    = sum(1 for r in rewards if r > prev_best)
        success_rate = n_success / self._lam
        self._sigma  = float(np.clip(
            self._sigma * (1.2 if success_rate > 0.2 else 0.85),
            1e-8, 1e2,
        ))

        return improved

    # ------------------------------------------------------------------
    # Policy interface (uses champion for inference)
    # ------------------------------------------------------------------

    def __call__(self, obs: dict | np.ndarray) -> np.ndarray:
        if self._champion is None:
            raise RuntimeError(
                "SC2CNNEvolutionPolicy: no champion yet — call "
                "sample_population() and update_distribution() first."
            )
        return self._champion(obs)

    def on_episode_start(self) -> None:
        pass

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
        pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        return {
            "policy_type":     "sc2_cnn",
            "n_channels":      self._template._n_channels,
            "obs_dim":         self._obs_spec.dim,
            "population_size": self._lam,
            "sigma":           float(self._sigma),
            "eval_episodes":   self._eval_episodes,
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

    def save_trainer_state(self, path: str) -> None:
        np.savez(
            path,
            mean=self._mean,
            sigma=np.float64(self._sigma),
            flat_dim=np.int64(self._flat_dim),
        )

    def load_trainer_state(self, path: str) -> None:
        with np.load(path) as data:
            saved_flat_dim = int(data["flat_dim"])
            if saved_flat_dim != self._flat_dim:
                raise ValueError(
                    f"SC2CNNEvolutionPolicy: trainer state flat_dim mismatch — "
                    f"saved={saved_flat_dim}, current={self._flat_dim}. "
                    f"Use --re-initialize to restart from scratch."
                )
            self._mean  = data["mean"].astype(np.float64)
            self._sigma = float(data["sigma"])
        logger.info(
            "[SC2CNNEvolutionPolicy] trainer state loaded from %s (sigma=%.4f)",
            path, self._sigma,
        )

    def load_champion(self, path: str) -> None:
        """Load champion weights from a .npz file saved by :meth:`save`."""
        with np.load(path) as data:
            saved_flat_dim = int(data["flat_dim"])
            if saved_flat_dim != self._flat_dim:
                raise ValueError(
                    f"SC2CNNEvolutionPolicy: champion flat_dim mismatch — "
                    f"saved={saved_flat_dim}, current={self._flat_dim}. "
                    f"Use --re-initialize to restart from scratch."
                )
            self._champion = self._template.with_flat(
                data["flat"].astype(np.float32)
            )
            self._mean = data["flat"].astype(np.float64)
        logger.info("[SC2CNNEvolutionPolicy] champion loaded from %s", path)
