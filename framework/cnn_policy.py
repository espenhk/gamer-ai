"""Game-agnostic CNN policy evolved by isotropic evolutionary strategy.

Architecture::

    spatial (C, H, W)
        │
    Conv2d(C → conv1_out, kernel×kernel, relu)
    Conv2d(conv1_out → conv2_out, kernel×kernel, relu)
    AdaptiveAvgPool2d(pool_h × pool_w)
    Flatten → (conv2_out * pool_h * pool_w,)
        │
    Concat with normalised flat obs (obs_dim,)
        │
    FC(pool_flat + obs_dim → fc_dim, relu)
        │
    output head(s)

Weights are evolved by isotropic ES (1/5 success-rule sigma adaptation) —
no backprop.  Observations must be a dict with keys ``"flat"`` and
``"spatial"``.

Usage with ``policy_type: cnn`` in training_params.yaml.  A game using this
policy must supply dict observations of the form::

    {"flat": np.ndarray (obs_dim,), "spatial": np.ndarray (C, H, W)}

The generic :class:`CNNEvolutionPolicy` (``policy_type: cnn``) outputs
``np.tanh(head)`` — values in ``(-1, 1)^n_outputs`` — and is intended for
Box-action games.  SC2-specific action encoding lives in
``games/sc2/cnn_policy.SC2CNNEvolutionPolicy``.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from framework.obs_spec import ObsSpec
from framework.policies import BasePolicy, register_policy, trainer_state_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared conv math helpers (also imported by games/sc2/cnn_policy.py)
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
        strides=(x.strides[0], x.strides[1], x.strides[2], x.strides[1], x.strides[2]),
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
# CNNBackbone — reusable feature extractor
# ---------------------------------------------------------------------------


class CNNBackbone:
    """Two conv layers + adaptive pool + FC feature extractor.

    Holds the shared trunk weights (no output head).  The output is a
    ``(fc_dim,)`` feature vector that game-specific heads consume.

    Parameters
    ----------
    n_channels :
        Number of spatial input channels (C in the (C, H, W) observation).
    obs_spec :
        Flat observation spec — used for normalisation scales.
    conv1_out, conv2_out :
        Output channel counts for the first and second conv layers.
    pool_h, pool_w :
        Adaptive-pool output spatial dimensions.
    kernel :
        Convolution kernel size (square).
    fc_dim :
        Width of the fully-connected trunk layer.
    seed :
        RNG seed for weight initialisation.
    """

    def __init__(
        self,
        n_channels: int,
        obs_spec: ObsSpec,
        *,
        conv1_out: int = 32,
        conv2_out: int = 64,
        pool_h: int = 4,
        pool_w: int = 4,
        kernel: int = 3,
        fc_dim: int = 256,
        seed: int | None = None,
    ) -> None:
        self._n_channels = n_channels
        self._obs_dim = obs_spec.dim
        self._scales = obs_spec.scales
        self._conv1_out = conv1_out
        self._conv2_out = conv2_out
        self._pool_h = pool_h
        self._pool_w = pool_w
        self._kernel = kernel
        self._fc_dim = fc_dim
        self._pool_flat = conv2_out * pool_h * pool_w

        rng = np.random.default_rng(seed)

        def _he(shape: tuple) -> np.ndarray:
            fan_in = int(np.prod(shape[1:]))
            return rng.standard_normal(shape).astype(np.float32) * np.sqrt(2.0 / fan_in)

        C, k = n_channels, kernel
        self.W1 = _he((conv1_out, C, k, k))
        self.b1 = np.zeros(conv1_out, dtype=np.float32)
        self.W2 = _he((conv2_out, conv1_out, k, k))
        self.b2 = np.zeros(conv2_out, dtype=np.float32)
        fc_in = self._pool_flat + obs_spec.dim
        self.W3 = _he((fc_dim, fc_in))
        self.b3 = np.zeros(fc_dim, dtype=np.float32)

    @property
    def fc_dim(self) -> int:
        return self._fc_dim

    @property
    def param_dim(self) -> int:
        """Total number of scalar parameters in this backbone."""
        C, k = self._n_channels, self._kernel
        fc_in = self._pool_flat + self._obs_dim
        return (
            self._conv1_out * C * k * k
            + self._conv1_out
            + self._conv2_out * self._conv1_out * k * k
            + self._conv2_out
            + self._fc_dim * fc_in
            + self._fc_dim
        )

    def extract(self, spatial: np.ndarray, flat_obs: np.ndarray) -> np.ndarray:
        """Run the backbone and return the ``(fc_dim,)`` feature vector."""
        x = _conv2d_valid_relu(spatial.astype(np.float32), self.W1, self.b1)
        x = _conv2d_valid_relu(x, self.W2, self.b2)
        x = _adaptive_avg_pool(x, self._pool_h, self._pool_w)
        cnn_feat = x.ravel()
        norm_flat = flat_obs.astype(np.float32) / self._scales
        combined = np.concatenate([cnn_feat, norm_flat])
        h = np.maximum(0.0, self.W3 @ combined + self.b3)
        return h

    def to_flat(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2, self.W3.ravel(), self.b3]).astype(
            np.float32
        )

    def with_flat(self, flat: np.ndarray) -> "CNNBackbone":
        """Return a new backbone with weights loaded from ``flat``."""
        obj = object.__new__(CNNBackbone)
        obj._n_channels = self._n_channels
        obj._obs_dim = self._obs_dim
        obj._scales = self._scales
        obj._conv1_out = self._conv1_out
        obj._conv2_out = self._conv2_out
        obj._pool_h = self._pool_h
        obj._pool_w = self._pool_w
        obj._kernel = self._kernel
        obj._fc_dim = self._fc_dim
        obj._pool_flat = self._pool_flat

        C, k = self._n_channels, self._kernel
        fc_in = self._pool_flat + self._obs_dim
        off = 0

        def _take(shape: tuple) -> np.ndarray:
            nonlocal off
            n = int(np.prod(shape))
            out = flat[off : off + n].reshape(shape).copy()
            off += n
            return out

        obj.W1 = _take((self._conv1_out, C, k, k))
        obj.b1 = _take((self._conv1_out,))
        obj.W2 = _take((self._conv2_out, self._conv1_out, k, k))
        obj.b2 = _take((self._conv2_out,))
        obj.W3 = _take((self._fc_dim, fc_in))
        obj.b3 = _take((self._fc_dim,))
        return obj


# ---------------------------------------------------------------------------
# CNNModel — generic model for non-SC2 pixel games
# ---------------------------------------------------------------------------


class CNNModel:
    """Generic CNN model: :class:`CNNBackbone` + single output head.

    ``__call__`` returns ``np.tanh(head_output)``, squashing to
    ``(-1, 1)^n_outputs``, suitable for symmetric Box action spaces.

    Parameters
    ----------
    n_channels :
        Number of spatial input channels.
    obs_spec :
        Flat observation spec.
    n_outputs :
        Output head size (e.g. ``action_dim`` for a Box action space).
    seed :
        RNG seed.
    backbone_kwargs :
        Forwarded to :class:`CNNBackbone` (``conv1_out``, ``fc_dim``, …).
    """

    def __init__(
        self,
        n_channels: int,
        obs_spec: ObsSpec,
        n_outputs: int,
        seed: int | None = None,
        **backbone_kwargs,
    ) -> None:
        self._backbone = CNNBackbone(n_channels, obs_spec, seed=seed, **backbone_kwargs)
        self._n_outputs = n_outputs
        fc_dim = self._backbone.fc_dim

        rng = np.random.default_rng(seed)
        self.W_out = rng.standard_normal((n_outputs, fc_dim)).astype(np.float32) * np.sqrt(2.0 / fc_dim)
        self.b_out = np.zeros(n_outputs, dtype=np.float32)

    @property
    def flat_dim(self) -> int:
        return self._backbone.param_dim + self._n_outputs * self._backbone.fc_dim + self._n_outputs

    def forward(self, spatial: np.ndarray, flat_obs: np.ndarray) -> np.ndarray:
        """Return ``(n_outputs,)`` raw logits."""
        h = self._backbone.extract(spatial, flat_obs)
        return self.W_out @ h + self.b_out

    def to_flat(self) -> np.ndarray:
        return np.concatenate([self._backbone.to_flat(), self.W_out.ravel(), self.b_out]).astype(np.float32)

    def with_flat(self, flat: np.ndarray) -> "CNNModel":
        flat = np.asarray(flat, dtype=np.float32)
        if flat.shape[0] != self.flat_dim:
            raise ValueError(f"CNNModel.with_flat: expected {self.flat_dim} params, got {flat.shape[0]}")
        obj = object.__new__(CNNModel)
        obj._n_outputs = self._n_outputs

        n_bb = self._backbone.param_dim
        obj._backbone = self._backbone.with_flat(flat[:n_bb])

        off = n_bb
        fc_dim = self._backbone.fc_dim
        obj.W_out = flat[off : off + self._n_outputs * fc_dim].reshape(self._n_outputs, fc_dim).copy()
        off += self._n_outputs * fc_dim
        obj.b_out = flat[off : off + self._n_outputs].copy()
        return obj

    def __call__(self, obs: dict) -> np.ndarray:
        if not isinstance(obs, dict):
            raise TypeError(
                "CNNModel expects a dict observation with keys 'flat' and 'spatial'.  Got: " + type(obs).__name__
            )
        logits = self.forward(obs["spatial"], obs["flat"])
        return np.tanh(logits).astype(np.float32)

    def on_episode_start(self, **kwargs) -> None:
        pass

    def on_episode_end(self) -> None:
        pass

    def update(
        self,
        obs,
        action,
        reward: float,
        next_obs,
        done: bool,
        **kwargs,
    ) -> None:
        pass


# ---------------------------------------------------------------------------
# _CNNESBase — shared isotropic-ES outer loop (not registered; abstract)
# ---------------------------------------------------------------------------


class _CNNESBase(BasePolicy):
    """Abstract isotropic-ES outer optimiser for CNN-style models.

    Concrete subclasses must:

    * Set ``POLICY_TYPE`` (non-empty string).
    * Override ``compatible_with()``.
    * Call ``_init_es()`` from ``__init__()`` after building the inner model
      template.
    * Override ``_construct_or_resume()`` to construct the policy from
      ``policy_params``.

    The inner model template only needs to implement:

    * ``flat_dim`` (property)
    * ``to_flat() -> np.ndarray``
    * ``with_flat(flat) -> same type``
    * ``__call__(obs) -> np.ndarray``
    * ``on_episode_start(**kwargs)``
    * ``on_episode_end()``
    * ``update(obs, action, reward, next_obs, done, **kwargs)``
    """

    POLICY_TYPE = ""  # abstract — not registered
    LOOP_TYPE = "cmaes"

    # ------------------------------------------------------------------
    # Shared initialisation (call from subclass __init__)
    # ------------------------------------------------------------------

    def _init_es(
        self,
        template,
        population_size: int,
        initial_sigma: float,
        eval_episodes: int,
        rng: np.random.Generator,
    ) -> None:
        """Initialise ES state from a template model."""
        self._template = template
        self._flat_dim: int = template.flat_dim
        self._mean: np.ndarray = template.to_flat().astype(np.float64)
        self._lam: int = int(population_size)
        self._sigma: float = float(initial_sigma)
        self._eval_episodes: int = max(1, int(eval_episodes))
        self._rng = rng

        mu = self._lam // 2
        self._mu = mu
        raw_w = np.array(
            [np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)],
            dtype=np.float64,
        )
        self._recomb_w: np.ndarray = raw_w / raw_w.sum()

        self._pop: list[np.ndarray] = []
        self._champion = None
        self._champion_reward: float = float("-inf")

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

    def sample_population(self) -> list:
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

        order = np.argsort(rewards)[::-1]
        prev_best = self._champion_reward
        improved = False

        best_r = rewards[order[0]]
        if best_r > self._champion_reward:
            self._champion_reward = best_r
            self._champion = self._template.with_flat(np.array(self._pop[order[0]], dtype=np.float32))
            improved = True

        elite_xs = np.stack([self._pop[order[i]] for i in range(self._mu)])
        self._mean = np.einsum("i,ij->j", self._recomb_w, elite_xs)

        n_success = sum(1 for r in rewards if r > prev_best)
        success_rate = n_success / self._lam
        self._sigma = float(
            np.clip(
                self._sigma * (1.2 if success_rate > 0.2 else 0.85),
                1e-8,
                1e2,
            )
        )
        return improved

    # ------------------------------------------------------------------
    # Policy interface (delegates to champion for inference)
    # ------------------------------------------------------------------

    def __call__(self, obs) -> np.ndarray:
        if self._champion is None:
            raise RuntimeError(
                f"{type(self).__name__}: no champion yet — call sample_population() and update_distribution() first."
            )
        return self._champion(obs)

    def on_episode_start(self, **kwargs) -> None:
        if self._champion is not None:
            self._champion.on_episode_start(**kwargs)

    def on_episode_end(self) -> None:
        pass

    def update(self, obs, action, reward: float, next_obs, done: bool, **kwargs) -> None:
        if self._champion is not None:
            self._champion.update(obs, action, reward, next_obs, done, **kwargs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save champion weights as a ``.npz`` file."""
        if self._champion is not None:
            flat = self._champion.to_flat()
            np.savez(
                path.replace(".yaml", ".npz") if path.endswith(".yaml") else path,
                flat=flat,
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
                    f"{type(self).__name__}: trainer state flat_dim mismatch — "
                    f"saved={saved_flat_dim}, current={self._flat_dim}. "
                    f"Use --re-initialize to restart from scratch."
                )
            self._mean = data["mean"].astype(np.float64)
            self._sigma = float(data["sigma"])
        logger.info(
            "[%s] trainer state loaded from %s (sigma=%.4f)",
            type(self).__name__,
            path,
            self._sigma,
        )

    def load_champion(self, path: str) -> None:
        """Load champion weights from a ``.npz`` file saved by :meth:`save`."""
        with np.load(path) as data:
            saved_flat_dim = int(data["flat_dim"])
            if saved_flat_dim != self._flat_dim:
                raise ValueError(
                    f"{type(self).__name__}: champion flat_dim mismatch — "
                    f"saved={saved_flat_dim}, current={self._flat_dim}. "
                    f"Use --re-initialize to restart from scratch."
                )
            self._champion = self._template.with_flat(data["flat"].astype(np.float32))
            self._mean = data["flat"].astype(np.float64)
        logger.info("[%s] champion loaded from %s", type(self).__name__, path)

    def _load_if_exists(self, weights_file: str, re_initialize: bool) -> None:
        """Load champion + trainer state from disk if available."""
        champion_path = weights_file.replace(".yaml", ".npz")
        if os.path.exists(champion_path) and not re_initialize:
            try:
                self.load_champion(champion_path)
                ts = trainer_state_path(weights_file)
                if os.path.exists(ts):
                    self.load_trainer_state(ts)
                    logger.info("[%s] loaded trainer state from %s", type(self).__name__, ts)
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "[%s] could not load saved state — %s; starting from random.",
                    type(self).__name__,
                    exc,
                )


# ---------------------------------------------------------------------------
# CNNEvolutionPolicy — generic, registered as "cnn"
# ---------------------------------------------------------------------------


@register_policy
class CNNEvolutionPolicy(_CNNESBase):
    """Generic CNN policy evolved by isotropic ES.

    Expects dict observations ``{"flat": ..., "spatial": ...}``.  Output is
    ``np.tanh(head)`` in ``(-1, 1)^n_outputs`` — suitable for symmetric Box
    action spaces.

    Parameters
    ----------
    n_channels :
        Number of spatial input channels.
    obs_spec :
        Flat observation spec.
    n_outputs :
        Output dimensionality (must equal the game's action dimension).
    population_size :
        λ — offspring evaluated per generation (default 20).
    initial_sigma :
        Starting perturbation scale (default 0.05).
    eval_episodes :
        Episodes per individual per generation (averaged for fitness).
    seed :
        RNG seed.
    backbone_kwargs :
        Forwarded to :class:`CNNBackbone` (``conv1_out``, ``fc_dim``, …).
    """

    POLICY_TYPE = "cnn"
    LOOP_TYPE = "cmaes"
    VALID_POLICY_PARAMS = frozenset(
        {
            "population_size",
            "initial_sigma",
            "eval_episodes",
            "n_outputs",
            "conv1_out",
            "conv2_out",
            "pool_h",
            "pool_w",
            "kernel",
            "fc_dim",
        }
    )

    @classmethod
    def compatible_with(cls, game_name: str) -> tuple[bool, str | None]:
        # SC2 uses its own sc2_cnn; otherwise any game that exposes dict obs.
        if game_name == "sc2":
            return False, "Use policy_type: sc2_cnn for StarCraft II."
        return True, None

    def __init__(
        self,
        n_channels: int,
        obs_spec: ObsSpec,
        n_outputs: int,
        population_size: int = 20,
        initial_sigma: float = 0.05,
        eval_episodes: int = 1,
        seed: int | None = None,
        **backbone_kwargs,
    ) -> None:
        template = CNNModel(
            n_channels=n_channels,
            obs_spec=obs_spec,
            n_outputs=n_outputs,
            seed=seed,
            **backbone_kwargs,
        )
        self._obs_spec = obs_spec
        self._n_outputs = n_outputs
        self._backbone_kwargs = backbone_kwargs
        self._init_es(
            template=template,
            population_size=population_size,
            initial_sigma=initial_sigma,
            eval_episodes=eval_episodes,
            rng=np.random.default_rng(seed),
        )
        logger.info(
            "[CNNEvolutionPolicy] n_channels=%d  obs_dim=%d  n_outputs=%d  flat_dim=%d  pop=%d  sigma=%.4f",
            n_channels,
            obs_spec.dim,
            n_outputs,
            self._flat_dim,
            self._lam,
            self._sigma,
        )

    def to_cfg(self) -> dict:
        return {
            "policy_type": "cnn",
            "n_channels": self._template._backbone._n_channels,
            "obs_dim": self._obs_spec.dim,
            "n_outputs": self._n_outputs,
            "population_size": self._lam,
            "sigma": float(self._sigma),
            "eval_episodes": self._eval_episodes,
            "champion_reward": float(self._champion_reward),
        }

    def save(self, path: str) -> None:
        if self._champion is not None:
            flat = self._champion.to_flat()
            np.savez(
                path.replace(".yaml", ".npz") if path.endswith(".yaml") else path,
                flat=flat,
                flat_dim=np.int64(self._flat_dim),
                n_channels=np.int64(self._template._backbone._n_channels),
                obs_dim=np.int64(self._obs_spec.dim),
                n_outputs=np.int64(self._n_outputs),
            )

    @classmethod
    def _construct_or_resume(
        cls, *, obs_spec, head_names, discrete_actions, weights_file, policy_params, re_initialize
    ):
        n_channels = int(policy_params.get("_n_channels", 0))
        if n_channels == 0:
            raise ValueError(
                "cnn policy requires at least one spatial layer.  "
                "Set screen_layers (or equivalent) in training_params.yaml."
            )
        n_outputs = int(policy_params.get("n_outputs", len(head_names)))
        backbone_kwargs = {
            k: policy_params[k]
            for k in ("conv1_out", "conv2_out", "pool_h", "pool_w", "kernel", "fc_dim")
            if k in policy_params
        }
        policy = cls(
            n_channels=n_channels,
            obs_spec=obs_spec,
            n_outputs=n_outputs,
            population_size=policy_params.get("population_size", 20),
            initial_sigma=policy_params.get("initial_sigma", 0.05),
            eval_episodes=policy_params.get("eval_episodes", 1),
            **backbone_kwargs,
        )
        policy._load_if_exists(weights_file, re_initialize)
        return policy
