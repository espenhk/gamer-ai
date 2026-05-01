"""Curiosity-driven exploration modules.

Two small pure-numpy models that produce an **intrinsic reward** per step:

* :class:`ICM`  — Intrinsic Curiosity Module (Pathak et al. 2017).
    Jointly trains a feature encoder phi, a forward dynamics model predicting
    phi(s') from (phi(s), a), and an inverse dynamics model predicting a from
    (phi(s), phi(s')). The intrinsic reward is the forward-model prediction
    error, and the inverse model forces the features to encode action-relevant
    information only (preventing a degenerate collapse).

* :class:`RND`  — Random Network Distillation (Burda et al. 2018).
    A frozen randomly-initialised target network and a trained predictor
    network produce feature embeddings of the current observation. The
    intrinsic reward is the prediction error. Novel states have higher error;
    the predictor rapidly learns visited states so their error decays.

Both classes implement :class:`CuriosityModule` and are game-agnostic (they
only see numpy obs / action vectors). The factory :func:`make_curiosity`
returns the right subclass given a string type, or ``None`` when disabled.

Updates are online: each call to :meth:`CuriosityModule.update` performs a
single SGD step on the given transition. No replay buffer is required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _he_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    std = np.sqrt(2.0 / max(fan_in, 1))
    return rng.standard_normal((fan_in, fan_out)).astype(np.float32) * std


def _as_row(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    return a


class _MLP:
    """Minimal MLP with manual forward/backward suitable for one-sample SGD.

    Layers: linear -> ReLU -> ... -> linear (no activation on the last layer).
    """

    def __init__(self, dims: list[int], lr: float = 1e-3, seed: int = 0) -> None:
        if len(dims) < 2:
            raise ValueError("MLP needs at least an input and output layer.")
        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = [
            _he_init(rng, dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ]
        self.biases: list[np.ndarray] = [
            np.zeros(dims[i + 1], dtype=np.float32) for i in range(len(dims) - 1)
        ]
        self.lr = float(lr)

    def forward(self, x: np.ndarray, cache: dict | None = None) -> np.ndarray:
        a = _as_row(x)
        if cache is not None:
            cache["acts"] = [a]
            cache["preacts"] = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            a = np.maximum(0.0, z) if i < len(self.weights) - 1 else z
            if cache is not None:
                cache["preacts"].append(z)
                cache["acts"].append(a)
        return a

    def backward(
        self, dout: np.ndarray, cache: dict
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Backprop *dout* (gradient w.r.t. output) through the cached pass.

        Returns ``(dx, grads_W, grads_b)`` where ``dx`` is the gradient w.r.t.
        the original input. Gradients are not applied — call :meth:`apply`.
        """
        delta = _as_row(dout)
        grads_W: list[np.ndarray | None] = [None] * len(self.weights)
        grads_b: list[np.ndarray | None] = [None] * len(self.biases)
        dx = delta
        for i in range(len(self.weights) - 1, -1, -1):
            a_prev = cache["acts"][i]
            grads_W[i] = a_prev.T @ delta
            grads_b[i] = delta.sum(axis=0)
            dx = delta @ self.weights[i].T
            if i > 0:
                z_prev = cache["preacts"][i - 1]
                delta = dx * (z_prev > 0).astype(np.float32)
        return dx, grads_W, grads_b  # type: ignore[return-value]

    def apply(self, grads_W: list[np.ndarray], grads_b: list[np.ndarray]) -> None:
        if self.lr == 0.0:
            return
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.lr * grads_W[i]
            self.biases[i]  = self.biases[i]  - self.lr * grads_b[i]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CuriosityModule(ABC):
    """Abstract interface for online-trained intrinsic reward modules."""

    @abstractmethod
    def reward(
        self,
        prev_obs: np.ndarray,
        action: np.ndarray,
        curr_obs: np.ndarray,
    ) -> float:
        """Return the scalar intrinsic reward for transition (s, a, s')."""

    @abstractmethod
    def update(
        self,
        prev_obs: np.ndarray,
        action: np.ndarray,
        curr_obs: np.ndarray,
    ) -> None:
        """Online gradient step on the transition."""

    def reset_episode(self) -> None:
        """Called at the start of each episode.  Override if stateful."""


class ICM(CuriosityModule):
    """Intrinsic Curiosity Module (Pathak et al. 2017).

    ``r_i = eta * 0.5 * ||phi(s') - phi_pred(s')||^2``

    Shared-encoder joint loss:
        L = beta * L_forward + (1 - beta) * L_inverse
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        feature_dim: int = 8,
        hidden_size: int = 32,
        lr: float = 1e-3,
        beta: float = 0.2,
        eta: float = 1.0,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must lie in [0, 1], got {beta}")
        if obs_dim <= 0 or action_dim <= 0 or feature_dim <= 0:
            raise ValueError("obs_dim, action_dim and feature_dim must be positive")
        self.obs_dim     = int(obs_dim)
        self.action_dim  = int(action_dim)
        self.feature_dim = int(feature_dim)
        self.beta        = float(beta)
        self.eta         = float(eta)
        self.encoder      = _MLP([obs_dim, hidden_size, feature_dim],
                                 lr=lr, seed=seed)
        self.forward_net  = _MLP([feature_dim + action_dim, hidden_size, feature_dim],
                                 lr=lr, seed=seed + 1)
        self.inverse_net  = _MLP([feature_dim * 2, hidden_size, action_dim],
                                 lr=lr, seed=seed + 2)

    # ---------------- inference -----------------------------------------

    def _encode(self, obs: np.ndarray, cache: dict | None = None) -> np.ndarray:
        return self.encoder.forward(obs, cache=cache)

    def reward(
        self,
        prev_obs: np.ndarray,
        action: np.ndarray,
        curr_obs: np.ndarray,
    ) -> float:
        f_prev = self._encode(prev_obs)
        f_curr = self._encode(curr_obs)
        a      = _as_row(action)
        if a.shape[1] != self.action_dim:
            raise ValueError(
                f"action dim mismatch: got {a.shape[1]}, expected {self.action_dim}")
        f_pred = self.forward_net.forward(np.concatenate([f_prev, a], axis=1))
        diff = f_curr - f_pred
        return float(self.eta * 0.5 * np.sum(diff * diff))

    # ---------------- learning ------------------------------------------

    def update(
        self,
        prev_obs: np.ndarray,
        action: np.ndarray,
        curr_obs: np.ndarray,
    ) -> None:
        a = _as_row(action)
        if a.shape[1] != self.action_dim:
            raise ValueError(
                f"action dim mismatch: got {a.shape[1]}, expected {self.action_dim}")

        # --- forward passes with caches so we can backprop -------------------
        enc_prev_cache: dict = {}
        enc_curr_cache: dict = {}
        f_prev = self._encode(prev_obs, cache=enc_prev_cache)
        f_curr = self._encode(curr_obs, cache=enc_curr_cache)

        fwd_cache: dict = {}
        f_pred = self.forward_net.forward(
            np.concatenate([f_prev, a], axis=1), cache=fwd_cache,
        )

        inv_cache: dict = {}
        a_pred = self.inverse_net.forward(
            np.concatenate([f_prev, f_curr], axis=1), cache=inv_cache,
        )

        # --- loss gradients --------------------------------------------------
        # L_fwd = beta * 0.5 * ||f_pred - f_curr||^2
        # L_inv = (1-beta) * 0.5 * ||a_pred - a||^2
        d_f_pred = self.beta * (f_pred - f_curr)
        d_f_curr_from_fwd = -d_f_pred
        d_a_pred = (1.0 - self.beta) * (a_pred - a)

        # --- forward net backward -------------------------------------------
        d_fwd_in, gW_fwd, gb_fwd = self.forward_net.backward(d_f_pred, fwd_cache)
        d_f_prev_from_fwd = d_fwd_in[:, : self.feature_dim]
        # (action gradient tail discarded — no learnable action embedding)

        # --- inverse net backward -------------------------------------------
        d_inv_in, gW_inv, gb_inv = self.inverse_net.backward(d_a_pred, inv_cache)
        d_f_prev_from_inv = d_inv_in[:, : self.feature_dim]
        d_f_curr_from_inv = d_inv_in[:, self.feature_dim :]

        # --- encoder backward (two caches with shared weights, sum grads) ----
        _, gW_prev, gb_prev = self.encoder.backward(
            d_f_prev_from_fwd + d_f_prev_from_inv, enc_prev_cache,
        )
        _, gW_curr, gb_curr = self.encoder.backward(
            d_f_curr_from_fwd + d_f_curr_from_inv, enc_curr_cache,
        )
        gW_enc = [p + c for p, c in zip(gW_prev, gW_curr)]
        gb_enc = [p + c for p, c in zip(gb_prev, gb_curr)]

        self.encoder.apply(gW_enc, gb_enc)
        self.forward_net.apply(gW_fwd, gb_fwd)
        self.inverse_net.apply(gW_inv, gb_inv)


class RND(CuriosityModule):
    """Random Network Distillation (Burda et al. 2018).

    ``r_i = eta * 0.5 * ||target(s') - predictor(s')||^2``

    The target network is frozen at initialisation; the predictor is trained
    to mimic it on visited states, so novel states retain high error.
    """

    def __init__(
        self,
        obs_dim: int,
        feature_dim: int = 8,
        hidden_size: int = 32,
        lr: float = 1e-3,
        eta: float = 1.0,
        seed: int = 0,
    ) -> None:
        if obs_dim <= 0 or feature_dim <= 0:
            raise ValueError("obs_dim and feature_dim must be positive")
        self.obs_dim     = int(obs_dim)
        self.feature_dim = int(feature_dim)
        self.eta         = float(eta)
        # lr=0 on the target net freezes it (apply() short-circuits).
        self.target    = _MLP([obs_dim, hidden_size, feature_dim], lr=0.0, seed=seed)
        self.predictor = _MLP([obs_dim, hidden_size, feature_dim], lr=lr,  seed=seed + 1)

    def reward(
        self,
        prev_obs: np.ndarray,
        action: np.ndarray,
        curr_obs: np.ndarray,
    ) -> float:
        del prev_obs, action  # RND only depends on s'
        f_t = self.target.forward(curr_obs)
        f_p = self.predictor.forward(curr_obs)
        diff = f_t - f_p
        return float(self.eta * 0.5 * np.sum(diff * diff))

    def update(
        self,
        prev_obs: np.ndarray,
        action: np.ndarray,
        curr_obs: np.ndarray,
    ) -> None:
        del prev_obs, action
        cache: dict = {}
        f_p = self.predictor.forward(curr_obs, cache=cache)
        f_t = self.target.forward(curr_obs)
        # L = 0.5 * ||f_p - f_t||^2  ->  d/df_p = (f_p - f_t)
        _, gW, gb = self.predictor.backward(f_p - f_t, cache)
        self.predictor.apply(gW, gb)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_curiosity(
    kind: str,
    *,
    obs_dim: int,
    action_dim: int,
    feature_dim: int = 8,
    hidden_size: int = 32,
    lr: float = 1e-3,
    beta: float = 0.2,
    eta: float = 1.0,
    seed: int = 0,
) -> CuriosityModule | None:
    """Build a curiosity module from a string type.

    Returns ``None`` for ``"none"`` (curiosity disabled).
    Raises ``ValueError`` for unknown types.
    """
    k = (kind or "none").strip().lower()
    if k == "none":
        return None
    if k == "icm":
        return ICM(obs_dim=obs_dim, action_dim=action_dim, feature_dim=feature_dim,
                   hidden_size=hidden_size, lr=lr, beta=beta, eta=eta, seed=seed)
    if k == "rnd":
        return RND(obs_dim=obs_dim, feature_dim=feature_dim, hidden_size=hidden_size,
                   lr=lr, eta=eta, seed=seed)
    raise ValueError(
        f"unknown curiosity_type: {kind!r} (expected 'none', 'icm', or 'rnd')"
    )
