"""Generic REINFORCE (Monte Carlo Policy Gradient) policies.

REINFORCEPolicy       — single softmax output head over a discrete action set.
TwoHeadREINFORCEPolicy — shared trunk + fn_idx softmax head + spatial sigmoid
                         head; mirrors the SC2 two-head design but parameterised
                         on obs_spec / n_fn_ids / n_spatial so it can be used by
                         any game integration that needs a combined discrete +
                         continuous action.
"""
from __future__ import annotations

import logging
from typing import Callable, NamedTuple

import numpy as np

from framework.obs_spec import ObsSpec
from framework.policies import BasePolicy

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Shared gradient-entry namedtuple for TwoHeadREINFORCEPolicy                 #
# --------------------------------------------------------------------------- #

class _GradEntry(NamedTuple):
    """Per-step trajectory entry stored by :class:`TwoHeadREINFORCEPolicy`.

    Fields match the tuple layout that the gradient-accumulation loop in
    ``on_episode_end`` unpacks, so the namedtuple can be used with
    ``_replace(fn_idx=...)`` in tests without any extra plumbing.
    """
    trunk_layer_inputs: list       # input to each trunk layer (for backprop)
    trunk_pre_relu:     list       # pre-activation values (for ReLU mask)
    h_last:             np.ndarray # shared trunk output, shape (h_dim,)
    fn_probs:           np.ndarray # masked softmax probabilities, shape (n_fn_ids,)
    fn_idx:             int        # sampled function index
    sp_sig:             np.ndarray # sigmoid spatial outputs, shape (n_spatial,)
    fn_mask:            np.ndarray # bool availability mask, shape (n_fn_ids,)

# --------------------------------------------------------------------------- #
# Sigmoid helper                                                               #
# --------------------------------------------------------------------------- #

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


# --------------------------------------------------------------------------- #
# REINFORCEPolicy                                                              #
# --------------------------------------------------------------------------- #

class REINFORCEPolicy(BasePolicy):
    """REINFORCE (Monte Carlo Policy Gradient) with optional entropy regularisation.

    Action head: softmax over *output_dim* discrete actions.
    Each episode accumulates (log_prob, reward) pairs; gradient update fires on
    episode end using discounted, normalised returns.

    Parameters
    ----------
    obs_spec :
        Observation spec providing ``dim`` and ``scales`` for normalisation.
    action_decoder :
        Callable ``(action_idx: int) -> np.ndarray`` that converts a sampled
        index to the action array returned by ``__call__``.
    output_dim :
        Number of discrete actions (= softmax output width).
    hidden_sizes :
        MLP hidden-layer widths (default ``[64, 64]``).
    learning_rate :
        Gradient-ascent step size.
    gamma :
        Discount factor for return computation.
    entropy_coeff :
        Entropy regularisation weight; 0 disables the term.
    baseline :
        ``"running_mean"`` (EMA of episode returns) or ``"none"``.
    available_actions_fn :
        Optional callable ``(info: dict) -> np.ndarray[bool]`` masking illegal
        actions.  When set, unavailable logits are set to ``-inf`` before
        sampling and their gradients are zeroed.
    seed :
        RNG seed for weight initialisation.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        action_decoder: Callable[[int], np.ndarray],
        *,
        output_dim: int,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
        baseline: str = "running_mean",
        available_actions_fn: Callable[..., np.ndarray] | None = None,
        seed: int | None = None,
    ) -> None:
        self._obs_spec       = obs_spec
        self._action_decoder = action_decoder
        self._output_dim     = int(output_dim)
        self._hidden         = list(hidden_sizes or [64, 64])
        self._lr             = float(learning_rate)
        self._gamma          = float(gamma)
        self._entropy_coeff  = float(entropy_coeff)
        self._baseline_type  = baseline
        self._avail_fn       = available_actions_fn

        # Attributes accessed by tests and downstream code
        self._obs_dim = obs_spec.dim
        self._scales  = obs_spec.scales

        self._weights, self._biases = self._build_net(seed)

        # Episode buffers (one entry per non-warmup step)
        self._ep_grads: list[tuple]   = []
        self._ep_rewards: list[float] = []

        # Running-mean baseline (EMA of total episode returns)
        self._baseline_val   = 0.0
        self._baseline_alpha = 0.05

        # Cached availability mask (None = all actions legal)
        self._available_mask: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_net(
        self, seed: int | None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        rng  = np.random.default_rng(seed)
        dims = [self._obs_dim] + self._hidden + [self._output_dim]
        weights, biases = [], []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            w = rng.standard_normal((dims[i + 1], fan_in)).astype(np.float32)
            w *= np.sqrt(2.0 / fan_in)
            b = np.zeros(dims[i + 1], dtype=np.float32)
            weights.append(w)
            biases.append(b)
        return weights, biases

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_s = z - z.max()
        e   = np.exp(z_s)
        return e / e.sum()

    def _forward(
        self, obs_norm: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Forward pass; caches (layer_inputs, pre_relu) for backprop."""
        x: np.ndarray      = obs_norm.astype(np.float32)
        layer_inputs: list = []
        pre_relu: list     = []
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            layer_inputs.append(x.copy())
            z = w @ x + b
            if i < len(self._weights) - 1:
                pre_relu.append(z.copy())
                x = np.maximum(0.0, z)
            else:
                logits = z
        probs = self._softmax(logits)
        return probs, layer_inputs, pre_relu

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def on_episode_start(self, **kwargs) -> None:
        self._ep_grads.clear()
        self._ep_rewards.clear()
        if self._avail_fn is not None:
            info = kwargs.get("info") or {}
            result = self._avail_fn(info)
            if result is not None:
                self._available_mask = np.asarray(result, dtype=bool)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_norm           = obs / self._scales
        probs, l_in, pre_r = self._forward(obs_norm)
        if self._available_mask is not None:
            masked = probs * self._available_mask
            total  = float(masked.sum())
            if total > 1e-10:
                masked = masked / total
            else:
                masked = self._available_mask.astype(np.float32)
                masked = masked / float(masked.sum())
            action_idx   = int(np.random.choice(self._output_dim, p=masked))
            stored_probs = masked
        else:
            action_idx   = int(np.random.choice(self._output_dim, p=probs))
            stored_probs = probs.copy()
        self._ep_grads.append((l_in, pre_r, stored_probs, action_idx))
        return self._action_decoder(action_idx).copy()

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray | int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        **kwargs,
    ) -> None:
        self._ep_rewards.append(float(reward))
        if self._avail_fn is not None:
            info   = kwargs.get("info") or {}
            result = self._avail_fn(info)
            if result is not None:
                self._available_mask = np.asarray(result, dtype=bool)

    def on_episode_end(self) -> None:
        T = min(len(self._ep_grads), len(self._ep_rewards))
        if T == 0:
            self._ep_grads.clear()
            self._ep_rewards.clear()
            return

        # Discounted returns
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

        dW = [np.zeros_like(w, dtype=np.float64) for w in self._weights]
        dB = [np.zeros_like(b, dtype=np.float64) for b in self._biases]

        for t in range(T):
            l_in, pre_r, probs, a_idx = self._ep_grads[t]
            advantage = float(G_norm[t])

            delta          = -probs.copy().astype(np.float64)
            delta[a_idx]  += 1.0
            delta          *= advantage

            if self._entropy_coeff > 0.0:
                log_p        = np.log(probs.astype(np.float64) + 1e-8)
                H            = -float(np.dot(probs, log_p))
                entropy_grad = -probs.astype(np.float64) * (log_p + H)
                delta       += self._entropy_coeff * entropy_grad

            g = delta
            for i in range(len(self._weights) - 1, -1, -1):
                dW[i] += np.outer(g, l_in[i])
                dB[i] += g
                if i > 0:
                    g = self._weights[i].T @ g * (pre_r[i - 1] > 0)

        lr_t = self._lr / T
        for i in range(len(self._weights)):
            self._weights[i] += (lr_t * dW[i]).astype(np.float32)
            self._biases[i]  += (lr_t * dB[i]).astype(np.float32)

        self._ep_grads.clear()
        self._ep_rewards.clear()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        return {
            "policy_type":    "reinforce",
            "hidden_sizes":   self._hidden,
            "learning_rate":  float(self._lr),
            "gamma":          float(self._gamma),
            "entropy_coeff":  float(self._entropy_coeff),
            "baseline":       self._baseline_type,
            "output_dim":     self._output_dim,
            "obs_dim":        self._obs_dim,
            "baseline_value": float(self._baseline_val),
            "weights":        [w.tolist() for w in self._weights],
            "biases":         [b.tolist() for b in self._biases],
        }

    @classmethod
    def from_cfg(
        cls,
        cfg: dict,
        obs_spec: ObsSpec,
        action_decoder: Callable[[int], np.ndarray],
    ) -> "REINFORCEPolicy":
        obj = cls(
            obs_spec       = obs_spec,
            action_decoder = action_decoder,
            output_dim     = cfg.get("output_dim",
                                     np.array(cfg["biases"][-1]).shape[0]
                                     if "biases" in cfg else 1),
            hidden_sizes   = cfg.get("hidden_sizes",  [64, 64]),
            learning_rate  = cfg.get("learning_rate", 0.001),
            gamma          = cfg.get("gamma",         0.99),
            entropy_coeff  = cfg.get("entropy_coeff", 0.01),
            baseline       = cfg.get("baseline",      "running_mean"),
        )
        if "weights" in cfg:
            obj._weights = [np.array(w, dtype=np.float32) for w in cfg["weights"]]
            obj._biases  = [np.array(b, dtype=np.float32) for b in cfg["biases"]]
        if "baseline_value" in cfg:
            obj._baseline_val = float(cfg["baseline_value"])
        return obj

    def save_trainer_state(self, path: str) -> None:
        """Persist running-mean baseline and obs_dim to an .npz file."""
        np.savez(
            path,
            baseline_val = np.float64(self._baseline_val),
            obs_dim      = np.int64(self._obs_dim),
        )
        logger.debug("[REINFORCEPolicy] trainer state saved → %s", path)

    def load_trainer_state(self, path: str) -> None:
        """Restore baseline from an .npz file.

        Raises ValueError if the saved obs_dim does not match.
        """
        with np.load(path) as data:
            saved_obs_dim = int(data["obs_dim"])
            if saved_obs_dim != self._obs_dim:
                raise ValueError(
                    f"REINFORCEPolicy: trainer state obs_dim mismatch — "
                    f"saved={saved_obs_dim}, current={self._obs_dim}. "
                    f"The observation space may have changed. "
                    f"Use --re-initialize to restart from scratch."
                )
            self._baseline_val = float(data["baseline_val"])
        logger.info("[REINFORCEPolicy] trainer state loaded from %s (baseline=%.4f)",
                    path, self._baseline_val)


# --------------------------------------------------------------------------- #
# TwoHeadREINFORCEPolicy                                                      #
# --------------------------------------------------------------------------- #

class TwoHeadREINFORCEPolicy(BasePolicy):
    """REINFORCE with a shared trunk and two independent output heads.

    * **fn_head** — *n_fn_ids* logits, softmax → sampled ``fn_idx``;
      unavailable IDs are masked to ``-∞`` before sampling.
    * **spatial_head** — *n_spatial* logits, sigmoid → continuous coordinates
      in ``[0, 1]^n_spatial``.

    Both heads are trained jointly via REINFORCE with a deterministic gradient
    for the spatial head (sigmoid policy gradient).

    Parameters
    ----------
    obs_spec :
        Observation spec.
    n_fn_ids :
        Width of the fn_idx softmax head.
    n_spatial :
        Width of the spatial sigmoid head (e.g. 2 for (x, y)).
    action_fn :
        Callable ``(fn_idx: int, sp_sig: np.ndarray) -> np.ndarray`` that
        assembles the policy's action array from sampled outputs.
    hidden_sizes :
        Shared trunk hidden-layer widths (default ``[128, 64]``).
    learning_rate :
        Gradient step size (default ``0.0003``).
    gamma :
        Discount factor (default ``0.995``).
    entropy_coeff :
        Entropy regularisation weight for the fn head (default ``0.05``).
    baseline :
        ``"running_mean"`` (EMA) or ``"none"``.
    available_fn_ids_fn :
        Optional callable ``(info: dict) -> set[int] | None`` that returns
        available fn IDs per step.  ``None`` means all IDs are available.
    seed :
        Optional RNG seed.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        n_fn_ids: int,
        n_spatial: int,
        action_fn: Callable[[int, np.ndarray], np.ndarray],
        *,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 0.0003,
        gamma: float = 0.995,
        entropy_coeff: float = 0.05,
        baseline: str = "running_mean",
        available_fn_ids_fn: Callable[..., "set[int] | None"] | None = None,
        seed: int | None = None,
    ) -> None:
        self._obs_spec         = obs_spec
        self._obs_dim          = obs_spec.dim
        self._scales           = obs_spec.scales
        self._n_fn_ids         = int(n_fn_ids)
        self._n_spatial        = int(n_spatial)
        self._action_fn        = action_fn
        self._hidden           = list(hidden_sizes) if hidden_sizes is not None else [128, 64]
        self._lr               = float(learning_rate)
        self._gamma            = float(gamma)
        self._entropy_coeff    = float(entropy_coeff)
        self._baseline_type    = baseline
        self._avail_fn_ids_fn  = available_fn_ids_fn

        self._rng = np.random.default_rng(seed)

        (
            self._trunk_w,
            self._trunk_b,
            self._fn_w,
            self._fn_b,
            self._sp_w,
            self._sp_b,
        ) = self._build_net(seed)

        self._ep_grads: list[_GradEntry] = []
        self._ep_rewards: list[float] = []

        self._baseline_val   = 0.0
        self._baseline_alpha = 0.05

        self._available_fn_ids: "set[int] | None" = None

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_net(
        self, seed: int | None
    ) -> tuple[
        list[np.ndarray], list[np.ndarray],
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
    ]:
        rng = np.random.default_rng(seed)

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

        fn_w = rng.standard_normal((self._n_fn_ids, h_dim)).astype(np.float32)
        fn_w *= np.sqrt(2.0 / h_dim)
        fn_b = np.zeros(self._n_fn_ids, dtype=np.float32)

        sp_w = rng.standard_normal((self._n_spatial, h_dim)).astype(np.float32)
        sp_w *= np.sqrt(2.0 / h_dim)
        sp_b = np.zeros(self._n_spatial, dtype=np.float32)

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
        x: np.ndarray      = obs_norm.astype(np.float32)
        layer_inputs: list = []
        pre_relu: list     = []
        for w, b in zip(self._trunk_w, self._trunk_b):
            layer_inputs.append(x.copy())
            z = w @ x + b
            pre_relu.append(z.copy())
            x = np.maximum(0.0, z)
        return x, layer_inputs, pre_relu

    def _build_fn_mask(self, available_fn_ids: "set[int] | None") -> np.ndarray:
        mask = np.ones(self._n_fn_ids, dtype=bool)
        if available_fn_ids is not None:
            for i in range(self._n_fn_ids):
                mask[i] = i in available_fn_ids
        if not mask.any():
            mask[0] = True
        return mask

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def on_episode_start(self, **kwargs) -> None:
        self._ep_grads.clear()
        self._ep_rewards.clear()
        info = kwargs.get("info") or {}
        if self._avail_fn_ids_fn is not None:
            available = self._avail_fn_ids_fn(info)
        else:
            available = info.get("available_fn_ids")
        if available is not None:
            self._available_fn_ids = set(available)
        else:
            self._available_fn_ids = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_norm = obs / self._scales
        h_last, l_in, pre_r = self._trunk_forward(obs_norm)

        fn_logits = self._fn_w @ h_last + self._fn_b
        fn_mask   = self._build_fn_mask(self._available_fn_ids)
        fn_logits_masked = fn_logits.copy()
        fn_logits_masked[~fn_mask] = -np.inf
        fn_probs  = self._softmax(fn_logits_masked)
        fn_idx    = int(self._rng.choice(self._n_fn_ids, p=fn_probs))

        sp_logits = self._sp_w @ h_last + self._sp_b
        sp_sig    = np.array(
            [_sigmoid(float(sp_logits[i])) for i in range(self._n_spatial)],
            dtype=np.float32,
        )

        self._ep_grads.append(_GradEntry(
            trunk_layer_inputs = l_in,
            trunk_pre_relu     = pre_r,
            h_last             = h_last.copy(),
            fn_probs           = fn_probs.copy(),
            fn_idx             = fn_idx,
            sp_sig             = sp_sig.copy(),
            fn_mask            = fn_mask.copy(),
        ))

        return self._action_fn(fn_idx, sp_sig)

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray | int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        **kwargs,
    ) -> None:
        self._ep_rewards.append(float(reward))
        info = kwargs.get("info") or {}
        if self._avail_fn_ids_fn is not None:
            available = self._avail_fn_ids_fn(info)
        else:
            available = info.get("available_fn_ids")
        if available is not None:
            self._available_fn_ids = set(available)

    def on_episode_end(self) -> None:
        T = min(len(self._ep_grads), len(self._ep_rewards))
        if T == 0:
            self._ep_grads.clear()
            self._ep_rewards.clear()
            return

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

        dW_trunk = [np.zeros_like(w, dtype=np.float64) for w in self._trunk_w]
        dB_trunk = [np.zeros_like(b, dtype=np.float64) for b in self._trunk_b]
        dW_fn    = np.zeros_like(self._fn_w, dtype=np.float64)
        dB_fn    = np.zeros_like(self._fn_b, dtype=np.float64)
        dW_sp    = np.zeros_like(self._sp_w, dtype=np.float64)
        dB_sp    = np.zeros_like(self._sp_b, dtype=np.float64)

        for t in range(T):
            l_in, pre_r, h_last, fn_probs, fn_idx, sp_sig, fn_mask = self._ep_grads[t]
            advantage = float(G_norm[t])

            delta_fn = -fn_probs.copy().astype(np.float64)
            delta_fn[fn_idx] += 1.0
            delta_fn[~fn_mask] = 0.0
            delta_fn *= advantage

            if self._entropy_coeff > 0.0:
                log_p_fn      = np.log(fn_probs.astype(np.float64) + 1e-8)
                H_fn          = -float(np.dot(fn_probs[fn_mask], log_p_fn[fn_mask]))
                ent_grad_fn   = np.zeros(self._n_fn_ids, dtype=np.float64)
                ent_grad_fn[fn_mask] = -(
                    fn_probs[fn_mask].astype(np.float64) * (log_p_fn[fn_mask] + H_fn)
                )
                delta_fn += self._entropy_coeff * ent_grad_fn

            sp_sig_d = sp_sig.astype(np.float64)
            delta_sp = advantage * (sp_sig_d * (1.0 - sp_sig_d))

            h_last_d = h_last.astype(np.float64)
            dW_fn += np.outer(delta_fn, h_last_d)
            dB_fn += delta_fn
            dW_sp += np.outer(delta_sp, h_last_d)
            dB_sp += delta_sp

            if self._trunk_w:
                g_trunk = (
                    self._fn_w.T.astype(np.float64) @ delta_fn
                    + self._sp_w.T.astype(np.float64) @ delta_sp
                )
                n_trunk = len(self._trunk_w)
                for i in range(n_trunk - 1, -1, -1):
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
            "policy_type":    "two_head_reinforce",
            "hidden_sizes":   self._hidden,
            "learning_rate":  float(self._lr),
            "gamma":          float(self._gamma),
            "entropy_coeff":  float(self._entropy_coeff),
            "baseline":       self._baseline_type,
            "n_fn_ids":       self._n_fn_ids,
            "n_spatial":      self._n_spatial,
            "obs_dim":        self._obs_dim,
            "baseline_value": float(self._baseline_val),
            "trunk_weights":  [w.tolist() for w in self._trunk_w],
            "trunk_biases":   [b.tolist() for b in self._trunk_b],
            "fn_weights":     self._fn_w.tolist(),
            "fn_biases":      self._fn_b.tolist(),
            "sp_weights":     self._sp_w.tolist(),
            "sp_biases":      self._sp_b.tolist(),
        }

    @classmethod
    def from_cfg(
        cls,
        cfg: dict,
        obs_spec: ObsSpec,
        action_fn: Callable[[int, np.ndarray], np.ndarray],
        available_fn_ids_fn: "Callable[..., set[int] | None] | None" = None,
    ) -> "TwoHeadREINFORCEPolicy":
        obj = cls(
            obs_spec          = obs_spec,
            n_fn_ids          = cfg.get("n_fn_ids",       6),
            n_spatial         = cfg.get("n_spatial",      2),
            action_fn         = action_fn,
            hidden_sizes      = cfg.get("hidden_sizes",   [128, 64]),
            learning_rate     = cfg.get("learning_rate",  0.0003),
            gamma             = cfg.get("gamma",          0.995),
            entropy_coeff     = cfg.get("entropy_coeff",  0.05),
            baseline          = cfg.get("baseline",       "running_mean"),
            available_fn_ids_fn = available_fn_ids_fn,
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
        np.savez(path,
                 baseline_val=np.float64(self._baseline_val),
                 obs_dim=np.int64(self._obs_dim))

    def load_trainer_state(self, path: str) -> None:
        with np.load(path) as data:
            saved_obs_dim = int(data["obs_dim"])
            if saved_obs_dim != self._obs_dim:
                raise ValueError(
                    f"TwoHeadREINFORCEPolicy: trainer state obs_dim mismatch — "
                    f"saved={saved_obs_dim}, current={self._obs_dim}. "
                    f"Use --re-initialize to restart from scratch."
                )
            self._baseline_val = float(data["baseline_val"])
        logger.info("[TwoHeadREINFORCEPolicy] trainer state loaded from %s", path)
