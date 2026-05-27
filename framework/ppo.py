"""Generic PPO (Proximal Policy Optimization) policy — pure numpy.

PPOPolicy — on-policy actor-critic with a clipped surrogate objective and
            Generalised Advantage Estimation (GAE).  A softmax actor over a
            discrete action set and a scalar-value critic, both pure-numpy MLPs
            trained with Adam.  Parameterised on an ObsSpec + an action decoder
            so any game integration with a discrete action set can use it.

The policy collects one episode of transitions (buffered through ``__call__`` /
``update``) and runs several epochs of minibatch PPO updates on ``on_episode_end``.
It therefore reuses the framework's ``q_learning`` greedy loop unchanged
(``LOOP_TYPE = "q_learning"``): the loop runs full episodes, feeds every step to
``update`` and calls ``on_episode_end`` once per episode.

PPO's continuous-action / multi-head variants are out of scope here; this is the
discrete-softmax baseline.  The discrete steer/accel/brake encoding is wrong for
SC2's ``[fn_idx, x, y, queue]`` action space, so the policy declares itself
incompatible with SC2 (use ``sc2_reinforce`` there).
"""

from __future__ import annotations

import logging
import os
from typing import Callable

import numpy as np
import yaml

from framework.obs_spec import ObsSpec
from framework.policies import (
    BasePolicy,
    check_continuous_action_compatible,
    register_policy,
    trainer_state_path,
)

logger = logging.getLogger(__name__)


@register_policy
class PPOPolicy(BasePolicy):
    """Proximal Policy Optimization with a clipped surrogate objective and GAE.

    Parameters
    ----------
    obs_spec :
        Observation spec providing ``dim`` and ``scales`` for normalisation.
    action_decoder :
        Callable ``(action_idx: int) -> np.ndarray`` mapping a sampled index to
        the action array returned by ``__call__``.
    output_dim :
        Number of discrete actions (= softmax output width).
    hidden_sizes :
        MLP hidden-layer widths for both the actor and the critic (default ``[64, 64]``).
    learning_rate :
        Adam step size (shared by actor and critic).
    gamma :
        Discount factor.
    gae_lambda :
        GAE smoothing parameter (1.0 = Monte-Carlo, 0.0 = one-step TD).
    clip_range :
        PPO clipping epsilon for the probability ratio.
    n_epochs :
        Number of optimisation passes over each collected episode.
    entropy_coeff :
        Entropy-bonus weight (encourages exploration; 0 disables the term).
    value_coeff :
        Weight on the value (critic) loss.
    minibatch_size :
        Transitions per gradient step within an epoch.
    available_actions_fn :
        Optional callable ``(info: dict) -> np.ndarray[bool]`` masking illegal
        actions before sampling (and during the update recompute).
    seed :
        RNG seed for weight init + action sampling.
    """

    POLICY_TYPE = "ppo"
    LOOP_TYPE = "q_learning"
    VALID_POLICY_PARAMS = frozenset(
        {
            "hidden_sizes",
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_range",
            "n_epochs",
            "entropy_coeff",
            "value_coeff",
            "minibatch_size",
        }
    )

    def __init__(
        self,
        obs_spec: ObsSpec,
        action_decoder: Callable[[int], np.ndarray],
        *,
        output_dim: int,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        n_epochs: int = 4,
        entropy_coeff: float = 0.0,
        value_coeff: float = 0.5,
        minibatch_size: int = 64,
        available_actions_fn: Callable[..., np.ndarray] | None = None,
        seed: int | None = None,
    ) -> None:
        self._obs_spec = obs_spec
        self._action_decoder = action_decoder
        self._output_dim = int(output_dim)
        self._hidden = list(hidden_sizes or [64, 64])
        self._lr = float(learning_rate)
        self._gamma = float(gamma)
        self._lam = float(gae_lambda)
        self._clip = float(clip_range)
        self._n_epochs = int(n_epochs)
        self._entropy_coeff = float(entropy_coeff)
        self._value_coeff = float(value_coeff)
        self._minibatch_size = int(minibatch_size)
        self._avail_fn = available_actions_fn

        self._obs_dim = obs_spec.dim
        self._scales = obs_spec.scales
        self._rng = np.random.default_rng(seed)

        self._actor_w, self._actor_b = self._build_net(self._output_dim, seed)
        self._critic_w, self._critic_b = self._build_net(1, None if seed is None else seed + 1)
        self._init_adam()

        # Rollout buffer (one entry per non-warmup step)
        self._reset_buffer()

        # Cached availability mask (None = all actions legal)
        self._available_mask: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_net(self, out_dim: int, seed: int | None) -> tuple[list[np.ndarray], list[np.ndarray]]:
        rng = np.random.default_rng(seed)
        dims = [self._obs_dim] + self._hidden + [out_dim]
        weights, biases = [], []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            w = (rng.standard_normal((dims[i + 1], fan_in)) * np.sqrt(2.0 / fan_in)).astype(np.float32)
            weights.append(w)
            biases.append(np.zeros(dims[i + 1], dtype=np.float32))
        return weights, biases

    def _init_adam(self) -> None:
        self._am_w = [np.zeros_like(w) for w in self._actor_w]
        self._am_b = [np.zeros_like(b) for b in self._actor_b]
        self._av_w = [np.zeros_like(w) for w in self._actor_w]
        self._av_b = [np.zeros_like(b) for b in self._actor_b]
        self._cm_w = [np.zeros_like(w) for w in self._critic_w]
        self._cm_b = [np.zeros_like(b) for b in self._critic_b]
        self._cv_w = [np.zeros_like(w) for w in self._critic_w]
        self._cv_b = [np.zeros_like(b) for b in self._critic_b]
        self._adam_t = 0

    def _reset_buffer(self) -> None:
        self._obs_buf: list[np.ndarray] = []
        self._act_buf: list[int] = []
        self._logp_buf: list[float] = []
        self._val_buf: list[float] = []
        self._rew_buf: list[float] = []
        self._done_buf: list[bool] = []
        self._mask_buf: list[np.ndarray | None] = []

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_s = z - z.max()
        e = np.exp(z_s)
        return e / e.sum()

    def _mlp_forward(
        self, weights: list[np.ndarray], biases: list[np.ndarray], obs_norm: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Forward pass; returns (output_pre_activation, layer_inputs, pre_relu)."""
        x = obs_norm.astype(np.float32)
        layer_inputs: list = []
        pre_relu: list = []
        out = x
        for i, (w, b) in enumerate(zip(weights, biases)):
            layer_inputs.append(x.copy())
            z = w @ x + b
            if i < len(weights) - 1:
                pre_relu.append(z.copy())
                x = np.maximum(0.0, z)
            else:
                out = z
        return out, layer_inputs, pre_relu

    def _actor_probs(
        self, obs_norm: np.ndarray, mask: np.ndarray | None
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        logits, l_in, pre_r = self._mlp_forward(self._actor_w, self._actor_b, obs_norm)
        if mask is not None and not mask.all():
            logits = np.where(mask, logits, -1e9)
        return self._softmax(logits), l_in, pre_r

    def _critic_value(self, obs_norm: np.ndarray) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        out, l_in, pre_r = self._mlp_forward(self._critic_w, self._critic_b, obs_norm)
        return float(out[0]), l_in, pre_r

    # ------------------------------------------------------------------
    # Backward + Adam
    # ------------------------------------------------------------------

    def _mlp_backward(
        self,
        weights: list[np.ndarray],
        layer_inputs: list[np.ndarray],
        pre_relu: list[np.ndarray],
        grad_out: np.ndarray,
        dW: list[np.ndarray],
        dB: list[np.ndarray],
    ) -> None:
        """Accumulate gradients of a scalar loss into dW/dB given grad wrt output."""
        g = grad_out
        for i in range(len(weights) - 1, -1, -1):
            dW[i] += np.outer(g, layer_inputs[i])
            dB[i] += g
            if i > 0:
                g = (weights[i].T @ g) * (pre_relu[i - 1] > 0)

    def _adam_step(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        dW: list[np.ndarray],
        dB: list[np.ndarray],
        m_w: list[np.ndarray],
        m_b: list[np.ndarray],
        v_w: list[np.ndarray],
        v_b: list[np.ndarray],
        t: int,
    ) -> None:
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(weights)):
            m_w[i] = b1 * m_w[i] + (1.0 - b1) * dW[i]
            v_w[i] = b2 * v_w[i] + (1.0 - b2) * dW[i] ** 2
            mw_hat = m_w[i] / (1.0 - b1**t)
            vw_hat = v_w[i] / (1.0 - b2**t)
            weights[i] -= (self._lr * mw_hat / (np.sqrt(vw_hat) + eps)).astype(np.float32)

            m_b[i] = b1 * m_b[i] + (1.0 - b1) * dB[i]
            v_b[i] = b2 * v_b[i] + (1.0 - b2) * dB[i] ** 2
            mb_hat = m_b[i] / (1.0 - b1**t)
            vb_hat = v_b[i] / (1.0 - b2**t)
            biases[i] -= (self._lr * mb_hat / (np.sqrt(vb_hat) + eps)).astype(np.float32)

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def on_episode_start(self, **kwargs) -> None:
        self._reset_buffer()
        if self._avail_fn is not None:
            result = self._avail_fn(kwargs.get("info") or {})
            self._available_mask = None if result is None else np.asarray(result, dtype=bool)
        else:
            self._available_mask = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_norm = (obs / self._scales).astype(np.float32)
        mask = self._available_mask
        probs, _, _ = self._actor_probs(obs_norm, mask)
        action_idx = int(self._rng.choice(self._output_dim, p=probs))
        value, _, _ = self._critic_value(obs_norm)

        self._obs_buf.append(obs_norm.copy())
        self._act_buf.append(action_idx)
        self._logp_buf.append(float(np.log(probs[action_idx] + 1e-8)))
        self._val_buf.append(value)
        self._mask_buf.append(None if mask is None else mask.copy())
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
        self._rew_buf.append(float(reward))
        self._done_buf.append(bool(done))
        if self._avail_fn is not None:
            result = self._avail_fn(kwargs.get("info") or {})
            self._available_mask = None if result is None else np.asarray(result, dtype=bool)

    def _compute_gae(self, values: np.ndarray, rewards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (advantages, returns) via GAE(λ) over one episode (terminal bootstrap = 0)."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float64)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_value = values[t + 1] if t + 1 < T else 0.0
            delta = rewards[t] + self._gamma * next_value - values[t]
            last_gae = delta + self._gamma * self._lam * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def on_episode_end(self) -> None:
        T = min(len(self._obs_buf), len(self._rew_buf))
        if T == 0:
            self._reset_buffer()
            return

        obs = np.array(self._obs_buf[:T], dtype=np.float32)
        acts = np.array(self._act_buf[:T], dtype=np.int64)
        old_logp = np.array(self._logp_buf[:T], dtype=np.float64)
        values = np.array(self._val_buf[:T], dtype=np.float64)
        rewards = np.array(self._rew_buf[:T], dtype=np.float64)
        masks = self._mask_buf[:T]

        advantages, returns = self._compute_gae(values, rewards)
        adv_std = float(advantages.std())
        if adv_std > 1e-8:
            adv_norm = (advantages - advantages.mean()) / (adv_std + 1e-8)
        else:
            adv_norm = advantages - advantages.mean()

        idx_all = np.arange(T)
        mb = max(1, self._minibatch_size)
        for _ in range(self._n_epochs):
            self._rng.shuffle(idx_all)
            for start in range(0, T, mb):
                batch = idx_all[start : start + mb]
                self._update_minibatch(obs, acts, old_logp, adv_norm, returns, masks, batch)

        self._reset_buffer()

    def _update_minibatch(
        self,
        obs: np.ndarray,
        acts: np.ndarray,
        old_logp: np.ndarray,
        adv_norm: np.ndarray,
        returns: np.ndarray,
        masks: list[np.ndarray | None],
        batch: np.ndarray,
    ) -> None:
        adW = [np.zeros_like(w, dtype=np.float64) for w in self._actor_w]
        adB = [np.zeros_like(b, dtype=np.float64) for b in self._actor_b]
        cdW = [np.zeros_like(w, dtype=np.float64) for w in self._critic_w]
        cdB = [np.zeros_like(b, dtype=np.float64) for b in self._critic_b]

        for j in batch:
            obs_j = obs[j]
            a = int(acts[j])
            adv = float(adv_norm[j])

            probs, a_lin, a_pre = self._actor_probs(obs_j, masks[j])
            p_a = float(probs[a])
            new_logp = float(np.log(p_a + 1e-8))
            ratio = float(np.exp(np.clip(new_logp - float(old_logp[j]), -20.0, 20.0)))

            unclipped = ratio * adv
            clipped = float(np.clip(ratio, 1.0 - self._clip, 1.0 + self._clip)) * adv
            # Gradient of the clipped surrogate wrt the log-prob flows only on the
            # active (min) branch; when clipping binds, the branch is constant → 0.
            pg_coeff = ratio * adv if unclipped <= clipped else 0.0

            indicator = np.zeros(self._output_dim, dtype=np.float64)
            indicator[a] = 1.0
            dlogp_dlogits = indicator - probs
            grad_obj = pg_coeff * dlogp_dlogits

            if self._entropy_coeff > 0.0:
                log_p = np.log(probs.astype(np.float64) + 1e-8)
                H = -float(np.dot(probs, log_p))
                grad_obj += self._entropy_coeff * (-probs.astype(np.float64) * (log_p + H))

            # Minimise loss = -objective → grad of loss wrt logits is -grad_obj.
            self._mlp_backward(self._actor_w, a_lin, a_pre, -grad_obj, adW, adB)

            value, c_lin, c_pre = self._critic_value(obs_j)
            grad_v = np.array([self._value_coeff * 2.0 * (value - float(returns[j]))], dtype=np.float64)
            self._mlp_backward(self._critic_w, c_lin, c_pre, grad_v, cdW, cdB)

        n = float(len(batch))
        adW = [g / n for g in adW]
        adB = [g / n for g in adB]
        cdW = [g / n for g in cdW]
        cdB = [g / n for g in cdB]

        self._adam_t += 1
        t = self._adam_t
        self._adam_step(self._actor_w, self._actor_b, adW, adB, self._am_w, self._am_b, self._av_w, self._av_b, t)
        self._adam_step(self._critic_w, self._critic_b, cdW, cdB, self._cm_w, self._cm_b, self._cv_w, self._cv_b, t)

    # ------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------

    @classmethod
    def compatible_with(cls, game_name: str) -> tuple[bool, str | None]:
        return check_continuous_action_compatible(game_name, cls.POLICY_TYPE)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        return {
            "policy_type": "ppo",
            "hidden_sizes": self._hidden,
            "learning_rate": float(self._lr),
            "gamma": float(self._gamma),
            "gae_lambda": float(self._lam),
            "clip_range": float(self._clip),
            "n_epochs": self._n_epochs,
            "entropy_coeff": float(self._entropy_coeff),
            "value_coeff": float(self._value_coeff),
            "minibatch_size": self._minibatch_size,
            "output_dim": self._output_dim,
            "obs_dim": self._obs_dim,
            "actor_weights": [w.tolist() for w in self._actor_w],
            "actor_biases": [b.tolist() for b in self._actor_b],
            "critic_weights": [w.tolist() for w in self._critic_w],
            "critic_biases": [b.tolist() for b in self._critic_b],
        }

    @classmethod
    def from_cfg(
        cls,
        cfg: dict,
        obs_spec: ObsSpec,
        action_decoder: Callable[[int], np.ndarray],
    ) -> "PPOPolicy":
        obj = cls(
            obs_spec=obs_spec,
            action_decoder=action_decoder,
            output_dim=cfg.get("output_dim", len(cfg["actor_biases"][-1]) if "actor_biases" in cfg else 1),
            hidden_sizes=cfg.get("hidden_sizes", [64, 64]),
            learning_rate=cfg.get("learning_rate", 3e-4),
            gamma=cfg.get("gamma", 0.99),
            gae_lambda=cfg.get("gae_lambda", 0.95),
            clip_range=cfg.get("clip_range", 0.2),
            n_epochs=cfg.get("n_epochs", 4),
            entropy_coeff=cfg.get("entropy_coeff", 0.0),
            value_coeff=cfg.get("value_coeff", 0.5),
            minibatch_size=cfg.get("minibatch_size", 64),
        )
        if "actor_weights" in cfg:
            loaded_w = [np.array(w, dtype=np.float32) for w in cfg["actor_weights"]]
            if loaded_w[0].shape[1] != obj._obs_dim:
                raise ValueError(
                    f"PPOPolicy.from_cfg: weight shape mismatch — first layer input dim "
                    f"{loaded_w[0].shape[1]} but obs_dim is {obj._obs_dim}"
                )
            obj._actor_w = loaded_w
            obj._actor_b = [np.array(b, dtype=np.float32) for b in cfg["actor_biases"]]
            obj._critic_w = [np.array(w, dtype=np.float32) for w in cfg["critic_weights"]]
            obj._critic_b = [np.array(b, dtype=np.float32) for b in cfg["critic_biases"]]
            obj._init_adam()
        return obj

    def save_trainer_state(self, path: str) -> None:
        """Persist Adam optimiser moments + obs_dim to an .npz file."""
        arrays: dict = {
            "obs_dim": np.int64(self._obs_dim),
            "adam_t": np.int64(self._adam_t),
            "n_actor_layers": np.int64(len(self._actor_w)),
            "n_critic_layers": np.int64(len(self._critic_w)),
        }
        for i in range(len(self._actor_w)):
            arrays[f"am_w_{i}"] = self._am_w[i]
            arrays[f"am_b_{i}"] = self._am_b[i]
            arrays[f"av_w_{i}"] = self._av_w[i]
            arrays[f"av_b_{i}"] = self._av_b[i]
        for i in range(len(self._critic_w)):
            arrays[f"cm_w_{i}"] = self._cm_w[i]
            arrays[f"cm_b_{i}"] = self._cm_b[i]
            arrays[f"cv_w_{i}"] = self._cv_w[i]
            arrays[f"cv_b_{i}"] = self._cv_b[i]
        np.savez(path, **arrays)
        logger.debug("[PPOPolicy] trainer state saved → %s", path)

    def load_trainer_state(self, path: str) -> None:
        """Restore Adam moments from an .npz file.  Raises ValueError on obs_dim mismatch."""
        with np.load(path) as data:
            saved_obs_dim = int(data["obs_dim"])
            if saved_obs_dim != self._obs_dim:
                raise ValueError(
                    f"PPOPolicy: trainer state obs_dim mismatch — saved={saved_obs_dim}, "
                    f"current={self._obs_dim}. Use --re-initialize to restart from scratch."
                )
            if int(data["n_actor_layers"]) != len(self._actor_w):
                raise ValueError("PPOPolicy: trainer state actor layer count mismatch.")
            self._adam_t = int(data["adam_t"])
            for i in range(len(self._actor_w)):
                self._am_w[i] = data[f"am_w_{i}"]
                self._am_b[i] = data[f"am_b_{i}"]
                self._av_w[i] = data[f"av_w_{i}"]
                self._av_b[i] = data[f"av_b_{i}"]
            for i in range(len(self._critic_w)):
                self._cm_w[i] = data[f"cm_w_{i}"]
                self._cm_b[i] = data[f"cm_b_{i}"]
                self._cv_w[i] = data[f"cv_w_{i}"]
                self._cv_b[i] = data[f"cv_b_{i}"]
        logger.info("[PPOPolicy] trainer state loaded from %s", path)

    # ------------------------------------------------------------------
    # Construction / resume (framework make() path)
    # ------------------------------------------------------------------

    @classmethod
    def _construct_or_resume(
        cls, *, obs_spec, head_names, discrete_actions, weights_file, policy_params, re_initialize
    ) -> "PPOPolicy":
        pp = policy_params
        action_decoder = lambda i: discrete_actions[i]  # noqa: E731

        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as f:
                cfg = yaml.safe_load(f) or {}
            if isinstance(cfg, dict) and cfg.get("policy_type") == "ppo":
                policy = cls.from_cfg(cfg, obs_spec, action_decoder)
                ts = trainer_state_path(weights_file)
                if os.path.exists(ts):
                    try:
                        policy.load_trainer_state(ts)
                        logger.info("[PPOPolicy] loaded trainer state from %s", ts)
                    except (ValueError, KeyError) as exc:
                        logger.warning(
                            "[PPOPolicy] could not load trainer state — %s; continuing with default state.", exc
                        )
                return policy

        return cls(
            obs_spec=obs_spec,
            action_decoder=action_decoder,
            output_dim=len(discrete_actions),
            hidden_sizes=pp.get("hidden_sizes", [64, 64]),
            learning_rate=pp.get("learning_rate", 3e-4),
            gamma=pp.get("gamma", 0.99),
            gae_lambda=pp.get("gae_lambda", 0.95),
            clip_range=pp.get("clip_range", 0.2),
            n_epochs=pp.get("n_epochs", 4),
            entropy_coeff=pp.get("entropy_coeff", 0.0),
            value_coeff=pp.get("value_coeff", 0.5),
            minibatch_size=pp.get("minibatch_size", 64),
        )
