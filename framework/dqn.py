"""Generic Deep Q-Network policy.

DQNPolicy — MLP Q-network (pure numpy) with experience replay, target network,
            and Adam optimiser.  Parameterised on an ObsSpec and a discrete
            action array so it can be used by any game integration.
            Optional ``available_actions_fn`` activates action-availability
            masking (switches internal buffer to MaskedReplayBuffer and masks
            Q-values before argmax).
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from framework.obs_spec import ObsSpec
from framework.policies import BasePolicy
from framework.replay import MaskedReplayBuffer, ReplayBuffer

logger = logging.getLogger(__name__)


class DQNPolicy(BasePolicy):
    """Deep Q-network with experience replay and a periodically-synced target network.

    Architecture: obs → Linear → ReLU → ... → Linear(n_actions)
    Pure numpy; Adam optimiser; ε-greedy exploration with linear decay.

    Parameters
    ----------
    obs_spec :
        Observation spec.  Provides ``dim`` (input width) and ``scales``
        (per-feature normalisation).
    discrete_actions :
        Array of shape ``(n_actions, action_dim)``; each row is one action.
    hidden_sizes :
        MLP hidden-layer widths (default ``[64, 64]``).
    replay_buffer_size :
        Capacity of the circular experience-replay buffer.
    batch_size :
        Mini-batch size for gradient steps.
    min_replay_size :
        Buffer fill level before gradient steps start.
    target_update_freq :
        Gradient steps between target-network syncs.
    learning_rate :
        Adam step size.
    epsilon_start / epsilon_end / epsilon_decay_steps :
        ε-greedy schedule (linear decay over *epsilon_decay_steps* env steps).
    gamma :
        Discount factor.
    available_actions_fn :
        Optional callable ``(info: dict) -> np.ndarray[bool]`` of shape
        ``(n_actions,)``.  When provided, illegal actions are masked to
        ``-inf`` before greedy selection, random exploration is restricted to
        legal actions, and the replay buffer stores the mask so the target
        network never bootstraps through unavailable Q-values.
    seed :
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        discrete_actions: np.ndarray,
        *,
        hidden_sizes: list[int] | None = None,
        replay_buffer_size: int = 10_000,
        batch_size: int = 64,
        min_replay_size: int = 500,
        target_update_freq: int = 200,
        learning_rate: float = 0.001,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5_000,
        gamma: float = 0.99,
        available_actions_fn: Callable[[dict], np.ndarray] | None = None,
        seed: int | None = None,
    ) -> None:
        self._obs_spec    = obs_spec
        self._obs_dim     = obs_spec.dim
        self._scales      = obs_spec.scales
        self._actions     = np.array(discrete_actions, dtype=np.float32)
        self._n_actions   = len(self._actions)
        self._hidden      = list(hidden_sizes or [64, 64])
        self._buf_maxlen  = int(replay_buffer_size)
        self._batch_size  = int(batch_size)
        self._min_replay  = int(min_replay_size)
        self._target_freq = int(target_update_freq)
        self._lr          = float(learning_rate)
        self._eps_start   = float(epsilon_start)
        self._eps         = float(epsilon_start)
        self._eps_end     = float(epsilon_end)
        self._eps_steps   = int(epsilon_decay_steps)
        self._eps_delta   = (float(epsilon_start) - float(epsilon_end)) / max(1, int(epsilon_decay_steps))
        self._gamma       = float(gamma)
        self._avail_fn    = available_actions_fn
        self._seed        = seed

        self._masked      = available_actions_fn is not None
        if self._masked:
            self._replay: ReplayBuffer = MaskedReplayBuffer(replay_buffer_size, self._n_actions)
            self._cached_mask = np.ones(self._n_actions, dtype=bool)
        else:
            self._replay = ReplayBuffer(replay_buffer_size)

        self._total_steps = 0
        self._grad_steps  = 0

        self._online = self._build_net()
        self._target = self._build_net()
        self._sync_target()

        self._m_w = [np.zeros_like(w) for w in self._online["weights"]]
        self._m_b = [np.zeros_like(b) for b in self._online["biases"]]
        self._v_w = [np.zeros_like(w) for w in self._online["weights"]]
        self._v_b = [np.zeros_like(b) for b in self._online["biases"]]
        self._adam_t = 0

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_cfg(
        cls,
        cfg: dict,
        obs_spec: ObsSpec,
        discrete_actions: np.ndarray,
        available_actions_fn: Callable[[dict], np.ndarray] | None = None,
    ) -> "DQNPolicy":
        obj = cls(
            obs_spec            = obs_spec,
            discrete_actions    = discrete_actions,
            hidden_sizes        = cfg.get("hidden_sizes",        [64, 64]),
            replay_buffer_size  = cfg.get("replay_buffer_size",  10_000),
            batch_size          = cfg.get("batch_size",          64),
            min_replay_size     = cfg.get("min_replay_size",     500),
            target_update_freq  = cfg.get("target_update_freq",  200),
            learning_rate       = cfg.get("learning_rate",       0.001),
            epsilon_start       = cfg.get("epsilon_start",       1.0),
            epsilon_end         = cfg.get("epsilon_end",         0.05),
            epsilon_decay_steps = cfg.get("epsilon_decay_steps", 5_000),
            gamma               = cfg.get("gamma",               0.99),
            available_actions_fn = available_actions_fn,
            seed                = cfg.get("seed",                None),
        )
        if "online_weights" in cfg:
            required = ["online_weights", "online_biases", "target_weights", "target_biases"]
            missing  = [k for k in required if k not in cfg]
            if missing:
                raise KeyError(f"DQNPolicy.from_cfg: missing keys {missing}")

            loaded_w = [np.array(w, dtype=np.float32) for w in cfg["online_weights"]]
            if loaded_w[0].shape[1] != obj._obs_dim:
                raise ValueError(
                    f"DQNPolicy.from_cfg: weight shape mismatch — "
                    f"first layer has input dim {loaded_w[0].shape[1]} "
                    f"but obs_dim is {obj._obs_dim}"
                )
            obj._online["weights"] = loaded_w
            obj._online["biases"]  = [np.array(b, dtype=np.float32) for b in cfg["online_biases"]]
            obj._target["weights"] = [np.array(w, dtype=np.float32) for w in cfg["target_weights"]]
            obj._target["biases"]  = [np.array(b, dtype=np.float32) for b in cfg["target_biases"]]
            obj._eps         = float(cfg.get("epsilon",      obj._eps_end))
            obj._total_steps = int(cfg.get("total_steps",   0))
            obj._grad_steps  = int(cfg.get("grad_steps",    0))
            obj._m_w = [np.zeros_like(w) for w in obj._online["weights"]]
            obj._m_b = [np.zeros_like(b) for b in obj._online["biases"]]
            obj._v_w = [np.zeros_like(w) for w in obj._online["weights"]]
            obj._v_b = [np.zeros_like(b) for b in obj._online["biases"]]
            logger.info("[DQNPolicy] loaded weights from cfg (eps=%.4f, steps=%d)",
                        obj._eps, obj._total_steps)
        return obj

    def _build_net(self) -> dict:
        rng  = np.random.default_rng(self._seed)
        dims = [self._obs_dim] + self._hidden + [self._n_actions]
        weights, biases = [], []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            w = rng.standard_normal((dims[i + 1], fan_in)).astype(np.float32)
            w *= np.sqrt(2.0 / fan_in)
            b = np.zeros(dims[i + 1], dtype=np.float32)
            weights.append(w)
            biases.append(b)
        return {"weights": weights, "biases": biases}

    def _sync_target(self) -> None:
        self._target["weights"] = [w.copy() for w in self._online["weights"]]
        self._target["biases"]  = [b.copy() for b in self._online["biases"]]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, net: dict, x: np.ndarray) -> tuple[np.ndarray, list, list]:
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]
        h: np.ndarray = x.astype(np.float32)
        layer_inputs: list[np.ndarray] = []
        pre_relu:     list[np.ndarray] = []
        for i, (w, b) in enumerate(zip(net["weights"], net["biases"])):
            layer_inputs.append(h)
            z = h @ w.T + b
            if i < len(net["weights"]) - 1:
                pre_relu.append(z)
                h = np.maximum(0.0, z)
            else:
                h = z
        return (h[0] if single else h), layer_inputs, pre_relu

    def _q_values(self, net: dict, obs_norm: np.ndarray) -> np.ndarray:
        q, _, _ = self._forward(net, obs_norm)
        return q

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _gradient_step(
        self,
        obs_b: np.ndarray,
        act_b: np.ndarray,
        rew_b: np.ndarray,
        next_b: np.ndarray,
        done_b: np.ndarray,
        mask_b: np.ndarray | None = None,
    ) -> None:
        obs_norm  = obs_b  / self._scales
        next_norm = next_b / self._scales
        B         = len(act_b)

        q_next = self._q_values(self._target, next_norm)
        if mask_b is not None:
            q_next_masked = q_next.copy()
            q_next_masked[~mask_b] = -np.inf
            # fall back to unmasked argmax if all masked for a row
            all_masked = ~mask_b.any(axis=1)
            if all_masked.any():
                q_next_masked[all_masked] = q_next[all_masked]
            targets = rew_b + self._gamma * np.max(q_next_masked, axis=1) * (1.0 - done_b)
        else:
            targets = rew_b + self._gamma * np.max(q_next, axis=1) * (1.0 - done_b)

        q_all, layer_inputs, pre_relu = self._forward(self._online, obs_norm)

        grad_out = np.zeros_like(q_all)
        grad_out[np.arange(B), act_b] = 2.0 * (q_all[np.arange(B), act_b] - targets) / B

        g = grad_out
        grad_params: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(self._online["weights"]) - 1, -1, -1):
            a_in = layer_inputs[i]
            dW   = g.T @ a_in
            db   = g.sum(axis=0)
            grad_params.append((dW, db))
            if i > 0:
                g = (g @ self._online["weights"][i]) * (pre_relu[i - 1] > 0)
        grad_params.reverse()

        self._adam_t += 1
        t      = self._adam_t
        b1, b2 = 0.9, 0.999
        eps_a  = 1e-8
        for i, (dW, db) in enumerate(grad_params):
            self._m_w[i] = b1 * self._m_w[i] + (1.0 - b1) * dW
            self._v_w[i] = b2 * self._v_w[i] + (1.0 - b2) * dW ** 2
            mw_hat = self._m_w[i] / (1.0 - b1 ** t)
            vw_hat = self._v_w[i] / (1.0 - b2 ** t)
            self._online["weights"][i] -= self._lr * mw_hat / (np.sqrt(vw_hat) + eps_a)

            self._m_b[i] = b1 * self._m_b[i] + (1.0 - b1) * db
            self._v_b[i] = b2 * self._v_b[i] + (1.0 - b2) * db ** 2
            mb_hat = self._m_b[i] / (1.0 - b1 ** t)
            vb_hat = self._v_b[i] / (1.0 - b2 ** t)
            self._online["biases"][i] -= self._lr * mb_hat / (np.sqrt(vb_hat) + eps_a)

        self._grad_steps += 1
        if self._grad_steps % self._target_freq == 0:
            self._sync_target()
            logger.debug("[DQNPolicy] target synced at grad_step %d", self._grad_steps)

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self._masked:
            mask = self._cached_mask
            if np.random.random() < self._eps:
                available = np.where(mask)[0]
                if len(available) == 0:
                    available = np.arange(self._n_actions)
                return self._actions[np.random.choice(available)].copy()
            obs_norm = (obs / self._scales).astype(np.float32)
            q = self._q_values(self._online, obs_norm).copy()
            q[~mask] = -np.inf
            if np.all(~mask):
                q = self._q_values(self._online, obs_norm)
            return self._actions[int(np.argmax(q))].copy()
        else:
            if np.random.random() < self._eps:
                return self._actions[np.random.randint(self._n_actions)].copy()
            obs_norm = (obs / self._scales).astype(np.float32)
            q        = self._q_values(self._online, obs_norm)
            return self._actions[int(np.argmax(q))].copy()

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray | int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        **kwargs,
    ) -> None:
        if isinstance(action, (int, np.integer)):
            action_idx = int(action)
        else:
            diffs = np.abs(self._actions - action[np.newaxis, :]).sum(axis=1)
            action_idx = int(np.argmin(diffs))

        if self._masked:
            info = kwargs.get("info") or {}
            if self._avail_fn is not None:
                new_mask = self._avail_fn(info)
                self._cached_mask = np.asarray(new_mask, dtype=bool)
            mask = self._cached_mask
            self._replay.push(obs, action_idx, reward, next_obs, done, mask)  # type: ignore[call-arg]
        else:
            self._replay.push(obs, action_idx, reward, next_obs, done)

        self._total_steps += 1
        self._eps = max(self._eps_end, self._eps - self._eps_delta)

        if len(self._replay) >= self._min_replay:
            if self._masked:
                obs_b, act_b, rew_b, next_b, done_b, mask_b = self._replay.sample(self._batch_size)  # type: ignore[misc]
                self._gradient_step(obs_b, act_b, rew_b, next_b, done_b, mask_b)
            else:
                obs_b, act_b, rew_b, next_b, done_b = self._replay.sample(self._batch_size)
                self._gradient_step(obs_b, act_b, rew_b, next_b, done_b)

    def on_episode_end(self) -> None:
        pass  # epsilon decays per step, not per episode

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        return {
            "policy_type":         "dqn",
            "hidden_sizes":        self._hidden,
            "replay_buffer_size":  self._buf_maxlen,
            "batch_size":          self._batch_size,
            "min_replay_size":     self._min_replay,
            "target_update_freq":  self._target_freq,
            "learning_rate":       float(self._lr),
            "epsilon_start":       float(self._eps_start),
            "epsilon_end":         float(self._eps_end),
            "epsilon_decay_steps": self._eps_steps,
            "gamma":               float(self._gamma),
            "epsilon":             float(self._eps),
            "total_steps":         self._total_steps,
            "grad_steps":          self._grad_steps,
            "online_weights":      [w.tolist() for w in self._online["weights"]],
            "online_biases":       [b.tolist() for b in self._online["biases"]],
            "target_weights":      [w.tolist() for w in self._target["weights"]],
            "target_biases":       [b.tolist() for b in self._target["biases"]],
        }

    # ------------------------------------------------------------------
    # Trainer state persistence
    # ------------------------------------------------------------------

    def save_trainer_state(self, path: str) -> None:
        """Persist replay buffer and Adam optimiser moments to an .npz file."""
        buf = list(self._replay._buf)
        n   = len(buf)
        if n > 0:
            obs_arr  = np.stack([t[0] for t in buf]).astype(np.float32)
            act_arr  = np.array([t[1] for t in buf], dtype=np.int32)
            rew_arr  = np.array([t[2] for t in buf], dtype=np.float32)
            next_arr = np.stack([t[3] for t in buf]).astype(np.float32)
            done_arr = np.array([t[4] for t in buf], dtype=np.float32)
            if self._masked:
                mask_arr = np.stack([t[5] for t in buf])  # (n, n_actions) bool
        else:
            obs_arr  = np.empty((0, self._obs_dim), dtype=np.float32)
            act_arr  = np.empty(0, dtype=np.int32)
            rew_arr  = np.empty(0, dtype=np.float32)
            next_arr = np.empty((0, self._obs_dim), dtype=np.float32)
            done_arr = np.empty(0, dtype=np.float32)
            if self._masked:
                mask_arr = np.empty((0, self._n_actions), dtype=bool)

        n_layers = len(self._m_w)
        arrays: dict = dict(
            replay_obs  = obs_arr,
            replay_act  = act_arr,
            replay_rew  = rew_arr,
            replay_next = next_arr,
            replay_done = done_arr,
            total_steps = np.int64(self._total_steps),
        )
        if self._masked:
            arrays["replay_mask"] = mask_arr
        arrays.update(
            grad_steps  = np.int64(self._grad_steps),
            adam_t      = np.int64(self._adam_t),
            epsilon     = np.float32(self._eps),
            obs_dim     = np.int64(self._obs_dim),
            n_layers    = np.int64(n_layers),
        )
        for i in range(n_layers):
            arrays[f"m_w_{i}"] = self._m_w[i]
            arrays[f"m_b_{i}"] = self._m_b[i]
            arrays[f"v_w_{i}"] = self._v_w[i]
            arrays[f"v_b_{i}"] = self._v_b[i]
        np.savez(path, **arrays)
        logger.debug("[DQNPolicy] trainer state saved → %s (buf=%d)", path, n)

    def load_trainer_state(self, path: str) -> None:
        """Restore replay buffer and Adam optimiser moments from an .npz file.

        Raises ValueError if obs_dim or layer count do not match the current
        network architecture.
        """
        with np.load(path) as data:
            saved_obs_dim = int(data["obs_dim"])
            if saved_obs_dim != self._obs_dim:
                raise ValueError(
                    f"DQNPolicy: trainer state obs_dim mismatch — "
                    f"saved={saved_obs_dim}, current={self._obs_dim}. "
                    f"Use --re-initialize to restart from scratch."
                )
            n_layers = int(data["n_layers"])
            if n_layers != len(self._m_w):
                raise ValueError(
                    f"DQNPolicy: trainer state n_layers mismatch — "
                    f"saved={n_layers}, current={len(self._m_w)}. "
                    f"Use --re-initialize to restart from scratch."
                )
            for i in range(n_layers):
                mw = data[f"m_w_{i}"]
                if mw.shape != self._m_w[i].shape:
                    raise ValueError(
                        f"DQNPolicy: Adam moment m_w[{i}] shape mismatch — "
                        f"saved={mw.shape}, current={self._m_w[i].shape}. "
                        f"Use --re-initialize to restart from scratch."
                    )

            if self._masked:
                self._replay = MaskedReplayBuffer(self._buf_maxlen, self._n_actions)
            else:
                self._replay = ReplayBuffer(self._buf_maxlen)
            has_masks = self._masked and "replay_mask" in data
            for i in range(len(data["replay_obs"])):
                if has_masks:
                    self._replay.push(
                        data["replay_obs"][i], int(data["replay_act"][i]),
                        float(data["replay_rew"][i]), data["replay_next"][i],
                        bool(data["replay_done"][i]),
                        mask=data["replay_mask"][i],
                    )
                else:
                    self._replay.push(
                        data["replay_obs"][i], int(data["replay_act"][i]),
                        float(data["replay_rew"][i]), data["replay_next"][i],
                        bool(data["replay_done"][i]),
                    )

            self._total_steps = int(data["total_steps"])
            self._grad_steps  = int(data["grad_steps"])
            self._adam_t      = int(data["adam_t"])
            self._eps         = float(data["epsilon"])
            for i in range(n_layers):
                self._m_w[i] = data[f"m_w_{i}"]
                self._m_b[i] = data[f"m_b_{i}"]
                self._v_w[i] = data[f"v_w_{i}"]
                self._v_b[i] = data[f"v_b_{i}"]
        logger.info(
            "[DQNPolicy] trainer state loaded from %s (buf=%d, steps=%d, eps=%.4f)",
            path, len(self._replay), self._total_steps, self._eps,
        )
