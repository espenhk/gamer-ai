"""SC2-specific policies: NeuralDQNPolicy, CMAESPolicy, REINFORCEPolicy,
LSTMPolicy, LSTMEvolutionPolicy.

These policies are adapted from the TMNF equivalents in games/tmnf/policies.py
but are parameterised by an :class:`framework.obs_spec.ObsSpec` instance and
the SC2 9-element discrete action set instead of the TMNF-specific obs space
and 25-element action set.

The CMAESPolicy operates over the flat weight vector of a framework
:class:`framework.policies.WeightedLinearPolicy`, exactly as it does in TMNF,
but the weight vector dimension is ``obs_spec.dim × len(head_names)`` rather
than the TMNF-fixed ``(BASE_OBS_DIM + n_lidar_rays) × 3``.

Training-loop dispatch:
    neural_dqn  → ``_greedy_loop_q_learning``
    cmaes       → ``_greedy_loop_cmaes``
    reinforce   → ``_greedy_loop_q_learning``
    lstm        → ``_greedy_loop_cmaes``
"""
from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

from framework.obs_spec import ObsSpec
from framework.policies import BasePolicy, WeightedLinearPolicy
from games.sc2.actions import DISCRETE_ACTIONS

logger = logging.getLogger(__name__)

_N_DISCRETE_ACTIONS = len(DISCRETE_ACTIONS)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _action_to_idx(action: np.ndarray) -> int:
    """Map a 4-vector action back to its nearest row index in DISCRETE_ACTIONS."""
    diffs = np.abs(DISCRETE_ACTIONS - action[np.newaxis, :]).sum(axis=1)
    return int(np.argmin(diffs))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class _ReplayBuffer:
    """Fixed-size circular buffer of (obs, action_idx, reward, next_obs, done)."""

    def __init__(self, maxlen: int) -> None:
        self._buf: deque = deque(maxlen=maxlen)

    def push(self, obs: np.ndarray, action_idx: int, reward: float,
             next_obs: np.ndarray, done: bool) -> None:
        self._buf.append((obs.copy(), int(action_idx), float(reward),
                          next_obs.copy(), bool(done)))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray,
                                               np.ndarray]:
        replace = batch_size > len(self._buf)
        idxs    = np.random.choice(len(self._buf), size=batch_size, replace=replace)
        batch   = [self._buf[i] for i in idxs]
        obs_b   = np.stack([t[0] for t in batch]).astype(np.float32)
        act_b   = np.array([t[1] for t in batch], dtype=np.int32)
        rew_b   = np.array([t[2] for t in batch], dtype=np.float32)
        next_b  = np.stack([t[3] for t in batch]).astype(np.float32)
        done_b  = np.array([t[4] for t in batch], dtype=np.float32)
        return obs_b, act_b, rew_b, next_b, done_b

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# NeuralDQNPolicy
# ---------------------------------------------------------------------------

class NeuralDQNPolicy(BasePolicy):
    """DQN over the 9-element SC2 discrete action set.

    Architecture: obs → Linear → ReLU → … → Linear(9)
    Pure numpy; Adam optimiser; ε-greedy exploration with linear decay.

    Parameters
    ----------
    obs_spec :
        Observation spec.  Provides ``dim`` (input width) and ``scales``
        (per-feature normalisation).
    hidden_sizes :
        MLP hidden layer widths (default ``[64, 64]``).
    replay_buffer_size :
        Maximum number of transitions stored in the replay buffer.
    batch_size :
        Mini-batch size for gradient updates.
    min_replay_size :
        Number of transitions before gradient updates start.
    target_update_freq :
        Gradient steps between target-network syncs.
    learning_rate :
        Adam step size.
    epsilon_start / epsilon_end / epsilon_decay_steps :
        ε-greedy schedule (linear decay over *epsilon_decay_steps* environment steps).
    gamma :
        Discount factor.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        hidden_sizes: list[int] | None = None,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
        min_replay_size: int = 500,
        target_update_freq: int = 200,
        learning_rate: float = 0.001,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5000,
        gamma: float = 0.99,
        seed: int | None = None,
    ) -> None:
        self._obs_spec     = obs_spec
        self._obs_dim      = obs_spec.dim
        self._scales       = obs_spec.scales
        self._hidden       = list(hidden_sizes or [64, 64])
        self._buf_maxlen   = int(replay_buffer_size)
        self._batch_size   = int(batch_size)
        self._min_replay   = int(min_replay_size)
        self._target_freq  = int(target_update_freq)
        self._lr           = float(learning_rate)
        self._eps_start    = float(epsilon_start)
        self._eps          = float(epsilon_start)
        self._eps_end      = float(epsilon_end)
        self._eps_steps    = int(epsilon_decay_steps)
        self._eps_delta    = (float(epsilon_start) - float(epsilon_end)) / max(1, int(epsilon_decay_steps))
        self._gamma        = float(gamma)
        self._seed         = seed

        self._replay       = _ReplayBuffer(replay_buffer_size)
        self._total_steps  = 0
        self._grad_steps   = 0

        self._online = self._build_net()
        self._target = self._build_net()
        self._sync_target()

        self._m_w = [np.zeros_like(w) for w in self._online["weights"]]
        self._m_b = [np.zeros_like(b) for b in self._online["biases"]]
        self._v_w = [np.zeros_like(w) for w in self._online["weights"]]
        self._v_b = [np.zeros_like(b) for b in self._online["biases"]]
        self._adam_t = 0

    @classmethod
    def from_cfg(cls, cfg: dict, obs_spec: ObsSpec) -> "NeuralDQNPolicy":
        obj = cls(
            obs_spec            = obs_spec,
            hidden_sizes        = cfg.get("hidden_sizes",        [64, 64]),
            replay_buffer_size  = cfg.get("replay_buffer_size",  10000),
            batch_size          = cfg.get("batch_size",          64),
            min_replay_size     = cfg.get("min_replay_size",     500),
            target_update_freq  = cfg.get("target_update_freq",  200),
            learning_rate       = cfg.get("learning_rate",       0.001),
            epsilon_start       = cfg.get("epsilon_start",       1.0),
            epsilon_end         = cfg.get("epsilon_end",         0.05),
            epsilon_decay_steps = cfg.get("epsilon_decay_steps", 5000),
            gamma               = cfg.get("gamma",               0.99),
            seed                = cfg.get("seed",                None),
        )
        if "online_weights" in cfg:
            required = ["online_weights", "online_biases", "target_weights", "target_biases"]
            missing  = [k for k in required if k not in cfg]
            if missing:
                raise KeyError(f"NeuralDQNPolicy.from_cfg: missing keys {missing}")
            loaded_w = [np.array(w, dtype=np.float32) for w in cfg["online_weights"]]
            if loaded_w[0].shape[1] != obj._obs_dim:
                raise ValueError(
                    f"NeuralDQNPolicy.from_cfg: weight shape mismatch — "
                    f"first layer has input dim {loaded_w[0].shape[1]}, "
                    f"but obs_dim is {obj._obs_dim}"
                )
            obj._online["weights"] = loaded_w
            obj._online["biases"]  = [np.array(b, dtype=np.float32) for b in cfg["online_biases"]]
            obj._target["weights"] = [np.array(w, dtype=np.float32) for w in cfg["target_weights"]]
            obj._target["biases"]  = [np.array(b, dtype=np.float32) for b in cfg["target_biases"]]
            obj._eps         = float(cfg.get("epsilon",     obj._eps_end))
            obj._total_steps = int(cfg.get("total_steps",  0))
            obj._grad_steps  = int(cfg.get("grad_steps",   0))
            obj._m_w = [np.zeros_like(w) for w in obj._online["weights"]]
            obj._m_b = [np.zeros_like(b) for b in obj._online["biases"]]
            obj._v_w = [np.zeros_like(w) for w in obj._online["weights"]]
            obj._v_b = [np.zeros_like(b) for b in obj._online["biases"]]
        return obj

    def _build_net(self) -> dict:
        rng  = np.random.default_rng(self._seed)
        dims = [self._obs_dim] + self._hidden + [_N_DISCRETE_ACTIONS]
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

    def _gradient_step(self, obs_b: np.ndarray, act_b: np.ndarray,
                       rew_b: np.ndarray, next_b: np.ndarray,
                       done_b: np.ndarray) -> None:
        obs_norm  = obs_b  / self._scales
        next_norm = next_b / self._scales
        B = len(act_b)

        q_next  = self._q_values(self._target, next_norm)
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

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if np.random.random() < self._eps:
            return DISCRETE_ACTIONS[np.random.randint(_N_DISCRETE_ACTIONS)].copy()
        obs_norm = (obs / self._scales).astype(np.float32)
        q        = self._q_values(self._online, obs_norm)
        return DISCRETE_ACTIONS[int(np.argmax(q))].copy()

    def update(self, obs: np.ndarray, action: np.ndarray | int, reward: float,
               next_obs: np.ndarray, done: bool) -> None:
        action_idx = int(action) if np.isscalar(action) else _action_to_idx(action)
        self._replay.push(obs, action_idx, reward, next_obs, done)
        self._total_steps += 1
        self._eps = max(self._eps_end, self._eps - self._eps_delta)
        if len(self._replay) >= self._min_replay:
            obs_b, act_b, rew_b, next_b, done_b = self._replay.sample(self._batch_size)
            self._gradient_step(obs_b, act_b, rew_b, next_b, done_b)

    def on_episode_end(self) -> None:
        pass  # ε decays per step

    def to_cfg(self) -> dict:
        return {
            "policy_type":         "neural_dqn",
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

    def save_trainer_state(self, path: str) -> None:
        buf = list(self._replay._buf)
        n   = len(buf)
        if n > 0:
            obs_arr  = np.stack([t[0] for t in buf]).astype(np.float32)
            act_arr  = np.array([t[1] for t in buf], dtype=np.int32)
            rew_arr  = np.array([t[2] for t in buf], dtype=np.float32)
            next_arr = np.stack([t[3] for t in buf]).astype(np.float32)
            done_arr = np.array([t[4] for t in buf], dtype=np.float32)
        else:
            obs_arr  = np.empty((0, self._obs_dim), dtype=np.float32)
            act_arr  = np.empty(0, dtype=np.int32)
            rew_arr  = np.empty(0, dtype=np.float32)
            next_arr = np.empty((0, self._obs_dim), dtype=np.float32)
            done_arr = np.empty(0, dtype=np.float32)

        n_layers = len(self._m_w)
        arrays: dict = dict(
            replay_obs  = obs_arr,
            replay_act  = act_arr,
            replay_rew  = rew_arr,
            replay_next = next_arr,
            replay_done = done_arr,
            total_steps = np.int64(self._total_steps),
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

    def load_trainer_state(self, path: str) -> None:
        with np.load(path) as data:
            saved_obs_dim = int(data["obs_dim"])
            if saved_obs_dim != self._obs_dim:
                raise ValueError(
                    f"NeuralDQNPolicy: trainer state obs_dim mismatch — "
                    f"saved={saved_obs_dim}, current={self._obs_dim}. "
                    f"Use --re-initialize to restart from scratch."
                )
            n_layers = int(data["n_layers"])
            if n_layers != len(self._m_w):
                raise ValueError(
                    f"NeuralDQNPolicy: trainer state n_layers mismatch — "
                    f"saved={n_layers}, current={len(self._m_w)}. "
                    f"Use --re-initialize to restart from scratch."
                )
            for i in range(n_layers):
                mw = data[f"m_w_{i}"]
                if mw.shape != self._m_w[i].shape:
                    raise ValueError(
                        f"NeuralDQNPolicy: Adam moment m_w[{i}] shape mismatch — "
                        f"saved={mw.shape}, current={self._m_w[i].shape}. "
                        f"Use --re-initialize to restart from scratch."
                    )
            self._replay = _ReplayBuffer(self._buf_maxlen)
            for i in range(len(data["replay_obs"])):
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
            "[SC2 NeuralDQNPolicy] trainer state loaded from %s "
            "(buf=%d, steps=%d, eps=%.4f)",
            path, len(self._replay), self._total_steps, self._eps,
        )


# ---------------------------------------------------------------------------
# CMAESPolicy
# ---------------------------------------------------------------------------

class CMAESPolicy(BasePolicy):
    """(μ/μ_w, λ)-CMA-ES over the flat weight vector of a WeightedLinearPolicy.

    The weight vector has dimension ``obs_spec.dim × len(head_names)`` — for
    the default SC2 action representation (4 heads: fn_idx, x, y, queue) on
    the 13-dim minigame obs this is 52 params; on the 21-dim ladder obs it is
    84 params.

    Parameters
    ----------
    obs_spec :
        Observation spec for the target environment.
    head_names :
        Output head names passed to WeightedLinearPolicy (e.g.
        ``["fn_idx", "x", "y", "queue"]``).
    population_size :
        λ — offspring sampled per generation (default 20).
    initial_sigma :
        Starting step size (default 0.3).
    eval_episodes :
        Episodes per individual per generation (default 1).
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        head_names: list[str],
        population_size: int = 20,
        initial_sigma: float = 0.3,
        seed: int | None = None,
        eval_episodes: int = 1,
    ) -> None:
        self._obs_spec     = obs_spec
        self._head_names   = list(head_names)
        self._lam          = int(population_size)
        self._eval_episodes = max(1, int(eval_episodes))
        n                  = obs_spec.dim * len(head_names)
        self._n            = n

        mu            = self._lam // 2
        self._mu      = mu
        raw_w         = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)],
                                 dtype=np.float64)
        self._weights = raw_w / raw_w.sum()
        self._mu_eff  = 1.0 / float(np.sum(self._weights ** 2))

        self._cs   = (self._mu_eff + 2) / (n + self._mu_eff + 5)
        self._ds   = (1 + 2 * max(0.0, float(np.sqrt((self._mu_eff - 1) / (n + 1))) - 1)
                      + self._cs)
        self._chin = float(np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2)))

        self._cc  = (4 + self._mu_eff / n) / (n + 4 + 2 * self._mu_eff / n)
        self._c1  = 2.0 / ((n + 1.3) ** 2 + self._mu_eff)
        self._cmu = min(
            1.0 - self._c1,
            2.0 * (self._mu_eff - 2 + 1.0 / self._mu_eff) / ((n + 2) ** 2 + self._mu_eff),
        )

        self._rng = np.random.default_rng(seed)

        self._mean      = self._rng.standard_normal(n).astype(np.float64)
        self._sigma     = float(initial_sigma)
        self._ps        = np.zeros(n, dtype=np.float64)
        self._pc        = np.zeros(n, dtype=np.float64)
        self._C         = np.eye(n, dtype=np.float64)
        self._B         = np.eye(n, dtype=np.float64)
        self._D         = np.ones(n, dtype=np.float64)
        self._invsqrtC  = np.eye(n, dtype=np.float64)
        self._eigengen  = 0
        self._gen       = 0

        self._pop_xs: list[np.ndarray] = []
        self._pop_ys: list[np.ndarray] = []

        self._champion: WeightedLinearPolicy | None = None
        self._champion_reward: float = float("-inf")

    @property
    def champion_reward(self) -> float:
        return self._champion_reward

    @property
    def population_size(self) -> int:
        return self._lam

    @property
    def sigma(self) -> float:
        return self._sigma

    def initialize_random(self) -> None:
        self._mean = np.zeros(self._n, dtype=np.float64)
        logger.info("[SC2 CMAESPolicy] initialised with zero mean, sigma=%.3f", self._sigma)

    def initialize_from_champion(self, champion: WeightedLinearPolicy) -> None:
        seeded_reward = None
        for attr_name in ("champion_reward", "reward"):
            reward_value = getattr(champion, attr_name, None)
            if reward_value is not None:
                try:
                    seeded_reward = float(reward_value)
                except (TypeError, ValueError):
                    seeded_reward = None
                else:
                    if math.isfinite(seeded_reward):
                        break
                    seeded_reward = None

        if seeded_reward is None and math.isfinite(self._champion_reward):
            seeded_reward = float(self._champion_reward)

        self._champion_reward = seeded_reward if seeded_reward is not None else float("-inf")
        self._champion = champion
        self._mean = champion.to_flat().astype(np.float64)
        logger.info(
            "[SC2 CMAESPolicy] seeded mean from champion%s",
            "" if seeded_reward is None else f" (baseline reward={self._champion_reward:.6f})",
        )

    def _flat_to_policy(self, flat: np.ndarray) -> WeightedLinearPolicy:
        """Build a WeightedLinearPolicy from a flat [head0|head1|…] weight vector."""
        n     = self._obs_spec.dim
        names = self._obs_spec.names
        cfg   = {
            f"{head}_weights": {names[i]: float(flat[k * n + i]) for i in range(n)}
            for k, head in enumerate(self._head_names)
        }
        return WeightedLinearPolicy.from_cfg(cfg, self._obs_spec, self._head_names)

    def _update_eigen(self) -> None:
        self._C = np.triu(self._C) + np.triu(self._C, 1).T
        eigvals, self._B = np.linalg.eigh(self._C)
        eigvals          = np.maximum(eigvals, 1e-20)
        self._D          = np.sqrt(eigvals)
        self._invsqrtC   = self._B @ np.diag(1.0 / self._D) @ self._B.T
        self._eigengen   = self._gen

    def sample_population(self) -> list[WeightedLinearPolicy]:
        n = self._n
        if self._gen - self._eigengen >= max(1, self._lam // max(1, 10 * n)):
            self._update_eigen()

        self._pop_xs = []
        self._pop_ys = []
        for _ in range(self._lam):
            z = self._rng.standard_normal(n)
            y = self._B @ (self._D * z)
            x = self._mean + self._sigma * y
            self._pop_xs.append(x)
            self._pop_ys.append(y)

        return [self._flat_to_policy(x) for x in self._pop_xs]

    def update_distribution(self, rewards: list[float]) -> bool:
        if len(rewards) != self._lam:
            raise ValueError(f"Expected {self._lam} rewards, got {len(rewards)}")
        if len(self._pop_xs) != self._lam or len(self._pop_ys) != self._lam:
            raise RuntimeError(
                "update_distribution() called before a matching sample_population()."
            )
        n = self._n

        order = np.argsort(rewards)[::-1]

        improved = False
        best_r   = rewards[order[0]]
        if best_r > self._champion_reward:
            self._champion_reward = best_r
            self._champion        = self._flat_to_policy(self._pop_xs[order[0]])
            improved              = True

        elite_ys = np.stack([self._pop_ys[order[i]] for i in range(self._mu)])
        step     = np.einsum("i,ij->j", self._weights, elite_ys)

        self._mean = self._mean + self._sigma * step

        ps_scale  = float(np.sqrt(self._cs * (2 - self._cs) * self._mu_eff))
        self._ps  = (1 - self._cs) * self._ps + ps_scale * (self._invsqrtC @ step)

        ps_norm     = float(np.linalg.norm(self._ps))
        self._sigma = float(np.clip(
            self._sigma * np.exp((self._cs / self._ds) * (ps_norm / self._chin - 1)),
            1e-10, 1e6,
        ))

        ps_norm_normed = ps_norm / float(np.sqrt(1 - (1 - self._cs) ** (2 * (self._gen + 1))))
        h_sigma = 1.0 if ps_norm_normed < (1.4 + 2.0 / (n + 1)) * self._chin else 0.0

        pc_scale  = float(np.sqrt(self._cc * (2 - self._cc) * self._mu_eff))
        self._pc  = (1 - self._cc) * self._pc + h_sigma * pc_scale * step

        delta_h = (1 - h_sigma) * self._cc * (2 - self._cc)
        rank1   = np.outer(self._pc, self._pc)
        rank_mu = np.einsum("i,ij,ik->jk", self._weights, elite_ys, elite_ys)
        self._C = (
            (1 - self._c1 - self._cmu) * self._C
            + self._c1 * (rank1 + delta_h * self._C)
            + self._cmu * rank_mu
        )

        self._gen += 1
        return improved

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self._champion is None:
            raise RuntimeError(
                "SC2 CMAESPolicy: no champion yet — call sample_population() "
                "and update_distribution() for at least one generation first."
            )
        return self._champion(obs)

    def to_cfg(self) -> dict:
        return {
            "policy_type":     "cmaes",
            "population_size": self._lam,
            "sigma":           self._sigma,
            "obs_dim":         self._obs_spec.dim,
            "head_names":      self._head_names,
            "eval_episodes":   self._eval_episodes,
            "champion_reward": float(self._champion_reward),
        }

    def save(self, path: str) -> None:
        """Save champion in WeightedLinearPolicy YAML format."""
        if self._champion is not None:
            self._champion.save(path)

    def save_trainer_state(self, path: str) -> None:
        np.savez(
            path,
            mean     = self._mean,
            sigma    = np.float64(self._sigma),
            C        = self._C,
            B        = self._B,
            D        = self._D,
            invsqrtC = self._invsqrtC,
            ps       = self._ps,
            pc       = self._pc,
            gen      = np.int64(self._gen),
            n        = np.int64(self._n),
        )

    def load_trainer_state(self, path: str) -> None:
        with np.load(path) as data:
            n_saved = int(data["n"])
            if n_saved != self._n:
                raise ValueError(
                    f"SC2 CMAESPolicy: trainer state dimension mismatch — "
                    f"saved n={n_saved}, current n={self._n}. "
                    f"Use --re-initialize to restart from scratch."
                )
            self._mean     = data["mean"].astype(np.float64)
            self._sigma    = float(data["sigma"])
            self._C        = data["C"].astype(np.float64)
            self._B        = data["B"].astype(np.float64)
            self._D        = data["D"].astype(np.float64)
            self._invsqrtC = data["invsqrtC"].astype(np.float64)
            self._ps       = data["ps"].astype(np.float64)
            self._pc       = data["pc"].astype(np.float64)
            self._gen      = int(data["gen"])
        logger.info("[SC2 CMAESPolicy] trainer state loaded from %s (gen=%d, sigma=%.4f)",
                    path, self._gen, self._sigma)


# ---------------------------------------------------------------------------
# REINFORCEPolicy
# ---------------------------------------------------------------------------

class REINFORCEPolicy(BasePolicy):
    """REINFORCE (Monte Carlo Policy Gradient) over the 9-element SC2 action set.

    Dispatched via ``_greedy_loop_q_learning`` (``update()`` per step,
    ``on_episode_end()`` per episode).

    Parameters
    ----------
    obs_spec :
        Observation spec.
    hidden_sizes :
        MLP hidden layer widths (default ``[64, 64]``).
    learning_rate :
        Gradient-ascent step size (default ``0.001``).
    gamma :
        Discount factor (default ``0.99``).
    entropy_coeff :
        Entropy regularisation weight (default ``0.01``).
    baseline :
        ``"running_mean"`` (EMA of episode returns) or ``"none"``.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
        baseline: str = "running_mean",
        seed: int | None = None,
    ) -> None:
        self._obs_spec      = obs_spec
        self._obs_dim       = obs_spec.dim
        self._scales        = obs_spec.scales
        self._hidden        = list(hidden_sizes or [64, 64])
        self._lr            = float(learning_rate)
        self._gamma         = float(gamma)
        self._entropy_coeff = float(entropy_coeff)
        self._baseline_type = baseline

        self._weights, self._biases = self._build_net(seed)

        self._ep_grads: list[tuple]   = []
        self._ep_rewards: list[float] = []

        self._baseline_val   = 0.0
        self._baseline_alpha = 0.05

    def _build_net(self, seed: int | None) -> tuple[list[np.ndarray], list[np.ndarray]]:
        rng  = np.random.default_rng(seed)
        dims = [self._obs_dim] + self._hidden + [_N_DISCRETE_ACTIONS]
        weights, biases = [], []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            w = rng.standard_normal((dims[i + 1], fan_in)).astype(np.float32)
            w *= np.sqrt(2.0 / fan_in)
            b = np.zeros(dims[i + 1], dtype=np.float32)
            weights.append(w)
            biases.append(b)
        return weights, biases

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_s = z - z.max()
        e   = np.exp(z_s)
        return e / e.sum()

    def _forward(self, obs_norm: np.ndarray):
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

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_norm            = obs / self._scales
        probs, l_in, pre_r  = self._forward(obs_norm)
        action_idx          = int(np.random.choice(_N_DISCRETE_ACTIONS, p=probs))
        self._ep_grads.append((l_in, pre_r, probs.copy(), action_idx))
        return DISCRETE_ACTIONS[action_idx].copy()

    def update(self, obs: np.ndarray, action: np.ndarray | int, reward: float,
               next_obs: np.ndarray, done: bool) -> None:
        self._ep_rewards.append(float(reward))

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
            self._baseline_val = ((1 - self._baseline_alpha) * self._baseline_val
                                  + self._baseline_alpha * float(G[0]))

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

            delta         = -probs.copy().astype(np.float64)
            delta[a_idx] += 1.0
            delta         *= advantage

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

    def to_cfg(self) -> dict:
        return {
            "policy_type":    "reinforce",
            "hidden_sizes":   self._hidden,
            "learning_rate":  float(self._lr),
            "gamma":          float(self._gamma),
            "entropy_coeff":  float(self._entropy_coeff),
            "baseline":       self._baseline_type,
            "obs_dim":        self._obs_dim,
            "baseline_value": float(self._baseline_val),
            "weights":        [w.tolist() for w in self._weights],
            "biases":         [b.tolist() for b in self._biases],
        }

    @classmethod
    def from_cfg(cls, cfg: dict, obs_spec: ObsSpec) -> "REINFORCEPolicy":
        obj = cls(
            obs_spec      = obs_spec,
            hidden_sizes  = cfg.get("hidden_sizes",  [64, 64]),
            learning_rate = cfg.get("learning_rate", 0.001),
            gamma         = cfg.get("gamma",         0.99),
            entropy_coeff = cfg.get("entropy_coeff", 0.01),
            baseline      = cfg.get("baseline",      "running_mean"),
        )
        if "weights" in cfg:
            obj._weights = [np.array(w, dtype=np.float32) for w in cfg["weights"]]
            obj._biases  = [np.array(b, dtype=np.float32) for b in cfg["biases"]]
        if "baseline_value" in cfg:
            obj._baseline_val = float(cfg["baseline_value"])
        return obj

    def save_trainer_state(self, path: str) -> None:
        np.savez(path, baseline_val=np.float64(self._baseline_val),
                 obs_dim=np.int64(self._obs_dim))

    def load_trainer_state(self, path: str) -> None:
        with np.load(path) as data:
            saved_obs_dim = int(data["obs_dim"])
            if saved_obs_dim != self._obs_dim:
                raise ValueError(
                    f"SC2 REINFORCEPolicy: trainer state obs_dim mismatch — "
                    f"saved={saved_obs_dim}, current={self._obs_dim}. "
                    f"Use --re-initialize to restart from scratch."
                )
            self._baseline_val = float(data["baseline_val"])
        logger.info("[SC2 REINFORCEPolicy] trainer state loaded from %s", path)


# ---------------------------------------------------------------------------
# LSTMPolicy (SC2 variant)
# ---------------------------------------------------------------------------

class LSTMPolicy(BasePolicy):
    """Single-layer LSTM policy for SC2.

    Outputs a 4-vector ``[fn_idx, x, y, queue]`` matching the SC2 action
    space.  Trained via an outer evolutionary optimiser (:class:`LSTMEvolutionPolicy`).
    Hidden state is reset at the start of each episode.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        hidden_size: int = 32,
        seed: int | None = None,
    ) -> None:
        self._obs_spec    = obs_spec
        self._hidden_size = hidden_size
        self._obs_dim     = obs_spec.dim
        self._scales      = obs_spec.scales

        h    = hidden_size
        c_in = h + self._obs_dim
        rng  = np.random.default_rng(seed)
        gain = np.sqrt(2.0 / c_in)

        self._W_f = rng.standard_normal((h, c_in)).astype(np.float32) * gain
        self._b_f = np.zeros(h, dtype=np.float32)
        self._W_i = rng.standard_normal((h, c_in)).astype(np.float32) * gain
        self._b_i = np.zeros(h, dtype=np.float32)
        self._W_g = rng.standard_normal((h, c_in)).astype(np.float32) * gain
        self._b_g = np.zeros(h, dtype=np.float32)
        self._W_o = rng.standard_normal((h, c_in)).astype(np.float32) * gain
        self._b_o = np.zeros(h, dtype=np.float32)

        # Output heads: fn_idx, x, y, queue
        self._W_fn    = rng.standard_normal(h).astype(np.float32) * np.sqrt(2.0 / h)
        self._W_x     = rng.standard_normal(h).astype(np.float32) * np.sqrt(2.0 / h)
        self._W_y     = rng.standard_normal(h).astype(np.float32) * np.sqrt(2.0 / h)
        self._W_queue = rng.standard_normal(h).astype(np.float32) * np.sqrt(2.0 / h)

        self._h = np.zeros(h, dtype=np.float32)
        self._c = np.zeros(h, dtype=np.float32)

        # Number of available function IDs (for scaling fn_idx output)
        from games.sc2.actions import FUNCTION_IDS
        self._n_funcs = max(FUNCTION_IDS) + 1

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def flat_dim(self) -> int:
        h    = self._hidden_size
        c_in = h + self._obs_dim
        return 4 * (h * c_in + h) + 4 * h  # 4 LSTM gates + 4 output heads

    def to_flat(self) -> np.ndarray:
        return np.concatenate([
            self._W_f.ravel(), self._b_f,
            self._W_i.ravel(), self._b_i,
            self._W_g.ravel(), self._b_g,
            self._W_o.ravel(), self._b_o,
            self._W_fn,
            self._W_x,
            self._W_y,
            self._W_queue,
        ]).astype(np.float32)

    def with_flat(self, flat: np.ndarray) -> "LSTMPolicy":
        flat = np.asarray(flat, dtype=np.float32)
        if flat.shape[0] != self.flat_dim:
            raise ValueError(
                f"LSTMPolicy.with_flat: expected {self.flat_dim}, got {flat.shape[0]}"
            )

        obj = object.__new__(LSTMPolicy)
        obj._obs_spec    = self._obs_spec
        obj._hidden_size = self._hidden_size
        obj._obs_dim     = self._obs_dim
        obj._scales      = self._scales
        obj._n_funcs     = self._n_funcs

        h    = self._hidden_size
        c_in = h + self._obs_dim
        off  = 0

        def _take(shape: tuple) -> np.ndarray:
            nonlocal off
            n   = int(np.prod(shape))
            out = flat[off: off + n].reshape(shape).copy()
            off += n
            return out

        obj._W_f    = _take((h, c_in))
        obj._b_f    = _take((h,))
        obj._W_i    = _take((h, c_in))
        obj._b_i    = _take((h,))
        obj._W_g    = _take((h, c_in))
        obj._b_g    = _take((h,))
        obj._W_o    = _take((h, c_in))
        obj._b_o    = _take((h,))
        obj._W_fn    = _take((h,))
        obj._W_x     = _take((h,))
        obj._W_y     = _take((h,))
        obj._W_queue = _take((h,))
        obj._h = np.zeros(h, dtype=np.float32)
        obj._c = np.zeros(h, dtype=np.float32)
        return obj

    def _reset_hidden_state(self) -> None:
        self._h = np.zeros(self._hidden_size, dtype=np.float32)
        self._c = np.zeros(self._hidden_size, dtype=np.float32)

    def on_episode_start(self) -> None:
        self._reset_hidden_state()

    def on_episode_end(self) -> None:
        self._reset_hidden_state()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x  = (obs / self._scales).astype(np.float32)
        hx = np.concatenate([self._h, x])

        f = _sigmoid(self._W_f @ hx + self._b_f)
        i = _sigmoid(self._W_i @ hx + self._b_i)
        g = np.tanh(self._W_g  @ hx + self._b_g)
        o = _sigmoid(self._W_o @ hx + self._b_o)

        self._c = f * self._c + i * g
        self._h = o * np.tanh(self._c)

        # Map hidden state to 4-vector action [fn_idx, x, y, queue].
        # fn_idx is a continuous value in [0, n_funcs-1]; SC2Client converts
        # it to an integer via int(action[0]) in action_to_function_call.
        fn_idx = float(_sigmoid(np.dot(self._W_fn, self._h)) * (self._n_funcs - 1))
        x_out  = float(_sigmoid(np.dot(self._W_x, self._h)))
        y_out  = float(_sigmoid(np.dot(self._W_y, self._h)))
        # queue is 0.0 or 1.0 — kept as float for array dtype consistency.
        queue  = float(int(_sigmoid(np.dot(self._W_queue, self._h)) > 0.5))
        return np.array([fn_idx, x_out, y_out, queue], dtype=np.float32)

    def update(self, obs: np.ndarray, action: np.ndarray | int, reward: float,
               next_obs: np.ndarray, done: bool) -> None:
        pass  # trained via outer evolutionary optimiser

    def to_cfg(self) -> dict:
        return {
            "policy_type": "lstm",
            "hidden_size": self._hidden_size,
            "obs_dim":     self._obs_dim,
            "W_f": self._W_f.tolist(), "b_f": self._b_f.tolist(),
            "W_i": self._W_i.tolist(), "b_i": self._b_i.tolist(),
            "W_g": self._W_g.tolist(), "b_g": self._b_g.tolist(),
            "W_o": self._W_o.tolist(), "b_o": self._b_o.tolist(),
            "W_fn":    self._W_fn.tolist(),
            "W_x":     self._W_x.tolist(),
            "W_y":     self._W_y.tolist(),
            "W_queue": self._W_queue.tolist(),
        }

    @classmethod
    def from_cfg(cls, cfg: dict, obs_spec: ObsSpec) -> "LSTMPolicy":
        obj = object.__new__(cls)
        obj._obs_spec    = obs_spec
        obj._hidden_size = int(cfg["hidden_size"])
        obj._obs_dim     = obs_spec.dim
        obj._scales      = obs_spec.scales
        from games.sc2.actions import FUNCTION_IDS
        obj._n_funcs     = max(FUNCTION_IDS) + 1
        obj._W_f    = np.array(cfg["W_f"],    dtype=np.float32)
        obj._b_f    = np.array(cfg["b_f"],    dtype=np.float32)
        obj._W_i    = np.array(cfg["W_i"],    dtype=np.float32)
        obj._b_i    = np.array(cfg["b_i"],    dtype=np.float32)
        obj._W_g    = np.array(cfg["W_g"],    dtype=np.float32)
        obj._b_g    = np.array(cfg["b_g"],    dtype=np.float32)
        obj._W_o    = np.array(cfg["W_o"],    dtype=np.float32)
        obj._b_o    = np.array(cfg["b_o"],    dtype=np.float32)
        obj._W_fn    = np.array(cfg["W_fn"],    dtype=np.float32)
        obj._W_x     = np.array(cfg["W_x"],     dtype=np.float32)
        obj._W_y     = np.array(cfg["W_y"],     dtype=np.float32)
        obj._W_queue = np.array(cfg["W_queue"], dtype=np.float32)
        h = obj._hidden_size
        obj._h = np.zeros(h, dtype=np.float32)
        obj._c = np.zeros(h, dtype=np.float32)
        return obj


# ---------------------------------------------------------------------------
# LSTMEvolutionPolicy
# ---------------------------------------------------------------------------

class LSTMEvolutionPolicy(BasePolicy):
    """(μ/μ_w, λ)-isotropic ES outer optimiser wrapping :class:`LSTMPolicy`.

    Uses the ``_greedy_loop_cmaes`` interface: ``sample_population()`` /
    ``update_distribution()``.  Step size is adapted via the 1/5 success rule.

    Parameters
    ----------
    obs_spec :
        Observation spec for the target environment.
    hidden_size :
        LSTM hidden state dimensionality (default 32).
    population_size :
        λ — offspring evaluated per generation (default 20).
    initial_sigma :
        Starting perturbation scale (default 0.05).
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        hidden_size: int = 32,
        population_size: int = 20,
        initial_sigma: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self._lam      = int(population_size)
        self._sigma    = float(initial_sigma)
        self._obs_spec = obs_spec
        self._rng      = np.random.default_rng(seed)

        self._template = LSTMPolicy(obs_spec=obs_spec, hidden_size=hidden_size)
        self._flat_dim = self._template.flat_dim
        self._mean     = self._template.to_flat().astype(np.float64)

        mu         = self._lam // 2
        self._mu   = mu
        raw_w      = np.array([np.log(mu + 0.5) - np.log(i + 1)
                               for i in range(mu)], dtype=np.float64)
        self._recomb_w = raw_w / raw_w.sum()

        self._pop: list[np.ndarray] = []
        self._champion: LSTMPolicy | None  = None
        self._champion_reward: float       = float("-inf")

    @property
    def population_size(self) -> int:
        return self._lam

    @property
    def champion_reward(self) -> float:
        return self._champion_reward

    @property
    def sigma(self) -> float:
        return self._sigma

    def initialize_from_champion(self, champion: LSTMPolicy) -> None:
        if champion.flat_dim != self._flat_dim:
            raise ValueError(
                f"LSTMEvolutionPolicy: flat_dim mismatch — "
                f"expected {self._flat_dim}, got {champion.flat_dim}. "
                f"Use --re-initialize to restart from scratch."
            )
        self._champion = champion
        self._mean     = champion.to_flat().astype(np.float64)
        logger.info("[SC2 LSTMEvolutionPolicy] seeded mean from champion")

    def sample_population(self) -> list[LSTMPolicy]:
        self._pop = []
        for _ in range(self._lam):
            z = self._rng.standard_normal(self._flat_dim)
            self._pop.append(self._mean + self._sigma * z)
        return [self._template.with_flat(x) for x in self._pop]

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

        elite_xs   = np.stack([self._pop[order[i]] for i in range(self._mu)])
        self._mean = np.einsum("i,ij->j", self._recomb_w, elite_xs)

        n_success    = sum(1 for r in rewards if r > prev_best)
        success_rate = n_success / self._lam
        self._sigma  = float(np.clip(
            self._sigma * (1.2 if success_rate > 0.2 else 0.85),
            1e-6, 1e2,
        ))

        return improved

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self._champion is None:
            raise RuntimeError(
                "SC2 LSTMEvolutionPolicy: no champion yet — call sample_population() "
                "and update_distribution() for at least one generation first."
            )
        return self._champion(obs)

    def on_episode_start(self) -> None:
        if self._champion is not None:
            self._champion.on_episode_start()

    def on_episode_end(self) -> None:
        if self._champion is not None:
            self._champion.on_episode_end()

    def to_cfg(self) -> dict:
        return {
            "policy_type":    "lstm",
            "hidden_size":    self._template._hidden_size,
            "population_size": self._lam,
            "sigma":          float(self._sigma),
            "obs_dim":        self._obs_spec.dim,
            "champion_reward": float(self._champion_reward),
        }

    def save(self, path: str) -> None:
        if self._champion is not None:
            self._champion.save(path)

    def save_trainer_state(self, path: str) -> None:
        np.savez(
            path,
            mean     = self._mean,
            sigma    = np.float64(self._sigma),
            flat_dim = np.int64(self._flat_dim),
        )

    def load_trainer_state(self, path: str) -> None:
        with np.load(path) as data:
            saved_flat_dim = int(data["flat_dim"])
            if saved_flat_dim != self._flat_dim:
                raise ValueError(
                    f"SC2 LSTMEvolutionPolicy: trainer state flat_dim mismatch — "
                    f"saved={saved_flat_dim}, current={self._flat_dim}. "
                    f"Use --re-initialize to restart from scratch."
                )
            self._mean  = data["mean"].astype(np.float64)
            self._sigma = float(data["sigma"])
        logger.info("[SC2 LSTMEvolutionPolicy] trainer state loaded from %s (sigma=%.4f)",
                    path, self._sigma)
