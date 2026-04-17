"""TMNF-specific policies: NeuralDQNPolicy (DQN) and CMAESPolicy (CMA-ES).

These policies are specific to TMNF because they are hardcoded to:
  - The 9-element discrete action set (_DISCRETE_ACTIONS)
  - The TMNF observation space (BASE_OBS_DIM + n_lidar_rays)
  - The WeightedLinearPolicy weight format (steer/accel/brake heads)

All other policies live in framework/policies.py and are game-agnostic.
"""
from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np
import yaml

from framework.policies import BasePolicy, WeightedLinearPolicy as _FrameworkWLP
from games.tmnf.actions import DISCRETE_ACTIONS as _DISCRETE_ACTIONS, _action_to_idx
from games.tmnf.obs_spec import BASE_OBS_DIM, TMNF_OBS_SPEC, obs_names_with_lidar, obs_scales_with_lidar

logger = logging.getLogger(__name__)

_N_DISCRETE_ACTIONS = len(_DISCRETE_ACTIONS)


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size circular buffer of (obs, action_idx, reward, next_obs, done) tuples."""

    def __init__(self, maxlen: int) -> None:
        self._buf: deque = deque(maxlen=maxlen)

    def push(self, obs: np.ndarray, action_idx: int, reward: float,
             next_obs: np.ndarray, done: bool) -> None:
        self._buf.append((obs.copy(), int(action_idx), float(reward), next_obs.copy(), bool(done)))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        replace = batch_size > len(self._buf)
        idxs  = np.random.choice(len(self._buf), size=batch_size, replace=replace)
        batch = [self._buf[i] for i in idxs]
        obs_b  = np.stack([t[0] for t in batch]).astype(np.float32)
        act_b  = np.array([t[1] for t in batch], dtype=np.int32)
        rew_b  = np.array([t[2] for t in batch], dtype=np.float32)
        next_b = np.stack([t[3] for t in batch]).astype(np.float32)
        done_b = np.array([t[4] for t in batch], dtype=np.float32)
        return obs_b, act_b, rew_b, next_b, done_b

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# NeuralDQNPolicy
# ---------------------------------------------------------------------------

class NeuralDQNPolicy(BasePolicy):
    """
    DQN over the 9-action discrete action set.

    Online network:  Q(s, a; theta_online)
    Target network:  Q(s, a; theta_target)  -- synced every target_update_freq gradient steps
    Replay buffer:   circular buffer of (s, a_idx, r, s', done)

    Architecture: obs -> Linear -> ReLU -> ... -> Linear(9)
    Pure numpy with Adam optimiser -- no external ML framework required.
    Epsilon decays linearly from epsilon_start -> epsilon_end over epsilon_decay_steps steps.
    """

    def __init__(
        self,
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
        n_lidar_rays: int = 0,
        seed: int | None = None,
    ) -> None:
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
        self._n_lidar_rays = n_lidar_rays
        self._obs_dim      = BASE_OBS_DIM + n_lidar_rays
        self._scales       = obs_scales_with_lidar(n_lidar_rays)
        self._seed         = seed

        self._replay      = ReplayBuffer(replay_buffer_size)
        self._total_steps = 0   # transitions pushed (drives epsilon schedule)
        self._grad_steps  = 0   # gradient updates (drives target sync)

        self._online = self._build_net()
        self._target = self._build_net()
        self._sync_target()

        # Adam first/second moments — one entry per layer
        self._m_w = [np.zeros_like(w) for w in self._online["weights"]]
        self._m_b = [np.zeros_like(b) for b in self._online["biases"]]
        self._v_w = [np.zeros_like(w) for w in self._online["weights"]]
        self._v_b = [np.zeros_like(b) for b in self._online["biases"]]
        self._adam_t = 0

    @classmethod
    def from_cfg(cls, cfg: dict, n_lidar_rays: int = 0) -> NeuralDQNPolicy:
        obj = cls(
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
            n_lidar_rays        = n_lidar_rays,
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
                    f"NeuralDQNPolicy.from_cfg: weight shape mismatch -- "
                    f"first layer expects input dim {loaded_w[0].shape[1]} "
                    f"but obs_dim is {obj._obs_dim} (n_lidar_rays={n_lidar_rays})"
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
            logger.info("[NeuralDQNPolicy] loaded weights from cfg (eps=%.4f, steps=%d)",
                        obj._eps, obj._total_steps)
        return obj

    def _build_net(self) -> dict:
        """He-initialised MLP: weights and biases as lists of float32 arrays."""
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
        """Forward pass; returns (q_values, layer_inputs, pre_relu_activations)."""
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

    def _gradient_step(self, obs_b, act_b, rew_b, next_b, done_b) -> None:
        """One Adam step on the online network using a sampled minibatch."""
        obs_norm  = obs_b  / self._scales
        next_norm = next_b / self._scales
        B = len(act_b)

        q_next   = self._q_values(self._target, next_norm)
        targets  = rew_b + self._gamma * np.max(q_next, axis=1) * (1.0 - done_b)

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
        t       = self._adam_t
        b1, b2  = 0.9, 0.999
        eps_a   = 1e-8
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
            logger.debug("[NeuralDQNPolicy] target synced at grad_step %d", self._grad_steps)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if np.random.random() < self._eps:
            return _DISCRETE_ACTIONS[np.random.randint(_N_DISCRETE_ACTIONS)].copy()
        obs_norm = (obs / self._scales).astype(np.float32)
        q        = self._q_values(self._online, obs_norm)
        return _DISCRETE_ACTIONS[int(np.argmax(q))].copy()

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
        pass  # epsilon decays per step, not per episode

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
            "n_lidar_rays":        self._n_lidar_rays,
            "epsilon":             float(self._eps),
            "total_steps":         self._total_steps,
            "grad_steps":          self._grad_steps,
            "online_weights":      [w.tolist() for w in self._online["weights"]],
            "online_biases":       [b.tolist() for b in self._online["biases"]],
            "target_weights":      [w.tolist() for w in self._target["weights"]],
            "target_biases":       [b.tolist() for b in self._target["biases"]],
        }


# ---------------------------------------------------------------------------
# CMAESPolicy
# ---------------------------------------------------------------------------

class CMAESPolicy(BasePolicy):
    """
    CMA-ES over the flat weight vector of a WeightedLinearPolicy.
    Uses the (mu/mu_w, lambda)-CMA-ES algorithm (Hansen 2016).

    Each generation:
      1. sample_population() -- draw lambda offspring from N(mean, sigma^2 * C)
      2. Evaluate each offspring for one episode
      3. update_distribution(rewards) -- update mean, sigma, C, and evolution paths

    Inference always uses the champion (best individual seen so far).
    save() writes the champion in WeightedLinearPolicy YAML format so
    existing analytics (weight heatmaps, etc.) work without changes.
    """

    def __init__(
        self,
        population_size: int = 20,
        initial_sigma: float = 0.3,
        n_lidar_rays: int = 0,
        seed: int | None = None,
    ) -> None:
        self._lam          = population_size
        self._n_lidar_rays = n_lidar_rays
        n                  = (BASE_OBS_DIM + n_lidar_rays) * 3   # steer + accel + brake heads
        self._n            = n

        # Recombination weights (elite half, log-based, normalised)
        mu            = self._lam // 2
        self._mu      = mu
        raw_w         = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)],
                                 dtype=np.float64)
        self._weights = raw_w / raw_w.sum()
        self._mu_eff  = 1.0 / float(np.sum(self._weights ** 2))

        # Step-size adaptation constants (Hansen 2016, S3)
        self._cs   = (self._mu_eff + 2) / (n + self._mu_eff + 5)
        self._ds   = (1 + 2 * max(0.0, float(np.sqrt((self._mu_eff - 1) / (n + 1))) - 1)
                      + self._cs)
        self._chin = float(np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2)))

        # Covariance adaptation constants (Hansen 2016, S3)
        self._cc  = (4 + self._mu_eff / n) / (n + 4 + 2 * self._mu_eff / n)
        self._c1  = 2.0 / ((n + 1.3) ** 2 + self._mu_eff)
        self._cmu = min(
            1.0 - self._c1,
            2.0 * (self._mu_eff - 2 + 1.0 / self._mu_eff) / ((n + 2) ** 2 + self._mu_eff),
        )

        self._rng = np.random.default_rng(seed)

        # Distribution state (float64 for numerical stability)
        self._mean      = self._rng.standard_normal(n).astype(np.float64)
        self._sigma     = float(initial_sigma)
        self._ps        = np.zeros(n, dtype=np.float64)   # step-size evolution path
        self._pc        = np.zeros(n, dtype=np.float64)   # covariance evolution path
        self._C         = np.eye(n, dtype=np.float64)     # covariance matrix
        self._B         = np.eye(n, dtype=np.float64)     # eigenvectors of C
        self._D         = np.ones(n, dtype=np.float64)    # sqrt(eigenvalues) of C
        self._invsqrtC  = np.eye(n, dtype=np.float64)     # C^{-1/2}
        self._eigengen  = 0
        self._gen       = 0

        self._pop_xs: list[np.ndarray] = []
        self._pop_ys: list[np.ndarray] = []

        self._champion: _FrameworkWLP | None = None  # may be TMNF WLP subclass at runtime
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
        """Initialise the search mean at zero; CMA-ES adapts scale via sigma."""
        self._mean = np.zeros(self._n, dtype=np.float64)
        logger.info("[CMAESPolicy] initialised with zero mean, sigma=%.3f", self._sigma)

    def initialize_from_champion(self, champion) -> None:
        """Seed the search mean from an existing champion's flat weight vector."""
        self._champion = champion

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
        self._mean = champion.to_flat().astype(np.float64)
        logger.info(
            "[CMAESPolicy] seeded mean from champion%s",
            "" if seeded_reward is None else f" (baseline reward={self._champion_reward:.6f})",
        )

    def _flat_to_policy(self, flat: np.ndarray):
        """Build a WeightedLinearPolicy from a flat [steer|accel|brake] weight vector.

        Uses a lazy import of the TMNF WeightedLinearPolicy (root policies shim) so
        that isinstance checks in tests pass.  The lazy import avoids a circular
        module-level dependency; by call time both modules are already loaded.
        """
        # Lazy import: avoids circular import at module level while returning
        # a TMNF-flavoured WLP so isinstance(result, policies.WeightedLinearPolicy) is True.
        from policies import WeightedLinearPolicy as _TMNF_WLP  # type: ignore[import]

        names = obs_names_with_lidar(self._n_lidar_rays)
        obs_n = len(names)
        cfg = {
            "steer_weights": {names[i]: float(flat[i])             for i in range(obs_n)},
            "accel_weights": {names[i]: float(flat[obs_n + i])     for i in range(obs_n)},
            "brake_weights": {names[i]: float(flat[2 * obs_n + i]) for i in range(obs_n)},
        }
        return _TMNF_WLP.from_cfg(cfg, self._n_lidar_rays)

    def _update_eigen(self) -> None:
        """Eigendecompose C and refresh B, D, invsqrtC."""
        self._C = np.triu(self._C) + np.triu(self._C, 1).T
        eigvals, self._B = np.linalg.eigh(self._C)
        eigvals          = np.maximum(eigvals, 1e-20)
        self._D          = np.sqrt(eigvals)
        self._invsqrtC   = self._B @ np.diag(1.0 / self._D) @ self._B.T
        self._eigengen   = self._gen

    def sample_population(self) -> list[_FrameworkWLP]:
        """Sample lambda offspring from N(mean, sigma^2 * C)."""
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
        """Apply (mu/mu_w, lambda)-CMA-ES update. Returns True if champion improved."""
        if len(rewards) != self._lam:
            raise ValueError(f"Expected {self._lam} rewards, got {len(rewards)}")
        if len(self._pop_xs) != self._lam or len(self._pop_ys) != self._lam:
            raise RuntimeError(
                "update_distribution() called before a matching sample_population(). "
                f"Expected {self._lam} samples in _pop_xs/_pop_ys, "
                f"got {len(self._pop_xs)}/{len(self._pop_ys)}. "
                "Call sample_population() first."
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
                "CMAESPolicy: no champion yet -- call sample_population() "
                "and update_distribution() for at least one generation first."
            )
        return self._champion(obs)

    def to_cfg(self) -> dict:
        return {
            "policy_type":     "cmaes",
            "population_size": self._lam,
            "sigma":           self._sigma,
            "n_lidar_rays":    self._n_lidar_rays,
            "champion_reward": float(self._champion_reward),
        }

    def save(self, path: str) -> None:
        """Save champion in WeightedLinearPolicy YAML format for analytics compatibility."""
        if self._champion is not None:
            self._champion.save(path)


# ---------------------------------------------------------------------------
# REINFORCEPolicy
# ---------------------------------------------------------------------------

class REINFORCEPolicy(BasePolicy):
    """
    REINFORCE (Monte Carlo Policy Gradient) with optional entropy regularisation.

    Action head: softmax over the 9 discrete TMNF actions.
    Each episode accumulates (log_prob, reward) pairs; gradient update fires
    on episode end using discounted, normalised returns.

    Training loop dispatch: "q_learning" (update() per step, on_episode_end() per episode).
    """

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
        baseline: str = "running_mean",
        n_lidar_rays: int = 0,
        seed: int | None = None,
    ) -> None:
        self._hidden       = list(hidden_sizes or [64, 64])
        self._lr           = float(learning_rate)
        self._gamma        = float(gamma)
        self._entropy_coeff = float(entropy_coeff)
        self._baseline_type = baseline
        self._n_lidar_rays = n_lidar_rays
        self._obs_dim      = BASE_OBS_DIM + n_lidar_rays
        self._scales       = obs_scales_with_lidar(n_lidar_rays)

        self._weights, self._biases = self._build_net(seed)

        # Episode buffers (aligned: one entry per non-warmup step)
        self._ep_grads: list[tuple]   = []  # (layer_inputs, pre_relu, probs, action_idx)
        self._ep_rewards: list[float] = []

        # Running-mean baseline (EMA of total episode returns)
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
        """Forward pass; caches (layer_inputs, pre_relu) for backprop."""
        x: np.ndarray         = obs_norm.astype(np.float32)
        layer_inputs: list    = []
        pre_relu: list        = []
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
        obs_norm               = obs / self._scales
        probs, l_in, pre_r    = self._forward(obs_norm)
        action_idx             = int(np.random.choice(_N_DISCRETE_ACTIONS, p=probs))
        self._ep_grads.append((l_in, pre_r, probs.copy(), action_idx))
        return _DISCRETE_ACTIONS[action_idx].copy()

    def update(self, obs: np.ndarray, action: np.ndarray | int, reward: float,
               next_obs: np.ndarray, done: bool) -> None:
        self._ep_rewards.append(float(reward))

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

        # Capture baseline before updating so this episode's returns don't bias
        # the advantages computed below (avoids action-dependent baseline).
        baseline_for_advantages = self._baseline_val

        # Baseline update (EMA of total episode return)
        if self._baseline_type == "running_mean":
            self._baseline_val = ((1 - self._baseline_alpha) * self._baseline_val
                                  + self._baseline_alpha * float(G[0]))

        # Normalise returns: scale by std when there is within-episode variance;
        # otherwise fall back to baseline-centred raw returns so single-step
        # episodes still produce a non-zero gradient.
        G_std = float(G.std())
        if G_std > 1e-6:
            G_norm = (G - G.mean()) / (G_std + 1e-8)
        else:
            G_norm = G - baseline_for_advantages

        # Accumulate gradients
        dW = [np.zeros_like(w, dtype=np.float64) for w in self._weights]
        dB = [np.zeros_like(b, dtype=np.float64) for b in self._biases]

        for t in range(T):
            l_in, pre_r, probs, a_idx = self._ep_grads[t]
            advantage = float(G_norm[t])

            # Output-layer gradient: ∂J/∂z = advantage * (one_hot(a) - probs) + entropy term
            delta                  = -probs.copy().astype(np.float64)
            delta[a_idx]          += 1.0
            delta                 *= advantage

            if self._entropy_coeff > 0.0:
                # ∂H/∂z_j = -p_j * (log(p_j) + H)  [gradient of entropy w.r.t. logits]
                log_p        = np.log(probs.astype(np.float64) + 1e-8)
                H            = -float(np.dot(probs, log_p))
                entropy_grad = -probs.astype(np.float64) * (log_p + H)
                delta       += self._entropy_coeff * entropy_grad

            # Backprop through MLP (gradient ascent)
            g = delta
            for i in range(len(self._weights) - 1, -1, -1):
                dW[i] += np.outer(g, l_in[i])
                dB[i] += g
                if i > 0:
                    g = self._weights[i].T @ g * (pre_r[i - 1] > 0)

        # Parameter update (gradient ascent: θ += lr * grad / T)
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
            "n_lidar_rays":   self._n_lidar_rays,
            "baseline_value": float(self._baseline_val),
            "weights":        [w.tolist() for w in self._weights],
            "biases":         [b.tolist() for b in self._biases],
        }

    @classmethod
    def from_cfg(cls, cfg: dict, n_lidar_rays: int = 0) -> "REINFORCEPolicy":
        obj = cls(
            hidden_sizes  = cfg.get("hidden_sizes",  [64, 64]),
            learning_rate = cfg.get("learning_rate", 0.001),
            gamma         = cfg.get("gamma",         0.99),
            entropy_coeff = cfg.get("entropy_coeff", 0.01),
            baseline      = cfg.get("baseline",      "running_mean"),
            n_lidar_rays  = n_lidar_rays,
        )
        if "weights" in cfg:
            obj._weights = [np.array(w, dtype=np.float32) for w in cfg["weights"]]
            obj._biases  = [np.array(b, dtype=np.float32) for b in cfg["biases"]]
        if "baseline_value" in cfg:
            obj._baseline_val = float(cfg["baseline_value"])
        return obj


# ---------------------------------------------------------------------------
# LSTMPolicy
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


class LSTMPolicy(BasePolicy):
    """
    Single-layer LSTM policy (pure numpy).

    Hidden state (h, c) persists across steps within an episode.
    Trained via an outer evolutionary optimiser (LSTMEvolutionPolicy).
    on_episode_end() resets (h, c) to zeros.

    Output heads:
      steer = tanh(W_steer · h)           → [-1, 1]
      accel = sigmoid(W_accel · h) > 0.5  → {0, 1}
      brake = sigmoid(W_brake · h) > 0.5  → {0, 1}
    """

    def __init__(
        self,
        hidden_size: int = 32,
        n_lidar_rays: int = 0,
        seed: int | None = None,
    ) -> None:
        self._hidden_size  = hidden_size
        self._n_lidar_rays = n_lidar_rays
        self._obs_dim      = BASE_OBS_DIM + n_lidar_rays
        self._scales       = obs_scales_with_lidar(n_lidar_rays)

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

        self._W_steer = rng.standard_normal(h).astype(np.float32) * np.sqrt(2.0 / h)
        self._W_accel = rng.standard_normal(h).astype(np.float32) * np.sqrt(2.0 / h)
        self._W_brake = rng.standard_normal(h).astype(np.float32) * np.sqrt(2.0 / h)

        self._h = np.zeros(h, dtype=np.float32)
        self._c = np.zeros(h, dtype=np.float32)

    # ------------------------------------------------------------------
    # Flat parameter interface (used by LSTMEvolutionPolicy)
    # ------------------------------------------------------------------

    @property
    def flat_dim(self) -> int:
        h    = self._hidden_size
        c_in = h + self._obs_dim
        return 4 * (h * c_in + h) + 3 * h

    def to_flat(self) -> np.ndarray:
        return np.concatenate([
            self._W_f.ravel(), self._b_f,
            self._W_i.ravel(), self._b_i,
            self._W_g.ravel(), self._b_g,
            self._W_o.ravel(), self._b_o,
            self._W_steer,
            self._W_accel,
            self._W_brake,
        ]).astype(np.float32)

    def with_flat(self, flat: np.ndarray) -> "LSTMPolicy":
        """Return a new LSTMPolicy whose weights come from a flat parameter vector."""
        flat = np.asarray(flat, dtype=np.float32)
        if flat.shape[0] != self.flat_dim:
            raise ValueError(
                f"LSTMPolicy.with_flat: expected flat vector of size {self.flat_dim}, "
                f"got {flat.shape[0]}"
            )

        obj = object.__new__(LSTMPolicy)
        obj._hidden_size  = self._hidden_size
        obj._n_lidar_rays = self._n_lidar_rays
        obj._obs_dim      = self._obs_dim
        obj._scales       = self._scales

        h    = self._hidden_size
        c_in = h + self._obs_dim

        off = 0
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
        obj._W_steer = _take((h,))
        obj._W_accel = _take((h,))
        obj._W_brake = _take((h,))
        obj._h      = np.zeros(h, dtype=np.float32)
        obj._c      = np.zeros(h, dtype=np.float32)
        return obj

    def mutated(self, scale: float, **_) -> "LSTMPolicy":
        """Return a new LSTMPolicy with Gaussian noise applied to all parameters."""
        flat  = self.to_flat()
        noise = np.random.default_rng().standard_normal(len(flat)).astype(np.float32)
        return self.with_flat(flat + scale * noise)

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x  = (obs / self._scales).astype(np.float32)
        hx = np.concatenate([self._h, x])

        f = _sigmoid(self._W_f @ hx + self._b_f)
        i = _sigmoid(self._W_i @ hx + self._b_i)
        g = np.tanh(self._W_g   @ hx + self._b_g)
        o = _sigmoid(self._W_o  @ hx + self._b_o)

        self._c = f * self._c + i * g
        self._h = o * np.tanh(self._c)

        steer = float(np.tanh(np.dot(self._W_steer, self._h)))
        accel = float(_sigmoid(np.dot(self._W_accel, self._h)) > 0.5)
        brake = float(_sigmoid(np.dot(self._W_brake, self._h)) > 0.5)
        return np.array([steer, accel, brake], dtype=np.float32)

    def update(self, obs: np.ndarray, action: np.ndarray | int, reward: float,
               next_obs: np.ndarray, done: bool) -> None:
        pass  # no online update; training via outer evolutionary optimiser

    def on_episode_end(self) -> None:
        self._h = np.zeros(self._hidden_size, dtype=np.float32)
        self._c = np.zeros(self._hidden_size, dtype=np.float32)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        return {
            "policy_type": "lstm",
            "hidden_size": self._hidden_size,
            "n_lidar_rays": self._n_lidar_rays,
            "obs_dim":     self._obs_dim,
            "W_f": self._W_f.tolist(), "b_f": self._b_f.tolist(),
            "W_i": self._W_i.tolist(), "b_i": self._b_i.tolist(),
            "W_g": self._W_g.tolist(), "b_g": self._b_g.tolist(),
            "W_o": self._W_o.tolist(), "b_o": self._b_o.tolist(),
            "W_steer": self._W_steer.tolist(),
            "W_accel": self._W_accel.tolist(),
            "W_brake": self._W_brake.tolist(),
        }

    @classmethod
    def from_cfg(cls, cfg: dict) -> "LSTMPolicy":
        obj = object.__new__(cls)
        obj._hidden_size  = int(cfg["hidden_size"])
        obj._n_lidar_rays = int(cfg.get("n_lidar_rays", 0))
        obj._obs_dim      = int(cfg.get("obs_dim", BASE_OBS_DIM + obj._n_lidar_rays))
        obj._scales       = obs_scales_with_lidar(obj._n_lidar_rays)
        obj._W_f    = np.array(cfg["W_f"],    dtype=np.float32)
        obj._b_f    = np.array(cfg["b_f"],    dtype=np.float32)
        obj._W_i    = np.array(cfg["W_i"],    dtype=np.float32)
        obj._b_i    = np.array(cfg["b_i"],    dtype=np.float32)
        obj._W_g    = np.array(cfg["W_g"],    dtype=np.float32)
        obj._b_g    = np.array(cfg["b_g"],    dtype=np.float32)
        obj._W_o    = np.array(cfg["W_o"],    dtype=np.float32)
        obj._b_o    = np.array(cfg["b_o"],    dtype=np.float32)
        obj._W_steer = np.array(cfg["W_steer"], dtype=np.float32)
        obj._W_accel = np.array(cfg["W_accel"], dtype=np.float32)
        obj._W_brake = np.array(cfg["W_brake"], dtype=np.float32)
        h = obj._hidden_size
        obj._h = np.zeros(h, dtype=np.float32)
        obj._c = np.zeros(h, dtype=np.float32)
        return obj


# ---------------------------------------------------------------------------
# LSTMEvolutionPolicy
# ---------------------------------------------------------------------------

class LSTMEvolutionPolicy(BasePolicy):
    """
    (μ/μ_w, λ)-ES outer optimiser wrapping LSTMPolicy as the inner individual.

    Uses the _greedy_loop_cmaes interface: sample_population() / update_distribution().
    Maintains an isotropic Gaussian search distribution (no full covariance matrix —
    infeasible for the ~7 K-dimensional LSTM parameter space).
    Step size is adapted via the 1/5 success rule.

    Inference delegates to the champion LSTMPolicy.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        population_size: int = 20,
        initial_sigma: float = 0.05,
        n_lidar_rays: int = 0,
        seed: int | None = None,
    ) -> None:
        self._lam          = int(population_size)
        self._sigma        = float(initial_sigma)
        self._n_lidar_rays = n_lidar_rays
        self._rng          = np.random.default_rng(seed)

        self._template = LSTMPolicy(hidden_size=hidden_size, n_lidar_rays=n_lidar_rays)
        self._flat_dim = self._template.flat_dim
        self._mean     = self._template.to_flat().astype(np.float64)

        # Weighted recombination (log-based, top-mu elites)
        mu         = self._lam // 2
        self._mu   = mu
        raw_w      = np.array([np.log(mu + 0.5) - np.log(i + 1)
                               for i in range(mu)], dtype=np.float64)
        self._recomb_w = raw_w / raw_w.sum()

        self._pop: list[np.ndarray] = []
        self._champion: LSTMPolicy | None        = None
        self._champion_reward: float             = float("-inf")

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
        champion_flat_dim = champion.flat_dim
        expected_hidden_size = getattr(self._template, "_hidden_size", None)
        champion_hidden_size = getattr(champion, "_hidden_size", None)
        expected_obs_dim = getattr(self._template, "obs_dim", None)
        champion_obs_dim = getattr(champion, "obs_dim", None)
        expected_lidar_rays = getattr(self, "_n_lidar_rays", None)
        champion_lidar_rays = getattr(champion, "_n_lidar_rays", None)

        mismatch_reasons: list[str] = []
        if champion_flat_dim != self._flat_dim:
            mismatch_reasons.append(
                f"flat_dim mismatch (expected {self._flat_dim}, got {champion_flat_dim})"
            )
        if (
            expected_hidden_size is not None
            and champion_hidden_size is not None
            and champion_hidden_size != expected_hidden_size
        ):
            mismatch_reasons.append(
                "hidden_size mismatch "
                f"(expected {expected_hidden_size}, got {champion_hidden_size})"
            )
        if (
            expected_obs_dim is not None
            and champion_obs_dim is not None
            and champion_obs_dim != expected_obs_dim
        ):
            mismatch_reasons.append(
                f"obs_dim mismatch (expected {expected_obs_dim}, got {champion_obs_dim})"
            )
        if (
            expected_lidar_rays is not None
            and champion_lidar_rays is not None
            and champion_lidar_rays != expected_lidar_rays
        ):
            mismatch_reasons.append(
                "n_lidar_rays mismatch "
                f"(expected {expected_lidar_rays}, got {champion_lidar_rays})"
            )

        if mismatch_reasons:
            raise ValueError(
                "Cannot initialize LSTMEvolutionPolicy from an incompatible champion: "
                + "; ".join(mismatch_reasons)
            )
        self._champion = champion
        self._mean     = champion.to_flat().astype(np.float64)
        logger.info("[LSTMEvolutionPolicy] seeded mean from champion")

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
            raise RuntimeError(
                "update_distribution() called before a matching sample_population(). "
                f"Expected {self._lam} samples in _pop, got {len(self._pop)}. "
                "Call sample_population() first."
            )

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

        # Weighted mean recombination (top-mu elites)
        elite_xs   = np.stack([self._pop[order[i]] for i in range(self._mu)])
        self._mean = np.einsum("i,ij->j", self._recomb_w, elite_xs)

        # 1/5 success rule for isotropic step-size adaptation
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
                "LSTMEvolutionPolicy: no champion yet — call sample_population() "
                "and update_distribution() for at least one generation first."
            )
        return self._champion(obs)

    def on_episode_end(self) -> None:
        if self._champion is not None:
            self._champion.on_episode_end()

    def to_cfg(self) -> dict:
        return {
            "policy_type":    "lstm",
            "hidden_size":    self._template._hidden_size,
            "population_size": self._lam,
            "sigma":          float(self._sigma),
            "n_lidar_rays":   self._n_lidar_rays,
            "champion_reward": float(self._champion_reward),
        }

    def save(self, path: str) -> None:
        if self._champion is not None:
            self._champion.save(path)
