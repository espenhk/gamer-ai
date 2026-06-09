"""AlphaZero-style model-based MCTS policy (pure numpy).

This is a *real* Monte-Carlo Tree Search agent — unlike the ``ucb_q`` policy
(an online UCB1 Q-learner with no tree).  At every decision it builds a search
tree, expanding nodes by **cloning the environment and stepping it forward**,
guided by a policy/value network (PUCT selection, à la AlphaGo Zero), and acts
on the resulting visit-count distribution.  The network is trained from
self-play with the AlphaZero loss (value regression + policy cross-entropy).

Because expansion clones and steps the environment, this policy requires a
**deterministic, cloneable simulator**.  None of gamer-ai's current games expose
that (they bind to live processes / sockets), so :meth:`compatible_with` gates
the policy off them and it is exercised on a cloneable toy env in the tests.
Adapting it to a real game means giving that game's env a cheap ``clone()``
(or making it ``copy.deepcopy``-able) and removing the game from the denylist.

Single-agent MDP adaptation: rather than the two-player ±1 outcome, edges carry
the environment reward and the value head predicts the (discounted)
return-to-go, so backups combine accumulated edge rewards with the leaf value.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, ClassVar

import numpy as np

from framework.obs_spec import ObsSpec
from framework.policies import BasePolicy, register_policy

logger = logging.getLogger(__name__)

# Games whose envs bind to live processes/sockets and cannot be cloned for tree
# search.  AlphaZero MCTS is gated off these (fail fast before connecting).
# These names match the CLI ``--game`` choices in main.py — note that the
# Assetto Corsa game's name is ``"assetto"`` (not the directory name
# ``assetto_corsa``).  MineRL uses a JVM subprocess; Atari/ALE does not expose
# a validated clone path.  The ``test_gated_off_non_cloneable_games`` test
# iterates this whole set, so any drift between names here and the actual
# ``game_name`` strings is caught.
_NON_CLONEABLE_GAMES: frozenset[str] = frozenset(
    {
        "tmnf",
        "sc2",
        "torcs",
        "beamng",
        "car_racing",
        "rocket_league",
        "iracing",
        "assetto",
        "atari",
        "minerl",
    }
)


# --------------------------------------------------------------------------- #
# Policy / value network (numpy MLP, two heads)                                #
# --------------------------------------------------------------------------- #


class _PVNet:
    """Small MLP with a softmax policy head and a scalar value head."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: tuple[int, ...] = (64, 64),
        learning_rate: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = tuple(hidden)
        self.lr = float(learning_rate)
        rng = np.random.default_rng(seed)
        dims = [obs_dim, *self.hidden]
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i in range(len(dims) - 1):
            self.W.append((rng.standard_normal((dims[i + 1], dims[i])) * np.sqrt(2.0 / dims[i])).astype(np.float32))
            self.b.append(np.zeros(dims[i + 1], dtype=np.float32))
        h = dims[-1]
        self.Wp = (rng.standard_normal((n_actions, h)) * np.sqrt(1.0 / h)).astype(np.float32)
        self.bp = np.zeros(n_actions, dtype=np.float32)
        self.Wv = (rng.standard_normal((1, h)) * np.sqrt(1.0 / h)).astype(np.float32)
        self.bv = np.zeros(1, dtype=np.float32)
        self._init_adam()

    def _params(self) -> list[np.ndarray]:
        return [*self.W, *self.b, self.Wp, self.bp, self.Wv, self.bv]

    def _init_adam(self) -> None:
        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]
        self._t = 0

    def _forward(self, x: np.ndarray):
        """x: (B, obs_dim). Returns (probs, value, cache)."""
        acts = [x]
        pre = []
        a = x
        for W, b in zip(self.W, self.b):
            z = a @ W.T + b
            pre.append(z)
            a = np.maximum(z, 0.0)
            acts.append(a)
        logits = a @ self.Wp.T + self.bp
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        value = (a @ self.Wv.T + self.bv)[:, 0]
        return probs, value, (acts, pre)

    def predict(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        probs, value, _ = self._forward(obs[np.newaxis, :].astype(np.float32))
        return probs[0], float(value[0])

    def train_step(self, obs_b: np.ndarray, pi_b: np.ndarray, z_b: np.ndarray, value_coef: float = 1.0) -> float:
        """One Adam step on the AlphaZero loss; returns the scalar loss."""
        B = len(obs_b)
        probs, value, (acts, pre) = self._forward(obs_b)
        eps = 1e-8
        policy_loss = -np.sum(pi_b * np.log(probs + eps)) / B
        value_loss = np.mean((value - z_b) ** 2)
        loss = policy_loss + value_coef * value_loss

        # Head gradients.
        dlogits = (probs - pi_b) / B  # (B, n_actions)
        dvalue = (2.0 * value_coef * (value - z_b) / B)[:, np.newaxis]  # (B, 1)
        a_last = acts[-1]
        gWp = dlogits.T @ a_last
        gbp = dlogits.sum(axis=0)
        gWv = dvalue.T @ a_last
        gbv = dvalue.sum(axis=0)

        # Backprop into the trunk (sum of both head contributions).
        da = dlogits @ self.Wp + dvalue @ self.Wv
        gW: list[np.ndarray] = [None] * len(self.W)  # type: ignore[list-item]
        gb: list[np.ndarray] = [None] * len(self.b)  # type: ignore[list-item]
        for i in range(len(self.W) - 1, -1, -1):
            dz = da * (pre[i] > 0)
            gW[i] = dz.T @ acts[i]
            gb[i] = dz.sum(axis=0)
            da = dz @ self.W[i]

        grads = [*gW, *gb, gWp, gbp, gWv, gbv]
        self._adam_apply(grads)
        return float(loss)

    def _adam_apply(self, grads: list[np.ndarray]) -> None:
        params = self._params()
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * g * g
            mhat = self._m[i] / (1 - b1**self._t)
            vhat = self._v[i] / (1 - b2**self._t)
            p -= self.lr * mhat / (np.sqrt(vhat) + eps)

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "W": [w.tolist() for w in self.W],
            "b": [b.tolist() for b in self.b],
            "Wp": self.Wp.tolist(),
            "bp": self.bp.tolist(),
            "Wv": self.Wv.tolist(),
            "bv": self.bv.tolist(),
        }

    def load_dict(self, d: dict) -> None:
        self.W = [np.array(w, dtype=np.float32) for w in d["W"]]
        self.b = [np.array(b, dtype=np.float32) for b in d["b"]]
        self.Wp = np.array(d["Wp"], dtype=np.float32)
        self.bp = np.array(d["bp"], dtype=np.float32)
        self.Wv = np.array(d["Wv"], dtype=np.float32)
        self.bv = np.array(d["bv"], dtype=np.float32)
        self._init_adam()


# --------------------------------------------------------------------------- #
# MCTS                                                                         #
# --------------------------------------------------------------------------- #


class _Node:
    __slots__ = ("env", "obs", "done", "reward_in", "P", "N", "W", "Q", "children", "expanded")

    def __init__(self, env: Any, obs: np.ndarray, done: bool, reward_in: float) -> None:
        self.env = env
        self.obs = obs
        self.done = done
        self.reward_in = reward_in
        self.expanded = False
        self.P: np.ndarray | None = None
        self.N: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.children: dict[int, _Node] = {}


def _clone_env(env: Any) -> Any:
    if hasattr(env, "clone"):
        return env.clone()
    return copy.deepcopy(env)


class _MCTS:
    def __init__(
        self,
        net: _PVNet,
        actions: np.ndarray,
        scales: np.ndarray,
        *,
        n_simulations: int,
        c_puct: float,
        gamma: float,
        dirichlet_alpha: float,
        dirichlet_frac: float,
        rng: np.random.Generator,
    ) -> None:
        self.net = net
        self.actions = actions
        self.scales = scales
        self.n_sim = int(n_simulations)
        self.c_puct = float(c_puct)
        self.gamma = float(gamma)
        self.dir_alpha = float(dirichlet_alpha)
        self.dir_frac = float(dirichlet_frac)
        self.rng = rng

    def _evaluate(self, node: _Node) -> float:
        probs, value = self.net.predict(node.obs / self.scales)
        n = len(self.actions)
        node.P = probs
        node.N = np.zeros(n, dtype=np.float64)
        node.W = np.zeros(n, dtype=np.float64)
        node.Q = np.zeros(n, dtype=np.float64)
        node.expanded = True
        return value

    def _select(self, node: _Node) -> int:
        total_n = node.N.sum()
        u = self.c_puct * node.P * np.sqrt(total_n + 1.0) / (1.0 + node.N)
        return int(np.argmax(node.Q + u))

    def _step_child(self, node: _Node, a: int) -> _Node:
        child_env = _clone_env(node.env)
        obs, reward, terminated, truncated, _ = child_env.step(self.actions[a])
        return _Node(child_env, np.asarray(obs, dtype=np.float32), bool(terminated or truncated), float(reward))

    def run(self, root: _Node, *, add_noise: bool) -> np.ndarray:
        """Run simulations from *root*; return per-action visit counts."""
        if not root.expanded:
            self._evaluate(root)
        if add_noise and self.dir_frac > 0.0:
            noise = self.rng.dirichlet([self.dir_alpha] * len(self.actions)).astype(np.float32)
            root.P = (1 - self.dir_frac) * root.P + self.dir_frac * noise

        for _ in range(self.n_sim):
            self._simulate(root)
        return root.N.copy()

    def _simulate(self, root: _Node) -> None:
        node = root
        path: list[tuple[_Node, int]] = []
        while True:
            if node.done:
                leaf_value = 0.0
                break
            a = self._select(node)
            path.append((node, a))
            if a not in node.children:
                child = self._step_child(node, a)
                node.children[a] = child
                leaf_value = 0.0 if child.done else self._evaluate(child)
                break
            node = node.children[a]

        g = leaf_value
        for n, a in reversed(path):
            child = n.children[a]
            g = child.reward_in + self.gamma * g
            n.N[a] += 1.0
            n.W[a] += g
            n.Q[a] = n.W[a] / n.N[a]


# --------------------------------------------------------------------------- #
# Policy                                                                       #
# --------------------------------------------------------------------------- #


@register_policy
class AlphaZeroMCTSPolicy(BasePolicy):
    """AlphaZero-style policy/value MCTS over a cloneable simulator."""

    POLICY_TYPE = "alphazero_mcts"
    LOOP_TYPE = "alphazero"
    VALID_POLICY_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {
            "n_simulations",
            "c_puct",
            "gamma",
            "hidden_sizes",
            "learning_rate",
            "temperature",
            "dirichlet_alpha",
            "dirichlet_frac",
            "value_loss_coef",
            "train_batch_size",
            "seed",
        }
    )

    def __init__(
        self,
        obs_spec: ObsSpec,
        discrete_actions: np.ndarray,
        *,
        n_simulations: int = 32,
        c_puct: float = 1.5,
        gamma: float = 0.99,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 1e-3,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_frac: float = 0.25,
        value_loss_coef: float = 1.0,
        train_batch_size: int = 32,
        seed: int | None = None,
    ) -> None:
        self._obs_spec = obs_spec
        self._scales = obs_spec.scales
        self._actions = np.asarray(discrete_actions, dtype=np.float32)
        self._n_actions = len(self._actions)
        self.n_simulations = int(n_simulations)
        self.c_puct = float(c_puct)
        self.gamma = float(gamma)
        self._hidden = tuple(hidden_sizes or [64, 64])
        self.learning_rate = float(learning_rate)
        self.temperature = float(temperature)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_frac = float(dirichlet_frac)
        self.value_loss_coef = float(value_loss_coef)
        self.train_batch_size = int(train_batch_size)
        self._seed = seed
        self._net = _PVNet(obs_spec.dim, self._n_actions, self._hidden, learning_rate, seed)

    # -- inference -----------------------------------------------------------
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Greedy action from the policy head (no tree — __call__ has no env)."""
        probs, _ = self._net.predict(np.asarray(obs, dtype=np.float32) / self._scales)
        return self._actions[int(np.argmax(probs))].copy()

    # -- compatibility -------------------------------------------------------
    @classmethod
    def compatible_with(cls, game_name: str) -> tuple[bool, str | None]:
        if game_name in _NON_CLONEABLE_GAMES:
            return False, (
                f"alphazero_mcts needs a deterministic, cloneable simulator to expand the search "
                f"tree, but {game_name!r}'s env binds to a live process/socket and cannot be cloned. "
                f"Give that game's env a clone() method (or make it deepcopy-able) and remove it from "
                f"framework.alphazero._NON_CLONEABLE_GAMES to enable it."
            )
        return True, None

    # -- serialisation -------------------------------------------------------
    def to_cfg(self) -> dict:
        return {
            "policy_type": self.POLICY_TYPE,
            "n_simulations": self.n_simulations,
            "c_puct": self.c_puct,
            "gamma": self.gamma,
            "hidden_sizes": list(self._hidden),
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "dirichlet_alpha": self.dirichlet_alpha,
            "dirichlet_frac": self.dirichlet_frac,
            "value_loss_coef": self.value_loss_coef,
            "train_batch_size": self.train_batch_size,
            "net": self._net.to_dict(),
        }

    @classmethod
    def from_cfg(cls, cfg: dict, obs_spec: ObsSpec, discrete_actions: np.ndarray) -> "AlphaZeroMCTSPolicy":
        obj = cls(
            obs_spec=obs_spec,
            discrete_actions=discrete_actions,
            n_simulations=cfg.get("n_simulations", 32),
            c_puct=cfg.get("c_puct", 1.5),
            gamma=cfg.get("gamma", 0.99),
            hidden_sizes=cfg.get("hidden_sizes", [64, 64]),
            learning_rate=cfg.get("learning_rate", 1e-3),
            temperature=cfg.get("temperature", 1.0),
            dirichlet_alpha=cfg.get("dirichlet_alpha", 0.3),
            dirichlet_frac=cfg.get("dirichlet_frac", 0.25),
            value_loss_coef=cfg.get("value_loss_coef", 1.0),
            train_batch_size=cfg.get("train_batch_size", 32),
        )
        if "net" in cfg and cfg["net"]:
            obj._net.load_dict(cfg["net"])
        return obj

    @classmethod
    def _construct_or_resume(
        cls, *, obs_spec, head_names, discrete_actions, weights_file, policy_params, re_initialize
    ) -> "AlphaZeroMCTSPolicy":
        import yaml

        pp = {k: v for k, v in (policy_params or {}).items() if not k.startswith("_")}
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as f:
                cfg = yaml.safe_load(f) or {}
            if isinstance(cfg, dict) and cfg.get("policy_type") == cls.POLICY_TYPE:
                logger.info("[AlphaZeroMCTSPolicy] resuming from %s", weights_file)
                return cls.from_cfg(cfg, obs_spec, discrete_actions)
        return cls(obs_spec=obs_spec, discrete_actions=discrete_actions, **pp)


# --------------------------------------------------------------------------- #
# Self-play training loop                                                      #
# --------------------------------------------------------------------------- #


def run_alphazero_loop(
    *,
    env,
    policy: AlphaZeroMCTSPolicy,
    n_sims: int,
    weights_file: str,
    training_params: dict,
    patience: int = 0,
    warmup_action: Any = None,
    warmup_steps: int = 0,
    live_monitor: Any = None,
    log_stats_every_n_sims: int = 0,
):
    """Self-play AlphaZero training. ``n_sims`` = number of self-play games."""
    from framework.analytics import GreedySimResult
    from framework.training import GreedyLoopResult

    # Fail fast if the env can't be cloned for tree search.
    try:
        _clone_env(env)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"alphazero_mcts requires a cloneable env (clone() or copy.deepcopy). Cloning failed: {exc}"
        ) from exc

    net = policy._net
    rng = np.random.default_rng(policy._seed)
    mcts = _MCTS(
        net,
        policy._actions,
        policy._scales,
        n_simulations=policy.n_simulations,
        c_puct=policy.c_puct,
        gamma=policy.gamma,
        dirichlet_alpha=policy.dirichlet_alpha,
        dirichlet_frac=policy.dirichlet_frac,
        rng=rng,
    )

    sims: list[GreedySimResult] = []
    best_reward = float("-inf")
    sims_since_improve = 0
    early_stopped = False
    early_stop_sim: int | None = None

    for ep in range(int(n_sims)):
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        traj_obs: list[np.ndarray] = []
        traj_pi: list[np.ndarray] = []
        traj_r: list[float] = []
        ep_reward = 0.0
        steps = 0
        done = False
        while not done:
            root = _Node(_clone_env(env), obs, done=False, reward_in=0.0)
            visits = mcts.run(root, add_noise=True)
            if visits.sum() <= 0:
                pi = np.ones(policy._n_actions) / policy._n_actions
            elif policy.temperature <= 1e-6:
                pi = np.zeros(policy._n_actions)
                pi[int(np.argmax(visits))] = 1.0
            else:
                scaled = visits ** (1.0 / policy.temperature)
                pi = scaled / scaled.sum()
            a = int(rng.choice(policy._n_actions, p=pi))
            traj_obs.append(obs.copy())
            traj_pi.append(pi.astype(np.float32))
            obs, reward, terminated, truncated, _ = env.step(policy._actions[a])
            obs = np.asarray(obs, dtype=np.float32)
            traj_r.append(float(reward))
            ep_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)

        # Discounted return-to-go value targets.
        z = np.zeros(len(traj_r), dtype=np.float32)
        running = 0.0
        for t in range(len(traj_r) - 1, -1, -1):
            running = traj_r[t] + policy.gamma * running
            z[t] = running

        # Train the net on this game's (obs, pi, z) tuples.
        obs_arr = np.stack(traj_obs) / policy._scales
        pi_arr = np.stack(traj_pi)
        idx = rng.permutation(len(obs_arr))
        bs = max(1, policy.train_batch_size)
        for start in range(0, len(idx), bs):
            sl = idx[start : start + bs]
            net.train_step(obs_arr[sl], pi_arr[sl], z[idx][start : start + len(sl)], policy.value_loss_coef)

        improved = ep_reward > best_reward
        if improved:
            best_reward = ep_reward
            sims_since_improve = 0
            policy.save(weights_file)
        else:
            sims_since_improve += 1
        sims.append(
            GreedySimResult(sim=ep, reward=ep_reward, improved=improved, throttle_counts=[0, 0, 0], total_steps=steps)
        )
        logger.info("game %d  r=%+.2f  steps=%d%s", ep, ep_reward, steps, "  *best*" if improved else "")

        if patience > 0 and sims_since_improve >= patience:
            early_stopped = True
            early_stop_sim = ep
            logger.info("[alphazero] early stop at game %d (no improvement for %d games)", ep, patience)
            break

    return GreedyLoopResult(
        policy=policy,
        best_reward=best_reward,
        greedy_sims=sims,
        early_stopped=early_stopped,
        early_stop_sim=early_stop_sim,
    )
