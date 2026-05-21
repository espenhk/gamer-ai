"""Generic (μ/μ_w, λ)-CMA-ES for reinforcement learning.

CMAESDistribution — pure-math CMA-ES distribution (Hansen 2016).
                    Operates on flat float64 parameter vectors; knows nothing
                    about observation spaces or policy representations.

CMAESPolicy       — wraps CMAESDistribution with a *policy_factory* callable
                    that converts flat vectors to evaluable policy objects.
                    Exposes the same interface as ``_greedy_loop_cmaes``
                    expects: ``sample_population()``, ``update_distribution()``,
                    ``save()``, ``save_trainer_state()``, ``load_trainer_state()``,
                    ``champion_reward``, ``sigma``, ``population_size``,
                    ``to_cfg()``.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, TypeVar

import numpy as np

from framework.obs_spec import ObsSpec
from framework.policies import BasePolicy

logger = logging.getLogger(__name__)

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# CMAESDistribution                                                            #
# --------------------------------------------------------------------------- #

class CMAESDistribution:
    """(μ/μ_w, λ)-CMA-ES distribution over *n*-dimensional parameter vectors.

    Implements step-size adaptation via CSA and full covariance adaptation via
    the rank-1 + rank-μ update (Hansen 2016, Section 3).

    Parameters
    ----------
    n :
        Dimensionality of the parameter space.
    population_size :
        λ — offspring sampled per generation.
    initial_sigma :
        Starting step size σ₀.
    seed :
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        n: int,
        *,
        population_size: int = 20,
        initial_sigma: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self._n   = int(n)
        self._lam = int(population_size)

        mu            = self._lam // 2
        self._mu      = mu
        raw_w         = np.array(
            [np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)],
            dtype=np.float64,
        )
        self._weights = raw_w / raw_w.sum()
        self._mu_eff  = 1.0 / float(np.sum(self._weights ** 2))

        # Step-size adaptation constants (Hansen 2016, S3)
        self._cs   = (self._mu_eff + 2) / (n + self._mu_eff + 5)
        self._ds   = (
            1 + 2 * max(0.0, float(np.sqrt((self._mu_eff - 1) / (n + 1))) - 1)
            + self._cs
        )
        self._chin = float(
            np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
        )

        # Covariance adaptation constants
        self._cc  = (4 + self._mu_eff / n) / (n + 4 + 2 * self._mu_eff / n)
        self._c1  = 2.0 / ((n + 1.3) ** 2 + self._mu_eff)
        self._cmu = min(
            1.0 - self._c1,
            2.0 * (self._mu_eff - 2 + 1.0 / self._mu_eff)
            / ((n + 2) ** 2 + self._mu_eff),
        )

        self._rng = np.random.default_rng(seed)

        # Distribution state (float64 for numerical stability)
        self._mean     = self._rng.standard_normal(n).astype(np.float64)
        self._sigma    = float(initial_sigma)
        self._ps       = np.zeros(n, dtype=np.float64)
        self._pc       = np.zeros(n, dtype=np.float64)
        self._C        = np.eye(n, dtype=np.float64)
        self._B        = np.eye(n, dtype=np.float64)
        self._D        = np.ones(n, dtype=np.float64)
        self._invsqrtC = np.eye(n, dtype=np.float64)
        self._eigengen = 0
        self._gen      = 0

        self._pop_xs: list[np.ndarray] = []
        self._pop_ys: list[np.ndarray] = []

    def initialize_random(self) -> None:
        """Set the search mean to zero (CMA-ES adapts scale via σ)."""
        self._mean = np.zeros(self._n, dtype=np.float64)
        logger.info("[CMAESDistribution] initialised with zero mean, sigma=%.3f", self._sigma)

    def _update_eigen(self) -> None:
        self._C = np.triu(self._C) + np.triu(self._C, 1).T
        eigvals, self._B = np.linalg.eigh(self._C)
        eigvals          = np.maximum(eigvals, 1e-20)
        self._D          = np.sqrt(eigvals)
        self._invsqrtC   = self._B @ np.diag(1.0 / self._D) @ self._B.T
        self._eigengen   = self._gen

    def sample(self) -> list[np.ndarray]:
        """Sample λ offspring from N(mean, σ² · C). Returns flat parameter vectors."""
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
        return list(self._pop_xs)

    def update(self, rewards: list[float]) -> tuple[float, int]:
        """Apply (μ/μ_w, λ)-CMA-ES update.

        Returns ``(best_r, best_idx)`` where ``best_r`` is the highest reward
        in this generation and ``best_idx`` is that offspring's index in the
        population list returned by the preceding ``sample()`` call.
        """
        if len(rewards) != self._lam:
            raise ValueError(f"Expected {self._lam} rewards, got {len(rewards)}")
        if len(self._pop_xs) != self._lam or len(self._pop_ys) != self._lam:
            raise RuntimeError(
                "update() called before a matching sample(). "
                f"Expected {self._lam} samples in _pop_xs/_pop_ys, "
                f"got {len(self._pop_xs)}/{len(self._pop_ys)}. "
                "Call sample() first."
            )
        n     = self._n
        order = np.argsort(rewards)[::-1]

        best_idx  = int(order[0])
        best_r    = float(rewards[best_idx])

        elite_ys  = np.stack([self._pop_ys[order[i]] for i in range(self._mu)])
        step      = np.einsum("i,ij->j", self._weights, elite_ys)

        self._mean = self._mean + self._sigma * step

        ps_scale  = float(np.sqrt(self._cs * (2 - self._cs) * self._mu_eff))
        self._ps  = (1 - self._cs) * self._ps + ps_scale * (self._invsqrtC @ step)

        ps_norm     = float(np.linalg.norm(self._ps))
        self._sigma = float(np.clip(
            self._sigma * np.exp((self._cs / self._ds) * (ps_norm / self._chin - 1)),
            1e-10, 1e6,
        ))

        ps_norm_normed = ps_norm / float(
            np.sqrt(1 - (1 - self._cs) ** (2 * (self._gen + 1)))
        )
        h_sigma  = 1.0 if ps_norm_normed < (1.4 + 2.0 / (n + 1)) * self._chin else 0.0
        pc_scale = float(np.sqrt(self._cc * (2 - self._cc) * self._mu_eff))
        self._pc = (1 - self._cc) * self._pc + h_sigma * pc_scale * step

        delta_h = (1 - h_sigma) * self._cc * (2 - self._cc)
        rank1   = np.outer(self._pc, self._pc)
        rank_mu = np.einsum("i,ij,ik->jk", self._weights, elite_ys, elite_ys)
        self._C = (
            (1 - self._c1 - self._cmu) * self._C
            + self._c1 * (rank1 + delta_h * self._C)
            + self._cmu * rank_mu
        )

        self._gen += 1
        return best_r, best_idx

    def save_state(self, path: str) -> None:
        """Persist full distribution state to an .npz file."""
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

    def load_state(self, path: str) -> None:
        """Restore distribution state from an .npz file.

        Raises ValueError if the saved dimension does not match.
        """
        with np.load(path) as data:
            n_saved = int(data["n"])
            if n_saved != self._n:
                raise ValueError(
                    f"CMAESDistribution: state dimension mismatch — "
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


# --------------------------------------------------------------------------- #
# CMAESPolicy                                                                  #
# --------------------------------------------------------------------------- #

class CMAESPolicy(BasePolicy):
    """(μ/μ_w, λ)-CMA-ES wrapped as a policy compatible with ``_greedy_loop_cmaes``.

    The parameter space is a flat float64 vector of length *n_params*.
    A *policy_factory* converts each sampled vector to an evaluable policy
    object (e.g. a ``WeightedLinearPolicy`` for TMNF or an
    ``SC2MultiHeadLinearPolicy`` for SC2).

    All CMA-ES distribution state is stored in the wrapped
    :class:`CMAESDistribution` instance; the delegating properties below
    expose it so downstream test and analytics code can access it unchanged.

    Parameters
    ----------
    obs_spec :
        Observation spec of the target environment (passed through to
        ``policy_factory``).
    policy_factory :
        Callable ``(flat: np.ndarray, obs_spec: ObsSpec) -> policy`` that
        builds an evaluable policy from a parameter vector.
    n_params :
        Dimensionality of the flat parameter vector.
    population_size :
        λ — offspring sampled per generation (default 20).
    initial_sigma :
        Starting step size σ₀ (default 0.3).
    eval_episodes :
        Episodes to average per individual (default 1).
    seed :
        Optional RNG seed.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        policy_factory: Callable[[np.ndarray, ObsSpec], Any],
        n_params: int,
        *,
        population_size: int = 20,
        initial_sigma: float = 0.3,
        eval_episodes: int = 1,
        seed: int | None = None,
    ) -> None:
        self._obs_spec        = obs_spec
        self._policy_factory  = policy_factory
        self._eval_episodes   = max(1, int(eval_episodes))
        self._dist            = CMAESDistribution(
            n_params,
            population_size = population_size,
            initial_sigma   = initial_sigma,
            seed            = seed,
        )

        self._champion        = None
        self._champion_reward = float("-inf")

    # ------------------------------------------------------------------
    # Delegating properties — expose distribution internals for tests /
    # analytics that inspect policy._mean, policy._C, etc.
    # ------------------------------------------------------------------

    @property
    def _n(self)        -> int:            return self._dist._n
    @property
    def _lam(self)      -> int:            return self._dist._lam
    @property
    def _mu(self)       -> int:            return self._dist._mu
    @property
    def _weights(self)  -> np.ndarray:     return self._dist._weights
    @property
    def _mean(self)     -> np.ndarray:     return self._dist._mean
    @property
    def _sigma(self)    -> float:          return self._dist._sigma
    @property
    def _C(self)        -> np.ndarray:     return self._dist._C
    @property
    def _B(self)        -> np.ndarray:     return self._dist._B
    @property
    def _D(self)        -> np.ndarray:     return self._dist._D
    @property
    def _invsqrtC(self) -> np.ndarray:     return self._dist._invsqrtC
    @property
    def _ps(self)       -> np.ndarray:     return self._dist._ps
    @property
    def _pc(self)       -> np.ndarray:     return self._dist._pc
    @property
    def _gen(self)      -> int:            return self._dist._gen
    @property
    def _pop_xs(self)   -> list:           return self._dist._pop_xs
    @property
    def _pop_ys(self)   -> list:           return self._dist._pop_ys

    # ------------------------------------------------------------------
    # _greedy_loop_cmaes interface
    # ------------------------------------------------------------------

    @property
    def population_size(self) -> int:
        return self._dist._lam

    @property
    def champion_reward(self) -> float:
        return self._champion_reward

    @property
    def sigma(self) -> float:
        return self._dist._sigma

    def initialize_random(self) -> None:
        """Seed the search mean at zero."""
        self._dist.initialize_random()

    def initialize_from_champion(self, champion: Any) -> None:
        """Seed the search mean from an existing champion's flat weight vector."""
        self._champion = champion

        seeded_reward: float | None = None
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
        self._dist._mean = champion.to_flat().astype(np.float64)
        logger.info(
            "[CMAESPolicy] seeded mean from champion%s",
            "" if seeded_reward is None
            else f" (baseline reward={self._champion_reward:.6f})",
        )

    def sample_population(self) -> list:
        """Sample λ offspring. Returns a list of policy objects."""
        xs = self._dist.sample()
        return [self._policy_factory(x, self._obs_spec) for x in xs]

    def update_distribution(self, rewards: list[float]) -> bool:
        """Apply (μ/μ_w, λ)-CMA-ES update. Returns True if champion improved."""
        best_r, best_idx = self._dist.update(rewards)

        improved = False
        if best_r > self._champion_reward:
            self._champion_reward = best_r
            self._champion = self._policy_factory(
                self._dist._pop_xs[best_idx], self._obs_spec
            )
            improved = True

        return improved

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self._champion is None:
            raise RuntimeError(
                "CMAESPolicy: no champion yet — call sample_population() and "
                "update_distribution() for at least one generation first."
            )
        return self._champion(obs)

    def to_cfg(self) -> dict:
        return {
            "policy_type":     "cmaes",
            "population_size": self._dist._lam,
            "sigma":           float(self._dist._sigma),
            "n_params":        self._dist._n,
            "eval_episodes":   self._eval_episodes,
            "champion_reward": float(self._champion_reward),
        }

    def save(self, path: str) -> None:
        """Save champion weights (must implement save())."""
        if self._champion is not None and hasattr(self._champion, "save"):
            self._champion.save(path)

    def save_trainer_state(self, path: str) -> None:
        """Persist full CMA-ES distribution state to an .npz file."""
        self._dist.save_state(path)
        logger.debug("[CMAESPolicy] trainer state saved → %s", path)

    def load_trainer_state(self, path: str) -> None:
        """Restore CMA-ES distribution state from an .npz file.

        Raises ValueError if the saved dimension does not match.
        """
        self._dist.load_state(path)
        logger.info("[CMAESPolicy] trainer state loaded from %s (gen=%d, sigma=%.4f)",
                    path, self._dist._gen, self._dist._sigma)
