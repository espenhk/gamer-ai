"""Generic LSTM-based policies for RL.

LSTMCore            — single-layer LSTM with TMNF-compatible output heads
                      (steer / accel / brake).  Parameterised on ObsSpec
                      instead of a raw n_lidar_rays integer.  The flat weight
                      interface (to_flat / with_flat / flat_dim) is identical
                      to the game-specific LSTMPolicy so CMA-ES / ES wrappers
                      can swap them interchangeably.

LSTMEvolutionPolicy — (μ/μ_w, λ)-ES outer optimiser wrapping LSTMCore as the
                      inner individual.  Uses the _greedy_loop_cmaes interface:
                      sample_population() / update_distribution().  Step size
                      adapted via the 1/5 success rule (isotropic Gaussian).
"""
from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import numpy as np

from framework.obs_spec import ObsSpec
from framework.policies import BasePolicy

logger = logging.getLogger(__name__)


@runtime_checkable
class _LSTMIndividual(Protocol):
    """Structural interface required by :class:`LSTMEvolutionPolicy` for its inner individuals.

    Any class that implements ``flat_dim``, ``to_flat``, ``with_flat``, ``save``,
    ``__call__``, ``on_episode_start``, and ``on_episode_end`` satisfies this
    Protocol — including both :class:`LSTMCore` and ``SC2LSTMPolicy``.
    """

    @property
    def flat_dim(self) -> int: ...
    def to_flat(self) -> np.ndarray: ...
    def with_flat(self, flat: np.ndarray) -> _LSTMIndividual: ...
    def save(self, path: str) -> None: ...
    def __call__(self, obs: np.ndarray) -> np.ndarray: ...
    def on_episode_start(self, **kwargs: Any) -> None: ...
    def on_episode_end(self) -> None: ...


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


# --------------------------------------------------------------------------- #
# LSTMCore                                                                     #
# --------------------------------------------------------------------------- #

class LSTMCore(BasePolicy):
    """Single-layer LSTM policy (pure numpy).

    Hidden state (h, c) persists across steps within an episode.
    Trained via an outer evolutionary optimiser (LSTMEvolutionPolicy).
    ``on_episode_end()`` resets (h, c) to zeros.

    Output heads:
      steer = tanh(W_steer · h)           → [-1, 1]
      accel = sigmoid(W_accel · h) > 0.5  → {0, 1}
      brake = sigmoid(W_brake · h) > 0.5  → {0, 1}

    Parameters
    ----------
    obs_spec :
        Observation spec providing ``dim`` and ``scales`` for input normalisation.
    hidden_size :
        LSTM hidden / cell state dimensionality (default 32).
    seed :
        Optional RNG seed for weight initialisation.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        hidden_size: int = 32,
        seed: int | None = None,
    ) -> None:
        self._obs_spec    = obs_spec
        self._obs_dim     = obs_spec.dim
        self._scales      = obs_spec.scales
        self._hidden_size = int(hidden_size)

        h    = self._hidden_size
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

    def with_flat(self, flat: np.ndarray) -> "LSTMCore":
        """Return a new LSTMCore whose weights come from a flat parameter vector."""
        flat = np.asarray(flat, dtype=np.float32)
        if flat.shape[0] != self.flat_dim:
            raise ValueError(
                f"LSTMCore.with_flat: expected flat vector of size {self.flat_dim}, "
                f"got {flat.shape[0]}"
            )

        obj = object.__new__(LSTMCore)
        obj._obs_spec    = self._obs_spec
        obj._obs_dim     = self._obs_dim
        obj._scales      = self._scales
        obj._hidden_size = self._hidden_size

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

    def mutated(self, scale: float, **_) -> "LSTMCore":
        """Return a new LSTMCore with Gaussian noise applied to all parameters."""
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

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray | int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        **kwargs,
    ) -> None:
        pass  # no online update; training via outer evolutionary optimiser

    def _reset_hidden_state(self) -> None:
        self._h = np.zeros(self._hidden_size, dtype=np.float32)
        self._c = np.zeros(self._hidden_size, dtype=np.float32)

    def on_episode_start(self, **kwargs) -> None:
        self._reset_hidden_state()

    def on_episode_end(self) -> None:
        self._reset_hidden_state()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        return {
            "policy_type": "lstm",
            "hidden_size": self._hidden_size,
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
    def from_cfg(cls, cfg: dict, obs_spec: ObsSpec) -> "LSTMCore":
        obj = object.__new__(cls)
        obj._obs_spec    = obs_spec
        obj._hidden_size = int(cfg["hidden_size"])
        obj._obs_dim     = obs_spec.dim
        obj._scales      = obs_spec.scales
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


# --------------------------------------------------------------------------- #
# LSTMEvolutionPolicy                                                          #
# --------------------------------------------------------------------------- #

class LSTMEvolutionPolicy(BasePolicy):
    """(μ/μ_w, λ)-ES outer optimiser wrapping LSTMCore as the inner individual.

    Uses the ``_greedy_loop_cmaes`` interface: ``sample_population()`` /
    ``update_distribution()``.  Maintains an isotropic Gaussian search
    distribution (no full covariance matrix — infeasible for the ~7K-dimensional
    LSTM parameter space).  Step size adapted via the 1/5 success rule.

    Inference delegates to the champion LSTMCore.

    Parameters
    ----------
    obs_spec :
        Observation spec passed to each LSTMCore individual.
    hidden_size :
        LSTM hidden / cell state dimensionality (default 32).
    population_size :
        λ — offspring sampled per generation (default 20).
    initial_sigma :
        Starting step size (default 0.05).
    seed :
        Optional RNG seed.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        hidden_size: int = 32,
        population_size: int = 20,
        initial_sigma: float = 0.05,
        seed: int | None = None,
        *,
        _template: "_LSTMIndividual | None" = None,
    ) -> None:
        self._obs_spec = obs_spec
        self._lam      = int(population_size)
        self._sigma    = float(initial_sigma)
        self._rng      = np.random.default_rng(seed)

        self._template: _LSTMIndividual = _template if _template is not None else LSTMCore(obs_spec=obs_spec, hidden_size=hidden_size)
        self._flat_dim = self._template.flat_dim
        self._mean     = self._template.to_flat().astype(np.float64)

        mu             = self._lam // 2
        self._mu       = mu
        raw_w          = np.array(
            [np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)],
            dtype=np.float64,
        )
        self._recomb_w = raw_w / raw_w.sum()

        self._pop: list[np.ndarray]              = []
        self._champion: _LSTMIndividual | None   = None
        self._champion_reward: float             = float("-inf")

    # ------------------------------------------------------------------
    # _greedy_loop_cmaes interface
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

    def initialize_from_champion(self, champion: _LSTMIndividual) -> None:
        champion_flat_dim    = champion.flat_dim
        expected_hidden_size = getattr(self._template, "_hidden_size", None)
        champion_hidden_size = getattr(champion,       "_hidden_size", None)
        expected_obs_dim     = getattr(self._template, "_obs_dim", None)
        champion_obs_dim     = getattr(champion,       "_obs_dim", None)

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
        if mismatch_reasons:
            raise ValueError(
                "Cannot initialise LSTMEvolutionPolicy from an incompatible champion: "
                + "; ".join(mismatch_reasons)
            )
        self._champion = champion
        self._mean     = champion.to_flat().astype(np.float64)
        logger.info("[LSTMEvolutionPolicy] seeded mean from champion")

    def sample_population(self) -> list[_LSTMIndividual]:
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
                "LSTMEvolutionPolicy: no champion yet — call sample_population() "
                "and update_distribution() for at least one generation first."
            )
        return self._champion(obs)

    def on_episode_start(self, **kwargs) -> None:
        if self._champion is not None:
            self._champion.on_episode_start(**kwargs)

    def on_episode_end(self) -> None:
        if self._champion is not None:
            self._champion.on_episode_end()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        return {
            "policy_type":     "lstm",
            "hidden_size":     self._template._hidden_size,
            "population_size": self._lam,
            "sigma":           float(self._sigma),
            "obs_dim":         self._template._obs_dim,
            "champion_reward": float(self._champion_reward),
        }

    def save(self, path: str) -> None:
        if self._champion is not None:
            self._champion.save(path)

    def save_trainer_state(self, path: str) -> None:
        """Persist isotropic ES distribution state (mean, sigma) to an .npz file."""
        np.savez(
            path,
            mean     = self._mean,
            sigma    = np.float64(self._sigma),
            flat_dim = np.int64(self._flat_dim),
        )
        logger.debug("[LSTMEvolutionPolicy] trainer state saved → %s", path)

    def load_trainer_state(self, path: str) -> None:
        """Restore ES distribution state from an .npz file.

        Raises ValueError if the saved flat_dim does not match.
        """
        with np.load(path) as data:
            saved_flat_dim = int(data["flat_dim"])
            if saved_flat_dim != self._flat_dim:
                raise ValueError(
                    f"LSTMEvolutionPolicy: trainer state flat_dim mismatch — "
                    f"saved={saved_flat_dim}, current={self._flat_dim}. "
                    f"The network architecture (hidden_size or obs_spec) may have changed. "
                    f"Use --re-initialize to restart from scratch."
                )
            self._mean  = data["mean"].astype(np.float64)
            self._sigma = float(data["sigma"])
        logger.info("[LSTMEvolutionPolicy] trainer state loaded from %s (sigma=%.4f)",
                    path, self._sigma)
