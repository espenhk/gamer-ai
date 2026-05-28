"""Self-play opponent management for multi-agent RL training.

Three modes
-----------
exact :
    Opponent is always an exact snapshot of the current champion policy.
    Refreshed at the end of each training generation / episode.
mutated :
    Opponent is a slightly mutated copy of the current champion.
    Mutation strength is controlled by *mutation_scale*.
    Refreshed at the end of each training generation / episode.
top_n :
    Maintains a pool of up to *top_n* historical champion snapshots.
    The pool grows whenever the champion improves; when at capacity the
    weakest snapshot is replaced if the new champion scores higher.
    A uniformly random pool member is selected as opponent each generation.

Usage
-----
In the training loop::

    manager = SelfPlayManager(mode="top_n", top_n=5)
    env.set_opponent_policy(manager.build_initial_opponent(policy))

    for gen in range(n_generations):
        rewards = evaluate(policy)
        improved = policy.update(rewards)
        new_opp = manager.step(policy, improved)
        if new_opp is not None:
            env.set_opponent_policy(new_opp)
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Valid self_play_mode values.
SELF_PLAY_MODES: frozenset[str] = frozenset({"exact", "mutated", "top_n"})


class SelfPlayManager:
    """Manages the opponent policy for self-play training.

    Parameters
    ----------
    mode :
        One of ``"exact"``, ``"mutated"``, or ``"top_n"``.
    mutation_scale :
        Std-dev of Gaussian weight perturbation applied in ``"mutated"``
        mode.  Ignored for the other modes.
    top_n :
        Maximum pool size for ``"top_n"`` mode.  Must be ≥ 1.  Ignored
        for ``"exact"`` and ``"mutated"``.
    seed :
        Optional integer RNG seed for reproducibility (controls pool
        sampling in ``"top_n"`` mode).
    """

    def __init__(
        self,
        mode: str = "exact",
        mutation_scale: float = 0.05,
        top_n: int = 5,
        seed: int | None = None,
    ) -> None:
        if mode not in SELF_PLAY_MODES:
            raise ValueError(
                f"Unknown self_play_mode {mode!r}; "
                f"must be one of {sorted(SELF_PLAY_MODES)}"
            )
        self._mode = mode
        self._mutation_scale = float(mutation_scale)
        self._top_n = max(1, int(top_n))
        # Pool entries: (score: float, callable: Any)
        self._pool: list[tuple[float, Any]] = []
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """The active self-play mode."""
        return self._mode

    def build_initial_opponent(self, policy: Any) -> Any:
        """Build the initial opponent at the start of a training run.

        Also seeds the pool (``"top_n"`` mode) with the initial policy
        snapshot so the very first generation has a valid opponent.

        Parameters
        ----------
        policy :
            The primary training policy before any training has occurred.

        Returns
        -------
        Any
            A callable ``(obs) -> action`` suitable for
            ``env.set_opponent_policy()``.
        """
        return self.step(policy, improved=True)

    def step(self, policy: Any, improved: bool) -> Any:
        """Update the opponent pool and return the next opponent callable.

        Call this once per training generation *after* the policy has
        been updated for that generation.

        Parameters
        ----------
        policy :
            The primary training policy (post-update).
        improved :
            Whether the champion score improved this generation.

        Returns
        -------
        Any
            A callable ``(obs) -> action``.  For ``"top_n"`` mode this
            is ``None`` only when the pool is empty (which cannot happen
            after :meth:`build_initial_opponent` has been called).
        """
        if self._mode == "exact":
            return self._snapshot_callable(policy)
        if self._mode == "mutated":
            return self._mutated_snapshot(policy)
        # top_n
        if improved or not self._pool:
            self._update_pool(policy)
        return self._pick_from_pool()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot_callable(self, policy: Any) -> Any:
        """Return a lightweight callable snapshot of the policy's champion.

        When the policy wrapper (e.g. ``SC2GeneticPolicy``) carries a
        ``_champion`` attribute that is itself callable (e.g.
        ``SC2MultiHeadLinearPolicy``), we deepcopy that individual rather
        than the whole population wrapper to avoid the cost of copying an
        entire population every generation.
        """
        champion = getattr(policy, "_champion", None)
        if champion is not None and callable(champion):
            return copy.deepcopy(champion)
        return copy.deepcopy(policy)

    def _mutated_snapshot(self, policy: Any) -> Any:
        """Return a mutated copy of the current champion.

        Uses the champion's ``mutated(scale, share)`` method when
        available (e.g. ``SC2MultiHeadLinearPolicy``).  Falls back to a
        plain deepcopy for policies without a ``mutated`` method.
        """
        champion = getattr(policy, "_champion", None)
        if champion is not None and hasattr(champion, "mutated"):
            return champion.mutated(scale=self._mutation_scale, share=1.0)
        if hasattr(policy, "mutated"):
            return policy.mutated(scale=self._mutation_scale)
        return self._snapshot_callable(policy)

    def _update_pool(self, policy: Any) -> None:
        """Add a champion snapshot to the pool, evicting the weakest entry
        when at capacity."""
        score = float(getattr(policy, "champion_reward", float("-inf")))
        snap = self._snapshot_callable(policy)
        entry: tuple[float, Any] = (score, snap)
        if len(self._pool) < self._top_n:
            self._pool.append(entry)
            logger.debug(
                "[SelfPlay/top_n] added champion (score=%.1f); pool %d/%d",
                score,
                len(self._pool),
                self._top_n,
            )
        else:
            worst_idx = min(range(len(self._pool)), key=lambda i: self._pool[i][0])
            if score > self._pool[worst_idx][0]:
                logger.debug(
                    "[SelfPlay/top_n] replaced pool[%d] (score %.1f → %.1f)",
                    worst_idx,
                    self._pool[worst_idx][0],
                    score,
                )
                self._pool[worst_idx] = entry

    def _pick_from_pool(self) -> Any | None:
        """Return a uniformly random callable from the pool."""
        if not self._pool:
            return None
        idx = int(self._rng.integers(len(self._pool)))
        return self._pool[idx][1]
