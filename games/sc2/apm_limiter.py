"""Token-bucket APM (actions per minute) limiter for SC2 agents.

Humans are constrained to a finite APM; this module lets training runs
simulate that constraint so policies learn under more human-like conditions.

Algorithm
---------
A standard token bucket: tokens refill at ``max_apm / 60`` tokens per
real-world second.  The bucket is capped at ``refill_rate * burst_s`` tokens
so short bursts are allowed but the agent cannot bank a full minute's worth
and spend them instantly.

Usage
-----
Create one limiter per env, call :meth:`reset` at the start of each episode,
and call :meth:`allow` before each step.  Pass the *same* monotonic clock
value that the env already tracks so no extra ``time.monotonic()`` call is
needed.  When :meth:`allow` returns ``False`` the caller should replace the
intended action with a no-op.

No-op actions (``fn_idx == 0``) are always allowed and **do not** consume a
token; they do not count as actions in real SC2 APM counting either.
"""

from __future__ import annotations


class ApmLimiter:
    """Rolling token-bucket rate limiter.

    Parameters
    ----------
    max_apm :
        Maximum actions per minute.  Must be > 0.
    burst_s :
        How many seconds' worth of tokens can accumulate.  Controls the
        maximum instantaneous burst size: ``max_apm / 60 * burst_s`` actions.
        Defaults to ``2.0`` (2 seconds) — short bursts are fine but the agent
        cannot execute all its budget in one frame.
    """

    def __init__(self, max_apm: int, burst_s: float = 2.0) -> None:
        if max_apm <= 0:
            raise ValueError(f"max_apm must be positive, got {max_apm}")
        if burst_s <= 0:
            raise ValueError(f"burst_s must be positive, got {burst_s}")
        self._refill_rate: float = max_apm / 60.0  # tokens per second
        self._max_tokens: float = self._refill_rate * burst_s
        self._tokens: float = self._max_tokens
        self._last_time: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tokens(self) -> float:
        """Current token count (read-only, for testing / diagnostics)."""
        return self._tokens

    @property
    def max_tokens(self) -> float:
        """Token bucket capacity."""
        return self._max_tokens

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, now: float) -> None:
        """Reset the limiter for a new episode.

        Parameters
        ----------
        now :
            Current wall-clock time in seconds (e.g. ``time.monotonic()``).
            The bucket is refilled to capacity.
        """
        self._tokens = self._max_tokens
        self._last_time = now

    def allow(self, now: float, fn_idx: int = -1) -> bool:
        """Decide whether an action may proceed.

        No-op actions (``fn_idx == 0``) are always permitted and do not
        consume a token, matching real SC2 APM conventions.

        Parameters
        ----------
        now :
            Current wall-clock time (seconds).
        fn_idx :
            Internal function index of the intended action.  Pass ``0`` for
            no-op (always allowed, no token consumed).  Any other value —
            including the default ``-1`` — is treated as a real action and
            will consume a token when the budget allows.

        Returns
        -------
        bool
            ``True`` → proceed with the action.
            ``False`` → throttled; caller should substitute a no-op.
        """
        # No-ops never cost a token.
        if fn_idx == 0:
            return True

        # Refill bucket based on elapsed time.
        elapsed = max(0.0, now - self._last_time)
        self._last_time = now
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False
