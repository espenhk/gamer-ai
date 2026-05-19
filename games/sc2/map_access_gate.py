"""Cross-process gate enforcing a minimum gap between SC2 map-file accesses.

Workaround for issue #254: when multiple PySC2 binaries boot at roughly
the same time (typically a ``grid_search.py --distribute --local-workers
N`` run, or an intra-run ``n_workers > 1`` pool) they race to read the
same ``.SC2Map`` file and one or more of them occasionally fails with a
"map not found" error.

The Popen-time stagger in ``grid_search._launch_local_workers`` only
spaces out the *initial* subprocess launches; every subsequent SC2Env
construction in those workers (one per experiment for distributed
workers, plus every parallel-eval worker startup) hits the race again.
This module guarantees a minimum gap between every SC2Env construction
on the same host, for the full lifetime of the grid-search run.

Usage::

    from games.sc2.map_access_gate import acquire_map_access_slot
    acquire_map_access_slot()        # blocks if needed, then returns
    sc2_env.SC2Env(...)              # safe to boot now

Behaviour:

* Maintains a single timestamp file (path from
  ``GAMER_AI_SC2_MAP_LOCK_PATH``, default ``<tempdir>/gamer-ai-sc2-map-access.lock``)
  storing the wall-clock time of the most recent grant.
* Each caller takes an exclusive ``fcntl.flock`` on the file, computes
  ``wait = (last + gap_s) - now``, sleeps that long if positive, writes
  the new ``now`` timestamp, and releases the lock.
* Because the sleep happens *while holding the lock*, callers are
  serialised: the next caller cannot read a stale timestamp and skip the
  wait.
* The gap defaults to 5.0 s and can be overridden per-process via
  ``GAMER_AI_SC2_MAP_GAP_S`` or per-call via the ``gap_s`` argument. A
  value of ``0`` disables the gate entirely (no I/O, no sleep).

Cross-platform: ``fcntl`` is POSIX-only. On Windows the gate falls back
to an unconditional ``sleep(gap_s)`` (single-process best-effort) and
logs a warning — the multi-worker SC2 scenario this fixes runs on Linux
in practice (see ``CLAUDE.md`` SC2 setup notes).
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_GAP_S = 5.0
_DEFAULT_LOCK_BASENAME = "gamer-ai-sc2-map-access.lock"


def _default_lock_path() -> str:
    return os.path.join(tempfile.gettempdir(), _DEFAULT_LOCK_BASENAME)


def _resolve_lock_path(explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    return os.environ.get("GAMER_AI_SC2_MAP_LOCK_PATH", _default_lock_path())


def _resolve_gap_s(explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get("GAMER_AI_SC2_MAP_GAP_S")
    if raw is None:
        return DEFAULT_GAP_S
    try:
        v = float(raw)
        if v < 0:
            raise ValueError
        return v
    except ValueError:
        logger.warning(
            "GAMER_AI_SC2_MAP_GAP_S=%r is not a non-negative float; "
            "using default %.1fs",
            raw, DEFAULT_GAP_S,
        )
        return DEFAULT_GAP_S


def acquire_map_access_slot(
    gap_s: float | None = None,
    lock_path: str | None = None,
    *,
    _sleep: Callable[[float], None] = time.sleep,
    _now: Callable[[], float] = time.time,
) -> float:
    """Block until ``gap_s`` seconds have elapsed since the last grant.

    Returns the number of seconds spent waiting (``0.0`` when the gate
    fired with no contention). The ``_sleep`` and ``_now`` keyword
    arguments are test seams.
    """
    resolved_gap = _resolve_gap_s(gap_s)
    if resolved_gap <= 0:
        return 0.0

    resolved_path = _resolve_lock_path(lock_path)
    parent = os.path.dirname(resolved_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        import fcntl  # POSIX-only
    except ImportError:
        logger.warning(
            "fcntl unavailable (Windows?) — falling back to unconditional "
            "%.1fs sleep before SC2Env construction. Multi-process "
            "map-access serialisation is best-effort on this platform.",
            resolved_gap,
        )
        _sleep(resolved_gap)
        return resolved_gap

    with open(resolved_path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read().strip()
            try:
                last = float(content) if content else 0.0
            except ValueError:
                last = 0.0

            now = _now()
            wait = (last + resolved_gap) - now
            if wait > 0:
                logger.info(
                    "SC2 map-access gate: waiting %.1fs to maintain %.1fs "
                    "gap since last access at %.3f",
                    wait, resolved_gap, last,
                )
                _sleep(wait)
                now = _now()
            else:
                wait = 0.0

            f.seek(0)
            f.truncate()
            f.write(f"{now}\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return max(wait, 0.0)
