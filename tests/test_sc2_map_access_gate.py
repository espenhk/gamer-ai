"""Tests for games.sc2.map_access_gate — cross-process SC2 map-access serialiser.

Covers issue #254: each call to ``acquire_map_access_slot`` must block
until at least ``gap_s`` seconds have elapsed since the previous grant,
so concurrent PySC2 workers don't race on the same ``.SC2Map`` file.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import pytest

from games.sc2 import map_access_gate
from games.sc2.map_access_gate import (
    DEFAULT_GAP_S,
    acquire_map_access_slot,
)


@pytest.fixture
def lock_path(tmp_path):
    return str(tmp_path / "sc2-map-access.lock")


class TestSingleProcess:
    def test_first_call_returns_zero_wait(self, lock_path):
        sleeps: list[float] = []
        wait = acquire_map_access_slot(
            gap_s=5.0,
            lock_path=lock_path,
            _sleep=sleeps.append,
            _now=lambda: 1000.0,
        )
        assert wait == 0.0
        assert sleeps == []
        with open(lock_path) as f:
            assert float(f.read().strip()) == 1000.0

    def test_second_call_within_gap_waits_remainder(self, lock_path):
        # Pre-seed last access timestamp
        with open(lock_path, "w") as f:
            f.write("1000.0\n")

        sleeps: list[float] = []
        wait = acquire_map_access_slot(
            gap_s=5.0,
            lock_path=lock_path,
            _sleep=sleeps.append,
            _now=lambda: 1002.0,  # 2s after last access
        )
        assert wait == pytest.approx(3.0)
        assert sleeps == [pytest.approx(3.0)]

    def test_second_call_after_gap_no_wait(self, lock_path):
        with open(lock_path, "w") as f:
            f.write("1000.0\n")

        sleeps: list[float] = []
        wait = acquire_map_access_slot(
            gap_s=5.0,
            lock_path=lock_path,
            _sleep=sleeps.append,
            _now=lambda: 1010.0,
        )
        assert wait == 0.0
        assert sleeps == []
        with open(lock_path) as f:
            assert float(f.read().strip()) == 1010.0

    def test_gap_zero_short_circuits_without_io(self, tmp_path, lock_path):
        sleeps: list[float] = []
        wait = acquire_map_access_slot(
            gap_s=0.0,
            lock_path=lock_path,
            _sleep=sleeps.append,
        )
        assert wait == 0.0
        assert sleeps == []
        # File should not have been touched.
        assert not os.path.exists(lock_path)

    def test_negative_gap_treated_as_disabled(self, lock_path):
        sleeps: list[float] = []
        wait = acquire_map_access_slot(
            gap_s=-1.0,
            lock_path=lock_path,
            _sleep=sleeps.append,
        )
        assert wait == 0.0
        assert sleeps == []
        assert not os.path.exists(lock_path)

    def test_corrupt_timestamp_treated_as_no_prior_access(self, lock_path):
        with open(lock_path, "w") as f:
            f.write("not-a-number\n")

        sleeps: list[float] = []
        wait = acquire_map_access_slot(
            gap_s=5.0,
            lock_path=lock_path,
            _sleep=sleeps.append,
            _now=lambda: 1000.0,
        )
        assert wait == 0.0
        assert sleeps == []

    def test_empty_lock_file_treated_as_no_prior_access(self, lock_path):
        open(lock_path, "w").close()

        sleeps: list[float] = []
        wait = acquire_map_access_slot(
            gap_s=5.0,
            lock_path=lock_path,
            _sleep=sleeps.append,
            _now=lambda: 1000.0,
        )
        assert wait == 0.0
        assert sleeps == []

    def test_clock_skew_future_timestamp_clamped_to_now(self, lock_path, caplog):
        """If the stored timestamp is ahead of the current clock (e.g. NTP
        rewinds the wall clock), the gate must clamp `last` to `now`
        rather than sleep for the entire skew + gap_s."""
        # Store a timestamp far in the future (simulating clock rewind).
        with open(lock_path, "w") as f:
            f.write("9999999999.0\n")

        sleeps: list[float] = []
        wait = acquire_map_access_slot(
            gap_s=5.0,
            lock_path=lock_path,
            _sleep=sleeps.append,
            _now=lambda: 1000.0,
        )
        # After clamping last → now, wait = (now + gap) - now = gap.
        assert wait == pytest.approx(5.0)
        assert sleeps == [pytest.approx(5.0)]


class TestEnvVarConfiguration:
    def test_lock_path_env_var(self, tmp_path, monkeypatch):
        custom = str(tmp_path / "custom.lock")
        monkeypatch.setenv("GAMER_AI_SC2_MAP_LOCK_PATH", custom)
        acquire_map_access_slot(
            gap_s=5.0,
            _sleep=lambda s: None,
            _now=lambda: 1000.0,
        )
        assert os.path.exists(custom)

    def test_gap_env_var(self, lock_path, monkeypatch):
        monkeypatch.setenv("GAMER_AI_SC2_MAP_GAP_S", "2.5")
        with open(lock_path, "w") as f:
            f.write("1000.0\n")

        sleeps: list[float] = []
        wait = acquire_map_access_slot(
            lock_path=lock_path,
            _sleep=sleeps.append,
            _now=lambda: 1001.0,
        )
        assert wait == pytest.approx(1.5)

    def test_gap_env_var_invalid_falls_back_to_default(
        self, lock_path, monkeypatch, caplog
    ):
        monkeypatch.setenv("GAMER_AI_SC2_MAP_GAP_S", "garbage")
        with open(lock_path, "w") as f:
            f.write("1000.0\n")

        sleeps: list[float] = []
        wait = acquire_map_access_slot(
            lock_path=lock_path,
            _sleep=sleeps.append,
            _now=lambda: 1000.0,
        )
        assert wait == pytest.approx(DEFAULT_GAP_S)
        assert sleeps == [pytest.approx(DEFAULT_GAP_S)]

    def test_gap_env_var_negative_falls_back_to_default(
        self, lock_path, monkeypatch
    ):
        monkeypatch.setenv("GAMER_AI_SC2_MAP_GAP_S", "-1")
        with open(lock_path, "w") as f:
            f.write("1000.0\n")

        sleeps: list[float] = []
        acquire_map_access_slot(
            lock_path=lock_path,
            _sleep=sleeps.append,
            _now=lambda: 1000.0,
        )
        # Default gap (5.0) applied, so a 5s wait occurs.
        assert sleeps == [pytest.approx(DEFAULT_GAP_S)]

    def test_gap_env_var_zero_disables_gate(self, lock_path, monkeypatch):
        monkeypatch.setenv("GAMER_AI_SC2_MAP_GAP_S", "0")
        sleeps: list[float] = []
        wait = acquire_map_access_slot(
            lock_path=lock_path,
            _sleep=sleeps.append,
        )
        assert wait == 0.0
        assert sleeps == []
        assert not os.path.exists(lock_path)


def _gate_worker(
    lock_path: str,
    gap_s: float,
    grant_queue: mp.Queue,
) -> None:
    """Child-process helper for multi-process serialisation test."""
    from games.sc2.map_access_gate import acquire_map_access_slot

    acquire_map_access_slot(gap_s=gap_s, lock_path=lock_path)
    grant_queue.put(time.time())


@pytest.mark.skipif(
    not hasattr(__import__("os"), "fork"),
    reason="requires POSIX fork (fcntl.flock-based serialisation)",
)
class TestMultiProcess:
    def test_concurrent_workers_are_serialised_with_gap(self, lock_path):
        """Two concurrent processes hit the gate at the same time: the
        second must be granted ≥ gap_s after the first."""
        gap = 1.0  # short for test runtime
        ctx = mp.get_context("fork")
        q: mp.Queue = ctx.Queue()

        # Seed lock file with a recent timestamp so the FIRST gate call
        # also has to wait — otherwise the test is sensitive to scheduling.
        with open(lock_path, "w") as f:
            f.write(f"{time.time()}\n")

        procs = [
            ctx.Process(target=_gate_worker, args=(lock_path, gap, q))
            for _ in range(3)
        ]
        for p in procs:
            p.start()
        try:
            for p in procs:
                p.join(timeout=15)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2)
                    if p.is_alive():
                        p.kill()
                        p.join(timeout=2)
                    pytest.fail(f"gate worker pid={p.pid} did not finish within 15s")
                assert p.exitcode == 0
        finally:
            # Defensive: ensure no stragglers leak into the rest of the test session.
            for p in procs:
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2)

        grants = sorted(q.get_nowait() for _ in range(3))
        # Each consecutive grant must be ≥ gap apart (allow small slack
        # for scheduling jitter on slow CI).
        for earlier, later in zip(grants, grants[1:]):
            assert later - earlier >= gap - 0.15, (
                f"grants spaced {later - earlier:.3f}s apart, expected ≥ {gap}s"
            )


class TestRealSleepIntegration:
    """One end-to-end test against the real clock to confirm the
    test-seam wiring matches actual behaviour."""

    def test_actual_clock_waits_remaining_gap(self, lock_path):
        with open(lock_path, "w") as f:
            f.write(f"{time.time()}\n")

        t0 = time.monotonic()
        acquire_map_access_slot(gap_s=0.3, lock_path=lock_path)
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.25  # allow a little jitter on the low side
