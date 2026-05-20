"""
Tests for distributed grid search — coordinator re-queue logic and JSON round-trip.

These tests do NOT require a live TMInterface session; they use synthetic data.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from distributed.protocol import (
    ComboSpec,
    ResultPayload,
    combo_to_dict,
    combo_from_dict,
    result_to_dict,
    result_from_dict,
    experiment_to_json,
    experiment_from_dict,
)
from distributed.coordinator import Coordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_combo(name: str = "test_combo") -> ComboSpec:
    return ComboSpec(
        name=name,
        track="a03_centerline",
        training_params={"speed": 10.0, "n_sims": 5, "mutation_scale": 0.05,
                         "in_game_episode_s": 30.0},
        reward_params={"progress_weight": 10000.0},
    )


def _make_experiment_data_dict(name: str = "test_combo") -> dict:
    """Return a minimal ExperimentData-compatible dict for round-trip tests."""
    from framework.analytics import (
        ExperimentData, GreedySimResult, RunTrace,
    )
    data = ExperimentData(
        experiment_name=name,
        probe_results=[],
        cold_start_restarts=[],
        greedy_sims=[
            GreedySimResult(
                sim=1, reward=123.4, improved=True,
                throttle_counts=[5, 10, 85], total_steps=100,
                trace=RunTrace(pos_x=[0.0, 1.0], pos_z=[0.0, 2.0],
                               throttle_state=[(0, 1)], total_reward=123.4),
            ),
            GreedySimResult(
                sim=2, reward=100.0, improved=False,
                throttle_counts=[10, 10, 80], total_steps=100,
            ),
        ],
        probe_floor=None,
        weights_file="experiments/a03_centerline/test_combo/policy_weights.yaml",
        reward_config_file="experiments/a03_centerline/test_combo/reward_config.yaml",
        training_params={"speed": 10.0, "n_sims": 5},
        timings={"greedy_s": 42.0},
        track="a03_centerline",
    )
    return data


# ---------------------------------------------------------------------------
# Protocol serialisation
# ---------------------------------------------------------------------------

class TestComboSpecSerialization:
    def test_round_trip(self):
        spec = _make_combo()
        d = combo_to_dict(spec)
        recovered = combo_from_dict(d)
        assert recovered.name == spec.name
        assert recovered.track == spec.track
        assert recovered.training_params == spec.training_params
        assert recovered.reward_params == spec.reward_params

    def test_dict_is_json_serialisable(self):
        spec = _make_combo()
        json.dumps(combo_to_dict(spec))  # must not raise


class TestResultPayloadSerialization:
    def test_round_trip(self):
        payload = ResultPayload(name="x", data_json='{"experiment_name": "x"}')
        d = result_to_dict(payload)
        recovered = result_from_dict(d)
        assert recovered.name == payload.name
        assert recovered.data_json == payload.data_json


class TestExperimentDataSerialization:
    def test_to_json_produces_valid_json(self):
        data = _make_experiment_data_dict()
        json_str = experiment_to_json(data)
        parsed = json.loads(json_str)
        assert parsed["experiment_name"] == "test_combo"

    def test_round_trip_greedy_sims(self):
        original = _make_experiment_data_dict()
        json_str = experiment_to_json(original)
        recovered = experiment_from_dict(json.loads(json_str))

        assert len(recovered.greedy_sims) == 2
        assert recovered.greedy_sims[0].reward == pytest.approx(123.4)
        assert recovered.greedy_sims[0].improved is True
        assert recovered.greedy_sims[1].improved is False

    def test_round_trip_preserves_throttle_counts(self):
        original = _make_experiment_data_dict()
        recovered = experiment_from_dict(json.loads(experiment_to_json(original)))
        assert recovered.greedy_sims[0].throttle_counts == [5, 10, 85]

    def test_round_trip_preserves_trace(self):
        original = _make_experiment_data_dict()
        recovered = experiment_from_dict(json.loads(experiment_to_json(original)))
        trace = recovered.greedy_sims[0].trace
        assert trace is not None
        assert trace.pos_x == [0.0, 1.0]
        assert trace.total_reward == pytest.approx(123.4)

    def test_round_trip_none_trace(self):
        original = _make_experiment_data_dict()
        recovered = experiment_from_dict(json.loads(experiment_to_json(original)))
        assert recovered.greedy_sims[1].trace is None

    def test_round_trip_metadata(self):
        original = _make_experiment_data_dict()
        recovered = experiment_from_dict(json.loads(experiment_to_json(original)))
        assert recovered.training_params == {"speed": 10.0, "n_sims": 5}
        assert recovered.timings["greedy_s"] == pytest.approx(42.0)
        assert recovered.track == "a03_centerline"

    def test_round_trip_task_metric_fields(self):
        """finish_time_s, mean_abs_lateral_offset, reward_components survive round-trip."""
        from framework.analytics import ExperimentData, GreedySimResult
        comps = {"progress": 8.5, "step_penalty": -0.1, "finish_bonus": 100.0}
        data = ExperimentData(
            experiment_name="task_metric_test",
            probe_results=[], cold_start_restarts=[],
            greedy_sims=[
                GreedySimResult(
                    sim=1, reward=99.0, improved=True,
                    throttle_counts=[0, 0, 100], total_steps=50,
                    finish_time_s=42.5,
                    mean_abs_lateral_offset=1.23,
                    reward_components=comps,
                ),
                GreedySimResult(
                    sim=2, reward=10.0, improved=False,
                    throttle_counts=[0, 50, 50], total_steps=50,
                    # None fields: ensure they round-trip as None
                ),
            ],
            probe_floor=None, weights_file="", reward_config_file="",
            training_params={}, timings={},
        )
        recovered = experiment_from_dict(json.loads(experiment_to_json(data)))
        s0 = recovered.greedy_sims[0]
        assert s0.finish_time_s == pytest.approx(42.5)
        assert s0.mean_abs_lateral_offset == pytest.approx(1.23)
        assert s0.reward_components == comps
        s1 = recovered.greedy_sims[1]
        assert s1.finish_time_s is None
        assert s1.mean_abs_lateral_offset is None
        assert s1.reward_components is None

    def test_round_trip_sc2_analytics_fields(self):
        """action_counts, obs_averages, xy_hist, skipped_frames survive round-trip.

        JSON serialises int dict keys as strings; the loader must restore them
        to int so callers continue to index by fn_idx integer.
        """
        from framework.analytics import ExperimentData, GreedySimResult
        hist = [[i + j for j in range(8)] for i in range(8)]
        data = ExperimentData(
            experiment_name="sc2_fields_test",
            probe_results=[], cold_start_restarts=[],
            greedy_sims=[
                GreedySimResult(
                    sim=1, reward=5.0, improved=True,
                    throttle_counts=[0, 0, 0], total_steps=50,
                    action_counts={0: 10, 1: 5, 2: 85},
                    obs_averages={"army_count": 3.0, "minerals": 150.0},
                    xy_hist=hist,
                    skipped_frames=12,
                ),
                GreedySimResult(
                    sim=2, reward=3.0, improved=False,
                    throttle_counts=[0, 0, 0], total_steps=50,
                ),
            ],
            probe_floor=None, weights_file="", reward_config_file="",
            training_params={}, timings={},
        )
        recovered = experiment_from_dict(json.loads(experiment_to_json(data)))
        s0 = recovered.greedy_sims[0]
        # action_counts keys must come back as int, not str.
        assert s0.action_counts == {0: 10, 1: 5, 2: 85}
        assert all(isinstance(k, int) for k in s0.action_counts)
        assert s0.obs_averages == {"army_count": 3.0, "minerals": 150.0}
        assert s0.xy_hist == hist
        assert s0.skipped_frames == 12
        s1 = recovered.greedy_sims[1]
        assert s1.action_counts is None
        assert s1.obs_averages is None
        assert s1.xy_hist is None
        assert s1.skipped_frames is None

    def test_numpy_arrays_serialised(self):
        """Numpy arrays in RunTrace should be serialised to lists without error."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        from framework.analytics import ExperimentData, GreedySimResult, RunTrace
        data = ExperimentData(
            experiment_name="np_test",
            probe_results=[], cold_start_restarts=[],
            greedy_sims=[GreedySimResult(
                sim=1, reward=1.0, improved=True,
                throttle_counts=[0, 0, 100], total_steps=100,
                trace=RunTrace(
                    pos_x=np.array([1.0, 2.0, 3.0]),
                    pos_z=np.array([4.0, 5.0, 6.0]),
                    throttle_state=[],
                    total_reward=1.0,
                ),
            )],
            probe_floor=None, weights_file="", reward_config_file="",
            training_params={}, timings={},
        )
        json_str = experiment_to_json(data)
        recovered = experiment_from_dict(json.loads(json_str))
        assert recovered.greedy_sims[0].trace.pos_x == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Coordinator — work queue and re-queue logic
# ---------------------------------------------------------------------------

_TEST_TOKEN = "test-secret"


class TestCoordinator:
    def _start_coord(self, combos, hb_timeout=30.0):
        """Start a coordinator on an OS-assigned free port (port=0)."""
        coord = Coordinator(combos, token=_TEST_TOKEN, port=0, heartbeat_timeout=hb_timeout)
        coord.start()
        time.sleep(0.05)  # give the server thread a moment to accept connections
        return coord

    def _url(self, coord, path: str) -> str:
        return f"http://localhost:{coord.port}{path}"

    def _auth_headers(self, extra: dict | None = None) -> dict:
        h = {"Authorization": f"Bearer {_TEST_TOKEN}"}
        if extra:
            h.update(extra)
        return h

    def _get(self, coord, path: str):
        import urllib.request
        req = urllib.request.Request(self._url(coord, path), headers=self._auth_headers())
        return urllib.request.urlopen(req, timeout=5)

    def _post(self, coord, path: str, body: bytes):
        import urllib.request
        req = urllib.request.Request(
            self._url(coord, path), data=body, method="POST",
            headers=self._auth_headers({
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            }),
        )
        return urllib.request.urlopen(req, timeout=5)

    def test_work_queue_serves_all_combos(self):
        combos = [_make_combo(f"c{i}") for i in range(3)]
        coord = self._start_coord(combos)

        names_received = []
        for _ in range(3):
            with self._get(coord, "/work") as r:
                assert r.status == 200
                spec = combo_from_dict(json.loads(r.read()))
                names_received.append(spec.name)

        # Fourth request should be 204 (queue empty, items in-progress, hb_timeout=30 s)
        with self._get(coord, "/work") as r:
            assert r.status == 204

        coord.stop()
        assert set(names_received) == {"c0", "c1", "c2"}

    def test_status_endpoint(self):
        combos = [_make_combo("s1"), _make_combo("s2")]
        coord = self._start_coord(combos)

        with self._get(coord, "/status") as r:
            status = json.loads(r.read())

        assert status["total"] == 2
        assert status["queued"] == 2
        assert status["done"] == 0
        coord.stop()

    def test_result_accepted_and_done_event_fires(self):
        combo = _make_combo("done_test")
        coord = self._start_coord([combo])

        # Fetch work
        with self._get(coord, "/work") as r:
            assert r.status == 200

        # Post result
        data = _make_experiment_data_dict("done_test")
        payload = ResultPayload(name="done_test", data_json=experiment_to_json(data))
        body = json.dumps(result_to_dict(payload)).encode()
        with self._post(coord, "/result", body) as r:
            assert r.status == 200

        assert coord._done_event.is_set()
        runs = coord.wait_for_all()
        assert len(runs) == 1
        assert runs[0][0] == "done_test"
        coord.stop()

    def test_unknown_combo_name_rejected(self):
        import urllib.error

        coord = self._start_coord([_make_combo("known")])

        data = _make_experiment_data_dict("unknown")
        payload = ResultPayload(name="unknown", data_json=experiment_to_json(data))
        body = json.dumps(result_to_dict(payload)).encode()
        try:
            self._post(coord, "/result", body)
            assert False, "expected 400"
        except urllib.error.HTTPError as e:
            assert e.code == 400

        assert not coord._done_event.is_set()
        coord.stop()

    def test_duplicate_result_ignored(self):
        combo = _make_combo("dup_test")
        coord = self._start_coord([combo])

        with self._get(coord, "/work") as r:
            assert r.status == 200

        def _post_result():
            data = _make_experiment_data_dict("dup_test")
            payload = ResultPayload(name="dup_test", data_json=experiment_to_json(data))
            body = json.dumps(result_to_dict(payload)).encode()
            with self._post(coord, "/result", body) as r:
                return json.loads(r.read())

        resp1 = _post_result()
        assert resp1.get("status") == "ok"
        assert coord._done_event.is_set()

        resp2 = _post_result()
        assert resp2.get("ignored") == "duplicate"
        assert len(coord._results) == 1
        coord.stop()

    def test_stale_worker_item_requeued(self):
        """An item not heartbeated within heartbeat_timeout should be re-queued."""
        combo = _make_combo("requeue_test")
        # hb_timeout=0.5 → monitor fires every 0.5 s; dispatch gives a one-timeout
        # grace period, so the item becomes stale after ~2×timeout ≈ 1 s.
        coord = self._start_coord([combo], hb_timeout=0.5)

        # Fetch work (marks as in-progress) but send NO heartbeats
        with self._get(coord, "/work") as r:
            assert r.status == 200

        # Wait long enough for the monitor to re-queue the stale item
        # (needs > 2×hb_timeout to clear the initial grace period)
        time.sleep(2.5)

        # Item should now be back in the queue
        with self._get(coord, "/work") as r:
            assert r.status == 200
            spec = combo_from_dict(json.loads(r.read()))
            assert spec.name == "requeue_test"

        coord.stop()

    def test_heartbeat_prevents_requeue(self):
        """A worker sending heartbeats should NOT have its item re-queued."""
        combo = _make_combo("hb_test")
        coord = self._start_coord([combo], hb_timeout=1.0)

        with self._get(coord, "/work") as r:
            assert r.status == 200

        # Send heartbeats every 0.3 s for 1.5 s — should keep item alive
        def send_hbs():
            for _ in range(5):
                body = json.dumps({"name": "hb_test", "worker_id": "tester"}).encode()
                try:
                    self._post(coord, "/heartbeat", body)
                except Exception:
                    pass
                time.sleep(0.3)

        hb_thread = threading.Thread(target=send_hbs, daemon=True)
        hb_thread.start()
        hb_thread.join()

        # Queue should still be empty (item is still in-progress, not re-queued)
        with self._get(coord, "/work") as r:
            assert r.status == 204

        coord.stop()

    def test_empty_queue_returns_immediately(self):
        coord = self._start_coord([])
        runs = coord.wait_for_all()
        assert runs == []
        coord.stop()

    def test_unauthorized_request_rejected(self):
        """Requests with wrong or missing token must receive 401."""
        import urllib.request
        import urllib.error

        coord = self._start_coord([_make_combo("auth_test")])

        # No token
        try:
            with urllib.request.urlopen(self._url(coord, "/work"), timeout=5):
                assert False, "expected 401"
        except urllib.error.HTTPError as e:
            assert e.code == 401

        # Wrong token
        req = urllib.request.Request(
            self._url(coord, "/work"),
            headers={"Authorization": "Bearer wrong-token"},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected 401"
        except urllib.error.HTTPError as e:
            assert e.code == 401

        coord.stop()

    def test_lan_only_allows_private_and_loopback_client_ips(self):
        coord = Coordinator([], token=_TEST_TOKEN, allow_non_lan=False)

        class _H:
            def __init__(self, ip):
                self.client_address = (ip, 12345)
                self.status = None

            def send_response(self, code):
                self.status = code

            def send_header(self, _k, _v):
                pass

            def end_headers(self):
                pass

        for ip in ("127.0.0.1", "192.168.1.25", "10.0.0.8", "169.254.1.2"):
            h = _H(ip)
            assert coord._check_client_allowed(h)
            assert h.status is None

    def test_lan_only_rejects_public_client_ips(self):
        coord = Coordinator([], token=_TEST_TOKEN, allow_non_lan=False)

        class _H:
            def __init__(self):
                self.client_address = ("8.8.8.8", 12345)
                self.status = None

            def send_response(self, code):
                self.status = code

            def send_header(self, _k, _v):
                pass

            def end_headers(self):
                pass

        h = _H()
        assert not coord._check_client_allowed(h)
        assert h.status == 403

    def test_allow_non_lan_flag_accepts_public_client_ips(self):
        coord = Coordinator([], token=_TEST_TOKEN, allow_non_lan=True)

        class _H:
            client_address = ("8.8.8.8", 12345)

            def send_response(self, _code):
                assert False, "should not reject"

            def send_header(self, _k, _v):
                assert False, "should not reject"

            def end_headers(self):
                assert False, "should not reject"

        assert coord._check_client_allowed(_H())

    def _get_with_game(self, coord, game: str):
        """GET /work with an X-Worker-Game filter header."""
        import urllib.request
        req = urllib.request.Request(
            self._url(coord, "/work"),
            headers=self._auth_headers({"X-Worker-Game": game}),
        )
        return urllib.request.urlopen(req, timeout=5)

    def test_game_filter_returns_matching_combo(self):
        """A worker with X-Worker-Game only receives work for that game."""
        tmnf_combo = ComboSpec(
            name="tmnf_job",
            track="a03",
            training_params={},
            reward_params={},
            game="tmnf",
        )
        sc2_combo = ComboSpec(
            name="sc2_job",
            track="MoveToBeacon",
            training_params={},
            reward_params={},
            game="sc2",
        )
        coord = self._start_coord([tmnf_combo, sc2_combo])

        # SC2 worker should receive the SC2 job (second in queue).
        with self._get_with_game(coord, "sc2") as r:
            assert r.status == 200
            spec = combo_from_dict(json.loads(r.read()))
            assert spec.name == "sc2_job"
            assert spec.game == "sc2"

        coord.stop()

    def test_game_filter_skips_non_matching_combos(self):
        """A worker asking for sc2 should get 204 when only tmnf jobs are queued."""
        tmnf_combo = ComboSpec(
            name="tmnf_only",
            track="a03",
            training_params={},
            reward_params={},
            game="tmnf",
        )
        coord = self._start_coord([tmnf_combo])

        with self._get_with_game(coord, "sc2") as r:
            assert r.status == 204

        # The tmnf job should still be in the queue (not consumed).
        with self._get(coord, "/status") as r:
            status = json.loads(r.read())
        assert status["queued"] == 1

        coord.stop()

    def test_game_filter_preserves_queue_order(self):
        """Non-matching items are returned to the queue in their original order."""
        combos = [
            ComboSpec(name="tmnf_1", track="a03", training_params={}, reward_params={}, game="tmnf"),
            ComboSpec(name="tmnf_2", track="a03", training_params={}, reward_params={}, game="tmnf"),
            ComboSpec(name="sc2_1",  track="map",  training_params={}, reward_params={}, game="sc2"),
        ]
        coord = self._start_coord(combos)

        # An SC2 worker should skip tmnf_1 and tmnf_2 and receive sc2_1.
        with self._get_with_game(coord, "sc2") as r:
            assert r.status == 200
            spec = combo_from_dict(json.loads(r.read()))
            assert spec.name == "sc2_1"

        # The two tmnf jobs must still be in the queue in original order.
        for expected in ["tmnf_1", "tmnf_2"]:
            with self._get_with_game(coord, "tmnf") as r:
                assert r.status == 200
                spec = combo_from_dict(json.loads(r.read()))
                assert spec.name == expected

        coord.stop()

    def test_skip_endpoint_returns_item_to_queue(self):
        """POST /skip puts the named in-progress item back at the front of the queue."""
        combo = ComboSpec(
            name="skip_me",
            track="a03",
            training_params={},
            reward_params={},
            game="tmnf",
        )
        coord = self._start_coord([combo])

        # Claim the work item (no game filter — generic worker).
        with self._get(coord, "/work") as r:
            assert r.status == 200

        # The item should now be in-progress.
        with self._get(coord, "/status") as r:
            status = json.loads(r.read())
        assert status["in_progress"] == 1
        assert status["queued"] == 0

        # Return the item via POST /skip.
        skip_body = json.dumps({"name": "skip_me"}).encode()
        with self._post(coord, "/skip", skip_body) as r:
            assert r.status == 200

        # The item should be back in the queue.
        with self._get(coord, "/status") as r:
            status = json.loads(r.read())
        assert status["queued"] == 1
        assert status["in_progress"] == 0

        coord.stop()

    def test_skip_unknown_item_is_no_op(self):
        """POST /skip for an item not in progress returns 200 with a note."""
        coord = self._start_coord([_make_combo("existing")])

        skip_body = json.dumps({"name": "nonexistent"}).encode()
        with self._post(coord, "/skip", skip_body) as r:
            assert r.status == 200
            resp = json.loads(r.read())
            assert "note" in resp

        coord.stop()


# ---------------------------------------------------------------------------
# Worker game filter (unit tests — no real coordinator needed)
# ---------------------------------------------------------------------------

class TestWorkerGameFilter:
    """Unit tests for the worker --game filtering logic via run_worker internals."""

    def test_combo_spec_game_field_default(self):
        """ComboSpec.game defaults to 'tmnf' for backward compatibility."""
        spec = ComboSpec(name="x", track="y", training_params={}, reward_params={})
        assert spec.game == "tmnf"

    def test_combo_spec_game_field_explicit(self):
        """ComboSpec.game can be set to any supported game."""
        for game in ("tmnf", "sc2", "torcs", "beamng", "car_racing"):
            spec = ComboSpec(name="x", track="y", training_params={}, reward_params={}, game=game)
            assert spec.game == game

    def test_combo_spec_round_trip_preserves_game(self):
        """game field survives combo_to_dict / combo_from_dict round-trip."""
        for game in ("tmnf", "sc2", "torcs", "beamng", "car_racing"):
            spec = ComboSpec(name="n", track="t", training_params={}, reward_params={}, game=game)
            recovered = combo_from_dict(combo_to_dict(spec))
            assert recovered.game == game
