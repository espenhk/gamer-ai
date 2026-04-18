"""
Distributed grid-search coordinator.

Hosts an HTTP server that workers poll for work and post results to.

Authentication: every request must carry  Authorization: Bearer <token>.
Missing or wrong token → 401 Unauthorized.

Endpoints:
  GET  /work       → 200 + ComboSpec JSON, or 204 when queue is empty
  POST /result     → 200 ack; stores ExperimentData from worker
  GET  /status     → 200 + JSON summary (queue depth, done, active workers)
  POST /heartbeat  → 200 ack; updates last-seen timestamp for a worker

Usage (called automatically by grid_search.py --distribute):
    from distributed.coordinator import Coordinator
    from distributed.protocol import ComboSpec

    combos = [ComboSpec(name=..., track=..., training_params=..., reward_params=...)]
    coord = Coordinator(combos, token="mysecret", port=5555, heartbeat_timeout=60.0)
    coord.start()
    print("Waiting for workers …")
    all_runs = coord.wait_for_all()   # blocks until all results received
    coord.stop()
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any

from distributed.protocol import (
    ComboSpec,
    combo_to_dict,
    result_from_dict,
    experiment_from_dict,
)

logger = logging.getLogger(__name__)


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Coordinator:
    """Work-queue coordinator for distributed grid search.

    Thread safety: all mutable state is protected by a single lock.
    """

    def __init__(
        self,
        combos: list[ComboSpec],
        token: str,
        port: int = 5555,
        heartbeat_timeout: float = 60.0,
    ) -> None:
        self._token = token
        self._work_queue: deque[ComboSpec] = deque(combos)
        self._known_names: set[str] = {spec.name for spec in combos}
        self._in_progress: dict[str, dict[str, Any]] = {}   # name → info
        self._results: dict[str, Any] = {}                   # name → ExperimentData
        self._total = len(combos)
        self._port = port
        self._heartbeat_timeout = heartbeat_timeout

        self._lock = threading.Lock()
        self._done_event = threading.Event()
        self._server: _ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the HTTP server and heartbeat monitor in background threads."""
        coord = self  # captured by handler class below

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # type: ignore[override]
                if not coord._check_auth(self):
                    return
                if self.path == "/work":
                    coord._handle_get_work(self)
                elif self.path == "/status":
                    coord._handle_get_status(self)
                else:
                    self._send_json(404, {"error": "not found"})

            def do_POST(self) -> None:  # type: ignore[override]
                if not coord._check_auth(self):
                    return
                body = self._read_body()
                if self.path == "/result":
                    coord._handle_post_result(self, body)
                elif self.path == "/heartbeat":
                    coord._handle_post_heartbeat(self, body)
                else:
                    self._send_json(404, {"error": "not found"})

            # ---- helpers ----

            def _send_json(self, code: int, data: Any) -> None:
                payload = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _read_body(self) -> bytes:
                length = int(self.headers.get("Content-Length", 0))
                return self.rfile.read(length)

            def log_message(self, fmt: str, *args: Any) -> None:
                pass  # suppress default access logging

        self._server = _ThreadingHTTPServer(("", self._port), Handler)
        actual_port = self._server.server_address[1]

        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="coord-http"
        )
        self._server_thread.start()

        self._monitor_thread = threading.Thread(
            target=self._heartbeat_monitor, daemon=True, name="coord-monitor"
        )
        self._monitor_thread.start()

        logger.info("Coordinator listening on port %d (%d combo(s) queued)", actual_port, self._total)

    @property
    def port(self) -> int:
        """The port the server is actually listening on (useful when port=0 was requested)."""
        if self._server:
            return self._server.server_address[1]
        return self._port

    def wait_for_all(self) -> list[tuple[str, Any]]:
        """Block until every combo has a result. Returns list of (name, ExperimentData)."""
        if self._total == 0:
            return []
        self._done_event.wait()
        with self._lock:
            return list(self._results.items())

    def stop(self) -> None:
        """Shut down the HTTP server and background threads."""
        self._done_event.set()  # signals monitor thread to exit
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=5)
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _check_auth(self, handler: Any) -> bool:
        """Check Bearer token. Sends 401 and returns False if invalid."""
        import hmac
        auth = handler.headers.get("Authorization", "")
        parts = auth.split(" ", 1)
        token_ok = (
            len(parts) == 2
            and parts[0].lower() == "bearer"
            and hmac.compare_digest(parts[1].strip(), self._token.strip())
        )
        if token_ok:
            return True
        handler.send_response(401)
        handler.send_header("WWW-Authenticate", 'Bearer realm="tmnf-grid"')
        handler.send_header("Content-Length", "0")
        handler.end_headers()
        return False

    # ------------------------------------------------------------------
    # HTTP handlers (called on handler threads)
    # ------------------------------------------------------------------

    def _handle_get_work(self, handler: Any) -> None:
        with self._lock:
            if not self._work_queue:
                handler.send_response(204)
                handler.end_headers()
                return
            spec = self._work_queue.popleft()
            worker_id = handler.headers.get("X-Worker-Id", "unknown")
            self._in_progress[spec.name] = {
                "spec": spec,
                "worker_id": worker_id,
                "last_hb": time.monotonic(),
            }

        logger.info("Dispatched %s → worker %s", spec.name, worker_id)
        handler._send_json(200, combo_to_dict(spec))

    def _handle_post_result(self, handler: Any, body: bytes) -> None:
        worker_id = handler.headers.get("X-Worker-Id", "unknown")
        try:
            payload = result_from_dict(json.loads(body.decode()))
            data = experiment_from_dict(json.loads(payload.data_json))
        except Exception as exc:
            logger.error("Failed to parse result payload: %s", exc)
            handler._send_json(400, {"error": str(exc)})
            return

        done = False
        with self._lock:
            if payload.name not in self._known_names:
                logger.warning("Rejected result for unknown combo %s from worker %s", payload.name, worker_id)
                handler._send_json(400, {"error": f"unknown combo: {payload.name}"})
                return

            if payload.name in self._results:
                logger.info("Ignoring duplicate result for %s from worker %s", payload.name, worker_id)
                handler._send_json(200, {"status": "ok", "ignored": "duplicate"})
                return

            self._in_progress.pop(payload.name, None)
            self._results[payload.name] = data
            progress = len(self._results)
            if progress == self._total:
                done = True

        logger.info(
            "Received result for %s  best_reward=%+.1f  (%d/%d done)",
            payload.name,
            max((s.reward for s in data.greedy_sims), default=float("-inf")),
            progress,
            self._total,
        )
        handler._send_json(200, {"status": "ok"})

        if done:
            self._done_event.set()

    def _handle_post_heartbeat(self, handler: Any, body: bytes) -> None:
        try:
            d = json.loads(body.decode())
            name = d["name"]
            worker_id = d.get("worker_id", "unknown")
        except Exception as exc:
            handler._send_json(400, {"error": str(exc)})
            return

        with self._lock:
            if name in self._in_progress:
                self._in_progress[name]["last_hb"] = time.monotonic()
                self._in_progress[name]["worker_id"] = worker_id

        handler._send_json(200, {"status": "ok"})

    def _handle_get_status(self, handler: Any) -> None:
        with self._lock:
            status = {
                "queued": len(self._work_queue),
                "in_progress": len(self._in_progress),
                "done": len(self._results),
                "total": self._total,
                "workers": [
                    {
                        "name": name,
                        "worker_id": info["worker_id"],
                        "idle_s": round(time.monotonic() - info["last_hb"], 1),
                    }
                    for name, info in self._in_progress.items()
                ],
            }
        handler._send_json(200, status)

    # ------------------------------------------------------------------
    # Heartbeat monitor
    # ------------------------------------------------------------------

    def _heartbeat_monitor(self) -> None:
        check_interval = max(0.5, self._heartbeat_timeout / 2)
        while not self._done_event.wait(timeout=check_interval):
            now = time.monotonic()
            with self._lock:
                stale = [
                    name
                    for name, info in self._in_progress.items()
                    if now - info["last_hb"] > self._heartbeat_timeout
                ]
                for name in stale:
                    info = self._in_progress.pop(name)
                    self._work_queue.appendleft(info["spec"])
                    logger.warning(
                        "Re-queued stale item %s (worker %s, idle %.0fs)",
                        name, info["worker_id"], now - info["last_hb"],
                    )
