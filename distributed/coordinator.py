"""
Distributed grid-search coordinator.

Hosts an HTTP server that workers poll for work and post results to.

Authentication: every request must carry  Authorization: Bearer <token>.
Missing or wrong token → 401 Unauthorized.

Network scope: by default, only LAN/loopback clients are accepted
(private, link-local, or loopback source IPs). Public/non-LAN clients are
rejected with 403 unless allow_non_lan=True.

Endpoints:
  GET  /work       → 200 + ComboSpec JSON, or 204 when queue is empty.
                     Workers may send an optional X-Worker-Game header to
                     receive only work items matching that game.
  POST /result     → 200 ack; stores ExperimentData from worker
  POST /skip       → 200 ack; returns a work item to the front of the queue
                     (used by game-filtered workers that receive an incompatible item)
  GET  /status     → 200 + JSON summary (queue depth, done, active workers)
  POST /heartbeat  → 200 ack; updates last-seen timestamp for a worker
  GET  /monitor    → mobile-friendly web UI with username/password login
  GET  /monitor/api/status → authenticated JSON payload for the web UI

Usage (called automatically by grid_search.py --distribute):
    from distributed.coordinator import Coordinator
    from distributed.protocol import ComboSpec

    combos = [ComboSpec(name=..., track=..., training_params=..., reward_params=...)]
    coord = Coordinator(
        combos,
        token="mysecret",
        port=5555,
        heartbeat_timeout=60.0,
        bind_host="0.0.0.0",
        allow_non_lan=False,
    )
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
import ipaddress
import hmac
import html
import secrets
from collections import deque
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import parse_qs, urlsplit

from distributed.protocol import (
    ComboSpec,
    DEFAULT_GAME,
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
        bind_host: str = "0.0.0.0",
        allow_non_lan: bool = False,
        monitor_username: str = "monitor",
        monitor_password: str | None = None,
    ) -> None:
        self._token = token
        self._work_queue: deque[ComboSpec] = deque(combos)
        self._specs_by_name: dict[str, ComboSpec] = {spec.name: spec for spec in combos}
        self._combo_order: list[str] = [spec.name for spec in combos]
        self._known_names: set[str] = {spec.name for spec in combos}
        self._in_progress: dict[str, dict[str, Any]] = {}   # name → info
        self._results: dict[str, Any] = {}                   # name → ExperimentData
        self._total = len(combos)
        self._port = port
        self._heartbeat_timeout = heartbeat_timeout
        self._bind_host = bind_host
        self._allow_non_lan = allow_non_lan
        self._monitor_username = monitor_username
        self._monitor_password = token if monitor_password is None else monitor_password
        self._monitor_sessions: dict[str, float] = {}
        self._monitor_session_ttl_s = 12 * 60 * 60

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
                if not coord._check_client_allowed(self):
                    return
                path = urlsplit(self.path).path
                if path == "/monitor" or path.startswith("/monitor/"):
                    coord._handle_monitor_get(self, path)
                    return
                if not coord._check_auth(self):
                    return
                if path == "/work":
                    coord._handle_get_work(self)
                elif path == "/status":
                    coord._handle_get_status(self)
                else:
                    self._send_json(404, {"error": "not found"})

            def do_POST(self) -> None:  # type: ignore[override]
                if not coord._check_client_allowed(self):
                    return
                path = urlsplit(self.path).path
                if path == "/monitor/login" or path == "/monitor/logout":
                    body = self._read_body()
                    coord._handle_monitor_post(self, path, body)
                    return
                if not coord._check_auth(self):
                    return
                body = self._read_body()
                if path == "/result":
                    coord._handle_post_result(self, body)
                elif path == "/heartbeat":
                    coord._handle_post_heartbeat(self, body)
                elif path == "/skip":
                    coord._handle_post_skip(self, body)
                else:
                    self._send_json(404, {"error": "not found"})

            # ---- helpers ----

            def _send_json(self, code: int, data: Any) -> None:
                payload = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def _send_html(
                self,
                code: int,
                content: str,
                extra_headers: dict[str, str] | None = None,
            ) -> None:
                payload = content.encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                for key, value in (extra_headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(payload)

            def _redirect(
                self,
                location: str,
                extra_headers: dict[str, str] | None = None,
            ) -> None:
                self.send_response(303)
                self.send_header("Location", location)
                self.send_header("Content-Length", "0")
                self.send_header("Cache-Control", "no-store")
                for key, value in (extra_headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()

            def _read_body(self) -> bytes:
                length = int(self.headers.get("Content-Length", 0))
                return self.rfile.read(length)

            def log_message(self, fmt: str, *args: Any) -> None:
                pass  # suppress default access logging

        self._server = _ThreadingHTTPServer((self._bind_host, self._port), Handler)
        actual_port = self._server.server_address[1]

        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="coord-http"
        )
        self._server_thread.start()

        self._monitor_thread = threading.Thread(
            target=self._heartbeat_monitor, daemon=True, name="coord-monitor"
        )
        self._monitor_thread.start()

        logger.info(
            "Coordinator listening on %s:%d (%d combo(s) queued, lan_only=%s)",
            self._bind_host,
            actual_port,
            self._total,
            not self._allow_non_lan,
        )

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

    def _check_client_allowed(self, handler: Any) -> bool:
        if self._allow_non_lan:
            return True
        client_ip = handler.client_address[0]
        try:
            addr = ipaddress.ip_address(client_ip)
        except ValueError:
            logger.warning("Rejected non-IP client address: %r", client_ip)
            handler.send_response(403)
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return False
        if addr.is_loopback or addr.is_private or addr.is_link_local:
            return True
        logger.warning("Rejected non-LAN client %s", client_ip)
        handler.send_response(403)
        handler.send_header("Content-Length", "0")
        handler.end_headers()
        return False

    def _check_auth(self, handler: Any) -> bool:
        """Check Bearer token. Sends 401 and returns False if invalid."""
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

    def _build_status_payload(self) -> dict[str, Any]:
        with self._lock:
            workers = [
                {
                    "name": name,
                    "worker_id": info["worker_id"],
                    "idle_s": round(max(0.0, time.monotonic() - info["last_hb"]), 1),
                }
                for name, info in self._in_progress.items()
            ]
            runs = []
            for name in self._combo_order:
                spec = self._specs_by_name[name]
                if name in self._results:
                    data = self._results[name]
                    best_reward = max(
                        (sim.reward for sim in data.greedy_sims),
                        default=None,
                    )
                    runs.append(
                        {
                            "name": name,
                            "state": "done",
                            "worker_id": None,
                            "idle_s": None,
                            "game": spec.game,
                            "track": spec.track,
                            "policy_type": spec.training_params.get("policy_type"),
                            "best_reward": best_reward,
                        }
                    )
                    continue
                if name in self._in_progress:
                    info = self._in_progress[name]
                    runs.append(
                        {
                            "name": name,
                            "state": "in_progress",
                            "worker_id": info["worker_id"],
                            "idle_s": round(max(0.0, time.monotonic() - info["last_hb"]), 1),
                            "game": spec.game,
                            "track": spec.track,
                            "policy_type": spec.training_params.get("policy_type"),
                            "best_reward": None,
                        }
                    )
                    continue
                runs.append(
                    {
                        "name": name,
                        "state": "queued",
                        "worker_id": None,
                        "idle_s": None,
                        "game": spec.game,
                        "track": spec.track,
                        "policy_type": spec.training_params.get("policy_type"),
                        "best_reward": None,
                    }
                )
            return {
                "queued": len(self._work_queue),
                "in_progress": len(self._in_progress),
                "done": len(self._results),
                "total": self._total,
                "workers": workers,
                "runs": runs,
            }

    def _monitor_cookie_header(self, session_id: str, expires_now: bool = False) -> str:
        max_age = 0 if expires_now else self._monitor_session_ttl_s
        value = "" if expires_now else session_id
        return (
            f"gamer_ai_monitor_session={value}; Path=/monitor; HttpOnly; "
            f"SameSite=Lax; Max-Age={max_age}"
        )

    def _prune_monitor_sessions(self) -> None:
        now = time.monotonic()
        with self._lock:
            expired = [sid for sid, expiry in self._monitor_sessions.items() if expiry <= now]
            for sid in expired:
                self._monitor_sessions.pop(sid, None)

    def _monitor_session_id(self, handler: Any) -> str | None:
        raw_cookie = handler.headers.get("Cookie", "")
        if not raw_cookie:
            return None
        cookie = SimpleCookie()
        try:
            cookie.load(raw_cookie)
        except Exception:
            return None
        morsel = cookie.get("gamer_ai_monitor_session")
        if morsel is None:
            return None
        return morsel.value

    def _has_monitor_session(self, handler: Any) -> bool:
        self._prune_monitor_sessions()
        session_id = self._monitor_session_id(handler)
        if not session_id:
            return False
        now = time.monotonic()
        with self._lock:
            expiry = self._monitor_sessions.get(session_id)
            if expiry is None or expiry <= now:
                self._monitor_sessions.pop(session_id, None)
                return False
            self._monitor_sessions[session_id] = now + self._monitor_session_ttl_s
            return True

    def _render_monitor_login(self, error: str | None = None) -> str:
        error_html = (
            f'<p class="error">{html.escape(error)}</p>' if error else ""
        )
        username = html.escape(self._monitor_username, quote=True)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>gamer-ai run monitor</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #f4f7fb;
      color: #142033;
      padding: 16px;
    }}
    .card {{
      width: min(100%, 420px);
      background: #fff;
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(20, 32, 51, 0.12);
      padding: 24px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 1.4rem; }}
    p {{ margin: 0 0 16px; color: #4f5d75; }}
    label {{ display: block; margin: 12px 0 6px; font-weight: 600; }}
    input {{
      box-sizing: border-box;
      width: 100%;
      padding: 14px;
      border: 1px solid #c9d2e3;
      border-radius: 12px;
      font-size: 1rem;
    }}
    button {{
      width: 100%;
      margin-top: 16px;
      padding: 14px;
      border: 0;
      border-radius: 12px;
      background: #2563eb;
      color: white;
      font-size: 1rem;
      font-weight: 700;
    }}
    .error {{
      margin: 0 0 12px;
      color: #b91c1c;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <main class="card">
    <h1>Run monitor login</h1>
    <p>Sign in to view queued, active, and completed distributed runs from your phone.</p>
    {error_html}
    <form method="post" action="/monitor/login">
      <label for="username">Username</label>
      <input id="username" name="username" autocomplete="username" value="{username}" required>
      <label for="password">Password</label>
      <input id="password" name="password" type="password" autocomplete="current-password" required>
      <button type="submit">Open monitor</button>
    </form>
  </main>
</body>
</html>"""

    def _render_monitor_app(self) -> str:
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>gamer-ai run monitor</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    body {
      margin: 0;
      background: #eef3f9;
      color: #152238;
    }
    .wrap {
      max-width: 720px;
      margin: 0 auto;
      padding: 16px;
    }
    .card {
      background: #fff;
      border-radius: 18px;
      box-shadow: 0 8px 24px rgba(20, 32, 51, 0.10);
      padding: 16px;
      margin-bottom: 14px;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    h1, h2 {
      margin: 0 0 8px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .stat {
      background: #f7f9fc;
      border-radius: 14px;
      padding: 12px;
      text-align: center;
    }
    .stat strong {
      display: block;
      font-size: 1.3rem;
    }
    select, button {
      min-height: 44px;
      border-radius: 12px;
      border: 1px solid #c9d2e3;
      font-size: 1rem;
    }
    select {
      width: 100%;
      padding: 10px 12px;
      background: white;
    }
    button {
      padding: 10px 14px;
      background: #2563eb;
      color: white;
      border: 0;
      font-weight: 700;
    }
    .subtle { color: #4f5d75; }
    .badge {
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: 700;
      text-transform: capitalize;
    }
    .queued { background: #e5e7eb; color: #374151; }
    .in_progress { background: #dbeafe; color: #1d4ed8; }
    .done { background: #dcfce7; color: #15803d; }
    dl {
      display: grid;
      grid-template-columns: minmax(110px, auto) 1fr;
      gap: 8px 12px;
      margin: 14px 0 0;
    }
    dt { font-weight: 700; }
    dd { margin: 0; }
    ul {
      list-style: none;
      padding: 0;
      margin: 12px 0 0;
    }
    li {
      padding: 10px 0;
      border-top: 1px solid #e5e7eb;
    }
    .actions {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    @media (max-width: 560px) {
      .stats {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <div class="topbar">
        <div>
          <h1>Run monitor</h1>
          <div class="subtle" id="updated-at">Loading…</div>
        </div>
        <form method="post" action="/monitor/logout">
          <button type="submit">Log out</button>
        </form>
      </div>
      <div class="stats">
        <div class="stat"><span class="subtle">Queued</span><strong id="queued-count">0</strong></div>
        <div class="stat"><span class="subtle">Running</span><strong id="active-count">0</strong></div>
        <div class="stat"><span class="subtle">Done</span><strong id="done-count">0</strong></div>
        <div class="stat"><span class="subtle">Total</span><strong id="total-count">0</strong></div>
      </div>
    </section>
    <section class="card">
      <h2>Select run</h2>
      <select id="run-select" aria-label="Select run"></select>
      <div id="run-detail" class="subtle" style="margin-top: 12px;">No runs available.</div>
    </section>
    <section class="card">
      <h2>Active workers</h2>
      <ul id="worker-list"><li class="subtle">No workers active.</li></ul>
    </section>
  </div>
  <script>
    let latestStatus = null;
    let selectedRun = null;

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[ch]);
    }

    function fmt(value) {
      return value == null ? "—" : value;
    }

    function fmtReward(value) {
      return value == null ? "—" : Number(value).toFixed(1);
    }

    function renderWorkers(workers) {
      const list = document.getElementById("worker-list");
      if (!workers.length) {
        list.innerHTML = '<li class="subtle">No workers active.</li>';
        return;
      }
      list.innerHTML = workers.map((worker) =>
        `<li><strong>${esc(worker.worker_id)}</strong><br><span class="subtle">${esc(worker.name)} · idle ${esc(worker.idle_s)}s</span></li>`
      ).join("");
    }

    function renderRunDetail() {
      const detail = document.getElementById("run-detail");
      const runs = latestStatus?.runs || [];
      const run = runs.find((item) => item.name === selectedRun);
      if (!run) {
        detail.innerHTML = '<p class="subtle">No runs available.</p>';
        return;
      }
      detail.innerHTML = `
        <span class="badge ${esc(run.state)}">${esc(run.state.replace("_", " "))}</span>
        <dl>
          <dt>Name</dt><dd>${esc(run.name)}</dd>
          <dt>Game</dt><dd>${esc(fmt(run.game))}</dd>
          <dt>Track / map</dt><dd>${esc(fmt(run.track))}</dd>
          <dt>Policy</dt><dd>${esc(fmt(run.policy_type))}</dd>
          <dt>Computer</dt><dd>${esc(fmt(run.worker_id))}</dd>
          <dt>Idle seconds</dt><dd>${esc(fmt(run.idle_s))}</dd>
          <dt>Best reward</dt><dd>${esc(fmtReward(run.best_reward))}</dd>
        </dl>
      `;
    }

    function renderRuns(runs) {
      const select = document.getElementById("run-select");
      if (!runs.length) {
        selectedRun = null;
        select.innerHTML = "";
        renderRunDetail();
        return;
      }
      if (!runs.some((run) => run.name === selectedRun)) {
        selectedRun = runs[0].name;
      }
      select.innerHTML = runs.map((run) => {
        const label = `${run.name} — ${run.state.replace("_", " ")}`;
        const selected = run.name === selectedRun ? " selected" : "";
        return `<option value="${esc(run.name)}"${selected}>${esc(label)}</option>`;
      }).join("");
      renderRunDetail();
    }

    async function refreshStatus() {
      const res = await fetch("/monitor/api/status", { credentials: "same-origin" });
      if (res.status === 401) {
        window.location = "/monitor";
        return;
      }
      latestStatus = await res.json();
      document.getElementById("queued-count").textContent = latestStatus.queued;
      document.getElementById("active-count").textContent = latestStatus.in_progress;
      document.getElementById("done-count").textContent = latestStatus.done;
      document.getElementById("total-count").textContent = latestStatus.total;
      document.getElementById("updated-at").textContent =
        `Updated ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;
      renderRuns(latestStatus.runs || []);
      renderWorkers(latestStatus.workers || []);
    }

    document.getElementById("run-select").addEventListener("change", (event) => {
      selectedRun = event.target.value;
      renderRunDetail();
    });

    refreshStatus();
    setInterval(refreshStatus, 5000);
  </script>
</body>
</html>"""

    def _handle_monitor_get(self, handler: Any, path: str) -> None:
        if path == "/monitor/api/status":
            if not self._has_monitor_session(handler):
                handler._send_json(401, {"error": "login required"})
                return
            handler._send_json(200, self._build_status_payload())
            return
        if path in ("/monitor", "/monitor/"):
            if self._has_monitor_session(handler):
                handler._send_html(200, self._render_monitor_app())
            else:
                handler._send_html(200, self._render_monitor_login())
            return
        handler._send_json(404, {"error": "not found"})

    def _handle_monitor_post(self, handler: Any, path: str, body: bytes) -> None:
        if path == "/monitor/logout":
            session_id = self._monitor_session_id(handler)
            if session_id:
                with self._lock:
                    self._monitor_sessions.pop(session_id, None)
            handler._redirect(
                "/monitor",
                extra_headers={
                    "Set-Cookie": self._monitor_cookie_header("", expires_now=True),
                },
            )
            return
        try:
            form = parse_qs(body.decode("utf-8"), keep_blank_values=True)
        except UnicodeDecodeError:
            handler._send_html(400, self._render_monitor_login("Malformed login body."))
            return
        username = form.get("username", [""])[0]
        password = form.get("password", [""])[0]
        if (
            hmac.compare_digest(username, self._monitor_username)
            and hmac.compare_digest(password, self._monitor_password)
        ):
            session_id = secrets.token_urlsafe(24)
            with self._lock:
                self._monitor_sessions[session_id] = (
                    time.monotonic() + self._monitor_session_ttl_s
                )
            handler._redirect(
                "/monitor",
                extra_headers={"Set-Cookie": self._monitor_cookie_header(session_id)},
            )
            return
        handler._send_html(401, self._render_monitor_login("Invalid username or password."))

    def _handle_get_work(self, handler: Any) -> None:
        worker_game = handler.headers.get("X-Worker-Game", None)
        with self._lock:
            if not self._work_queue:
                handler.send_response(204)
                handler.end_headers()
                return

            if worker_game:
                # Find the first item in the queue that matches the worker's game.
                spec = None
                skipped: list[Any] = []
                while self._work_queue:
                    candidate = self._work_queue.popleft()
                    if getattr(candidate, "game", DEFAULT_GAME) == worker_game:
                        spec = candidate
                        break
                    skipped.append(candidate)
                # Put non-matching items back at the front (preserving order).
                for item in reversed(skipped):
                    self._work_queue.appendleft(item)
                if spec is None:
                    # No matching work for this game yet.
                    handler.send_response(204)
                    handler.end_headers()
                    return
            else:
                spec = self._work_queue.popleft()

            worker_id = handler.headers.get("X-Worker-Id", "unknown")
            self._in_progress[spec.name] = {
                "spec": spec,
                "worker_id": worker_id,
                # Offset last_hb into the future by one timeout so the worker
                # has 2×timeout to send its first heartbeat without being
                # re-queued.  Once the first heartbeat arrives it resets
                # last_hb to the real receive time, giving normal 1×timeout
                # tracking for all subsequent heartbeats.
                "last_hb": time.monotonic() + self._heartbeat_timeout,
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
                self._done_event.set()

        logger.info(
            "Received result for %s  best_reward=%+.1f  (%d/%d done)",
            payload.name,
            max((s.reward for s in data.greedy_sims), default=float("-inf")),
            progress,
            self._total,
        )
        handler._send_json(200, {"status": "ok"})

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

    def _handle_post_skip(self, handler: Any, body: bytes) -> None:
        """Return a work item to the front of the queue.

        Called by game-filtered workers that received an item for a different
        game (should not happen with server-side filtering, but kept as a
        fallback for older coordinator versions or race conditions).
        """
        worker_id = handler.headers.get("X-Worker-Id", "unknown")
        try:
            d = json.loads(body.decode())
            name = d["name"]
        except Exception as exc:
            handler._send_json(400, {"error": str(exc)})
            return

        with self._lock:
            info = self._in_progress.pop(name, None)
            if info is None:
                # Already re-queued or completed — ignore silently.
                handler._send_json(200, {"status": "ok", "note": "not in progress"})
                return
            self._work_queue.appendleft(info["spec"])

        logger.info("Returned %s to queue (skipped by worker %s)", name, worker_id)
        handler._send_json(200, {"status": "ok"})

    def _handle_get_status(self, handler: Any) -> None:
        handler._send_json(200, self._build_status_payload())

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
