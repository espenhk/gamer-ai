"""
Distributed grid-search worker.

Polls the coordinator for work items, runs train_rl() locally (requires a live
TMInterface game session), and posts results back.

Authentication: every request carries  Authorization: Bearer <token>.
Supply the token via --token or the TMNF_GRID_TOKEN environment variable.

Usage:
    python -m distributed.worker --coordinator http://<host>:5555 --token <secret>

Options:
    --coordinator URL    Coordinator base URL (required)
    --token SECRET       Shared secret; falls back to TMNF_GRID_TOKEN env var
    --worker-id ID       Identifier shown in coordinator logs (default: hostname)
    --heartbeat-interval Seconds between heartbeat POSTs during a run (default: 15)
    --no-interrupt       Pass --no-interrupt to each train_rl call
    --re-initialize      Re-initialize weights for every experiment
    --log-level LEVEL    DEBUG/INFO/WARNING/ERROR (default: INFO)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request

from distributed.protocol import (
    ComboSpec,
    ResultPayload,
    combo_from_dict,
    experiment_to_json,
    result_to_dict,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only — no requests dependency)
# ---------------------------------------------------------------------------

def _http_get(url: str, headers: dict | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise PermissionError("401 Unauthorized — check your --token") from exc
        return exc.code, exc.read()


def _http_post(url: str, body: dict | str, headers: dict | None = None) -> tuple[int, bytes]:
    if isinstance(body, dict):
        data = json.dumps(body).encode()
    else:
        data = body.encode() if isinstance(body, str) else body
    h = {"Content-Type": "application/json", "Content-Length": str(len(data))}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise PermissionError("401 Unauthorized — check your --token") from exc
        return exc.code, exc.read()


# ---------------------------------------------------------------------------
# Heartbeat thread
# ---------------------------------------------------------------------------

class _HeartbeatThread(threading.Thread):
    """Sends POST /heartbeat to the coordinator at a fixed interval."""

    def __init__(self, coordinator_url: str, combo_name: str, worker_id: str,
                 interval: float, auth_headers: dict) -> None:
        super().__init__(daemon=True, name="worker-heartbeat")
        self._url = f"{coordinator_url.rstrip('/')}/heartbeat"
        self._name = combo_name
        self._worker_id = worker_id
        self._interval = interval
        self._auth_headers = auth_headers
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(timeout=self._interval):
            try:
                _http_post(self._url, {"name": self._name, "worker_id": self._worker_id},
                           headers=self._auth_headers)
            except Exception as exc:
                logger.debug("Heartbeat failed: %s", exc)

    def stop(self) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Core worker loop
# ---------------------------------------------------------------------------

def run_worker(
    coordinator_url: str,
    token: str,
    worker_id: str,
    heartbeat_interval: float,
    no_interrupt: bool,
    re_initialize: bool,
) -> None:
    """Poll for work, run experiments, post results. Returns when queue is empty."""
    base = coordinator_url.rstrip("/")
    auth_headers = {
        "Authorization": f"Bearer {token}",
        "X-Worker-Id": worker_id,
    }

    from framework.game_adapter import GAME_ADAPTERS
    from framework.run_config import RunConfig
    from framework.training import train_rl

    from grid_search import _build_policy_params, _setup_experiment_dir

    logger.info("Worker %s connecting to %s", worker_id, base)

    while True:
        # --- fetch work item ---
        try:
            status, body = _http_get(f"{base}/work", headers=auth_headers)
        except Exception as exc:
            logger.error("GET /work failed: %s — retrying in 5 s", exc)
            time.sleep(5)
            continue

        if status == 204:
            # Queue is empty, but items may still be in-progress (assigned to other workers
            # that could crash). Poll /status until done == total before exiting so this
            # worker remains available to pick up any re-queued stale items.
            try:
                s_status, s_body = _http_get(f"{base}/status", headers=auth_headers)
                if s_status == 200:
                    s = json.loads(s_body.decode())
                    if s.get("done", 0) >= s.get("total", 0):
                        logger.info("Grid complete — worker %s exiting", worker_id)
                        return
                    logger.info(
                        "Queue empty but grid not complete (%d/%d done) — waiting for re-queue",
                        s.get("done", 0), s.get("total", 0),
                    )
                    time.sleep(5)
                    continue
            except Exception as exc:
                logger.debug("GET /status failed: %s", exc)
            logger.info("Queue empty — worker %s exiting", worker_id)
            return

        if status != 200:
            logger.error("GET /work returned %d — retrying in 5 s", status)
            time.sleep(5)
            continue

        spec: ComboSpec = combo_from_dict(json.loads(body.decode()))
        logger.info("Running experiment: %s", spec.name)

        # --- prepare local experiment dir ---
        track = spec.track
        t = spec.training_params
        r = spec.reward_params
        game_name = getattr(spec, "game", "tmnf")
        adapter = GAME_ADAPTERS[game_name]()
        track_override = track if game_name != "tmnf" else None
        experiment_dir, weights_file, reward_cfg_file = _setup_experiment_dir(
            adapter, spec.name, t, r, track_override
        )

        # --- start heartbeat ---
        hb = _HeartbeatThread(base, spec.name, worker_id, heartbeat_interval, auth_headers)
        hb.start()

        # --- run training ---
        t_with_pp = dict(t)
        t_with_pp["policy_params"] = _build_policy_params(t)
        try:
            game_spec = adapter.build_game_spec(
                spec.name, experiment_dir, weights_file, reward_cfg_file,
                t_with_pp, track_override,
            )
            data = train_rl(
                game=game_spec,
                config=RunConfig.from_training_params(t_with_pp),
                probe=adapter.build_probe(t_with_pp),
                warmup=adapter.build_warmup(t_with_pp),
                extras=adapter.build_extras(weights_file, t_with_pp, re_initialize),
                no_interrupt=no_interrupt,
                re_initialize=re_initialize,
            )
        except Exception as exc:
            logger.error("train_rl failed for %s: %s", spec.name, exc, exc_info=True)
            hb.stop()
            hb.join(timeout=2)
            continue
        finally:
            hb.stop()
            hb.join(timeout=2)

        # --- save local results ---
        if game_spec.save_results_fn is not None:
            game_spec.save_results_fn(data, results_dir=f"{experiment_dir}/results")
        best = max((s.reward for s in data.greedy_sims), default=float("-inf"))
        logger.info("Finished %s  best_reward=%+.1f", spec.name, best)

        # --- post result to coordinator ---
        payload = ResultPayload(name=spec.name, data_json=experiment_to_json(data))
        try:
            status, _ = _http_post(f"{base}/result", json.dumps(result_to_dict(payload)),
                                   headers=auth_headers)
            if status != 200:
                logger.error("POST /result returned %d for %s", status, spec.name)
        except Exception as exc:
            logger.error("POST /result failed for %s: %s", spec.name, exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distributed grid-search worker — polls coordinator for work"
    )
    parser.add_argument("--coordinator", required=True, metavar="URL",
                        help="Coordinator base URL, e.g. http://192.168.1.10:5555")
    parser.add_argument("--token", default=None, metavar="SECRET",
                        help="Shared secret; falls back to TMNF_GRID_TOKEN env var")
    parser.add_argument("--worker-id", default=socket.gethostname(), metavar="ID",
                        help="Identifier shown in coordinator status (default: hostname)")
    parser.add_argument("--heartbeat-interval", type=float, default=15.0, metavar="S",
                        help="Seconds between heartbeat POSTs (default: 15)")
    parser.add_argument("--no-interrupt", action="store_true",
                        help="Skip all 'Press Enter' prompts")
    parser.add_argument("--re-initialize", action="store_true",
                        help="Ignore existing weights files and restart from scratch")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    token = args.token or os.environ.get("TMNF_GRID_TOKEN")
    if not token:
        parser.error("No token provided. Use --token <secret> or set TMNF_GRID_TOKEN env var.")

    run_worker(
        coordinator_url=args.coordinator,
        token=token,
        worker_id=args.worker_id,
        heartbeat_interval=args.heartbeat_interval,
        no_interrupt=args.no_interrupt,
        re_initialize=args.re_initialize,
    )


if __name__ == "__main__":
    main()
