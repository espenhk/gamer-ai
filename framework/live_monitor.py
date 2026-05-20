"""Live training monitor UI for reward components and observations.

Optional Tkinter window that updates at every env.step() call.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_IDX_RE = re.compile(r"^(?P<base>.+)_(?P<idx>\d+)$")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _derive_step_components(
    info: dict,
    reward: float,
    prev_episode_components: dict[str, float] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (step_components, current_episode_components)."""
    step = info.get("step_reward_components")
    if isinstance(step, dict):
        out = {str(k): _safe_float(v) for k, v in step.items()}
        out["total_reward"] = _safe_float(reward)
        current_ep = info.get("episode_reward_components")
        if isinstance(current_ep, dict):
            return out, {str(k): _safe_float(v) for k, v in current_ep.items()}
        return out, dict(prev_episode_components or {})

    current_episode = info.get("episode_reward_components")
    if isinstance(current_episode, dict):
        current = {str(k): _safe_float(v) for k, v in current_episode.items()}
        prev = prev_episode_components or {}
        out = {k: current.get(k, 0.0) - prev.get(k, 0.0) for k in current}
        out["total_reward"] = _safe_float(reward)
        return out, current

    return {"total_reward": _safe_float(reward)}, dict(prev_episode_components or {})


def _rolling_means(
    history: dict[str, deque[float]],
    step_components: dict[str, float],
    window: int,
) -> dict[str, float]:
    for key, value in step_components.items():
        history.setdefault(key, deque(maxlen=window)).append(_safe_float(value))
    return {
        key: (sum(vals) / len(vals)) for key, vals in history.items() if len(vals) > 0
    }


@dataclass
class ObservationGroups:
    xy_pairs: list[tuple[str, str, str]]
    indexed: list[tuple[str, list[int]]]
    quads: list[tuple[str, dict[str, int]]]
    scalar_idxs: list[int]


def _classify_observation_features(obs_names: list[str]) -> ObservationGroups:
    used: set[int] = set()
    by_name = {name: i for i, name in enumerate(obs_names)}

    xy_pairs: list[tuple[str, str, str]] = []
    for x_suf, y_suf in (("_x", "_y"), ("_cx", "_cy")):
        for name in obs_names:
            if not name.endswith(x_suf):
                continue
            base = name[: -len(x_suf)]
            y_name = f"{base}{y_suf}"
            if y_name in by_name:
                xi, yi = by_name[name], by_name[y_name]
                if xi in used or yi in used:
                    continue
                used.update((xi, yi))
                xy_pairs.append((base, name, y_name))

    indexed_map: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, name in enumerate(obs_names):
        if i in used:
            continue
        m = _IDX_RE.match(name)
        if not m:
            continue
        indexed_map[m.group("base")].append((int(m.group("idx")), i))

    indexed: list[tuple[str, list[int]]] = []
    for base, pairs in indexed_map.items():
        if len(pairs) < 2:
            continue
        ordered = [i for _idx, i in sorted(pairs, key=lambda t: t[0])]
        used.update(ordered)
        indexed.append((base, ordered))

    quad_suffixes = ("_NE", "_NW", "_SE", "_SW")
    quads: list[tuple[str, dict[str, int]]] = []
    for name in obs_names:
        if any(name.endswith(s) for s in quad_suffixes):
            for suf in quad_suffixes:
                if not name.endswith(suf):
                    continue
                base = name[: -len(suf)]
                members = {s: f"{base}{s}" for s in quad_suffixes}
                if all(n in by_name for n in members.values()):
                    idx_map = {s: by_name[n] for s, n in members.items()}
                    if any(idx in used for idx in idx_map.values()):
                        continue
                    used.update(idx_map.values())
                    quads.append((base, idx_map))
                break

    scalar_idxs = [i for i in range(len(obs_names)) if i not in used]
    return ObservationGroups(
        xy_pairs=xy_pairs,
        indexed=indexed,
        quads=quads,
        scalar_idxs=scalar_idxs,
    )


class LiveTelemetryMonitor:
    """Small Tkinter dashboard that updates during training."""

    def __init__(
        self,
        obs_names: list[str],
        obs_scales: np.ndarray,
        rolling_window: int = 5,
    ) -> None:
        self._obs_names = list(obs_names)
        self._obs_scales = np.asarray(obs_scales, dtype=np.float32)
        self._rolling_window = max(1, int(rolling_window))
        self._reward_history: dict[str, deque[float]] = {}
        self._prev_episode_components: dict[str, float] = {}
        self._groups = _classify_observation_features(self._obs_names)
        self._step_idx = 0

        self._tk = None
        self._reward_canvas = None
        self._obs_canvas = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        try:
            import tkinter as tk
        except Exception as exc:  # pragma: no cover - env dependent
            logger.warning("Live GUI disabled (tkinter unavailable): %s", exc)
            return

        try:
            self._tk = tk.Tk()
            self._tk.title("gamer-ai live monitor")
            self._tk.geometry("1200x850")
            header = tk.Label(
                self._tk,
                text="Live reward/observation telemetry (rolling mean window=5)",
                anchor="w",
            )
            header.pack(fill="x", padx=8, pady=6)
            self._reward_canvas = tk.Canvas(self._tk, width=1180, height=300, bg="white")
            self._reward_canvas.pack(fill="both", padx=8, pady=4)
            self._obs_canvas = tk.Canvas(self._tk, width=1180, height=500, bg="white")
            self._obs_canvas.pack(fill="both", padx=8, pady=4)
            self._started = True
            self._tk.update_idletasks()
            self._tk.update()
        except Exception as exc:  # pragma: no cover - env dependent
            logger.warning("Live GUI disabled (failed to create window): %s", exc)
            self._tk = None
            self._reward_canvas = None
            self._obs_canvas = None
            self._started = False

    def close(self) -> None:
        if self._tk is None:
            return
        try:
            self._tk.destroy()
        except Exception:  # pragma: no cover - env dependent
            pass
        finally:
            self._tk = None
            self._reward_canvas = None
            self._obs_canvas = None
            self._started = False

    def on_step(self, obs: np.ndarray, reward: float, info: dict) -> None:
        if not self._started or self._tk is None:
            return
        self._step_idx += 1
        step_components, self._prev_episode_components = _derive_step_components(
            info, reward, self._prev_episode_components,
        )
        avg_components = _rolling_means(
            self._reward_history, step_components, self._rolling_window,
        )
        self._draw_reward_panel(avg_components)
        self._draw_observation_panel(np.asarray(obs, dtype=np.float32))
        try:
            self._tk.update_idletasks()
            self._tk.update()
        except Exception:  # pragma: no cover - env dependent
            self.close()

    def _draw_reward_panel(self, avg_components: dict[str, float]) -> None:
        if self._reward_canvas is None:
            return
        c = self._reward_canvas
        c.delete("all")
        c.create_text(10, 10, anchor="nw", text=f"Rewards @ step {self._step_idx}", fill="black")
        if not avg_components:
            c.create_text(10, 35, anchor="nw", text="No reward components yet.", fill="gray")
            return

        items = sorted(avg_components.items(), key=lambda kv: kv[0])
        max_abs = max(1.0, max(abs(v) for _, v in items))
        left = 220
        right = int(c.winfo_width()) - 20
        y = 35
        bar_h = 18
        gap = 8
        zero_x = left + (right - left) // 2
        c.create_line(zero_x, y - 8, zero_x, y + len(items) * (bar_h + gap), fill="#888")
        for name, value in items:
            frac = value / max_abs
            px = int((right - left) * min(1.0, abs(frac)) / 2)
            x0, x1 = (zero_x, zero_x + px) if frac >= 0 else (zero_x - px, zero_x)
            color = "#2e8b57" if value >= 0 else "#d9534f"
            c.create_rectangle(x0, y, x1, y + bar_h, fill=color, outline=color)
            c.create_text(10, y + 2, anchor="nw", text=f"{name:>24}")
            c.create_text(right + 5, y + 2, anchor="nw", text=f"{value:+.3f}")
            y += bar_h + gap

    def _draw_observation_panel(self, obs: np.ndarray) -> None:
        if self._obs_canvas is None:
            return
        c = self._obs_canvas
        c.delete("all")
        c.create_text(10, 10, anchor="nw", text="Observations (live)", fill="black")
        y = 34

        # Scalars: show first 12 most informative by abs(normalized value).
        norm = np.divide(
            obs,
            np.where(np.abs(self._obs_scales) < 1e-9, 1.0, self._obs_scales),
            out=np.zeros_like(obs, dtype=np.float32),
        )
        scalar_idxs = self._groups.scalar_idxs[:]
        scalar_idxs.sort(key=lambda i: abs(float(norm[i])), reverse=True)
        scalar_idxs = scalar_idxs[:12]
        c.create_text(10, y, anchor="nw", text="Scalar bars", fill="#333")
        y += 20
        left = 210
        right = int(c.winfo_width()) - 20
        zero_x = left + (right - left) // 2
        c.create_line(zero_x, y - 6, zero_x, y + len(scalar_idxs) * 20, fill="#aaa")
        for idx in scalar_idxs:
            val = float(obs[idx])
            nval = max(-1.0, min(1.0, float(norm[idx])))
            px = int((right - left) * abs(nval) / 2)
            x0, x1 = (zero_x, zero_x + px) if nval >= 0 else (zero_x - px, zero_x)
            color = "#4c78a8" if nval >= 0 else "#f58518"
            c.create_rectangle(x0, y, x1, y + 14, fill=color, outline=color)
            c.create_text(10, y, anchor="nw", text=f"{self._obs_names[idx]}")
            c.create_text(right + 5, y, anchor="nw", text=f"{val:+.3f}")
            y += 20

        # XY pairs (recommendation: point/arrow-like features).
        if self._groups.xy_pairs:
            y += 4
            c.create_text(10, y, anchor="nw", text="XY pairs (recommended directional view)", fill="#333")
            y += 20
            for base, xn, yn in self._groups.xy_pairs[:6]:
                xi = self._obs_names.index(xn)
                yi = self._obs_names.index(yn)
                xv = float(norm[xi])
                yv = float(norm[yi])
                c.create_text(10, y, anchor="nw", text=f"{base}: ({obs[xi]:+.2f}, {obs[yi]:+.2f})")
                box_l, box_t = 260, y
                box_s = 44
                c.create_rectangle(box_l, box_t, box_l + box_s, box_t + box_s, outline="#888")
                cx, cy = box_l + box_s / 2, box_t + box_s / 2
                c.create_line(cx, cy, cx + 16 * xv, cy - 16 * yv, fill="#2e8b57", width=2)
                y += 52

        # Indexed vectors (recommendation: strip heat).
        if self._groups.indexed:
            y += 4
            c.create_text(10, y, anchor="nw", text="Indexed vectors (recommended strip heatmap)", fill="#333")
            y += 20
            for base, idxs in self._groups.indexed[:4]:
                c.create_text(10, y + 2, anchor="nw", text=f"{base}")
                x = 260
                for idx in idxs[:24]:
                    v = float(norm[idx])
                    intensity = int(max(0, min(255, 127 + 120 * v)))
                    color = f"#{intensity:02x}64{(255-intensity):02x}"
                    c.create_rectangle(x, y, x + 14, y + 14, fill=color, outline="")
                    x += 16
                y += 22

        # Quadrants (recommendation: 2x2 grid heat).
        if self._groups.quads:
            y += 4
            c.create_text(10, y, anchor="nw", text="Quadrants (recommended 2x2 grid)", fill="#333")
            y += 20
            for base, idx_map in self._groups.quads[:3]:
                c.create_text(10, y, anchor="nw", text=base)
                grid_x, grid_y = 260, y
                cell = 20
                mapping = {
                    "_NW": (0, 0),
                    "_NE": (1, 0),
                    "_SW": (0, 1),
                    "_SE": (1, 1),
                }
                for suf, (gx, gy) in mapping.items():
                    idx = idx_map[suf]
                    v = float(norm[idx])
                    intensity = int(max(0, min(255, 127 + 120 * v)))
                    color = f"#{intensity:02x}{intensity:02x}c8"
                    x0 = grid_x + gx * cell
                    y0 = grid_y + gy * cell
                    c.create_rectangle(x0, y0, x0 + cell, y0 + cell, fill=color, outline="#666")
                y += 48


def make_live_monitor(training_params: dict, obs_spec: Any) -> LiveTelemetryMonitor | None:
    """Create a live monitor if enabled; otherwise return None."""
    enabled = bool(training_params.get("live_gui", False))
    if not enabled:
        return None
    names = list(getattr(obs_spec, "names", []))
    scales = np.asarray(getattr(obs_spec, "scales", np.ones(len(names))), dtype=np.float32)
    if not names:
        logger.warning("live_gui requested but observation spec is empty; disabling.")
        return None
    monitor = LiveTelemetryMonitor(names, scales, rolling_window=5)
    monitor.start()
    return monitor
