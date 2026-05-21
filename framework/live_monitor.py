"""Live training monitor UI for reward components and observations.

Optional Tkinter window that updates at every env.step() call.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_IDX_RE = re.compile(r"^(?P<base>.+)_(?P<idx>\d+)$")
_MID_IDX_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)_(?P<suffix>.+)$")

# Logical display order for well-known reward-component keys. Keys not in this
# list are appended alphabetically after those that are.
_REWARD_ORDER = [
    "total_reward",
    "progress_weight",
    "speed_weight",
    "centerline_weight",
    "finish_bonus",
    "finish_time_weight",
    "accel_bonus",
    "airborne_penalty",
    "lidar_wall_weight",
    "step_penalty",
    "crash_threshold_m",
    # SC2
    "score_weight",
    "win_bonus",
    "loss_penalty",
    "economy_weight",
    "idle_penalty",
    "idle_bonus",
    "attack_bonus",
    "click_attack_bonus",
    "move_exploration_bonus",
    "move_repeat_penalty",
    "move_self_penalty",
    "unit_loss_penalty",
    "damage_taken_penalty",
    "passive_under_fire_penalty",
]
_REWARD_ORDER_INDEX = {name: idx for idx, name in enumerate(_REWARD_ORDER)}


def _reward_sort_key(name: str) -> tuple[int, str]:
    return (_REWARD_ORDER_INDEX.get(name, len(_REWARD_ORDER)), name)


def _split_into_columns_preserving_order(items: list[Any], n_cols: int) -> list[list[Any]]:
    n_cols = max(1, int(n_cols))
    if not items:
        return [[]]
    if n_cols == 1:
        return [list(items)]
    rows = int(math.ceil(len(items) / n_cols))
    return [list(items[i: i + rows]) for i in range(0, len(items), rows)]


def _observation_column_count(canvas_width: int) -> int:
    return 4 if int(canvas_width) >= 1200 else 3


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
        if m:
            indexed_map[m.group("base")].append((int(m.group("idx")), i))
            continue
        m_mid = _MID_IDX_RE.match(name)
        if m_mid:
            base = f"{m_mid.group('prefix')}_{m_mid.group('suffix')}"
            indexed_map[base].append((int(m_mid.group("idx")), i))

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


def _fmt_action(action: Any) -> str:
    """Format an action value for display in the actions list."""
    if action is None:
        return "—"
    try:
        arr = np.asarray(action, dtype=np.float32).flatten()
        if len(arr) == 4 and np.isfinite(arr[:4]).all():
            try:
                from games.sc2.actions import FUNCTION_IDS
            except Exception:
                FUNCTION_IDS = {}
            fn_idx = int(round(float(arr[0])))
            fn_name = FUNCTION_IDS.get(fn_idx)
            if fn_name == "no_op":
                return ""
            if fn_name is not None:
                x = int(round(float(np.clip(arr[1], 0.0, 1.0) * 63.0)))
                y = int(round(float(np.clip(arr[2], 0.0, 1.0) * 63.0)))
                base_name = fn_name.replace("_screen", "").replace("_minimap", "")
                base = base_name.replace("_quick", "").replace("_", " ").lower()
                if fn_name == "select_point":
                    return f"select screen: ({x},{y})"
                if fn_name.endswith("_screen"):
                    return f"{base} screen: ({x},{y})"
                if fn_name.endswith("_minimap"):
                    return f"{base} minimap: ({x},{y})"
                return base
        if len(arr) == 3 and np.isfinite(arr[:3]).all():
            effective_zero = 0.01
            steer = float(np.clip(arr[0], -1.0, 1.0))
            accel_raw = float(np.clip(arr[1], 0.0, 1.0))
            brake_raw = float(np.clip(arr[2], 0.0, 1.0))
            accel_active = accel_raw > effective_zero
            brake_active = brake_raw > effective_zero
            accel = int(round(accel_raw * 100)) if accel_active else 0
            brake = int(round(brake_raw * 100)) if brake_active else 0
            steer_pct = int(round(abs(steer) * 100))
            if steer_pct == 0:
                steer_label = "straight"
            elif steer < 0:
                steer_label = f"left {steer_pct}%"
            else:
                steer_label = f"right {steer_pct}%"
            if accel_active and not brake_active:
                pedal_label = f"accel {accel}%"
            elif brake_active and not accel_active:
                pedal_label = f"brake {brake}%"
            else:
                pedal_label = f"accel {accel}% / brake {brake}%"
            return f"{pedal_label} | steer {steer_label}"
        return "[" + ", ".join(f"{v:+.2f}" for v in arr[:6]) + ("]" if len(arr) <= 6 else "…]")
    except Exception:
        return str(action)[:40]


class LiveTelemetryMonitor:
    """Compact Tkinter dashboard that updates during training."""

    _WINDOW_W = 960
    _WINDOW_H = 720

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
        self._last_actions: deque[tuple[int, Any]] = deque(maxlen=10)

        self._tk = None
        self._reward_canvas = None
        self._obs_canvas = None
        self._action_canvas = None
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
            self._tk.geometry(f"{self._WINDOW_W}x{self._WINDOW_H}")
            self._tk.resizable(True, True)

            header = tk.Label(
                self._tk,
                text=f"Live reward/observation telemetry (rolling mean window={self._rolling_window})",
                anchor="w",
                font=("TkDefaultFont", 9),
            )
            header.pack(fill="x", padx=6, pady=3)

            # Top row: rewards (left) + actions (right)
            top_frame = tk.Frame(self._tk)
            top_frame.pack(fill="both", expand=False, padx=6, pady=2)

            reward_label = tk.Label(top_frame, text="Rewards (rolling avg)", anchor="w", font=("TkDefaultFont", 8, "bold"))
            reward_label.grid(row=0, column=0, sticky="w")
            action_label = tk.Label(top_frame, text="Last 10 actions", anchor="w", font=("TkDefaultFont", 8, "bold"))
            action_label.grid(row=0, column=1, sticky="w", padx=(12, 0))

            top_frame.columnconfigure(0, weight=3)
            top_frame.columnconfigure(1, weight=1)

            self._reward_canvas, _rf = self._make_scrollable_canvas(top_frame, tk, height=260, bg="white")
            _rf.grid(row=1, column=0, sticky="nsew")

            self._action_canvas, _af = self._make_scrollable_canvas(top_frame, tk, height=260, bg="#f8f8f8")
            _af.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

            top_frame.rowconfigure(1, weight=1)

            obs_label = tk.Label(self._tk, text="Observations (live)", anchor="w", font=("TkDefaultFont", 8, "bold"))
            obs_label.pack(fill="x", padx=6)

            self._obs_canvas, _of = self._make_scrollable_canvas(self._tk, tk, height=360, bg="white")
            _of.pack(fill="both", expand=True, padx=6, pady=(0, 4))

            self._started = True
            self._tk.update_idletasks()
            self._tk.update()
        except Exception as exc:  # pragma: no cover - env dependent
            logger.warning("Live GUI disabled (failed to create window): %s", exc)
            self._tk = None
            self._reward_canvas = None
            self._obs_canvas = None
            self._action_canvas = None
            self._started = False

    def _make_scrollable_canvas(self, parent: Any, tk: Any, height: int, bg: str = "white"):
        """Return (canvas, outer_frame) with a vertical scrollbar."""
        frame = tk.Frame(parent)
        vscroll = tk.Scrollbar(frame, orient="vertical")
        canvas = tk.Canvas(frame, height=height, bg=bg, yscrollcommand=vscroll.set, bd=0, highlightthickness=1)
        vscroll.config(command=canvas.yview)
        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        def _on_wheel(event):
            delta = getattr(event, "delta", 0)
            if delta:
                units = max(1, abs(int(delta)) // 120)
                canvas.yview_scroll(-units if delta > 0 else units, "units")
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        canvas.bind("<MouseWheel>", _on_wheel)
        canvas.bind("<Button-4>", _on_wheel)
        canvas.bind("<Button-5>", _on_wheel)
        return canvas, frame

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
            self._action_canvas = None
            self._started = False

    def on_step(self, obs: np.ndarray, reward: float, info: dict, action: Any = None) -> None:
        if not self._started or self._tk is None:
            return
        self._step_idx += 1
        if action is not None:
            self._last_actions.append((self._step_idx, action))
        step_components, self._prev_episode_components = _derive_step_components(
            info, reward, self._prev_episode_components,
        )
        avg_components = _rolling_means(
            self._reward_history, step_components, self._rolling_window,
        )
        self._draw_reward_panel(avg_components)
        self._draw_action_panel()
        self._draw_observation_panel(np.asarray(obs, dtype=np.float32))
        try:
            self._tk.update_idletasks()
            self._tk.update()
        except Exception:  # pragma: no cover - env dependent
            self.close()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_grid_lines(self, canvas: Any, left: int, right: int, y_top: int, y_bot: int, n_lines: int = 4) -> None:
        """Draw subtle vertical grid lines between left and right."""
        for i in range(1, n_lines):
            x = left + (right - left) * i // n_lines
            canvas.create_line(x, y_top, x, y_bot, fill="#dddddd", dash=(2, 3))

    def _draw_reward_panel(self, avg_components: dict[str, float]) -> None:
        if self._reward_canvas is None:
            return
        c = self._reward_canvas
        c.delete("all")
        c.create_text(8, 6, anchor="nw", text=f"step {self._step_idx}", fill="#666", font=("TkDefaultFont", 8))
        if not avg_components:
            c.create_text(8, 26, anchor="nw", text="No reward components yet.", fill="gray")
            c.configure(scrollregion=(0, 0, 200, 60))
            return

        items = sorted(avg_components.items(), key=lambda kv: _reward_sort_key(kv[0]))
        item_cols = _split_into_columns_preserving_order(items, n_cols=2)
        max_abs = max(1.0, max(abs(v) for _, v in items))
        canvas_w = max(420, int(c.winfo_width()) or 420)
        col_gap = 20
        n_cols = max(1, len(item_cols))
        total_inner = canvas_w - 16 - col_gap * (n_cols - 1)
        col_w = max(180, total_inner // n_cols)
        y_top = 24
        bar_h = 16
        gap = 6
        max_rows = max((len(col_items) for col_items in item_cols), default=0)
        total_height = y_top + max_rows * (bar_h + gap) + 10
        for col_i, col_items in enumerate(item_cols):
            col_left = 8 + col_i * (col_w + col_gap)
            label_x = col_left
            left = col_left + 110
            right = col_left + col_w - 48
            if right <= left + 20:
                right = left + 20
            zero_x = left + (right - left) // 2
            self._draw_grid_lines(c, left, right, y_top - 4, total_height, n_lines=4)
            c.create_line(zero_x, y_top - 4, zero_x, total_height, fill="#999", width=1)

            y = y_top
            for name, value in col_items:
                frac = value / max_abs
                px = int((right - left) * min(1.0, abs(frac)) / 2)
                x0, x1 = (zero_x, zero_x + px) if frac >= 0 else (zero_x - px, zero_x)
                color = "#2e8b57" if value >= 0 else "#d9534f"
                c.create_rectangle(x0, y, x1, y + bar_h, fill=color, outline=color)
                c.create_text(label_x, y + 2, anchor="nw", text=f"{name}", font=("TkDefaultFont", 8))
                c.create_text(right + 4, y + 2, anchor="nw", text=f"{value:+.3f}", font=("TkFixedFont", 8))
                y += bar_h + gap

        c.configure(scrollregion=(0, 0, canvas_w, total_height + 4))

    def _draw_action_panel(self) -> None:
        if self._action_canvas is None:
            return
        c = self._action_canvas
        c.delete("all")
        if not self._last_actions:
            c.create_text(8, 8, anchor="nw", text="(no actions yet)", fill="gray", font=("TkDefaultFont", 8))
            c.configure(scrollregion=(0, 0, 200, 40))
            return

        y = 8
        row_h = 18
        actions = list(self._last_actions)
        shown = 0
        for step_i, act in reversed(actions):
            action_text = _fmt_action(act)
            if not action_text:
                continue
            label = f"#{step_i:>5}: {action_text}"
            c.create_text(8, y, anchor="nw", text=label, font=("TkFixedFont", 8), fill="#222")
            y += row_h
            shown += 1

        if shown == 0:
            c.create_text(8, 8, anchor="nw", text="(no non-no-op actions yet)", fill="gray", font=("TkDefaultFont", 8))
            c.configure(scrollregion=(0, 0, 200, 40))
            return

        c.configure(scrollregion=(0, 0, int(c.winfo_width()) or 200, y + 4))

    def _draw_observation_panel(self, obs: np.ndarray) -> None:
        if self._obs_canvas is None:
            return
        c = self._obs_canvas
        c.delete("all")
        norm = np.divide(
            obs,
            np.where(np.abs(self._obs_scales) < 1e-9, 1.0, self._obs_scales),
            out=np.zeros_like(obs, dtype=np.float32),
        )

        canvas_w = max(680, int(c.winfo_width()) or 680)
        n_cols = _observation_column_count(canvas_w)
        col_gap = 16
        total_inner = canvas_w - 16 - col_gap * (n_cols - 1)
        col_w = max(150, total_inner // n_cols)
        col_lefts = [8 + i * (col_w + col_gap) for i in range(n_cols)]
        col_ys = [8 for _ in range(n_cols)]

        def _next_col() -> int:
            return min(range(n_cols), key=lambda i: col_ys[i])

        def _draw_scalar_section(col_i: int, idxs: list[int]) -> None:
            if not idxs:
                return
            x0 = col_lefts[col_i]
            y = col_ys[col_i]
            c.create_text(x0, y, anchor="nw", text="Scalars", fill="#333", font=("TkDefaultFont", 8, "bold"))
            y += 18
            left = x0 + 85
            right = x0 + col_w - 35
            if right <= left + 20:
                right = left + 20
            zero_x = left + (right - left) // 2
            bar_h = 14
            gap = 4
            section_bot = y + len(idxs) * (bar_h + gap) + 4
            self._draw_grid_lines(c, left, right, y - 2, section_bot, n_lines=4)
            c.create_line(zero_x, y - 2, zero_x, section_bot, fill="#bbb", width=1)
            for idx in idxs:
                val = float(obs[idx])
                nval = max(-1.0, min(1.0, float(norm[idx])))
                px = int((right - left) * abs(nval) / 2)
                x1, x2 = (zero_x, zero_x + px) if nval >= 0 else (zero_x - px, zero_x)
                color = "#4c78a8" if nval >= 0 else "#f58518"
                c.create_rectangle(x1, y, x2, y + bar_h, fill=color, outline=color)
                c.create_text(x0, y, anchor="nw", text=self._obs_names[idx], font=("TkDefaultFont", 8))
                c.create_text(right + 3, y, anchor="nw", text=f"{val:+.1f}", font=("TkFixedFont", 8))
                y += bar_h + gap
            col_ys[col_i] = y + 6

        def _draw_xy_section(col_i: int, pairs: list[tuple[str, str, str]]) -> None:
            if not pairs:
                return
            x0 = col_lefts[col_i]
            y = col_ys[col_i]
            c.create_text(x0, y, anchor="nw", text="XY pairs", fill="#333", font=("TkDefaultFont", 8, "bold"))
            y += 18
            for base, xn, yn in pairs[:6]:
                xi = self._obs_names.index(xn)
                yi = self._obs_names.index(yn)
                xv = float(norm[xi])
                yv = float(norm[yi])
                c.create_text(x0, y, anchor="nw", text=f"{base}: ({obs[xi]:+.1f}, {obs[yi]:+.1f})", font=("TkDefaultFont", 8))
                box_s = 36
                box_l = x0 + col_w - box_s - 4
                box_t = y
                c.create_rectangle(box_l, box_t, box_l + box_s, box_t + box_s, outline="#888")
                cx, cy = box_l + box_s / 2, box_t + box_s / 2
                c.create_line(cx, cy, cx + 12 * xv, cy - 12 * yv, fill="#2e8b57", width=2)
                y += 44
            col_ys[col_i] = y + 6

        def _draw_indexed_section(col_i: int, indexed: list[tuple[str, list[int]]]) -> None:
            if not indexed:
                return
            x0 = col_lefts[col_i]
            y = col_ys[col_i]
            c.create_text(x0, y, anchor="nw", text="Indexed vectors", fill="#333", font=("TkDefaultFont", 8, "bold"))
            y += 18
            for base, idxs in indexed[:5]:
                c.create_text(x0, y + 2, anchor="nw", text=base, font=("TkDefaultFont", 8))
                x = x0 + 95
                for idx in idxs[:20]:
                    v = float(norm[idx])
                    intensity = int(max(0, min(255, 127 + 120 * v)))
                    color = f"#{intensity:02x}64{(255-intensity):02x}"
                    c.create_rectangle(x, y, x + 10, y + 10, fill=color, outline="")
                    x += 12
                    if x + 10 > (x0 + col_w):
                        break
                y += 18
            col_ys[col_i] = y + 6

        def _draw_quad_section(col_i: int, quads: list[tuple[str, dict[str, int]]]) -> None:
            if not quads:
                return
            x0 = col_lefts[col_i]
            y = col_ys[col_i]
            c.create_text(x0, y, anchor="nw", text="Quadrants", fill="#333", font=("TkDefaultFont", 8, "bold"))
            y += 18
            for base, idx_map in quads[:4]:
                c.create_text(x0, y, anchor="nw", text=base, font=("TkDefaultFont", 8))
                cell = 16
                grid_x = x0 + col_w - (2 * cell + 6)
                grid_y = y
                mapping = {"_NW": (0, 0), "_NE": (1, 0), "_SW": (0, 1), "_SE": (1, 1)}
                for suf, (gx, gy) in mapping.items():
                    idx = idx_map[suf]
                    v = float(norm[idx])
                    intensity = int(max(0, min(255, 127 + 120 * v)))
                    color = f"#{intensity:02x}{intensity:02x}c8"
                    x1 = grid_x + gx * cell
                    y1 = grid_y + gy * cell
                    c.create_rectangle(x1, y1, x1 + cell, y1 + cell, fill=color, outline="#666")
                y += 38
            col_ys[col_i] = y + 6

        _draw_scalar_section(_next_col(), sorted(self._groups.scalar_idxs))
        _draw_xy_section(_next_col(), self._groups.xy_pairs)
        _draw_indexed_section(_next_col(), self._groups.indexed)
        _draw_quad_section(_next_col(), self._groups.quads)

        c.configure(scrollregion=(0, 0, canvas_w, max(col_ys) + 8))


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
