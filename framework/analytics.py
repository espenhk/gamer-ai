"""Generic data containers and reward/training plots.

Game-specific analytics (paths, weight heatmaps, throttle traces) live in
games/<name>/analytics.py.  This module only depends on framework/ symbols.

Entry points used by the training loop:
    ExperimentData        — top-level result container
    RunTrace              — per-episode sampled trajectory
    ProbeResult           — one fixed-action probe episode
    ColdStartSimResult    — one sim inside a cold-start restart
    ColdStartRestartResult — one cold-start restart
    GreedySimResult       — one greedy simulation
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg', force=True)  # prevent TkAgg GC-from-daemon-thread crashes (issue #73)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.cm as cm
    from matplotlib.figure import Figure
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RunTrace:
    """Sampled trajectory for one episode.

    pos_x / pos_z are world-space coordinates (game units).
    throttle_state entries are (accel_val, brake_val) float tuples in [0, 1].
    These are populated from the step info dict and action arrays; for games
    that don't provide them the lists stay empty.
    """
    pos_x: list          # world X, sampled every TRACE_SAMPLE_EVERY steps
    pos_z: list          # world Z (horizontal plane; Y is up in most engines)
    throttle_state: list # per step: (accel_val, brake_val) floats in [0, 1]
    total_reward: float


@dataclass
class ProbeResult:
    action_idx: int
    action_name: str
    reward: float
    trace: RunTrace | None = None


@dataclass
class ColdStartSimResult:
    sim: int           # 1-based within its restart
    reward: float
    throttle_counts: list  # [brake_steps, coast_steps, accel_steps]
    total_steps: int
    trace: RunTrace | None = None
    termination_reason: str | None = None


@dataclass
class ColdStartRestartResult:
    restart: int
    sims: list         # list[ColdStartSimResult]
    best_reward: float
    beat_probe_floor: bool


@dataclass
class GreedySimResult:
    sim: int
    reward: float
    improved: bool
    throttle_counts: list  # [brake_steps, coast_steps, accel_steps]
    total_steps: int
    trace: RunTrace | None = None
    weights: dict | None = None
    final_track_progress: float = 0.0
    laps_completed: int = 0
    mutation_scale: float | None = None
    termination_reason: str | None = None
    # --- Option A: config-independent task metrics ---
    finish_time_s: float | None = None        # elapsed_s when finished; None if not
    mean_abs_lateral_offset: float | None = None  # mean |lateral_offset| for episode
    # --- Option C: per-component reward totals ---
    reward_components: dict | None = None     # {component_name: total_contribution}


@dataclass
class ExperimentData:
    experiment_name: str
    probe_results: list        # list[ProbeResult]; empty if weights pre-existed
    cold_start_restarts: list  # list[ColdStartRestartResult]; empty if skipped
    greedy_sims: list          # list[GreedySimResult]
    probe_floor: float | None  # best probe reward, or None if probe was skipped
    weights_file: str          # absolute or relative path to policy_weights.yaml
    reward_config_file: str    # path to the experiment's reward_config.yaml
    training_params: dict      # hyperparams dict
    timings: dict              # start, end, total_s, probe_s, cold_start_s, greedy_s
    track: str = ""
    early_stopped: bool = False          # True if patience-based early stopping fired
    early_stop_sim: int | None = None    # sim index where early stopping fired


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: "Figure", path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generic plots
# ---------------------------------------------------------------------------

def plot_probe_rewards(data: ExperimentData, results_dir: str) -> None:
    if not _HAS_MPL:
        return
    probes  = sorted(data.probe_results, key=lambda p: p.action_idx)
    names   = [p.action_name for p in probes]
    rewards = [p.reward for p in probes]
    best_r  = max(rewards)
    colors  = ["#f1c40f" if r == best_r else "#3498db" for r in rewards]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, rewards, color=colors, edgecolor="white", linewidth=0.6)

    if data.probe_floor is not None:
        ax.axhline(data.probe_floor, color="#e74c3c", linestyle="--",
                   linewidth=1.4, label=f"probe floor ({data.probe_floor:+.1f})")
        ax.legend(fontsize=9)

    ax.set_title(f"{data.experiment_name} — Probe Phase: Reward per Constant Action")
    ax.set_xlabel("Action")
    ax.set_ylabel("Total Episode Reward")
    ax.tick_params(axis="x", rotation=20)
    for bar, r in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(bar.get_height()) * 0.01,
                f"{r:+.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "probe_rewards.png"))


def plot_cold_start_rewards(data: ExperimentData, results_dir: str) -> None:
    if not _HAS_MPL:
        return
    restarts = data.cold_start_restarts
    xs       = [r.restart for r in restarts]
    rewards  = [r.best_reward for r in restarts]
    colors   = ["#27ae60" if r.beat_probe_floor else "#c0392b" for r in restarts]

    fig, ax = plt.subplots(figsize=(max(6, len(xs) * 0.8), 5))
    ax.bar(xs, rewards, color=colors, edgecolor="white", linewidth=0.6)
    if data.probe_floor is not None:
        ax.axhline(data.probe_floor, color="#f39c12", linestyle="--",
                   linewidth=1.4, label=f"probe floor ({data.probe_floor:+.1f})")
    ax.legend(handles=[
        mpatches.Patch(color="#27ae60", label="beat probe floor"),
        mpatches.Patch(color="#c0392b", label="below probe floor"),
    ] + ([ax.get_legend_handles_labels()[0][0]] if data.probe_floor is not None else []),
        fontsize=9)
    ax.set_title(f"{data.experiment_name} — Cold-Start: Best Reward per Restart")
    ax.set_xlabel("Restart")
    ax.set_ylabel("Best Episode Reward")
    ax.set_xticks(xs)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "cold_start_best_rewards.png"))


def plot_greedy_rewards(data: ExperimentData, results_dir: str) -> None:
    if not _HAS_MPL:
        return
    sims    = [s.sim for s in data.greedy_sims]
    rewards = [s.reward for s in data.greedy_sims]

    best_so_far   = []
    running_best  = float("-inf")
    for r in rewards:
        running_best = max(running_best, r)
        best_so_far.append(running_best)

    improvement_xs = [s.sim for s in data.greedy_sims if s.improved]
    improvement_ys = [s.reward for s in data.greedy_sims if s.improved]

    fig, ax = plt.subplots(figsize=(max(8, len(sims) * 0.15), 5))
    ax.scatter(sims, rewards, color="#95a5a6", s=18, alpha=0.7, zorder=2, label="candidate reward")
    ax.step(sims, best_so_far, where="post", color="#e67e22",
            linewidth=2.0, zorder=3, label="best so far")
    ax.scatter(improvement_xs, improvement_ys, color="#27ae60",
               s=60, zorder=4, marker="^", label="improvement")
    if getattr(data, "early_stopped", False) and data.early_stop_sim is not None:
        ax.axvline(data.early_stop_sim, color="#e74c3c", linestyle="--",
                   linewidth=1.5, label=f"early stop (sim {data.early_stop_sim})")
    ax.set_title(f"{data.experiment_name} — Greedy Phase: Reward per Simulation")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Total Episode Reward")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "greedy_rewards.png"))


def plot_reward_trajectory(data: ExperimentData, results_dir: str) -> None:
    if not _HAS_MPL:
        return
    xs, ys, colors = [], [], []
    x = 0

    for p in sorted(data.probe_results, key=lambda p: p.action_idx):
        xs.append(x); ys.append(p.reward); colors.append("#3498db")
        x += 1

    for restart in data.cold_start_restarts:
        for s in restart.sims:
            xs.append(x); ys.append(s.reward); colors.append("#9b59b6")
            x += 1

    for s in data.greedy_sims:
        xs.append(x); ys.append(s.reward); colors.append("#e67e22")
        x += 1

    running_best = float("-inf")
    best_xs, best_ys = [], []
    for xi, yi in zip(xs, ys):
        running_best = max(running_best, yi)
        best_xs.append(xi); best_ys.append(running_best)

    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.12), 5))
    ax.scatter(xs, ys, c=colors, s=14, alpha=0.6, zorder=2)
    ax.step(best_xs, best_ys, where="post", color="black",
            linewidth=1.8, zorder=3, label="best so far")

    boundary = len(data.probe_results or [])
    if boundary > 0 and (data.cold_start_restarts or data.greedy_sims):
        ax.axvline(boundary - 0.5, color="#3498db", linestyle=":", linewidth=1, alpha=0.6)
    cs_total = sum(len(r.sims) for r in (data.cold_start_restarts or []))
    if cs_total > 0 and data.greedy_sims:
        ax.axvline(boundary + cs_total - 0.5, color="#9b59b6", linestyle=":", linewidth=1, alpha=0.6)

    legend_patches = []
    if data.probe_results:
        legend_patches.append(mpatches.Patch(color="#3498db", label="probe"))
    if data.cold_start_restarts:
        legend_patches.append(mpatches.Patch(color="#9b59b6", label="cold-start"))
    if data.greedy_sims:
        legend_patches.append(mpatches.Patch(color="#e67e22", label="greedy"))
    ax.legend(handles=legend_patches + [ax.get_legend_handles_labels()[0][-1]], fontsize=9)
    ax.set_title(f"{data.experiment_name} — Reward Trajectory Across All Phases")
    ax.set_xlabel("Cumulative simulation")
    ax.set_ylabel("Total Episode Reward")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "reward_trajectory.png"))


def plot_task_metrics(data: ExperimentData, results_dir: str) -> None:
    """Plot config-independent task metrics for the greedy phase.

    Shows finish-time evolution and finish rate in a two-panel figure.
    Runs with no finish data produce only the finish-rate panel.
    """
    if not _HAS_MPL:
        return
    sims = data.greedy_sims
    if not sims:
        return

    xs = [s.sim for s in sims]
    has_finish = any(s.finish_time_s is not None for s in sims)

    n_panels = 2 if has_finish else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # Panel 1 (optional): finish time per sim
    if has_finish:
        ax_ft = axes[0]
        ax_rate = axes[1]
        ft_xs = [s.sim for s in sims if s.finish_time_s is not None]
        ft_ys = [s.finish_time_s for s in sims if s.finish_time_s is not None]
        ax_ft.scatter(ft_xs, ft_ys, color="#27ae60", s=40, zorder=3, label="finish time")
        if ft_ys:
            best_ft = ft_ys[0]
            best_xs_ft, best_ys_ft = [], []
            for xi, yi in zip(ft_xs, ft_ys):
                if yi < best_ft:
                    best_ft = yi
                best_xs_ft.append(xi)
                best_ys_ft.append(best_ft)
            ax_ft.step(best_xs_ft, best_ys_ft, where="post", color="black",
                       linewidth=1.8, label="best so far")
        ax_ft.set_title("Finish Time per Sim")
        ax_ft.set_xlabel("Simulation")
        ax_ft.set_ylabel("Finish time (s)")
        ax_ft.legend(fontsize=9)
    else:
        ax_rate = axes[0]

    # Panel: rolling finish rate (window = min(20, len(sims)))
    window = min(20, len(sims))
    finished_flags = [1 if s.finish_time_s is not None else 0 for s in sims]
    rates = [
        sum(finished_flags[max(0, i - window + 1): i + 1]) / min(i + 1, window)
        for i in range(len(sims))
    ]
    ax_rate.plot(xs, rates, color="#3498db", linewidth=1.8, label=f"finish rate (window={window})")
    ax_rate.set_ylim(0, 1.05)
    ax_rate.set_title("Rolling Finish Rate")
    ax_rate.set_xlabel("Simulation")
    ax_rate.set_ylabel("Fraction finished")
    ax_rate.legend(fontsize=9)

    fig.suptitle(f"{data.experiment_name} — Task Metrics (config-independent)", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "task_metrics.png"))


def plot_reward_components(data: ExperimentData, results_dir: str) -> None:
    """Plot per-component reward totals across greedy-phase simulations.

    Each component (progress, centerline, speed, …) is drawn as a separate
    line.  Components whose total is zero in every sim are omitted.
    """
    if not _HAS_MPL:
        return
    sims = data.greedy_sims
    if not sims:
        return
    # Only plot if at least one sim has component data.
    if not any(s.reward_components for s in sims):
        return

    # Collect all component names present in any sim.
    all_keys: list[str] = []
    seen: set[str] = set()
    for s in sims:
        if s.reward_components:
            for k in s.reward_components:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

    xs = [s.sim for s in sims]
    series: dict[str, list[float]] = {k: [] for k in all_keys}
    for s in sims:
        comps = s.reward_components or {}
        for k in all_keys:
            series[k].append(comps.get(k, 0.0))

    # Drop components that are zero everywhere.
    active_keys = [k for k in all_keys if any(v != 0.0 for v in series[k])]
    if not active_keys:
        return

    cmap = cm.tab10(np.linspace(0, 1, min(len(active_keys), 10)))
    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.15), 5))
    for i, k in enumerate(active_keys):
        color = cmap[i % len(cmap)]
        ax.plot(xs, series[k], color=color, linewidth=1.2, alpha=0.85, label=k)

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_title(f"{data.experiment_name} — Reward Components per Sim")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Component total (episode sum)")
    ax.legend(fontsize=8, loc="best", ncol=max(1, len(active_keys) // 5))
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "reward_components.png"))


# ---------------------------------------------------------------------------
# Markdown tables
# ---------------------------------------------------------------------------

def _probe_table_md(data: ExperimentData) -> str:
    best_reward = max(p.reward for p in data.probe_results)
    lines = [
        "## Probe Phase\n\n",
        f"Best probe reward: **{best_reward:+.1f}**\n\n",
        "| Action | Name            | Reward   |          |\n",
        "|--------|-----------------|----------|----------|\n",
    ]
    for p in sorted(data.probe_results, key=lambda x: x.action_idx):
        marker = "← best" if p.reward == best_reward else ""
        lines.append(f"| {p.action_idx:6d} | {p.action_name:15s} | {p.reward:+8.1f} | {marker} |\n")
    return "".join(lines)


def _cold_start_table_md(data: ExperimentData) -> str:
    best_r = max(r.best_reward for r in data.cold_start_restarts)
    lines = [
        "## Cold-Start Search\n\n",
        f"Best cold-start reward: **{best_r:+.1f}**\n",
        f"Probe floor: **{data.probe_floor:+.1f}**\n\n" if data.probe_floor is not None else "\n",
        "| Restart | Best Reward | Beat Probe Floor |          |\n",
        "|---------|-------------|------------------|----------|\n",
    ]
    for r in data.cold_start_restarts:
        marker = "← best" if r.best_reward == best_r else ""
        beat   = "yes" if r.beat_probe_floor else "no"
        lines.append(f"| {r.restart:7d} | {r.best_reward:+11.1f} | {beat:16s} | {marker} |\n")
    return "".join(lines)


def _greedy_table_md(data: ExperimentData) -> str:
    best_r = max((s.reward for s in data.greedy_sims), default=float("-inf"))
    lines = [
        "## Greedy Phase\n\n",
        f"Best reward: **{best_r:+.1f}**\n\n",
        "| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |\n",
        "|------|----------|----------|-------------|--------------|--------------|-------------|\n",
    ]
    for s in data.greedy_sims:
        tag    = "**NEW BEST**" if s.improved else ""
        reason = s.termination_reason or ""
        prog   = f"{s.final_track_progress:.3f}"
        ft     = f"{s.finish_time_s:.1f}s" if s.finish_time_s is not None else "—"
        lat    = (f"{s.mean_abs_lateral_offset:.2f}m"
                  if s.mean_abs_lateral_offset is not None else "—")
        lines.append(
            f"| {s.sim:4d} | {s.reward:+8.1f} | {prog:8s} | {ft:11s} | {lat:7s} | {reason:12s} | {tag} |\n"
        )
    return "".join(lines)


def _task_metrics_table_md(data: ExperimentData) -> str:
    """Markdown table of config-independent task metrics for the greedy phase."""
    sims = data.greedy_sims
    if not sims:
        return ""
    finished_sims = [s for s in sims if s.finish_time_s is not None]
    finish_rate   = len(finished_sims) / len(sims)
    best_progress = max(s.final_track_progress for s in sims)
    mean_progress = sum(s.final_track_progress for s in sims) / len(sims)
    lats   = [s.mean_abs_lateral_offset for s in sims if s.mean_abs_lateral_offset is not None]
    mean_lat = sum(lats) / len(lats) if lats else None
    lines  = [
        "## Task Metrics (config-independent)\n\n",
        "| Metric | Value |\n",
        "|--------|-------|\n",
        f"| Finish rate | {finish_rate:.1%} ({len(finished_sims)}/{len(sims)} sims) |\n",
        f"| Best track progress | {best_progress:.4f} |\n",
        f"| Mean track progress | {mean_progress:.4f} |\n",
    ]
    if finished_sims:
        finish_times = [s.finish_time_s for s in finished_sims if s.finish_time_s is not None]
        if finish_times:
            best_ft  = min(finish_times)
            mean_ft  = sum(finish_times) / len(finish_times)
            lines += [
                f"| Best finish time | {best_ft:.1f}s |\n",
                f"| Mean finish time | {mean_ft:.1f}s |\n",
            ]
    if mean_lat is not None:
        lines.append(f"| Mean abs lateral offset | {mean_lat:.3f}m |\n")
    return "".join(lines) + "\n"


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h}h {m:02d}m {s:04.1f}s"
    if m:
        return f"{m}m {s:04.1f}s"
    return f"{s:.1f}s"


def _timings_md(data: ExperimentData) -> str:
    t = data.timings
    lines = [
        "## Timings\n\n",
        f"- **Start:** {t['start']}\n",
        f"- **End:** {t['end']}\n",
        f"- **Total runtime:** {_fmt_duration(t['total_s'])}\n\n",
        "| Phase | Duration |\n",
        "|-------|----------|\n",
    ]
    if t.get("probe_s") is not None:
        lines.append(f"| Probe | {_fmt_duration(t['probe_s'])} |\n")
    if t.get("cold_start_s") is not None:
        lines.append(f"| Cold-start | {_fmt_duration(t['cold_start_s'])} |\n")
    if t.get("greedy_s") is not None:
        lines.append(f"| Greedy | {_fmt_duration(t['greedy_s'])} |\n")
    return "".join(lines) + "\n"


def _summary_md(data: ExperimentData) -> str:
    lines = ["## Run Parameters\n\n", "### Training\n\n",
             "| Parameter | Value |\n", "|-----------|-------|\n"]
    if data.track:
        lines.append(f"| track | {data.track} |\n")
    for k, v in data.training_params.items():
        lines.append(f"| {k} | {v} |\n")
    lines.append("\n### Reward Config\n\n")
    if os.path.exists(data.reward_config_file):
        with open(data.reward_config_file) as f:
            cfg = yaml.safe_load(f)
        lines.append("| Parameter | Value |\n|-----------|-------|\n")
        for k, v in cfg.items():
            lines.append(f"| {k} | {v} |\n")
    else:
        lines.append(f"_(reward config not found at `{data.reward_config_file}`)_\n")
    return "".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Grid search summary
# ---------------------------------------------------------------------------

def _gs_stats(data: ExperimentData) -> dict:
    sims = data.greedy_sims
    if not sims:
        return {
            "best_reward": float("-inf"), "n_improvements": 0,
            "first_improvement_sim": None, "accel_pct": None,
            "greedy_runtime_s": data.timings.get("greedy_s"),
            "finish_rate": 0.0, "best_track_progress": 0.0,
            "best_finish_time_s": None,
        }
    best_reward = max(s.reward for s in sims)
    n_improvements = sum(1 for s in sims if s.improved)
    improved_sims = [s.sim for s in sims if s.improved]
    first_improvement_sim = improved_sims[0] if improved_sims else None
    best_sim = max(sims, key=lambda s: s.reward)
    b, c, a = best_sim.throttle_counts
    total = (b + c + a) or 1
    accel_pct = 100 * a / total
    # Task metrics (config-independent).
    finished_sims = [s for s in sims if s.finish_time_s is not None]
    finish_rate = len(finished_sims) / len(sims)
    best_track_progress = max(s.final_track_progress for s in sims)
    finish_times = [s.finish_time_s for s in finished_sims if s.finish_time_s is not None]
    best_finish_time_s = min(finish_times) if finish_times else None
    return {
        "best_reward": best_reward,
        "n_improvements": n_improvements,
        "first_improvement_sim": first_improvement_sim,
        "accel_pct": accel_pct,
        "greedy_runtime_s": data.timings.get("greedy_s"),
        "finish_rate": finish_rate,
        "best_track_progress": best_track_progress,
        "best_finish_time_s": best_finish_time_s,
    }


def plot_gs_comparison_rewards(runs: list[tuple[str, dict]], summary_dir: str) -> None:
    if not _HAS_MPL:
        return
    runs_sorted = sorted(runs, key=lambda x: x[1]["best_reward"])
    names   = [r[0] for r in runs_sorted]
    rewards = [r[1]["best_reward"] for r in runs_sorted]
    n = len(names)
    colors = cm.RdYlGn(np.linspace(0.15, 0.85, n))
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.45)))
    bars = ax.barh(names, rewards, color=colors, edgecolor="white", linewidth=0.5)
    for bar, r in zip(bars, rewards):
        ax.text(r, bar.get_y() + bar.get_height() / 2, f"  {r:+.1f}", va="center", fontsize=8)
    ax.set_xlabel("Best Greedy Reward")
    ax.set_title("Grid Search — Best Reward per Experiment")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    _save(fig, os.path.join(summary_dir, "comparison_rewards.png"))


def plot_gs_comparison_task_metrics(runs: list[tuple[str, dict]], summary_dir: str) -> None:
    """Horizontal bar chart of best track progress per experiment (config-independent)."""
    if not _HAS_MPL:
        return
    runs_sorted = sorted(runs, key=lambda x: x[1]["best_track_progress"])
    names    = [r[0] for r in runs_sorted]
    progress = [r[1]["best_track_progress"] for r in runs_sorted]
    n = len(names)
    colors = cm.RdYlGn(np.linspace(0.15, 0.85, n))
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.45)))
    bars = ax.barh(names, progress, color=colors, edgecolor="white", linewidth=0.5)
    for bar, p in zip(bars, progress):
        ax.text(p, bar.get_y() + bar.get_height() / 2, f"  {p:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Best Track Progress (fraction, config-independent)")
    ax.set_title("Grid Search — Best Track Progress per Experiment")
    ax.set_xlim(0, max(max(progress) * 1.1, 1.05))
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    _save(fig, os.path.join(summary_dir, "comparison_task_metrics.png"))


def save_grid_summary(
    runs: list[tuple[str, ExperimentData]],
    varied_keys: list[str],
    summary_dir: str,
    base_name: str,
    extra_plots_fn=None,
) -> None:
    """Generate a cross-experiment summary report and plots in *summary_dir*.

    extra_plots_fn: optional callable(runs, summary_dir) for game-specific plots.
    """
    os.makedirs(summary_dir, exist_ok=True)
    stats  = [(name, _gs_stats(data)) for name, data in runs]
    # Primary ranking: best track progress (config-independent); secondary: reward.
    ranked_by_progress = sorted(stats,
                                key=lambda x: (-x[1]["best_track_progress"],
                                               -x[1]["best_reward"]))
    ranked_by_reward   = sorted(stats, key=lambda x: -x[1]["best_reward"])

    plot_gs_comparison_rewards([(name, s) for name, s in ranked_by_reward], summary_dir)
    plot_gs_comparison_task_metrics([(name, s) for name, s in ranked_by_progress], summary_dir)
    if extra_plots_fn is not None:
        extra_plots_fn(runs, summary_dir)

    lines = [
        f"# Grid Search Summary: {base_name}\n\n",
        f"{len(runs)} experiments.\n\n",
        "## Rankings by Task Metrics (config-independent)\n\n",
        "Ranked by best track progress, then by best reward.\n\n",
        "![Task metrics comparison](comparison_task_metrics.png)\n\n",
        "| Rank | Experiment | Best Progress | Finish Rate | Best Finish Time | Best Reward |\n",
        "|------|-----------|---------------|-------------|-----------------|-------------|\n",
    ]
    for rank, (name, s) in enumerate(ranked_by_progress, 1):
        prog = f"{s['best_track_progress']:.4f}"
        fr   = f"{s['finish_rate']:.1%}"
        bft  = f"{s['best_finish_time_s']:.1f}s" if s["best_finish_time_s"] is not None else "—"
        lines.append(
            f"| {rank} | {name} | {prog} | {fr} | {bft} | {s['best_reward']:+.1f} |\n"
        )
    lines.append("\n")

    lines += [
        "## Rankings by Reward\n\n",
        "![Reward comparison](comparison_rewards.png)\n\n",
        "| Rank | Experiment | Best Reward | Improvements | First Improv. Sim | Accel % | Greedy Time |\n",
        "|------|-----------|-------------|--------------|-------------------|---------|-------------|\n",
    ]
    for rank, (name, s) in enumerate(ranked_by_reward, 1):
        fi  = str(s["first_improvement_sim"]) if s["first_improvement_sim"] is not None else "—"
        acc = f"{s['accel_pct']:.0f}%" if s["accel_pct"] is not None else "—"
        rt  = _fmt_duration(s["greedy_runtime_s"]) if s["greedy_runtime_s"] else "—"
        lines.append(
            f"| {rank} | {name} | {s['best_reward']:+.1f} | {s['n_improvements']} "
            f"| {fi} | {acc} | {rt} |\n"
        )
    lines.append("\n")

    for rank, (name, data) in enumerate(
        sorted(runs, key=lambda x: (-_gs_stats(x[1])["best_track_progress"],
                                    -_gs_stats(x[1])["best_reward"])), 1
    ):
        s = _gs_stats(data)
        results_rel = f"../{name}/results"
        lines.append(f"---\n\n## {rank}. {name}\n\n"
                     f"**Best reward: {s['best_reward']:+.1f}** | "
                     f"**Best progress: {s['best_track_progress']:.4f}** | "
                     f"**Finish rate: {s['finish_rate']:.1%}**\n\n")
        if varied_keys and data.training_params:
            reward_cfg = {}
            if os.path.exists(data.reward_config_file):
                with open(data.reward_config_file) as f:
                    reward_cfg = yaml.safe_load(f) or {}
            all_params = {**data.training_params, **reward_cfg}
            lines.append("| Param | Value |\n|---|---|\n")
            for k in varied_keys:
                lines.append(f"| `{k}` | {all_params.get(k, '?')} |\n")
            lines.append("\n")
        fi  = str(s["first_improvement_sim"]) if s["first_improvement_sim"] is not None else "—"
        acc = f"{s['accel_pct']:.1f}%" if s["accel_pct"] is not None else "—"
        bft = f"{s['best_finish_time_s']:.1f}s" if s["best_finish_time_s"] is not None else "—"
        lines += [
            "| Stat | Value |\n|---|---|\n",
            f"| Best track progress | {s['best_track_progress']:.4f} |\n",
            f"| Finish rate | {s['finish_rate']:.1%} |\n",
            f"| Best finish time | {bft} |\n",
            f"| Greedy improvements | {s['n_improvements']} |\n",
            f"| First improvement (sim) | {fi} |\n",
            f"| Accel % of best run | {acc} |\n",
        ]
        if s["greedy_runtime_s"]:
            lines.append(f"| Greedy runtime | {_fmt_duration(s['greedy_runtime_s'])} |\n")
        lines.append("\n")
        lines += [
            f"![Best run path + throttle]({results_rel}/greedy_best_run.png)\n\n",
            f"![Weight evolution]({results_rel}/greedy_weight_evolution.png)\n\n",
            f"![Reward trajectory]({results_rel}/reward_trajectory.png)\n\n",
            f"![Task metrics]({results_rel}/task_metrics.png)\n\n",
        ]

    report_path = os.path.join(summary_dir, "summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    # Eagerly close all figures to prevent tkinter GC crashes from daemon threads
    if _HAS_MPL:
        plt.close('all')
    logger.info("Saved grid summary → %s", report_path)


# ---------------------------------------------------------------------------
# Experiment data JSON persistence (for grid-search consolidation)
# ---------------------------------------------------------------------------

def save_experiment_data_json(data: ExperimentData, results_dir: str) -> str:
    """Serialise *data* to ``experiment_data.json`` inside *results_dir*.

    Returns the path to the written file.
    """
    from distributed.protocol import experiment_to_json

    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "experiment_data.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(experiment_to_json(data))
    logger.info("Saved experiment data → %s", path)
    return path


def load_experiment_data(experiment_dir: str) -> ExperimentData:
    """Load an ``ExperimentData`` from *experiment_dir*/results/experiment_data.json.

    *experiment_dir* is the top-level experiment folder (e.g.
    ``games/tmnf/experiments/a03_centerline/gs_v1__ms0.05``).
    """
    from distributed.protocol import experiment_from_dict

    path = os.path.join(experiment_dir, "results", "experiment_data.json")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    data = experiment_from_dict(d)
    logger.debug("Loaded experiment data ← %s", path)
    return data
