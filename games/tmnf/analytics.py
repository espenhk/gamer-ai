"""TMNF-specific analytics plots.

These plots use TMNF-only concepts: bird's-eye position traces (pos_x / pos_z),
throttle-state traces, and WeightedLinearPolicy weight heatmaps / evolution charts.

Entry point called by save_experiment_results():
    save_tmnf_plots(data: ExperimentData, results_dir: str) -> None
"""
from __future__ import annotations

import logging
import os
import numpy as np
import yaml

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure


from framework.analytics import (
    ExperimentData,
    RunTrace,
    plot_probe_rewards,
    plot_cold_start_rewards,
    plot_greedy_rewards,
    plot_reward_trajectory,
    _probe_table_md,
    _cold_start_table_md,
    _greedy_table_md,
    _timings_md,
    _summary_md,
    save_grid_summary as _framework_save_grid_summary,
)
from games.tmnf.obs_spec import OBS_NAMES

logger = logging.getLogger(__name__)



def _save(fig: "Figure", path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


_THROTTLE_COLORS = ["#c0392b", "#95a5a6", "#27ae60"]


# ---------------------------------------------------------------------------
# Throttle / position helpers
# ---------------------------------------------------------------------------

def _plot_throttle_trace(ax: "Axes", throttle_state: list, title: str) -> None:
    steps = range(len(throttle_state))
    accel = [t[0] for t in throttle_state]
    brake = [t[1] for t in throttle_state]
    ax.plot(steps, accel, color="#27ae60", linewidth=0.8, alpha=0.85, label="accel")
    ax.plot(steps, brake, color="#c0392b", linewidth=0.8, alpha=0.85, label="brake")
    ax.set_ylim(-0.05, 1.1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, loc="upper right")


# ---------------------------------------------------------------------------
# Probe paths
# ---------------------------------------------------------------------------

def plot_probe_paths(data: ExperimentData, results_dir: str) -> None:
    probes = [p for p in sorted(data.probe_results, key=lambda p: p.action_idx)
              if p.trace and p.trace.pos_x]
    if not probes:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    colors  = cm.tab10(np.linspace(0, 1, len(probes)))
    for p, color in zip(probes, colors):
        ax.plot(p.trace.pos_x, p.trace.pos_z, color=color, linewidth=1.2,
                label=p.action_name, alpha=0.85)
        ax.plot(p.trace.pos_x[0], p.trace.pos_z[0], "o", color=color, markersize=5)
    ax.set_title(f"{data.experiment_name} — Probe Phase: Paths (bird's eye)")
    ax.set_xlabel("World X")
    ax.set_ylabel("World Z")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "probe_paths.png"))


# ---------------------------------------------------------------------------
# Cold-start best run
# ---------------------------------------------------------------------------

def plot_cold_start_best_run(data: ExperimentData, results_dir: str) -> None:
    best_sim = None
    best_r   = float("-inf")
    for restart in data.cold_start_restarts:
        for s in restart.sims:
            if s.reward > best_r:
                best_r   = s.reward
                best_sim = s
    if best_sim is None or not best_sim.trace or not best_sim.trace.pos_x:
        return

    trace = best_sim.trace
    fig, (ax_path, ax_thr) = plt.subplots(1, 2, figsize=(14, 6))
    ax_path.plot(trace.pos_x, trace.pos_z, color="#9b59b6", linewidth=1.4)
    ax_path.plot(trace.pos_x[0], trace.pos_z[0], "o", color="#9b59b6", markersize=6)
    ax_path.set_title("Path (bird's eye)")
    ax_path.set_xlabel("World X")
    ax_path.set_ylabel("World Z")
    ax_path.set_aspect("equal", adjustable="datalim")
    _plot_throttle_trace(ax_thr, trace.throttle_state,
                         f"Throttle/brake  (reward {trace.total_reward:+.1f})")
    fig.suptitle(f"{data.experiment_name} — Cold-Start Best Run", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "cold_start_best_run.png"))


def plot_cold_start_action_dist(data: ExperimentData, results_dir: str) -> None:
    restarts = data.cold_start_restarts
    xs = [r.restart for r in restarts]
    accel_pcts, brake_pcts = [], []
    for r in restarts:
        total_b = total_c = total_a = 0
        for s in r.sims:
            b, c, a = s.throttle_counts
            total_b += b; total_c += c; total_a += a
        total = total_b + total_c + total_a or 1
        accel_pcts.append(100 * total_a / total)
        brake_pcts.append(100 * total_b / total)

    fig, ax = plt.subplots(figsize=(max(6, len(xs) * 0.8), 5))
    ax.plot(xs, accel_pcts, color=_THROTTLE_COLORS[2], linewidth=1.8,
            marker="o", markersize=4, label="accel %")
    ax.plot(xs, brake_pcts, color=_THROTTLE_COLORS[0], linewidth=1.8,
            marker="o", markersize=4, label="brake %")
    ax.set_title(f"{data.experiment_name} — Cold-Start: Accel / Brake % per Restart")
    ax.set_xlabel("Restart")
    ax.set_ylabel("% Steps (threshold ≥ 0.5)")
    ax.set_xticks(xs)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "cold_start_action_dist.png"))


# ---------------------------------------------------------------------------
# Greedy best run
# ---------------------------------------------------------------------------

def plot_greedy_best_run(data: ExperimentData, results_dir: str) -> None:
    if not data.greedy_sims:
        return
    best  = max(data.greedy_sims, key=lambda s: s.reward)
    trace = best.trace
    if not trace or not trace.pos_x:
        return

    fig, (ax_path, ax_thr) = plt.subplots(1, 2, figsize=(14, 6))
    ax_path.plot(trace.pos_x, trace.pos_z, color="#e67e22", linewidth=1.4)
    ax_path.plot(trace.pos_x[0], trace.pos_z[0], "o", color="#e67e22", markersize=6)
    ax_path.set_title("Path (bird's eye)")
    ax_path.set_xlabel("World X")
    ax_path.set_ylabel("World Z")
    ax_path.set_aspect("equal", adjustable="datalim")
    _plot_throttle_trace(ax_thr, trace.throttle_state,
                         f"Throttle/brake  (reward {trace.total_reward:+.1f})")
    fig.suptitle(f"{data.experiment_name} — Greedy Best Run (sim {best.sim})", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "greedy_best_run.png"))


def plot_greedy_action_dist(data: ExperimentData, results_dir: str) -> None:
    sims = data.greedy_sims
    xs   = [s.sim for s in sims]
    accel_pcts, brake_pcts = [], []
    for s in sims:
        b, c, a = s.throttle_counts
        total = (b + c + a) or 1
        accel_pcts.append(100 * a / total)
        brake_pcts.append(100 * b / total)

    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.15), 5))
    ax.plot(xs, accel_pcts, color=_THROTTLE_COLORS[2], linewidth=1.2,
            alpha=0.85, label="accel %")
    ax.plot(xs, brake_pcts, color=_THROTTLE_COLORS[0], linewidth=1.2,
            alpha=0.85, label="brake %")
    ax.set_title(f"{data.experiment_name} — Greedy Phase: Accel / Brake % per Sim")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("% Steps (threshold ≥ 0.5)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "greedy_action_dist.png"))


def plot_greedy_progress(data: ExperimentData, results_dir: str) -> None:
    sims = data.greedy_sims
    xs   = [s.sim for s in sims]
    ys   = [s.laps_completed + s.final_track_progress for s in sims]

    running_best = 0.0
    best_so_far  = []
    for y in ys:
        running_best = max(running_best, y)
        best_so_far.append(running_best)

    improvement_xs = [s.sim for s in sims if s.improved]
    improvement_ys = [s.laps_completed + s.final_track_progress
                      for s in sims if s.improved]

    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.15), 5))
    ax.scatter(xs, ys, color="#95a5a6", s=18, alpha=0.7, zorder=2,
               label="candidate progress")
    ax.step(xs, best_so_far, where="post", color="#e67e22",
            linewidth=2.0, zorder=3, label="best so far")
    ax.scatter(improvement_xs, improvement_ys, color="#27ae60",
               s=60, zorder=4, marker="^", label="improvement")

    max_prog = max(ys) if ys else 1.0
    for lap in range(1, int(max_prog) + 2):
        if lap <= max_prog + 0.1:
            ax.axhline(lap, color="#bdc3c7", linestyle="--", linewidth=0.8, zorder=1)
            ax.text(xs[-1] + 0.3, lap, f"lap {lap}", va="center", fontsize=7,
                    color="#7f8c8d")

    ax.set_title(f"{data.experiment_name} — Greedy Phase: Track Progress per Sim")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Effective progress (laps + fraction)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "greedy_progress.png"))


# ---------------------------------------------------------------------------
# Weight heatmap and evolution (WLP-specific)
# ---------------------------------------------------------------------------

def plot_weight_heatmap(data: ExperimentData, results_dir: str) -> None:
    with open(data.weights_file) as f:
        cfg = yaml.safe_load(f) or {}
    if "steer_weights" not in cfg:
        return

    obs_names = OBS_NAMES
    steer_w = np.array([cfg["steer_weights"].get(n, 0.0) for n in obs_names])
    accel_w = np.array([cfg["accel_weights"].get(n, 0.0) for n in obs_names])
    brake_w = np.array([cfg["brake_weights"].get(n, 0.0) for n in obs_names])
    matrix  = np.vstack([steer_w, accel_w, brake_w])
    vmax    = max(abs(matrix).max(), 1e-6)

    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(matrix, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_xticks(range(len(obs_names)))
    ax.set_xticklabels(obs_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["steer", "accel", "brake"])
    ax.set_title(f"{data.experiment_name} — Final Policy Weight Heatmap")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "policy_weights_heatmap.png"))


def plot_weight_evolution(data: ExperimentData, results_dir: str) -> None:
    sims = [s for s in data.greedy_sims
            if s.weights is not None and "steer_weights" in s.weights]
    if len(sims) < 2:
        return

    obs_names = OBS_NAMES
    xs = [s.sim for s in sims]
    improvement_xs = [s.sim for s in sims if s.improved]

    steer_m = np.array([[s.weights["steer_weights"].get(n, 0.0) for n in obs_names]
                        for s in sims])
    accel_m = np.array([[s.weights["accel_weights"].get(n, 0.0) for n in obs_names]
                        for s in sims])
    brake_m = np.array([[s.weights["brake_weights"].get(n, 0.0) for n in obs_names]
                        for s in sims])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    cmap = plt.cm.get_cmap("tab20", len(obs_names))
    for ax, matrix, title in zip(
        axes,
        [steer_m, accel_m, brake_m],
        ["Steer weights", "Accel weights", "Brake weights"],
    ):
        for ix in improvement_xs:
            ax.axvline(ix, color="grey", alpha=0.25, linewidth=1)
        for i, name in enumerate(obs_names):
            ax.plot(xs, matrix[:, i], label=name, color=cmap(i),
                    linewidth=0.9, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Weight value")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=6, ncol=3, framealpha=0.6)
    axes[-1].set_xlabel("Simulation #")
    fig.suptitle(f"{data.experiment_name} — Weight evolution (greedy phase)\n"
                 "Grey lines = improvements")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "greedy_weight_evolution.png"))


# ---------------------------------------------------------------------------
# Termination reason breakdown
# ---------------------------------------------------------------------------

_REASON_COLORS = {
    "finish":     "#27ae60",
    "crash":      "#c0392b",
    "hard_crash": "#e74c3c",
    "timeout":    "#f39c12",
}
_REASON_ORDER = ["finish", "crash", "hard_crash", "timeout"]


def plot_termination_reasons(data: ExperimentData, results_dir: str) -> None:
    sims = data.greedy_sims
    if not sims:
        return

    counts: dict[str, int] = {}
    for s in sims:
        r = s.termination_reason or "unknown"
        counts[r] = counts.get(r, 0) + 1

    order  = _REASON_ORDER + sorted(k for k in counts if k not in _REASON_ORDER)
    labels = [r for r in order if r in counts]
    values = [counts[r] for r in labels]
    colors = [_REASON_COLORS.get(r, "#95a5a6") for r in labels]
    total  = len(sims)

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.4), 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(total * 0.005, 0.3),
            f"{v} ({100 * v / total:.0f}%)",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_title(
        f"{data.experiment_name} — Greedy Phase: Termination Reasons ({total} sims)"
    )
    ax.set_xlabel("Reason")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "termination_reasons.png"))


# ---------------------------------------------------------------------------
# Grid-search path comparison
# ---------------------------------------------------------------------------

def plot_gs_comparison_paths(
    runs: list[tuple[str, ExperimentData]],
    summary_dir: str,
) -> None:
    traced = []
    for name, data in runs:
        if not data.greedy_sims:
            continue
        best = max(data.greedy_sims, key=lambda s: s.reward)
        if best.trace and best.trace.pos_x:
            traced.append((name, best.reward, best.trace))
    if not traced:
        return

    traced.sort(key=lambda x: x[1])
    n      = len(traced)
    colors = cm.RdYlGn(np.linspace(0.15, 0.85, n))

    fig, ax = plt.subplots(figsize=(9, 9))
    for (name, reward, trace), color in zip(traced, colors):
        ax.plot(trace.pos_x, trace.pos_z, color=color, linewidth=1.0,
                label=f"{name}  ({reward:+.0f})", alpha=0.8)
        ax.plot(trace.pos_x[0], trace.pos_z[0], "o", color=color, markersize=4)
    ax.set_title("Grid Search — Best-Run Paths (green = higher reward)")
    ax.set_xlabel("World X")
    ax.set_ylabel("World Z")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=6, loc="best", framealpha=0.6)
    fig.tight_layout()
    _save(fig, os.path.join(summary_dir, "comparison_paths.png"))


def plot_gs_comparison_progress(
    runs: list[tuple[str, ExperimentData]],
    summary_dir: str,
) -> None:
    series = []
    for name, data in runs:
        if not data.greedy_sims:
            continue
        xs           = [s.sim for s in data.greedy_sims]
        running_best = 0.0
        ys           = []
        for s in data.greedy_sims:
            p = s.laps_completed + s.final_track_progress
            running_best = max(running_best, p)
            ys.append(running_best)
        series.append((name, xs, ys, running_best))
    if not series:
        return

    series.sort(key=lambda x: -x[3])
    n      = len(series)
    colors = cm.RdYlGn(np.linspace(0.15, 0.85, n))

    fig, ax = plt.subplots(figsize=(12, 6))
    for (name, xs, ys, final_best), color in zip(series, colors):
        ax.step(xs, ys, where="post", color=color, linewidth=1.4, alpha=0.85,
                label=f"{name}  ({final_best:.2f})")
    max_prog = max(s[3] for s in series) if series else 1.0
    for lap in range(1, int(max_prog) + 2):
        if lap <= max_prog + 0.1:
            ax.axhline(lap, color="#bdc3c7", linestyle="--", linewidth=0.7, zorder=1)
    ax.set_title("Grid Search — Best-So-Far Track Progress per Simulation")
    ax.set_xlabel("Simulation #")
    ax.set_ylabel("Effective progress (laps + fraction)")
    ax.legend(fontsize=6, loc="upper left", framealpha=0.7,
              ncol=max(1, n // 12))
    fig.tight_layout()
    _save(fig, os.path.join(summary_dir, "comparison_progress.png"))


# ---------------------------------------------------------------------------
# Entry point: called by save_experiment_results
# ---------------------------------------------------------------------------

def save_tmnf_plots(data: ExperimentData, results_dir: str) -> None:
    """Generate all TMNF-specific plots into results_dir."""
    if data.probe_results:
        plot_probe_paths(data, results_dir)
    if data.cold_start_restarts:
        plot_cold_start_action_dist(data, results_dir)
        plot_cold_start_best_run(data, results_dir)
    if data.greedy_sims:
        plot_greedy_best_run(data, results_dir)
        plot_greedy_action_dist(data, results_dir)
        plot_greedy_progress(data, results_dir)
        plot_weight_evolution(data, results_dir)
        plot_termination_reasons(data, results_dir)
    plot_weight_heatmap(data, results_dir)


# ---------------------------------------------------------------------------
# Combined entry points (generic framework report + TMNF-specific plots)
# ---------------------------------------------------------------------------

def save_experiment_results(data: ExperimentData, results_dir: str) -> None:
    """Generate all plots and write a single results.md report to *results_dir*."""
    os.makedirs(results_dir, exist_ok=True)

    track_line = f"\n**Track:** {data.track}\n" if data.track else ""
    sections = [
        f"# Experiment: {data.experiment_name}\n{track_line}\n",
        _timings_md(data),
        _summary_md(data),
    ]

    if data.probe_results:
        plot_probe_rewards(data, results_dir)
        plot_probe_paths(data, results_dir)
        sections.append(_probe_table_md(data))
        sections.append("\n![Probe rewards](probe_rewards.png)\n\n")
        sections.append("![Probe paths](probe_paths.png)\n\n")

    if data.cold_start_restarts:
        plot_cold_start_rewards(data, results_dir)
        plot_cold_start_action_dist(data, results_dir)
        plot_cold_start_best_run(data, results_dir)
        sections.append(_cold_start_table_md(data))
        sections.append("\n![Cold-start best rewards](cold_start_best_rewards.png)\n\n")
        sections.append("![Cold-start action distribution](cold_start_action_dist.png)\n\n")
        sections.append("![Cold-start best run](cold_start_best_run.png)\n\n")

    if data.greedy_sims:
        plot_greedy_rewards(data, results_dir)
        plot_greedy_progress(data, results_dir)
        plot_greedy_best_run(data, results_dir)
        plot_weight_evolution(data, results_dir)
        plot_termination_reasons(data, results_dir)
        sections.append(_greedy_table_md(data))
        sections.append("\n![Greedy rewards](greedy_rewards.png)\n\n")
        sections.append("![Greedy progress](greedy_progress.png)\n\n")
        sections.append("![Greedy best run](greedy_best_run.png)\n\n")
        sections.append("![Weight evolution](greedy_weight_evolution.png)\n\n")
        sections.append("![Termination reasons](termination_reasons.png)\n\n")

    plot_greedy_action_dist(data, results_dir)
    plot_reward_trajectory(data, results_dir)
    plot_weight_heatmap(data, results_dir)
    sections.append("## Additional Plots\n\n")
    if data.greedy_sims:
        sections.append("![Greedy action distribution](greedy_action_dist.png)\n\n")
    sections.append("![Reward trajectory](reward_trajectory.png)\n\n")
    if os.path.exists(data.weights_file):
        sections.append("![Policy weight heatmap](policy_weights_heatmap.png)\n\n")

    report_path = os.path.join(results_dir, "results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(sections)

    n = len(os.listdir(results_dir))
    logger.info("Saved %d file(s) to %s/ (report: results.md)", n, results_dir)


def save_grid_summary(
    runs: list[tuple[str, ExperimentData]],
    varied_keys: list[str],
    summary_dir: str,
    base_name: str,
) -> None:
    """Grid search cross-experiment summary with TMNF path/progress comparison plots."""

    def _tmnf_extra(r, d):
        plot_gs_comparison_paths(r, d)
        plot_gs_comparison_progress(r, d)

    _framework_save_grid_summary(
        runs, varied_keys, summary_dir, base_name,
        extra_plots_fn=_tmnf_extra,
    )
