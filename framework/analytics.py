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

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
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
        "| Sim  | Reward   | Result       |\n",
        "|------|----------|--------------|\n",
    ]
    for s in data.greedy_sims:
        tag = "**NEW BEST**" if s.improved else ""
        lines.append(f"| {s.sim:4d} | {s.reward:+8.1f} | {tag} |\n")
    return "".join(lines)


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
        }
    best_reward = max(s.reward for s in sims)
    n_improvements = sum(1 for s in sims if s.improved)
    improved_sims = [s.sim for s in sims if s.improved]
    first_improvement_sim = improved_sims[0] if improved_sims else None
    best_sim = max(sims, key=lambda s: s.reward)
    b, c, a = best_sim.throttle_counts
    total = (b + c + a) or 1
    accel_pct = 100 * a / total
    return {
        "best_reward": best_reward,
        "n_improvements": n_improvements,
        "first_improvement_sim": first_improvement_sim,
        "accel_pct": accel_pct,
        "greedy_runtime_s": data.timings.get("greedy_s"),
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
    ranked = sorted(stats, key=lambda x: -x[1]["best_reward"])

    plot_gs_comparison_rewards([(name, s) for name, s in ranked], summary_dir)
    if extra_plots_fn is not None:
        extra_plots_fn(runs, summary_dir)

    lines = [
        f"# Grid Search Summary: {base_name}\n\n",
        f"{len(runs)} experiments, ranked by best greedy reward.\n\n",
        "![Reward comparison](comparison_rewards.png)\n\n",
    ]
    lines += [
        "## Rankings\n\n",
        "| Rank | Experiment | Best Reward | Improvements | First Improv. Sim | Accel % | Greedy Time |\n",
        "|------|-----------|-------------|--------------|-------------------|---------|-------------|\n",
    ]
    for rank, (name, s) in enumerate(ranked, 1):
        fi  = str(s["first_improvement_sim"]) if s["first_improvement_sim"] is not None else "—"
        acc = f"{s['accel_pct']:.0f}%" if s["accel_pct"] is not None else "—"
        rt  = _fmt_duration(s["greedy_runtime_s"]) if s["greedy_runtime_s"] else "—"
        lines.append(
            f"| {rank} | {name} | {s['best_reward']:+.1f} | {s['n_improvements']} "
            f"| {fi} | {acc} | {rt} |\n"
        )
    lines.append("\n")

    for rank, (name, data) in enumerate(
        sorted(runs, key=lambda x: -_gs_stats(x[1])["best_reward"]), 1
    ):
        s = _gs_stats(data)
        results_rel = f"../{name}/results"
        lines.append(f"---\n\n## {rank}. {name}\n\n**Best reward: {s['best_reward']:+.1f}**\n\n")
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
        lines += [
            "| Stat | Value |\n|---|---|\n",
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
        ]

    report_path = os.path.join(summary_dir, "summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info("Saved grid summary → %s", report_path)
