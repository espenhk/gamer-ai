"""
analytics.py — Generate plots and summary tables at the end of a TMNF experiment.

Entry point:
    save_experiment_results(data: ExperimentData, results_dir: str) -> None

All output goes to results_dir/. The directory is created if it doesn't exist.
Files that belong to a skipped phase (e.g. probe/cold-start on a resumed run)
are simply not written.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import yaml
from policies import WeightedLinearPolicy


# ---------------------------------------------------------------------------
# Data containers (populated by main.py during training)
# ---------------------------------------------------------------------------

@dataclass
class RunTrace:
    """Sampled position + per-step throttle state for one episode."""
    pos_x: list          # world X, sampled every TRACE_SAMPLE_EVERY steps
    pos_z: list          # world Z (horizontal plane in TMNF; Y is up)
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
    weights: dict | None = None  # candidate.to_cfg() — steer/throttle weights for this sim
    final_track_progress: float = 0.0   # track_progress at episode end [0, 1)
    laps_completed: int = 0             # full laps finished (auto-respawn); 0 normally
    mutation_scale: float | None = None  # active mutation scale at this sim (adaptive)


@dataclass
class ExperimentData:
    experiment_name: str
    probe_results: list        # list[ProbeResult]; empty if weights pre-existed
    cold_start_restarts: list  # list[ColdStartRestartResult]; empty if skipped
    greedy_sims: list          # list[GreedySimResult]
    probe_floor: float | None  # best probe reward, or None if probe was skipped
    weights_file: str          # absolute or relative path to policy_weights.yaml
    reward_config_file: str    # path to the experiment's reward_config.yaml
    training_params: dict      # SPEED, N_SIMS, etc. from main()
    timings: dict              # start, end, total_s, probe_s, cold_start_s, greedy_s
    track: str = ""            # centerline stem, e.g. "a03_centerline"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


_THROTTLE_COLORS = ["#c0392b", "#95a5a6", "#27ae60"]   # brake / coast / accel
_THROTTLE_LABELS = ["brake", "coast", "accel"]


# ---------------------------------------------------------------------------
# Probe phase
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


def plot_probe_rewards(data: ExperimentData, results_dir: str) -> None:

    probes = sorted(data.probe_results, key=lambda p: p.action_idx)
    names  = [p.action_name for p in probes]
    rewards = [p.reward for p in probes]
    best_r = max(rewards)

    colors = ["#f1c40f" if r == best_r else "#3498db" for r in rewards]

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
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + abs(bar.get_height()) * 0.01,
                f"{r:+.0f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "probe_rewards.png"))


# ---------------------------------------------------------------------------
# Cold-start phase
# ---------------------------------------------------------------------------

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


def plot_cold_start_rewards(data: ExperimentData, results_dir: str) -> None:

    restarts = data.cold_start_restarts
    xs      = [r.restart for r in restarts]
    rewards = [r.best_reward for r in restarts]
    colors  = ["#27ae60" if r.beat_probe_floor else "#c0392b" for r in restarts]

    fig, ax = plt.subplots(figsize=(max(6, len(xs) * 0.8), 5))
    bars = ax.bar(xs, rewards, color=colors, edgecolor="white", linewidth=0.6)

    if data.probe_floor is not None:
        ax.axhline(data.probe_floor, color="#f39c12", linestyle="--",
                   linewidth=1.4, label=f"probe floor ({data.probe_floor:+.1f})")

    # legend patches for beat/miss
    ax.legend(handles=[
        mpatches.Patch(color="#27ae60", label="beat probe floor"),
        mpatches.Patch(color="#c0392b", label="below probe floor"),
    ] + ([ax.get_legend_handles_labels()[0][0]] if data.probe_floor is not None else []),
        fontsize=9)

    ax.set_title(f"{data.experiment_name} — Cold-Start: Best Reward per Restart")
    ax.set_xlabel("Restart")
    ax.set_ylabel("Best Episode Reward")
    ax.set_xticks(xs)

    for bar, r in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{r:+.0f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "cold_start_best_rewards.png"))


def plot_cold_start_action_dist(data: ExperimentData, results_dir: str) -> None:
    """Line graph: accel% and brake% per restart (thresholded at 0.5)."""

    restarts = data.cold_start_restarts
    xs = [r.restart for r in restarts]

    accel_pcts = []
    brake_pcts = []
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
# Greedy phase
# ---------------------------------------------------------------------------

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


def plot_greedy_rewards(data: ExperimentData, results_dir: str) -> None:

    sims    = [s.sim for s in data.greedy_sims]
    rewards = [s.reward for s in data.greedy_sims]

    # best-so-far step function
    best_so_far = []
    running_best = float("-inf")
    for r in rewards:
        if r > running_best:
            running_best = r
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


# ---------------------------------------------------------------------------
# Suggested extras
# ---------------------------------------------------------------------------

def plot_greedy_action_dist(data: ExperimentData, results_dir: str) -> None:
    """Line graph: accel% and brake% per greedy sim — shows if policy shifts toward accel."""

    sims = data.greedy_sims
    xs   = [s.sim for s in sims]
    accel_pcts = []
    brake_pcts = []
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


def plot_reward_trajectory(data: ExperimentData, results_dir: str) -> None:
    """All-phases best reward on a single cumulative-sim axis."""

    xs, ys, colors = [], [], []
    x = 0

    # Probe phase — one horizontal line segment per action (not hill-climbing)
    if data.probe_results:
        probe_sorted = sorted(data.probe_results, key=lambda p: p.action_idx)
        for p in probe_sorted:
            xs.append(x); ys.append(p.reward); colors.append("#3498db")
            x += 1

    # Cold-start — one point per sim, across all restarts
    if data.cold_start_restarts:
        for restart in data.cold_start_restarts:
            for s in restart.sims:
                xs.append(x); ys.append(s.reward); colors.append("#9b59b6")
                x += 1

    # Greedy
    for s in data.greedy_sims:
        xs.append(x); ys.append(s.reward); colors.append("#e67e22")
        x += 1

    # best-so-far overlay
    running_best = float("-inf")
    best_xs, best_ys = [], []
    for xi, yi in zip(xs, ys):
        if yi > running_best:
            running_best = yi
        best_xs.append(xi)
        best_ys.append(running_best)

    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.12), 5))
    ax.scatter(xs, ys, c=colors, s=14, alpha=0.6, zorder=2)
    ax.step(best_xs, best_ys, where="post", color="black",
            linewidth=1.8, zorder=3, label="best so far")

    # phase boundary markers
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
    ax.legend(handles=legend_patches + [ax.get_legend_handles_labels()[0][-1]],
              fontsize=9)

    ax.set_title(f"{data.experiment_name} — Reward Trajectory Across All Phases")
    ax.set_xlabel("Cumulative simulation")
    ax.set_ylabel("Total Episode Reward")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "reward_trajectory.png"))


def plot_weight_heatmap(data: ExperimentData, results_dir: str) -> None:
    """3×15 heatmap of steer/accel/brake weights from the saved policy YAML."""

    if not os.path.exists(data.weights_file):
        return

    with open(data.weights_file) as f:
        cfg = yaml.safe_load(f)

    if "steer_weights" not in cfg:
        return  # non-WLP policy (neural_net, q-table, etc.) — skip heatmap

    obs_names = WeightedLinearPolicy.OBS_NAMES
    steer_w = np.array([cfg["steer_weights"][n] for n in obs_names])
    accel_w = np.array([cfg["accel_weights"][n] for n in obs_names])
    brake_w = np.array([cfg["brake_weights"][n] for n in obs_names])
    matrix  = np.vstack([steer_w, accel_w, brake_w])

    vmax = max(abs(matrix).max(), 1e-6)

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
    """Line plot of every steer/accel/brake weight across greedy simulations.

    Each feature gets one line per sub-plot (steer / accel / brake).
    Iterations where the policy improved are marked with a vertical grey band.
    """

    sims = [s for s in data.greedy_sims
            if s.weights is not None and "steer_weights" in s.weights]
    if len(sims) < 2:
        return  # no WLP weight data (non-linear policy type) — skip evolution plot

    obs_names = WeightedLinearPolicy.OBS_NAMES
    xs = [s.sim for s in sims]
    improvement_xs = [s.sim for s in sims if s.improved]

    steer_matrix = np.array([[s.weights["steer_weights"][n] for n in obs_names] for s in sims])
    accel_matrix = np.array([[s.weights["accel_weights"][n] for n in obs_names] for s in sims])
    brake_matrix = np.array([[s.weights["brake_weights"][n] for n in obs_names] for s in sims])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    cmap = plt.cm.get_cmap("tab20", len(obs_names))

    for ax, matrix, title in zip(
        axes,
        [steer_matrix, accel_matrix, brake_matrix],
        ["Steer weights", "Accel weights", "Brake weights"],
    ):
        for ix in improvement_xs:
            ax.axvline(ix, color="grey", alpha=0.25, linewidth=1)
        for i, name in enumerate(obs_names):
            ax.plot(xs, matrix[:, i], label=name, color=cmap(i), linewidth=0.9, alpha=0.85)
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
# Entry point
# ---------------------------------------------------------------------------

def _plot_throttle_trace(ax: Axes, throttle_state: list, title: str) -> None:
    """Draw accel (green) and brake (red) as continuous line traces on *ax*.

    throttle_state entries are (accel_val, brake_val) float tuples in [0, 1].
    Both lines share a [0, 1] y-axis.
    """
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


def plot_probe_paths(data: ExperimentData, results_dir: str) -> None:
    """One path per probe action, all overlaid on a single bird's-eye plot."""

    probes = [p for p in sorted(data.probe_results, key=lambda p: p.action_idx)
              if p.trace and p.trace.pos_x]
    if not probes:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = cm.tab10(np.linspace(0, 1, len(probes)))

    for p, color in zip(probes, colors):
        ax.plot(p.trace.pos_x, p.trace.pos_z, color=color, linewidth=1.2,
                label=p.action_name, alpha=0.85)
        ax.plot(p.trace.pos_x[0], p.trace.pos_z[0], "o", color=color, markersize=5)

    ax.set_title(f"{data.experiment_name} — Probe Phase: Paths (bird's eye)")
    ax.set_xlabel("World X")
    ax.set_ylabel("World Z")
    ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "probe_paths.png"))


def _best_cold_start_trace(data: ExperimentData) -> RunTrace | None:
    """Return the RunTrace of the highest-reward sim across all cold-start restarts."""
    best_sim = None
    best_reward = float("-inf")
    for restart in data.cold_start_restarts:
        for s in restart.sims:
            if s.reward > best_reward:
                best_reward = s.reward
                best_sim = s
    return best_sim.trace if best_sim else None


def plot_cold_start_best_run(data: ExperimentData, results_dir: str) -> None:
    """Path + throttle trace for the best cold-start sim."""

    trace = _best_cold_start_trace(data)
    if not trace or not trace.pos_x:
        return

    fig, (ax_path, ax_throttle) = plt.subplots(1, 2, figsize=(14, 6))

    ax_path.plot(trace.pos_x, trace.pos_z, color="#9b59b6", linewidth=1.4)
    ax_path.plot(trace.pos_x[0], trace.pos_z[0], "o", color="#9b59b6", markersize=6)
    ax_path.set_title("Path (bird's eye)")
    ax_path.set_xlabel("World X")
    ax_path.set_ylabel("World Z")
    ax_path.set_aspect("equal", adjustable="datalim")

    _plot_throttle_trace(ax_throttle, trace.throttle_state,
                         f"Throttle/brake trace  (reward {trace.total_reward:+.1f})")

    fig.suptitle(f"{data.experiment_name} — Cold-Start Best Run", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "cold_start_best_run.png"))


def plot_greedy_best_run(data: ExperimentData, results_dir: str) -> None:
    """Path + throttle trace for the highest-reward greedy sim."""

    if not data.greedy_sims:
        return
    best = max(data.greedy_sims, key=lambda s: s.reward)
    trace = best.trace
    if not trace or not trace.pos_x:
        return

    fig, (ax_path, ax_throttle) = plt.subplots(1, 2, figsize=(14, 6))

    ax_path.plot(trace.pos_x, trace.pos_z, color="#e67e22", linewidth=1.4)
    ax_path.plot(trace.pos_x[0], trace.pos_z[0], "o", color="#e67e22", markersize=6)
    ax_path.set_title("Path (bird's eye)")
    ax_path.set_xlabel("World X")
    ax_path.set_ylabel("World Z")
    ax_path.set_aspect("equal", adjustable="datalim")

    _plot_throttle_trace(ax_throttle, trace.throttle_state,
                         f"Throttle/brake trace  (reward {trace.total_reward:+.1f})")

    fig.suptitle(f"{data.experiment_name} — Greedy Best Run (sim {best.sim})", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "greedy_best_run.png"))


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

    lines = ["## Run Parameters\n\n"]

    # Training hyperparameters
    lines.append("### Training\n\n")
    lines.append("| Parameter | Value |\n")
    lines.append("|-----------|-------|\n")
    if data.track:
        lines.append(f"| track | {data.track} |\n")
    for k, v in data.training_params.items():
        lines.append(f"| {k} | {v} |\n")

    # Reward config
    lines.append("\n### Reward Config\n\n")
    if os.path.exists(data.reward_config_file):
        with open(data.reward_config_file) as f:
            cfg = yaml.safe_load(f)
        lines.append("| Parameter | Value |\n")
        lines.append("|-----------|-------|\n")
        for k, v in cfg.items():
            lines.append(f"| {k} | {v} |\n")
    else:
        lines.append(f"_(reward config not found at `{data.reward_config_file}`)_\n")

    return "".join(lines) + "\n"


def _effective_progress(s: GreedySimResult) -> float:
    """laps_completed + final_track_progress — continuous progress metric."""
    return s.laps_completed + s.final_track_progress


def plot_greedy_progress(data: ExperimentData, results_dir: str) -> None:
    """Scatter + best-so-far line of effective track progress per greedy sim.

    Effective progress = laps_completed + final_track_progress, so finishing
    one full lap and then reaching 50% of lap 2 = 1.5.
    """

    sims = data.greedy_sims
    xs   = [s.sim for s in sims]
    ys   = [_effective_progress(s) for s in sims]

    best_so_far = []
    running_best = 0.0
    for y in ys:
        if y > running_best:
            running_best = y
        best_so_far.append(running_best)

    improvement_xs = [s.sim for s in sims if s.improved]
    improvement_ys = [_effective_progress(s) for s in sims if s.improved]

    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.15), 5))
    ax.scatter(xs, ys, color="#95a5a6", s=18, alpha=0.7, zorder=2, label="candidate progress")
    ax.step(xs, best_so_far, where="post", color="#e67e22",
            linewidth=2.0, zorder=3, label="best so far")
    ax.scatter(improvement_xs, improvement_ys, color="#27ae60",
               s=60, zorder=4, marker="^", label="improvement")

    # Reference lines at each full lap
    max_prog = max(ys) if ys else 1.0
    for lap in range(1, int(max_prog) + 2):
        if lap <= max_prog + 0.1:
            ax.axhline(lap, color="#bdc3c7", linestyle="--", linewidth=0.8, zorder=1)
            ax.text(xs[-1] + 0.3, lap, f"lap {lap}", va="center", fontsize=7, color="#7f8c8d")

    ax.set_title(f"{data.experiment_name} — Greedy Phase: Track Progress per Sim")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Effective progress (laps + fraction)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "greedy_progress.png"))


def save_experiment_results(data: ExperimentData, results_dir: str) -> None:
    """Generate all plots and write a single results.md report to *results_dir*."""
    os.makedirs(results_dir, exist_ok=True)

    track_line = f"\n**Track:** {data.track}\n" if data.track else ""
    sections = [f"# Experiment: {data.experiment_name}\n{track_line}\n", _timings_md(data), _summary_md(data)]

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
        sections.append(_greedy_table_md(data))
        sections.append("\n![Greedy rewards](greedy_rewards.png)\n\n")
        sections.append("![Greedy progress](greedy_progress.png)\n\n")
        sections.append("![Greedy best run](greedy_best_run.png)\n\n")
        sections.append("![Weight evolution](greedy_weight_evolution.png)\n\n")

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


# ---------------------------------------------------------------------------
# Grid search summary
# ---------------------------------------------------------------------------

def _gs_stats(data: ExperimentData) -> dict:
    """Derive key summary statistics from one ExperimentData."""
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


def plot_gs_comparison_rewards(
    runs: list[tuple[str, dict]],
    summary_dir: str,
) -> None:
    """Horizontal bar chart of best greedy reward per experiment, sorted descending."""

    runs_sorted = sorted(runs, key=lambda x: x[1]["best_reward"])
    names   = [r[0] for r in runs_sorted]
    rewards = [r[1]["best_reward"] for r in runs_sorted]

    n = len(names)
    colors = cm.RdYlGn(np.linspace(0.15, 0.85, n))

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.45)))
    bars = ax.barh(names, rewards, color=colors, edgecolor="white", linewidth=0.5)

    for bar, r in zip(bars, rewards):
        ax.text(r, bar.get_y() + bar.get_height() / 2,
                f"  {r:+.1f}", va="center", fontsize=8)

    ax.set_xlabel("Best Greedy Reward")
    ax.set_title("Grid Search — Best Reward per Experiment")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    _save(fig, os.path.join(summary_dir, "comparison_rewards.png"))


def plot_gs_comparison_paths(
    runs: list[tuple[str, ExperimentData]],
    summary_dir: str,
) -> None:
    """All experiments' best-run paths overlaid; coloured green→red by reward rank."""

    # Filter to runs that have a usable trace in the best greedy sim
    traced = []
    for name, data in runs:
        if not data.greedy_sims:
            continue
        best_sim = max(data.greedy_sims, key=lambda s: s.reward)
        if best_sim.trace and best_sim.trace.pos_x:
            traced.append((name, best_sim.reward, best_sim.trace))

    if not traced:
        return

    traced.sort(key=lambda x: x[1])  # ascending reward → red first, green on top
    n = len(traced)
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
    """Best-so-far effective progress vs simulation number, one line per experiment.

    Shows how quickly and how far each configuration's policy learned to drive.
    Coloured green→red by final best progress (green = furthest).
    """

    # Compute best-so-far progress series per experiment
    series = []
    for name, data in runs:
        if not data.greedy_sims:
            continue
        xs = [s.sim for s in data.greedy_sims]
        running_best = 0.0
        ys = []
        for s in data.greedy_sims:
            p = _effective_progress(s)
            if p > running_best:
                running_best = p
            ys.append(running_best)
        series.append((name, xs, ys, running_best))

    if not series:
        return

    # Sort by final best progress so legend is ranked
    series.sort(key=lambda x: -x[3])
    n = len(series)
    colors = cm.RdYlGn(np.linspace(0.15, 0.85, n))

    fig, ax = plt.subplots(figsize=(12, 6))
    for (name, xs, ys, final_best), color in zip(series, colors):
        ax.step(xs, ys, where="post", color=color, linewidth=1.4, alpha=0.85,
                label=f"{name}  ({final_best:.2f})")

    # Reference lines at full laps
    max_prog = max(s[3] for s in series) if series else 1.0
    for lap in range(1, int(max_prog) + 2):
        if lap <= max_prog + 0.1:
            ax.axhline(lap, color="#bdc3c7", linestyle="--", linewidth=0.7, zorder=1)
            ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 1,
                    lap, f" lap {lap}", va="center", fontsize=7, color="#7f8c8d")

    ax.set_title("Grid Search — Best-So-Far Track Progress per Simulation\n"
                 "(green = further; line = best seen up to that sim)")
    ax.set_xlabel("Simulation #")
    ax.set_ylabel("Effective progress (laps + fraction)")
    ax.legend(fontsize=6, loc="upper left", framealpha=0.7,
              ncol=max(1, n // 12))
    fig.tight_layout()
    _save(fig, os.path.join(summary_dir, "comparison_progress.png"))


def save_grid_summary(
    runs: list[tuple[str, ExperimentData]],
    varied_keys: list[str],
    summary_dir: str,
    base_name: str,
) -> None:
    """Generate a cross-experiment summary report and plots in *summary_dir*.

    Produces:
        summary.md                  — ranked table + per-experiment sections
        comparison_rewards.png      — horizontal bar chart of best rewards
        comparison_paths.png        — all best-run paths overlaid
    """
    os.makedirs(summary_dir, exist_ok=True)

    # Compute stats for every run
    stats = [(name, _gs_stats(data)) for name, data in runs]
    ranked = sorted(stats, key=lambda x: -x[1]["best_reward"])

    # --- Plots ---
    plot_gs_comparison_rewards(
        [(name, s) for name, s in ranked],
        summary_dir,
    )
    plot_gs_comparison_paths(runs, summary_dir)
    plot_gs_comparison_progress(runs, summary_dir)

    # --- Markdown ---
    lines = [
        f"# Grid Search Summary: {base_name}\n\n",
        f"{len(runs)} experiments, ranked by best greedy reward.\n\n",
        "![Reward comparison](comparison_rewards.png)\n\n",
        "![Progress over training](comparison_progress.png)\n\n",
        "![Path comparison](comparison_paths.png)\n\n",
    ]

    # Rankings table
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

    # Per-experiment sections (in rank order)
    for rank, (name, data) in enumerate(
        sorted(runs, key=lambda x: -_gs_stats(x[1])["best_reward"]), 1
    ):
        s = _gs_stats(data)
        results_rel = f"../{name}/results"  # relative path from summary_dir

        lines.append(f"---\n\n## {rank}. {name}\n\n**Best reward: {s['best_reward']:+.1f}**\n\n")

        # Varied-param values for this experiment
        if varied_keys and data.training_params:
            reward_cfg = {}
            if os.path.exists(data.reward_config_file):
                with open(data.reward_config_file) as f:
                    reward_cfg = yaml.safe_load(f) or {}
            all_params = {**data.training_params, **reward_cfg}

            lines.append("| Param | Value |\n|---|---|\n")
            for k in varied_keys:
                v = all_params.get(k, "?")
                lines.append(f"| `{k}` | {v} |\n")
            lines.append("\n")

        # Key stats
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

        # Images — link to per-experiment results
        best_run_img = f"{results_rel}/greedy_best_run.png"
        weight_evo_img = f"{results_rel}/greedy_weight_evolution.png"
        reward_traj_img = f"{results_rel}/reward_trajectory.png"

        lines += [
            f"![Best run path + throttle]({best_run_img})\n\n",
            f"![Weight evolution]({weight_evo_img})\n\n",
            f"![Reward trajectory]({reward_traj_img})\n\n",
        ]

    report_path = os.path.join(summary_dir, "summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    n = len(os.listdir(summary_dir))
    logger.info("Saved %d file(s) to %s/ (report: summary.md)", n, summary_dir)
