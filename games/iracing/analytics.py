"""iRacing-specific analytics.

Entry point called by main.py::

    save_experiment_results(data: ExperimentData, results_dir: str) -> None

iRacing exposes the richest telemetry of the racing games (lap times, tyre
loads/temps, fuel, RPM, brake bias).  The headline metric is lap-time
improvement, and secondary plots cover throttle/brake distribution and
termination reasons.

Obs-average panels (tyre temps/loads, fuel, RPM, brake bias) use
``GreedySimResult.obs_averages`` when the env populates
``episode_obs_averages`` in terminal-step info; they are silently skipped
when that dict is absent.
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib
import numpy as np

if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from framework.analytics import (
    ExperimentData,
    _cold_start_table_md,
    _greedy_table_md,
    _probe_table_md,
    _summary_md,
    _timings_md,
    plot_cold_start_rewards,
    plot_greedy_rewards,
    plot_probe_rewards,
    plot_reward_trajectory,
)

logger = logging.getLogger(__name__)

_THROTTLE_COLORS = ["#c0392b", "#95a5a6", "#27ae60"]
_REASON_COLORS = {
    "finish": "#27ae60",
    "crash": "#c0392b",
    "timeout": "#f39c12",
}
_REASON_ORDER = ["finish", "crash", "timeout"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _save(fig: "Figure", path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Lap-time improvement (headline metric for iRacing)
# ---------------------------------------------------------------------------


def plot_lap_time_improvement(data: ExperimentData, results_dir: str) -> bool:
    """Plot finished-lap times over greedy sims with a best-so-far curve.

    Only sims where ``finish_time_s`` is not None (i.e. the episode ended
    with a completed lap) contribute a data point.  DNF/crash/timeout sims
    are shown as vertical markers without a lap time value.
    """
    sims = data.greedy_sims
    if not sims:
        return False

    finished_xs = [s.sim for s in sims if s.finish_time_s is not None]
    finished_ys = [s.finish_time_s for s in sims if s.finish_time_s is not None]
    dnf_xs = [s.sim for s in sims if s.finish_time_s is None]

    if not finished_xs:
        return False

    best_so_far: list[float] = []
    running_best = float("inf")
    for t in finished_ys:
        running_best = min(running_best, t)
        best_so_far.append(running_best)

    fig, ax = plt.subplots(figsize=(max(8, len(sims) * 0.15), 5))
    ax.scatter(
        finished_xs,
        finished_ys,
        color="#3498db",
        s=25,
        alpha=0.8,
        zorder=3,
        label="lap time (finished)",
    )
    ax.step(
        finished_xs,
        best_so_far,
        where="post",
        color="#e67e22",
        linewidth=2.0,
        zorder=4,
        label="best so far",
    )
    improved_xs = [s.sim for s in sims if s.improved and s.finish_time_s is not None]
    improved_ys = [s.finish_time_s for s in sims if s.improved and s.finish_time_s is not None]
    if improved_xs:
        ax.scatter(
            improved_xs,
            improved_ys,
            color="#27ae60",
            s=60,
            marker="^",
            zorder=5,
            label="improvement",
        )
    for x in dnf_xs:
        ax.axvline(x, color="#c0392b", alpha=0.25, linewidth=0.8)

    ax.set_title(f"{data.experiment_name} — iRacing Lap Time Improvement")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Lap time (s)")
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "iracing_lap_time_improvement.png"))
    return True


# ---------------------------------------------------------------------------
# Throttle / brake distribution
# ---------------------------------------------------------------------------


def plot_greedy_action_dist(data: ExperimentData, results_dir: str) -> bool:
    """Plot per-sim accel and brake percentage over the greedy phase."""
    sims = data.greedy_sims
    if not sims:
        return False

    xs = [s.sim for s in sims]
    accel_pcts, brake_pcts = [], []
    for s in sims:
        b, c, a = s.throttle_counts
        total = (b + c + a) or 1
        accel_pcts.append(100 * a / total)
        brake_pcts.append(100 * b / total)

    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.15), 5))
    ax.plot(
        xs,
        accel_pcts,
        color=_THROTTLE_COLORS[2],
        linewidth=1.2,
        alpha=0.85,
        label="accel %",
    )
    ax.plot(
        xs,
        brake_pcts,
        color=_THROTTLE_COLORS[0],
        linewidth=1.2,
        alpha=0.85,
        label="brake %",
    )
    ax.set_title(f"{data.experiment_name} — iRacing: Accel / Brake % per Sim")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("% Steps (threshold ≥ 0.5)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "iracing_action_dist.png"))
    return True


# ---------------------------------------------------------------------------
# Obs-averages plots (tire temps, RPM, gear, fuel, brake bias)
# ---------------------------------------------------------------------------

_OBS_AVG_PLOTS: list[tuple[list[str], str, str, str]] = [
    # (feature_names, y_label, title_suffix, filename)
    (
        ["tire_temp_fl", "tire_temp_fr", "tire_temp_rl", "tire_temp_rr"],
        "Mean temp (°C)",
        "Mean Tyre Temperature per Sim",
        "iracing_tire_temps.png",
    ),
    (
        ["tire_load_fl", "tire_load_fr", "tire_load_rl", "tire_load_rr"],
        "Mean load (N)",
        "Mean Tyre Load per Sim",
        "iracing_tire_loads.png",
    ),
    (
        ["fuel_pct"],
        "Mean fuel fraction",
        "Mean Fuel Level per Sim",
        "iracing_fuel.png",
    ),
    (
        ["rpm"],
        "Mean RPM",
        "Mean RPM per Sim",
        "iracing_rpm.png",
    ),
    (
        ["brake_bias"],
        "Brake bias",
        "Mean Brake Bias per Sim",
        "iracing_brake_bias.png",
    ),
]

_CORNER_COLORS = ["#3498db", "#e67e22", "#27ae60", "#9b59b6"]
_CORNER_LABELS = ["FL", "FR", "RL", "RR"]


def _plot_obs_avg_series(
    data: ExperimentData,
    results_dir: str,
    feature_names: list[str],
    y_label: str,
    title_suffix: str,
    filename: str,
) -> bool:
    sims = [s for s in data.greedy_sims if s.obs_averages]
    if not sims:
        return False
    xs = [s.sim for s in sims]
    series: list[list[float]] = []
    for feat in feature_names:
        ys = [s.obs_averages.get(feat) for s in sims]  # type: ignore[union-attr]
        if any(v is not None for v in ys):
            series.append([float(v) if v is not None else float("nan") for v in ys])
        else:
            series.append([])

    if not any(s for s in series):
        return False

    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.15), 4))
    multi = len(feature_names) > 1
    for i, (feat, ys) in enumerate(zip(feature_names, series)):
        if not ys:
            continue
        label = _CORNER_LABELS[i] if multi and i < len(_CORNER_LABELS) else feat
        color = _CORNER_COLORS[i % len(_CORNER_COLORS)]
        ax.plot(xs, ys, color=color, linewidth=1.2, marker="o", markersize=3, label=label)

    ax.set_title(f"{data.experiment_name} — iRacing: {title_suffix}")
    ax.set_xlabel("Simulation")
    ax.set_ylabel(y_label)
    if multi:
        ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, filename))
    return True


def plot_obs_avg_panels(data: ExperimentData, results_dir: str) -> list[str]:
    """Generate all obs-average plots; return list of written filenames."""
    written: list[str] = []
    for features, ylabel, title, fname in _OBS_AVG_PLOTS:
        if _plot_obs_avg_series(data, results_dir, features, ylabel, title, fname):
            written.append(fname)
    return written


# ---------------------------------------------------------------------------
# Termination reason breakdown
# ---------------------------------------------------------------------------


def plot_termination_reasons(data: ExperimentData, results_dir: str) -> bool:
    sims = data.greedy_sims
    if not sims:
        return False

    counts: dict[str, int] = {}
    for s in sims:
        r = s.termination_reason or "unknown"
        counts[r] = counts.get(r, 0) + 1

    order = _REASON_ORDER + sorted(k for k in counts if k not in _REASON_ORDER)
    labels = [r for r in order if r in counts]
    values = [counts[r] for r in labels]
    colors = [_REASON_COLORS.get(r, "#95a5a6") for r in labels]
    total = len(sims)

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.4), 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(total * 0.005, 0.3),
            f"{v} ({100 * v / total:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(f"{data.experiment_name} — iRacing: Termination Reasons ({total} sims)")
    ax.set_xlabel("Reason")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "iracing_termination_reasons.png"))
    return True


# ---------------------------------------------------------------------------
# iRacing summary table (markdown)
# ---------------------------------------------------------------------------


def _iracing_summary_md(data: ExperimentData) -> str:
    sims = data.greedy_sims
    if not sims:
        return ""

    finished = [s for s in sims if s.finish_time_s is not None]
    finish_rate = len(finished) / len(sims)
    lines = [
        "## iRacing Metrics\n\n",
        "| Metric | Value |\n",
        "|--------|-------|\n",
        f"| Finish rate | {finish_rate:.1%} ({len(finished)}/{len(sims)} sims) |\n",
    ]
    if finished:
        best_lap = min(s.finish_time_s for s in finished)  # type: ignore[arg-type]
        mean_lap = float(np.mean([s.finish_time_s for s in finished]))
        lines += [
            f"| Best lap time | {best_lap:.3f} s |\n",
            f"| Mean lap time (finished) | {mean_lap:.3f} s |\n",
        ]

    accel_pcts = []
    for s in sims:
        b, c, a = s.throttle_counts
        total = (b + c + a) or 1
        accel_pcts.append(100 * a / total)
    lines.append(f"| Mean accel % | {float(np.mean(accel_pcts)):.1f}% |\n")

    return "".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def save_experiment_results(data: ExperimentData, results_dir: str) -> None:
    """Generate all plots and write a results.md report to *results_dir*."""
    os.makedirs(results_dir, exist_ok=True)

    sections = [
        f"# Experiment: {data.experiment_name}\n\n**Game:** iRacing\n\n",
        _timings_md(data),
        _summary_md(data),
    ]

    if data.probe_results:
        plot_probe_rewards(data, results_dir)
        sections.append(_probe_table_md(data))
        sections.append("\n![Probe rewards](probe_rewards.png)\n\n")

    if data.cold_start_restarts:
        plot_cold_start_rewards(data, results_dir)
        sections.append(_cold_start_table_md(data))
        sections.append("\n![Cold-start best rewards](cold_start_best_rewards.png)\n\n")

    if data.greedy_sims:
        plot_greedy_rewards(data, results_dir)
        sections.append(_greedy_table_md(data))
        sections.append("\n![Greedy rewards](greedy_rewards.png)\n\n")

    wrote_lap_time = plot_lap_time_improvement(data, results_dir)
    wrote_action_dist = plot_greedy_action_dist(data, results_dir)
    wrote_termination = plot_termination_reasons(data, results_dir)
    obs_avg_files = plot_obs_avg_panels(data, results_dir)

    sections.append(_iracing_summary_md(data))
    plot_reward_trajectory(data, results_dir)

    sections.append("## Additional Plots\n\n")
    if wrote_lap_time:
        sections.append("![iRacing lap time improvement](iracing_lap_time_improvement.png)\n\n")
    if wrote_action_dist:
        sections.append("![iRacing action distribution](iracing_action_dist.png)\n\n")
    if wrote_termination:
        sections.append("![iRacing termination reasons](iracing_termination_reasons.png)\n\n")
    for fname in obs_avg_files:
        label = fname.replace("iracing_", "").replace(".png", "").replace("_", " ").title()
        sections.append(f"![iRacing {label}]({fname})\n\n")
    sections.append("![Reward trajectory](reward_trajectory.png)\n\n")

    report_path = os.path.join(results_dir, "results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(sections).rstrip("\n") + "\n")

    n = len(os.listdir(results_dir))
    logger.info("Saved %d file(s) to %s/ (report: results.md)", n, results_dir)
