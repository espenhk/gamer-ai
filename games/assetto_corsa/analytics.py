"""Assetto Corsa-specific analytics (issue #464).

Entry point called by main.py::

    save_experiment_results(data: ExperimentData, results_dir: str) -> None

Assetto Corsa is a TMNF-style racing game whose distinctive telemetry is the
per-wheel slip ratio (traction/grip analysis), plus engine RPM and gear.
The shared racing plots (track progress, best-run throttle trace, action
distribution, termination reasons) are reused from ``games/torcs/analytics.py``
— they only touch generic ``ExperimentData`` / ``GreedySimResult`` fields.

The AC-specific panels (wheel slip, RPM/gear, centerline distribution) read
``GreedySimResult.obs_averages``, which the AC env populates via
``episode_obs_averages`` in terminal-step info; they are silently skipped for
old experiment data recorded before that field existed.
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib
import yaml

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
from framework.analytics import (
    save_grid_summary as _framework_save_grid_summary,
)
from games.torcs.analytics import (
    plot_cold_start_action_dist,
    plot_cold_start_best_run,
    plot_greedy_action_dist,
    plot_greedy_best_run,
    plot_greedy_progress,
    plot_termination_reasons,
)

logger = logging.getLogger(__name__)

_CORNER_COLORS = ["#3498db", "#e67e22", "#27ae60", "#9b59b6"]
_CORNER_LABELS = ["FL", "FR", "RL", "RR"]
_WHEEL_SLIP_KEYS = ["wheel_0_slip", "wheel_1_slip", "wheel_2_slip", "wheel_3_slip"]


def _save(fig: "Figure", path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _obs_avg_sims(data: ExperimentData) -> list:
    return [s for s in data.greedy_sims if s.obs_averages]


# ---------------------------------------------------------------------------
# Wheel slip (the distinctive AC signal)
# ---------------------------------------------------------------------------


def plot_wheel_slip(data: ExperimentData, results_dir: str) -> bool:
    """Per-wheel mean slip ratio per sim — traction/grip analysis."""
    sims = _obs_avg_sims(data)
    if not sims:
        return False
    xs = [s.sim for s in sims]
    any_series = False

    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.15), 4))
    for i, key in enumerate(_WHEEL_SLIP_KEYS):
        ys = [s.obs_averages.get(key) for s in sims]
        if not any(v is not None for v in ys):
            continue
        any_series = True
        ax.plot(
            xs,
            [float(v) if v is not None else float("nan") for v in ys],
            color=_CORNER_COLORS[i],
            linewidth=1.2,
            marker="o",
            markersize=3,
            label=_CORNER_LABELS[i],
        )
    if not any_series:
        plt.close(fig)
        return False

    ax.set_title(f"{data.experiment_name} — AC: Mean Wheel Slip per Sim")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Mean slip ratio")
    ax.legend(fontsize=9, title="wheel")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "ac_wheel_slip.png"))
    return True


# ---------------------------------------------------------------------------
# RPM / gear usage
# ---------------------------------------------------------------------------


def plot_rpm_gear(data: ExperimentData, results_dir: str) -> bool:
    """Mean engine RPM and mean gear per sim, side by side."""
    sims = _obs_avg_sims(data)
    if not sims:
        return False
    xs = [s.sim for s in sims]
    rpm = [s.obs_averages.get("engine_rpm") for s in sims]
    gear = [s.obs_averages.get("gear") for s in sims]
    has_rpm = any(v is not None for v in rpm)
    has_gear = any(v is not None for v in gear)
    if not (has_rpm or has_gear):
        return False

    fig, (ax_rpm, ax_gear) = plt.subplots(1, 2, figsize=(max(10, len(xs) * 0.2), 4))
    if has_rpm:
        ax_rpm.plot(
            xs,
            [float(v) if v is not None else float("nan") for v in rpm],
            color="#c0392b",
            linewidth=1.2,
            marker="o",
            markersize=3,
        )
    ax_rpm.set_title("Mean engine RPM per sim")
    ax_rpm.set_xlabel("Simulation")
    ax_rpm.set_ylabel("RPM")
    if has_gear:
        ax_gear.plot(
            xs,
            [float(v) if v is not None else float("nan") for v in gear],
            color="#8e44ad",
            linewidth=1.2,
            marker="o",
            markersize=3,
        )
    ax_gear.set_title("Mean gear per sim (0=R, 1=N, 2..7 fwd)")
    ax_gear.set_xlabel("Simulation")
    ax_gear.set_ylabel("Gear")
    fig.suptitle(f"{data.experiment_name} — AC: RPM / Gear Usage", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "ac_rpm_gear.png"))
    return True


# ---------------------------------------------------------------------------
# Centerline distribution
# ---------------------------------------------------------------------------


def plot_centerline_distribution(data: ExperimentData, results_dir: str) -> bool:
    """Histogram of per-sim mean |lateral offset| across the greedy phase."""
    values = [s.mean_abs_lateral_offset for s in data.greedy_sims if s.mean_abs_lateral_offset is not None]
    if not values:
        # Fall back to the obs-average key when the generic field is absent.
        values = [
            s.obs_averages["abs_lateral_offset_m"]
            for s in _obs_avg_sims(data)
            if "abs_lateral_offset_m" in s.obs_averages
        ]
    if not values:
        return False

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=min(20, max(5, len(values) // 2)), color="#2980b9", edgecolor="white")
    ax.set_title(f"{data.experiment_name} — AC: Mean |Lateral Offset| Distribution")
    ax.set_xlabel("Mean |lateral offset| per sim (m)")
    ax.set_ylabel("Sims")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "ac_centerline_dist.png"))
    return True


# ---------------------------------------------------------------------------
# Lap time vs par
# ---------------------------------------------------------------------------


def plot_lap_time_progression(data: ExperimentData, results_dir: str) -> bool:
    """Finish time of each finished sim over the greedy phase, vs par_time_s."""
    finished = [(s.sim, s.finish_time_s) for s in data.greedy_sims if s.finish_time_s is not None]
    if not finished:
        return False
    xs = [sim for sim, _ in finished]
    ys = [t for _, t in finished]

    # Best-effort par line: a missing, malformed, or non-numeric reward config
    # must never crash results generation — the line is simply skipped.
    par_time = None
    try:
        if data.reward_config_file and os.path.exists(data.reward_config_file):
            with open(data.reward_config_file, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            loaded = cfg.get("par_time_s") if isinstance(cfg, dict) else None
            if loaded is not None:
                par_time = float(loaded)
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        par_time = None

    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.3), 4))
    ax.plot(xs, ys, color="#16a085", linewidth=1.4, marker="o", markersize=4, label="finish time")
    best_so_far = []
    running = float("inf")
    for t in ys:
        running = min(running, t)
        best_so_far.append(running)
    ax.step(xs, best_so_far, where="post", color="#e67e22", linewidth=1.8, label="best so far")
    if par_time is not None:
        ax.axhline(float(par_time), color="#7f8c8d", linestyle="--", linewidth=1.0, label=f"par ({par_time:g}s)")
    ax.set_title(f"{data.experiment_name} — AC: Lap Time Progression (finished sims)")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Finish time (s)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "ac_lap_times.png"))
    return True


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def save_experiment_results(data: ExperimentData, results_dir: str) -> None:
    """Generate all plots and write a results.md report into *results_dir*."""
    os.makedirs(results_dir, exist_ok=True)

    track_line = f"\n**Track:** {data.track}\n" if data.track else ""
    sections = [
        f"# Experiment: {data.experiment_name}\n\n**Game:** Assetto Corsa\n{track_line}\n",
        _timings_md(data),
        _summary_md(data),
    ]

    if data.probe_results:
        plot_probe_rewards(data, results_dir)
        sections.append(_probe_table_md(data))
        sections.append("\n![Probe rewards](probe_rewards.png)\n\n")

    if data.cold_start_restarts:
        plot_cold_start_rewards(data, results_dir)
        plot_cold_start_action_dist(data, results_dir)
        plot_cold_start_best_run(data, results_dir)
        sections.append(_cold_start_table_md(data))
        sections.append("\n![Cold-start best rewards](cold_start_best_rewards.png)\n\n")
        sections.append("![Cold-start action distribution](cold_start_action_dist.png)\n\n")
        if os.path.exists(os.path.join(results_dir, "cold_start_best_run.png")):
            sections.append("![Cold-start best run](cold_start_best_run.png)\n\n")

    if data.greedy_sims:
        plot_greedy_rewards(data, results_dir)
        plot_greedy_progress(data, results_dir)
        plot_greedy_action_dist(data, results_dir)
        plot_greedy_best_run(data, results_dir)
        plot_termination_reasons(data, results_dir)
        sections.append(_greedy_table_md(data))
        sections.append("\n![Greedy rewards](greedy_rewards.png)\n\n")
        sections.append("![Greedy progress](greedy_progress.png)\n\n")
        sections.append("![Greedy action distribution](greedy_action_dist.png)\n\n")
        if os.path.exists(os.path.join(results_dir, "greedy_best_run.png")):
            sections.append("![Greedy best run](greedy_best_run.png)\n\n")
        sections.append("![Termination reasons](termination_reasons.png)\n\n")

    ac_plots = [
        (plot_wheel_slip, "ac_wheel_slip.png", "Wheel slip (traction)"),
        (plot_rpm_gear, "ac_rpm_gear.png", "RPM / gear usage"),
        (plot_centerline_distribution, "ac_centerline_dist.png", "Centerline distribution"),
        (plot_lap_time_progression, "ac_lap_times.png", "Lap time progression"),
    ]
    ac_sections = []
    for plot_fn, fname, label in ac_plots:
        if plot_fn(data, results_dir):
            ac_sections.append(f"![{label}]({fname})\n\n")
    if ac_sections:
        sections.append("## Assetto Corsa Plots\n\n")
        sections.extend(ac_sections)

    plot_reward_trajectory(data, results_dir)
    sections.append("## Additional Plots\n\n")
    sections.append("![Reward trajectory](reward_trajectory.png)\n\n")

    report_path = os.path.join(results_dir, "results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(sections).rstrip("\n") + "\n")

    n = len(os.listdir(results_dir))
    logger.info("Saved %d file(s) to %s/ (report: results.md)", n, results_dir)


def save_grid_summary(*args, **kwargs) -> None:
    """AC grid-summary wrapper: framework defaults, no extra plots."""
    _framework_save_grid_summary(*args, **kwargs)
