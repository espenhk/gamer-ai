"""BeamNG-specific analytics (issue #462).

Entry point called by main.py::

    save_experiment_results(data: ExperimentData, results_dir: str) -> None

BeamNG is a TMNF-style racing game; the shared racing plots (track progress,
best-run throttle trace, action distribution, termination reasons) are reused
from ``games/torcs/analytics.py`` — they only touch generic ``ExperimentData``
/ ``GreedySimResult`` fields.

The BeamNG-specific panels (airborne fraction / wheel contact, mean speed,
centerline distribution) read ``GreedySimResult.obs_averages``, which the
BeamNG env populates via ``episode_obs_averages`` in terminal-step info; they
are silently skipped for old experiment data recorded before that field
existed.
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
_WHEEL_CONTACT_KEYS = ["wheel_0_contact", "wheel_1_contact", "wheel_2_contact", "wheel_3_contact"]


def _save(fig: "Figure", path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _obs_avg_sims(data: ExperimentData) -> list:
    return [s for s in data.greedy_sims if s.obs_averages]


# ---------------------------------------------------------------------------
# Airborne fraction / wheel contact (the distinctive BeamNG signal)
# ---------------------------------------------------------------------------


def plot_airborne(data: ExperimentData, results_dir: str) -> bool:
    """Airborne fraction and per-wheel mean ground contact per sim."""
    sims = _obs_avg_sims(data)
    if not sims:
        return False
    xs = [s.sim for s in sims]
    airborne = [s.obs_averages.get("airborne") for s in sims]
    has_airborne = any(v is not None for v in airborne)
    contact_series = []
    for key in _WHEEL_CONTACT_KEYS:
        ys = [s.obs_averages.get(key) for s in sims]
        contact_series.append(ys if any(v is not None for v in ys) else None)
    has_contacts = any(s is not None for s in contact_series)
    if not (has_airborne or has_contacts):
        return False

    fig, (ax_air, ax_contact) = plt.subplots(1, 2, figsize=(max(10, len(xs) * 0.2), 4))
    if has_airborne:
        ax_air.plot(
            xs,
            [float(v) if v is not None else float("nan") for v in airborne],
            color="#c0392b",
            linewidth=1.2,
            marker="o",
            markersize=3,
        )
    ax_air.set_title("Airborne fraction per sim (≤1 wheel grounded)")
    ax_air.set_xlabel("Simulation")
    ax_air.set_ylabel("Fraction of steps airborne")
    ax_air.set_ylim(0, 1)
    for i, ys in enumerate(contact_series):
        if ys is None:
            continue
        ax_contact.plot(
            xs,
            [float(v) if v is not None else float("nan") for v in ys],
            color=_CORNER_COLORS[i],
            linewidth=1.2,
            marker="o",
            markersize=3,
            label=_CORNER_LABELS[i],
        )
    ax_contact.set_title("Mean wheel ground contact per sim")
    ax_contact.set_xlabel("Simulation")
    ax_contact.set_ylabel("Mean contact (0-1)")
    ax_contact.set_ylim(0, 1)
    if has_contacts:
        ax_contact.legend(fontsize=9, title="wheel")
    fig.suptitle(f"{data.experiment_name} — BeamNG: Airborne / Wheel Contact", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "beamng_airborne.png"))
    return True


# ---------------------------------------------------------------------------
# Mean speed
# ---------------------------------------------------------------------------


def plot_mean_speed(data: ExperimentData, results_dir: str) -> bool:
    """Mean speed per sim over the greedy phase."""
    sims = _obs_avg_sims(data)
    if not sims:
        return False
    speed = [s.obs_averages.get("speed_ms") for s in sims]
    if not any(v is not None for v in speed):
        return False
    xs = [s.sim for s in sims]

    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.15), 4))
    ax.plot(
        xs,
        [float(v) if v is not None else float("nan") for v in speed],
        color="#16a085",
        linewidth=1.2,
        marker="o",
        markersize=3,
    )
    ax.set_title(f"{data.experiment_name} — BeamNG: Mean Speed per Sim")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Mean speed (m/s)")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "beamng_mean_speed.png"))
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
    ax.set_title(f"{data.experiment_name} — BeamNG: Mean |Lateral Offset| Distribution")
    ax.set_xlabel("Mean |lateral offset| per sim (m)")
    ax.set_ylabel("Sims")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "beamng_centerline_dist.png"))
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
        ax.axhline(par_time, color="#7f8c8d", linestyle="--", linewidth=1.0, label=f"par ({par_time:g}s)")
    ax.set_title(f"{data.experiment_name} — BeamNG: Lap Time Progression (finished sims)")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Finish time (s)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "beamng_lap_times.png"))
    return True


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def save_experiment_results(data: ExperimentData, results_dir: str) -> None:
    """Generate all plots and write a results.md report to *results_dir*."""
    os.makedirs(results_dir, exist_ok=True)

    sections = [
        f"# Experiment: {data.experiment_name}\n\n**Game:** BeamNG.drive\n\n",
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

    beamng_plots = [
        (plot_airborne, "beamng_airborne.png", "Airborne / wheel contact"),
        (plot_mean_speed, "beamng_mean_speed.png", "Mean speed"),
        (plot_centerline_distribution, "beamng_centerline_dist.png", "Centerline distribution"),
        (plot_lap_time_progression, "beamng_lap_times.png", "Lap time progression"),
    ]
    beamng_sections = []
    for plot_fn, fname, label in beamng_plots:
        if plot_fn(data, results_dir):
            beamng_sections.append(f"![{label}]({fname})\n\n")
    if beamng_sections:
        sections.append("## BeamNG Plots\n\n")
        sections.extend(beamng_sections)

    plot_reward_trajectory(data, results_dir)
    sections.append("## Additional Plots\n\n")
    sections.append("![Reward trajectory](reward_trajectory.png)\n\n")

    report_path = os.path.join(results_dir, "results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(sections).rstrip("\n") + "\n")

    n = len(os.listdir(results_dir))
    logger.info("Saved %d file(s) to %s/ (report: results.md)", n, results_dir)
