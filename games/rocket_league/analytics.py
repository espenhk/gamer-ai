"""Rocket League-specific analytics (issue #466).

Entry point called by main.py::

    save_experiment_results(data: ExperimentData, results_dir: str) -> None

Rocket League is an adversarial ball-sport env, so the game-specific plots
are match analytics, not racing plots: match-result breakdown (an episode
ends the moment a goal is scored or conceded, so ``termination_reason``
*is* the result), ball-touch counts, boost usage, and ball pursuit
(distance / velocity-towards-ball means).

The touch/boost/pursuit panels read ``GreedySimResult.obs_averages``, which
the env populates via ``episode_obs_averages`` in terminal-step info; they
are silently skipped for old experiment data recorded before that field
existed. Per-step traces and position heatmaps need per-step telemetry the
framework does not record yet (see issues #443/#444) and are deferred.
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib

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

#: termination_reason → match result. Episodes end on the first goal, so the
#: reason doubles as the result; "done"/"timeout" episodes had no goal.
_RESULT_LABELS = {
    "goal_scored": ("win", "#27ae60"),
    "goal_conceded": ("loss", "#c0392b"),
    "timeout": ("draw", "#95a5a6"),
    "done": ("draw", "#95a5a6"),
}


def _save(fig: "Figure", path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _obs_avg_sims(data: ExperimentData) -> list:
    return [s for s in data.greedy_sims if s.obs_averages]


# ---------------------------------------------------------------------------
# Match results (the headline metric)
# ---------------------------------------------------------------------------


def plot_match_results(data: ExperimentData, results_dir: str) -> bool:
    """Win / loss / draw breakdown across the greedy phase."""
    sims = data.greedy_sims
    if not sims:
        return False

    counts = {"win": 0, "loss": 0, "draw": 0}
    for s in sims:
        label, _ = _RESULT_LABELS.get(s.termination_reason or "", ("draw", "#95a5a6"))
        counts[label] += 1
    total = len(sims)

    labels = ["win", "loss", "draw"]
    values = [counts[label] for label in labels]
    colors = ["#27ae60", "#c0392b", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, values):
        ax.annotate(
            f"{v} ({100 * v / total:.0f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_title(f"{data.experiment_name} — RL: Match Results ({total} episodes)")
    ax.set_ylabel("Episodes")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "rl_match_results.png"))
    return True


# ---------------------------------------------------------------------------
# Ball touches
# ---------------------------------------------------------------------------


def plot_ball_touches(data: ExperimentData, results_dir: str) -> bool:
    """Ball-touch step count per episode over the greedy phase."""
    sims = _obs_avg_sims(data)
    touches = [s.obs_averages.get("ball_touches") for s in sims]
    if not any(v is not None for v in touches):
        return False
    xs = [s.sim for s in sims]

    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.15), 4))
    ax.plot(
        xs,
        [float(v) if v is not None else float("nan") for v in touches],
        color="#8e44ad",
        linewidth=1.2,
        marker="o",
        markersize=3,
    )
    ax.set_title(f"{data.experiment_name} — RL: Ball-Touch Steps per Episode")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Steps with a ball touch")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "rl_ball_touches.png"))
    return True


# ---------------------------------------------------------------------------
# Boost usage
# ---------------------------------------------------------------------------


def plot_boost_usage(data: ExperimentData, results_dir: str) -> bool:
    """Mean boost fuel per episode over the greedy phase."""
    sims = _obs_avg_sims(data)
    boost = [s.obs_averages.get("boost_amount") for s in sims]
    if not any(v is not None for v in boost):
        return False
    xs = [s.sim for s in sims]

    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.15), 4))
    ax.plot(
        xs,
        [float(v) if v is not None else float("nan") for v in boost],
        color="#e67e22",
        linewidth=1.2,
        marker="o",
        markersize=3,
    )
    ax.set_title(f"{data.experiment_name} — RL: Mean Boost per Episode")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Mean boost fuel [0, 1]")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "rl_boost_usage.png"))
    return True


# ---------------------------------------------------------------------------
# Ball pursuit
# ---------------------------------------------------------------------------


def plot_ball_pursuit(data: ExperimentData, results_dir: str) -> bool:
    """Mean distance-to-ball and velocity-towards-ball per episode."""
    sims = _obs_avg_sims(data)
    if not sims:
        return False
    xs = [s.sim for s in sims]
    dist = [s.obs_averages.get("dist_to_ball") for s in sims]
    vel = [s.obs_averages.get("vel_towards_ball") for s in sims]
    has_dist = any(v is not None for v in dist)
    has_vel = any(v is not None for v in vel)
    if not (has_dist or has_vel):
        return False

    fig, (ax_dist, ax_vel) = plt.subplots(1, 2, figsize=(max(10, len(xs) * 0.2), 4))
    if has_dist:
        ax_dist.plot(
            xs,
            [float(v) if v is not None else float("nan") for v in dist],
            color="#2980b9",
            linewidth=1.2,
            marker="o",
            markersize=3,
        )
    ax_dist.set_title("Mean distance to ball per episode")
    ax_dist.set_xlabel("Simulation")
    ax_dist.set_ylabel("Distance (UU)")
    if has_vel:
        ax_vel.plot(
            xs,
            [float(v) if v is not None else float("nan") for v in vel],
            color="#16a085",
            linewidth=1.2,
            marker="o",
            markersize=3,
        )
        ax_vel.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=0.8)
    ax_vel.set_title("Mean velocity towards ball per episode")
    ax_vel.set_xlabel("Simulation")
    ax_vel.set_ylabel("Velocity (UU/s, + = approaching)")
    fig.suptitle(f"{data.experiment_name} — RL: Ball Pursuit", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "rl_ball_pursuit.png"))
    return True


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def save_experiment_results(data: ExperimentData, results_dir: str) -> None:
    """Generate all plots and write a results.md report to *results_dir*."""
    os.makedirs(results_dir, exist_ok=True)

    sections = [
        f"# Experiment: {data.experiment_name}\n\n**Game:** Rocket League (via RLGym)\n\n",
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

    rl_plots = [
        (plot_match_results, "rl_match_results.png", "Match results"),
        (plot_ball_touches, "rl_ball_touches.png", "Ball touches"),
        (plot_boost_usage, "rl_boost_usage.png", "Boost usage"),
        (plot_ball_pursuit, "rl_ball_pursuit.png", "Ball pursuit"),
    ]
    rl_sections = []
    for plot_fn, fname, label in rl_plots:
        if plot_fn(data, results_dir):
            rl_sections.append(f"![{label}]({fname})\n\n")
    if rl_sections:
        sections.append("## Rocket League Plots\n\n")
        sections.extend(rl_sections)

    plot_reward_trajectory(data, results_dir)
    sections.append("## Additional Plots\n\n")
    sections.append("![Reward trajectory](reward_trajectory.png)\n\n")

    report_path = os.path.join(results_dir, "results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(sections).rstrip("\n") + "\n")

    n = len(os.listdir(results_dir))
    logger.info("Saved %d file(s) to %s/ (report: results.md)", n, results_dir)
