"""CarRacing-specific analytics.

Reuses the game-agnostic action-distribution / best-run / termination-reason
plots from games/torcs/analytics.py (they only touch generic ExperimentData /
RunTrace fields) and adds two CarRacing-specific plots: action-value
histograms (gas / brake / steering) and a speed trace of the best run.

Entry point called by main.py:
    save_experiment_results(data: ExperimentData, results_dir: str) -> None
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib

if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from framework.analytics import (
    ExperimentData,
    GreedySimResult,
    _cold_start_table_md,
    _greedy_table_md,
    _probe_table_md,
    _reward_moving_average_md,
    _summary_md,
    _timings_md,
    plot_cold_start_rewards,
    plot_greedy_rewards,
    plot_probe_rewards,
    plot_reward_moving_average,
    plot_reward_trajectory,
)
from games.torcs.analytics import (
    plot_cold_start_action_dist,
    plot_cold_start_best_run,
    plot_greedy_action_dist,
    plot_greedy_best_run,
    plot_termination_reasons,
)

logger = logging.getLogger(__name__)

# CarRacing-v2's published "solved" benchmark: average reward >= 900 over
# 100 consecutive episodes (see games/car_racing/config/gs_sac.yaml, issue #482).
SOLVED_REWARD_THRESHOLD = 900.0
SOLVED_WINDOW_EPISODES = 100


def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _best_sim(data: ExperimentData) -> GreedySimResult | None:
    if not data.greedy_sims:
        return None
    return max(data.greedy_sims, key=lambda s: s.reward)


def plot_action_histograms(data: ExperimentData, results_dir: str) -> None:
    """Gas / brake / steering value histograms over the best greedy run."""
    best = _best_sim(data)
    if best is None or not best.trace or not best.trace.throttle_state:
        return

    gas = [t[0] for t in best.trace.throttle_state]
    brake = [t[1] for t in best.trace.throttle_state]
    steer = [t[2] for t in best.trace.throttle_state if len(t) > 2]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(gas, bins=20, range=(0.0, 1.0), color="#27ae60", edgecolor="white")
    axes[0].set_title("Gas")
    axes[1].hist(brake, bins=20, range=(0.0, 1.0), color="#c0392b", edgecolor="white")
    axes[1].set_title("Brake")
    if steer:
        axes[2].hist(steer, bins=20, range=(-1.0, 1.0), color="#2980b9", edgecolor="white")
    axes[2].set_title("Steering")

    for ax in axes:
        ax.set_xlabel("Value")
        ax.set_ylabel("Steps")
    fig.suptitle(f"{data.experiment_name} — Best Run Action Distribution (sim {best.sim}, reward {best.reward:+.1f})")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "action_histograms.png"))


def plot_speed_trace(data: ExperimentData, results_dir: str) -> None:
    """Speed trace (m/s) of the best greedy run."""
    best = _best_sim(data)
    if best is None or not best.trace or not best.trace.speed_trace:
        return

    speeds = best.trace.speed_trace
    xs = range(len(speeds))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, speeds, color="#8e44ad", linewidth=1.0, alpha=0.9)
    ax.set_title(f"{data.experiment_name} — Best Run Speed Trace (sim {best.sim}, reward {best.reward:+.1f})")
    ax.set_xlabel("Sampled step")
    ax.set_ylabel("Speed (m/s)")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "speed_trace.png"))


def save_experiment_results(data: ExperimentData, results_dir: str) -> None:
    """Generate all plots and write a results.md report to *results_dir*."""
    os.makedirs(results_dir, exist_ok=True)

    sections = [
        f"# Experiment: {data.experiment_name}\n\n**Game:** CarRacing-v2\n\n",
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
        sections.append("![Cold-start best run](cold_start_best_run.png)\n\n")

    if data.greedy_sims:
        plot_greedy_rewards(data, results_dir)
        plot_greedy_action_dist(data, results_dir)
        plot_greedy_best_run(data, results_dir)
        plot_termination_reasons(data, results_dir)
        plot_action_histograms(data, results_dir)
        plot_speed_trace(data, results_dir)
        sections.append(_greedy_table_md(data))
        sections.append("\n![Greedy rewards](greedy_rewards.png)\n\n")
        sections.append("![Greedy action distribution](greedy_action_dist.png)\n\n")
        sections.append("![Greedy best run](greedy_best_run.png)\n\n")
        sections.append("![Termination reasons](termination_reasons.png)\n\n")
        if os.path.exists(os.path.join(results_dir, "action_histograms.png")):
            sections.append("![Action histograms](action_histograms.png)\n\n")
        if os.path.exists(os.path.join(results_dir, "speed_trace.png")):
            sections.append("![Speed trace](speed_trace.png)\n\n")

        plot_reward_moving_average(
            data,
            results_dir,
            window=SOLVED_WINDOW_EPISODES,
            solved_threshold=SOLVED_REWARD_THRESHOLD,
        )
        sections.append(
            _reward_moving_average_md(
                data,
                window=SOLVED_WINDOW_EPISODES,
                solved_threshold=SOLVED_REWARD_THRESHOLD,
            )
        )
        sections.append("![Reward moving average](reward_moving_average.png)\n\n")

    plot_reward_trajectory(data, results_dir)
    sections.append("## Additional Plots\n\n")
    sections.append("![Reward trajectory](reward_trajectory.png)\n\n")

    report_path = os.path.join(results_dir, "results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(sections).rstrip("\n") + "\n")

    n = len(os.listdir(results_dir))
    logger.info("Saved %d file(s) to %s/ (report: results.md)", n, results_dir)
