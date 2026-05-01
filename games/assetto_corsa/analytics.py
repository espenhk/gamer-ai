"""Assetto Corsa analytics: thin wrapper around framework analytics.

The AC integration doesn't yet ship game-specific plots — the framework
plots (probe rewards, cold-start, greedy, reward trajectory) are sufficient
for early experiments. Game-specific plots (track-relative trajectory,
slip heatmap, …) can be added here later without touching the framework.
"""

from __future__ import annotations

import logging
import os

from framework.analytics import (
    ExperimentData,
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

logger = logging.getLogger(__name__)


def save_experiment_results(data: ExperimentData, results_dir: str) -> None:
    """Generate framework plots and a results.md report into *results_dir*."""
    os.makedirs(results_dir, exist_ok=True)

    track_line = f"\n**Track:** {data.track}\n" if data.track else ""
    sections = [
        f"# Experiment: {data.experiment_name}\n{track_line}\n",
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

    plot_reward_trajectory(data, results_dir)
    sections.append("## Additional Plots\n\n")
    sections.append("![Reward trajectory](reward_trajectory.png)\n\n")

    report_path = os.path.join(results_dir, "results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(sections)

    n = len(os.listdir(results_dir))
    logger.info("Saved %d file(s) to %s/ (report: results.md)", n, results_dir)


def save_grid_summary(*args, **kwargs) -> None:
    """AC grid-summary wrapper: framework defaults, no extra plots."""
    _framework_save_grid_summary(*args, **kwargs)
