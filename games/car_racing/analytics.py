"""CarRacing-specific analytics.

Entry point called by main.py:
    save_experiment_results(data: ExperimentData, results_dir: str) -> None
"""

from __future__ import annotations

import logging
import os

from framework.analytics import (
    ExperimentData,
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

logger = logging.getLogger(__name__)

# CarRacing-v2's published "solved" benchmark: average reward >= 900 over
# 100 consecutive episodes (see games/car_racing/config/gs_sac.yaml, issue #482).
SOLVED_REWARD_THRESHOLD = 900.0
SOLVED_WINDOW_EPISODES = 100


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
        sections.append(_cold_start_table_md(data))
        sections.append("\n![Cold-start best rewards](cold_start_best_rewards.png)\n\n")

    if data.greedy_sims:
        plot_greedy_rewards(data, results_dir)
        sections.append(_greedy_table_md(data))
        sections.append("\n![Greedy rewards](greedy_rewards.png)\n\n")

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
