"""SC2-specific analytics entry point.

SC2 minigames don't have throttle / steering / centerline concepts, so
this module is intentionally thin — it delegates to the framework's
generic reward-trajectory and probe/cold-start/greedy plots and writes
a minimal Markdown summary.

Entry point called by main.py:
    save_experiment_results(data: ExperimentData, results_dir: str) -> None
"""
from __future__ import annotations

import logging
import os
import sys

import matplotlib
if 'matplotlib.pyplot' not in sys.modules:
    matplotlib.use('Agg')

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
    """Generate generic plots and write a results.md report to *results_dir*."""
    os.makedirs(results_dir, exist_ok=True)

    sections = [
        f"# Experiment: {data.experiment_name}\n\n**Game:** StarCraft 2\n\n",
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
    sections.append("\n![Reward trajectory](reward_trajectory.png)\n\n")

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
    """Grid search cross-experiment summary using framework defaults."""
    _framework_save_grid_summary(runs, varied_keys, summary_dir, base_name)
