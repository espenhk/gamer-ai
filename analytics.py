"""Backward-compatibility shim.

Data classes and generic plots live in framework.analytics.
TMNF-specific plots live in games.tmnf.analytics.
This module re-exports everything and provides the combined
save_experiment_results() / save_grid_summary() entry points.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-export data classes and generic plots
# ---------------------------------------------------------------------------

from framework.analytics import (  # noqa: F401
    RunTrace,
    ProbeResult,
    ColdStartSimResult,
    ColdStartRestartResult,
    GreedySimResult,
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

# ---------------------------------------------------------------------------
# Re-export TMNF-specific plots
# ---------------------------------------------------------------------------

from games.tmnf.analytics import (  # noqa: F401
    plot_probe_paths,
    plot_cold_start_action_dist,
    plot_cold_start_best_run,
    plot_greedy_best_run,
    plot_greedy_action_dist,
    plot_greedy_progress,
    plot_weight_heatmap,
    plot_weight_evolution,
    plot_gs_comparison_paths,
    plot_gs_comparison_progress,
)


# ---------------------------------------------------------------------------
# Combined entry point (generic report + TMNF-specific plots)
# ---------------------------------------------------------------------------

def save_experiment_results(data: ExperimentData, results_dir: str) -> None:
    """Generate all plots and write a single results.md report to *results_dir*."""
    os.makedirs(results_dir, exist_ok=True)

    track_line = f"\n**Track:** {data.track}\n" if data.track else ""
    sections = [
        f"# Experiment: {data.experiment_name}\n{track_line}\n",
        _timings_md(data),
        _summary_md(data),
    ]

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


def save_grid_summary(
    runs: list[tuple[str, ExperimentData]],
    varied_keys: list[str],
    summary_dir: str,
    base_name: str,
) -> None:
    """Grid search cross-experiment summary with TMNF path/progress comparison plots."""

    def _tmnf_extra(r, d):
        plot_gs_comparison_paths(r, d)
        plot_gs_comparison_progress(r, d)

    _framework_save_grid_summary(
        runs, varied_keys, summary_dir, base_name,
        extra_plots_fn=_tmnf_extra,
    )
