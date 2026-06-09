"""Atari-specific analytics.

Entry point called by main.py::

    save_experiment_results(data: ExperimentData, results_dir: str) -> None

Atari games expose 128-byte RAM observations and a small discrete action set,
so the report focuses on return, episode length, and action-choice behaviour
rather than centreline-style spatial geometry.
"""

from __future__ import annotations

import logging
import os

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

import numpy as np

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


def _save(fig: "Figure", path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_episode_returns(data: ExperimentData, results_dir: str) -> bool:
    """Plot per-episode shaped training reward with a best-so-far curve.

    Uses ``GreedySimResult.reward``, which is the shaped reward produced by
    ``AtariRewardCalculator`` (applies ``native_reward_scale``, ``clip_sign``,
    and ``step_penalty``).  It reflects training performance, not the raw
    un-scaled native game score.
    """
    if not _HAS_MPL or not data.greedy_sims:
        return False

    sims = [s.sim for s in data.greedy_sims]
    rewards = [s.reward for s in data.greedy_sims]
    best_so_far: list[float] = []
    running_best = float("-inf")
    for score in rewards:
        running_best = max(running_best, score)
        best_so_far.append(running_best)

    fig, ax = plt.subplots(figsize=(max(8, len(sims) * 0.15), 5))
    ax.plot(sims, rewards, color="#3498db", linewidth=1.2, marker="o", markersize=3, label="training reward")
    ax.step(sims, best_so_far, where="post", color="#e67e22", linewidth=2.0, label="best so far")
    improved_xs = [s.sim for s in data.greedy_sims if s.improved]
    improved_ys = [s.reward for s in data.greedy_sims if s.improved]
    if improved_xs:
        ax.scatter(improved_xs, improved_ys, color="#27ae60", s=50, marker="^", zorder=4, label="improved")
    ax.set_title(f"{data.experiment_name} — Atari Training Reward")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Episode reward (shaped)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "atari_episode_returns.png"))
    return True


def plot_episode_lengths(data: ExperimentData, results_dir: str) -> bool:
    """Plot episode length in environment steps for each greedy sim."""
    if not _HAS_MPL or not data.greedy_sims:
        return False

    sims = [s.sim for s in data.greedy_sims]
    lengths = [s.total_steps for s in data.greedy_sims]
    fig, ax = plt.subplots(figsize=(max(8, len(sims) * 0.15), 4))
    ax.plot(sims, lengths, color="#8e44ad", linewidth=1.2, marker="o", markersize=3)
    improved_xs = [s.sim for s in data.greedy_sims if s.improved]
    improved_ys = [s.total_steps for s in data.greedy_sims if s.improved]
    if improved_xs:
        ax.scatter(improved_xs, improved_ys, color="#27ae60", s=50, marker="^", zorder=4, label="improved")
        ax.legend(fontsize=9)
    ax.set_title(f"{data.experiment_name} — Atari Episode Length")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Steps")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "atari_episode_lengths.png"))
    return True


def _normalized_action_counts(action_counts: dict | None) -> dict[int, int]:
    normalized: dict[int, int] = {}
    for action_id, count in (action_counts or {}).items():
        try:
            normalized[int(action_id)] = normalized.get(int(action_id), 0) + int(count)
        except (TypeError, ValueError):
            continue
    return normalized


def plot_action_histogram(data: ExperimentData, results_dir: str) -> bool:
    """Plot aggregate action choices over the greedy phase."""
    if not _HAS_MPL:
        return False
    sim_counts = [_normalized_action_counts(s.action_counts) for s in data.greedy_sims]
    sim_counts = [counts for counts in sim_counts if counts]
    if not sim_counts:
        return False

    action_ids = sorted({action_id for counts in sim_counts for action_id in counts})
    totals = [sum(counts.get(action_id, 0) for counts in sim_counts) for action_id in action_ids]
    if not any(totals):
        return False

    fig, ax = plt.subplots(figsize=(max(8, len(action_ids) * 0.55), 4))
    colors = cm.tab20(np.linspace(0, 1, max(len(action_ids), 1)))
    ax.bar([str(action_id) for action_id in action_ids], totals, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_title(f"{data.experiment_name} — Atari Action Histogram")
    ax.set_xlabel("Action index")
    ax.set_ylabel("Steps")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "atari_action_histogram.png"))
    return True


def _atari_summary_md(data: ExperimentData) -> str:
    sims = data.greedy_sims
    if not sims:
        return ""

    rewards = [s.reward for s in sims]
    lengths = [s.total_steps for s in sims]
    lines = [
        "## Atari Metrics\n\n",
        "| Metric | Value |\n",
        "|--------|-------|\n",
        f"| Best training reward | {max(rewards):+.1f} |\n",
        f"| Mean training reward | {float(np.mean(rewards)):+.1f} |\n",
        f"| Mean episode length | {float(np.mean(lengths)):.1f} steps |\n",
    ]

    aggregate_counts: dict[int, int] = {}
    for sim in sims:
        for action_id, count in _normalized_action_counts(sim.action_counts).items():
            aggregate_counts[action_id] = aggregate_counts.get(action_id, 0) + count
    total = sum(aggregate_counts.values())
    if total > 0:
        top_action, top_count = max(aggregate_counts.items(), key=lambda item: item[1])
        lines.append(f"| Most-used action | {top_action} ({top_count / total:.1%} of steps) |\n")
    return "".join(lines) + "\n"


def save_experiment_results(data: ExperimentData, results_dir: str) -> None:
    """Generate plots and write a results.md report to *results_dir*."""
    os.makedirs(results_dir, exist_ok=True)

    sections = [
        f"# Experiment: {data.experiment_name}\n\n**Game:** Atari\n\n",
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
    wrote_episode_returns = plot_episode_returns(data, results_dir)
    wrote_episode_lengths = plot_episode_lengths(data, results_dir)
    wrote_action_histogram = plot_action_histogram(data, results_dir)
    sections.append(_atari_summary_md(data))
    sections.append("## Additional Plots\n\n")
    if wrote_episode_returns:
        sections.append("![Atari episode return](atari_episode_returns.png)\n\n")
    if wrote_episode_lengths:
        sections.append("![Atari episode length](atari_episode_lengths.png)\n\n")
    if wrote_action_histogram:
        sections.append("![Atari action histogram](atari_action_histogram.png)\n\n")
    sections.append("![Reward trajectory](reward_trajectory.png)\n\n")

    report_path = os.path.join(results_dir, "results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(sections).rstrip("\n") + "\n")

    n = len(os.listdir(results_dir))
    logger.info("Saved %d file(s) to %s/ (report: results.md)", n, results_dir)
