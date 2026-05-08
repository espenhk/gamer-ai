"""SC2-specific analytics entry point.

SC2 minigames don't have throttle / steering / centerline concepts, so this
module skips those racing-specific plots entirely and instead produces:

- Generic reward-trajectory / probe / cold-start / greedy plots (framework)
- Reward-component breakdown (2b — framework helper, SC2 env populates components)
- Action-frequency breakdown (2a — bar chart, aggregate bar, entropy)
- Economy/state feature averages (2c — per-sim line chart)
- Spatial target heatmap (2d — 8×8 imshow, log-scaled)
- Episode outcome breakdown (2e — win/loss/finish/timeout stacked bars)
- Skipped-frame trend (2f — per-sim bars, with improved-run markers)

Flags consumed by any caller that needs to know which plot families this
game supports:
    SUPPORTS_THROTTLE = False   — no accel/brake distribution plots
    SUPPORTS_PATH     = False   — no pos_x / pos_z bird's-eye path traces

Entry point called by main.py / grid_search.py:
    save_experiment_results(data: ExperimentData, results_dir: str) -> None
"""
from __future__ import annotations

import dataclasses
import logging
import os

try:
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.figure import Figure
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

import numpy as np
import yaml

from framework.analytics import (
    ExperimentData,
    plot_probe_rewards,
    plot_cold_start_rewards,
    plot_greedy_rewards,
    plot_reward_components,
    plot_reward_trajectory,
    _probe_table_md,
    _cold_start_table_md,
    _greedy_table_md,
    _timings_md,
    _summary_md,
    save_grid_summary as _framework_save_grid_summary,
)
from games.sc2.actions import FUNCTION_IDS
from games.sc2.reward import SC2RewardConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flags (Part 1)
# ---------------------------------------------------------------------------
SUPPORTS_THROTTLE: bool = False   # no accel/brake distribution plots for SC2
SUPPORTS_PATH:     bool = False   # no pos_x / pos_z path-trace plots for SC2

_REWARD_COMPONENT_TO_CFG_KEY: dict[str, str] = {
    "score": "score_weight",
    "economy": "economy_weight",
    "idle_penalty": "idle_penalty",
    "idle_bonus": "idle_bonus",
    "move_exploration": "move_exploration_bonus",
    "move_repeat_penalty": "move_repeat_penalty",
    "move_self_penalty": "move_self_penalty",
    "attack_move_bonus": "attack_move_bonus",
    "click_attack_bonus": "click_attack_bonus",
    "attack_friendly_penalty": "attack_friendly_penalty",
    "step_penalty": "step_penalty",
}
_DEFAULT_REWARD_CFG: dict[str, float | int] = dataclasses.asdict(SC2RewardConfig())
# Components that are inherently scale-free and pass through unscaled.
_NO_SCALE_COMPONENT_KEYS: set[str] = {"scout"}
_WARNED_NON_NUMERIC_WEIGHT: bool = False

# ---------------------------------------------------------------------------
# Action-label helpers
# ---------------------------------------------------------------------------

def _action_label(fn_idx: int) -> str:
    """Short human-readable label for a FUNCTION_IDS fn_idx integer."""
    return FUNCTION_IDS.get(fn_idx, f"fn{fn_idx}")


# ---------------------------------------------------------------------------
# 2a — Action-frequency breakdown
# ---------------------------------------------------------------------------

def plot_action_frequency(data: ExperimentData, results_dir: str) -> None:
    """Bar chart of per-sim action-type counts + aggregate bar + entropy line.

    Three panels written to ``action_frequency.png``:
    - Top: stacked vertical bars, one bar per greedy sim, coloured by fn_idx.
    - Middle: aggregate total-count bar chart across all sims.
    - Bottom: action-entropy per sim H = -Σ pᵢ log₂ pᵢ over unique fn_idx values.
    """
    if not _HAS_MPL:
        return
    sims = [s for s in data.greedy_sims if s.action_counts]
    if not sims:
        return

    # Collect all fn_idx keys that appear in any sim.
    all_fn: list[int] = []
    seen: set[int] = set()
    for s in sims:
        for k in s.action_counts:
            if k not in seen:
                all_fn.append(k)
                seen.add(k)
    all_fn.sort()

    labels = [_action_label(k) for k in all_fn]
    xs = [s.sim for s in sims]
    n_fns = len(all_fn)
    cmap_colors = [cm.tab10(i / max(n_fns - 1, 1)) for i in range(n_fns)]

    # Build per-sim count matrix (n_sims × n_fns).
    counts_mat = np.array([
        [s.action_counts.get(k, 0) for k in all_fn]
        for s in sims
    ], dtype=float)

    # Normalise rows to fractions.
    row_totals = counts_mat.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1
    fracs_mat = counts_mat / row_totals

    # Entropy per sim. Compute log2 only where p > 0 so zero-probability
    # terms contribute exactly 0 without biasing nonzero probabilities.
    log_fracs = np.zeros_like(fracs_mat)
    positive = fracs_mat > 0
    log_fracs[positive] = np.log2(fracs_mat[positive])
    entropy = -np.sum(fracs_mat * log_fracs, axis=1)

    # Aggregate totals.
    agg_counts = counts_mat.sum(axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(xs) * 0.25), 12),
                              gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

    # --- Panel 1: stacked bar per sim ---
    ax1 = axes[0]
    left = np.zeros(len(sims))
    for i, (fn, label, color) in enumerate(zip(all_fn, labels, cmap_colors)):
        vals = fracs_mat[:, i]
        ax1.bar(xs, vals, bottom=left, color=color, label=label,
                width=0.8, edgecolor="none")
        left += vals
    ax1.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    ax1.set_ylim(0, 1)
    ax1.set_title(f"{data.experiment_name} — Action Distribution per Sim (fraction)")
    ax1.set_xlabel("Simulation")
    ax1.set_ylabel("Fraction")
    ax1.legend(fontsize=8, loc="upper right", ncol=max(1, n_fns // 3))

    # --- Panel 2: aggregate counts ---
    ax2 = axes[1]
    bar_colors = [cm.tab10(i / max(n_fns - 1, 1)) for i in range(n_fns)]
    ax2.bar(labels, agg_counts, color=bar_colors, edgecolor="white")
    ax2.set_title("Aggregate Action Counts (all sims)")
    ax2.set_ylabel("Total steps")
    ax2.tick_params(axis="x", rotation=20)
    for j, v in enumerate(agg_counts):
        ax2.text(j, v + max(agg_counts) * 0.01, f"{int(v)}", ha="center",
                 va="bottom", fontsize=8)

    # --- Panel 3: entropy over training ---
    ax3 = axes[2]
    ax3.plot(xs, entropy, color="#3498db", linewidth=1.6, marker="o",
             markersize=3, label="action entropy (bits)")
    ax3.set_title("Action Entropy per Sim")
    ax3.set_xlabel("Simulation")
    ax3.set_ylabel("H (bits)")
    ax3.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "action_frequency.png"))


# ---------------------------------------------------------------------------
# 2c — Economy / game-state feature averages
# ---------------------------------------------------------------------------

_OBS_FEATURE_LABELS: dict[str, str] = {
    "army_count":         "Army count",
    "food_used":          "Food used",
    "food_cap":           "Food cap",
    "minerals":           "Minerals",
    "vespene":            "Vespene",
    "screen_self_count":  "Screen self (px)",
    "screen_enemy_count": "Screen enemy (px)",
}


def plot_obs_averages(data: ExperimentData, results_dir: str) -> None:
    """Multi-panel line chart of per-sim episode-average game-state features.

    One panel per feature that has at least one non-zero sim value.  Written
    to ``obs_averages.png``.
    """
    if not _HAS_MPL:
        return
    sims = [s for s in data.greedy_sims if s.obs_averages]
    if not sims:
        return

    # Determine which features to plot.
    all_feats = list(_OBS_FEATURE_LABELS.keys())
    active = [f for f in all_feats
              if any(s.obs_averages.get(f, 0.0) != 0.0 for s in sims)]
    if not active:
        return

    xs = [s.sim for s in sims]
    n = len(active)
    fig, axes = plt.subplots(n, 1, figsize=(max(10, len(xs) * 0.2), 3 * n),
                              sharex=True)
    if n == 1:
        axes = [axes]

    colors = cm.tab10(np.linspace(0, 1, n))
    for ax, feat, color in zip(axes, active, colors):
        ys = [s.obs_averages.get(feat, 0.0) for s in sims]
        ax.plot(xs, ys, color=color, linewidth=1.4)
        ax.set_ylabel(_OBS_FEATURE_LABELS.get(feat, feat), fontsize=9)
        improvement_xs = [s.sim for s in sims if s.improved]
        improvement_ys = [s.obs_averages.get(feat, 0.0)
                          for s in sims if s.improved]
        if improvement_xs:
            ax.scatter(improvement_xs, improvement_ys, color="#27ae60",
                       s=40, zorder=4, marker="^")

    axes[-1].set_xlabel("Simulation")
    fig.suptitle(f"{data.experiment_name} — Game-State Feature Averages", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "obs_averages.png"))


# ---------------------------------------------------------------------------
# 2d — Spatial target heatmap
# ---------------------------------------------------------------------------

def plot_spatial_heatmap(data: ExperimentData, results_dir: str) -> None:
    """Aggregate 8×8 heatmap of (x, y) action targets across all greedy sims.

    Written to ``spatial_heatmap.png``.  Log-scaled so infrequent cells are
    still visible.  Shows whether the policy is stuck in one screen region.
    """
    if not _HAS_MPL:
        return
    sims_with_hist = [s for s in data.greedy_sims if s.xy_hist is not None]
    if not sims_with_hist:
        return

    agg = np.zeros((8, 8), dtype=float)
    for s in sims_with_hist:
        agg += np.array(s.xy_hist, dtype=float)

    if agg.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    # Log-scale: log1p so zero cells show as 0.
    img = ax.imshow(np.log1p(agg), cmap="YlOrRd", origin="upper",
                    extent=[0, 1, 1, 0], aspect="equal")
    fig.colorbar(img, ax=ax, label="log1p(steps)")
    ax.set_title(f"{data.experiment_name} — Action Target Heatmap (all greedy sims)")
    ax.set_xlabel("Screen X (normalised)")
    ax.set_ylabel("Screen Y (normalised)")
    # Draw grid lines at cell boundaries.
    for v in np.linspace(0, 1, 9):
        ax.axhline(v, color="white", linewidth=0.4, alpha=0.5)
        ax.axvline(v, color="white", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "spatial_heatmap.png"))


# ---------------------------------------------------------------------------
# 2e — Episode outcome breakdown
# ---------------------------------------------------------------------------

_OUTCOME_COLORS: dict[str, str] = {
    "win":     "#27ae60",
    "finish":  "#2ecc71",
    "timeout": "#f39c12",
    "loss":    "#c0392b",
    "other":   "#95a5a6",
}


def plot_outcome_breakdown(data: ExperimentData, results_dir: str) -> None:
    """Stacked-bar chart of win / finish / timeout / loss per greedy sim.

    Only rendered when at least one sim has a non-None termination_reason.
    Written to ``outcome_breakdown.png``.
    """
    if not _HAS_MPL:
        return
    sims = data.greedy_sims
    if not sims:
        return
    reasons = [s.termination_reason or "other" for s in sims]
    if all(r == "other" for r in reasons):
        return

    xs = [s.sim for s in sims]
    categories = ["win", "finish", "timeout", "loss", "other"]
    # Build per-sim one-hot for each category.
    data_map: dict[str, list[int]] = {c: [] for c in categories}
    for r in reasons:
        r_key = r if r in categories else "other"
        for c in categories:
            data_map[c].append(1 if c == r_key else 0)

    # Drop categories that never appear.
    active_cats = [c for c in categories if any(data_map[c])]

    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.15), 4))
    bottom = np.zeros(len(xs))
    for cat in active_cats:
        vals = np.array(data_map[cat], dtype=float)
        ax.bar(xs, vals, bottom=bottom, color=_OUTCOME_COLORS[cat],
               label=cat, width=1.0, edgecolor="none")
        bottom += vals

    ax.set_ylim(0, 1.05)
    ax.set_title(f"{data.experiment_name} — Episode Outcome per Greedy Sim")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Outcome")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "outcome_breakdown.png"))


# ---------------------------------------------------------------------------
# 2f — Skipped frames per sim
# ---------------------------------------------------------------------------

def plot_skipped_frames(data: ExperimentData, results_dir: str) -> None:
    """Bar chart of skipped frames per greedy sim."""
    if not _HAS_MPL:
        return
    sims = [s for s in data.greedy_sims if s.skipped_frames is not None]
    if not sims:
        return

    xs = [s.sim for s in sims]
    ys = [max(0, int(s.skipped_frames or 0)) for s in sims]
    colors = ["#e74c3c" if v > 0 else "#27ae60" for v in ys]

    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.15), 4))
    ax.bar(xs, ys, color=colors, edgecolor="none", width=1.0)

    improved_xs = [s.sim for s in sims if s.improved]
    improved_ys = [max(0, int(s.skipped_frames or 0)) for s in sims if s.improved]
    if improved_xs:
        ax.scatter(improved_xs, improved_ys, color="#2c3e50", s=40,
                   zorder=4, marker="^", label="improved")
        ax.legend(fontsize=9)

    ax.set_ylim(0, max(1.0, float(max(ys) * 1.1)))
    ax.set_title(f"{data.experiment_name} — Skipped Frames per Greedy Sim")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Skipped frames")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "skipped_frames.png"))


# ---------------------------------------------------------------------------
# 2g — Supply-capped fraction per sim
# ---------------------------------------------------------------------------

def plot_supply_capped(data: ExperimentData, results_dir: str) -> None:
    """Bar chart showing the fraction of each greedy sim spent supply-capped.

    Supply-capped means ``food_used >= food_cap`` for that step.  Written to
    ``supply_capped.png``.  Only rendered when at least one sim has a non-None
    ``supply_capped_fraction`` value.
    """
    if not _HAS_MPL:
        return
    sims = [s for s in data.greedy_sims if s.supply_capped_fraction is not None]
    if not sims:
        return

    xs     = [s.sim for s in sims]
    ys     = [s.supply_capped_fraction for s in sims]
    colors = ["#e74c3c" if v > 0.5 else "#f39c12" if v > 0.25 else "#27ae60"
              for v in ys]

    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.15), 4))
    ax.bar(xs, ys, color=colors, edgecolor="none", width=1.0)

    # Mark sims where the policy improved.
    improved_xs = [s.sim for s in sims if s.improved]
    improved_ys = [s.supply_capped_fraction for s in sims if s.improved]
    if improved_xs:
        ax.scatter(improved_xs, improved_ys, color="#2c3e50", s=40,
                   zorder=4, marker="^", label="improved")
        ax.legend(fontsize=9)

    ax.set_ylim(0, 1.0)
    ax.set_title(f"{data.experiment_name} — Time Supply-Capped per Greedy Sim")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Fraction of steps supply-capped")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{100*v:.0f}%")
    )
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "supply_capped.png"))


# ---------------------------------------------------------------------------
# 2h — Resources available over time (best run)
# ---------------------------------------------------------------------------

def _best_sim(data: ExperimentData) -> "GreedySimResult | None":
    """Return the last improved greedy sim, or the last sim if none improved."""
    if not data.greedy_sims:
        return None
    improved = [s for s in data.greedy_sims if s.improved]
    return improved[-1] if improved else data.greedy_sims[-1]


def plot_resource_series(data: ExperimentData, results_dir: str) -> None:
    """Line chart of minerals + vespene over game time for the best greedy sim.

    Written to ``resource_series.png``.  Only rendered when the best sim has
    a non-empty ``resource_series`` list.
    """
    if not _HAS_MPL:
        return
    best = _best_sim(data)
    if best is None or not best.resource_series:
        return

    series = best.resource_series
    xs = [pt[0] for pt in series]
    ys = [pt[1] for pt in series]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(xs, ys, alpha=0.25, color="#3498db")
    ax.plot(xs, ys, color="#3498db", linewidth=1.4)

    # Average line.
    avg = sum(ys) / len(ys)
    ax.axhline(avg, color="#e74c3c", linestyle="--", linewidth=1.0,
               label=f"avg {avg:.0f}")

    ax.set_title(
        f"{data.experiment_name} — Resources Available Over Time"
        f" (sim {best.sim}, reward {best.reward:+.1f})"
    )
    ax.set_xlabel("Game time (s)")
    ax.set_ylabel("Minerals + Vespene")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "resource_series.png"))


# ---------------------------------------------------------------------------
# 2i — Army count over time (best run)
# ---------------------------------------------------------------------------

def plot_army_count(data: ExperimentData, results_dir: str) -> None:
    """Line chart of army count over game time for the best greedy sim.

    Written to ``army_count.png``.  Only rendered when the best sim has a
    non-empty ``army_count_series`` list.
    """
    if not _HAS_MPL:
        return
    best = _best_sim(data)
    if best is None or not best.army_count_series:
        return

    series = best.army_count_series
    xs = [pt[0] for pt in series]
    ys = [pt[1] for pt in series]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(xs, ys, alpha=0.2, color="#27ae60")
    ax.plot(xs, ys, color="#27ae60", linewidth=1.4)

    ax.set_title(
        f"{data.experiment_name} — Army Count Over Time"
        f" (sim {best.sim}, reward {best.reward:+.1f})"
    )
    ax.set_xlabel("Game time (s)")
    ax.set_ylabel("Army count (units)")
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "army_count.png"))


# ---------------------------------------------------------------------------
# 2j — Build order (best run)
# ---------------------------------------------------------------------------

def _game_time_to_mmss(game_time_s: float) -> str:
    """Convert a float game-time in seconds to ``mm:ss`` string."""
    m = int(game_time_s) // 60
    s = int(game_time_s) % 60
    return f"{m:02d}:{s:02d}"


def plot_build_order(data: ExperimentData, results_dir: str) -> None:
    """Horizontal timeline of unit-build events for the best greedy sim.

    Each dot on the timeline marks when a new unit of that type was first
    detected (unit count increased).  Written to ``build_order.png``.  Only
    rendered when the best sim has a non-empty ``build_order`` list.
    """
    if not _HAS_MPL:
        return
    best = _best_sim(data)
    if best is None or not best.build_order:
        return

    events = best.build_order   # [[game_time_s, unit_name], ...]

    # Collect unique unit names in order of first appearance.
    seen: dict[str, int] = {}
    for _, uname in events:
        if uname not in seen:
            seen[uname] = len(seen)
    unit_names = list(seen.keys())
    n_units = len(unit_names)

    fig, ax = plt.subplots(figsize=(12, max(3, n_units * 0.55 + 1.5)))
    colors = [cm.tab10(i / max(n_units - 1, 1)) for i in range(n_units)]
    color_map = dict(zip(unit_names, colors))

    for t, uname in events:
        y = seen[uname]
        ax.scatter(t, y, color=color_map[uname], s=60, zorder=3)

    ax.set_yticks(range(n_units))
    ax.set_yticklabels(unit_names)
    ax.set_xlabel("Game time (s)")

    # Secondary x-axis with mm:ss labels.
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_positions = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 6)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([_game_time_to_mmss(v) for v in tick_positions], fontsize=8)

    ax.set_title(
        f"{data.experiment_name} — Build Order"
        f" (sim {best.sim}, reward {best.reward:+.1f})"
    )
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, "build_order.png"))


# ---------------------------------------------------------------------------
# Grid-search SC2-specific summary plots
# ---------------------------------------------------------------------------

def _entropy_from_counts(action_counts: dict[int, int] | None) -> float | None:
    """Return action entropy in bits for one sim's action-count mapping."""
    if not action_counts:
        return None
    counts = np.array([max(0.0, float(v)) for v in action_counts.values()], dtype=float)
    total = float(counts.sum())
    if total <= 0.0:
        return None
    probs = counts / total
    positive = probs > 0
    return float(-np.sum(probs[positive] * np.log2(probs[positive])))


def plot_gs_action_entropy(
    runs: list[tuple[str, ExperimentData]],
    summary_dir: str,
) -> None:
    """Cross-run comparison of mean per-sim action entropy (bits)."""
    if not _HAS_MPL:
        return
    rows: list[tuple[str, float]] = []
    for name, data in runs:
        entropies = [
            e
            for e in (_entropy_from_counts(s.action_counts) for s in data.greedy_sims)
            if e is not None
        ]
        if entropies:
            rows.append((name, float(np.mean(entropies))))
    if not rows:
        return

    rows.sort(key=lambda x: x[1])
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    n = len(rows)
    colors = cm.RdYlGn(np.linspace(0.15, 0.85, n))

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.45)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(v, bar.get_y() + bar.get_height() / 2, f"  {v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Mean action entropy (bits)")
    ax.set_title("SC2 Grid Search — Mean Action Entropy per Experiment")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    _save(fig, os.path.join(summary_dir, "comparison_action_entropy.png"))


def plot_gs_outcome_breakdown(
    runs: list[tuple[str, ExperimentData]],
    summary_dir: str,
) -> None:
    """Cross-run stacked horizontal bars of greedy-sim outcome fractions."""
    if not _HAS_MPL:
        return
    categories = ["win", "finish", "timeout", "loss", "other"]
    rows: list[tuple[str, dict[str, float]]] = []
    for name, data in runs:
        sims = data.greedy_sims
        if not sims:
            continue
        counts = {c: 0 for c in categories}
        for s in sims:
            r = s.termination_reason or "other"
            counts[r if r in counts else "other"] += 1
        total = float(sum(counts.values()))
        if total <= 0:
            continue
        rows.append((name, {k: (v / total) for k, v in counts.items()}))
    if not rows:
        return

    rows.sort(key=lambda x: (-(x[1]["win"] + x[1]["finish"]), x[1]["loss"]))
    names = [r[0] for r in rows]
    n = len(rows)
    y = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.5)))
    left = np.zeros(n, dtype=float)
    for cat in categories:
        vals = np.array([r[1][cat] for r in rows], dtype=float)
        if np.all(vals == 0):
            continue
        ax.barh(
            y, vals, left=left, label=cat,
            color=_OUTCOME_COLORS[cat], edgecolor="white", linewidth=0.5
        )
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Greedy-sim outcome fraction")
    ax.set_title("SC2 Grid Search — Outcome Breakdown per Experiment")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, os.path.join(summary_dir, "comparison_outcomes.png"))


def plot_gs_supply_capped(
    runs: list[tuple[str, ExperimentData]],
    summary_dir: str,
) -> None:
    """Cross-run comparison of average supply-capped fraction."""
    if not _HAS_MPL:
        return
    rows: list[tuple[str, float]] = []
    for name, data in runs:
        values = [
            float(s.supply_capped_fraction)
            for s in data.greedy_sims
            if s.supply_capped_fraction is not None
        ]
        if values:
            rows.append((name, float(np.mean(values))))
    if not rows:
        return

    rows.sort(key=lambda x: x[1], reverse=True)
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = ["#e74c3c" if v > 0.5 else "#f39c12" if v > 0.25 else "#27ae60" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.45)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(v, bar.get_y() + bar.get_height() / 2, f"  {100*v:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Mean supply-capped fraction")
    ax.set_title("SC2 Grid Search — Supply-Capped Time per Experiment")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    _save(fig, os.path.join(summary_dir, "comparison_supply_capped.png"))


def plot_gs_skipped_frames(
    runs: list[tuple[str, ExperimentData]],
    summary_dir: str,
) -> None:
    """Cross-run comparison of average skipped frames per greedy sim."""
    if not _HAS_MPL:
        return
    rows: list[tuple[str, float]] = []
    for name, data in runs:
        values = [
            float(max(0, int(s.skipped_frames or 0)))
            for s in data.greedy_sims
            if s.skipped_frames is not None
        ]
        if values:
            rows.append((name, float(np.mean(values))))
    if not rows:
        return

    rows.sort(key=lambda x: x[1], reverse=True)
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = ["#e74c3c" if v > 0 else "#27ae60" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.45)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(v, bar.get_y() + bar.get_height() / 2, f"  {v:.1f}", va="center", fontsize=8)
    ax.set_xlabel("Mean skipped frames / sim")
    ax.set_title("SC2 Grid Search — Skipped Frames per Experiment")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    _save(fig, os.path.join(summary_dir, "comparison_skipped_frames.png"))


def plot_gs_spatial_heatmap(
    runs: list[tuple[str, ExperimentData]],
    summary_dir: str,
) -> None:
    """Aggregate cross-run spatial heatmap using all greedy-sim target histograms."""
    if not _HAS_MPL:
        return
    agg = np.zeros((8, 8), dtype=float)
    for _, data in runs:
        for s in data.greedy_sims:
            if s.xy_hist is None:
                continue
            agg += np.array(s.xy_hist, dtype=float)
    if agg.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(np.log1p(agg), cmap="YlOrRd", origin="upper", extent=[0, 1, 1, 0], aspect="equal")
    fig.colorbar(img, ax=ax, label="log1p(steps)")
    ax.set_title("SC2 Grid Search — Aggregate Spatial Target Heatmap")
    ax.set_xlabel("Screen X (normalised)")
    ax.set_ylabel("Screen Y (normalised)")
    for v in np.linspace(0, 1, 9):
        ax.axhline(v, color="white", linewidth=0.4, alpha=0.5)
        ax.axvline(v, color="white", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    _save(fig, os.path.join(summary_dir, "comparison_spatial_heatmap.png"))


def _append_sc2_grid_summary_section(summary_dir: str) -> None:
    """Append SC2-specific chart links to framework-generated summary.md."""
    summary_path = os.path.join(summary_dir, "summary.md")
    if not os.path.exists(summary_path):
        return
    items = [
        ("comparison_action_entropy.png", "Action entropy comparison"),
        ("comparison_outcomes.png", "Outcome breakdown comparison"),
        ("comparison_skipped_frames.png", "Skipped-frames comparison"),
        ("comparison_supply_capped.png", "Supply-capped comparison"),
        ("comparison_spatial_heatmap.png", "Aggregate spatial heatmap"),
    ]
    present = [(f, alt) for f, alt in items if os.path.exists(os.path.join(summary_dir, f))]
    if not present:
        return
    with open(summary_path, encoding="utf-8") as f:
        content = f.read()
    if "## SC2-specific cross-run charts" in content:
        return
    lines = ["\n## SC2-specific cross-run charts\n\n"]
    lines.extend(f"![{alt}]({filename})\n\n" for filename, alt in present)
    with open(summary_path, "a", encoding="utf-8") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save(fig: "Figure", path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _safe_scale(weight: float | int | None) -> float:
    """Normalisation scale for a reward component weight.

    Returns ``max(abs(weight), 1.0)`` so that:

    * Weights **≥ 1.0** are divided by their actual value — making
      large-weight components comparable across runs where that weight
      differs (e.g. ``score_weight`` swept from 1 to 100 in a grid search).
    * Weights **< 1.0** produce a scale of 1.0, passing the raw reward
      value through unchanged.  For per-step background costs such as
      ``step_penalty`` (weight ≈ 0.001) the raw value already *encodes*
      the weight (``raw = steps × weight``), so dividing again would
      amplify it by ×1000 and swamp every other component.

    Two runs that differ only in a sub-1.0 weight (e.g. 0.001 vs 0.002)
    therefore compare on their proportional raw contributions — the larger
    weight naturally produces a larger raw value, preserving the correct
    relative ordering without any hard-coded component allowlist.

    A zero or absent weight returns 1.0 (pass-through).
    """
    global _WARNED_NON_NUMERIC_WEIGHT
    try:
        scale = abs(float(weight or 0.0))
    except (TypeError, ValueError):
        if not _WARNED_NON_NUMERIC_WEIGHT:
            logger.warning(
                "SC2 reward normalisation: non-numeric reward weight encountered; "
                "using neutral scale 1.0."
            )
            _WARNED_NON_NUMERIC_WEIGHT = True
        return 1.0
    return max(scale, 1.0) if scale > 1e-12 else 1.0


def _normalised_reward_for_sim(sim, reward_cfg: dict[str, float | int]) -> float:
    """Return config-normalized reward for one greedy sim.

    Falls back to the stored reward when reward-components are unavailable.
    """
    if not sim.reward_components:
        return float(sim.reward)

    total = 0.0
    unknown_keys: set[str] = set()
    for key, value in sim.reward_components.items():
        v = float(value)
        if key == "terminal":
            if v > 0.0:
                scale = _safe_scale(reward_cfg["win_bonus"])
            elif v < 0.0:
                # Divide by abs(loss_penalty): exact configured loss maps to -1.0.
                scale = _safe_scale(reward_cfg["loss_penalty"])
            else:
                scale = 1.0
        else:
            if key in _NO_SCALE_COMPONENT_KEYS:
                total += v
                continue
            cfg_key = _REWARD_COMPONENT_TO_CFG_KEY.get(key)
            if cfg_key:
                scale = _safe_scale(reward_cfg[cfg_key])
            else:
                scale = 1.0
                unknown_keys.add(key)
        total += v / scale
    if unknown_keys:
        logger.warning(
            "SC2 reward normalisation: unmapped reward component key(s): %s",
            sorted(unknown_keys),
        )
    return float(total)


def _normalise_rewards_for_summary(data: ExperimentData) -> ExperimentData:
    """Return a copy of *data* with greedy rewards normalized for comparisons."""
    if not data.greedy_sims:
        return data
    reward_cfg: dict[str, float | int] = dict(_DEFAULT_REWARD_CFG)
    if data.reward_config_file and os.path.exists(data.reward_config_file):
        try:
            with open(data.reward_config_file) as f:
                loaded_cfg = yaml.safe_load(f) or {}
            if isinstance(loaded_cfg, dict):
                non_string_keys: list[object] = []
                string_key_cfg: dict[str, object] = {}
                for k, v in loaded_cfg.items():
                    if isinstance(k, str):
                        string_key_cfg[k] = v
                    else:
                        non_string_keys.append(k)
                if non_string_keys:
                    logger.warning(
                        "SC2 reward normalisation: ignored non-string reward config keys in %s: %s",
                        data.reward_config_file,
                        sorted(repr(k) for k in non_string_keys),
                    )
                reward_cfg.update(string_key_cfg)
            else:
                logger.warning(
                    "SC2 reward normalisation: reward config %s is not a mapping; "
                    "falling back to defaults.",
                    data.reward_config_file,
                )
        except (yaml.YAMLError, TypeError, ValueError) as exc:
            logger.warning(
                "SC2 reward normalisation: failed to parse reward config %s (%s); "
                "falling back to neutral scales.",
                data.reward_config_file,
                exc,
            )
    sims = [
        dataclasses.replace(sim, reward=_normalised_reward_for_sim(sim, reward_cfg))
        for sim in data.greedy_sims
    ]
    return dataclasses.replace(data, greedy_sims=sims)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def save_experiment_results(data: ExperimentData, results_dir: str) -> None:
    """Generate SC2-specific and generic plots; write results.md to *results_dir*."""
    os.makedirs(results_dir, exist_ok=True)

    def _img(filename: str, alt: str) -> str:
        """Return a Markdown image tag only if the file was actually written."""
        if os.path.exists(os.path.join(results_dir, filename)):
            return f"\n![{alt}]({filename})\n\n"
        return ""

    sections = [
        f"# Experiment: {data.experiment_name}\n\n**Game:** StarCraft 2\n\n",
        _timings_md(data),
        _summary_md(data),
    ]

    if data.probe_results:
        plot_probe_rewards(data, results_dir)
        sections.append(_probe_table_md(data))
        sections.append(_img("probe_rewards.png", "Probe rewards"))

    if data.cold_start_restarts:
        plot_cold_start_rewards(data, results_dir)
        sections.append(_cold_start_table_md(data))
        sections.append(_img("cold_start_best_rewards.png", "Cold-start best rewards"))

    if data.greedy_sims:
        plot_greedy_rewards(data, results_dir)
        sections.append(_greedy_table_md(data))
        sections.append(_img("greedy_rewards.png", "Greedy rewards"))

        # 2b — Reward-component breakdown.  Only adds a section if the env
        # populated info["episode_reward_components"] AND at least one
        # component is non-zero.
        plot_reward_components(data, results_dir)
        sections.append(_img("reward_components.png", "Reward components"))

        # 2a — Action-frequency breakdown.
        plot_action_frequency(data, results_dir)
        sections.append(_img("action_frequency.png", "Action frequency"))

        # 2c — Economy / game-state feature averages.
        plot_obs_averages(data, results_dir)
        sections.append(_img("obs_averages.png", "Game-state averages"))

        # 2d — Spatial target heatmap.
        plot_spatial_heatmap(data, results_dir)
        sections.append(_img("spatial_heatmap.png", "Spatial target heatmap"))

        # 2e — Episode outcome breakdown.
        plot_outcome_breakdown(data, results_dir)
        sections.append(_img("outcome_breakdown.png", "Outcome breakdown"))

        # 2f — Skipped frames per sim.
        plot_skipped_frames(data, results_dir)
        sections.append(_img("skipped_frames.png", "Skipped frames"))

        # 2g — Supply-capped fraction per sim.
        plot_supply_capped(data, results_dir)
        sections.append(_img("supply_capped.png", "Time supply-capped"))

        # 2h — Resources available over time (best run).
        plot_resource_series(data, results_dir)
        sections.append(_img("resource_series.png", "Resources available over time"))

        # 2i — Army count over time (best run).
        plot_army_count(data, results_dir)
        sections.append(_img("army_count.png", "Army count over time"))

        # 2j — Build order (best run).
        plot_build_order(data, results_dir)
        sections.append(_img("build_order.png", "Build order"))

    plot_reward_trajectory(data, results_dir)
    sections.append(_img("reward_trajectory.png", "Reward trajectory"))

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
    """Grid search summary with config-normalized SC2 rewards for fair comparison."""
    normalised_runs = [
        (name, _normalise_rewards_for_summary(data))
        for name, data in runs
    ]

    def _sc2_extra(r: list[tuple[str, ExperimentData]], d: str) -> None:
        plot_gs_action_entropy(r, d)
        plot_gs_outcome_breakdown(r, d)
        plot_gs_skipped_frames(r, d)
        plot_gs_supply_capped(r, d)
        plot_gs_spatial_heatmap(r, d)

    _framework_save_grid_summary(
        normalised_runs, varied_keys, summary_dir, base_name,
        extra_plots_fn=_sc2_extra,
    )
    _append_sc2_grid_summary_section(summary_dir)
