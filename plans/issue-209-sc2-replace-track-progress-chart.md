# Issue #209 — SC2: replace "track progress" chart in grid summary

## Problem

`framework/analytics.py` `save_grid_summary` unconditionally uses
`best_track_progress` (always 0.0 for SC2) as the primary ranking metric and
chart for the "Task Metrics" section.  Five locations emit this bad output:

| File | ~Line | Symptom |
|---|---|---|
| `framework/analytics.py` | 680–694 | `plot_gs_comparison_task_metrics` — title/axis hardcode "Best Track Progress" |
| `framework/analytics.py` | 750–756 | Primary sort key for ranking table is `best_track_progress` |
| `framework/analytics.py` | 762–776 | `summary.md` section heading + table column "Best Progress" |
| `framework/analytics.py` | 796–805 | Per-experiment block header `**Best progress: 0.0000**` |
| `framework/analytics.py` | 824–826 | Per-experiment stats table row `| Best track progress | 0.0000 |` |

## Replacement metric

**Win/success rate** — fraction of greedy sims with
`termination_reason in {"win", "finish"}`.  Already fully populated for every
SC2 sim; directly analogous to TMNF's track-completion signal.

## Implementation

### 1. Parameterise `plot_gs_comparison_task_metrics` (`framework/analytics.py`)

Add `metric_label: str = "Best Track Progress"` parameter; substitute into
x-axis label and chart title.

### 2. Add plugin parameters to `save_grid_summary` (`framework/analytics.py`)

```python
def save_grid_summary(
    runs, varied_keys, summary_dir, base_name,
    extra_plots_fn=None,
    task_metric_fn: Callable[[ExperimentData], float] | None = None,
    task_metric_label: str = "Best Track Progress",
) -> None:
```

- Compute `s["task_metric"]` per experiment: `task_metric_fn(data)` when
  provided, else fall back to `s["best_track_progress"]`.
- Sort by `task_metric` instead of `best_track_progress`.
- Thread `task_metric_label` through chart call, markdown headings, table
  column, per-experiment block header, and stats table row.
- Format: `:.1%` when `task_metric_fn` is provided; `:.4f` for the default
  (TMNF) path — preserves existing output exactly.

### 3. Add `_sc2_task_metric` helper (`games/sc2/analytics.py`)

```python
def _sc2_task_metric(data: ExperimentData) -> float:
    sims = data.greedy_sims
    if not sims:
        return 0.0
    return sum(1 for s in sims if s.termination_reason in {"win", "finish"}) / len(sims)
```

### 4. Pass the plugin from SC2's `save_grid_summary` (`games/sc2/analytics.py`)

```python
_framework_save_grid_summary(
    normalised_runs, varied_keys, summary_dir, base_name,
    extra_plots_fn=_sc2_extra,
    task_metric_fn=_sc2_task_metric,
    task_metric_label="Win/Success Rate",
)
```

## What does NOT change

- `_gs_stats` — still returns `best_track_progress`; no changes needed there.
- All TMNF call sites — `task_metric_fn=None` default preserves current behaviour.
- `comparison_outcomes.png` — still appended by `_append_sc2_grid_summary_section`.
- Per-experiment `results.md` / `task_metrics.png` — handled separately, unaffected.

## Acceptance criteria

1. SC2 grid summary `comparison_task_metrics.png` shows win/success rate
   values in `[0, 1]` reflecting actual episode outcomes.
2. SC2 `summary.md` ranking section labels are "Win/Success Rate" with
   percentage formatting.
3. TMNF grid summary is byte-for-byte identical to pre-change output.
