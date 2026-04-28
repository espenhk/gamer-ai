# Plan: Canonical Score for Cross-Experiment Reward Comparability

## Context

The training reward is a weighted sum of physical signals (progress, centerline deviation, speed, etc.). When grid-searching over reward weights, running the same drive with `progress_weight=20000` instead of `10000` produces double the accumulated reward — even though the car drove identically. This makes cross-experiment reward plots misleading.

The fix is two-pronged:
1. **Track physical metrics per episode** (weight-independent): `track_progress`, `finished`, `mean_lateral_offset_m`
2. **Compute a canonical score** from those metrics using fixed hardcoded weights, so all experiments are judged by the same ruler

Raw reward plots are kept for within-experiment diagnostics (they show the actual training signal). Canonical score is added alongside for cross-experiment comparison.

---

## Canonical Score Formula

Defined in `framework/analytics.py` as `compute_canonical_score(track_progress, finished, mean_lateral_offset_m)`:

```
canonical_score = 1000.0 * track_progress
                + 500.0  * finished         (bool → 0.0 or 1.0)
                - 2.0    * mean_lateral_offset_m²
```

**Rationale for constants:**
- Full clean lap: `1000 + 500 - ~0 ≈ 1500`
- Half-lap crash at 2 m avg offset: `500 + 0 - 8 = 492`
- Progress and finish terms dominate; offset penalty is meaningful but can't swamp a good drive (2 m avg → −8 pts, 10 m avg → −200 pts)
- Weights are fixed in code, never configurable

---

## Changes

### 1. `framework/analytics.py`

**Add** `compute_canonical_score()` function before the dataclass definitions.

**Extend `RunTrace`** with 4 new optional fields (default `None` for backward compat):
```python
track_progress: float | None = None
finished: bool | None = None
mean_lateral_offset_m: float | None = None
canonical_score: float | None = None
```

**Extend `GreedySimResult`** with 1 new optional field:
```python
canonical_score: float | None = None
```

**Add `plot_gs_comparison_canonical()`** — horizontal bar chart ranking experiments by best canonical score (analogous to existing `plot_gs_comparison_rewards`). Guards on `if not eligible: return` so it's a no-op when no canonical scores exist.

**Update `_gs_stats()`** (if it exists) or wherever grid summary stats are gathered to include `best_canonical_score`.

**Update `save_grid_summary()`** to call the new canonical bar chart and add a "Best Canonical" column to the markdown rankings table.

### 2. `framework/training.py`

**In `_run_episode()`**, add two accumulators before the loop:
```python
lateral_offset_sum   = 0.0
lateral_offset_steps = 0
```

Inside the loop, after each `env.step()`, accumulate:
```python
lo = info.get("lateral_offset")
if lo is not None:
    lateral_offset_sum   += abs(float(lo))
    lateral_offset_steps += 1
```

After the loop, compute metrics and pass to `RunTrace`:
```python
mean_lo        = lateral_offset_sum / lateral_offset_steps if lateral_offset_steps > 0 else 0.0
final_progress = float(info.get("track_progress", 0.0))
final_finished = bool(info.get("finished", False))
canon          = compute_canonical_score(final_progress, final_finished, mean_lo)

trace = RunTrace(
    pos_x=pos_x, pos_z=pos_z,
    throttle_state=throttle_state, total_reward=total_reward,
    track_progress=final_progress,
    finished=final_finished,
    mean_lateral_offset_m=mean_lo,
    canonical_score=canon,
)
```

Add `compute_canonical_score` to the existing `from framework.analytics import (...)` block at the top.

**In each greedy loop** (`_greedy_loop`, `_greedy_loop_cmaes`, `_greedy_loop_genetic`, `_greedy_loop_q_learning`), when constructing `GreedySimResult`, add:
```python
canonical_score=trace.canonical_score if trace else None,
```

### 3. `distributed/protocol.py`

**In `_trace()`**, add new field reads with `None` fallbacks:
```python
track_progress=t.get("track_progress"),
finished=t.get("finished"),
mean_lateral_offset_m=t.get("mean_lateral_offset_m"),
canonical_score=t.get("canonical_score"),
```

**In `_greedy_sim()`**, add:
```python
canonical_score=s.get("canonical_score"),
```

Serialization (`experiment_to_json`) uses `dataclasses.asdict()` — no changes needed there.

### 4. `games/tmnf/analytics.py`

**Add `plot_greedy_canonical()`** — per-experiment scatter + best-so-far line of canonical score over greedy simulations. Same structure as `plot_greedy_rewards()`. Guards: skip if no sims have canonical scores. Saves to `greedy_canonical.png`.

**Add `plot_gs_comparison_canonical_progress()`** — cross-experiment canonical-score best-so-far curves over simulation number, analogous to existing `plot_gs_comparison_progress`. Saves to `comparison_canonical_progress.png`.

**Update `save_tmnf_plots()`** to call `plot_greedy_canonical(data, results_dir)` alongside the existing greedy plots.

**Update grid summary call** to include `plot_gs_comparison_canonical_progress`.

---

## Files to Modify

| File | Change |
|---|---|
| `framework/analytics.py` | Add `compute_canonical_score`, extend `RunTrace` + `GreedySimResult`, add canonical grid plot, update grid summary |
| `framework/training.py` | Accumulate `lateral_offset` in `_run_episode`, populate new RunTrace fields, thread `canonical_score` into `GreedySimResult` construction |
| `distributed/protocol.py` | Update `_trace()` and `_greedy_sim()` deserialization to read new fields |
| `games/tmnf/analytics.py` | Add `plot_greedy_canonical`, add `plot_gs_comparison_canonical_progress`, update `save_tmnf_plots` |

---

## Backward Compatibility

- All new dataclass fields default to `None` — existing construction sites that don't pass them are unaffected
- Deserialization uses `.get("key")` → `None` for old JSON without the new keys
- New plot functions all guard with early-return when no canonical scores exist
- No changes to training logic or reward computation

---

## Verification

1. Run `python main.py <any_experiment_name>` — greedy phase should log canonical score in RunTrace, `greedy_canonical.png` should appear in results
2. Run a small grid search with 2 configs having different `progress_weight` values — `comparison_canonical_progress.png` should show similar curves for runs that drove equally well despite different raw rewards
3. Run `python -m pytest tests/` — no regressions
4. Load old experiment JSON via `experiment_from_dict` — all new fields deserialize to `None` without error
