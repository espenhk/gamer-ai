# Plan: Per-Episode Termination Reason Tracking

## Problem

When an episode ends, there are four distinct outcomes:

| Outcome | Cause |
|---------|-------|
| `finish` | `track_progress >= 1.0` — car completed the track |
| `crash` | `\|lateral_offset\| > crash_threshold_m` — car went off the road |
| `hard_crash` | `step.done` from client safety net — extreme offset (>50 m) |
| `timeout` | `elapsed_s > max_episode_time_s` — episode time exceeded |

Currently `TMNFEnv.step()` sets `terminated = finished or crashed` and
`truncated = (step.done and not terminated) or time_over`, but **does not record which
outcome occurred**. The `info` dict contains `track_progress` and `finished` but not the
termination category.

This makes it impossible to answer questions like:
- Is the policy crashing more often than timing out?
- As training progresses, do crashes decrease while timeouts stay flat (policy is going
  further but still not finishing)?
- What fraction of simulations end in finish vs crash in the greedy phase?

## Proposed Solution

### 1. Add `termination_reason` to the `info` dict in `TMNFEnv.step()`

```python
# Determine reason (mutually exclusive, checked in priority order)
if finished:
    termination_reason = "finish"
elif crashed:
    termination_reason = "crash"
elif step.done:  # hard crash via client safety net
    termination_reason = "hard_crash"
elif time_over:
    termination_reason = "timeout"
else:
    termination_reason = None  # episode still running

info["termination_reason"] = termination_reason
```

### 2. Thread `termination_reason` through `_run_episode()` in `main.py`

`_run_episode()` returns `(reward, info, throttle_counts, total_steps, trace)`.
Extend to return the final `info` dict (already returned), and read
`info.get("termination_reason")` at the call site.

### 3. Add `termination_reason` to `GreedySimResult` and `ColdStartSimResult`

```python
@dataclass
class GreedySimResult:
    ...
    termination_reason: str | None = None  # "finish", "crash", "hard_crash", "timeout"
```

Populate from the final `info` dict in `_greedy_loop()` and `_cold_start_phase()`.

### 4. Add a termination breakdown chart to `analytics.py`

In `save_experiment_results()`, after the greedy reward chart, add a stacked bar or
pie chart showing the distribution of termination reasons across all greedy sims:

```
Greedy phase terminations (100 sims):
  finish:     12  (12%)
  crash:      61  (61%)
  hard_crash:  5   (5%)
  timeout:    22  (22%)
```

Use `matplotlib` (already imported) to produce `results/termination_reasons.png`.

Also include per-sim reason in the existing `greedy_sims.yaml` output so it can be
inspected without re-running.

## Files to Change

| File | Change |
|------|--------|
| `rl/env.py` | Add `termination_reason` to `info` dict in `step()` |
| `analytics.py` | Add `termination_reason` field to `GreedySimResult`, `ColdStartSimResult`; add breakdown chart |
| `main.py` | Read `termination_reason` from info and populate result dataclasses |

## Testing

1. Unit test in `tests/test_reward.py` or a new `tests/test_env.py`: mock a step that
   triggers each outcome and assert `info["termination_reason"]` is the correct string
2. Run a short training session (10 sims) and confirm `termination_reasons.png` is
   generated in `results/`
3. Inspect `greedy_sims.yaml` and confirm each entry has a `termination_reason` field
4. Regression: existing tests that check `info` dict keys should still pass (new key
   is additive)
