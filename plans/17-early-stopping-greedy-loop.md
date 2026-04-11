# Plan: Early Stopping in the Greedy Training Loop

## Problem

`_greedy_loop()` in `main.py` always runs exactly `N_SIMS` simulations regardless of
training progress. Two common failure modes:

**Premature waste:** The policy converges after 20 sims (no new best reward for a long
plateau), but the remaining 80 sims keep running, consuming wall-clock time and game
compute without improving the result.

**No signal of convergence:** The user has no way to know the policy has converged
without manually inspecting reward logs or waiting for the full run to finish.

This matters most at the default `SPEED = 10.0` where episodes are fast, but for longer
tracks or lower speeds each wasted sim is expensive.

## Proposed Solution

Add **patience-based early stopping**: if the best reward has not improved after
`patience` consecutive simulations, stop the greedy loop early.

### New parameter: `patience: int`

Add to `_greedy_loop()` signature:

```python
def _greedy_loop(
    ...
    patience: int = 0,   # 0 = disabled (current behaviour)
) -> tuple[BasePolicy, float, list[GreedySimResult]]:
```

`patience=0` means disabled — the loop runs all `n_sims`, preserving the current
default behaviour.

### Implementation in `_greedy_loop()`

```python
no_improve_streak: int = 0

# Inside the sim loop, after evaluating the candidate:
if improved:
    no_improve_streak = 0
else:
    no_improve_streak += 1

if patience > 0 and no_improve_streak >= patience:
    logger.info(
        "Early stopping: no improvement in last %d sims (best=%.1f). "
        "Stopping at sim %d/%d.",
        patience, best_reward, sim, n_sims,
    )
    break
```

Apply to both the ES gradient path (WeightedLinearPolicy) and the single-candidate
fallback path (NeuralNetPolicy) — both have the same inner loop structure.

### Expose in `train_rl()` and `main()`

Add `PATIENCE = 30` as a tunable constant at the top of `main()` alongside `N_SIMS`:

```python
PATIENCE = 30   # Stop greedy loop if no improvement in this many sims; 0 = run all
```

Pass through to `_greedy_loop()`:
```python
best_policy, best_reward, greedy_sims = _greedy_loop(
    ..., patience=patience
)
```

Also expose in `grid_search.py` as a grid-searchable training param.

### Logging

Log the early-stop event at `INFO` level with: sim index, patience threshold, and
final best reward. Add `early_stopped: bool` and `early_stop_sim: int | None` fields
to `ExperimentData` so analytics can note it on the reward curve plot.

## Why Patience, Not a Threshold?

A fixed reward threshold (`stop if reward > X`) requires knowing a good value in
advance. Patience is self-calibrating: it adapts to whatever reward scale the current
experiment happens to use, and it fires naturally when the optimiser is stuck regardless
of absolute reward level.

## Files to Change

| File | Change |
|------|--------|
| `main.py` | Add `patience` param to `_greedy_loop()`; add `PATIENCE` constant in `main()`; pass through `train_rl()` |
| `analytics.py` | Add `early_stopped` / `early_stop_sim` to `ExperimentData`; annotate reward curve plot |
| `grid_search.py` | Add `patience` to `_ABBREV` and pass through to `train_rl()` |

No changes needed to `policies.py` or `rl/env.py`.

## Testing

1. Unit test: call `_greedy_loop()` with `n_sims=100, patience=5` and a mock env that
   always returns the same reward — confirm it stops after exactly 5 sims
2. Verify `patience=0` runs all 100 sims (backward compatibility)
3. Run a real session with `patience=20` and confirm `early_stopped=True` is reflected
   in the analytics output when training converges early
4. Regression: existing tests that check `_greedy_loop` return values should pass
   (return signature is unchanged)
