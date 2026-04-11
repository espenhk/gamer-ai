# Plan: Consistent `n_ticks` Scaling Across All Reward Components

## Problem

When the RL thread is slower than the game (common at 10× speed), multiple game ticks
fire between consecutive `env.step()` calls. `StepState.ticks_this_step` records how
many ticks were covered. Some reward components in `RewardCalculator.compute()` scale
by `n_ticks`; others do not:

| Component | Scales with n_ticks? |
|-----------|----------------------|
| `accel_bonus` | Yes (`reward += cfg.accel_bonus * n_ticks`) |
| `step_penalty` | Yes (`reward += cfg.step_penalty * n_ticks`) |
| `centerline_weight` | **No** — sampled once at the latest tick |
| `speed_weight` | **No** — sampled once at the latest tick |
| `airborne_penalty` | **No** — sampled once at the latest tick |
| `finish_bonus` | Correct — one-time event |
| `progress_weight` | Correct — delta is inherently multi-tick |

The inconsistency means:
- A step with `n_ticks=3` accumulates 3× the accel_bonus and step_penalty, but only
  1× the centerline/speed/airborne reward signal.
- At high game speeds with frequent skips, the accel and penalty signals dominate
  relative to the physical-state signals, skewing the policy toward always pressing
  throttle regardless of position.
- Weight tuning in `reward_config.yaml` cannot easily compensate because the scaling
  factor varies dynamically per step.

## Proposed Solution

Scale all per-tick reward components by `n_ticks`:

```python
# centerline: car was off-centreline for n_ticks game ticks
if curr.lateral_offset is not None:
    reward += (
        cfg.centerline_weight * abs(curr.lateral_offset) ** cfg.centerline_exp * n_ticks
    )

# speed: car was travelling at this speed for n_ticks game ticks
reward += cfg.speed_weight * curr.velocity.magnitude() * n_ticks

# airborne: car was airborne for n_ticks game ticks
if airborne and curr.vertical_offset <= 0.0:
    reward += cfg.airborne_penalty * n_ticks
```

`finish_bonus` and `progress_weight` are already correct and should not change.

## Reward Magnitude Impact

Applying `n_ticks` to previously unscaled components will increase their magnitude at
high skip rates. To keep total reward magnitudes comparable to existing experiments,
rescale the affected weights in `config/reward_config.yaml`:

```
# Approximate correction: divide by expected avg ticks_per_step at 10x speed.
# At 10x with ~1.2 avg ticks/step:
centerline_weight: -0.42    # was -0.5, divided by 1.2
speed_weight:       0.042   # was 0.05, divided by 1.2
airborne_penalty:  -0.83    # was -1.0, divided by 1.2
```

Existing experiments are unaffected because they use their own copy of
`reward_config.yaml` in `experiments/<name>/`. Only the master template changes.

## Files to Change

| File | Change |
|------|--------|
| `rl/reward.py` | Add `* n_ticks` to centerline, speed, and airborne reward terms |
| `config/reward_config.yaml` | Rescale `centerline_weight`, `speed_weight`, `airborne_penalty` |

## Testing

1. Unit test in `tests/test_reward.py`: call `compute()` with `n_ticks=1` and `n_ticks=3`;
   verify all per-tick components scale linearly and `finish_bonus` / `progress_weight`
   do not
2. Sanity check: run a 20-sim greedy session and confirm total episode reward magnitude
   is similar to pre-fix (within ~20%) — not wildly different due to the rescaling
3. Verify skip-heavy episodes (high `ep_max_skip`) no longer show disproportionate
   accel/penalty signal by comparing `throttle_counts` to reward breakdown
