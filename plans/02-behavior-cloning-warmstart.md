# Plan: Behavior Cloning Warm-Start from PD Controller

## Problem

The cold-start phase uses random hill-climbing to find an initial policy. This is
expensive and noisy — it requires many episodes of largely random driving before
finding anything useful. Meanwhile, `SimplePolicy` in `policies.py` already contains
a working hand-tuned PD controller that tracks the centerline successfully. Its
trajectories are free expert demonstrations that are currently wasted.

## Proposed Solution

Before the cold-start search, optionally run `SimplePolicy` for `N_DEMO_LAPS` laps,
record `(obs, action)` pairs, and fit a `WeightedLinearPolicy` (or `NeuralNetPolicy`)
to those pairs via least-squares regression. The result is a policy that already roughly
tracks the centerline on the first training episode, dramatically accelerating convergence.

## New File: `rl/pretrain.py`

```python
"""Behavior cloning: fit a WeightedLinearPolicy to SimplePolicy demonstrations."""

N_DEMO_LAPS = 3

def collect_demos(env, n_laps: int) -> tuple[np.ndarray, np.ndarray]:
    """Drive n_laps with SimplePolicy, return (obs_matrix, action_matrix)."""
    ...

def fit_weighted_linear(obs_matrix, action_matrix, n_lidar: int = 0) -> WeightedLinearPolicy:
    """Fit steer/accel/brake heads via lstsq. Returns a new WeightedLinearPolicy."""
    # Normalise obs: divide each column by OBS_SCALES
    # Fit steer: lstsq(norm_obs, steer_labels) → w_steer
    # Fit accel: lstsq(norm_obs, accel_labels) → w_accel  (labels are 0/1)
    # Fit brake: lstsq(norm_obs, brake_labels) → w_brake
    ...

def run(env, experiment_dir: Path, policy_type: str = "hill_climbing") -> None:
    """Collect demos and save pre-trained weights to experiment_dir."""
    ...
```

### `collect_demos` implementation
```python
expert = SimplePolicy()
obs, _ = env.reset()
obs_list, act_list = [], []
laps_done = 0
while laps_done < n_laps:
    action = expert(obs)          # SimplePolicy returns [steer, accel, brake]
    obs_list.append(obs.copy())
    act_list.append(action.copy())
    obs, _, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        laps_done += 1
        obs, _ = env.reset()
return np.array(obs_list), np.array(act_list)
```

### `fit_weighted_linear` implementation
```python
scales = obs_scales_with_lidar(n_lidar)            # from obs_spec.py
norm_obs = obs_matrix / scales[np.newaxis, :]      # shape (N, obs_dim)

w_steer, _, _, _ = np.linalg.lstsq(norm_obs, act_matrix[:, 0], rcond=None)
w_accel, _, _, _ = np.linalg.lstsq(norm_obs, act_matrix[:, 1], rcond=None)
w_brake, _, _, _ = np.linalg.lstsq(norm_obs, act_matrix[:, 2], rcond=None)

return WeightedLinearPolicy.from_cfg({
    "steer_weights": w_steer.tolist(),
    "accel_weights": w_accel.tolist(),
    "brake_weights": w_brake.tolist(),
})
```

## Changes to `main.py`

Add parameter `do_pretrain: bool = False` to `train_rl()`.

In the cold-start section (around line 726), before `_run_probes()`:
```python
if do_pretrain and not weights_file.exists():
    print("--- Pre-training from PD controller demos ---")
    from rl.pretrain import run as pretrain_run
    pretrain_run(env, experiment_dir, policy_type=policy_type)
    print("--- Pre-training complete ---")
```

If `do_pretrain=True` and weights already exist, skip (same logic as cold-start).

## Key Implementation Details

- `SimplePolicy.__call__(obs)` already returns `np.ndarray` of shape `(3,)` with
  `[steer ∈ [-1,1], accel ∈ {0,1}, brake ∈ {0,1}]` — no conversion needed
- Least-squares fits a linear model: this is exactly what `WeightedLinearPolicy`
  implements (dot product of weights with normalised obs)
- For `NeuralNetPolicy`, behavior cloning requires a different fitting procedure
  (gradient descent via backprop); restrict pre-training to `WeightedLinearPolicy`
  and `hill_climbing` policy type for now
- The fitted policy may not be perfect (linear model can't represent the full PD
  controller), but it provides a much better starting point than random weights

## Changes to `grid_search.py`

Two additions are needed so grid search can enable pre-training:

**1. Add `do_pretrain` to `_ABBREV`** (so it gets a short label in experiment directory names when used as a search axis):
```python
"do_pretrain": "pt",
```

**2. Forward `do_pretrain` in the `train_rl()` call** inside `main()`:
```python
data = train_rl(
    ...
    do_pretrain=t.get("do_pretrain", False),
    ...
)
```

Grid search configs can then either fix or sweep it:
```yaml
training_params:
    do_pretrain: true        # fixed — always pretrain
    # or:
    do_pretrain: [true, false]  # search axis — compare pretrain vs cold-start
```

## Files to Change

| File | Change |
|------|--------|
| `rl/pretrain.py` | New file (~80 lines) |
| `main.py` | Add `do_pretrain` param, call `pretrain.run()` before cold-start |
| `grid_search.py` | Add `do_pretrain` to `_ABBREV`; forward param in `train_rl()` call |

## Testing

1. Run `python -c "from rl.pretrain import collect_demos, fit_weighted_linear; ..."` standalone
2. Check that `experiments/<name>/policy_weights.yaml` is created after pretrain
3. Load the pre-trained policy and run one episode; compare average reward to a
   randomly-initialised policy over the same episode — pre-trained should score higher
4. Run full training with `do_pretrain=True` and compare convergence speed (reward
   at sim 10) vs `do_pretrain=False`
