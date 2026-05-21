# `ObsSpec` and `ObsDim`

**Source:** `framework/obs_spec.py` · **You implement:**
`games/<name>/obs_spec.py`

`ObsSpec` is the single source of truth for a game's flat observation
vector: feature **names**, **scales**, and **order**. All framework code
that needs feature names, normalisation scales, or dimensionality operates
on an `ObsSpec` rather than bare lists, so it never has to know what any
particular feature means.

You create one module-level `ObsSpec` constant per game and hand it to the
framework via `GameSpec.obs_spec` (see [`run_config.md`](run_config.md)).

## `ObsDim` — one feature

```python
@dataclass(frozen=True)
class ObsDim:
    name: str          # unique feature name, e.g. "speed_ms"
    scale: float       # divisor applied before the value reaches a policy
    description: str    # human-readable, shown in analytics labels
```

## `ObsSpec` — the ordered collection

```python
spec = ObsSpec([
    ObsDim("speed_ms",         50.0, "Vehicle speed in m/s"),
    ObsDim("lateral_offset_m",  5.0, "Metres from centreline"),
    ...
])
```

| Member | Returns | Use |
|---|---|---|
| `spec.dim` (also `len(spec)`) | `int` | Total feature count → your env's `observation_space` shape. |
| `spec.names` | `list[str]` | Ordered feature names → policy weight keys, analytics labels. |
| `spec.scales` | `np.ndarray` (float32) | Divisor array, shape `(dim,)`. |
| `spec.dims` | `list[ObsDim]` | The full ordered list. |
| `spec.with_lidar(n_rays)` | `ObsSpec` | Returns a new spec with `n_rays` extra `lidar_i` dims (scale `1.0`); returns `self` unchanged when `n_rays == 0`. |

## Normalisation convention

Policies normalise the raw observation by dividing by the scales:

```python
obs_normalised = raw_obs / spec.scales
```

Choose each `scale` so the normalised value lands roughly in `[-1, 1]`
across typical play. This matters because linear and tabular policies treat
all features on an equal footing — a feature with a 50× larger raw range
would otherwise dominate every dot product and every mutation step. Pick
the scale as the rough max magnitude of the raw feature (e.g. top speed
~50 m/s → `scale=50.0`).

## Ordering invariant

The order of `ObsDim` entries **is** the layout of the observation vector.
Your env's `_build_obs` (see [`base_env.md`](base_env.md)) must emit
values in exactly this order, and policy weight files key off the names in
this order. Append new features at the **end** — see auto-migration below.

## The auto-migration story ("missing key → 0.0")

Because weight files key on feature **names**, you can extend an `ObsSpec`
without invalidating existing champions:

- **Adding a feature** (append a new `ObsDim` at the end): old weight files
  simply lack that key. `WeightedLinearPolicy` initialises any missing
  feature weight to `0.0`, logging a one-line warning that the loaded dim
  doesn't match the current obs dim. The old champion loads and behaves
  exactly as before (the new feature contributes nothing until trained).
- **LIDAR** is the canonical example: `spec.with_lidar(n_rays)` appends
  `lidar_0 … lidar_{n-1}`, and pre-LIDAR weight files migrate via the same
  missing-key path.

This is why you should only ever **append** features and never rename or
reorder existing ones — a rename looks like "drop the old feature, add a
new zero-initialised one". See [`policies.md`](policies.md) for the
weight-file format that makes this work.

## Worked example

```python
# games/mygame/obs_spec.py
from framework.obs_spec import ObsSpec, ObsDim

MYGAME_OBS_SPEC = ObsSpec([
    ObsDim("speed_ms",         50.0, "Vehicle speed in m/s"),
    ObsDim("lateral_offset_m",  5.0, "Metres from centreline (neg=left)"),
    ObsDim("yaw_error_rad",     3.14159, "Track heading minus car heading"),
    ObsDim("track_progress",    1.0, "Fraction of track completed [0,1]"),
])

# Optionally append n LIDAR rays at runtime:
#   spec = MYGAME_OBS_SPEC.with_lidar(training_params["n_lidar_rays"])
```
