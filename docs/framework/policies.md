# `BasePolicy`

**Source:** `framework/policies.py` · **You implement:** a subclass in
`framework/policies.py` (game-agnostic) or `games/<name>/` (game-specific,
e.g. `sc2_genetic`)

Every policy inherits `BasePolicy`. The framework selects the active
policy by the `policy_type` string in `training_params.yaml`; built-in
types are wired in `framework/`, and games register their own via
`PolicyExtras` (see [`run_config.md`](run_config.md)).

A policy is fundamentally a **callable**: given a normalised observation it
returns an action. Trainable policies additionally accept feedback and
serialise their state.

## The interface

```python
class BasePolicy(ABC):
    @abstractmethod
    def __call__(self, obs: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def to_cfg(self) -> dict: ...

    def update(self, obs, action, reward, next_obs, done, **kwargs) -> None: ...
    def on_episode_start(self, **kwargs) -> None: ...
    def on_episode_end(self) -> None: ...
    def save(self, path: str) -> None: ...
    def save_trainer_state(self, path: str) -> None: ...
    def load_trainer_state(self, path: str) -> None: ...
```

### `__call__(obs)` — **abstract** (this is "act")

Select an action given an observation array. The loop calls
`action = policy(obs)`. The framework hands you the **raw** observation;
the policy divides by `obs_spec.scales` itself (see
[`obs_spec.md`](obs_spec.md)). Return a `float32` action array shaped for
the env's `action_space`.

> Note: the abstract method is `__call__`, not a method named `act`. The
> `CONTRIBUTING.md` shorthand "`act(obs)`" refers to this callable.

### `to_cfg()` — **abstract**

Return a YAML-serialisable dict fully representing the policy's state. The
default `save()` dumps this to YAML. Tabular policies additionally pickle
their Q-table alongside (see below).

### `update(obs, action, reward, next_obs, done, **kwargs)` — optional

Per-step feedback from the environment. **No-op by default**, so
hand-coded baselines and evolutionary policies (which learn from
episode-level returns, not per-step transitions) simply don't override it.
Online learners (Q-learning, DQN) implement the Bellman update here.

### `on_episode_start(**kwargs)` / `on_episode_end()` — optional

Lifecycle hooks, both no-ops by default. `on_episode_start` is called
before the first step of each episode and is forwarded the `info` dict
from `env.reset()` as `info=` when available — override to reset
episode-scoped hidden state (e.g. an LSTM's `h`/`c`) or prime an
available-actions mask. `on_episode_end` is called once after the episode.

### `save(path)` — default writes `to_cfg()` to YAML

```python
def save(self, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(self.to_cfg(), f, default_flow_style=False, sort_keys=False)
```

### `save_trainer_state(path)` / `load_trainer_state(path)` — optional

Persist/restore **trainer-internal** state that lives beyond the champion
weights — the CMA-ES distribution (mean, σ, covariance), a DQN replay
buffer, etc. Both are no-ops by default. The framework calls these so a
run can resume mid-training. The path is `trainer_state.npz` in the
experiment directory.

## How loading works (there is no `load()` method)

`BasePolicy` has no `load` method. Policies **load through construction**
instead, taking the weights-file path (`GameSpec.weights_file`) as a
constructor argument, or via a `from_cfg(cfg, ...)` classmethod:

```python
# File-backed: reads weights_file if present, else inits random and writes it.
policy = WeightedLinearPolicy(obs_spec, head_names, weights_file)

# From an in-memory dict (not file-backed):
policy = WeightedLinearPolicy.from_cfg(cfg, obs_spec, head_names)
```

The `CONTRIBUTING.md` shorthand "`save(path)` / `load(path)`" maps onto
"`save(path)` writes; construction reads".

## Weight-file format conventions

`WeightedLinearPolicy` (the linear baseline) stores one weight per
`(head, feature)` pair, keyed by **feature name**:

```yaml
steer_weights:
  speed_ms: -0.12
  lateral_offset_m: 0.88
  yaw_error_rad: 1.40
accel_weights:
  speed_ms: 0.30
  ...
brake_weights:
  ...
```

- Head names come from `GameSpec.head_names` (e.g.
  `["steer","accel","brake"]`), serialised as `{head}_weights`.
- The output convention: head `[0]` is continuous (clipped to `[-1, 1]`);
  heads `[1:]` are binary (thresholded at `0`).
- Existing TMNF files (`steer_weights` / `accel_weights` / `brake_weights`)
  are byte-compatible with this scheme.
- Other policy types serialise differently: `NeuralNetPolicy` dumps nested
  weight/bias lists; tabular policies (`epsilon_greedy`, `ucb_q`) write
  hyperparameters to YAML **and** pickle the Q-table to a sibling
  `*_qtable.pkl`; `GeneticPolicy.save()` writes its champion in the
  `WeightedLinearPolicy` YAML format so analytics and inference work
  unchanged.

## The "missing key → 0.0" migration rule

Because weights key on feature **name**, an old champion stays loadable
when you append a new observation feature (see
[`obs_spec.md`](obs_spec.md)). On load, `WeightedLinearPolicy` builds each
head's weight vector by looking up every current feature name in the
loaded dict and **defaulting any missing name to `0.0`**:

```python
self._weights[head] = np.array(
    [float(head_cfg.get(n, 0.0)) for n in names],   # missing → 0.0
    dtype=np.float32,
)
```

It logs a one-line warning when the loaded dimensionality differs from the
current obs dim, then proceeds. The new feature contributes nothing
(weight `0.0`) until training adjusts it — so adding observation features
never invalidates a saved champion, as long as you only **append** and
never rename or reorder. This is the same invariant that lets LIDAR rays
be switched on for an existing weight file.

## Minimal custom policy

```python
import numpy as np
from framework.policies import BasePolicy


class ConstantPolicy(BasePolicy):
    """Always issues the same action — a trivial non-trainable baseline."""

    def __init__(self, action: np.ndarray):
        self._action = np.asarray(action, dtype=np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self._action.copy()

    def to_cfg(self) -> dict:
        return {"policy_type": "constant", "action": self._action.tolist()}
    # update / save / trainer-state all inherit the no-op / YAML defaults.
```

Register a new `policy_type` by adding it to the framework dispatch (for
game-agnostic policies) or to your adapter's `PolicyExtras.factories` and
`PolicyExtras.loop_dispatch` (for game-specific ones). The full policy
taxonomy and per-type hyperparameters live in
[`../../CLAUDE.md`](../../CLAUDE.md#policies).
