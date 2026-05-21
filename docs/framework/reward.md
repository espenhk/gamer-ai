# `RewardCalculatorBase` and the `reward_config.yaml` round-trip

**Source:** `framework/base_reward.py` · **You implement:**
`games/<name>/reward.py`

In the current architecture, the **game env** usually owns the reward
calculator and calls it during `env.step()`. The outer training loop then
consumes the scalar `reward` returned by `env.step()` without recomputing
it. `RewardCalculatorBase` documents the calculator contract shared by
those env implementations.

## The interface

```python
from framework.base_reward import RewardCalculatorBase


class MyGameRewardCalculator(RewardCalculatorBase):
    def compute(self, prev_state, curr_state, finished,
                elapsed_s, info, n_ticks=1) -> float:
        ...
```

### `compute(...)` — **abstract, you must implement**

Returns the scalar reward for one RL step.

| Param | Type | Meaning |
|---|---|---|
| `prev_state` | `Any` | Game-specific state snapshot from the *previous* step. |
| `curr_state` | `Any` | Game-specific state snapshot from the *current* step. |
| `finished` | `bool` | `True` when the game signalled episode completion (finish line, reached goal, …). |
| `elapsed_s` | `float` | Wall-clock seconds elapsed in the current episode. |
| `info` | `dict` | The per-step metadata dict your env passes in. In practice this is usually the same data the env returns from `reset()` / `step()`, whether assembled directly there or via a helper such as `_get_game_info()`. **Read reward signals from here** rather than adding bespoke positional params. |
| `n_ticks` | `int` | Number of game ticks covered by this RL step (`≥ 1`). Tick-rate-independent rewards should be scaled by `n_ticks`. |

### `compute_with_components(...)` — optional, default delegates

```python
def compute_with_components(...) -> tuple[float, dict]:
    return self.compute(...), {}   # default: no breakdown
```

Override to expose a per-term breakdown for analytics. Return
`(scalar, components)` where `components: dict[str, float]` maps each named
reward term to its contribution, and the scalar equals the **sum** of all
component values (identical to what `compute` returns). The default
returns an empty dict, signalling "no breakdown available". Component
breakdowns drive the reward-component charts in analytics, so populate
this if you want them.

### `reset()` — optional, default no-op

```python
def reset(self) -> None:
    pass   # default
```

Called at the start of each episode. Override if your calculator is
stateful (e.g. tracks a previous-position or a rolling baseline).

## The `RewardConfig` convention

`RewardCalculatorBase` only mandates `compute`. The repo-wide **convention**
(not enforced by the framework) is to pair each calculator with a
`RewardConfig` dataclass that:

1. declares every reward knob as a typed field with a default, and
2. provides a `from_yaml(path)` classmethod that loads
   `reward_config.yaml`.

Most existing integrations (TMNF, TORCS, SC2) validate the keys and raise
on unknown ones; a few lightweight integrations (for example CarRacing
and BeamNG) currently filter unknown keys instead.

```python
import dataclasses, yaml


@dataclasses.dataclass
class MyGameRewardConfig:
    native_reward_scale: float = 1.0
    step_penalty: float = -0.1
    finish_bonus: float = 100.0

    @classmethod
    def from_yaml(cls, path: str) -> "MyGameRewardConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data.keys()) - valid_keys
        if unknown:
            raise ValueError(f"{path}: unknown reward config keys: {sorted(unknown)}")
        return cls(**data)
```

## The config round-trip

How a `reward_config.yaml` becomes a live calculator:

1. **Master copy.** `main.py` copies
   `games/<name>/config/reward_config.yaml` into the experiment directory
   on first run (so editing the experiment copy never touches the master).
2. **Decoration.** `main.py` loads that copy and calls
   `adapter.decorate_reward_cfg(reward_cfg, training_params, track_override)`,
   which **mutates the dict in place** to inject game-specific keys the
   YAML can't carry by itself — e.g. TMNF resolves the track's
   `centerline_path` and writes it in. The decorated dict is written back
   to disk. (See [`game_adapter.md`](game_adapter.md).)
3. **Load.** Your `GameSpec.reward_config_file` points at that decorated
   file; your calculator's `RewardConfig.from_yaml(...)` reads it and
   constructs the calculator.
4. **Per-step.** Your env calls `calc.compute(...)` (or
   `calc.compute_with_components(...)`) during `step()`, then returns that
   scalar as the Gymnasium `reward`. The calculator sees whatever `info`
   dict the env chooses to pass in.

`decorate_reward_cfg` is the hook for anything that depends on the chosen
track/map and therefore can't live as a static default in the master YAML.
If your game has no such keys, make it a no-op (`pass`) — as CarRacing
does.

## Worked example: TORCS

TORCS follows the full `RewardCalculatorBase` contract and validates
`reward_config.yaml` keys strictly:

```python
@dataclasses.dataclass
class TorcsRewardConfig:
    progress_weight: float = 10.0
    centerline_weight: float = -0.5
    centerline_exp: float = 2.0
    speed_weight: float = 0.05
    step_penalty: float = -0.01
    finish_bonus: float = 100.0
    finish_time_weight: float = -0.1
    par_time_s: float = 120.0
    accel_bonus: float = 0.5
    crash_threshold_m: float = 8.0

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data.keys()) - valid_keys
        if unknown:
            raise ValueError(f"{path}: unknown reward config keys: {sorted(unknown)}")
        return cls(**data)


class TorcsRewardCalculator(RewardCalculatorBase):
    def __init__(self, config: TorcsRewardConfig):
        self.config = config

    def compute(self, prev_state, curr_state, finished, elapsed_s, info, n_ticks=1):
        reward = 0.0
        delta = info.get("track_progress", 0.0) - info.get("prev_progress", 0.0)
        reward += delta * self.config.progress_weight
        reward += (
            self.config.centerline_weight
            * abs(info.get("lateral_offset", 0.0)) ** self.config.centerline_exp
            * n_ticks
        )
        reward += self.config.speed_weight * info.get("speed_ms", 0.0) * n_ticks
        reward += self.config.step_penalty * n_ticks
        if finished:
            reward += self.config.finish_bonus
        return float(reward)
```

Here the env computes the final Gymnasium reward by calling the
calculator during `step()`, and the calculator reads the per-step signals
from the env-provided `info` dict.
