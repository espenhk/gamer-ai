# Plan: Strict YAML Validation in `RewardConfig.from_yaml()`

## Problem

`RewardConfig.from_yaml()` uses `cls(**data)` to construct the dataclass from the
parsed YAML dict:

```python
@classmethod
def from_yaml(cls, path: str) -> RewardConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return cls(**data)
```

This has two failure modes:

**1. Unknown keys raise a confusing TypeError.**
If `reward_config.yaml` contains a key that is not a `RewardConfig` field — for example
a typo (`centeline_weight` instead of `centerline_weight`), a key from a future version
of the code, or a comment-like annotation — Python raises:

```
TypeError: __init__() got an unexpected keyword argument 'centeline_weight'
```

There is no indication of which file the bad key came from or what the valid keys are.

**2. Silent typos go undetected.**
A misspelled key silently falls back to the Python default value. For example,
`ceterline_weight: -2.0` loads without error, `centerline_weight` stays at `-0.5`, and
the experiment runs with unintended reward weights. This is especially dangerous because
`reward_config.yaml` is hand-edited per experiment.

## Proposed Solution

Replace the bare `cls(**data)` with an explicit key check before construction:

```python
@classmethod
def from_yaml(cls, path: str) -> RewardConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    valid_keys = {f.name for f in fields(cls)}
    unknown = set(data.keys()) - valid_keys
    if unknown:
        raise ValueError(
            f"{path}: unknown reward config keys: {sorted(unknown)}\n"
            f"Valid keys: {sorted(valid_keys)}"
        )

    return cls(**data)
```

Add `from dataclasses import fields` to the existing import at the top of `rl/reward.py`.

## Why Not Just Catch the TypeError?

Catching `TypeError` would conflate unknown-key errors with genuine bugs (e.g., passing
a `str` where a `float` is expected). An explicit set-difference check catches only the
unknown-key case and surfaces the exact offending key names along with the file path.

## Migration Concern

`grid_search.py` builds reward configs by merging user-provided YAML with base defaults.
If any grid config YAML has extra keys (e.g. `track_name` before it was added to the
dataclass), those runs will now fail explicitly rather than silently ignoring the key.
This is the desired behaviour — the fix surfaces latent bugs.

If backward compatibility for old experiment configs is needed, a `--lenient` flag could
be added to skip validation, but this is not recommended.

## Files to Change

| File | Change |
|------|--------|
| `rl/reward.py` | Add unknown-key check in `RewardConfig.from_yaml()`; add `fields` import |

## Testing

1. Add a test in `tests/test_reward.py`:
   - Write a temp YAML with a misspelled key (`centeline_weight: -1.0`)
   - Call `RewardConfig.from_yaml(path)` — expect `ValueError` with the key name in the message
2. Write a temp YAML with all valid keys — expect no exception
3. Write a temp YAML missing some keys — expect no exception (defaults fill in)
4. Verify existing `config/reward_config.yaml` and all `experiments/*/reward_config.yaml`
   load without error
