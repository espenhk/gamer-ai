# Plan: Multi-Track Support

## Problem

The entire system is coupled to a single track. The centerline path
(`tracks/a03_centerline.npy`) and par time (`par_time_s = 60.0`) are hardcoded in
`rl/env.py` and `rl/reward_config.yaml`. Adding a second track requires editing source
files and risks breaking existing experiments.

## Proposed Solution

1. Add `track_name` and `centerline_path` fields to `RewardConfig` with defaults that
   preserve current behaviour — no existing experiments break
2. `make_env()` reads `centerline_path` from the experiment's `reward_config.yaml`
3. Add a `tracks/registry.yaml` as a human-readable index of available tracks
4. Update `build_centerline.py` to register new tracks automatically

## Files to Change

### `rl/reward.py` — `RewardConfig` dataclass

Add two fields after `par_time_s`:
```python
track_name: str = "a03"
centerline_path: str = "tracks/a03_centerline.npy"
```

Update `from_yaml()` to read these fields (they'll fall back to defaults if absent from
old YAML files, so all existing experiments continue to work unmodified).

### `rl/reward_config.yaml` (master template)

Add two lines:
```yaml
track_name: a03
centerline_path: tracks/a03_centerline.npy
```

These are copied into each new experiment directory on first run (existing logic in
`main.py`), so new experiments get the fields by default.

### `rl/env.py` — `make_env()` (lines 324–346)

Replace the hardcoded centerline path:
```python
# Before:
centerline_file = Path("tracks/a03_centerline.npy")

# After:
centerline_file = Path(reward_config.centerline_path)
```

No other changes needed — `TMNFEnv.__init__` already accepts `centerline_file` as a
parameter.

### `build_centerline.py`

Add CLI argument `--track-name` (defaults to stem of the output path):
```
python build_centerline.py replay.Replay.Gbx \
    --output tracks/b05_centerline.npy \
    --track-name b05 \
    --spacing 2.0
```

After saving the `.npy`, upsert an entry in `tracks/registry.yaml`:
```python
registry_path = Path("tracks/registry.yaml")
registry = yaml.safe_load(registry_path.read_text()) if registry_path.exists() else {}
registry[track_name] = {
    "centerline_path": str(output_path),
    "default_par_time_s": None,   # user fills in manually
    "source_replay": str(replay_path),
}
registry_path.write_text(yaml.dump(registry, sort_keys=True))
```

### New file: `tracks/registry.yaml`

```yaml
a03:
  centerline_path: tracks/a03_centerline.npy
  default_par_time_s: 60.0
  source_replay: replays/a03_centerline.Replay.Gbx
```

This file is committed to the repo. It serves as the human-readable index of all
available tracks and their default parameters.

## Workflow for Adding a New Track

1. Record a replay on the new track
2. Run: `python build_centerline.py path/to/replay.Replay.Gbx --track-name b05 --output tracks/b05_centerline.npy`
3. Edit `tracks/registry.yaml` to fill in `default_par_time_s`
4. Create experiment: `python main.py my_b05_run` and edit
   `experiments/my_b05_run/reward_config.yaml`:
   ```yaml
   track_name: b05
   centerline_path: tracks/b05_centerline.npy
   par_time_s: 45.0
   ```
5. Run training as normal — `make_env()` will load the B05 centerline automatically

## Backward Compatibility

- All existing `experiments/*/reward_config.yaml` files lack `track_name` and
  `centerline_path`; `RewardConfig.from_yaml()` will fall back to defaults
  (`"a03"` and `"tracks/a03_centerline.npy"`) — **no change in behaviour**
- The master `rl/reward_config.yaml` gains two new lines; new experiments get them
  automatically via the copy-on-first-run logic in `main.py`

## Files to Change

| File | Change |
|------|--------|
| `rl/reward.py` | Add `track_name` and `centerline_path` to `RewardConfig` dataclass |
| `rl/reward_config.yaml` | Add `track_name: a03` and `centerline_path: tracks/a03_centerline.npy` |
| `rl/env.py` | Replace hardcoded path in `make_env()` with `reward_config.centerline_path` |
| `build_centerline.py` | Add `--track-name` arg; upsert entry in `tracks/registry.yaml` |
| `tracks/registry.yaml` | New file — a03 entry |

## Testing

1. Confirm existing experiments still load: `python main.py existing_exp` — no changes
   in behaviour expected
2. Create a duplicate centerline under a different name:
   ```bash
   cp tracks/a03_centerline.npy tracks/a03_copy.npy
   ```
3. Create experiment with `centerline_path: tracks/a03_copy.npy` in its reward config
4. Run one episode — observations should be identical to the original a03 experiment
   (same geometry, different file path)
5. Test `build_centerline.py` with `--track-name` and verify `tracks/registry.yaml` is
   updated correctly
