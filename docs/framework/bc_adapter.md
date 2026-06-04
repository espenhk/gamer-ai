# `BCAdapter`

**Source:** `framework/bc.py`, `framework/bc_io.py` · **You implement:**
`games/<name>/bc_adapter.py` · **You wire up:** `GameAdapter.bc`

Behaviour cloning (BC) pre-trains a policy from demonstration data before
the live RL loop starts. The `BCAdapter` Protocol defines the per-game
seam: your adapter owns the game's replay format, dataset assembly, and
per-target fit logic. The framework `run()` orchestrator in
`framework/bc.py` drives the full flow — validate, build, fit, save —
without knowing anything game-specific.

`BCAdapter` is a `typing.Protocol`: your adapter does **not** need to
subclass anything. It just needs to expose the attributes and methods
below (duck typing) and be attached to the `GameAdapter.bc` attribute.

## Attributes

| Attribute | Type | Meaning |
|---|---|---|
| `name` | `str` | Game identifier, matching `GameAdapter.name` for the same game (e.g. `"sc2"`, `"tmnf"`). |
| `supported_targets` | `tuple[str, ...]` | `policy_type` strings this adapter knows how to pre-train. The orchestrator validates the requested target against this tuple before calling `fit_bc`. |
| `default_target` | `str` | Target used when neither `--bc-target` nor `bc_target` in config is set. Must appear in `supported_targets`. |

## Methods

```python
def validate_replay_dir(replay_dir, *, race=None) -> Any
def build_dataset(replay_dir, save_path, *, obs_spec, training_params,
                  race=None, max_replays=None) -> dict
def fit_bc(dataset, obs_spec, *, target, training_params) -> tuple[Any, float]
def summary_extras(dataset, meta, *, target, training_params) -> dict   # optional
```

| Method | Returns | Notes |
|---|---|---|
| `validate_replay_dir` | anything | Fail fast if `replay_dir` cannot be used as a BC source. The return value is ignored by the orchestrator; raise on error. May accept `None` for sources that do not need a replay directory (e.g. TMNF's live-demo source). |
| `build_dataset` | `dict` with `n_episodes`, `n_steps` | Parse `replay_dir` → `demos.npz` at `save_path`. Must write a file readable by `framework.bc_io.load_dataset`. Read all extra knobs (player id, step_mul, …) from `training_params`. |
| `fit_bc` | `(policy, final_bc_loss)` | Pre-train `target` on `dataset`. `policy` must have a `save(path)` method; if it is a `BasePolicy`, `save_trainer_state` is also called by the orchestrator. |
| `summary_extras` | `dict` | *Optional.* Called after `fit_bc` with the fully-loaded dataset in scope. Return game-specific stats to embed under `bc_summary.json["extras"]` (e.g. SC2's `fn_idx_histogram`). Default from the Protocol returns `{}`. |

## `demos.npz` dataset schema

A BC dataset NPZ file contains six required keys:

| Key | dtype | Shape | Description |
|---|---|---|---|
| `obs` | `float32` | `[N, D]` | Concatenated observation vectors across all episodes |
| `actions` | `float32` | `[N, A]` | Corresponding action vectors (game-specific `A`; e.g. SC2 uses `[fn_idx, x, y, queue]`, TMNF uses `[steer, accel, brake]`) |
| `episode_starts` | `int64` | `[E]` | Start index of each episode in `obs` / `actions` |
| `episode_lengths` | `int64` | `[E]` | Step count for each episode |
| `episode_id` | `int64` | `[N]` | Per-sample episode index ∈ `[0, E)` |
| `meta` | `0-d unicode str` | — | JSON-encoded metadata dict (at minimum `n_episodes` and `n_steps`) |

Call `framework.bc_io.load_dataset(path)` to get a flat dict, or pass
`as_episodes=True` to get a list of `(obs_seq, act_seq)` tuples in
episode order (needed by recurrent targets like `sc2_lstm`).

The schema is byte-compatible with the one SC2's `games/sc2/replay_bc.py`
has written since issue #351, so existing `demos.npz` files load unchanged.

## `bc_summary.json` schema

`framework.bc.run` writes `bc_summary.json` into the experiment directory
after each BC run:

```json
{
  "game": "sc2",
  "bc_target": "sc2_reinforce",
  "n_episodes": 12,
  "n_pairs": 8743,
  "bc_race": "terran",
  "final_bc_loss": 0.42,
  "extras": { "fn_idx_histogram": { "0": 312, "2": 148 } }
}
```

| Key | Type | Meaning |
|---|---|---|
| `game` | `str` | `BCAdapter.name` |
| `bc_target` | `str` | Policy type that was pre-trained |
| `n_episodes` | `int` | Number of episodes in the dataset |
| `n_pairs` | `int` | Total `(obs, action)` pairs |
| `bc_race` | `str` | Race filter applied (`"any"` if none) |
| `final_bc_loss` | `float` | Final training loss (gradient methods) or residual metric (least-squares) |
| `extras` | `dict` | Game-specific stats from `summary_extras()` |

`grid_search.py` reads `bc_target` from `bc_summary.json` when validating
warm-start compatibility — keep the key name stable.

## Wiring the adapter

Expose your adapter on the game's `GameAdapter` via the `bc` attribute:

```python
class MyGameAdapter:
    name = "my_game"
    ...
    bc = MyGameBCAdapter()   # or a property if construction is deferred
```

Games that do not support BC simply omit the `bc` attribute (or set it to
`None`). The `--bc` CLI flag checks for a wired adapter and exits with a
clear error if none is present.

## Worked example: adding BC to a new game

Here is a minimal adapter skeleton for a hypothetical game `"mygame"` whose
BC source is a directory of binary replay files and whose only supported
target is `hill_climbing`:

```python
# games/mygame/bc_adapter.py
from __future__ import annotations
import pathlib
import numpy as np
from framework.obs_spec import ObsSpec
from framework.bc_io import load_dataset

class MyGameBCAdapter:
    name = "mygame"
    supported_targets = ("hill_climbing",)
    default_target = "hill_climbing"

    def validate_replay_dir(self, replay_dir, *, race=None):
        path = pathlib.Path(replay_dir or "")
        replays = sorted(path.glob("*.myreplay"))
        if not replays:
            raise ValueError(f"No .myreplay files found in {replay_dir!r}")
        return replays

    def build_dataset(self, replay_dir, save_path, *,
                      obs_spec, training_params, race=None, max_replays=None):
        import json
        replays = sorted(pathlib.Path(replay_dir).glob("*.myreplay"))
        if max_replays:
            replays = replays[:max_replays]

        obs_list, act_list, episode_lengths = [], [], []
        for rpath in replays:
            obs_ep, act_ep = _parse_replay(rpath, obs_spec)
            obs_list.append(obs_ep)
            act_list.append(act_ep)
            episode_lengths.append(len(obs_ep))

        obs = np.concatenate(obs_list).astype(np.float32)
        actions = np.concatenate(act_list).astype(np.float32)
        ep_lengths = np.array(episode_lengths, dtype=np.int64)
        ep_starts = np.concatenate(([0], np.cumsum(ep_lengths[:-1]))).astype(np.int64)
        ep_id = np.repeat(np.arange(len(ep_lengths), dtype=np.int64), ep_lengths)
        meta = {"n_episodes": len(replays), "n_steps": int(obs.shape[0])}
        np.savez_compressed(str(save_path),
                            obs=obs, actions=actions,
                            episode_starts=ep_starts, episode_lengths=ep_lengths,
                            episode_id=ep_id, meta=np.array(json.dumps(meta)))
        return meta

    def fit_bc(self, dataset, obs_spec, *, target, training_params):
        from games.mygame.policies import WeightedLinearPolicy
        obs = dataset["obs"]
        actions = dataset["actions"]
        # Simple least-squares fit
        scales = obs_spec.scales.astype(float)
        norm_obs = obs / scales[np.newaxis, :]
        w, *_ = np.linalg.lstsq(norm_obs, actions[:, 0], rcond=None)
        policy = WeightedLinearPolicy(w)
        loss = float(np.mean((norm_obs @ w - actions[:, 0]) ** 2))
        return policy, loss


def make_bc_adapter() -> MyGameBCAdapter:
    return MyGameBCAdapter()
```

Then wire it up in `games/mygame/adapter.py`:

```python
class MyGameAdapter:
    name = "mygame"
    config_dir = "games/mygame/config"
    bc = MyGameBCAdapter()
    ...
```

That is the entire contract. The framework `run()` orchestrator handles the
rest: it calls `validate_replay_dir`, `build_dataset`, `fit_bc`, saves
`policy_weights.yaml` and `bc_summary.json`, and returns the summary dict
to the caller.

## Existing implementations

| Game | Module | Sources | Targets |
|---|---|---|---|
| SC2 | `games/sc2/bc_adapter.py` | `.SC2Replay` files (requires `--replay-dir`) | `sc2_reinforce`, `sc2_genetic`, `sc2_cmaes`, `sc2_neural_net`, `sc2_neural_dqn`, `sc2_lstm`, `sc2_cnn`, `epsilon_greedy`, `ucb_q` |
| TMNF | `games/tmnf/bc_adapter.py` | In-game `SimplePolicy` live demos (no `--replay-dir` needed) | `hill_climbing` |

See the per-game READMEs (`games/sc2/README.md`, `games/tmnf/README.md`)
for game-specific workflow details and CLI examples.
