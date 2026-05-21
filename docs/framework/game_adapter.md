# `GameAdapter`

**Source:** `framework/game_adapter.py` · **You implement:**
`games/<name>/adapter.py`

The `GameAdapter` is the single seam between the game-agnostic training
loop and any one game. `main.py` looks the adapter up in the
`GAME_ADAPTERS` registry by `--game` name, then calls its methods to build
the config bundles ([`run_config.md`](run_config.md)) that `train_rl()`
consumes. The framework never imports `games/<name>/` directly — it only
ever calls these methods.

`GameAdapter` is a `typing.Protocol`: your adapter does **not** need to
subclass anything. It just needs to expose the attributes and methods
below (duck typing) and a module-level `make_adapter()` factory.

## Attributes

| Attribute | Type | Meaning |
|---|---|---|
| `name` | `str` | Game key, matches the `--game` choice (e.g. `"car_racing"`). |
| `config_dir` | `str` | Where the master `training_params.yaml` / `reward_config.yaml` live, e.g. `"games/car_racing/config"`. `main.py` copies these into each new experiment. |

## Methods

```python
def experiment_dir(self, experiment_name, training_params, track_override) -> str
def experiment_dir_root(self, training_params, track_override) -> str
def track_label(self, training_params, track_override) -> str
def decorate_reward_cfg(self, reward_cfg, training_params, track_override) -> None
def build_game_spec(self, experiment_name, experiment_dir,
                    weights_file, reward_cfg_file,
                    training_params, track_override) -> GameSpec
def build_probe(self, training_params) -> ProbeSpec | None
def build_warmup(self, training_params) -> WarmupSpec | None
def build_extras(self, weights_file, training_params, re_initialize) -> PolicyExtras | None
```

| Method | Returns | Notes |
|---|---|---|
| `experiment_dir` | path | The directory for **this** run. Created by `main.py`; receives the copied configs and all results. |
| `experiment_dir_root` | path | The parent directory all runs of this game/policy/track share. Used by cross-grid reporting. |
| `track_label` | `str` | Human-readable track/map label used in directory naming. |
| `decorate_reward_cfg` | `None` | **Mutates** `reward_cfg` in place to inject game-specific keys the master YAML can't carry (e.g. TMNF resolves the `centerline_path` from the track). No-op if you have none. |
| `build_game_spec` | `GameSpec` | The core wiring — see [`run_config.md`](run_config.md). |
| `build_probe` | `ProbeSpec` or `None` | Return `None` to skip the probe + cold-start phases (only TMNF uses them today). |
| `build_warmup` | `WarmupSpec` or `None` | Return `None` to skip forced-action warmup. |
| `build_extras` | `PolicyExtras` or `None` | Return `None` unless the game adds its own `policy_type` strings (e.g. SC2's `sc2_genetic`). |

### `experiment_dir` naming convention

The repo nests experiments as
`experiments/<game>/<policy>/<track>/<experiment_name>/`. Derive every
component from the args you are handed so grid search and analytics can
find runs by walking the tree:

```python
def experiment_dir(self, experiment_name, training_params, track_override):
    policy = training_params.get("policy_type", "hill_climbing")
    track  = self.track_label(training_params, track_override)
    return f"experiments/{self.name}/{policy}/{track}/{experiment_name}"

def experiment_dir_root(self, training_params, track_override):
    policy = training_params.get("policy_type", "hill_climbing")
    track  = self.track_label(training_params, track_override)
    return f"experiments/{self.name}/{policy}/{track}"
```

Keep `experiment_dir_root` equal to `experiment_dir` minus the trailing
`experiment_name` segment — cross-grid reporting relies on that.

## Registering the adapter

Add a lazy factory entry to `GAME_ADAPTERS` in
`framework/game_adapter.py`, and add `<name>` to the `--game` choices in
`main.py`:

```python
GAME_ADAPTERS = {
    ...
    "car_racing": lambda: __import__(
        "games.car_racing.adapter", fromlist=["make_adapter"]
    ).make_adapter(),
}
```

The lambda is important: it defers the import so loading one adapter never
pulls in another game's heavy dependencies (`tminterface`, `pysc2`, …).
Keep game-SDK imports **inside** your methods, not at module top, for the
same reason.

## Worked example: `CarRacingAdapter`

`games/car_racing/` is the smallest reference adapter in the repo. The
whole file:

```python
from framework.run_config import GameSpec, ProbeSpec, WarmupSpec, PolicyExtras


class CarRacingAdapter:
    name = "car_racing"
    config_dir = "games/car_racing/config"

    def experiment_dir(self, experiment_name, training_params, track_override):
        policy = training_params.get("policy_type", "hill_climbing")
        track = self.track_label(training_params, track_override)
        return f"experiments/car_racing/{policy}/{track}/{experiment_name}"

    def experiment_dir_root(self, training_params, track_override):
        policy = training_params.get("policy_type", "hill_climbing")
        track = self.track_label(training_params, track_override)
        return f"experiments/car_racing/{policy}/{track}"

    def track_label(self, training_params, track_override):
        return track_override or "car_racing"

    def decorate_reward_cfg(self, reward_cfg, training_params, track_override):
        pass  # CarRacing has no game-specific reward keys to inject

    def build_game_spec(self, experiment_name, experiment_dir,
                        weights_file, reward_cfg_file,
                        training_params, track_override):
        # Imports kept inside the method so a bare `import adapter` stays light.
        from games.car_racing.obs_spec import CAR_RACING_OBS_SPEC
        from games.car_racing.actions import DISCRETE_ACTIONS
        from games.car_racing.analytics import save_experiment_results

        def _make_env():
            from games.car_racing.env import make_env
            return make_env(
                experiment_dir=experiment_dir,
                max_episode_time_s=training_params["in_game_episode_s"],
            )

        return GameSpec(
            experiment_name=experiment_name,
            track=self.track_label(training_params, track_override),
            make_env_fn=_make_env,
            obs_spec=CAR_RACING_OBS_SPEC,
            head_names=["steer", "accel", "brake"],
            discrete_actions=DISCRETE_ACTIONS,
            weights_file=weights_file,
            reward_config_file=reward_cfg_file,
            save_results_fn=save_experiment_results,
        )

    def build_probe(self, training_params):
        return None   # no probe / cold-start

    def build_warmup(self, training_params):
        return None   # no forced warmup

    def build_extras(self, weights_file, training_params, re_initialize):
        return None   # no game-specific policy types


def make_adapter() -> CarRacingAdapter:
    return CarRacingAdapter()
```

That's the entire contract. Everything else a game needs lives in the
objects this adapter references — see the per-protocol pages linked from
[`README.md`](README.md).
