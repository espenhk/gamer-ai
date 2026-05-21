# Config bundles: `GameSpec`, `RunConfig`, `ProbeSpec`, `WarmupSpec`, `PolicyExtras`

**Source:** `framework/run_config.py` · **Built by:** your
`GameAdapter` ([`game_adapter.md`](game_adapter.md))

These five frozen dataclasses are the typed arguments to `train_rl()`.
Grouping related settings keeps the training-loop signature small. Your
adapter builds `GameSpec` once per experiment; `RunConfig` carries
algorithm-level knobs that apply to every game; the optional `ProbeSpec`,
`WarmupSpec`, and `PolicyExtras` are `None` for most games.

```python
data = train_rl(
    game   = game_spec,                          # GameSpec   (required)
    config = RunConfig.from_training_params(p),  # RunConfig  (required)
    probe  = adapter.build_probe(p),             # ProbeSpec  | None
    warmup = adapter.build_warmup(p),            # WarmupSpec | None
    extras = adapter.build_extras(...),          # PolicyExtras | None
)
```

## `GameSpec`

The game/track binding, built once per experiment by the adapter. This is
the object that hands the framework everything game-specific it needs.

| Field | Type | Meaning |
|---|---|---|
| `experiment_name` | `str` | Run name; used in results/dir naming. |
| `track` | `str` | Track/map label for results/dir naming. |
| `make_env_fn` | `Callable[[], BaseGameEnv]` | Zero-arg factory returning a fresh env. Called by the loop; keep game-SDK imports inside it. See [`base_env.md`](base_env.md). |
| `obs_spec` | `ObsSpec` | The observation spec instance. See [`obs_spec.md`](obs_spec.md). |
| `head_names` | `list[str]` | Output-head names, e.g. `["steer", "accel", "brake"]`. Drives `WeightedLinearPolicy` weight keys — see [`policies.md`](policies.md). |
| `discrete_actions` | `np.ndarray` | Discrete action table for tabular policies. |
| `weights_file` | `str` | Path the champion policy is saved to / loaded from. |
| `reward_config_file` | `str` | Path to the (already decorated) `reward_config.yaml`. See [`reward.md`](reward.md). |
| `save_results_fn` | `Callable \| None` | Optional `callable(data, results_dir)` for game-specific analytics. `None` skips it. |

## `RunConfig`

Algorithm-level config shared by all games. Build it with
`RunConfig.from_training_params(p)`, which reads a training-params dict and
applies safe defaults:

| Field | Default | Source key |
|---|---|---|
| `n_sims` | *(required)* | `n_sims` |
| `in_game_episode_s` | *(required)* | `in_game_episode_s` |
| `speed` | `1.0` | `speed` |
| `mutation_scale` | `0.05` | `mutation_scale` |
| `mutation_share` | `1.0` | `mutation_share` |
| `adaptive_mutation` | `True` | `adaptive_mutation` |
| `do_pretrain` | `False` | `do_pretrain` |
| `patience` | `0` | `patience` |
| `policy_type` | `"hill_climbing"` | `policy_type` |
| `policy_params` | `{}` | `policy_params` |
| `training_params` | *(the whole dict)* | — kept verbatim for record-keeping |

## `ProbeSpec` (optional — `None` to skip)

Configures the **probe + cold-start** phases. Only `hill_climbing` runs
them, and today only TMNF returns a non-`None` `ProbeSpec`.

| Field | Type | Meaning |
|---|---|---|
| `actions` | `list` | `(action_array, name)` tuples — the fixed probe actions. |
| `probe_in_game_s` | `float` | In-game seconds per probe action run. |
| `cold_start_restarts` | `int` | Max random restarts in cold-start search. |
| `cold_start_sims` | `int` | Hill-climb sims per restart. |

Returning `None` from `build_probe` skips straight to greedy optimisation.

## `WarmupSpec` (optional — `None` to skip)

A forced-action warmup at the start of each episode (e.g. TMNF forces
full-throttle-straight for the braking-start phase so weights aren't
updated during forced behaviour).

| Field | Type | Meaning |
|---|---|---|
| `action` | `np.ndarray` | The action applied during warmup. |
| `steps` | `int` | Number of warmup steps. |

## `PolicyExtras` (optional — `None` to skip)

Registers game-specific `policy_type` strings the framework doesn't know
about (e.g. SC2's `sc2_genetic`, `sc2_cnn`).

| Field | Type | Meaning |
|---|---|---|
| `factories` | `dict` | Maps `policy_type` name → zero-arg callable building the policy. |
| `loop_dispatch` | `dict` | Maps `policy_type` name → loop-type string telling `train_rl` which training loop to run. |

Return `None` when all your policies are framework built-ins
(`hill_climbing`, `genetic`, `cmaes`, …). See the SC2 adapter for a
populated example.
