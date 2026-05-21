# Framework protocols

`gamer-ai` keeps a single game-agnostic training loop (`framework/`) and
plugs each game in behind a small set of protocols. This directory
documents every framework-side seam a contributor touches when shipping a
new game or a new policy, so you can read it end-to-end and write a
`_template` adapter **without** reading any `games/<name>/` code.

The framework mostly avoids depending on `games/`. Core training and
config flow only hold the abstract/protocol types below; concrete game
code lives under `games/<name>/` and is wired in through the adapter.
There are a few optional best-effort imports for nicer logs (for example
SC2 action names), but correctness does not depend on them.

| Doc | Protocol(s) | Lives in | You implement it in |
|---|---|---|---|
| [`game_adapter.md`](game_adapter.md) | `GameAdapter` | `framework/game_adapter.py` | `games/<name>/adapter.py` |
| [`run_config.md`](run_config.md) | `GameSpec`, `RunConfig`, `ProbeSpec`, `WarmupSpec`, `PolicyExtras` | `framework/run_config.py` | built by your adapter |
| [`base_env.md`](base_env.md) | `BaseGameEnv` | `framework/base_env.py` | `games/<name>/env.py` |
| [`reward.md`](reward.md) | `RewardCalculatorBase` (+ `RewardConfig` convention) | `framework/base_reward.py` | `games/<name>/reward.py` |
| [`policies.md`](policies.md) | `BasePolicy` | `framework/policies.py` | `framework/policies.py` or `games/<name>/` |
| [`obs_spec.md`](obs_spec.md) | `ObsSpec`, `ObsDim` | `framework/obs_spec.py` | `games/<name>/obs_spec.py` |

## How the pieces fit together

```
main.py
  └─ adapter = GAME_ADAPTERS[game]()          # GameAdapter
       ├─ adapter.decorate_reward_cfg(...)     # inject game-specific reward keys
       ├─ game_spec = adapter.build_game_spec(...)   # GameSpec (env factory, obs_spec, …)
       └─ train_rl(
              game   = game_spec,              # GameSpec
              config = RunConfig.from_training_params(p),
              probe  = adapter.build_probe(p),     # ProbeSpec | None
              warmup = adapter.build_warmup(p),    # WarmupSpec | None
              extras = adapter.build_extras(...),  # PolicyExtras | None
          )
               └─ env = game_spec.make_env_fn()     # BaseGameEnv
                    loop: obs,reward,term,trunc,info = env.step(action)
                          action = policy(obs)                # BasePolicy
```

Game envs that use a `RewardCalculatorBase` own it internally and call it
from `env.step()`; the outer training loop consumes the returned reward.

Read [`game_adapter.md`](game_adapter.md) first — it is the entry point
that builds everything else. The "Adding a new game" walkthrough in
[`../../CONTRIBUTING.md`](../../CONTRIBUTING.md#adding-a-new-game) links
back here for each protocol.
