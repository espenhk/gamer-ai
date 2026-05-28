# Framework Policies

This directory contains the game-agnostic RL policy framework used by all games
(TMNF, TORCS, SC2, …).  Each algorithm lives in one canonical module; per-game
policy files are thin registered subclasses that inject game-specific adapters
without duplicating algorithmic code.

---

## Table of Contents

- [Policy registry](#policy-registry)
- [Algorithm modules](#algorithm-modules)
  - [replay.py — ReplayBuffer](#replaypy--replaybuffer)
  - [dqn.py — DQNPolicy](#dqnpy--dqnpolicy)
  - [reinforce.py — REINFORCEPolicy / TwoHeadREINFORCEPolicy](#reinforcepy--reinforcepolicy--twoheadreinforcepolicy)
  - [cmaes.py — CMAESPolicy](#cmaespy--cmaespolicy)
  - [lstm.py — LSTMCore / LSTMEvolutionPolicy](#lstmpy--lstmcore--lstmevolutionpolicy)
- [Simple policies in policies.py](#simple-policies-in-policiespy)
- [Creating a new game's policies](#creating-a-new-games-policies)
- [Adaptation hooks reference](#adaptation-hooks-reference)

---

## Policy registry

Every trainable policy class is stored in `POLICY_REGISTRY`, a `dict[str, type[BasePolicy]]`
populated by the `@register_policy` decorator.

```python
from framework.policies import POLICY_REGISTRY, register_policy, BasePolicy

@register_policy
class MyGamePolicy(BasePolicy):
    POLICY_TYPE = "my_policy"    # key in training_params.yaml
    LOOP_TYPE   = "hill_climbing"  # training loop: hill_climbing | q_learning | cmaes | genetic
    VALID_POLICY_PARAMS = frozenset({"hidden_size"})

    @classmethod
    def compatible_with(cls, game_name: str) -> tuple[bool, str | None]:
        if game_name != "mygame":
            return False, "This policy only works with game='mygame'."
        return True, None

    @classmethod
    def _construct_or_resume(cls, *, obs_spec, head_names, discrete_actions,
                             weights_file, policy_params, re_initialize):
        ...
```

### `BasePolicy` class-level attributes

| Attribute | Type | Purpose |
|---|---|---|
| `POLICY_TYPE` | `str` | Key used in `training_params.yaml` |
| `LOOP_TYPE` | `str` | Training loop (`hill_climbing`, `q_learning`, `cmaes`, `genetic`) |
| `VALID_POLICY_PARAMS` | `frozenset[str]` | Allowed keys in `policy_params` section |

### Required methods

| Method | Notes |
|---|---|
| `__call__(obs)` | Return action array |
| `update(obs, action, reward, next_obs, done, **info)` | Per-step update |
| `on_episode_start(**kwargs)` | Called before first step with `info=reset_info` |
| `on_episode_end()` | Called after terminal step |
| `to_cfg()` | Return serialisable dict |
| `save(path)` | Persist weights to YAML |
| `_construct_or_resume(...)` | Factory: build from params **or** resume from file |

---

## Algorithm modules

### `replay.py` — ReplayBuffer

```python
from framework.replay import ReplayBuffer, MaskedReplayBuffer

buf = ReplayBuffer(maxlen=10_000)
buf.push(obs, action_idx, reward, next_obs, done)
obs_b, act_b, rew_b, next_b, done_b = buf.sample(batch_size=64)
```

`MaskedReplayBuffer` extends `ReplayBuffer` with a boolean `mask` column for
storing available-actions masks alongside each transition.

**No game-specific code.** Used directly by `DQNPolicy`.

---

### `dqn.py` — DQNPolicy

Adam MLP Q-network + ε-greedy exploration + experience replay + target network.

```python
from framework.dqn import DQNPolicy

policy = DQNPolicy(
    obs_spec          = my_obs_spec,
    discrete_actions  = my_actions,        # np.ndarray shape (N, action_dim)
    hidden_sizes      = [64, 64],
    available_actions_fn = my_mask_fn,     # optional
)
```

#### Game-specific adaptation hooks

| Hook | Type | Purpose |
|---|---|---|
| `discrete_actions` | `np.ndarray` | The discrete action table (rows = actions) |
| `available_actions_fn` | `(info: dict) -> np.ndarray[bool]` | Per-step action mask; `None` = no masking |

**TMNF (`neural_dqn`)** — uses the 9-element TMNF `DISCRETE_ACTIONS` array; no
available-actions mask (all actions always legal).

**SC2 (`sc2_neural_dqn`)** — uses SC2's `DISCRETE_ACTIONS` (no_op + select_army
+ Move_screen grid rows); masks unavailable SC2 function IDs via
`build_available_actions_mask`.

---

### `reinforce.py` — REINFORCEPolicy / TwoHeadREINFORCEPolicy

Two classes:

1. **`REINFORCEPolicy`** — single softmax head over a discrete action set.
   Used by TMNF's `reinforce` policy type.

2. **`TwoHeadREINFORCEPolicy`** — shared trunk + fn_idx softmax head +
   spatial sigmoid head.  Used by SC2's `sc2_reinforce` policy type.
   Also exports `_GradEntry` (a NamedTuple) for per-step trajectory storage.

```python
from framework.reinforce import REINFORCEPolicy, TwoHeadREINFORCEPolicy

# Single head (TMNF):
policy = REINFORCEPolicy(
    obs_spec       = my_obs_spec,
    action_decoder = lambda idx: DISCRETE_ACTIONS[idx],
    output_dim     = len(DISCRETE_ACTIONS),
)

# Two-head (SC2):
policy = TwoHeadREINFORCEPolicy(
    obs_spec            = sc2_obs_spec,
    n_fn_ids            = N_FUNCTION_IDS,   # 6
    n_spatial           = 2,                # (x, y)
    action_fn           = lambda fn, sp: np.array([fn, sp[0], sp[1], 0.0]),
    available_fn_ids_fn = lambda info: info.get("available_fn_ids"),
)
```

#### Game-specific adaptation hooks

**`REINFORCEPolicy`:**
| Hook | Type | Purpose |
|---|---|---|
| `action_decoder` | `(int) -> np.ndarray` | Convert discrete index to game action |
| `output_dim` | `int` | Logit vector width |
| `available_actions_fn` | `(info) -> np.ndarray[bool]` | Optional per-step mask |

**`TwoHeadREINFORCEPolicy`:**
| Hook | Type | Purpose |
|---|---|---|
| `n_fn_ids` | `int` | Width of the function-ID softmax head |
| `n_spatial` | `int` | Width of the spatial sigmoid head |
| `action_fn` | `(fn_idx, sp_sig) -> np.ndarray` | Assemble action array from head outputs |
| `available_fn_ids_fn` | `(info) -> set[int] \| None` | Available function IDs per step |

---

### `cmaes.py` — CMAESPolicy

`(μ/μ_w, λ)-CMA-ES` (Hansen 2016) outer loop.  The inner individual type is
game-injectable via a `parameter_decoder` factory callable.

```python
from framework.cmaes import CMAESPolicy

policy = CMAESPolicy(
    obs_spec          = my_obs_spec,
    parameter_decoder = lambda flat, spec: MyInnerPolicy(spec).with_flat(flat),
    flat_dim          = my_inner_policy.flat_dim,
    population_size   = 20,
    initial_sigma     = 0.3,
)
```

#### Game-specific adaptation hooks

| Hook | Type | Purpose |
|---|---|---|
| `parameter_decoder` | `(flat: np.ndarray, obs_spec) -> BasePolicy` | Build an inner individual from a flat weight vector |
| `flat_dim` | `int` | Number of parameters in the flat vector |

The inner policy is evaluated by the training loop and its reward is fed back
to `update_distribution(rewards)`.  The inner policy can expose an
`available_actions` hook for masked evaluation if needed.

**TMNF (`cmaes`)** — `parameter_decoder` builds a `WeightedLinearPolicy` from a
flat weight vector.

**SC2 (`sc2_cmaes`)** — `parameter_decoder` builds a `SC2MultiHeadLinearPolicy`.
The outer `CMAESPolicy` additionally overrides `__call__` to apply
available-actions masking during inference from the champion.

---

### `lstm.py` — LSTMCore / LSTMEvolutionPolicy

Two classes:

1. **`LSTMCore`** — single-layer LSTM with TMNF-compatible steer/accel/brake
   output heads.  Supports flat parameter serialisation for use with
   `LSTMEvolutionPolicy`.

2. **`LSTMEvolutionPolicy`** — isotropic-σ ES outer loop wrapping any LSTM
   inner individual.  Accepts an optional `_template` argument to inject a
   custom inner model (e.g. `SC2LSTMPolicy`).

```python
from framework.lstm import LSTMCore, LSTMEvolutionPolicy

# TMNF (uses LSTMCore as inner individual):
policy = LSTMEvolutionPolicy(
    obs_spec        = tmnf_obs_spec,
    hidden_size     = 32,
    population_size = 20,
    initial_sigma   = 0.05,
)

# Any game with a custom inner individual:
my_template = MyGameLSTM(obs_spec, hidden_size=64)
policy = LSTMEvolutionPolicy(
    obs_spec        = my_obs_spec,
    population_size = 20,
    initial_sigma   = 0.03,
    _template       = my_template,   # keyword-only; bypasses LSTMCore
)
```

The `_template` object must implement:
- `flat_dim: int` — total parameter count
- `to_flat() -> np.ndarray` — serialise weights to a 1-D float32 array
- `with_flat(flat: np.ndarray) -> <same type>` — return a new instance from flat weights
- standard policy interface (`__call__`, `update`, `on_episode_start`, `on_episode_end`)

#### Game-specific adaptation hooks

| Hook | Type | Purpose |
|---|---|---|
| `_template` | custom LSTM object | Inner individual type; defaults to `LSTMCore` |

**TMNF (`lstm`)** — default `LSTMCore` template; steer/accel/brake output.

**SC2 (`sc2_lstm`)** — `SC2LSTMPolicy` template; 6-logit fn_idx + 9-cell spatial
output with available-actions masking.

---

## Simple policies in `policies.py`

The following policies are defined in `framework/policies.py` and registered
directly (not parameterised on game-specific hooks):

| `POLICY_TYPE` | Class | Algorithm |
|---|---|---|
| `hill_climbing` | `WeightedLinearPolicy` | Mutate-and-keep linear heads |
| `neural_net` | `NeuralNetPolicy` | MLP, mutate-and-keep |
| `epsilon_greedy` | `EpsilonGreedyPolicy` | Tabular Q-learning, ε-greedy |
| `ucb_q` | `UCBQPolicy` | Tabular UCB1 online Q-learning (renamed from `mcts`) |
| `genetic` | `GeneticPolicy` | Population + crossover + mutation |

Gradient deep-RL policies (`ppo`, `a2c`, `sac`, `td3`, `qr_dqn`,
`recurrent_ppo`) are Stable-Baselines3-backed and registered in
`framework/sb3_policies.py` (`LOOP_TYPE = "sb3"`, install with
`poetry install --with deep_rl`); `alphazero_mcts` (`framework/alphazero.py`)
is a real model-based MCTS that needs a cloneable simulator.

These use the TMNF continuous steer/accel/brake action encoding and are
**incompatible with SC2** (which needs `fn_idx ∈ [0, 5]` plus spatial
coordinates).  The SC2-native equivalents are prefixed `sc2_`.

---

## Creating a new game's policies

### 1. Choose the algorithm module

Pick the algorithm that fits your game:

- **Evolutionary (no gradient)**: `CMAESPolicy` or `LSTMEvolutionPolicy`
- **Deep Q-learning**: `DQNPolicy`
- **Policy gradient**: `REINFORCEPolicy` (discrete) or `TwoHeadREINFORCEPolicy` (fn+spatial)
- **Simple hill-climbing**: subclass `WeightedLinearPolicy` or `NeuralNetPolicy`

### 2. Define a thin registered subclass

```python
# games/mygame/policies.py

from framework.dqn import DQNPolicy as _DQNPolicy
from framework.policies import register_policy, trainer_state_path
from games.mygame.actions import DISCRETE_ACTIONS as _ACTIONS
import os, yaml

@register_policy
class MyGameDQNPolicy(_DQNPolicy):
    POLICY_TYPE = "mygame_dqn"
    LOOP_TYPE   = "q_learning"
    VALID_POLICY_PARAMS = frozenset({
        "hidden_sizes", "learning_rate", "gamma",
        "epsilon_start", "epsilon_end", "epsilon_decay_steps",
    })

    @classmethod
    def compatible_with(cls, game_name):
        if game_name != "mygame":
            return False, "Use game='mygame'."
        return True, None

    def __init__(self, obs_spec, hidden_sizes=None, **kwargs):
        super().__init__(
            obs_spec         = obs_spec,
            discrete_actions = _ACTIONS,
            hidden_sizes     = hidden_sizes,
            **kwargs,
        )

    def to_cfg(self):
        cfg = super().to_cfg()
        cfg["policy_type"] = "mygame_dqn"
        return cfg

    @classmethod
    def from_cfg(cls, cfg, obs_spec):
        obj = cls(obs_spec, hidden_sizes=cfg.get("hidden_sizes", [64, 64]))
        # restore weights from cfg ...
        return obj

    @classmethod
    def _construct_or_resume(cls, *, obs_spec, head_names, discrete_actions,
                             weights_file, policy_params, re_initialize):
        pp = policy_params
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as f:
                cfg = yaml.safe_load(f) or {}
            if isinstance(cfg, dict) and cfg.get("policy_type") == "mygame_dqn":
                return cls.from_cfg(cfg, obs_spec)
        return cls(obs_spec, **{k: pp[k] for k in cls.VALID_POLICY_PARAMS if k in pp})
```

### 3. Import the module from the adapter

The policy registry is populated by `@register_policy` at import time.  Your
game's adapter must import the policy module so the decorators run:

```python
# games/mygame/adapter.py
import games.mygame.policies  # noqa: F401 — populates POLICY_REGISTRY
```

### 4. Set `policy_type` in your config

```yaml
# experiments/mygame/<name>/training_params.yaml
policy_type: mygame_dqn
policy_params:
  hidden_sizes: [128, 64]
  learning_rate: 0.001
```

---

## Adaptation hooks reference

Quick reference — what each algorithm needs from a game integration:

| Algorithm | Mandatory hooks | Optional hooks |
|---|---|---|
| `DQNPolicy` | `discrete_actions` | `available_actions_fn(info) -> bool[N]` |
| `REINFORCEPolicy` | `action_decoder(idx) -> array`, `output_dim` | `available_actions_fn(info) -> bool[N]` |
| `TwoHeadREINFORCEPolicy` | `n_fn_ids`, `n_spatial`, `action_fn(fn_idx, sp_sig) -> array` | `available_fn_ids_fn(info) -> set[int] \| None` |
| `CMAESPolicy` | `parameter_decoder(flat, obs_spec) -> policy`, `flat_dim` | — |
| `LSTMEvolutionPolicy` | — | `_template` (custom inner LSTM object) |
| `WeightedLinearPolicy` / `NeuralNetPolicy` | — | — |
| `GeneticPolicy` | — | Override `_make_member(cfg)` for custom individual type |
