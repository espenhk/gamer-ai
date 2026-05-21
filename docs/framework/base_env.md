# `BaseGameEnv`

**Source:** `framework/base_env.py` · **You implement:**
`games/<name>/env.py`

Every game environment subclasses `BaseGameEnv`, which itself inherits
`gymnasium.Env`. That inheritance is deliberate: your env is a drop-in
standard Gymnasium environment, so SB3, Gymnasium wrappers, and the rest
of the RL ecosystem work unchanged. The framework training loop only ever
holds a reference of type `BaseGameEnv` and never imports from `games/`.

Your adapter's `GameSpec.make_env_fn` (see
[`run_config.md`](run_config.md)) is the zero-arg factory that returns one
of these.

## What `train_rl()` expects

`BaseGameEnv` is the standard Gymnasium contract plus two small hooks.

### `reset()` / `step()` — the Gymnasium contract

You define `observation_space` and `action_space` in `__init__` as usual,
then implement the standard Gymnasium methods:

```python
obs, info = env.reset(seed=None, options=None)
obs, reward, terminated, truncated, info = env.step(action)
```

- **`obs`** — a `float32` array whose shape matches `observation_space`
  (and your [`ObsSpec`](obs_spec.md)). In practice you build it in
  `_build_obs(step)` (see below).
- **`reward`** — the training reward your env returns from `env.step()`.
  Many envs compute or shape this internally via a game-specific
  [`RewardCalculator`](reward.md); the framework loop consumes the value
  as-is.
- **`terminated`** vs **`truncated`** — keep them distinct:
  - `terminated = True` → the episode reached a real end state (finished
    the track, won/lost the game, crashed past a threshold).
  - `truncated = True` → the episode was cut off by a time/step limit, not
    by reaching a terminal state.
- **`info`** — a dict; merge your game-specific metrics in here (see
  `_get_game_info`).

### `_build_obs(step)` — **abstract, you must implement**

```python
@abstractmethod
def _build_obs(self, step: Any) -> np.ndarray:
    """Build the float32 observation vector from current game state."""
```

Build and return the observation array from the current game state. The
shape must match `self.observation_space` and the feature order must match
your `ObsSpec`. This is the single source of the obs vector — `reset()`
and `step()` both route through it.

### `_get_game_info()` — optional helper, defaults to `{}`

```python
def _get_game_info(self) -> dict:
    return {}   # default
```

Return a dict of game-specific metrics for the current step **if your env
chooses to use this helper**. `BaseGameEnv` does not merge this into the
Gymnasium `info` dict for you; several envs build `info` directly in
`reset()` / `step()` and never implement `_get_game_info()`. Recommended
keys for racing-style games:

| Key | Meaning |
|---|---|
| `pos_x`, `pos_z` | For bird's-eye trajectory plots. |
| `track_progress` | Fraction of track completed, `[0, 1]`. |
| `laps_completed` | Cumulative lap count. |
| `lateral_offset` | Metres from the centreline. |

If your env uses a reward calculator, this is also a good place to gather
the signals you will pass through `info` rather than inventing bespoke
positional parameters (see [`reward.md`](reward.md)).

## Episode time limit (optional capability)

Two hooks let the framework scale episode length progressively (the
4-step `25% → 50% → 75% → 100%` schedule). Both have safe defaults, so
envs without a configurable limit work unchanged:

```python
def get_episode_time_limit(self) -> float | None:
    return None   # default: no configurable limit

def set_episode_time_limit(self, seconds: float) -> None:
    pass          # default: no-op
```

Override both if your env supports a variable per-episode wall-clock
limit. `get_…` returns the current limit (or `None`); the loop reads it
once at the start of each phase. `set_…` is called each simulation to
apply the progressive schedule.

## Threading expectations

Most envs (CarRacing, TORCS, SC2) are **step-driven**: `env.step()` runs
the game one tick and returns. No special threading needed — implement
`_build_obs` and you're done.

TMNF is the exception: TMInterface is **callback-driven**
(`on_run_step`), so its env bridges the callback thread and the RL thread
with queues and events (an `_action` queue, a `_state_queue` of
`maxsize=1`, and an `_episode_ready` event). You only need that machinery
if your game pushes state at you instead of letting you pull it. See the
"Threading Model" section of [`../../CLAUDE.md`](../../CLAUDE.md) for the
TMNF specifics.

## Minimal skeleton

```python
import gymnasium as gym
import numpy as np
from framework.base_env import BaseGameEnv
from games.mygame.obs_spec import MYGAME_OBS_SPEC


class MyGameEnv(BaseGameEnv):
    def __init__(self, max_episode_time_s: float):
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(MYGAME_OBS_SPEC.dim,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32,
        )
        self._limit_s = max_episode_time_s

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        step = self._start_game()
        return self._build_obs(step), self._get_game_info()

    def step(self, action):
        step = self._advance_game(action)
        obs = self._build_obs(step)
        terminated = self._reached_goal(step)
        truncated  = self._elapsed_s(step) >= self._limit_s
        return obs, 0.0, terminated, truncated, self._get_game_info()

    def _build_obs(self, step) -> np.ndarray:
        return np.array([...], dtype=np.float32)   # order matches MYGAME_OBS_SPEC

    def _get_game_info(self) -> dict:
        return {"track_progress": ..., "native_reward": ...}

    def get_episode_time_limit(self) -> float | None:
        return self._limit_s

    def set_episode_time_limit(self, seconds: float) -> None:
        self._limit_s = seconds
```
