"""Atari Gymnasium environment wrapper.

Wraps ``gymnasium.make("ALE/<Game>-v5", obs_type="ram")`` from `ale-py`,
which ships the MIT-licensed Atari ROMs.  Default obs is the 128-byte
console RAM exposed as a flat float32 vector — directly consumable by all
framework flat-observation policies.

The framework's policy contract delivers actions as a 1-D float array; we
map ``action[0]`` to an integer index in the underlying Discrete action
space:

* Tabular policies (``epsilon_greedy``, ``mcts``) emit ``[i]`` rows from
  ``DISCRETE_ACTIONS`` — i.e. ``i ∈ [0, N_ACTIONS_FULL)`` as a float.  We
  ``int(round(...))`` and clamp.
* Continuous policies (``hill_climbing``, ``neural_net``, ``genetic``,
  ``cmaes``, …) emit a ``tanh``/clip value in ``[-1, 1]``.  We linearly
  map this to ``[0, N_legal − 1]`` using the env's *legal* action set,
  which is typically smaller than 18.

Games whose legal action set is smaller than 18 (e.g. Pong with 6) clamp
out-of-range full-space indices to ``NOOP`` (0) — a safe fallback that
keeps ill-trained policies from crashing the env.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as _exc:  # pragma: no cover — exercised only when gym missing
    raise ImportError(
        "Atari support requires gymnasium and ale-py.  Install with:\n    pip install ale-py gymnasium\n"
    ) from _exc

from framework.base_env import BaseGameEnv
from games.atari.actions import N_ACTIONS_FULL
from games.atari.obs_spec import BASE_OBS_DIM, RAM_SIZE
from games.atari.reward import AtariRewardCalculator, AtariRewardConfig

_DEFAULT_MAP_NAME = "Pong-v5"
_ALE_PREFIX = "ALE/"


def _resolve_env_id(map_name: str) -> str:
    """Return the gymnasium env id for *map_name*.

    Accepts either the bare game name (``"Pong-v5"``) or the fully qualified
    ALE id (``"ALE/Pong-v5"``); the second form is returned unchanged.
    """
    if map_name.startswith(_ALE_PREFIX):
        return map_name
    return _ALE_PREFIX + map_name


class AtariEnv(BaseGameEnv):
    """Gymnasium wrapper around an ALE Atari env in RAM observation mode.

    Parameters
    ----------
    map_name :
        Atari game name (e.g. ``"Pong-v5"``, ``"Breakout-v5"``) or the
        fully qualified ``"ALE/<name>"`` id.
    reward_config :
        ``AtariRewardConfig`` instance.  ``None`` uses Python defaults.
    max_episode_steps :
        Gymnasium episode-step cap; ALE games already have built-in time
        limits, this lets us shorten episodes for fast smoke tests.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        map_name: str = _DEFAULT_MAP_NAME,
        reward_config: AtariRewardConfig | None = None,
        max_episode_steps: int = 5000,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self._map_name = map_name
        self._reward_config = reward_config or AtariRewardConfig()
        self._reward_calc = AtariRewardCalculator(self._reward_config)
        self._max_episode_steps = int(max_episode_steps)

        # Register the ALE namespace with gymnasium.  Importing ale_py is
        # what triggers its env registration; do it lazily so plain
        # ``import games.atari`` works in environments without ale-py.
        try:
            import ale_py  # noqa: F401,PLC0415

            if hasattr(gym, "register_envs"):
                gym.register_envs(ale_py)  # type: ignore[attr-defined]
        except ImportError as exc:  # pragma: no cover — only when ale-py missing
            raise ImportError(
                "Atari support requires the ale-py package.  Install with:\n    pip install ale-py\n"
            ) from exc

        env_id = _resolve_env_id(map_name)
        self._env = gym.make(
            env_id,
            obs_type="ram",
            max_episode_steps=self._max_episode_steps,
            render_mode=render_mode,
        )

        # The Discrete action space size for this particular game (≤ 18).
        underlying_action_space = self._env.action_space
        if not isinstance(underlying_action_space, spaces.Discrete):
            raise TypeError(
                f"Atari env {env_id!r} has a non-discrete action space "
                f"({type(underlying_action_space).__name__}); only Discrete is supported."
            )
        self._n_legal_actions: int = int(underlying_action_space.n)

        self.observation_space = spaces.Box(
            low=0.0,
            high=255.0,
            shape=(BASE_OBS_DIM,),
            dtype=np.float32,
        )
        # We accept either a continuous [-1, 1] head or a discrete index in
        # [0, N_ACTIONS_FULL).  Expose a Box matching the discrete-actions
        # row width (1) so policies that introspect action_space.shape work.
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([float(N_ACTIONS_FULL - 1)], dtype=np.float32),
            dtype=np.float32,
        )

        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_legal_actions(self) -> int:
        """Number of valid actions for the current ALE game."""
        return self._n_legal_actions

    @property
    def map_name(self) -> str:
        return self._map_name

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        raw_obs, info = self._env.reset(seed=seed, options=options)
        self._step_count = 0
        self._reward_calc.reset()
        return self._build_obs(raw_obs), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        idx = self._action_to_index(action)
        raw_obs, native_reward, terminated, truncated, info = self._env.step(idx)
        self._step_count += 1

        info["native_reward"] = float(native_reward)
        info["action_index"] = idx
        info.setdefault("termination_reason", None)
        if terminated:
            info["termination_reason"] = "finish"
        elif truncated:
            info["termination_reason"] = "timeout"

        reward = self._reward_calc.compute(
            prev_state=None,
            curr_state=None,
            finished=terminated,
            elapsed_s=self._step_count / 60.0,  # Atari ticks at ~60 fps
            info=info,
        )
        return self._build_obs(raw_obs), reward, terminated, truncated, info

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # BaseGameEnv hooks
    # ------------------------------------------------------------------

    def _build_obs(self, step: Any) -> np.ndarray:
        """Coerce the raw ALE RAM observation into a float32 vector.

        ALE returns a uint8 array of length 128.  Some wrapper stacks pass
        through other shapes (e.g. (1, 128) batched); this helper accepts
        any 1-D-like input and pads/truncates to ``RAM_SIZE``.
        """
        arr = np.asarray(step, dtype=np.float32).reshape(-1)
        if arr.shape[0] != RAM_SIZE:
            # Should never happen with obs_type="ram", but be defensive in
            # case Gymnasium adds frame-stacking by default in a future
            # ALE release.
            if arr.shape[0] > RAM_SIZE:
                arr = arr[:RAM_SIZE]
            else:
                arr = np.pad(arr, (0, RAM_SIZE - arr.shape[0]))
        return arr.astype(np.float32, copy=False)

    def get_episode_time_limit(self) -> float:
        return float(self._max_episode_steps) / 60.0

    def set_episode_time_limit(self, seconds: float) -> None:
        self._max_episode_steps = max(1, int(seconds * 60))
        # Propagate to the Gymnasium TimeLimit wrapper so the underlying env
        # honours the updated cap — gym.make creates a TimeLimit wrapper whose
        # own counter is independent of our field.
        inner = self._env
        while inner is not None:
            if hasattr(inner, "_max_episode_steps"):
                inner._max_episode_steps = self._max_episode_steps
                break
            inner = getattr(inner, "env", None)

    # ------------------------------------------------------------------
    # Action conversion
    # ------------------------------------------------------------------

    def _action_to_index(self, action: Any) -> int:
        """Map a framework action vector to a legal ALE action index."""
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return 0  # NOOP
        val = float(arr[0])

        # Discrete/tabular policies emit integer-valued indices in [0, N_ACTIONS_FULL).
        # Detect these first so that discrete indices 0 and 1 are not misread as
        # continuous [-1, 1] values.
        _EPS = 1e-4
        if abs(val - round(val)) < _EPS and val >= 0.0:
            idx = int(round(val))
        elif -1.0 <= val <= 1.0 and self._n_legal_actions > 1:
            # Continuous head ([-1, 1]) → linearly map to [0, n_legal − 1].
            idx = int(round((val + 1.0) * 0.5 * (self._n_legal_actions - 1)))
        else:
            idx = int(round(val))

        # Out-of-range (e.g. tabular DISCRETE_ACTIONS row 12 on a Pong-6
        # legal set) clamps to NOOP rather than crashing the env.
        if idx < 0 or idx >= self._n_legal_actions:
            return 0
        return idx


def make_env(
    experiment_dir: str | Path,
    map_name: str = _DEFAULT_MAP_NAME,
    max_episode_time_s: float = 60.0,
    render_mode: str | None = None,
) -> AtariEnv:
    """Factory that wires up an AtariEnv from an experiment directory."""
    experiment_dir = Path(experiment_dir)
    reward_cfg_path = experiment_dir / "reward_config.yaml"
    if reward_cfg_path.exists():
        reward_config = AtariRewardConfig.from_yaml(str(reward_cfg_path))
    else:
        reward_config = AtariRewardConfig()
    max_steps = int(max_episode_time_s * 60)
    return AtariEnv(
        map_name=map_name,
        reward_config=reward_config,
        max_episode_steps=max_steps,
        render_mode=render_mode,
    )
