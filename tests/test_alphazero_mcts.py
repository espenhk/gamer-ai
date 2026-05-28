"""Tests for the AlphaZero-style MCTS policy (framework/alphazero.py).

Exercised on a tiny deterministic, deepcopy-able corridor MDP — the kind of
cloneable simulator AlphaZero needs.  No game binary or torch required.
"""

from __future__ import annotations

import threading

import gymnasium as gym
import numpy as np
import pytest

from framework.alphazero import _NON_CLONEABLE_GAMES, AlphaZeroMCTSPolicy, run_alphazero_loop
from framework.base_env import BaseGameEnv
from framework.obs_spec import ObsDim, ObsSpec
from framework.policies import POLICY_REGISTRY
from framework.run_config import GameSpec, RunConfig
from framework.training import train_rl

_SIZE = 5
_ACTIONS = np.array([[-1.0], [1.0]], dtype=np.float32)  # left, right
_OBS_SPEC = ObsSpec([ObsDim(f"p{i}", 1.0, "pos one-hot") for i in range(_SIZE)])


class _CorridorEnv(BaseGameEnv):
    """1-D corridor: start at 0, reach the far end (+1), small step cost."""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.pos = 0
        self.steps = 0

    def _onehot(self) -> np.ndarray:
        o = np.zeros(_SIZE, dtype=np.float32)
        o[self.pos] = 1.0
        return o

    def _build_obs(self, step):
        return self._onehot()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = 0
        self.steps = 0
        return self._onehot(), {}

    def step(self, action):
        move = 1 if float(np.asarray(action).reshape(-1)[0]) > 0 else -1
        self.pos = int(np.clip(self.pos + move, 0, _SIZE - 1))
        self.steps += 1
        reached = self.pos == _SIZE - 1
        reward = 1.0 if reached else -0.05
        truncated = self.steps >= 20
        return self._onehot(), reward, reached, truncated, {}


class _NonCloneableEnv(_CorridorEnv):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()  # cannot be deepcopied


def _policy(**kw) -> AlphaZeroMCTSPolicy:
    base = dict(n_simulations=16, c_puct=1.5, gamma=0.99, hidden_sizes=[32], seed=0)
    base.update(kw)
    return AlphaZeroMCTSPolicy(_OBS_SPEC, _ACTIONS, **base)


def test_registered():
    assert "alphazero_mcts" in POLICY_REGISTRY
    assert POLICY_REGISTRY["alphazero_mcts"].LOOP_TYPE == "alphazero"


def test_gated_off_non_cloneable_games():
    # Iterate the whole denylist so any drift between entries here and the
    # actual ``game_name`` strings (in particular Assetto Corsa, which uses
    # ``"assetto"`` rather than the dir name ``assetto_corsa``) is caught.
    assert _NON_CLONEABLE_GAMES, "denylist is empty — gating would be a no-op"
    for g in _NON_CLONEABLE_GAMES:
        ok, hint = AlphaZeroMCTSPolicy.compatible_with(g)
        assert ok is False, f"{g!r} should be gated off (in _NON_CLONEABLE_GAMES)"
        assert hint and "clone" in hint.lower()


def test_allowed_on_cloneable_game_name():
    ok, _ = AlphaZeroMCTSPolicy.compatible_with("corridor")
    assert ok is True


def test_call_returns_valid_action():
    p = _policy()
    a = p(np.zeros(_SIZE, dtype=np.float32))
    assert a.shape == (1,)
    assert a[0] in (-1.0, 1.0)


def test_loop_raises_on_non_cloneable_env():
    p = _policy()
    with pytest.raises(RuntimeError, match="cloneable"):
        run_alphazero_loop(env=_NonCloneableEnv(), policy=p, n_sims=1, weights_file="/tmp/_az.yaml", training_params={})


def test_mcts_finds_goal_and_trains(tmp_path):
    weights = str(tmp_path / "policy_weights.yaml")
    spec = GameSpec(
        experiment_name="az_test",
        track="corridor",
        make_env_fn=_CorridorEnv,
        obs_spec=_OBS_SPEC,
        head_names=["move"],
        discrete_actions=_ACTIONS,
        weights_file=weights,
        reward_config_file=str(tmp_path / "reward.yaml"),
        game_name="corridor",
    )
    cfg = RunConfig(
        n_sims=12,
        in_game_episode_s=10.0,
        policy_type="alphazero_mcts",
        policy_params={"n_simulations": 24, "hidden_sizes": [32], "seed": 0},
    )
    data = train_rl(spec, cfg, no_interrupt=True, re_initialize=True)

    assert data.greedy_sims, "no self-play games recorded"
    # With 24 sims/move over a depth-4 corridor, MCTS should reach the goal at
    # least once (a positive-reward game).
    best = max(s.reward for s in data.greedy_sims)
    assert best > 0.0, f"MCTS never reached the goal (best reward {best})"

    import os

    assert os.path.exists(weights)
    resumed = POLICY_REGISTRY["alphazero_mcts"].make(
        obs_spec=_OBS_SPEC,
        head_names=["move"],
        discrete_actions=_ACTIONS,
        weights_file=weights,
        policy_params={},
        re_initialize=False,
    )
    assert resumed(np.eye(_SIZE, dtype=np.float32)[0]).shape == (1,)
