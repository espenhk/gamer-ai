"""Tests for the DQN upgrades: Double-DQN, Huber loss, gradient clipping.

These cover the framework-level :class:`framework.dqn.DQNPolicy` directly so
the behaviour is exercised independent of any game wrapper.
"""

from __future__ import annotations

import numpy as np

from framework.dqn import DQNPolicy
from framework.obs_spec import ObsDim, ObsSpec

_SPEC = ObsSpec([ObsDim(f"f{i}", 1.0, "feat") for i in range(4)])
_ACTIONS = np.eye(3, dtype=np.float32)  # 3 discrete actions, 3-dim action vectors


def _policy(**kw) -> DQNPolicy:
    base = dict(min_replay_size=1, batch_size=1, learning_rate=0.01, seed=0)
    base.update(kw)
    return DQNPolicy(_SPEC, _ACTIONS, **base)


def test_defaults_are_upgraded():
    p = DQNPolicy(_SPEC, _ACTIONS)
    assert p._double is True
    assert p._huber is True
    assert p._huber_kappa == 1.0
    assert p._max_grad_norm == 10.0


def test_to_cfg_from_cfg_roundtrips_new_knobs():
    p = _policy(double_dqn=False, huber_loss=False, huber_kappa=2.0, max_grad_norm=None)
    cfg = p.to_cfg()
    assert cfg["double_dqn"] is False
    assert cfg["huber_loss"] is False
    assert cfg["huber_kappa"] == 2.0
    assert cfg["max_grad_norm"] is None

    restored = DQNPolicy.from_cfg(cfg, _SPEC, _ACTIONS)
    assert restored._double is False
    assert restored._huber is False
    assert restored._huber_kappa == 2.0
    assert restored._max_grad_norm is None


def test_legacy_cfg_without_new_keys_loads_upgraded_defaults():
    """An old weight file (no double_dqn/huber/clip keys) loads with the upgraded defaults."""
    legacy = {
        "hidden_sizes": [8, 8],
        "replay_buffer_size": 100,
        "batch_size": 8,
        "gamma": 0.9,
    }
    p = DQNPolicy.from_cfg(legacy, _SPEC, _ACTIONS)
    assert p._double is True
    assert p._huber is True
    assert p._max_grad_norm == 10.0


def test_huber_clamps_loss_gradient():
    p = _policy(huber_loss=True, huber_kappa=1.0)
    assert p._loss_grad(np.array([1000.0]))[0] == 1.0
    assert p._loss_grad(np.array([-1000.0]))[0] == -1.0
    # Inside the kappa band the Huber gradient is the raw residual.
    assert abs(p._loss_grad(np.array([0.3]))[0] - 0.3) < 1e-6


def test_mse_loss_gradient_is_unbounded():
    p = _policy(huber_loss=False)
    assert p._loss_grad(np.array([1000.0]))[0] == 2000.0
    assert p._loss_grad(np.array([-2.5]))[0] == -5.0


def test_huber_kappa_band_width():
    p = _policy(huber_loss=True, huber_kappa=5.0)
    assert p._loss_grad(np.array([3.0]))[0] == 3.0  # within band
    assert p._loss_grad(np.array([100.0]))[0] == 5.0  # clamped to kappa


def test_grad_clip_keeps_weights_finite_under_extreme_error():
    p = _policy(max_grad_norm=0.5, learning_rate=1.0)
    obs = np.ones(4, dtype=np.float32)
    for _ in range(5):
        p.update(obs, 1, 1e6, obs, False)
    assert np.all(np.isfinite(p._online["weights"][0]))
    assert np.all(np.isfinite(p._online["weights"][-1]))


def test_double_dqn_runs_and_is_finite():
    p = _policy(double_dqn=True)
    obs = np.ones(4, dtype=np.float32)
    for r in (1.0, -1.0, 0.5):
        p.update(obs, 0, r, obs, False)
    q = p._q_values(p._online, (obs / _SPEC.scales).astype(np.float32))
    assert np.all(np.isfinite(q))
