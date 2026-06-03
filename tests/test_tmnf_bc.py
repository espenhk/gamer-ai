"""Tests for the TMNF BC adapter (issue #395).

Phase 3 wires :class:`games.tmnf.bc_adapter.TMNFBCAdapter` into the
framework BC seam (#393) and replaces the legacy ``do_pretrain: true``
path.  This suite covers:

- Protocol surface (``name``, ``supported_targets``, ``default_target``,
  required methods).
- ``validate_replay_dir`` synthetic-marker behaviour for the SimplePolicy
  source, and the explicit ``#396`` defer when a ``.Replay.Gbx``
  directory is passed.
- ``fit_bc`` lstsq round-trip on synthetic demos — verifies parity with
  the historical ``rl/pretrain.fit_weighted_linear`` (which produced
  exactly the same numerical fit).
- ``build_dataset`` writes a framework-compatible ``demos.npz`` when
  driven by a fake env (the real TMNF env binds to TMInterface and is
  not exercisable in unit tests).
- End-to-end :func:`framework.bc.run` through the adapter with a stubbed
  TMNF env, asserting the canonical framework summary shape.
"""

from __future__ import annotations

import json
import pathlib
from unittest.mock import patch

import numpy as np
import pytest

from framework.bc import run as fw_run
from framework.bc_io import load_dataset
from framework.obs_spec import ObsSpec
from games.tmnf.bc_adapter import (
    DEFAULT_N_DEMO_LAPS,
    TMNFBCAdapter,
    _bc_residual_loss,
    collect_demos,
    fit_weighted_linear,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tmnf_like_obs_spec(n_lidar: int = 0) -> ObsSpec:
    """The real TMNF obs_spec (possibly extended with N lidar rays).

    Using the real spec is important because
    :meth:`games.tmnf.policies.WeightedLinearPolicy.from_cfg` reconstructs
    its own ``TMNF_OBS_SPEC.with_lidar(n_lidar_rays)`` internally; the
    feature names must match for ``_normalize_weight_cfg`` to see a
    populated weights dict."""
    from games.tmnf.obs_spec import TMNF_OBS_SPEC

    return TMNF_OBS_SPEC.with_lidar(n_lidar)


class _FakeTMNFEnv:
    """Episode-emitting fake.  Each ``step`` returns ``finished=True`` after
    ``ep_steps`` ticks so ``collect_demos`` can finish ``n_laps`` cleanly."""

    def __init__(self, obs_dim: int, ep_steps: int = 4) -> None:
        self._obs_dim = obs_dim
        self._ep_steps = ep_steps
        self._step_in_ep = 0
        self._lap = 0
        self.reset_count = 0
        self.step_count = 0

    def _obs(self) -> np.ndarray:
        # Mildly varying obs so the lstsq is non-degenerate.
        return np.linspace(
            0.1 * (self._lap + 1),
            0.1 * (self._lap + 1) + 0.5,
            self._obs_dim,
            dtype=np.float32,
        )

    def reset(self):
        self.reset_count += 1
        self._step_in_ep = 0
        return self._obs(), {}

    def step(self, _action):
        self._step_in_ep += 1
        self.step_count += 1
        terminated = self._step_in_ep >= self._ep_steps
        truncated = False
        info = {"finished": terminated}
        if terminated:
            self._lap += 1
        return self._obs(), 0.0, terminated, truncated, info

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def test_protocol_surface():
    a = TMNFBCAdapter()
    assert a.name == "tmnf"
    assert a.supported_targets == ("hill_climbing",)
    assert a.default_target == "hill_climbing"
    for method in ("validate_replay_dir", "build_dataset", "fit_bc"):
        assert callable(getattr(a, method))


def test_validate_replay_dir_none_returns_marker():
    a = TMNFBCAdapter()
    result = a.validate_replay_dir(None)
    assert isinstance(result, str)
    assert "simple-policy" in result


def test_validate_replay_dir_with_gbx_defers_to_396(tmp_path):
    (tmp_path / "a.Replay.Gbx").touch()
    (tmp_path / "b.Replay.Gbx").touch()
    a = TMNFBCAdapter()
    with pytest.raises(ValueError, match="#396"):
        a.validate_replay_dir(tmp_path)


def test_build_dataset_with_replay_dir_defers_to_396(tmp_path):
    a = TMNFBCAdapter()
    with pytest.raises(ValueError, match="#396"):
        a.build_dataset(
            tmp_path,
            tmp_path / "demos.npz",
            obs_spec=_tmnf_like_obs_spec(),
            training_params={},
        )


# ---------------------------------------------------------------------------
# fit_bc / fit_weighted_linear
# ---------------------------------------------------------------------------


def test_fit_weighted_linear_recovers_target_weights():
    """Synthesise demos from a known linear policy; the lstsq fit should
    recover those weights tightly when the residual is zero by construction."""
    obs_spec = _tmnf_like_obs_spec(n_lidar=0)
    rng = np.random.default_rng(0)
    n = 500
    obs = rng.uniform(-1.0, 1.0, size=(n, obs_spec.dim)).astype(np.float32)
    target_w = {
        "steer": rng.uniform(-1.0, 1.0, size=obs_spec.dim).astype(np.float32),
        "accel": rng.uniform(-1.0, 1.0, size=obs_spec.dim).astype(np.float32),
        "brake": rng.uniform(-1.0, 1.0, size=obs_spec.dim).astype(np.float32),
    }
    scales = obs_spec.scales.astype(float)
    norm = obs / scales
    actions = np.stack(
        [norm @ target_w["steer"], norm @ target_w["accel"], norm @ target_w["brake"]],
        axis=1,
    )

    policy = fit_weighted_linear(obs, actions, obs_spec)
    fitted_steer = policy._weights["steer"]
    np.testing.assert_allclose(fitted_steer, target_w["steer"], atol=1e-4)


def test_fit_bc_returns_policy_and_loss():
    a = TMNFBCAdapter()
    obs_spec = _tmnf_like_obs_spec()
    rng = np.random.default_rng(1)
    n = 200
    obs = rng.uniform(-1.0, 1.0, size=(n, obs_spec.dim)).astype(np.float32)
    actions = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    dataset = {"obs": obs, "actions": actions}

    policy, loss = a.fit_bc(dataset, obs_spec, target="hill_climbing", training_params={})

    assert hasattr(policy, "save")
    # Residual MSE should match _bc_residual_loss on the same inputs.
    expected = _bc_residual_loss(obs, actions, obs_spec)
    assert loss == pytest.approx(expected)


def test_fit_bc_rejects_unsupported_target():
    a = TMNFBCAdapter()
    with pytest.raises(ValueError, match="hill_climbing"):
        a.fit_bc(
            {"obs": np.zeros((1, 6), dtype=np.float32), "actions": np.zeros((1, 3), dtype=np.float32)},
            _tmnf_like_obs_spec(),
            target="neural_dqn",
            training_params={},
        )


# ---------------------------------------------------------------------------
# collect_demos + build_dataset (with fake env)
# ---------------------------------------------------------------------------


def test_collect_demos_finishes_requested_laps():
    env = _FakeTMNFEnv(obs_dim=6, ep_steps=5)
    obs, acts = collect_demos(env, n_laps=2)
    assert obs.shape == (10, 6)  # 2 laps × 5 steps
    assert acts.shape == (10, 3)
    assert obs.dtype == np.float32
    assert acts.dtype == np.float32


def test_build_dataset_writes_framework_compatible_npz(tmp_path):
    a = TMNFBCAdapter()
    obs_spec = _tmnf_like_obs_spec()
    save_path = tmp_path / "demos.npz"

    with patch("games.tmnf.bc_adapter._make_tmnf_env", return_value=_FakeTMNFEnv(obs_dim=obs_spec.dim, ep_steps=3)):
        meta = a.build_dataset(
            None,
            save_path,
            obs_spec=obs_spec,
            training_params={"bc_n_demo_laps": 2},
        )

    assert save_path.exists()
    assert meta["source"] == "simple_policy"
    assert meta["n_episodes"] == 2
    assert meta["n_steps"] == 6
    assert meta["bc_n_demo_laps"] == 2

    # Loadable via the framework loader.
    data = load_dataset(save_path)
    assert data["obs"].shape == (6, obs_spec.dim)
    assert data["actions"].shape == (6, 3)
    # SimplePolicy demos are written as a single contiguous "episode".
    assert data["episode_starts"].tolist() == [0]
    assert data["episode_lengths"].tolist() == [6]


def test_build_dataset_default_n_demo_laps_matches_historical_constant(tmp_path):
    a = TMNFBCAdapter()
    obs_spec = _tmnf_like_obs_spec()
    fake = _FakeTMNFEnv(obs_dim=obs_spec.dim, ep_steps=1)

    with patch("games.tmnf.bc_adapter._make_tmnf_env", return_value=fake):
        meta = a.build_dataset(
            None,
            tmp_path / "demos.npz",
            obs_spec=obs_spec,
            training_params={},
        )
    assert meta["n_episodes"] == DEFAULT_N_DEMO_LAPS == 3


def test_build_dataset_rejects_zero_step_run(tmp_path):
    a = TMNFBCAdapter()
    obs_spec = _tmnf_like_obs_spec()
    with pytest.raises(ValueError, match="bc_n_demo_laps must be > 0"):
        a.build_dataset(
            None,
            tmp_path / "demos.npz",
            obs_spec=obs_spec,
            training_params={"bc_n_demo_laps": 0},
        )


# ---------------------------------------------------------------------------
# End-to-end through framework.bc.run
# ---------------------------------------------------------------------------


def test_framework_run_end_to_end(tmp_path):
    obs_spec = _tmnf_like_obs_spec(n_lidar=2)
    adapter = TMNFBCAdapter()

    with patch(
        "games.tmnf.bc_adapter._make_tmnf_env",
        return_value=_FakeTMNFEnv(obs_dim=obs_spec.dim, ep_steps=4),
    ):
        summary = fw_run(
            adapter,
            replay_dir=None,
            experiment_dir=tmp_path,
            obs_spec=obs_spec,
            target="hill_climbing",
            training_params={"bc_n_demo_laps": 2},
        )

    assert summary["game"] == "tmnf"
    assert summary["bc_target"] == "hill_climbing"
    assert summary["n_episodes"] == 2
    assert summary["n_pairs"] == 8  # 2 laps × 4 steps
    assert summary["bc_race"] == "any"
    assert isinstance(summary["final_bc_loss"], float)
    assert (tmp_path / "policy_weights.yaml").exists()
    on_disk = json.loads((tmp_path / "bc_summary.json").read_text())
    assert on_disk == summary


# ---------------------------------------------------------------------------
# Wiring through TMNFAdapter.bc
# ---------------------------------------------------------------------------


def test_tmnf_game_adapter_exposes_bc():
    from games.tmnf.adapter import make_adapter

    adapter = make_adapter()
    assert isinstance(adapter.bc, TMNFBCAdapter)


# ---------------------------------------------------------------------------
# do_pretrain removal — make sure the old hooks are really gone
# ---------------------------------------------------------------------------


def test_do_pretrain_removed_from_run_config():
    from dataclasses import fields

    from framework.run_config import RunConfig

    assert "do_pretrain" not in {f.name for f in fields(RunConfig)}


def test_rl_pretrain_module_removed():
    """rl/pretrain.py was the legacy do_pretrain landing pad; phase 3
    deletes it.  The framework BC seam is the supported path now."""
    rl_pretrain = pathlib.Path(__file__).parent.parent / "rl" / "pretrain.py"
    assert not rl_pretrain.exists()


def test_run_config_from_training_params_ignores_do_pretrain():
    """Stale configs with a stray ``do_pretrain: true`` line must not
    crash — RunConfig just drops the key."""
    from framework.run_config import RunConfig

    cfg = RunConfig.from_training_params(
        {
            "n_sims": 1,
            "in_game_episode_s": 1.0,
            "do_pretrain": True,  # legacy — ignored
        }
    )
    assert not hasattr(cfg, "do_pretrain")
