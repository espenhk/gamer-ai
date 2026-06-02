"""Tests for the SC2 ↔ framework BC seam introduced in phase 2 (#394).

Covers the integration shape — that ``SC2BCAdapter`` plugs into
``framework.bc.run`` end-to-end and produces the canonical framework
summary shape (with SC2 stats under ``extras``).  The detailed SC2 BC
test suite lives in ``tests/test_sc2_replay_bc.py`` and exercises the
backward-compat shim plus the per-target fitters; it remains the
behavioural baseline.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Inject minimal fake PySC2 + s2protocol modules so games.sc2.replay_bc imports
# cleanly without the real SC2 binary.  Mirrors the setup in
# tests/test_sc2_replay_bc.py — kept private here to keep this test file
# standalone.


def _inject_pysc2_stubs() -> None:
    if "pysc2" in sys.modules:
        return
    pysc2 = types.ModuleType("pysc2")
    run_configs = types.ModuleType("pysc2.run_configs")
    run_configs.get = lambda *a, **kw: types.SimpleNamespace()
    pysc2.run_configs = run_configs
    features = types.ModuleType("pysc2.lib.features")
    features.AgentInterfaceFormat = object
    features.Dimensions = object
    features.features_from_game_info = lambda *a, **kw: None
    pysc2.lib = types.ModuleType("pysc2.lib")
    pysc2.lib.features = features
    sys.modules["pysc2"] = pysc2
    sys.modules["pysc2.run_configs"] = run_configs
    sys.modules["pysc2.lib"] = pysc2.lib
    sys.modules["pysc2.lib.features"] = features

    s2 = types.ModuleType("s2protocol")
    versions = types.ModuleType("s2protocol.versions")

    def _build_decoders(_ver):
        return None

    versions.build = _build_decoders
    s2.versions = versions
    sys.modules["s2protocol"] = s2
    sys.modules["s2protocol.versions"] = versions


_inject_pysc2_stubs()

from framework.bc import run as fw_run  # noqa: E402
from games.sc2.bc_adapter import SC2BCAdapter  # noqa: E402
from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC  # noqa: E402

_OBS_DIM = len(SC2_MINIGAME_OBS_SPEC.names)


def _synthetic_dataset(n: int = 30) -> dict:
    actions = np.column_stack(
        [
            np.full(n, 2, np.float32),
            np.full(n, 0.5, np.float32),
            np.full(n, 0.5, np.float32),
            np.zeros(n, np.float32),
        ]
    )
    return {
        "obs": np.zeros((n, _OBS_DIM), dtype=np.float32),
        "actions": actions,
        "episode_starts": np.array([0], dtype=np.int64),
        "episode_lengths": np.array([n], dtype=np.int64),
        "episode_id": np.zeros(n, dtype=np.int64),
        "meta": {"n_episodes": 1, "n_steps": n},
    }


def _patch_sc2_pipeline(dataset: dict):
    """Context manager patching the parts of games.sc2.replay_bc that touch
    real replays / PySC2.  Lets the orchestrator drive SC2BCAdapter end to
    end without an SC2 binary."""
    meta = {
        "n_episodes": 1,
        "n_steps": int(dataset["obs"].shape[0]),
        "obs_dim": _OBS_DIM,
        "n_replays_skipped_race": 0,
        "player_id": "winner",
        "race_filter": None,
    }
    return patch.multiple(
        "games.sc2.replay_bc",
        validate_replay_dir=lambda d, race=None, version=None: [Path(d) / "fake.SC2Replay"],
        build_dataset=lambda *a, **kw: meta,
    ), meta


def test_sc2_adapter_satisfies_framework_protocol_surface():
    a = SC2BCAdapter()
    assert a.name == "sc2"
    assert "sc2_reinforce" in a.supported_targets
    assert a.default_target == "sc2_reinforce"
    for method in ("validate_replay_dir", "build_dataset", "fit_bc", "summary_extras"):
        assert callable(getattr(a, method))


def test_run_through_framework_writes_canonical_summary(tmp_path):
    dataset = _synthetic_dataset(n=20)
    patch_ctx, _meta = _patch_sc2_pipeline(dataset)

    # Patch framework.bc_io.load_dataset too — the framework orchestrator
    # imports it directly, so the SC2 module's patch doesn't bite.
    with patch_ctx, patch("framework.bc.load_dataset", lambda p: dataset):
        replay_dir = tmp_path / "replays"
        replay_dir.mkdir()
        (replay_dir / "g.SC2Replay").touch()
        experiment_dir = tmp_path / "exp"

        summary = fw_run(
            SC2BCAdapter(),
            replay_dir,
            experiment_dir,
            obs_spec=SC2_MINIGAME_OBS_SPEC,
            target="sc2_reinforce",
            training_params={
                "bc_player_id": "winner",
                "bc_epochs": 1,
                "bc_batch_size": 5,
                "bc_ignore_noop": False,
                "seed": 0,
            },
            race="any",
        )

    # Canonical framework summary shape: SC2-specific stats nested under "extras".
    assert summary["game"] == "sc2"
    assert summary["bc_target"] == "sc2_reinforce"
    assert summary["n_pairs"] == 20
    assert summary["bc_race"] == "any"
    assert "extras" in summary
    extras = summary["extras"]
    assert "fn_idx_histogram" in extras
    assert extras["fn_idx_histogram"] == {2: 20}
    assert extras["bc_player_id"] == "winner"

    # bc_summary.json matches the returned dict (modulo JSON key-stringification
    # of the int-keyed fn_idx_histogram).
    on_disk = json.loads((experiment_dir / "bc_summary.json").read_text())
    assert on_disk["extras"]["fn_idx_histogram"] == {"2": 20}
    on_disk_extras = dict(on_disk["extras"])
    on_disk_extras["fn_idx_histogram"] = {int(k): v for k, v in on_disk_extras["fn_idx_histogram"].items()}
    assert {**on_disk, "extras": on_disk_extras} == summary

    # policy_weights.yaml exists.
    assert (experiment_dir / "policy_weights.yaml").exists()


def test_validate_replay_dir_rejects_none():
    a = SC2BCAdapter()
    with pytest.raises(ValueError, match="non-None|--replay-dir|bc_replay_dir"):
        a.validate_replay_dir(None)


def test_supported_targets_covers_all_per_target_fitters():
    """If a per-target fitter is added in games.sc2.replay_bc.fit_bc but not
    declared in SC2BCAdapter.supported_targets, the framework orchestrator
    will refuse to dispatch to it.  Sanity check the union stays in sync."""
    a = SC2BCAdapter()
    expected = {
        "sc2_reinforce",
        "sc2_genetic",
        "sc2_cmaes",
        "sc2_neural_net",
        "sc2_neural_dqn",
        "sc2_lstm",
        "sc2_cnn",
        "epsilon_greedy",
        "ucb_q",
    }
    assert set(a.supported_targets) == expected
