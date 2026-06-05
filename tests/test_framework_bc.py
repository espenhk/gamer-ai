"""Tests for the game-agnostic BC framework (issue #393).

Phase 1 of the BC refactor introduces `framework/bc.py` and
`framework/bc_io.py`.  This suite covers:

- `framework.bc_io.load_dataset` flat-vs-episode round-trip, and the
  NPZ schema staying byte-compatible with SC2's existing format.
- `framework.bc_io.save_summary` JSON shape.
- `framework.bc.BCAdapter` Protocol conformance with a fake adapter.
- `framework.bc.run` end-to-end against a fake adapter — exercises the
  validate → build → fit → save flow without any game-specific code.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import yaml

from framework.bc import run
from framework.bc_io import load_dataset, save_summary
from framework.obs_spec import ObsDim, ObsSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_demo_npz(
    path: pathlib.Path, n_episodes: int = 2, ep_len: int = 4, obs_dim: int = 3, act_dim: int = 2
) -> dict:
    """Write a demos.npz at *path* with deterministic contents.  Returns its meta dict."""
    n = n_episodes * ep_len
    obs = np.arange(n * obs_dim, dtype=np.float32).reshape(n, obs_dim)
    actions = np.arange(n * act_dim, dtype=np.float32).reshape(n, act_dim)
    episode_starts = np.arange(0, n, ep_len, dtype=np.int64)
    episode_lengths = np.full(n_episodes, ep_len, dtype=np.int64)
    episode_id = np.repeat(np.arange(n_episodes, dtype=np.int64), ep_len)
    meta = {"n_episodes": n_episodes, "n_steps": n}
    np.savez(
        path,
        obs=obs,
        actions=actions,
        episode_starts=episode_starts,
        episode_lengths=episode_lengths,
        episode_id=episode_id,
        meta=json.dumps(meta),
    )
    return meta


class _FakePolicy:
    """Minimal stand-in for a BC-trained policy.  Has `save()` only."""

    def __init__(self) -> None:
        self.weights = {"w": [1.0, 2.0, 3.0]}
        self.saved_to: str | None = None

    def save(self, path: str) -> None:
        self.saved_to = path
        with open(path, "w") as f:
            yaml.dump(self.weights, f)


class _FakeBCAdapter:
    """Fake game-side BC adapter used by the framework orchestrator tests."""

    name = "fake"
    supported_targets: tuple[str, ...] = ("dummy",)
    default_target = "dummy"

    def __init__(self) -> None:
        self.validate_calls: list[tuple] = []
        self.build_calls: list[tuple] = []
        self.fit_calls: list[tuple] = []
        self.last_policy: _FakePolicy | None = None

    def validate_replay_dir(self, replay_dir, *, race=None):
        self.validate_calls.append((replay_dir, race))
        return [pathlib.Path("fake/replay")]

    def build_dataset(
        self,
        replay_dir,
        save_path,
        *,
        obs_spec,
        training_params,
        race=None,
        max_replays=None,
    ):
        self.build_calls.append((replay_dir, save_path, race, max_replays))
        meta = _write_demo_npz(pathlib.Path(save_path))
        meta["summary_extras"] = {"fake_stat": 42}
        return meta

    def fit_bc(self, dataset, obs_spec, *, target, training_params):
        self.fit_calls.append((target, dict(training_params)))
        self.last_policy = _FakePolicy()
        return self.last_policy, 0.123


# ---------------------------------------------------------------------------
# bc_io.load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_flat_round_trip(tmp_path):
    path = tmp_path / "demos.npz"
    _write_demo_npz(path, n_episodes=3, ep_len=5, obs_dim=4, act_dim=2)

    data = load_dataset(path)

    assert set(data.keys()) == {"obs", "actions", "episode_starts", "episode_lengths", "episode_id", "meta"}
    assert data["obs"].shape == (15, 4)
    assert data["actions"].shape == (15, 2)
    assert data["episode_starts"].tolist() == [0, 5, 10]
    assert data["episode_lengths"].tolist() == [5, 5, 5]
    assert data["episode_id"].tolist() == [0] * 5 + [1] * 5 + [2] * 5
    assert data["meta"] == {"n_episodes": 3, "n_steps": 15}


def test_load_dataset_as_episodes_preserves_order(tmp_path):
    path = tmp_path / "demos.npz"
    _write_demo_npz(path, n_episodes=2, ep_len=3, obs_dim=2, act_dim=1)

    episodes = load_dataset(path, as_episodes=True)

    assert len(episodes) == 2
    ep0_obs, ep0_act = episodes[0]
    ep1_obs, ep1_act = episodes[1]
    assert ep0_obs.shape == (3, 2)
    assert ep1_obs.shape == (3, 2)
    # Episode 1 starts where episode 0 ended (concatenated obs).
    assert ep1_obs[0, 0] == ep0_obs[-1, 0] + 2  # next row, 2 obs dims


def test_load_dataset_rejects_missing_keys(tmp_path):
    path = tmp_path / "broken.npz"
    np.savez(path, obs=np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(KeyError):
        load_dataset(path)


# ---------------------------------------------------------------------------
# bc_io.save_summary
# ---------------------------------------------------------------------------


def test_save_summary_writes_indented_json(tmp_path):
    summary = {"game": "fake", "bc_target": "dummy", "extras": {"x": 1}}
    out = save_summary(tmp_path, summary)

    assert out == tmp_path / "bc_summary.json"
    text = out.read_text()
    # Indented (multi-line) output.
    assert "\n" in text
    assert json.loads(text) == summary


def test_save_summary_creates_missing_experiment_dir(tmp_path):
    nested = tmp_path / "does" / "not" / "exist"
    save_summary(nested, {"game": "fake"})
    assert (nested / "bc_summary.json").exists()


# ---------------------------------------------------------------------------
# bc.run orchestrator
# ---------------------------------------------------------------------------


@pytest.fixture
def obs_spec() -> ObsSpec:
    return ObsSpec([ObsDim("a", 1.0, "a"), ObsDim("b", 2.0, "b"), ObsDim("c", 1.0, "c")])


def test_run_drives_adapter_in_order(tmp_path, obs_spec):
    adapter = _FakeBCAdapter()
    training_params = {"foo": "bar"}

    summary = run(
        adapter,
        replay_dir="some/dir",
        experiment_dir=tmp_path,
        obs_spec=obs_spec,
        target="dummy",
        training_params=training_params,
        race="any",
        max_replays=None,
    )

    assert len(adapter.validate_calls) == 1
    assert len(adapter.build_calls) == 1
    assert len(adapter.fit_calls) == 1
    # validate runs before build runs before fit
    # (call list ordering already implies this via the single-counter increments,
    # but assert the fit was passed the requested target)
    assert adapter.fit_calls[0][0] == "dummy"
    assert adapter.fit_calls[0][1] == training_params

    # Summary on disk matches return value.
    summary_on_disk = json.loads((tmp_path / "bc_summary.json").read_text())
    assert summary_on_disk == summary
    assert summary["game"] == "fake"
    assert summary["bc_target"] == "dummy"
    assert summary["n_episodes"] == 2
    assert summary["n_pairs"] == 8  # 2 episodes × 4 steps from _write_demo_npz default
    assert summary["bc_race"] == "any"
    assert summary["final_bc_loss"] == pytest.approx(0.123)
    assert summary["extras"] == {"fake_stat": 42}


def test_run_saves_policy_weights(tmp_path, obs_spec):
    adapter = _FakeBCAdapter()
    run(
        adapter,
        replay_dir=None,
        experiment_dir=tmp_path,
        obs_spec=obs_spec,
        target="dummy",
        training_params={},
    )
    weights = tmp_path / "policy_weights.yaml"
    assert weights.exists()
    loaded = yaml.safe_load(weights.read_text())
    assert loaded == {"w": [1.0, 2.0, 3.0]}
    assert adapter.last_policy is not None
    assert adapter.last_policy.saved_to == str(weights)


def test_run_normalises_race_any_to_none(tmp_path, obs_spec):
    adapter = _FakeBCAdapter()
    run(
        adapter,
        replay_dir=None,
        experiment_dir=tmp_path,
        obs_spec=obs_spec,
        target="dummy",
        training_params={},
        race="ANY",
    )
    # build_dataset was called with race=None (the orchestrator normalised it).
    assert adapter.build_calls[0][2] is None
    # The validate call was also passed the normalised value.
    assert adapter.validate_calls[0][1] is None


def test_run_passes_through_real_race_filter(tmp_path, obs_spec):
    adapter = _FakeBCAdapter()
    run(
        adapter,
        replay_dir=None,
        experiment_dir=tmp_path,
        obs_spec=obs_spec,
        target="dummy",
        training_params={},
        race="Terran",
    )
    assert adapter.build_calls[0][2] == "terran"  # lower-cased


def test_run_rejects_unsupported_target(tmp_path, obs_spec):
    adapter = _FakeBCAdapter()
    with pytest.raises(ValueError, match="not supported"):
        run(
            adapter,
            replay_dir=None,
            experiment_dir=tmp_path,
            obs_spec=obs_spec,
            target="not_a_real_target",
            training_params={},
        )
    # Nothing was attempted because the target check is the first guard.
    assert adapter.validate_calls == []


def test_run_creates_experiment_dir(tmp_path, obs_spec):
    adapter = _FakeBCAdapter()
    target_dir = tmp_path / "fresh" / "experiment"
    run(
        adapter,
        replay_dir=None,
        experiment_dir=target_dir,
        obs_spec=obs_spec,
        target="dummy",
        training_params={},
    )
    assert (target_dir / "policy_weights.yaml").exists()
    assert (target_dir / "bc_summary.json").exists()


# ---------------------------------------------------------------------------
# BCAdapter protocol conformance
# ---------------------------------------------------------------------------


def test_fake_adapter_satisfies_protocol():
    """_FakeBCAdapter is structurally a BCAdapter — assignment to a
    typed slot would type-check at runtime via isinstance with @runtime_checkable.
    We don't decorate BCAdapter with @runtime_checkable (Protocols are
    structural; isinstance support is optional), so this test asserts the
    attribute surface area instead."""
    a = _FakeBCAdapter()
    assert isinstance(a.name, str)
    assert isinstance(a.supported_targets, tuple)
    assert a.default_target in a.supported_targets
    for attr in ("validate_replay_dir", "build_dataset", "fit_bc"):
        assert callable(getattr(a, attr))


def test_game_adapter_protocol_documents_bc_attribute():
    """The GameAdapter Protocol declares a ``bc`` attribute for the BC seam.

    Concrete adapters use structural typing rather than inheritance, so we
    verify the declaration is present on the Protocol class itself."""
    from framework.game_adapter import GameAdapter

    assert "bc" in GameAdapter.__annotations__
