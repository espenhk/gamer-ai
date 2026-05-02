"""Tests for the game adapter protocol and registry.

Pure-logic tests — no heavyweight env imports, no live game sessions.
Verifies that each registered adapter:
  - can be instantiated via make_adapter()
  - returns sensible values for experiment_dir, track_label
  - returns ProbeSpec/WarmupSpec for TMNF, None for others
"""
from __future__ import annotations

import pytest
from framework.game_adapter import GAME_ADAPTERS


class TestGameAdapterRegistry:
    """The GAME_ADAPTERS dict has entries for all supported games."""

    def test_all_games_registered(self):
        assert set(GAME_ADAPTERS.keys()) == {"tmnf", "torcs", "sc2", "beamng", "car_racing"}

    @pytest.mark.parametrize("game", list(GAME_ADAPTERS.keys()))
    def test_adapter_can_be_instantiated(self, game):
        adapter = GAME_ADAPTERS[game]()
        assert adapter.name == game
        assert adapter.config_dir.startswith("games/")


class TestTMNFAdapter:
    def _adapter(self):
        return GAME_ADAPTERS["tmnf"]()

    def test_experiment_dir_includes_track(self):
        a = self._adapter()
        d = a.experiment_dir("myrun", {"track": "a03_centerline"}, None)
        assert "a03_centerline" in d
        assert "myrun" in d

    def test_track_override(self):
        a = self._adapter()
        d = a.experiment_dir("myrun", {"track": "a03_centerline"}, "custom_track")
        assert "custom_track" in d
        assert "a03_centerline" not in d

    def test_track_label_default(self):
        a = self._adapter()
        assert a.track_label({"track": "a03"}, None) == "a03"

    def test_track_label_override(self):
        a = self._adapter()
        assert a.track_label({"track": "a03"}, "b04") == "b04"

    def test_build_probe_returns_spec(self):
        a = self._adapter()
        probe = a.build_probe({"probe_s": 15.0, "cold_restarts": 20, "cold_sims": 5})
        assert probe is not None
        assert probe.probe_in_game_s == 15.0
        assert probe.cold_start_restarts == 20
        assert probe.cold_start_sims == 5

    def test_build_warmup_returns_spec(self):
        a = self._adapter()
        warmup = a.build_warmup({})
        assert warmup is not None
        assert warmup.steps == 5

    def test_build_extras_returns_policy_extras(self):
        import os
        import tempfile

        a = self._adapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            wf = os.path.join(tmpdir, "policy_weights.yaml")
            extras = a.build_extras(wf, {"n_lidar_rays": 0}, False)
            assert extras is not None
            assert "neural_dqn" in extras.factories
            assert "cmaes" in extras.factories
            assert "reinforce" in extras.factories
            assert "lstm" in extras.factories
            assert extras.loop_dispatch["neural_dqn"] == "q_learning"
            assert extras.loop_dispatch["cmaes"] == "cmaes"

    def test_decorate_reward_cfg_adds_track_keys(self):
        a = self._adapter()
        cfg = {}
        a.decorate_reward_cfg(cfg, {"track": "a03_centerline"}, None)
        assert cfg["track_name"] == "a03_centerline"
        assert "centerline_path" in cfg

    def test_experiment_dir_root(self):
        a = self._adapter()
        root = a.experiment_dir_root({"track": "a03_centerline"}, None)
        assert "a03_centerline" in root


class TestTorcsAdapter:
    def _adapter(self):
        return GAME_ADAPTERS["torcs"]()

    def test_experiment_dir(self):
        a = self._adapter()
        d = a.experiment_dir("myrun", {}, None)
        assert "torcs" in d
        assert "myrun" in d

    def test_track_label_default(self):
        a = self._adapter()
        assert a.track_label({}, None) == "torcs"

    def test_track_label_override(self):
        a = self._adapter()
        assert a.track_label({}, "aalborg") == "aalborg"

    def test_build_probe_returns_none(self):
        a = self._adapter()
        assert a.build_probe({}) is None

    def test_build_warmup_returns_none(self):
        a = self._adapter()
        assert a.build_warmup({}) is None

    def test_build_extras_returns_none(self):
        a = self._adapter()
        assert a.build_extras("/fake/weights.yaml", {}, False) is None


class TestSC2Adapter:
    def _adapter(self):
        return GAME_ADAPTERS["sc2"]()

    def test_experiment_dir_includes_map_name(self):
        a = self._adapter()
        d = a.experiment_dir("myrun", {"map_name": "MoveToBeacon"}, None)
        assert "sc2_MoveToBeacon" in d
        assert "myrun" in d

    def test_track_override_overrides_map(self):
        a = self._adapter()
        d = a.experiment_dir("myrun", {"map_name": "MoveToBeacon"}, "CollectMineralShards")
        assert "sc2_CollectMineralShards" in d
        assert "MoveToBeacon" not in d

    def test_track_label(self):
        a = self._adapter()
        assert a.track_label({"map_name": "MoveToBeacon"}, None) == "sc2_MoveToBeacon"

    def test_build_probe_returns_none(self):
        a = self._adapter()
        assert a.build_probe({}) is None

    def test_build_warmup_returns_none(self):
        a = self._adapter()
        assert a.build_warmup({}) is None


class TestBeamNGAdapter:
    def _adapter(self):
        return GAME_ADAPTERS["beamng"]()

    def test_experiment_dir(self):
        a = self._adapter()
        d = a.experiment_dir("myrun", {}, None)
        assert "beamng" in d

    def test_build_probe_returns_none(self):
        a = self._adapter()
        assert a.build_probe({}) is None


class TestCarRacingAdapter:
    def _adapter(self):
        return GAME_ADAPTERS["car_racing"]()

    def test_experiment_dir(self):
        a = self._adapter()
        d = a.experiment_dir("myrun", {}, None)
        assert "car_racing" in d

    def test_build_probe_returns_none(self):
        a = self._adapter()
        assert a.build_probe({}) is None
