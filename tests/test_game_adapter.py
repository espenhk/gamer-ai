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

    # ------------------------------------------------------------------
    # Fail-fast policy type validation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("bad_type,expected_hint", [
        ("hill_climbing", "sc2_genetic"),
        ("genetic",       "sc2_genetic"),
        ("neural_net",    "sc2_neural_dqn"),
    ])
    def test_build_extras_rejects_incompatible_policy_type(self, bad_type, expected_hint):
        """Incompatible policy types must raise ValueError naming the bad type and migration hint."""
        import os
        import tempfile

        a = self._adapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            wf = os.path.join(tmpdir, "policy_weights.yaml")
            with pytest.raises(ValueError) as exc_info:
                a.build_extras(wf, {"policy_type": bad_type}, False)
            msg = str(exc_info.value)
            assert bad_type in msg
            assert expected_hint in msg

    def test_build_extras_incompatible_type_error_contains_docs_reference(self):
        """ValueError for incompatible type includes a reference to the docs."""
        import os
        import tempfile

        a = self._adapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            wf = os.path.join(tmpdir, "policy_weights.yaml")
            with pytest.raises(ValueError, match="CLAUDE.md"):
                a.build_extras(wf, {"policy_type": "hill_climbing"}, False)

    # ------------------------------------------------------------------
    # Fail-fast policy_params validation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("policy_type,bad_param", [
        ("sc2_genetic",  "hidden_sizes"),
        ("cmaes",        "mutation_scale"),
        ("sc2_cmaes",    "learning_rate"),
        ("sc2_lstm",     "mutation_scale"),
        ("sc2_reinforce", "population_size"),
    ])
    def test_build_extras_rejects_invalid_policy_params(self, policy_type, bad_param):
        """Unknown policy_params key must raise ValueError naming the bad key."""
        import os
        import tempfile

        a = self._adapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            wf = os.path.join(tmpdir, "policy_weights.yaml")
            with pytest.raises(ValueError, match=bad_param):
                a.build_extras(wf, {
                    "policy_type": policy_type,
                    "policy_params": {bad_param: 0.1},
                }, False)

    def test_build_extras_accepts_valid_params_for_sc2_genetic(self):
        """Valid policy_params for sc2_genetic must not raise."""
        import os
        import tempfile

        a = self._adapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            wf = os.path.join(tmpdir, "policy_weights.yaml")
            extras = a.build_extras(wf, {
                "policy_type": "sc2_genetic",
                "policy_params": {"population_size": 10, "elite_k": 3},
            }, False)
            assert extras is not None
            assert "sc2_genetic" in extras.factories

    def test_build_extras_accepts_empty_policy_params(self):
        """Empty policy_params must not raise for any valid SC2 policy type."""
        import os
        import tempfile

        a = self._adapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            wf = os.path.join(tmpdir, "policy_weights.yaml")
            extras = a.build_extras(wf, {
                "policy_type": "sc2_genetic",
                "policy_params": {},
            }, False)
            assert extras is not None


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
