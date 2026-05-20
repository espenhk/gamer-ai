"""Tests for grid_search.py — grid expansion and name generation."""
from __future__ import annotations

import glob
import os
import subprocess
import tempfile

import pytest
import yaml

from grid_search import (
    _ABBREV,
    _expand_grid,
    _make_experiment_name,
    _fmt_value,
    _build_policy_params,
    _POLICY_PARAM_MAP,
    _load_grid_config,
    _launch_local_workers,
    _parse_non_negative_int,
    _stop_local_workers,
)


class TestExpandGrid:
    def test_no_variation_returns_single_combo(self):
        t = {"speed": 10.0, "n_sims": 50, "mutation_scale": 0.05}
        r = {"progress_weight": 10000.0}
        combos, varied_keys = _expand_grid(t, r)
        assert len(combos) == 1
        assert varied_keys == []
        assert combos[0]["training_params"]["speed"] == 10.0

    def test_single_training_axis(self):
        t = {"mutation_scale": [0.05, 0.1, 0.2], "n_sims": 50}
        r = {"progress_weight": 10000.0}
        combos, varied_keys = _expand_grid(t, r)
        assert len(combos) == 3
        assert varied_keys == ["mutation_scale"]
        scales = [c["training_params"]["mutation_scale"] for c in combos]
        assert scales == [0.05, 0.1, 0.2]

    def test_single_reward_axis(self):
        t = {"n_sims": 50}
        r = {"centerline_weight": [-0.1, -0.5]}
        combos, varied_keys = _expand_grid(t, r)
        assert len(combos) == 2
        assert varied_keys == ["centerline_weight"]

    def test_cartesian_product(self):
        t = {"mutation_scale": [0.05, 0.1]}
        r = {"centerline_weight": [-0.1, -0.5, -1.0]}
        combos, varied_keys = _expand_grid(t, r)
        assert len(combos) == 6   # 2 × 3
        assert set(varied_keys) == {"mutation_scale", "centerline_weight"}

    def test_fixed_params_preserved_across_combos(self):
        t = {"speed": 10.0, "mutation_scale": [0.05, 0.1]}
        r = {"progress_weight": 9999.0}
        combos, _ = _expand_grid(t, r)
        for c in combos:
            assert c["training_params"]["speed"] == 10.0
            assert c["reward_params"]["progress_weight"] == 9999.0

    def test_flat_dict_contains_varied_values(self):
        t = {"mutation_scale": [0.05, 0.2]}
        r = {}
        combos, _ = _expand_grid(t, r)
        assert combos[0]["_flat"]["mutation_scale"] == 0.05
        assert combos[1]["_flat"]["mutation_scale"] == 0.2

    def test_no_variation_has_no_flat_key(self):
        t = {"n_sims": 50}
        r = {}
        combos, _ = _expand_grid(t, r)
        # Single-combo result has no _flat since it's not added
        assert "_flat" not in combos[0]


class TestMakeExperimentName:
    def test_no_varied_keys(self):
        name = _make_experiment_name("gs_v1", {}, [])
        assert name == "gs_v1"

    def test_single_varied_key(self):
        name = _make_experiment_name("gs_v1", {"mutation_scale": 0.05}, ["mutation_scale"])
        assert name == "gs_v1__ms0.05"

    def test_negative_float_uses_n_prefix(self):
        name = _make_experiment_name("gs", {"centerline_weight": -0.1}, ["centerline_weight"])
        assert "n0.1" in name

    def test_multiple_varied_keys(self):
        flat = {"mutation_scale": 0.1, "centerline_weight": -0.5}
        name = _make_experiment_name("gs", flat, ["mutation_scale", "centerline_weight"])
        assert name == "gs__ms0.1__cwn0.5"

    def test_unknown_key_uses_key_itself(self):
        name = _make_experiment_name("gs", {"my_custom_param": 42}, ["my_custom_param"])
        assert "my_custom_param42" in name


class TestBuildPolicyParams:
    def test_nested_policy_params_passed_through(self):
        t = {"policy_params": {"hidden_sizes": [64, 64], "gamma": 0.99}}
        pp = _build_policy_params(t)
        assert pp["hidden_sizes"] == [64, 64]
        assert pp["gamma"] == 0.99

    def test_top_level_key_promoted_to_policy_params(self):
        t = {"learning_rate": 0.001, "batch_size": 32}
        pp = _build_policy_params(t)
        assert pp["learning_rate"] == 0.001
        assert pp["batch_size"] == 32

    def test_top_level_overrides_nested(self):
        t = {"learning_rate": 0.005, "policy_params": {"learning_rate": 0.001}}
        pp = _build_policy_params(t)
        assert pp["learning_rate"] == 0.005

    def test_all_new_policy_param_keys_are_mapped(self):
        new_keys = {
            "hidden_sizes", "hidden_size", "learning_rate", "entropy_coeff",
            "baseline", "batch_size", "target_update_freq", "epsilon_decay_steps",
        }
        assert new_keys.issubset(_POLICY_PARAM_MAP.keys())

    def test_promoted_keys_have_correct_target_names(self):
        identity_mapped = {
            "hidden_sizes", "hidden_size", "learning_rate", "entropy_coeff",
            "baseline", "batch_size", "target_update_freq", "epsilon_decay_steps",
        }
        for key in identity_mapped:
            assert _POLICY_PARAM_MAP[key] == key, (
                f"_POLICY_PARAM_MAP[{key!r}] = {_POLICY_PARAM_MAP[key]!r}, expected {key!r}"
            )

    def test_no_policy_params_key_returns_empty(self):
        t = {"speed": 10.0, "n_sims": 50}
        pp = _build_policy_params(t)
        assert pp == {}

    def test_lstm_hidden_size_promoted(self):
        t = {"hidden_size": 64}
        pp = _build_policy_params(t)
        assert pp["hidden_size"] == 64

    def test_reinforce_baseline_promoted(self):
        t = {"baseline": "none", "entropy_coeff": 0.05}
        pp = _build_policy_params(t)
        assert pp["baseline"] == "none"
        assert pp["entropy_coeff"] == 0.05

    def test_genetic_mutation_scale_promoted(self):
        """mutation_scale top-level key should be forwarded into policy_params."""
        t = {"mutation_scale": 0.2}
        pp = _build_policy_params(t)
        assert pp["mutation_scale"] == 0.2

    def test_genetic_mutation_share_promoted(self):
        """mutation_share top-level key should be forwarded into policy_params."""
        t = {"mutation_share": 0.5}
        pp = _build_policy_params(t)
        assert pp["mutation_share"] == 0.5

    def test_genetic_mutation_params_can_be_grid_axes(self):
        """Ensure mutation_scale and mutation_share are in _POLICY_PARAM_MAP."""
        assert "mutation_scale" in _POLICY_PARAM_MAP
        assert "mutation_share" in _POLICY_PARAM_MAP
        assert _POLICY_PARAM_MAP["mutation_scale"] == "mutation_scale"
        assert _POLICY_PARAM_MAP["mutation_share"] == "mutation_share"


class TestFmtValue:
    def test_integer(self):
        assert _fmt_value(50) == "50"

    def test_float_strips_trailing_zeros(self):
        assert _fmt_value(10.0) == "10"
        assert _fmt_value(0.050) == "0.05"

    def test_negative_float(self):
        assert _fmt_value(-0.1) == "n0.1"

    def test_string(self):
        assert _fmt_value("hill_climbing") == "hill_climbing"


class TestLoadGridConfig:
    def test_default_game_is_tmnf(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"base_name": "test", "training_params": {"n_sims": 10}}, f)
            f.flush()
            base_name, game, track, t, r, d = _load_grid_config(f.name)
        os.unlink(f.name)
        assert game == "tmnf"

    def test_game_field_honoured(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"base_name": "test", "game": "torcs",
                        "training_params": {"n_sims": 10}}, f)
            f.flush()
            base_name, game, track, t, r, d = _load_grid_config(f.name)
        os.unlink(f.name)
        assert game == "torcs"

    def test_track_field_honoured(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"base_name": "test", "track": "aalborg",
                        "training_params": {"n_sims": 10}}, f)
            f.flush()
            base_name, game, track, t, r, d = _load_grid_config(f.name)
        os.unlink(f.name)
        assert track == "aalborg"

    def test_track_default_is_none(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"base_name": "test", "training_params": {"n_sims": 10}}, f)
            f.flush()
            base_name, game, track, t, r, d = _load_grid_config(f.name)
        os.unlink(f.name)
        assert track is None

    def test_grid_expansion_unaffected_by_game_field(self):
        """Adding game: field should not affect grid expansion."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "base_name": "test",
                "game": "torcs",
                "training_params": {"mutation_scale": [0.05, 0.1]},
                "reward_params": {}
            }, f)
            f.flush()
            _, _, _, t, r, _ = _load_grid_config(f.name)
        os.unlink(f.name)
        combos, varied = _expand_grid(t, r)
        assert len(combos) == 2
        assert varied == ["mutation_scale"]


class TestAbbreviationCoverage:
    def test_all_default_training_and_reward_keys_have_abbreviations(self):
        config_paths = glob.glob("games/**/config/training_params.yaml", recursive=True)
        config_paths += glob.glob("games/**/config/reward_config.yaml", recursive=True)

        keys = set()
        for path in config_paths:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            keys.update(data.keys())

        missing = sorted(k for k in keys if k not in _ABBREV)
        assert missing == []

    def test_all_promoted_policy_params_have_abbreviations(self):
        missing = sorted(k for k in _POLICY_PARAM_MAP if k not in _ABBREV)
        assert missing == []


class TestLocalDistributedWorkers:
    def test_launch_local_workers_uses_configured_coordinator_host(self, monkeypatch):
        calls = []

        class _DummyProc:
            def __init__(self, cmd):
                self.cmd = cmd

        def _fake_popen(cmd):
            calls.append(cmd)
            return _DummyProc(cmd)

        monkeypatch.setattr("grid_search.subprocess.Popen", _fake_popen)
        _launch_local_workers(
            coordinator_host="192.168.1.50",
            coordinator_port=6000,
            token="tok",
            game_name="tmnf",
            local_workers=1,
            no_interrupt=False,
            re_initialize=False,
            start_stagger_s=0,
        )
        assert len(calls) == 1
        assert "http://192.168.1.50:6000" in calls[0]

    def test_launch_local_workers_spawns_expected_commands(self, monkeypatch):
        calls = []

        class _DummyProc:
            def __init__(self, cmd):
                self.cmd = cmd

        def _fake_popen(cmd):
            calls.append(cmd)
            return _DummyProc(cmd)

        monkeypatch.setattr("grid_search.subprocess.Popen", _fake_popen)

        procs = _launch_local_workers(
            coordinator_host="127.0.0.1",
            coordinator_port=5555,
            token="tok",
            game_name="sc2",
            local_workers=2,
            no_interrupt=True,
            re_initialize=True,
            start_stagger_s=0,
        )

        assert len(procs) == 2
        assert len(calls) == 2
        for i, cmd in enumerate(calls, 1):
            assert cmd[1:4] == ["-m", "distributed.worker", "--coordinator"]
            assert "http://127.0.0.1:5555" in cmd
            assert "--token" in cmd
            assert "--game" in cmd
            assert "--no-interrupt" in cmd
            assert "--re-initialize" in cmd
            assert f"local-{i}" in cmd

    def test_launch_local_workers_cascading_stagger_between_spawns(
        self, monkeypatch
    ):
        """Issue #254: each worker after the first waits ``start_stagger_s``
        seconds before launching, so SC2 binaries don't race on the same
        ``.SC2Map`` file at boot."""
        spawn_order: list[tuple[str, object]] = []

        class _DummyProc:
            def __init__(self, cmd):
                self.cmd = cmd

        def _fake_popen(cmd):
            spawn_order.append(("popen", cmd[cmd.index("--worker-id") + 1]))
            return _DummyProc(cmd)

        sleep_calls: list[float] = []

        def _fake_sleep(s):
            spawn_order.append(("sleep", s))
            sleep_calls.append(s)

        monkeypatch.setattr("grid_search.subprocess.Popen", _fake_popen)
        monkeypatch.setattr("grid_search.sleep", _fake_sleep)

        procs = _launch_local_workers(
            coordinator_host="127.0.0.1",
            coordinator_port=5555,
            token="tok",
            game_name="sc2",
            local_workers=3,
            no_interrupt=False,
            re_initialize=False,
            start_stagger_s=5.0,
        )

        assert len(procs) == 3
        # 3 spawns + 2 sleeps (no sleep after the last spawn)
        assert sleep_calls == [5.0, 5.0]
        assert spawn_order == [
            ("popen", "local-1"),
            ("sleep", 5.0),
            ("popen", "local-2"),
            ("sleep", 5.0),
            ("popen", "local-3"),
        ]

    def test_launch_local_workers_no_sleep_when_stagger_zero(self, monkeypatch):
        monkeypatch.setattr(
            "grid_search.subprocess.Popen", lambda cmd: type("P", (), {"cmd": cmd})()
        )

        sleep_calls: list[float] = []
        monkeypatch.setattr("grid_search.sleep", lambda s: sleep_calls.append(s))

        _launch_local_workers(
            coordinator_host="127.0.0.1",
            coordinator_port=5555,
            token="tok",
            game_name="sc2",
            local_workers=3,
            no_interrupt=False,
            re_initialize=False,
            start_stagger_s=0.0,
        )
        assert sleep_calls == []

    def test_launch_local_workers_cleans_up_started_procs_on_launch_error(
        self, monkeypatch
    ):
        started = []

        class _FakeProc:
            def __init__(self):
                self.terminated = False
                self.pid = 999

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        calls = {"n": 0}

        def _fake_popen(cmd):
            calls["n"] += 1
            if calls["n"] == 1:
                p = _FakeProc()
                started.append(p)
                return p
            raise OSError("boom")

        monkeypatch.setattr("grid_search.subprocess.Popen", _fake_popen)

        with pytest.raises(OSError):
            _launch_local_workers(
                coordinator_host="127.0.0.1",
                coordinator_port=5555,
                token="tok",
                game_name="sc2",
                local_workers=2,
                no_interrupt=False,
                re_initialize=False,
                start_stagger_s=0,
            )
        assert len(started) == 1
        assert started[0].terminated is True

    def test_stop_local_workers_terminates_and_kills_on_timeout(self):
        class _Proc:
            def __init__(self, already_done=False, timeout=False):
                self._already_done = already_done
                self._timeout = timeout
                self.terminated = False
                self.killed = False
                self.wait_calls = 0
                self.pid = 123

            def poll(self):
                return 0 if self._already_done else None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.wait_calls += 1
                if self._timeout and self.wait_calls == 1:
                    raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
                return 0

            def kill(self):
                self.killed = True

        done = _Proc(already_done=True)
        graceful = _Proc()
        needs_kill = _Proc(timeout=True)
        _stop_local_workers([done, graceful, needs_kill])

        assert done.terminated is False
        assert graceful.terminated is True
        assert graceful.killed is False
        assert needs_kill.terminated is True
        assert needs_kill.killed is True


class TestLocalWorkerParsing:
    def test_parse_non_negative_int_accepts_numeric_values(self):
        assert _parse_non_negative_int(0, "x") == 0
        assert _parse_non_negative_int("3", "x") == 3

    def test_parse_non_negative_int_rejects_negative(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            _parse_non_negative_int(-1, "x")

    def test_parse_non_negative_int_rejects_non_integer(self):
        with pytest.raises(ValueError, match="must be an integer >= 0"):
            _parse_non_negative_int("abc", "x")


class TestGridSearchCliFlags:
    def test_help_includes_live_gui_flag(self):
        proc = subprocess.run(
            ["python", "grid_search.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0
        assert "--live-gui" in proc.stdout
