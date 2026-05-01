"""Tests for grid_search.py — grid expansion and name generation."""
from __future__ import annotations

from grid_search import _expand_grid, _make_experiment_name, _fmt_value, _build_policy_params, _POLICY_PARAM_MAP


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
