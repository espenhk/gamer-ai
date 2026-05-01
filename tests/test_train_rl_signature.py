"""Tests for the train_rl function signature.

Guards against future parameter drift by verifying that the new bundle-based
parameters exist and have the expected names.
"""
from __future__ import annotations

import inspect

from framework.training import train_rl


class TestTrainRLSignature:
    """Verify train_rl accepts the new bundle-based interface."""

    def test_accepts_game_and_config_params(self):
        sig = inspect.signature(train_rl)
        param_names = list(sig.parameters.keys())
        assert "game" in param_names
        assert "config" in param_names

    def test_accepts_optional_specs(self):
        sig = inspect.signature(train_rl)
        param_names = list(sig.parameters.keys())
        assert "probe" in param_names
        assert "warmup" in param_names
        assert "extras" in param_names

    def test_accepts_control_flags(self):
        sig = inspect.signature(train_rl)
        param_names = list(sig.parameters.keys())
        assert "no_interrupt" in param_names
        assert "re_initialize" in param_names

    def test_legacy_params_still_accepted(self):
        """Back-compat: the legacy flat parameter list should still work."""
        sig = inspect.signature(train_rl)
        param_names = list(sig.parameters.keys())
        assert "experiment_name" in param_names
        assert "make_env_fn" in param_names
        assert "obs_spec" in param_names
        assert "weights_file" in param_names
        assert "policy_type" in param_names
