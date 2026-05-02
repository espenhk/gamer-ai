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

    def test_no_legacy_flat_params(self):
        """Legacy flat parameter list has been removed."""
        sig = inspect.signature(train_rl)
        param_names = set(sig.parameters.keys())
        expected = {"game", "config", "probe", "warmup", "extras",
                    "no_interrupt", "re_initialize"}
        assert param_names == expected
