"""Tests that the games/_template/ skeleton is valid, importable Python.

Guards against drift between the template and the real framework APIs:
if a refactor renames a base class or protocol method, these tests will
catch it before a contributor copies a broken template.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "games" / "_template"

# All Python files in the template directory.
_PY_FILES = sorted(_TEMPLATE_DIR.glob("*.py"))


class TestTemplateParseable:
    """Every .py file in games/_template/ must be valid Python syntax."""

    @pytest.mark.parametrize("py_file", _PY_FILES, ids=lambda p: p.name)
    def test_syntax(self, py_file: Path) -> None:
        source = py_file.read_text(encoding="utf-8")
        # ast.parse raises SyntaxError on invalid Python.
        ast.parse(source, filename=str(py_file))


class TestTemplateImportable:
    """Key template modules must be importable without error."""

    def test_import_adapter(self) -> None:
        mod = importlib.import_module("games._template.adapter")
        assert hasattr(mod, "make_adapter")
        assert hasattr(mod, "_TemplateAdapter")

    def test_import_obs_spec(self) -> None:
        mod = importlib.import_module("games._template.obs_spec")
        assert hasattr(mod, "TEMPLATE_OBS_SPEC")
        assert hasattr(mod, "BASE_OBS_DIM")
        assert mod.BASE_OBS_DIM >= 1

    def test_import_actions(self) -> None:
        mod = importlib.import_module("games._template.actions")
        assert hasattr(mod, "DISCRETE_ACTIONS")

    def test_import_reward(self) -> None:
        mod = importlib.import_module("games._template.reward")
        assert hasattr(mod, "RewardCalculator")
        assert hasattr(mod, "RewardConfig")

    def test_import_analytics(self) -> None:
        mod = importlib.import_module("games._template.analytics")
        assert callable(getattr(mod, "save_experiment_results", None))

    def test_import_env(self) -> None:
        mod = importlib.import_module("games._template.env")
        assert hasattr(mod, "_TemplateEnv")
        assert hasattr(mod, "make_env")


class TestTemplateAdapterProtocol:
    """The template adapter exposes every required GameAdapter method."""

    def test_adapter_has_required_attributes(self) -> None:
        from games._template.adapter import make_adapter

        adapter = make_adapter()
        assert hasattr(adapter, "name")
        assert hasattr(adapter, "config_dir")

    def test_adapter_has_required_methods(self) -> None:
        from framework.game_adapter import GameAdapter
        from games._template.adapter import make_adapter

        adapter = make_adapter()
        required_methods = tuple(
            name for name, value in GameAdapter.__dict__.items() if callable(value) and not name.startswith("_")
        )
        for method_name in required_methods:
            assert callable(getattr(adapter, method_name, None)), f"Adapter missing method: {method_name}"

    def test_build_probe_returns_none(self) -> None:
        from games._template.adapter import make_adapter

        adapter = make_adapter()
        assert adapter.build_probe({}) is None

    def test_build_warmup_returns_none(self) -> None:
        from games._template.adapter import make_adapter

        adapter = make_adapter()
        assert adapter.build_warmup({}) is None


class TestTemplateNotRegistered:
    """The _template must NOT appear in the live game registry or CLI."""

    def test_not_in_game_adapters(self) -> None:
        from framework.game_adapter import GAME_ADAPTERS

        assert "_template" not in GAME_ADAPTERS

    def test_not_in_main_parser(self) -> None:
        import main

        parser = main._build_arg_parser()
        game_action = None
        for action in parser._actions:
            if hasattr(action, "dest") and action.dest == "game":
                game_action = action
                break
        assert game_action is not None
        assert "_template" not in game_action.choices


class TestTemplateConfigFiles:
    """Template YAML config files must exist and be loadable."""

    def test_training_params_exists(self) -> None:
        path = _TEMPLATE_DIR / "config" / "training_params.yaml"
        assert path.exists()

    def test_reward_config_exists(self) -> None:
        path = _TEMPLATE_DIR / "config" / "reward_config.yaml"
        assert path.exists()

    def test_training_params_loadable(self) -> None:
        import yaml

        path = _TEMPLATE_DIR / "config" / "training_params.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "game" in data

    def test_reward_config_loadable(self) -> None:
        import yaml

        path = _TEMPLATE_DIR / "config" / "reward_config.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
