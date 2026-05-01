"""Regression test: analytics modules must be importable without matplotlib.

These tests verify that neither framework.analytics nor games.tmnf.analytics
raises ImportError when matplotlib is absent from sys.modules.
"""
from __future__ import annotations

import importlib
import sys
import types


def _import_without_matplotlib(module_name: str) -> None:
    """Import *module_name* after hiding matplotlib from sys.modules."""
    # Stash real modules
    stashed = {k: v for k, v in sys.modules.items()
               if k == "matplotlib" or k.startswith("matplotlib.")}

    # Block matplotlib by inserting a dummy that raises on attribute access
    blocker = types.ModuleType("matplotlib")
    blocker.__spec__ = None  # prevents importlib from treating it as a package

    def _raise(*a, **kw):
        raise ImportError("matplotlib blocked for test")

    blocker.__getattr__ = _raise  # type: ignore[assignment]

    # Remove any real matplotlib entries and install blocker
    for k in list(sys.modules):
        if k == "matplotlib" or k.startswith("matplotlib."):
            del sys.modules[k]
    sys.modules["matplotlib"] = blocker
    # Also block submodules that analytics files import directly
    for sub in ("matplotlib.pyplot", "matplotlib.cm", "matplotlib.patches",
                "matplotlib.axes", "matplotlib.figure", "matplotlib.colors"):
        sys.modules[sub] = blocker  # type: ignore[assignment]

    # Remove target module so it re-executes
    for k in list(sys.modules):
        if k == module_name or k.startswith(module_name + "."):
            del sys.modules[k]

    try:
        importlib.import_module(module_name)
    finally:
        # Restore original state
        for k in list(sys.modules):
            if k == "matplotlib" or k.startswith("matplotlib."):
                del sys.modules[k]
        sys.modules.update(stashed)
        # Re-import the target so subsequent tests see the real module
        for k in list(sys.modules):
            if k == module_name or k.startswith(module_name + "."):
                del sys.modules[k]


def test_framework_analytics_importable_without_matplotlib():
    """framework.analytics must not raise ImportError when matplotlib is absent."""
    _import_without_matplotlib("framework.analytics")


def test_tmnf_analytics_importable_without_matplotlib():
    """games.tmnf.analytics must not raise ImportError when matplotlib is absent."""
    _import_without_matplotlib("games.tmnf.analytics")
