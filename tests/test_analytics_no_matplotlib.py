"""Regression test: analytics modules must be importable without matplotlib.

These tests verify that neither framework.analytics nor games.tmnf.analytics
raises ImportError when matplotlib is absent from sys.modules.
"""
from __future__ import annotations

import importlib
import sys
import types


def _import_without_matplotlib(module_name: str) -> None:
    """Import *module_name* after hiding matplotlib from sys.modules.

    The function removes ALL newly imported modules from sys.modules after the
    test completes, including freshly-imported dependency modules (e.g.
    ``framework.analytics`` imported as a side-effect of testing
    ``games.tmnf.analytics``) that were stamped with ``_HAS_MPL = False`` during
    the matplotlib-blocked import.  Without this cleanup those contaminated
    modules persist in sys.modules and cause subsequent tests that rely on
    matplotlib to silently skip plot generation.

    Only modules that were NOT already present in sys.modules before the test
    (i.e. newly imported as side-effects) are removed; pre-existing C extension
    modules such as numpy are never touched.
    """
    # Snapshot the set of module keys BEFORE any manipulation so we can tell
    # which modules were freshly imported as side effects of the test.
    pre_existing = set(sys.modules.keys())

    # Stash real matplotlib modules
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
        # Remove every module that was freshly imported during this test
        # (not present before the blocker was installed, excluding matplotlib
        # entries handled separately below).  This prevents dependency modules
        # like framework.analytics — imported with _HAS_MPL=False as a side
        # effect — from contaminating subsequent tests.
        for k in list(sys.modules):
            if (k not in pre_existing
                    and k != "matplotlib"
                    and not k.startswith("matplotlib.")):
                del sys.modules[k]
        # Also explicitly remove the target module and any submodules (handles
        # the case where the target was already in pre_existing and was removed
        # then re-imported with _HAS_MPL=False during the test).
        for k in list(sys.modules):
            if k == module_name or k.startswith(module_name + "."):
                del sys.modules[k]
        # Restore original matplotlib state
        for k in list(sys.modules):
            if k == "matplotlib" or k.startswith("matplotlib."):
                del sys.modules[k]
        sys.modules.update(stashed)


def test_framework_analytics_importable_without_matplotlib():
    """framework.analytics must not raise ImportError when matplotlib is absent."""
    _import_without_matplotlib("framework.analytics")


def test_tmnf_analytics_importable_without_matplotlib():
    """games.tmnf.analytics must not raise ImportError when matplotlib is absent."""
    _import_without_matplotlib("games.tmnf.analytics")


def test_sc2_analytics_importable_without_matplotlib():
    """games.sc2.analytics must not raise ImportError when matplotlib is absent."""
    _import_without_matplotlib("games.sc2.analytics")
