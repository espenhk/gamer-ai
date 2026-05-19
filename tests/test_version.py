"""Unit tests for framework.version (code_version reporting)."""
from __future__ import annotations

import re

from framework import version
from framework.version import PACKAGE_VERSION, code_version, git_revision


def test_package_version_looks_like_semver():
    assert re.match(r"^\d+\.\d+\.\d+$", PACKAGE_VERSION), PACKAGE_VERSION


def test_code_version_starts_with_package_version():
    code_version.cache_clear()
    s = code_version()
    assert s.startswith(PACKAGE_VERSION), s


def test_code_version_shape_when_in_git_repo():
    """In a git repo, code_version() should be `<version>+g<sha7>[.dirty]`."""
    code_version.cache_clear()
    sha, _ = git_revision()
    s = code_version()
    if sha is None:
        # Not a git repo (or `git` missing): bare version, nothing appended.
        assert s == PACKAGE_VERSION
    else:
        assert re.match(
            rf"^{re.escape(PACKAGE_VERSION)}\+g[0-9a-f]{{7}}(\.dirty)?$", s
        ), s


def test_code_version_is_cached():
    """Repeated calls return the identical string (functools.lru_cache)."""
    code_version.cache_clear()
    a = code_version()
    b = code_version()
    assert a is b  # lru_cache guarantees identity, not just equality


def test_git_revision_handles_missing_git(monkeypatch):
    """If `git` isn't available, both fields default safely."""
    def _raise(*args, **kwargs):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(version.subprocess, "run", _raise)
    code_version.cache_clear()
    sha, dirty = git_revision()
    assert sha is None
    assert dirty is False
    assert code_version() == PACKAGE_VERSION
    code_version.cache_clear()
