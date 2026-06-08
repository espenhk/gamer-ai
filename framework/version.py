"""Code-version reporting for `gamer-ai`.

`code_version()` returns a string identifying the exact state of the
codebase at runtime, so it can be persisted alongside experiment results
and used to reproduce / debug a run later.

Format follows PEP 440 local version identifiers:

    <package_version>+g<sha7>            # clean working tree
    <package_version>+g<sha7>.dirty      # uncommitted changes
    <package_version>                    # not in a git repo

`PACKAGE_VERSION` is the single in-tree source of truth for the
human-readable version number.  `scripts/release.py` keeps it in sync
with `pyproject.toml` on every release.
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

# Source of truth for the package version.  Bumped by scripts/release.py
# at release time; pyproject.toml is updated to the same value in the
# same commit.  Keep both in sync.
PACKAGE_VERSION: str = "0.6.7"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_git(*args: str) -> str | None:
    """Run `git <args>` in the repo root, return stripped stdout or None."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_revision() -> tuple[str | None, bool]:
    """Return ``(short_sha, dirty)`` or ``(None, False)`` if not in a git repo.

    A working tree with any unstaged or uncommitted change is considered
    dirty.  Untracked files do not flip the flag — they're noise from
    experiment dumps under `experiments/` more often than not.
    """
    sha = _run_git("rev-parse", "--short=7", "HEAD")
    if sha is None:
        return None, False
    status = _run_git("status", "--porcelain", "--untracked-files=no")
    dirty = bool(status)
    return sha, dirty


@functools.lru_cache(maxsize=1)
def code_version() -> str:
    """Return a single string identifying the running code version."""
    sha, dirty = git_revision()
    if sha is None:
        return PACKAGE_VERSION
    suffix = f"+g{sha}"
    if dirty:
        suffix += ".dirty"
    return f"{PACKAGE_VERSION}{suffix}"
