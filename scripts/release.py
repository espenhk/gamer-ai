#!/usr/bin/env python3
"""Cut a release of `gamer-ai`.

Usage:
    python scripts/release.py <new_version> [--allow-dirty] [--no-tag]

The script:
  1. Validates the new version looks like SemVer (`X.Y.Z`).
  2. Validates the working tree is clean (unless --allow-dirty).
  3. Bumps `version` in `pyproject.toml`.
  4. Bumps `PACKAGE_VERSION` in `framework/version.py`.
  5. Rewrites `CHANGELOG.md` — promotes the contents of `## [Unreleased]`
     into a dated `## [<new_version>] - YYYY-MM-DD` section and leaves a
     fresh empty `## [Unreleased]` at the top.
  6. Commits with message `Release v<new_version>`.
  7. Creates an annotated git tag `v<new_version>` (unless --no-tag).

After this runs:
    git push origin main --tags

You can then refer to `v<new_version>` in issues, experiment notes, and
when investigating an `experiment_data.json` whose `code_version` field
starts with the same prefix.
"""
from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
VERSION_PY = REPO_ROOT / "framework" / "version.py"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _run(*args: str) -> str:
    out = subprocess.run(
        list(args), cwd=str(REPO_ROOT),
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        sys.exit(f"command failed: {' '.join(args)}\n{out.stderr}")
    return out.stdout.strip()


def _working_tree_clean() -> bool:
    return _run("git", "status", "--porcelain", "--untracked-files=no") == ""


def _bump_pyproject(new_version: str) -> None:
    text = PYPROJECT.read_text()
    new_text, n = re.subn(
        r'^version\s*=\s*"[^"]+"',
        f'version = "{new_version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        sys.exit("could not find a `version = \"...\"` line in pyproject.toml")
    PYPROJECT.write_text(new_text)


def _bump_version_py(new_version: str) -> None:
    text = VERSION_PY.read_text()
    new_text, n = re.subn(
        r'^PACKAGE_VERSION:\s*str\s*=\s*"[^"]+"',
        f'PACKAGE_VERSION: str = "{new_version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        sys.exit("could not find PACKAGE_VERSION in framework/version.py")
    VERSION_PY.write_text(new_text)


def _roll_changelog(new_version: str, date: str) -> None:
    text = CHANGELOG.read_text()

    # Anchor on the H2 heading itself (start-of-line), not the bare
    # string, so prose mentions like `## [Unreleased]` inside backticks
    # don't trip the matcher.
    unreleased_h2 = re.search(r"^## \[Unreleased\]\s*$", text, flags=re.MULTILINE)
    if unreleased_h2 is None:
        sys.exit("CHANGELOG.md is missing the '## [Unreleased]' heading")

    head = text[: unreleased_h2.end()]
    tail = text[unreleased_h2.end() :]

    next_h2 = re.search(r"^## \[", tail, flags=re.MULTILINE)
    if next_h2:
        unreleased_body = tail[: next_h2.start()]
        rest = tail[next_h2.start() :]
    else:
        unreleased_body = tail
        rest = ""

    # Strip leading/trailing blank lines and `---` rules around the body
    # so the move is clean.
    body_stripped = unreleased_body.strip()
    body_stripped = re.sub(r"^-{3,}\s*", "", body_stripped).strip()
    body_stripped = re.sub(r"\s*-{3,}$", "", body_stripped).strip()

    new_section = (
        "\n\n---\n\n"
        f"## [{new_version}] - {date}\n\n"
    )
    if body_stripped:
        new_section += body_stripped + "\n\n"
    new_section += "---\n\n"

    CHANGELOG.write_text(head + new_section + rest.lstrip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("new_version", help='New version, e.g. "0.2.0"')
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="Don't bail if the working tree has uncommitted changes",
    )
    parser.add_argument(
        "--no-tag", action="store_true",
        help="Skip creating the annotated git tag (only commit)",
    )
    args = parser.parse_args()

    if not _SEMVER_RE.match(args.new_version):
        sys.exit(f"version must look like X.Y.Z, got {args.new_version!r}")

    if not args.allow_dirty and not _working_tree_clean():
        sys.exit("working tree is dirty; commit/stash first or pass --allow-dirty")

    tag = f"v{args.new_version}"
    existing_tags = _run("git", "tag", "--list", tag)
    if existing_tags:
        sys.exit(f"git tag {tag} already exists")

    today = datetime.date.today().isoformat()
    _bump_pyproject(args.new_version)
    _bump_version_py(args.new_version)
    _roll_changelog(args.new_version, today)

    _run("git", "add", "pyproject.toml", "framework/version.py", "CHANGELOG.md")
    _run("git", "commit", "-m", f"Release {tag}")

    if not args.no_tag:
        _run("git", "tag", "-a", tag, "-m", f"Release {tag}")

    print(f"Release commit created for {tag}.")
    if not args.no_tag:
        print(f"Tagged {tag}.")
    print("Next: git push origin <branch> --tags")


if __name__ == "__main__":
    main()
