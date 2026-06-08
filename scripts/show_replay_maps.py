#!/usr/bin/env python3
"""List unique maps referenced by .SC2Replay files without starting SC2.

Usage:
    pip install mpyq
    python scripts/show_replay_maps.py "C:/Program Files (x86)/StarCraft II/ReplaysMiniPack"
"""

from __future__ import annotations

import pathlib
import sys


def _strings_from_s2blob(data: bytes) -> list[str]:
    """Extract UTF-8 strings stored as s2-protocol blob fields (tag byte 0x02)."""
    out: list[str] = []
    i = 0
    while i < len(data) - 2:
        if data[i] == 0x02:
            n, shift, j = 0, 0, i + 1
            while j < len(data):
                b = data[j]
                j += 1
                n |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
            if 0 < n <= 512 and j + n <= len(data):
                try:
                    s = data[j : j + n].decode("utf-8")
                    out.append(s)
                    i = j + n
                    continue
                except UnicodeDecodeError:
                    pass
        i += 1
    return out


def map_title_from_replay(path: pathlib.Path) -> str:
    import mpyq  # type: ignore[import-untyped]

    archive = mpyq.MPQArchive(str(path))
    raw = archive.read_file("replay.details")
    if raw is None:
        raise ValueError("replay.details not found in archive")
    strings = _strings_from_s2blob(raw)
    # Map names are printable ASCII and usually contain a space or a
    # well-known SC2 ladder suffix (LE / TE / RE / CE).
    for s in strings:
        if (
            len(s) >= 5
            and all(0x20 <= ord(c) <= 0x7E for c in s)
            and any(kw in s for kw in (" ", "LE", "TE", "RE", "CE", "GSL"))
        ):
            return s
    return strings[0] if strings else "unknown"


def main() -> None:
    try:
        import mpyq  # noqa: F401
    except ImportError:
        sys.exit("mpyq not installed — run: pip install mpyq")

    folder = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    replays = sorted(folder.glob("*.SC2Replay"))
    if not replays:
        sys.exit(f"No .SC2Replay files found in {folder}")

    map_counts: dict[str, int] = {}
    errors: list[str] = []

    for path in replays:
        try:
            title = map_title_from_replay(path)
            map_counts[title] = map_counts.get(title, 0) + 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name}: {exc}")

    print(f"{len(replays)} replay(s) → {len(map_counts)} unique map(s):\n")
    for title, n in sorted(map_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d}×  {title}")

    if errors:
        print(f"\n{len(errors)} unreadable replay(s):")
        for e in errors:
            print(f"  {e}")

    print(
        "\nTo populate the Battle.net cache: open SC2 through Battle.net and"
        "\nplay (or load) each map once — the client will download the .s2ma"
        "\nfiles automatically."
    )


if __name__ == "__main__":
    main()
