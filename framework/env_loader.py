"""Minimal .env file loader — no third-party dependency required.

Reads KEY=VALUE pairs from a .env file in the repo root and injects any
missing keys into os.environ so that game clients (e.g. SC2PATH for pysc2)
pick them up before their first import.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(dotenv_path: str | Path | None = None) -> None:
    """Load a .env file into os.environ (existing vars are not overwritten)."""
    if dotenv_path is None:
        dotenv_path = Path(__file__).parent.parent / ".env"
    path = Path(dotenv_path)
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip optional surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
