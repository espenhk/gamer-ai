"""Backward-compatibility shim. Moved to games.tmnf.tools.debug_straight.

Run with:
    python -m games.tmnf.tools.debug_straight
"""
from games.tmnf.tools.debug_straight import main  # noqa: F401

if __name__ == "__main__":
    main()
