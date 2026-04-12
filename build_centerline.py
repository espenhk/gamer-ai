"""Backward-compatibility shim. Moved to games.tmnf.tools.build_centerline.

Run with:
    python -m games.tmnf.tools.build_centerline path/to/replay.Replay.Gbx
"""
from games.tmnf.tools.build_centerline import (  # noqa: F401
    extract_positions,
    resample_centerline,
    update_registry,
    main,
)

if __name__ == "__main__":
    main()
