"""Top-level RL training entry point.

Dispatches to a per-game entry module based on --game. Game-specific
wiring (env factory, obs_spec, policy factories, analytics) lives in
games/<name>/entry.py so this file stays game-agnostic.

Defaults to --game tmnf to preserve existing CLI invocations.
"""

from __future__ import annotations

import argparse
import importlib
import logging


# Per-game entry modules. Each module exposes ``run(args)``.
GAMES: dict[str, str] = {
    "tmnf":    "games.tmnf.entry",
    "assetto": "games.assetto_corsa.entry",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="RL training (multi-game)")
    parser.add_argument(
        "experiment",
        help="Experiment name — files stored in experiments/<...>/<name>/",
    )
    parser.add_argument(
        "--game", default="tmnf", choices=sorted(GAMES.keys()),
        help="Game integration to train against (default: tmnf)",
    )
    parser.add_argument(
        "--no-interrupt", action="store_true",
        help="Skip all 'Press Enter' prompts and run all phases automatically",
    )
    parser.add_argument(
        "--re-initialize", action="store_true",
        help="Ignore any existing weights file and restart from scratch, "
             "including probe and cold-start phases.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    module = importlib.import_module(GAMES[args.game])
    module.run(args)


if __name__ == "__main__":
    main()
