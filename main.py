"""RL training entry point.

Thin glue layer: reads experiment config, wires game-specific objects into the
game-agnostic framework.training.train_rl(), then saves results.

Supports multiple games via the ``--game`` flag:
    python main.py <experiment>                       # default: tmnf
    python main.py <experiment> --game tmnf
    python main.py <experiment> --game beamng
    python main.py <experiment> --game assetto
    python main.py <experiment> --game car_racing
    python main.py <experiment> --game torcs
    python main.py <experiment> --game sc2

All algorithm logic lives in framework/training.py.
Game-specific logic lives in games/<name>/.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil

import yaml

from framework.game_adapter import GAME_ADAPTERS
from framework.run_config import RunConfig
from framework.training import train_rl

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="RL training (multi-game)")
    parser.add_argument(
        "experiment",
        help="Experiment name — files stored in experiments/<...>/<name>/",
    )
    parser.add_argument(
        "--game",
        default="tmnf",
        choices=["tmnf", "beamng", "assetto", "car_racing", "torcs", "sc2"],
        help=(
            "Select which simulator to use. "
            "Choices: tmnf (default), beamng, assetto, car_racing, torcs, sc2. "
            "beamng and assetto require optional simulator dependencies."
        ),
    )
    parser.add_argument(
        "--track",
        default=None,
        help="Override the track / map name from the config (e.g. aalborg, CollectMineralShards).",
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
        "--play", action="store_true",
        help=(
            "Human-vs-AI interactive play mode (SC2 only).  "
            "Loads the champion policy from the experiment and launches a "
            "two-player PySC2 session where you play via the SC2 UI against "
            "the trained agent.  No weight updates occur."
        ),
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

    if args.game == "assetto":
        _run_assetto(args)
        return

    if args.play:
        if args.game != "sc2":
            raise SystemExit("--play is only supported with --game sc2")
        _run_play_sc2(args)
        return

    adapter = GAME_ADAPTERS[args.game]()
    _run_one(adapter, args)


# ======================================================================
# Unified runner (all games except assetto)
# ======================================================================

def _run_one(adapter, args: argparse.Namespace) -> None:
    # Read master config to learn the track before the experiment dir exists.
    master_cfg = os.path.join(adapter.config_dir, "training_params.yaml")
    with open(master_cfg) as f:
        master_p = yaml.safe_load(f)

    experiment_dir = adapter.experiment_dir(args.experiment, master_p, args.track)
    weights_file = f"{experiment_dir}/policy_weights.yaml"
    trainer_state_file = f"{experiment_dir}/trainer_state.npz"
    reward_cfg_file = f"{experiment_dir}/reward_config.yaml"
    training_params_file = f"{experiment_dir}/training_params.yaml"

    os.makedirs(experiment_dir, exist_ok=True)
    if not os.path.exists(reward_cfg_file):
        shutil.copy(os.path.join(adapter.config_dir, "reward_config.yaml"), reward_cfg_file)
        logger.info("Copied master reward config → %s", reward_cfg_file)
    if not os.path.exists(training_params_file):
        shutil.copy(master_cfg, training_params_file)
        logger.info("Copied master training params → %s", training_params_file)

    with open(training_params_file) as f:
        p = yaml.safe_load(f)

    # Decorate reward config with game-specific keys (e.g. TMNF centerline_path).
    with open(reward_cfg_file) as f:
        reward_cfg = yaml.safe_load(f) or {}
    adapter.decorate_reward_cfg(reward_cfg, p, args.track)
    with open(reward_cfg_file, "w") as f:
        yaml.dump(reward_cfg, f, default_flow_style=False, sort_keys=False)

    re_initialize = args.re_initialize
    if re_initialize:
        if os.path.exists(trainer_state_file):
            os.remove(trainer_state_file)
            logger.info("Removed existing trainer state for re-initialization: %s",
                        trainer_state_file)
        if os.path.exists(weights_file):
            os.remove(weights_file)
            logger.info("Removed existing policy weights for re-initialization: %s",
                        weights_file)

    game_spec = adapter.build_game_spec(
        args.experiment, experiment_dir, weights_file, reward_cfg_file,
        p, args.track,
    )
    data = train_rl(
        game=game_spec,
        config=RunConfig.from_training_params(p),
        probe=adapter.build_probe(p),
        warmup=adapter.build_warmup(p),
        extras=adapter.build_extras(weights_file, p, re_initialize),
        no_interrupt=args.no_interrupt,
        re_initialize=re_initialize,
    )


# ======================================================================
# SC2 play entry point
# ======================================================================

def _run_play_sc2(args: argparse.Namespace) -> None:
    try:
        from games.sc2.play import play_sc2  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            f"Cannot import SC2 play dependencies: {exc}\n"
            "Install pysc2 with:  poetry install --with sc2"
        ) from exc

    play_sc2(args.experiment, args)


# ======================================================================
# Assetto Corsa entry point (separate — uses its own runner)
# ======================================================================

def _run_assetto(args: argparse.Namespace) -> None:
    try:
        from games.assetto_corsa.entry import run as _ac_run  # noqa: PLC0415
    except ImportError as exc:
        raise ValueError(
            f"Cannot import Assetto Corsa dependencies: {exc}\n"
            "Install the assetto-corsa-rl package, then:\n"
            "    poetry install --with assetto_corsa"
        ) from exc

    _ac_run(args)


if __name__ == "__main__":
    main()
