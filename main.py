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
    python main.py <experiment> --game rocket_league
    python main.py <experiment> --game iracing
    python main.py <experiment> --game atari

All algorithm logic lives in framework/training.py.
Game-specific logic lives in games/<name>/.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys

import yaml

from framework.env_loader import load_dotenv

load_dotenv()

from framework.game_adapter import GAME_ADAPTERS
from framework.run_config import RunConfig
from framework.training import train_rl
from framework.version import code_version

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the fully configured argument parser for main().

    Extracted so tests can import and exercise the real parser without
    invoking the full training stack.
    """
    parser = argparse.ArgumentParser(description="RL training (multi-game)")
    parser.add_argument(
        "experiment",
        help="Experiment name — files stored in experiments/<...>/<name>/",
    )
    parser.add_argument(
        "--game",
        default="tmnf",
        choices=["tmnf", "beamng", "assetto", "car_racing", "torcs", "sc2", "rocket_league", "iracing", "atari"],
        help=(
            "Select which simulator to use. "
            "Choices: tmnf (default), beamng, assetto, car_racing, torcs, sc2, "
            "rocket_league, iracing, atari. "
            "beamng, assetto, rocket_league and iracing require optional simulator "
            "dependencies; atari requires ale-py."
        ),
    )
    parser.add_argument(
        "--track",
        default=None,
        help="Override the track / map name from the config (e.g. aalborg, CollectMineralShards).",
    )
    parser.add_argument(
        "--no-interrupt",
        action="store_true",
        help="Skip all 'Press Enter' prompts and run all phases automatically",
    )
    parser.add_argument(
        "--re-initialize",
        action="store_true",
        help="Ignore any existing weights file and restart from scratch, including probe and cold-start phases.",
    )
    parser.add_argument(
        "--live-gui",
        action="store_true",
        help=(
            "Show a live GUI window with per-step reward components (rolling avg "
            "of 5 steps) and observation values during training."
        ),
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help=(
            "Open a human-visible game window while training (Atari only). "
            "Ignored for all other games. "
            "Slows training to roughly real-time (~60 fps); not suitable for grid search."
        ),
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--play",
        action="store_true",
        help=(
            "Human-vs-AI interactive play mode (SC2 only).  "
            "Loads the champion policy from the experiment and launches a "
            "two-player PySC2 session where you play via the SC2 UI against "
            "the trained agent.  No weight updates occur."
        ),
    )
    mode_group.add_argument(
        "--eval",
        action="store_true",
        help=(
            "Evaluation mode (SC2 and Atari).  "
            "Loads the champion policy from the experiment and runs it for "
            "--num-episodes episodes, reporting aggregate statistics.  "
            "For Atari, opens a human-visible game window automatically.  "
            "No weight updates occur."
        ),
    )
    mode_group.add_argument(
        "--bc",
        action="store_true",
        help=(
            "Behaviour-cloning pre-training mode.  Available for any game whose "
            "adapter exposes a BCAdapter (SC2 and TMNF today).  "
            "Reads replay files from --replay-dir (or from a per-game live "
            "demonstration source when None), fits a policy to the data, and "
            "writes policy_weights.yaml into the experiment directory.  Run "
            "without --bc afterwards to fine-tune from the pre-trained weights."
        ),
    )

    parser.add_argument(
        "--replay-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory of replay files for --bc mode (e.g. .SC2Replay for SC2).  "
            "May also be set via bc_replay_dir in training_params.yaml.  Some "
            "games (e.g. TMNF live-demo source) accept no replay-dir."
        ),
    )
    parser.add_argument(
        "--bc-player",
        default=None,
        choices=["winner", "1", "2"],
        help=("Which player to clone (SC2 only): 'winner' (default), '1', or '2'.  Overrides bc_player_id in config."),
    )
    parser.add_argument(
        "--bc-race",
        default=None,
        choices=["terran", "protoss", "zerg", "any"],
        help=("Race filter for --bc mode (SC2 only, default: any).  Overrides bc_race in config."),
    )
    parser.add_argument(
        "--bc-target",
        default=None,
        metavar="POLICY_TYPE",
        help=(
            "Policy type to pre-train.  Overrides bc_target in config.  "
            "Validated against the active game's BCAdapter.supported_targets "
            "at runtime."
        ),
    )

    def _positive_int(name: str):
        def _check(v: str) -> int:
            iv = int(v)
            if iv < 1:
                raise argparse.ArgumentTypeError(f"{name} must be ≥ 1, got {v}")
            return iv

        return _check

    parser.add_argument(
        "--num-episodes",
        type=_positive_int("--num-episodes"),
        default=1,
        help="Number of evaluation episodes to run (default: 1, used with --eval)",
    )
    parser.add_argument(
        "--bot-difficulty",
        default=None,
        choices=[
            "very_easy",
            "easy",
            "medium",
            "medium_hard",
            "hard",
            "harder",
            "very_hard",
            "cheat_vision",
            "cheat_money",
            "cheat_insane",
        ],
        help=(
            "SC2 built-in bot difficulty for ladder maps during --eval "
            "(default: use experiment config, fallback very_easy).  "
            "Ignored for minigame maps."
        ),
    )
    parser.add_argument(
        "--eval-speed",
        type=_positive_int("--eval-speed"),
        default=None,
        metavar="STEP_MUL",
        help=(
            "Override step_mul during --eval.  Controls how many game ticks "
            "advance between policy calls — i.e. observation granularity, "
            "not action rate (max_apm governs that).  Defaults to the "
            "experiment's configured step_mul; best left there so the agent "
            "sees the same state deltas it was trained on."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help=(
            "Write logs to PATH in addition to the terminal (tee).  "
            "The file is opened fresh on each run (overwrites).  "
            "Combine with --log-level DEBUG to capture SC2 state snapshots "
            "(available actions, units, buildings) every ~10 s."
        ),
    )
    parser.add_argument(
        "--workers",
        type=_positive_int("--workers"),
        default=None,
        metavar="N",
        help=(
            "Override training_params n_workers — number of local SC2 binaries "
            "used to evaluate population members in parallel (issue #229).  "
            "1 = serial.  Only meaningful for population-based SC2 policies "
            "(sc2_genetic, sc2_cmaes, sc2_lstm, sc2_cnn)."
        ),
    )
    return parser


def main() -> None:
    # Handle --version before argparse so it doesn't trip over the
    # otherwise-required `experiment` positional.
    if "--version" in sys.argv[1:]:
        print(code_version())
        return

    parser = _build_arg_parser()
    args = parser.parse_args()

    _log_level = getattr(logging, args.log_level)
    _log_fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    _log_datefmt = "%H:%M:%S"
    logging.basicConfig(level=_log_level, format=_log_fmt, datefmt=_log_datefmt)
    if args.log_file:
        _fh = logging.FileHandler(args.log_file, mode="w", encoding="utf-8", delay=False)
        _fh.setLevel(_log_level)
        _fh.setFormatter(logging.Formatter(_log_fmt, datefmt=_log_datefmt))
        logging.getLogger().addHandler(_fh)
        logger.info("Logging to file: %s", args.log_file)
    logger.info("gamer-ai code version: %s", code_version())

    if args.play and args.game != "sc2":
        raise SystemExit("--play is only supported with --game sc2")

    if args.eval and args.game not in ("sc2", "atari"):
        raise SystemExit("--eval is only supported with --game sc2 or --game atari")

    if args.play:
        _run_play_sc2(args)
        return

    if args.eval:
        if args.game == "sc2":
            _run_eval_sc2(args)
        else:
            _run_eval_atari(args)
        return

    adapter = GAME_ADAPTERS[args.game]()

    if args.bc:
        _run_bc(adapter, args)
        return

    _run_one(adapter, args)


# ======================================================================
# Unified runner
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

    # CLI override for intra-run parallel evaluation (issue #229).
    if getattr(args, "workers", None) is not None:
        p["n_workers"] = int(args.workers)
        logger.info("Overriding training_params n_workers = %d from --workers", p["n_workers"])
    if getattr(args, "live_gui", False):
        p["live_gui"] = True
        logger.info("Live GUI telemetry enabled via --live-gui")
    if getattr(args, "render", False):
        if adapter.name == "atari":
            p["render_mode"] = "human"
            logger.info("Human-visible rendering enabled via --render")
        else:
            logger.warning("--render is only supported for Atari; flag ignored for game %r", adapter.name)

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
            logger.info("Removed existing trainer state for re-initialization: %s", trainer_state_file)
        if os.path.exists(weights_file):
            os.remove(weights_file)
            logger.info("Removed existing policy weights for re-initialization: %s", weights_file)

    game_spec = adapter.build_game_spec(
        args.experiment,
        experiment_dir,
        weights_file,
        reward_cfg_file,
        p,
        args.track,
    )
    train_rl(
        game=game_spec,
        config=RunConfig.from_training_params(p),
        probe=adapter.build_probe(p),
        warmup=adapter.build_warmup(p),
        no_interrupt=args.no_interrupt,
        re_initialize=re_initialize,
    )


# ======================================================================
# SC2 evaluation entry point
# ======================================================================


def _run_eval_sc2(args: argparse.Namespace) -> None:
    try:
        from games.sc2.eval import eval_sc2  # noqa: PLC0415

        eval_sc2(args.experiment, args)
    except ImportError as exc:
        raise SystemExit(
            f"Cannot import SC2 eval dependencies: {exc}\nInstall pysc2 with:  poetry install --with sc2"
        ) from exc


def _run_eval_atari(args: argparse.Namespace) -> None:
    from games.atari.eval import eval_atari  # noqa: PLC0415

    eval_atari(args.experiment, args)


# ======================================================================
# SC2 play entry point
# ======================================================================


def _run_play_sc2(args: argparse.Namespace) -> None:
    try:
        from games.sc2.play import play_sc2  # noqa: PLC0415

        play_sc2(args.experiment, args)
    except ImportError as exc:
        raise SystemExit(
            f"Cannot import SC2 play dependencies: {exc}\nInstall pysc2 with:  poetry install --with sc2"
        ) from exc


# ======================================================================
# SC2 behaviour-cloning entry point
# ======================================================================


def _run_bc(adapter, args: argparse.Namespace) -> None:
    """Game-agnostic behaviour-cloning entry point.

    Drives the framework BC orchestrator (:func:`framework.bc.run`) via the
    game's :class:`framework.bc.BCAdapter`.  Games without a BC adapter wired
    on their :class:`GameAdapter` raise a clean error here.
    """
    bc_adapter = getattr(adapter, "bc", None)
    if bc_adapter is None:
        raise SystemExit(
            f"--bc is not supported for --game {args.game}: no BCAdapter wired on "
            f"the game adapter.  See framework/bc.py for the integration contract."
        )

    from framework.bc import run as bc_run

    master_cfg = os.path.join(adapter.config_dir, "training_params.yaml")
    with open(master_cfg) as f:
        master_p = yaml.safe_load(f)

    experiment_dir = adapter.experiment_dir(args.experiment, master_p, args.track)
    training_params_file = f"{experiment_dir}/training_params.yaml"
    reward_cfg_file = f"{experiment_dir}/reward_config.yaml"

    os.makedirs(experiment_dir, exist_ok=True)
    # Mirror _run_one for the master-config copy step.  Some games (e.g. TMNF)
    # build an env during BC and need reward_config.yaml on disk too.
    master_reward_cfg = os.path.join(adapter.config_dir, "reward_config.yaml")
    if not os.path.exists(reward_cfg_file) and os.path.exists(master_reward_cfg):
        shutil.copy(master_reward_cfg, reward_cfg_file)
        logger.info("Copied master reward config → %s", reward_cfg_file)
    if not os.path.exists(training_params_file):
        shutil.copy(master_cfg, training_params_file)
        logger.info("Copied master training params → %s", training_params_file)

    with open(training_params_file) as f:
        p = yaml.safe_load(f)

    # Apply game-specific reward-config decoration (e.g. TMNF centerline_path)
    # so the BC-driven env constructs cleanly.
    if os.path.exists(reward_cfg_file):
        with open(reward_cfg_file) as f:
            reward_cfg = yaml.safe_load(f) or {}
        adapter.decorate_reward_cfg(reward_cfg, p, args.track)
        with open(reward_cfg_file, "w") as f:
            yaml.dump(reward_cfg, f, default_flow_style=False, sort_keys=False)

    obs_spec = _build_bc_obs_spec(args.game, p)

    # CLI flags override config; config overrides BCAdapter default.
    replay_dir = getattr(args, "replay_dir", None) or p.get("bc_replay_dir")
    race = getattr(args, "bc_race", None) or p.get("bc_race", "any")
    target = getattr(args, "bc_target", None) or p.get("bc_target", bc_adapter.default_target)

    # SC2-specific: --bc-player overrides bc_player_id in the training params
    # the BCAdapter will read.  Other games can ignore the key.
    if getattr(args, "bc_player", None):
        p = dict(p, bc_player_id=args.bc_player)

    # Pass the resolved experiment directory through so adapters that build
    # an env during BC (e.g. TMNF) can find reward_config.yaml on disk.
    p = dict(p, _bc_experiment_dir=experiment_dir)

    try:
        bc_run(
            bc_adapter,
            replay_dir,
            experiment_dir,
            obs_spec=obs_spec,
            target=target,
            training_params=p,
            race=race,
            max_replays=p.get("bc_max_replays"),
        )
    except ValueError as exc:
        # Adapter/orchestrator validation errors (missing --replay-dir,
        # unsupported target, empty replay directory, race filter dropped
        # all replays, ...) should exit cleanly with the message rather
        # than dump a traceback at the user.
        raise SystemExit(str(exc)) from exc
    except ImportError as exc:
        # Optional game-specific deps (pysc2 for SC2, tmnf group for TMNF,
        # etc.) are imported lazily inside the adapter; surface a clean
        # install hint instead of a raw traceback.
        raise SystemExit(
            f"Cannot import {args.game} BC dependencies: {exc}\n"
            f"Install the {args.game} dependencies with:  poetry install --with {args.game}"
        ) from exc
    logger.info(
        "BC complete — run 'python main.py %s --game %s' to fine-tune from pre-trained weights.",
        args.experiment,
        args.game,
    )


def _build_bc_obs_spec(game: str, training_params: dict):
    """Build the active observation spec for a BC run.

    Each game decides its own obs_spec wiring.  The CLI asks the game
    module directly; new games add a branch here when they wire a BCAdapter.
    """
    if game == "sc2":
        from games.sc2.adapter import _get_obs_spec

        map_name = training_params.get("map_name", "MoveToBeacon")
        preset = training_params.get("obs_spec_preset")
        enable_belief = training_params.get("enable_belief", False)
        return _get_obs_spec(map_name, preset, enable_belief)

    if game == "tmnf":
        from games.tmnf.obs_spec import TMNF_OBS_SPEC

        n_lidar_rays = training_params.get("n_lidar_rays", 0)
        return TMNF_OBS_SPEC.with_lidar(n_lidar_rays)

    raise SystemExit(
        f"BC obs_spec for --game {game} is not wired into main.py yet.  See _build_bc_obs_spec in main.py."
    )


if __name__ == "__main__":
    main()
