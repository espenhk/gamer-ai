"""TMNF RL training entry point.

Thin glue layer: reads experiment config, wires TMNF-specific objects into the
game-agnostic framework.training.train_rl(), then saves results.

All algorithm logic lives in framework/training.py.
All TMNF game logic lives in games/tmnf/.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil

import yaml

from framework.training import train_rl
from games.tmnf.obs_spec import TMNF_OBS_SPEC
from games.tmnf.actions import DISCRETE_ACTIONS, PROBE_ACTIONS, WARMUP_ACTION
from games.tmnf.env import make_env
from analytics import save_experiment_results  # backward-compat shim

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="TMNF RL training")
    parser.add_argument(
        "experiment",
        help="Experiment name — files stored in experiments/<track>/<name>/",
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

    # Bootstrap: read track from master config before the experiment dir exists,
    # then re-read the experiment-local copy once it has been created.
    with open("config/training_params.yaml") as f:
        master_p = yaml.safe_load(f)
    track = master_p.get("track", "a03_centerline")

    experiment_dir       = f"experiments/{track}/{args.experiment}"
    weights_file         = f"{experiment_dir}/policy_weights.yaml"
    reward_cfg_file      = f"{experiment_dir}/reward_config.yaml"
    training_params_file = f"{experiment_dir}/training_params.yaml"

    os.makedirs(experiment_dir, exist_ok=True)
    if not os.path.exists(reward_cfg_file):
        shutil.copy("config/reward_config.yaml", reward_cfg_file)
        logger.info("Copied master reward config → %s", reward_cfg_file)
    if not os.path.exists(training_params_file):
        shutil.copy("config/training_params.yaml", training_params_file)
        logger.info("Copied master training params → %s", training_params_file)

    with open(training_params_file) as f:
        p = yaml.safe_load(f)
    track = p.get("track", track)

    n_lidar_rays = p.get("n_lidar_rays", 0)
    obs_spec     = TMNF_OBS_SPEC.with_lidar(n_lidar_rays)

    data = train_rl(
        experiment_name     = args.experiment,
        make_env_fn         = lambda: make_env(
            experiment_dir    = experiment_dir,
            speed             = p["speed"],
            in_game_episode_s = p["in_game_episode_s"],
            n_lidar_rays      = n_lidar_rays,
        ),
        obs_spec            = obs_spec,
        head_names          = ["steer", "accel", "brake"],
        discrete_actions    = DISCRETE_ACTIONS,
        speed               = p["speed"],
        n_sims              = p["n_sims"],
        in_game_episode_s   = p["in_game_episode_s"],
        weights_file        = weights_file,
        reward_config_file  = reward_cfg_file,
        mutation_scale      = p["mutation_scale"],
        mutation_share      = p.get("mutation_share", 1.0),
        probe_actions       = PROBE_ACTIONS,
        probe_in_game_s     = p["probe_s"],
        cold_start_restarts = p["cold_restarts"],
        cold_start_sims     = p["cold_sims"],
        warmup_action       = WARMUP_ACTION,
        warmup_steps        = 100,
        training_params     = p,
        no_interrupt        = args.no_interrupt,
        re_initialize       = args.re_initialize,
        policy_type         = p.get("policy_type", "hill_climbing"),
        policy_params       = p.get("policy_params") or {},
        track               = track,
        adaptive_mutation   = p.get("adaptive_mutation", True),
    )

    save_experiment_results(data, results_dir=f"{experiment_dir}/results")


if __name__ == "__main__":
    main()
