"""TMNF training entry point.

The body of this module was previously inlined in main.py. Moving it here
keeps games/tmnf/ self-contained and lets main.py dispatch on --game.
The behaviour and CLI-visible side effects (config copying, train_rl
arguments, analytics output paths) are unchanged.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil

import yaml

import games.tmnf.policies  # noqa: F401 — side-effect: registers TMNF policies
from framework.training import train_rl
from games.tmnf.actions import DISCRETE_ACTIONS, PROBE_ACTIONS, WARMUP_ACTION
from games.tmnf.analytics import save_experiment_results
from games.tmnf.env import make_env
from games.tmnf.obs_spec import TMNF_OBS_SPEC

logger = logging.getLogger(__name__)


def run(args: argparse.Namespace) -> None:
    # Bootstrap: read track from master config before the experiment dir exists,
    # then re-read the experiment-local copy once it has been created.
    with open("config/training_params.yaml") as f:
        master_p = yaml.safe_load(f)
    track = master_p.get("track", "a03_centerline")

    experiment_dir = f"experiments/{track}/{args.experiment}"
    weights_file = f"{experiment_dir}/policy_weights.yaml"
    reward_cfg_file = f"{experiment_dir}/reward_config.yaml"
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
    obs_spec = TMNF_OBS_SPEC.with_lidar(n_lidar_rays)
    policy_type = p.get("policy_type", "hill_climbing")
    policy_params = p.get("policy_params") or {}
    re_initialize = args.re_initialize

    data = train_rl(
        experiment_name=args.experiment,
        make_env_fn=lambda: make_env(
            experiment_dir=experiment_dir,
            speed=p["speed"],
            in_game_episode_s=p["in_game_episode_s"],
            n_lidar_rays=n_lidar_rays,
        ),
        obs_spec=obs_spec,
        head_names=["steer", "accel", "brake"],
        discrete_actions=DISCRETE_ACTIONS,
        speed=p["speed"],
        n_sims=p["n_sims"],
        in_game_episode_s=p["in_game_episode_s"],
        weights_file=weights_file,
        reward_config_file=reward_cfg_file,
        mutation_scale=p["mutation_scale"],
        mutation_share=p.get("mutation_share", 1.0),
        probe_actions=PROBE_ACTIONS,
        probe_in_game_s=p["probe_s"],
        cold_start_restarts=p["cold_restarts"],
        cold_start_sims=p["cold_sims"],
        warmup_action=WARMUP_ACTION,
        warmup_steps=5,
        training_params=p,
        no_interrupt=args.no_interrupt,
        re_initialize=re_initialize,
        policy_type=policy_type,
        policy_params=policy_params,
        track=track,
        adaptive_mutation=p.get("adaptive_mutation", True),
        patience=p.get("patience", 0),
    )

    save_experiment_results(data, results_dir=f"{experiment_dir}/results")
