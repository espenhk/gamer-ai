"""Assetto Corsa training entry point.

Reads experiment config, wires AC-specific objects into the game-agnostic
framework.training.train_rl(), then saves results. Mirrors
games.tmnf.entry.run() so main.py only has to dispatch on --game.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import yaml

from framework.training import train_rl
from games.assetto_corsa.actions import DISCRETE_ACTIONS, PROBE_ACTIONS, WARMUP_ACTION
from games.assetto_corsa.analytics import save_experiment_results
from games.assetto_corsa.env import make_env
from games.assetto_corsa.obs_spec import AC_OBS_SPEC, with_vision

logger = logging.getLogger(__name__)


_PKG_DIR    = Path(__file__).resolve().parent
_CONFIG_DIR = _PKG_DIR / "config"


def run(args: argparse.Namespace) -> None:
    """Train an AC experiment from a parsed argparse Namespace.

    The Namespace must expose: experiment, no_interrupt, re_initialize.
    """
    # Master-config bootstrap, mirroring games.tmnf.entry.run().
    master_training = _CONFIG_DIR / "training_params.yaml"
    master_reward   = _CONFIG_DIR / "reward_config.yaml"

    with open(master_training) as f:
        master_p = yaml.safe_load(f)
    track = master_p.get("track", "assetto_default")

    experiment_dir       = Path(f"experiments/assetto_corsa/{track}/{args.experiment}")
    weights_file         = str(experiment_dir / "policy_weights.yaml")
    reward_cfg_file      = str(experiment_dir / "reward_config.yaml")
    training_params_file = str(experiment_dir / "training_params.yaml")

    experiment_dir.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(reward_cfg_file):
        shutil.copy(master_reward, reward_cfg_file)
        logger.info("Copied master AC reward config → %s", reward_cfg_file)
    if not os.path.exists(training_params_file):
        shutil.copy(master_training, training_params_file)
        logger.info("Copied master AC training params → %s", training_params_file)

    with open(training_params_file) as f:
        p = yaml.safe_load(f)
    track = p.get("track", track)

    n_vision      = int(p.get("n_vision", 0))
    obs_spec      = with_vision(n_vision)
    policy_type   = p.get("policy_type", "genetic")
    policy_params = p.get("policy_params") or {}

    data = train_rl(
        experiment_name     = args.experiment,
        make_env_fn         = lambda: make_env(
            experiment_dir    = str(experiment_dir),
            speed             = p["speed"],
            in_game_episode_s = p["in_game_episode_s"],
            n_vision          = n_vision,
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
        warmup_steps        = 5,
        training_params     = p,
        no_interrupt        = args.no_interrupt,
        re_initialize       = args.re_initialize,
        policy_type         = policy_type,
        policy_params       = policy_params,
        track               = track,
        adaptive_mutation   = p.get("adaptive_mutation", True),
        patience            = p.get("patience", 0),
    )

    save_experiment_results(data, results_dir=str(experiment_dir / "results"))
