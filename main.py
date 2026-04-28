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
from games.tmnf.policies import NeuralDQNPolicy, CMAESPolicy, REINFORCEPolicy, LSTMPolicy, LSTMEvolutionPolicy
from games.tmnf.analytics import save_experiment_results

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
    policy_type  = p.get("policy_type", "hill_climbing")
    policy_params = p.get("policy_params") or {}
    re_initialize = args.re_initialize

    # Factory callables for TMNF-specific policy types (injected into framework).
    def _make_neural_dqn() -> NeuralDQNPolicy:
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as _f:
                _cfg = yaml.safe_load(_f)
            if isinstance(_cfg, dict) and _cfg.get("policy_type") == "neural_dqn":
                return NeuralDQNPolicy.from_cfg(_cfg, n_lidar_rays=n_lidar_rays)
        return NeuralDQNPolicy(
            hidden_sizes        = policy_params.get("hidden_sizes",        [64, 64]),
            replay_buffer_size  = policy_params.get("replay_buffer_size",  10000),
            batch_size          = policy_params.get("batch_size",          64),
            min_replay_size     = policy_params.get("min_replay_size",     500),
            target_update_freq  = policy_params.get("target_update_freq",  200),
            learning_rate       = policy_params.get("learning_rate",       0.001),
            epsilon_start       = policy_params.get("epsilon_start",       1.0),
            epsilon_end         = policy_params.get("epsilon_end",         0.05),
            epsilon_decay_steps = policy_params.get("epsilon_decay_steps", 5000),
            gamma               = policy_params.get("gamma",               0.99),
            n_lidar_rays        = n_lidar_rays,
        )

    def _make_cmaes() -> CMAESPolicy:
        pop_size = policy_params.get("population_size", 20)
        sigma    = policy_params.get("initial_sigma",   0.3)
        policy   = CMAESPolicy(
            population_size = pop_size,
            initial_sigma   = sigma,
            n_lidar_rays    = n_lidar_rays,
        )
        if os.path.exists(weights_file) and not re_initialize:
            from games.tmnf.policies import WeightedLinearPolicy as _WLP
            champion = _WLP.from_cfg(
                yaml.safe_load(open(weights_file)) or {}, n_lidar_rays=n_lidar_rays
            )
            policy.initialize_from_champion(champion)
        else:
            policy.initialize_random()
        return policy

    def _make_reinforce() -> REINFORCEPolicy:
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as _f:
                _cfg = yaml.safe_load(_f) or {}
            if isinstance(_cfg, dict) and _cfg.get("policy_type") == "reinforce":
                return REINFORCEPolicy.from_cfg(_cfg, n_lidar_rays=n_lidar_rays)
        return REINFORCEPolicy(
            hidden_sizes  = policy_params.get("hidden_sizes",  [64, 64]),
            learning_rate = policy_params.get("learning_rate", 0.001),
            gamma         = policy_params.get("gamma",         0.99),
            entropy_coeff = policy_params.get("entropy_coeff", 0.01),
            baseline      = policy_params.get("baseline",      "running_mean"),
            n_lidar_rays  = n_lidar_rays,
        )

    def _make_lstm() -> LSTMEvolutionPolicy:
        hidden_size = policy_params.get("hidden_size",     32)
        pop_size    = policy_params.get("population_size", 20)
        sigma       = policy_params.get("initial_sigma",   0.05)
        policy      = LSTMEvolutionPolicy(
            hidden_size     = hidden_size,
            population_size = pop_size,
            initial_sigma   = sigma,
            n_lidar_rays    = n_lidar_rays,
        )
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as _f:
                _cfg = yaml.safe_load(_f) or {}
            if isinstance(_cfg, dict) and _cfg.get("policy_type") == "lstm":
                saved_hidden_size = _cfg.get("hidden_size")
                saved_n_lidar_rays = _cfg.get("n_lidar_rays")
                if saved_hidden_size is not None and saved_hidden_size != hidden_size:
                    raise ValueError(
                        "Saved LSTM champion hidden_size does not match current run: "
                        f"saved={saved_hidden_size}, current={hidden_size}"
                    )
                if saved_n_lidar_rays is not None and saved_n_lidar_rays != n_lidar_rays:
                    raise ValueError(
                        "Saved LSTM champion n_lidar_rays does not match current run: "
                        f"saved={saved_n_lidar_rays}, current={n_lidar_rays}"
                    )
                champion = LSTMPolicy.from_cfg(_cfg)
                policy.initialize_from_champion(champion)
        return policy

    extra_policy_types = {
        "neural_dqn": _make_neural_dqn,
        "cmaes":      _make_cmaes,
        "reinforce":  _make_reinforce,
        "lstm":       _make_lstm,
    }
    extra_loop_dispatch = {
        "neural_dqn": "q_learning",
        "cmaes":      "cmaes",
        "reinforce":  "q_learning",
        "lstm":       "cmaes",
    }

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
        warmup_steps        = 5,
        training_params     = p,
        no_interrupt        = args.no_interrupt,
        re_initialize       = re_initialize,
        do_pretrain         = p.get("do_pretrain", False),
        policy_type         = policy_type,
        policy_params       = policy_params,
        track               = track,
        adaptive_mutation   = p.get("adaptive_mutation", True),
        extra_policy_types  = extra_policy_types,
        extra_loop_dispatch = extra_loop_dispatch,
        patience            = p.get("patience", 0),
    )

    save_experiment_results(data, results_dir=f"{experiment_dir}/results")


if __name__ == "__main__":
    main()
