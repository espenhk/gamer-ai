"""RL training entry point.

Thin glue layer: reads experiment config, wires game-specific objects into the
game-agnostic framework.training.train_rl(), then saves results.

Supports multiple games via the ``--game`` flag:
    python main.py <experiment> --game tmnf   (default)
    python main.py <experiment> --game torcs

All algorithm logic lives in framework/training.py.
Game-specific logic lives in games/<name>/.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil

import yaml

from framework.training import train_rl

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="RL training")
    parser.add_argument(
        "experiment",
        help="Experiment name — files stored in experiments/<track>/<name>/",
    )
    parser.add_argument(
        "--game", default="tmnf", choices=["tmnf", "torcs"],
        help="Game to train on (default: tmnf)",
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

    if args.game == "torcs":
        _run_torcs(args)
    else:
        _run_tmnf(args)


# ======================================================================
# TMNF entry point (original logic, unchanged)
# ======================================================================

def _run_tmnf(args: argparse.Namespace) -> None:
    from games.tmnf.obs_spec import TMNF_OBS_SPEC
    from games.tmnf.actions import DISCRETE_ACTIONS, PROBE_ACTIONS, WARMUP_ACTION
    from games.tmnf.env import make_env
    from games.tmnf.policies import NeuralDQNPolicy, CMAESPolicy, REINFORCEPolicy, LSTMPolicy, LSTMEvolutionPolicy
    from games.tmnf.analytics import save_experiment_results

    # Bootstrap: read track from master config before the experiment dir exists,
    # then re-read the experiment-local copy once it has been created.
    with open("games/tmnf/config/training_params.yaml") as f:
        master_p = yaml.safe_load(f)
    track = master_p.get("track", "a03_centerline")

    experiment_dir       = f"experiments/{track}/{args.experiment}"
    weights_file         = f"{experiment_dir}/policy_weights.yaml"
    trainer_state_file   = f"{experiment_dir}/trainer_state.npz"
    reward_cfg_file      = f"{experiment_dir}/reward_config.yaml"
    training_params_file = f"{experiment_dir}/training_params.yaml"

    os.makedirs(experiment_dir, exist_ok=True)
    if not os.path.exists(reward_cfg_file):
        shutil.copy("games/tmnf/config/reward_config.yaml", reward_cfg_file)
        logger.info("Copied master reward config → %s", reward_cfg_file)
    if not os.path.exists(training_params_file):
        shutil.copy("games/tmnf/config/training_params.yaml", training_params_file)
        logger.info("Copied master training params → %s", training_params_file)

    with open(training_params_file) as f:
        p = yaml.safe_load(f)
    track = p.get("track", track)

    n_lidar_rays = p.get("n_lidar_rays", 0)
    obs_spec     = TMNF_OBS_SPEC.with_lidar(n_lidar_rays)
    policy_type  = p.get("policy_type", "hill_climbing")
    policy_params = p.get("policy_params") or {}
    action_window_ticks = p.get("action_window_ticks", 1)
    decision_offset_pct = p.get("decision_offset_pct", 0.75)
    re_initialize = args.re_initialize

    # Delete persisted state when re-initializing so stale policy/trainer state
    # doesn't survive the restart.
    if re_initialize:
        if os.path.exists(trainer_state_file):
            os.remove(trainer_state_file)
            logger.info("Removed existing trainer state for re-initialization: %s",
                        trainer_state_file)
        if os.path.exists(weights_file):
            os.remove(weights_file)
            logger.info("Removed existing policy weights for re-initialization: %s",
                        weights_file)

    # Factory callables for TMNF-specific policy types (injected into framework).
    def _make_neural_dqn() -> NeuralDQNPolicy:
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as _f:
                _cfg = yaml.safe_load(_f)
            if isinstance(_cfg, dict) and _cfg.get("policy_type") == "neural_dqn":
                policy = NeuralDQNPolicy.from_cfg(_cfg, n_lidar_rays=n_lidar_rays)
                if os.path.exists(trainer_state_file):
                    try:
                        policy.load_trainer_state(trainer_state_file)
                        logger.info("[NeuralDQNPolicy] loaded trainer state from %s",
                                    trainer_state_file)
                    except (ValueError, KeyError) as exc:
                        logger.warning(
                            "[NeuralDQNPolicy] could not load trainer state from %s — %s; "
                            "continuing with default state.",
                            trainer_state_file, exc,
                        )
                return policy
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
            if os.path.exists(trainer_state_file):
                try:
                    policy.load_trainer_state(trainer_state_file)
                    logger.info("[CMAESPolicy] loaded trainer state from %s",
                                trainer_state_file)
                except (ValueError, KeyError) as exc:
                    logger.warning(
                        "[CMAESPolicy] could not load trainer state from %s — %s; "
                        "continuing with champion weights and default distribution.",
                        trainer_state_file, exc,
                    )
        else:
            policy.initialize_random()
        return policy

    def _make_reinforce() -> REINFORCEPolicy:
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as _f:
                _cfg = yaml.safe_load(_f) or {}
            if isinstance(_cfg, dict) and _cfg.get("policy_type") == "reinforce":
                policy = REINFORCEPolicy.from_cfg(_cfg, n_lidar_rays=n_lidar_rays)
                if os.path.exists(trainer_state_file):
                    try:
                        policy.load_trainer_state(trainer_state_file)
                        logger.info("[REINFORCEPolicy] loaded trainer state from %s",
                                    trainer_state_file)
                    except (ValueError, KeyError) as exc:
                        logger.warning(
                            "[REINFORCEPolicy] could not load trainer state from %s — %s; "
                            "continuing with default state.",
                            trainer_state_file, exc,
                        )
                return policy
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
                if os.path.exists(trainer_state_file):
                    try:
                        policy.load_trainer_state(trainer_state_file)
                        logger.info("[LSTMEvolutionPolicy] loaded trainer state from %s",
                                    trainer_state_file)
                    except (ValueError, KeyError) as exc:
                        logger.warning(
                            "[LSTMEvolutionPolicy] could not load trainer state from %s — %s; "
                            "continuing with champion weights and default distribution.",
                            trainer_state_file, exc,
                        )
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
            experiment_dir      = experiment_dir,
            speed               = p["speed"],
            in_game_episode_s   = p["in_game_episode_s"],
            n_lidar_rays        = n_lidar_rays,
            decision_offset_pct = decision_offset_pct,
            action_window_ticks = p.get("action_window_ticks", 1),
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


# ======================================================================
# TORCS entry point
# ======================================================================

def _run_torcs(args: argparse.Namespace) -> None:
    from games.torcs.obs_spec import TORCS_OBS_SPEC
    from games.torcs.actions import DISCRETE_ACTIONS, PROBE_ACTIONS, WARMUP_ACTION
    from games.torcs.env import make_env
    from games.torcs.analytics import save_experiment_results

    # Read TORCS master config.
    with open("games/torcs/config/training_params.yaml") as f:
        master_p = yaml.safe_load(f)

    experiment_dir       = f"experiments/torcs/{args.experiment}"
    weights_file         = f"{experiment_dir}/policy_weights.yaml"
    reward_cfg_file      = f"{experiment_dir}/reward_config.yaml"
    training_params_file = f"{experiment_dir}/training_params.yaml"

    os.makedirs(experiment_dir, exist_ok=True)
    if not os.path.exists(reward_cfg_file):
        shutil.copy("games/torcs/config/reward_config.yaml", reward_cfg_file)
        logger.info("Copied TORCS reward config → %s", reward_cfg_file)
    if not os.path.exists(training_params_file):
        shutil.copy("games/torcs/config/training_params.yaml", training_params_file)
        logger.info("Copied TORCS training params → %s", training_params_file)

    with open(training_params_file) as f:
        p = yaml.safe_load(f)

    obs_spec     = TORCS_OBS_SPEC
    policy_type  = p.get("policy_type", "hill_climbing")
    policy_params = p.get("policy_params") or {}

    data = train_rl(
        experiment_name     = args.experiment,
        make_env_fn         = lambda: make_env(
            experiment_dir    = experiment_dir,
            max_episode_time_s = p["in_game_episode_s"],
        ),
        obs_spec            = obs_spec,
        head_names          = ["steer", "accel", "brake"],
        discrete_actions    = DISCRETE_ACTIONS,
        speed               = p.get("speed", 1.0),
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
        do_pretrain         = p.get("do_pretrain", False),
        policy_type         = policy_type,
        policy_params       = policy_params,
        track               = "torcs",
        adaptive_mutation   = p.get("adaptive_mutation", True),
        patience            = p.get("patience", 0),
    )

    save_experiment_results(data, results_dir=f"{experiment_dir}/results")
    logger.info("TORCS training complete. Results saved to %s", experiment_dir)


if __name__ == "__main__":
    main()
