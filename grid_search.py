"""
grid_search.py — Run a Cartesian-product sweep over training and reward params.

Usage (from tmnf/):
    python grid_search.py config/my_grid.yaml [--no-interrupt]

Config format (YAML):
    base_name: "gs_v1"
    track: a03_centerline       # stem of tracks/<track>.npy (default: a03_centerline)
    training_params:
        speed: 10.0
        n_sims: 50
        mutation_scale: [0.05, 0.1, 0.2]   # list = search axis
        ...
    reward_params:
        centerline_weight: [-0.1, -0.5]     # list = search axis
        accel_bonus: 1.0
        ...

Any param set to a list becomes a search axis; all others are fixed.
One experiment is run per unique combination. Names encode only the varied params.
Results are written to experiments/<track>/<name>/.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
from time import sleep
from typing import Any
import uuid as _uuid

import yaml

from distributed.protocol import ComboSpec
from distributed.coordinator import Coordinator

from games.tmnf.analytics import save_experiment_results, save_grid_summary
from framework.training import train_rl
from games.tmnf.obs_spec import TMNF_OBS_SPEC
from games.tmnf.actions import DISCRETE_ACTIONS, PROBE_ACTIONS, WARMUP_ACTION
from games.tmnf.env import make_env
from games.tmnf.policies import (
    CMAESPolicy,
    LSTMEvolutionPolicy,
    LSTMPolicy,
    NeuralDQNPolicy,
    REINFORCEPolicy,
    WeightedLinearPolicy,
)

# Game-specific and analytics imports are deferred to the functions that need them
# so that importing grid_search for testing (utility functions only) doesn't require
# a live game environment or Windows-only modules.

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Param name → short abbreviation for experiment directory names
# ---------------------------------------------------------------------------

_ABBREV = {
    # training params
    "speed": "speed",
    "n_sims": "nsims",
    "in_game_episode_s": "ep",
    "mutation_scale": "ms",
    "mutation_share": "mshare",
    "probe_s": "probe",
    "cold_restarts": "cr",
    "cold_sims": "cs",
    "policy_type": "pt",
    "do_pretrain": "dpt",
    "patience": "pat",
    "action_window_ticks": "win",
    "decision_offset_pct": "dec",
    # neural_net policy params
    "hidden_sizes": "hs",
    # genetic policy params
    "population_size": "pop",
    "elite_k": "ek",
    # cmaes policy params
    "initial_sigma": "sigma",
    # epsilon-greedy params
    "epsilon": "eps",
    "epsilon_decay": "ed",
    "epsilon_min": "emin",
    "alpha": "alpha",
    "gamma": "gamma",
    "n_bins": "bins",
    # mcts params
    "mcts_c": "mc",
    # reward params
    "progress_weight": "pw",
    "centerline_weight": "cw",
    "centerline_exp": "ce",
    "speed_weight": "sw",
    "step_penalty": "sp",
    "finish_bonus": "fb",
    "finish_time_weight": "ftw",
    "par_time_s": "par",
    "accel_bonus": "ab",
    "airborne_penalty": "ap",
    "crash_threshold_m": "ct",
    "lidar_wall_weight": "lww",
}

# Top-level training_params keys that should be forwarded into policy_params.
# Allows grid axes like `epsilon: [0.5, 1.0]` without nesting inside policy_params.
# mcts_c is renamed to c because that's what MCTSPolicy.from_cfg expects.
_POLICY_PARAM_MAP = {
    "hidden_sizes": "hidden_sizes",  # neural_net
    "epsilon": "epsilon",  # epsilon_greedy
    "epsilon_decay": "epsilon_decay",  # epsilon_greedy
    "epsilon_min": "epsilon_min",  # epsilon_greedy
    "alpha": "alpha",  # epsilon_greedy / mcts
    "gamma": "gamma",  # epsilon_greedy / mcts
    "n_bins": "n_bins",  # epsilon_greedy / mcts
    "mcts_c": "c",  # mcts (renamed)
    "population_size": "population_size",  # genetic / cmaes
    "elite_k": "elite_k",  # genetic
    "initial_sigma": "initial_sigma",  # cmaes
}


def _fmt_value(v: Any) -> str:
    """Format a param value for use in a directory name.

    - Integers: plain digits (50 → '50')
    - Floats: strip trailing zeros; replace leading '-' with 'n' (−0.1 → 'n0.1')
    - Others: str()
    """
    if isinstance(v, float):
        s = f"{v:g}"  # e.g. '10', '-0.1', '0.05'
        s = s.replace("-", "n")
        return s
    if isinstance(v, int):
        return str(v)
    return str(v).replace("-", "n")


def _make_experiment_name(
    base_name: str, combo: dict[str, Any], varied_keys: list[str]
) -> str:
    """Build experiment name from base + only the varied param values."""
    parts = [base_name]
    for key in varied_keys:
        abbrev = _ABBREV.get(key, key)
        parts.append(f"{abbrev}{_fmt_value(combo[key])}")
    return "__".join(parts)


# ---------------------------------------------------------------------------
# Config loading and grid expansion
# ---------------------------------------------------------------------------


def _load_grid_config(
    path: str,
) -> tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load grid config YAML. Returns (base_name, track, training_spec, reward_spec, distribute_cfg)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base_name = cfg.get("base_name", "gs")
    track = cfg.get("track", "a03_centerline")
    training_spec = cfg.get("training_params", {})
    reward_spec = cfg.get("reward_params", {})
    distribute_cfg = cfg.get("distribute", {})
    return base_name, track, training_spec, reward_spec, distribute_cfg


def _expand_grid(
    training_spec: dict[str, Any], reward_spec: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Expand all list-valued params into a Cartesian product.
    Each element of the returned list is a flat dict:
        {"training_params": {...}, "reward_params": {...}}
    Also returns the list of varied keys (in order) for name generation.
    """
    # Collect axes: (key, [values], "training"|"reward")
    axes = []
    for key, val in training_spec.items():
        if isinstance(val, list):
            axes.append((key, val, "training"))
    for key, val in reward_spec.items():
        if isinstance(val, list):
            axes.append((key, val, "reward"))

    if not axes:
        # No variation — single run with fixed values
        return [
            {"training_params": dict(training_spec), "reward_params": dict(reward_spec)}
        ], []

    varied_keys = [a[0] for a in axes]
    value_lists = [a[1] for a in axes]
    sources = [a[2] for a in axes]

    combos = []
    for combo_values in itertools.product(*value_lists):
        t_params = dict(training_spec)
        r_params = dict(reward_spec)
        flat = {}
        for key, val, src in zip(varied_keys, combo_values, sources):
            flat[key] = val
            if src == "training":
                t_params[key] = val
            else:
                r_params[key] = val
        combos.append(
            {"training_params": t_params, "reward_params": r_params, "_flat": flat}
        )

    return combos, varied_keys


# ---------------------------------------------------------------------------
# Policy param helpers
# ---------------------------------------------------------------------------


def _build_policy_params(t: dict[str, Any]) -> dict[str, Any]:
    """
    Build the policy_params dict for train_rl from a training combo.

    Policy-specific hyperparams can be specified in two ways in a grid config:
      1. Nested:   policy_params: {epsilon: 0.5}  (passes through as-is)
      2. Top-level: epsilon: 0.5                  (promoted via _POLICY_PARAM_MAP)

    Top-level keys take precedence so that grid search axes like
    `epsilon: [0.5, 1.0]` work without nesting inside policy_params.
    """
    params = dict(t.get("policy_params") or {})
    for tkey, pkey in _POLICY_PARAM_MAP.items():
        if tkey in t:
            params[pkey] = t[tkey]
    return params


def _build_tmnf_extras(
    weights_file: str,
    n_lidar_rays: int,
    re_initialize: bool,
    policy_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Build (extra_policy_types, extra_loop_dispatch) for TMNF-specific policy
    types so train_rl() can construct and route them. Mirrors main.py's setup
    so `policy_type: cmaes | neural_dqn | reinforce | lstm` works from grid
    search and distributed workers, not just from main.py.
    """

    def _make_neural_dqn() -> NeuralDQNPolicy:
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as _f:
                _cfg = yaml.safe_load(_f)
            if isinstance(_cfg, dict) and _cfg.get("policy_type") == "neural_dqn":
                return NeuralDQNPolicy.from_cfg(_cfg, n_lidar_rays=n_lidar_rays)
        return NeuralDQNPolicy(
            hidden_sizes=policy_params.get("hidden_sizes", [64, 64]),
            replay_buffer_size=policy_params.get("replay_buffer_size", 10000),
            batch_size=policy_params.get("batch_size", 64),
            min_replay_size=policy_params.get("min_replay_size", 500),
            target_update_freq=policy_params.get("target_update_freq", 200),
            learning_rate=policy_params.get("learning_rate", 0.001),
            epsilon_start=policy_params.get("epsilon_start", 1.0),
            epsilon_end=policy_params.get("epsilon_end", 0.05),
            epsilon_decay_steps=policy_params.get("epsilon_decay_steps", 5000),
            gamma=policy_params.get("gamma", 0.99),
            n_lidar_rays=n_lidar_rays,
        )

    def _make_cmaes() -> CMAESPolicy:
        policy = CMAESPolicy(
            population_size=policy_params.get("population_size", 20),
            initial_sigma=policy_params.get("initial_sigma", 0.3),
            n_lidar_rays=n_lidar_rays,
        )
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as _f:
                champion = WeightedLinearPolicy.from_cfg(
                    yaml.safe_load(_f) or {}, n_lidar_rays=n_lidar_rays
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
            hidden_sizes=policy_params.get("hidden_sizes", [64, 64]),
            learning_rate=policy_params.get("learning_rate", 0.001),
            gamma=policy_params.get("gamma", 0.99),
            entropy_coeff=policy_params.get("entropy_coeff", 0.01),
            baseline=policy_params.get("baseline", "running_mean"),
            n_lidar_rays=n_lidar_rays,
        )

    def _make_lstm() -> LSTMEvolutionPolicy:
        hidden_size = policy_params.get("hidden_size", 32)
        policy = LSTMEvolutionPolicy(
            hidden_size=hidden_size,
            population_size=policy_params.get("population_size", 20),
            initial_sigma=policy_params.get("initial_sigma", 0.05),
            n_lidar_rays=n_lidar_rays,
        )
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as _f:
                _cfg = yaml.safe_load(_f) or {}
            if isinstance(_cfg, dict) and _cfg.get("policy_type") == "lstm":
                saved_hidden = _cfg.get("hidden_size")
                saved_lidar = _cfg.get("n_lidar_rays")
                if saved_hidden is not None and saved_hidden != hidden_size:
                    raise ValueError(
                        "Saved LSTM champion hidden_size does not match current run: "
                        f"saved={saved_hidden}, current={hidden_size}"
                    )
                if saved_lidar is not None and saved_lidar != n_lidar_rays:
                    raise ValueError(
                        "Saved LSTM champion n_lidar_rays does not match current run: "
                        f"saved={saved_lidar}, current={n_lidar_rays}"
                    )
                policy.initialize_from_champion(LSTMPolicy.from_cfg(_cfg))
        return policy

    extras = {
        "neural_dqn": _make_neural_dqn,
        "cmaes": _make_cmaes,
        "reinforce": _make_reinforce,
        "lstm": _make_lstm,
    }
    dispatch = {
        "neural_dqn": "q_learning",
        "cmaes": "cmaes",
        "reinforce": "q_learning",
        "lstm": "cmaes",
    }
    return extras, dispatch


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _setup_experiment_dir(
    name: str, track: str, t: dict[str, Any], r: dict[str, Any]
) -> tuple[str, str, str]:
    """Create experiment dir, write config files. Returns (experiment_dir, weights_file, reward_cfg_file)."""
    centerline_path = f"games/tmnf/tracks/{track}.npy"
    experiment_dir = f"games/tmnf/experiments/{track}/{name}"
    weights_file = f"{experiment_dir}/policy_weights.yaml"
    reward_cfg_file = f"{experiment_dir}/reward_config.yaml"
    training_params_file = f"{experiment_dir}/training_params.yaml"

    os.makedirs(experiment_dir, exist_ok=True)

    with open("games/tmnf/config/reward_config.yaml") as f:
        reward_cfg = yaml.safe_load(f) or {}
    reward_cfg.update(r)
    reward_cfg["track_name"] = track
    reward_cfg["centerline_path"] = centerline_path
    with open(reward_cfg_file, "w") as f:
        yaml.dump(reward_cfg, f, default_flow_style=False, sort_keys=False)
    with open(training_params_file, "w") as f:
        yaml.dump(t, f, default_flow_style=False, sort_keys=False)

    return experiment_dir, weights_file, reward_cfg_file


def _run_local(
    combos: list[dict[str, Any]],
    names: list[str],
    track: str,
    no_interrupt: bool,
    re_initialize: bool,
) -> list[tuple[str, Any]]:
    """Run all combos sequentially on this machine. Returns list of (name, ExperimentData)."""

    all_runs = []
    n = len(combos)
    for i, (combo, name) in enumerate(zip(combos, names), 1):
        t = combo["training_params"]
        r = combo["reward_params"]
        logger.info("=== Run %d/%d: %s ===", i, n, name)

        experiment_dir, weights_file, reward_cfg_file = _setup_experiment_dir(
            name, track, t, r
        )
        n_lidar_rays = t.get("n_lidar_rays", 0)
        obs_spec = TMNF_OBS_SPEC.with_lidar(n_lidar_rays)
        policy_params = _build_policy_params(t)
        extras, dispatch = _build_tmnf_extras(
            weights_file=weights_file,
            n_lidar_rays=n_lidar_rays,
            re_initialize=re_initialize,
            policy_params=policy_params,
        )

        data = train_rl(
            experiment_name=name,
            make_env_fn=lambda _dir=experiment_dir, _sp=t["speed"], _ep=t[
                "in_game_episode_s"
            ], _lr=n_lidar_rays: make_env(
                experiment_dir=_dir,
                speed=_sp,
                in_game_episode_s=_ep,
                n_lidar_rays=_lr,
            ),
            obs_spec=obs_spec,
            head_names=["steer", "accel", "brake"],
            discrete_actions=DISCRETE_ACTIONS,
            speed=t["speed"],
            n_sims=t["n_sims"],
            in_game_episode_s=t["in_game_episode_s"],
            weights_file=weights_file,
            reward_config_file=reward_cfg_file,
            mutation_scale=t.get("mutation_scale", 0.05),
            mutation_share=t.get("mutation_share", 1.0),
            probe_actions=PROBE_ACTIONS,
            probe_in_game_s=t.get("probe_s", 0),
            cold_start_restarts=t.get("cold_restarts", 0),
            cold_start_sims=t.get("cold_sims", 0),
            warmup_action=WARMUP_ACTION,
            warmup_steps=5,
            training_params=t,
            no_interrupt=no_interrupt or i > 1,
            re_initialize=re_initialize,
            policy_type=t.get("policy_type", "hill_climbing"),
            policy_params=policy_params,
            extra_policy_types=extras,
            extra_loop_dispatch=dispatch,
            track=track,
            do_pretrain=t.get("do_pretrain", False),
            patience=t.get("patience", 0),
        )

        save_experiment_results(data, results_dir=f"{experiment_dir}/results")
        all_runs.append((name, data))
        best = max((s.reward for s in data.greedy_sims), default=float("-inf"))
        logger.info("[%d/%d] %s  best_reward=%+.1f", i, n, name, best)
        sleep(10)  # brief pause between runs to avoid overwhelming the system

    return all_runs


def _run_distributed(
    combos: list[dict[str, Any]],
    names: list[str],
    track: str,
    token: str,
    port: int,
    heartbeat_timeout: float,
) -> list[tuple[str, Any]]:
    """Start coordinator, write local config files, block until all results arrive."""

    # Build ComboSpec list and write local experiment dirs so reward_config_file
    # resolves on this machine when save_grid_summary reads it.
    combo_specs = []
    for combo, name in zip(combos, names):
        t = combo["training_params"]
        r = combo["reward_params"]
        _setup_experiment_dir(name, track, t, r)
        combo_specs.append(
            ComboSpec(name=name, track=track, training_params=t, reward_params=r)
        )

    coord = Coordinator(
        combo_specs, token=token, port=port, heartbeat_timeout=heartbeat_timeout
    )
    coord.start()
    logger.info(
        "Coordinator ready on port %d — start workers with:\n"
        "  python -m distributed.worker --coordinator http://<this-host>:%d --token <token>",
        port,
        port,
    )

    raw_runs = coord.wait_for_all()
    coord.stop()

    # Override reward_config_file to the local path written above, then save results.
    all_runs = []
    for name, data in raw_runs:
        experiment_dir = f"experiments/{track}/{name}"
        data.reward_config_file = f"{experiment_dir}/reward_config.yaml"
        data.weights_file = f"{experiment_dir}/policy_weights.yaml"
        save_experiment_results(data, results_dir=f"{experiment_dir}/results")
        all_runs.append((name, data))

    return all_runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search over TMNF training/reward params"
    )
    parser.add_argument("config", help="Path to grid search YAML config")
    parser.add_argument(
        "--no-interrupt",
        action="store_true",
        help="Skip all 'Press Enter' prompts (run fully automated)",
    )
    parser.add_argument(
        "--re-initialize",
        action="store_true",
        help="Start each run from fresh random small-positive weights, "
        "ignoring any existing weights file. Skips probe and cold-start.",
    )
    parser.add_argument(
        "--distribute",
        action="store_true",
        help="Act as coordinator: serve work items over HTTP and wait for "
        "workers to post results instead of running locally",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Coordinator HTTP port when --distribute is set (default: 5555, "
        "or value from config distribute.port)",
    )
    parser.add_argument(
        "--token",
        default=None,
        metavar="SECRET",
        help="Shared secret for worker authentication; falls back to "
        "TMNF_GRID_TOKEN env var (auto-generated UUID if neither provided)",
    )
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=None,
        help="Seconds before a silent worker's item is re-queued "
        "(default: 60, or value from config distribute.heartbeat_timeout)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    base_name, track, training_spec, reward_spec, distribute_cfg = _load_grid_config(
        args.config
    )
    combos, varied_keys = _expand_grid(training_spec, reward_spec)

    n = len(combos)
    logger.info("  Grid search:       %d combination(s)", n)
    logger.info("  Base name:         %s", base_name)
    logger.info("  Track:             %s", track)
    logger.info("  Training config:")
    for k, v in training_spec.items():
        logger.info("    %s: %s", k, v)
    logger.info("  Reward config:")
    for k, v in reward_spec.items():
        logger.info("    %s: %s", k, v)
    if args.distribute:
        logger.info("  Distribute config: %s", distribute_cfg)
        for k, v in distribute_cfg.items():
            logger.info("    %s: %s", k, v)
    if varied_keys:
        logger.info("  Varied:      %s", ", ".join(varied_keys))
    logger.info("%s", "=" * 60)

    names = []
    for c in combos:
        name = _make_experiment_name(base_name, c.get("_flat", {}), varied_keys)
        names.append(name)
        logger.info("  %s", name)

    if args.distribute:
        port = args.port or distribute_cfg.get("port", 5555)
        hb_timeout = args.heartbeat_timeout or distribute_cfg.get(
            "heartbeat_timeout", 60.0
        )
        token = args.token or os.environ.get("TMNF_GRID_TOKEN") or str(_uuid.uuid4())
        if not (args.token or os.environ.get("TMNF_GRID_TOKEN")):
            token_preview = f"{token[:8]}..." if len(token) > 8 else "[redacted]"
            logger.info(
                "Auto-generated token for distributed run (%s). Pass it to workers via --token or TMNF_GRID_TOKEN.",
                token_preview,
            )
        all_runs = _run_distributed(
            combos, names, track, token=token, port=port, heartbeat_timeout=hb_timeout
        )
    else:
        all_runs = _run_local(
            combos,
            names,
            track,
            no_interrupt=args.no_interrupt,
            re_initialize=args.re_initialize,
        )

    # Final summary table
    logger.info("=== Grid search complete — %d run(s) ===", n)
    logger.info("  %-50s  %12s", "Experiment", "Best Reward")
    for exp_name, exp_data in sorted(
        all_runs,
        key=lambda x: -max((s.reward for s in x[1].greedy_sims), default=float("-inf")),
    ):
        best = max((s.reward for s in exp_data.greedy_sims), default=float("-inf"))
        logger.info("  %-50s  %+12.1f", exp_name, best)

    # Cross-experiment summary report

    summary_dir = f"games/tmnf/experiments/{track}/{base_name}__summary" # TODO needs to be game adapted
    save_grid_summary(all_runs, varied_keys, summary_dir, base_name)
    logger.info("Summary report: %s/summary.md", summary_dir)


if __name__ == "__main__":
    main()
