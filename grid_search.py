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
from typing import Any

import yaml
from analytics import save_experiment_results, save_grid_summary
from framework.training import train_rl
from games.tmnf.obs_spec import TMNF_OBS_SPEC
from games.tmnf.actions import DISCRETE_ACTIONS, PROBE_ACTIONS, WARMUP_ACTION
from games.tmnf.env import make_env

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Param name → short abbreviation for experiment directory names
# ---------------------------------------------------------------------------

_ABBREV = {
    # training params
    "speed":           "speed",
    "n_sims":          "nsims",
    "in_game_episode_s": "ep",
    "mutation_scale":  "ms",
    "mutation_share":  "mshare",
    "probe_s":         "probe",
    "cold_restarts":   "cr",
    "cold_sims":       "cs",
    "policy_type":     "pt",
    "do_pretrain":     "dpt",
    # neural_net policy params
    "hidden_sizes":    "hs",
    # genetic policy params
    "population_size": "pop",
    "elite_k":         "ek",
    # epsilon-greedy params
    "epsilon":         "eps",
    "epsilon_decay":   "ed",
    "epsilon_min":     "emin",
    "alpha":           "alpha",
    "gamma":           "gamma",
    "n_bins":          "bins",
    # mcts params
    "mcts_c":          "mc",
    # reward params
    "progress_weight": "pw",
    "centerline_weight": "cw",
    "centerline_exp":  "ce",
    "speed_weight":    "sw",
    "step_penalty":    "sp",
    "finish_bonus":    "fb",
    "finish_time_weight": "ftw",
    "par_time_s":      "par",
    "accel_bonus":     "ab",
    "airborne_penalty": "ap",
    "crash_threshold_m": "ct",
    "lidar_wall_weight": "lww",
}

# Top-level training_params keys that should be forwarded into policy_params.
# Allows grid axes like `epsilon: [0.5, 1.0]` without nesting inside policy_params.
# mcts_c is renamed to c because that's what MCTSPolicy.from_cfg expects.
_POLICY_PARAM_MAP = {
    "hidden_sizes":    "hidden_sizes",   # neural_net
    "epsilon":         "epsilon",        # epsilon_greedy
    "epsilon_decay":   "epsilon_decay",  # epsilon_greedy
    "epsilon_min":     "epsilon_min",    # epsilon_greedy
    "alpha":           "alpha",          # epsilon_greedy / mcts
    "gamma":           "gamma",          # epsilon_greedy / mcts
    "n_bins":          "n_bins",         # epsilon_greedy / mcts
    "mcts_c":          "c",              # mcts (renamed)
    "population_size": "population_size", # genetic
    "elite_k":         "elite_k",        # genetic
}


def _fmt_value(v: Any) -> str:
    """Format a param value for use in a directory name.

    - Integers: plain digits (50 → '50')
    - Floats: strip trailing zeros; replace leading '-' with 'n' (−0.1 → 'n0.1')
    - Others: str()
    """
    if isinstance(v, float):
        s = f"{v:g}"          # e.g. '10', '-0.1', '0.05'
        s = s.replace("-", "n")
        return s
    if isinstance(v, int):
        return str(v)
    return str(v).replace("-", "n")


def _make_experiment_name(base_name: str, combo: dict[str, Any], varied_keys: list[str]) -> str:
    """Build experiment name from base + only the varied param values."""
    parts = [base_name]
    for key in varied_keys:
        abbrev = _ABBREV.get(key, key)
        parts.append(f"{abbrev}{_fmt_value(combo[key])}")
    return "__".join(parts)


# ---------------------------------------------------------------------------
# Config loading and grid expansion
# ---------------------------------------------------------------------------

def _load_grid_config(path: str) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    """Load grid config YAML. Returns (base_name, track, training_spec, reward_spec)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base_name = cfg.get("base_name", "gs")
    track = cfg.get("track", "a03_centerline")
    training_spec = cfg.get("training_params", {})
    reward_spec = cfg.get("reward_params", {})
    return base_name, track, training_spec, reward_spec


def _expand_grid(training_spec: dict[str, Any], reward_spec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
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
        return [{"training_params": dict(training_spec), "reward_params": dict(reward_spec)}], []

    varied_keys = [a[0] for a in axes]
    value_lists = [a[1] for a in axes]
    sources     = [a[2] for a in axes]

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
        combos.append({"training_params": t_params, "reward_params": r_params, "_flat": flat})

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search over TMNF training/reward params")
    parser.add_argument("config", help="Path to grid search YAML config")
    parser.add_argument("--no-interrupt", action="store_true",
                        help="Skip all 'Press Enter' prompts (run fully automated)")
    parser.add_argument("--re-initialize", action="store_true",
                        help="Start each run from fresh random small-positive weights, "
                             "ignoring any existing weights file. Skips probe and cold-start.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    base_name, track, training_spec, reward_spec = _load_grid_config(args.config)
    combos, varied_keys = _expand_grid(training_spec, reward_spec)
    centerline_path = f"tracks/{track}.npy"

    n = len(combos)
    logger.info(f"  Grid search: {n} combination(s)")
    logger.info(f"  Base name:   {base_name}")
    logger.info(f"  Track:       {track}")
    if varied_keys:
        logger.info(f"  Varied:      {', '.join(varied_keys)}")
    if varied_keys:
        logger.info("  Varied:      %s", ', '.join(varied_keys))
    logger.info("%s", "="*60)

    # Log all experiment names upfront so the user knows what's coming
    names = []
    for c in combos:
        name = _make_experiment_name(base_name, c["_flat"], varied_keys)
        names.append(name)
        logger.info("  %s", name)


    all_runs = []   # list of (name, ExperimentData) for the summary

    for i, (combo, name) in enumerate(zip(combos, names), 1):
        t = combo["training_params"]
        r = combo["reward_params"]

        logger.info("=== Run %d/%d: %s ===", i, n, name)

        experiment_dir = f"experiments/{track}/{name}"
        weights_file   = f"{experiment_dir}/policy_weights.yaml"
        reward_cfg_file = f"{experiment_dir}/reward_config.yaml"
        training_params_file = f"{experiment_dir}/training_params.yaml"

        os.makedirs(experiment_dir, exist_ok=True)

        # Write reward config: master defaults merged with combo overrides.
        # This ensures params not listed in the grid config (e.g. lidar_wall_weight)
        # still get their master-config values rather than silently defaulting.
        with open("config/reward_config.yaml") as f:
            reward_cfg = yaml.safe_load(f) or {}
        reward_cfg.update(r)
        reward_cfg["track_name"] = track
        reward_cfg["centerline_path"] = centerline_path
        with open(reward_cfg_file, "w") as f:
            yaml.dump(reward_cfg, f, default_flow_style=False, sort_keys=False)
        with open(training_params_file, "w") as f:
            yaml.dump(t, f, default_flow_style=False, sort_keys=False)

        n_lidar_rays = t.get("n_lidar_rays", 0)
        obs_spec     = TMNF_OBS_SPEC.with_lidar(n_lidar_rays)

        data = train_rl(
            experiment_name     = name,
            make_env_fn         = lambda _dir=experiment_dir, _sp=t["speed"], _ep=t["in_game_episode_s"], _lr=n_lidar_rays: make_env(
                experiment_dir    = _dir,
                speed             = _sp,
                in_game_episode_s = _ep,
                n_lidar_rays      = _lr,
            ),
            obs_spec            = obs_spec,
            head_names          = ["steer", "accel", "brake"],
            discrete_actions    = DISCRETE_ACTIONS,
            speed               = t["speed"],
            n_sims              = t["n_sims"],
            in_game_episode_s   = t["in_game_episode_s"],
            weights_file        = weights_file,
            reward_config_file  = reward_cfg_file,
            mutation_scale      = t["mutation_scale"],
            mutation_share      = t.get("mutation_share", 1.0),
            probe_actions       = PROBE_ACTIONS,
            probe_in_game_s     = t.get("probe_s", 0),
            cold_start_restarts = t.get("cold_restarts", 0),
            cold_start_sims     = t.get("cold_sims", 0),
            warmup_action       = WARMUP_ACTION,
            warmup_steps        = 100,
            training_params     = t,
            no_interrupt        = args.no_interrupt or i > 1,
            re_initialize       = args.re_initialize,
            policy_type         = t.get("policy_type", "hill_climbing"),
            policy_params       = _build_policy_params(t),
            track               = track,
            do_pretrain         = t.get("do_pretrain", False),
        )

        save_experiment_results(data, results_dir=f"{experiment_dir}/results")
        all_runs.append((name, data))

        best = max((s.reward for s in data.greedy_sims), default=float("-inf"))
        logger.info("[%d/%d] %s  best_reward=%+.1f", i, n, name, best)

    # Final summary table
    logger.info("=== Grid search complete — %d run(s) ===", n)
    logger.info("  %-50s  %12s", "Experiment", "Best Reward")
    for exp_name, exp_data in sorted(
        all_runs, key=lambda x: -max((s.reward for s in x[1].greedy_sims), default=float("-inf"))
    ):
        best = max((s.reward for s in exp_data.greedy_sims), default=float("-inf"))
        logger.info("  %-50s  %+12.1f", exp_name, best)

    # Cross-experiment summary report
    summary_dir = f"experiments/{track}/{base_name}__summary"
    save_grid_summary(all_runs, varied_keys, summary_dir, base_name)
    logger.info("Summary report: %s/summary.md", summary_dir)


if __name__ == "__main__":
    main()
