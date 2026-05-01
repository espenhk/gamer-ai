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

from framework.game_adapter import GAME_ADAPTERS
from framework.run_config import RunConfig
from framework.training import train_rl

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
    "decision_offset_pct": "dec",
    "action_window_ticks": "awt",
    # neural_net policy params
    "hidden_sizes": "hs",
    "hidden_size": "hsize",
    "learning_rate": "lr",
    "entropy_coeff": "ec",
    "baseline": "base",
    # neural_dqn params
    "batch_size": "bs",
    "target_update_freq": "tuf",
    "epsilon_decay_steps": "eds",
    # genetic policy params
    "population_size": "pop",
    "elite_k": "ek",
    # cmaes policy params
    "initial_sigma": "sigma",
    # shared genetic/cmaes policy params
    "eval_episodes": "evep",
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
    # curiosity params (issue #24)
    "curiosity_type": "ck",
    "curiosity_weight": "cwgt",
    "curiosity_feature_dim": "cfd",
    "curiosity_hidden_size": "chs",
    "curiosity_lr": "clr",
    "curiosity_beta": "cbeta",
}

# Top-level training_params keys that should be forwarded into policy_params.
# Allows grid axes like `epsilon: [0.5, 1.0]` without nesting inside policy_params.
# mcts_c is renamed to c because that's what MCTSPolicy.from_cfg expects.
_POLICY_PARAM_MAP = {
    "hidden_sizes": "hidden_sizes",  # neural_net / reinforce / neural_dqn
    "hidden_size": "hidden_size",  # lstm
    "learning_rate": "learning_rate",  # reinforce / neural_dqn
    "entropy_coeff": "entropy_coeff",  # reinforce
    "baseline": "baseline",  # reinforce
    "batch_size": "batch_size",  # neural_dqn
    "target_update_freq": "target_update_freq",  # neural_dqn
    "epsilon_decay_steps": "epsilon_decay_steps",  # neural_dqn
    "epsilon": "epsilon",  # epsilon_greedy
    "epsilon_decay": "epsilon_decay",  # epsilon_greedy
    "epsilon_min": "epsilon_min",  # epsilon_greedy
    "alpha": "alpha",  # epsilon_greedy / mcts
    "gamma": "gamma",  # epsilon_greedy / mcts / neural_dqn / reinforce
    "n_bins": "n_bins",  # epsilon_greedy / mcts
    "mcts_c": "c",  # mcts (renamed)
    "population_size": "population_size",  # genetic / cmaes / lstm
    "elite_k": "elite_k",  # genetic
    "initial_sigma": "initial_sigma",  # cmaes
    "eval_episodes": "eval_episodes",  # genetic / cmaes
}

# Guard against silent regressions when adding or editing promoted top-level
# training_params keys. If one of these entries is removed or renamed
# incorrectly, grid configs can silently fall back to policy defaults.
_EXPECTED_POLICY_PARAM_MAP = {
    "hidden_sizes": "hidden_sizes",
    "hidden_size": "hidden_size",
    "learning_rate": "learning_rate",
    "entropy_coeff": "entropy_coeff",
    "baseline": "baseline",
    "batch_size": "batch_size",
    "target_update_freq": "target_update_freq",
    "epsilon_decay_steps": "epsilon_decay_steps",
    "epsilon": "epsilon",
    "epsilon_decay": "epsilon_decay",
    "epsilon_min": "epsilon_min",
    "alpha": "alpha",
    "gamma": "gamma",
    "n_bins": "n_bins",
    "mcts_c": "c",
    "population_size": "population_size",
    "elite_k": "elite_k",
    "initial_sigma": "initial_sigma",
}


def _validate_policy_param_map() -> None:
    """Fail fast if promoted training_params keys are miswired.

    This provides a lightweight regression check in environments where the
    pure-logic promotion path is imported without the full training stack.
    """
    mismatches = {
        src: (expected_dst, _POLICY_PARAM_MAP.get(src))
        for src, expected_dst in _EXPECTED_POLICY_PARAM_MAP.items()
        if _POLICY_PARAM_MAP.get(src) != expected_dst
    }
    if mismatches:
        details = ", ".join(
            f"{src}->{actual!r} (expected {expected!r})"
            for src, (expected, actual) in sorted(mismatches.items())
        )
        raise RuntimeError(f"Invalid _POLICY_PARAM_MAP entries: {details}")


_validate_policy_param_map()
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
) -> tuple[str, str, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load grid config YAML. Returns (base_name, game, track, training_spec, reward_spec, distribute_cfg)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base_name = cfg.get("base_name", "gs")
    game = cfg.get("game", "tmnf")
    track = cfg.get("track", None)
    training_spec = cfg.get("training_params", {})
    reward_spec = cfg.get("reward_params", {})
    distribute_cfg = cfg.get("distribute", {})
    return base_name, game, track, training_spec, reward_spec, distribute_cfg


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



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _setup_experiment_dir(
    adapter, name: str, t: dict[str, Any], r: dict[str, Any],
    track_override: str | None,
) -> tuple[str, str, str]:
    """Create experiment dir, write config files. Returns (experiment_dir, weights_file, reward_cfg_file)."""
    experiment_dir = adapter.experiment_dir(name, t, track_override)
    weights_file = f"{experiment_dir}/policy_weights.yaml"
    reward_cfg_file = f"{experiment_dir}/reward_config.yaml"
    training_params_file = f"{experiment_dir}/training_params.yaml"

    os.makedirs(experiment_dir, exist_ok=True)

    reward_master = os.path.join(adapter.config_dir, "reward_config.yaml")
    with open(reward_master) as f:
        reward_cfg = yaml.safe_load(f) or {}
    reward_cfg.update(r)
    adapter.decorate_reward_cfg(reward_cfg, t, track_override)
    with open(reward_cfg_file, "w") as f:
        yaml.dump(reward_cfg, f, default_flow_style=False, sort_keys=False)
    with open(training_params_file, "w") as f:
        yaml.dump(t, f, default_flow_style=False, sort_keys=False)

    return experiment_dir, weights_file, reward_cfg_file


def _run_local(
    adapter,
    combos: list[dict[str, Any]],
    names: list[str],
    track_override: str | None,
    no_interrupt: bool,
    re_initialize: bool,
) -> list[tuple[str, Any]]:
    """Run all combos sequentially on this machine. Returns list of (name, ExperimentData)."""
    from framework.analytics import save_experiment_data_json

    all_runs = []
    n = len(combos)
    for i, (combo, name) in enumerate(zip(combos, names), 1):
        t = combo["training_params"]
        r = combo["reward_params"]
        logger.info("=== Run %d/%d: %s ===", i, n, name)

        experiment_dir, weights_file, reward_cfg_file = _setup_experiment_dir(
            adapter, name, t, r, track_override
        )

        # Merge promoted policy params so grid axes like `epsilon: [0.5, 1.0]` work.
        t_with_pp = dict(t)
        t_with_pp["policy_params"] = _build_policy_params(t)

        game_spec = adapter.build_game_spec(
            name, experiment_dir, weights_file, reward_cfg_file, t_with_pp, track_override,
        )
        data = train_rl(
            game=game_spec,
            config=RunConfig.from_training_params(t_with_pp),
            probe=adapter.build_probe(t_with_pp),
            warmup=adapter.build_warmup(t_with_pp),
            extras=adapter.build_extras(weights_file, t_with_pp, re_initialize),
            no_interrupt=no_interrupt or i > 1,
            re_initialize=re_initialize,
        )

        if game_spec.save_results_fn is not None:
            game_spec.save_results_fn(data, results_dir=f"{experiment_dir}/results")
        save_experiment_data_json(data, results_dir=f"{experiment_dir}/results")
        all_runs.append((name, data))
        best = max((s.reward for s in data.greedy_sims), default=float("-inf"))
        logger.info("[%d/%d] %s  best_reward=%+.1f", i, n, name, best)
        sleep(10)  # brief pause between runs to avoid overwhelming the system

    return all_runs


def _run_distributed(
    adapter,
    combos: list[dict[str, Any]],
    names: list[str],
    track_override: str | None,
    token: str,
    port: int,
    heartbeat_timeout: float,
    game_name: str,
) -> list[tuple[str, Any]]:
    """Start coordinator, write local config files, block until all results arrive."""

    # Build ComboSpec list and write local experiment dirs so reward_config_file
    # resolves on this machine when save_grid_summary reads it.
    combo_specs = []
    for combo, name in zip(combos, names):
        t = combo["training_params"]
        r = combo["reward_params"]
        _setup_experiment_dir(adapter, name, t, r, track_override)
        track = adapter.track_label(t, track_override)
        combo_specs.append(
            ComboSpec(name=name, track=track, training_params=t,
                      reward_params=r, game=game_name)
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
        experiment_dir = adapter.experiment_dir(name, data.training_params, track_override)
        data.reward_config_file = f"{experiment_dir}/reward_config.yaml"
        data.weights_file = f"{experiment_dir}/policy_weights.yaml"
        if hasattr(adapter, 'build_game_spec'):
            # Use adapter's save_results_fn via build_game_spec
            game_spec = adapter.build_game_spec(
                name, experiment_dir, data.weights_file, data.reward_config_file,
                data.training_params, track_override,
            )
            if game_spec.save_results_fn is not None:
                game_spec.save_results_fn(data, results_dir=f"{experiment_dir}/results")
        from framework.analytics import save_experiment_data_json
        save_experiment_data_json(data, results_dir=f"{experiment_dir}/results")
        all_runs.append((name, data))

    return all_runs


def _consolidate(
    experiment_dirs: list[str],
    summary_name: str,
    summary_dir: str | None,
) -> None:
    """Load experiment data from *experiment_dirs* and produce a combined summary."""
    from framework.analytics import load_experiment_data

    all_runs: list[tuple[str, Any]] = []
    for d in experiment_dirs:
        try:
            data = load_experiment_data(d)
        except FileNotFoundError:
            logger.error(
                "No experiment_data.json found in %s/results/ — skipping. "
                "(Was this experiment run with a version that saves experiment_data.json?)",
                d,
            )
            continue
        all_runs.append((data.experiment_name, data))
        best = max((s.reward for s in data.greedy_sims), default=float("-inf"))
        logger.info("  Loaded %-50s  best_reward=%+.1f", data.experiment_name, best)

    if not all_runs:
        logger.error("No experiment data loaded — nothing to consolidate.")
        return

    # Infer varied keys: collect all training_params keys whose values differ
    # across runs.
    all_keys: set[str] = set()
    for _, data in all_runs:
        all_keys.update(data.training_params.keys())
    varied_keys: list[str] = []
    for k in sorted(all_keys):
        values = [data.training_params.get(k) for _, data in all_runs]
        if len(set(str(v) for v in values)) > 1:
            varied_keys.append(k)

    # Determine summary output dir
    if summary_dir is None:
        # Place summary next to the experiment dirs (common parent)
        parent = os.path.commonpath(experiment_dirs)
        summary_dir = os.path.join(parent, f"{summary_name}__summary")

    logger.info(
        "Consolidating %d experiment(s) into %s", len(all_runs), summary_dir
    )
    from framework.analytics import save_grid_summary
    save_grid_summary(all_runs, varied_keys, summary_dir, summary_name)
    logger.info("Summary report: %s/summary.md", summary_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search over training/reward params (multi-game)"
    )
    parser.add_argument("config", nargs="?", default=None,
                        help="Path to grid search YAML config (not needed with --consolidate)")
    parser.add_argument(
        "--game",
        default=None,
        choices=["tmnf", "beamng", "car_racing", "torcs", "sc2"],
        help="Override game (default: from YAML 'game:' field, or tmnf)",
    )
    parser.add_argument(
        "--track",
        default=None,
        help="Override the track / map name from the config.",
    )
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
        "--consolidate",
        nargs="+",
        metavar="DIR",
        help="Consolidate previous grid-search experiment folders into one "
        "summary. Each DIR is a path to an experiment directory containing "
        "results/experiment_data.json. Example: "
        "python grid_search.py --consolidate experiments/a03/gs__ms0.05 "
        "experiments/a03/gs__ms0.1 --summary-name my_summary",
    )
    parser.add_argument(
        "--summary-name",
        default="consolidated",
        help="Base name for the consolidated summary (default: 'consolidated'). "
        "Used with --consolidate.",
    )
    parser.add_argument(
        "--summary-dir",
        default=None,
        help="Output directory for the consolidated summary report. "
        "If not given, inferred from experiment paths. Used with --consolidate.",
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

    if args.consolidate:
        _consolidate(args.consolidate, args.summary_name, args.summary_dir)
        return

    if args.config is None:
        parser.error("config is required when not using --consolidate")

    base_name, game_name, track_override, training_spec, reward_spec, distribute_cfg = _load_grid_config(
        args.config
    )

    # CLI --game / --track override YAML values
    game_name = getattr(args, "game", None) or game_name
    track_override = getattr(args, "track", None) or track_override

    adapter = GAME_ADAPTERS[game_name]()
    combos, varied_keys = _expand_grid(training_spec, reward_spec)

    n = len(combos)
    logger.info("  Grid search:       %d combination(s)", n)
    logger.info("  Base name:         %s", base_name)
    logger.info("  Game:              %s", game_name)
    logger.info("  Track override:    %s", track_override or "(default)")
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
            adapter, combos, names, track_override,
            token=token, port=port, heartbeat_timeout=hb_timeout,
            game_name=game_name,
        )
    else:
        all_runs = _run_local(
            adapter,
            combos,
            names,
            track_override=track_override,
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
    summary_root = adapter.experiment_dir_root(training_spec, track_override)
    summary_dir = f"{summary_root}/{base_name}__summary"
    # Lazy import — save_grid_summary is game-specific analytics
    try:
        game_spec = adapter.build_game_spec(
            base_name, summary_dir, "", "", training_spec, track_override,
        )
        # Import the game-specific save_grid_summary via its analytics module
        _analytics_mod = __import__(f"games.{game_name}.analytics", fromlist=["save_grid_summary"])
        _analytics_mod.save_grid_summary(all_runs, varied_keys, summary_dir, base_name)
    except (ImportError, AttributeError):
        from framework.analytics import save_grid_summary
        save_grid_summary(all_runs, varied_keys, summary_dir, base_name)
    logger.info("Summary report: %s/summary.md", summary_dir)


if __name__ == "__main__":
    main()
