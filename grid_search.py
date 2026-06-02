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
Results are written to experiments/<game>/<policy>/<map>/<base_name>/<params>/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid as _uuid
from time import sleep
from typing import Any

import yaml

from framework.env_loader import load_dotenv

load_dotenv()

from distributed.coordinator import Coordinator
from distributed.protocol import ComboSpec
from framework.game_adapter import GAME_ADAPTERS
from framework.run_config import RunConfig
from framework.training import train_rl
from framework.version import code_version

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Param name → short abbreviation for experiment directory names
# ---------------------------------------------------------------------------

_ABBREV = {
    # training params
    "track": "trk",
    "track_name": "tn",
    "speed": "speed",
    "game": "gm",
    "n_sims": "nsims",
    "in_game_episode_s": "ep",
    "mutation_scale": "ms",
    "mutation_share": "mshare",
    "adaptive_mutation": "am",
    "probe_s": "probe",
    "cold_restarts": "cr",
    "cold_sims": "cs",
    "n_lidar_rays": "nlr",
    "n_vision": "nv",
    "policy_type": "pt",
    "policy_params": "pp",
    "do_pretrain": "dpt",
    "patience": "pat",
    "decision_offset_pct": "dec",
    "action_window_ticks": "awt",
    "map_name": "map",
    "agent_race": "ar",
    "bot_difficulty": "bd",
    "step_mul": "stm",
    "screen_size": "ss",
    "screen_layers": "scl",
    "minimap_size": "mms",
    "minimap_layers": "mml",
    "obs_spec_preset": "osp",
    "enable_belief": "eb",
    "self_play": "slp",
    "max_apm": "apm",
    "apm_burst_s": "apmb",
    "log_stats_every_n_sims": "lsens",
    "initial_extreme_random_fraction": "ierf",
    "initial_extreme_random_runs": "ierr",
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
    # neural_dqn upgrade params (v0.3.1)
    "double_dqn": "ddqn",
    "dueling": "duel",
    "huber_loss": "hub",
    "huber_kappa": "hkap",
    "max_grad_norm": "mgn",
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
    "native_reward_scale": "nrs",
    "centerline_weight": "cw",
    "centerline_exp": "ce",
    "centerline_path": "cp",
    "score_weight": "scw",
    "speed_weight": "sw",
    "step_penalty": "sp",
    "finish_bonus": "fb",
    "finish_time_weight": "ftw",
    "par_time_s": "par",
    "accel_bonus": "ab",
    "idle_penalty": "ip",
    "idle_worker_penalty": "iwp",
    "idle_bonus": "ib",
    "move_exploration_bonus": "meb",
    "move_exploration_grid_size": "megs",
    "move_exploration_decay_steps": "meds",
    "move_repeat_penalty": "mrp",
    "move_self_penalty": "msp",
    "attack_move_bonus": "amb",
    "click_attack_bonus": "cab",
    "click_attack_cooldown_steps": "cacs",
    "attack_friendly_penalty": "afp",
    "win_bonus": "wb",
    "loss_penalty": "lp",
    "economy_weight": "ew",
    "unit_loss_penalty": "ulp",
    "damage_taken_penalty": "dtp",
    "passive_under_fire_penalty": "pufp",
    "small_selection_bonus": "ssb",
    "attack_bonus": "atb",
    "early_random_action_bonus": "erab",
    "new_action_unlock_bonus": "naub",
    "resource_banking_penalty": "rbp",
    "mineral_banking_threshold": "mbt",
    "gas_banking_threshold": "gbt",
    "airborne_penalty": "ap",
    "crash_threshold_m": "ct",
    "off_track_penalty": "otp",
    "lap_time_improvement_bonus": "ltib",
    "lidar_wall_weight": "lww",
    # rocket_league reward params
    "vel_to_ball_weight": "vtb",
    "boost_weight": "bw",
    "touch_bonus": "tb",
    "goal_weight": "gw",
    "concede_penalty": "cp",
    # rocket_league training params
    "tick_skip": "ts",
    # iracing params
    "action_mode": "actm",
    # atari params
    "clip_sign": "csign",
    # behaviour-cloning params
    "bc_target": "bct",
    "bc_player_id": "bcpid",
    "bc_race": "bcrace",
    "bc_step_mul": "bcstm",
    "bc_epochs": "bcep",
    "bc_learning_rate": "bclr",
    "bc_batch_size": "bcbs",
    "bc_ignore_noop": "bcin",
    # curiosity params (issue #24)
    "curiosity_type": "ck",
    "curiosity_weight": "cwgt",
    "curiosity_feature_dim": "cfd",
    "curiosity_hidden_size": "chs",
    "curiosity_lr": "clr",
    "curiosity_beta": "cbeta",
    "curiosity_seed": "csd",
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
    "double_dqn": "double_dqn",  # neural_dqn (v0.3.1)
    "dueling": "dueling",  # neural_dqn (v0.3.1)
    "huber_loss": "huber_loss",  # neural_dqn (v0.3.1)
    "huber_kappa": "huber_kappa",  # neural_dqn (v0.3.1)
    "max_grad_norm": "max_grad_norm",  # neural_dqn (v0.3.1)
    "epsilon": "epsilon",  # epsilon_greedy
    "epsilon_decay": "epsilon_decay",  # epsilon_greedy
    "epsilon_min": "epsilon_min",  # epsilon_greedy
    "alpha": "alpha",  # epsilon_greedy / mcts
    "gamma": "gamma",  # epsilon_greedy / mcts / neural_dqn / reinforce
    "n_bins": "n_bins",  # epsilon_greedy / mcts
    "mcts_c": "c",  # mcts (renamed)
    "population_size": "population_size",  # genetic / cmaes / lstm
    "elite_k": "elite_k",  # genetic
    "mutation_scale": "mutation_scale",  # genetic (sc2_genetic / genetic)
    "mutation_share": "mutation_share",  # genetic (sc2_genetic / genetic)
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
    "mutation_scale": "mutation_scale",
    "mutation_share": "mutation_share",
    "initial_sigma": "initial_sigma",
    "eval_episodes": "eval_episodes",
    "double_dqn": "double_dqn",
    "dueling": "dueling",
    "huber_loss": "huber_loss",
    "huber_kappa": "huber_kappa",
    "max_grad_norm": "max_grad_norm",
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
            f"{src}->{actual!r} (expected {expected!r})" for src, (expected, actual) in sorted(mismatches.items())
        )
        raise RuntimeError(f"Invalid _POLICY_PARAM_MAP entries: {details}")


_validate_policy_param_map()


# ---------------------------------------------------------------------------
# BC warm-start helpers
# ---------------------------------------------------------------------------

# Maps bc_target → frozenset of compatible training policy_type values.
# sc2_genetic and sc2_cmaes share the SC2MultiHeadLinearPolicy weight format
# and are cross-compatible for warm-starting.
_BC_COMPATIBLE_POLICY_TYPES: dict[str, frozenset[str]] = {
    "sc2_genetic": frozenset({"sc2_genetic", "sc2_cmaes"}),
    "sc2_cmaes": frozenset({"sc2_cmaes", "sc2_genetic"}),
    "sc2_reinforce": frozenset({"sc2_reinforce"}),
    "sc2_neural_net": frozenset({"sc2_neural_net"}),
    "sc2_neural_dqn": frozenset({"sc2_neural_dqn"}),
    "sc2_lstm": frozenset({"sc2_lstm"}),
    "sc2_cnn": frozenset({"sc2_cnn"}),
    "epsilon_greedy": frozenset({"epsilon_greedy"}),
    "ucb_q": frozenset({"ucb_q"}),
}


def _validate_bc_warmstart_combos(
    bc_warmstart_dir: str,
    combos: list[dict[str, Any]],
    names: list[str],
) -> str:
    """Check every combo's policy_type is compatible with the BC target.

    Reads ``bc_summary.json`` from *bc_warmstart_dir*, then for every
    combo checks that ``policy_type`` is in
    ``_BC_COMPATIBLE_POLICY_TYPES[bc_target]``.  All incompatible combos
    are collected and reported in a single ``ValueError`` so the user can
    fix them all at once.

    Returns the ``bc_target`` string from the summary on success.
    """
    summary_path = os.path.join(bc_warmstart_dir, "bc_summary.json")
    if not os.path.exists(summary_path):
        raise ValueError(
            f"No bc_summary.json found in {bc_warmstart_dir!r}. "
            "Cannot verify policy compatibility. "
            "Run BC pre-training first: python main.py <exp> --game sc2 --bc"
        )
    with open(summary_path) as f:
        summary = json.load(f)
    bc_target = summary.get("bc_target")
    if not bc_target:
        raise ValueError(f"bc_summary.json in {bc_warmstart_dir!r} is missing the 'bc_target' field.")
    compatible = _BC_COMPATIBLE_POLICY_TYPES.get(bc_target, frozenset())
    errors = []
    for name, combo in zip(names, combos):
        t = combo["training_params"]
        policy_type = t.get("policy_type", "sc2_genetic")
        if policy_type not in compatible:
            errors.append(f"  {name}: bc_target={bc_target!r} incompatible with policy_type={policy_type!r}")
    if errors:
        raise ValueError(
            f"BC warm-start (target={bc_target!r}) is incompatible with the following "
            "combo(s):\n" + "\n".join(errors) + f"\nCompatible policy_type values: {sorted(compatible)}"
        )
    logger.info(
        "BC warm-start compatibility check passed: target=%s compatible with all %d combo(s).",
        bc_target,
        len(combos),
    )
    return bc_target


def _copy_bc_weights(bc_warmstart_dir: str, experiment_dir: str) -> None:
    """Copy BC-trained weight files from *bc_warmstart_dir* into *experiment_dir*.

    Copies ``policy_weights.yaml`` (evolutionary/gradient targets),
    ``policy_weights.npz`` (``sc2_cnn``), ``trainer_state.npz`` (MLP
    gradient targets such as ``sc2_reinforce``), and
    ``policy_weights_qtable.pkl`` (tabular targets ``epsilon_greedy`` /
    ``ucb_q``) when they exist in *bc_warmstart_dir*.
    """
    copied = []
    for fname in (
        "policy_weights.yaml",
        "policy_weights.npz",
        "trainer_state.npz",
        "policy_weights_qtable.pkl",
    ):
        src = os.path.join(bc_warmstart_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(experiment_dir, fname)
            shutil.copy2(src, dst)
            copied.append(fname)
    if not copied:
        raise FileNotFoundError(
            f"No BC weight files found in {bc_warmstart_dir!r}. "
            "Expected at least policy_weights.yaml or policy_weights.npz."
        )
    logger.debug("BC warm-start: copied %s → %s", copied, experiment_dir)


def _run_inline_bc(
    bc_cfg: dict[str, Any],
    adapter: Any,
    base_name: str,
    training_spec: dict[str, Any],
    track_override: str | None,
    game_name: str,
) -> str:
    """Run BC pre-training from the grid config ``bc:`` section.

    Creates a shared ``<base_name>__bc_warmstart`` experiment directory
    adjacent to the grid experiments.  If that directory already contains
    ``policy_weights.yaml`` (or ``policy_weights.npz`` for ``sc2_cnn``) and
    ``bc_summary.json`` the BC step is skipped,
    so restarting a grid run after an interruption is cheap.

    Returns the path to the BC experiment directory.

    Raises ``ValueError`` if *game_name* is not ``"sc2"`` or ``bc.replay_dir``
    is missing.  Raises ``SystemExit`` when the SC2/PySC2 stack cannot be
    imported.
    """
    if game_name != "sc2":
        raise ValueError(
            f"Inline BC pre-training (grid config 'bc:' section) is only supported for game='sc2', got {game_name!r}."
        )
    try:
        from games.sc2.adapter import _get_obs_spec  # noqa: PLC0415
        from games.sc2.replay_bc import run as bc_run  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(f"Cannot import SC2 BC dependencies: {exc}\nInstall with: poetry install --with sc2") from exc

    replay_dir = bc_cfg.get("replay_dir") or bc_cfg.get("bc_replay_dir")
    if not replay_dir:
        raise ValueError("Grid config 'bc.replay_dir' is required when a 'bc:' section is present.")

    # Build obs_spec from the base training spec (before grid expansion).
    map_name = training_spec.get("map_name", "MoveToBeacon")
    obs_spec_preset = training_spec.get("obs_spec_preset")
    enable_belief = bool(training_spec.get("enable_belief", False))
    obs_spec = _get_obs_spec(map_name, obs_spec_preset, enable_belief)

    # BC experiment dir is a sibling of the per-combo experiment dirs.
    summary_root = adapter.experiment_dir_root(training_spec, track_override)
    bc_dir = os.path.join(summary_root, f"{base_name}__bc_warmstart")
    os.makedirs(bc_dir, exist_ok=True)

    # Skip if already completed.  sc2_cnn saves policy_weights.npz instead of
    # policy_weights.yaml, so accept either file alongside bc_summary.json.
    bc_summary_path = os.path.join(bc_dir, "bc_summary.json")
    weights_yaml = os.path.join(bc_dir, "policy_weights.yaml")
    weights_npz = os.path.join(bc_dir, "policy_weights.npz")
    if os.path.exists(bc_summary_path) and (os.path.exists(weights_yaml) or os.path.exists(weights_npz)):
        logger.info(
            "BC warm-start already exists at %s — skipping re-run. Delete %s to force a fresh BC run.",
            bc_dir,
            bc_dir,
        )
        return bc_dir

    # Resolve BC options; bc_cfg keys take precedence over training_spec fallbacks.
    bc_target = bc_cfg.get("bc_target") or bc_cfg.get("target") or training_spec.get("bc_target", "sc2_reinforce")
    player_id: str | int = bc_cfg.get("player_id") or bc_cfg.get("bc_player_id", "winner")
    if player_id in ("1", "2"):
        player_id = int(player_id)
    race = bc_cfg.get("race") or bc_cfg.get("bc_race", "any")
    max_replays = bc_cfg.get("max_replays") or bc_cfg.get("bc_max_replays")
    step_mul = bc_cfg.get("step_mul") or bc_cfg.get("bc_step_mul") or training_spec.get("step_mul", 1)
    screen_size = bc_cfg.get("screen_size") or training_spec.get("screen_size", 64)
    minimap_size = bc_cfg.get("minimap_size") or training_spec.get("minimap_size", 64)
    bc_epochs = bc_cfg.get("bc_epochs") or bc_cfg.get("epochs", 10)
    bc_lr = bc_cfg.get("bc_learning_rate") or bc_cfg.get("learning_rate", 1e-3)
    bc_batch = bc_cfg.get("bc_batch_size") or bc_cfg.get("batch_size", 256)
    # bc_ignore_noop may legitimately be False, so don't use `or`.
    if "bc_ignore_noop" in bc_cfg:
        bc_ignore_noop = bool(bc_cfg["bc_ignore_noop"])
    elif "ignore_noop" in bc_cfg:
        bc_ignore_noop = bool(bc_cfg["ignore_noop"])
    else:
        bc_ignore_noop = True

    logger.info(
        "=== Inline BC pre-training (target=%s, replay_dir=%s) ===",
        bc_target,
        replay_dir,
    )
    bc_run(
        replay_dir,
        bc_dir,
        obs_spec,
        target=bc_target,
        player_id=player_id,
        race=race,
        max_replays=max_replays,
        step_mul=step_mul,
        screen_size=screen_size,
        minimap_size=minimap_size,
        bc_epochs=bc_epochs,
        bc_learning_rate=bc_lr,
        bc_batch_size=bc_batch,
        bc_ignore_noop=bc_ignore_noop,
        seed=bc_cfg.get("seed"),
        n_channels=bc_cfg.get("n_channels", 1),
        n_bins=bc_cfg.get("n_bins", 3),
        bc_lstm_hidden_size=bc_cfg.get("bc_lstm_hidden_size") or bc_cfg.get("lstm_hidden_size", 64),
        hidden_sizes=bc_cfg.get("hidden_sizes"),
    )
    logger.info("=== Inline BC pre-training complete → %s ===", bc_dir)
    return bc_dir


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


def _make_experiment_name(base_name: str, combo: dict[str, Any], varied_keys: list[str]) -> str:
    """Build experiment name from base + only the varied param values."""
    parts = [base_name]
    for key in varied_keys:
        abbrev = _ABBREV.get(key, key)
        parts.append(f"{abbrev}{_fmt_value(combo[key])}")
    return "__".join(parts)


def _split_grid_run_name(name: str) -> tuple[str, str | None]:
    """Split ``<base>__<param1>__...`` into (base, ``param1__...``)."""
    if "__" not in name:
        return name, None
    base, suffix = name.split("__", 1)
    return base, suffix


# ---------------------------------------------------------------------------
# Config loading and grid expansion
# ---------------------------------------------------------------------------


def _load_grid_config(
    path: str,
) -> tuple[str, str, str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load grid config YAML.

    Returns ``(base_name, game, track, training_spec, reward_spec, distribute_cfg, bc_cfg)``.
    *bc_cfg* is the optional ``bc:`` section used for inline BC pre-training; it is
    an empty dict when the section is absent.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base_name = cfg.get("base_name", "gs")
    game = cfg.get("game", "tmnf")
    track = cfg.get("track", None)
    training_spec = cfg.get("training_params", {})
    reward_spec = cfg.get("reward_params", {})
    distribute_cfg = cfg.get("distribute", {})
    bc_cfg = cfg.get("bc", {})
    return base_name, game, track, training_spec, reward_spec, distribute_cfg, bc_cfg


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


def _setup_experiment_dir(
    adapter,
    name: str,
    t: dict[str, Any],
    r: dict[str, Any],
    track_override: str | None,
) -> tuple[str, str, str]:
    """Create experiment dir, write config files. Returns (experiment_dir, weights_file, reward_cfg_file)."""
    base_name, param_suffix = _split_grid_run_name(name)
    experiment_dir = adapter.experiment_dir(base_name, t, track_override)
    if param_suffix:
        experiment_dir = f"{experiment_dir}/{param_suffix}"
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
    bc_warmstart_dir: str | None = None,
) -> list[tuple[str, Any]]:
    """Run all combos sequentially on this machine. Returns list of (name, ExperimentData)."""
    from framework.analytics import save_experiment_data_json

    all_runs = []
    n = len(combos)
    for i, (combo, name) in enumerate(zip(combos, names), 1):
        t = combo["training_params"]
        r = combo["reward_params"]
        logger.info("=== Run %d/%d: %s ===", i, n, name)

        experiment_dir, weights_file, reward_cfg_file = _setup_experiment_dir(adapter, name, t, r, track_override)

        if bc_warmstart_dir is not None:
            _copy_bc_weights(bc_warmstart_dir, experiment_dir)

        # Merge promoted policy params so grid axes like `epsilon: [0.5, 1.0]` work.
        t_with_pp = dict(t)
        t_with_pp["policy_params"] = _build_policy_params(t)

        game_spec = adapter.build_game_spec(
            name,
            experiment_dir,
            weights_file,
            reward_cfg_file,
            t_with_pp,
            track_override,
        )
        data = train_rl(
            game=game_spec,
            config=RunConfig.from_training_params(t_with_pp),
            probe=adapter.build_probe(t_with_pp),
            warmup=adapter.build_warmup(t_with_pp),
            no_interrupt=no_interrupt or i > 1,
            re_initialize=re_initialize,
        )

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
    bind_host: str,
    allow_non_lan: bool,
    game_name: str,
    monitor_username: str,
    monitor_password: str,
    local_workers: int = 1,
    local_worker_start_stagger_s: float = 5.0,
    no_interrupt: bool = False,
    re_initialize: bool = False,
    bc_warmstart_dir: str | None = None,
) -> list[tuple[str, Any]]:
    """Start coordinator, write local config files, block until all results arrive."""

    # Build ComboSpec list and write local experiment dirs so reward_config_file
    # resolves on this machine when save_grid_summary reads it.
    combo_specs = []
    for combo, name in zip(combos, names):
        t = combo["training_params"]
        r = combo["reward_params"]
        experiment_dir, _, _ = _setup_experiment_dir(adapter, name, t, r, track_override)
        if bc_warmstart_dir is not None:
            _copy_bc_weights(bc_warmstart_dir, experiment_dir)
        track = adapter.track_label(t, track_override)
        combo_specs.append(
            ComboSpec(
                name=name,
                track=track,
                training_params=t,
                reward_params=r,
                game=game_name,
            )
        )

    coord = Coordinator(
        combo_specs,
        token=token,
        port=port,
        heartbeat_timeout=heartbeat_timeout,
        bind_host=bind_host,
        allow_non_lan=allow_non_lan,
        monitor_username=monitor_username,
        monitor_password=monitor_password,
    )
    worker_procs: list[subprocess.Popen] = []
    try:
        coord.start()
        local_worker_host = bind_host
        if bind_host in ("", "0.0.0.0", "::"):
            local_worker_host = "127.0.0.1"
        worker_procs = _launch_local_workers(
            coordinator_host=local_worker_host,
            coordinator_port=coord.port,
            token=token,
            game_name=game_name,
            local_workers=local_workers,
            no_interrupt=no_interrupt,
            re_initialize=re_initialize,
            start_stagger_s=local_worker_start_stagger_s,
        )
        logger.info(
            "Coordinator ready on port %d — start workers with:\n"
            "  python -m distributed.worker --coordinator http://<this-host>:%d --token <token>",
            coord.port,
            coord.port,
        )
        raw_runs = coord.wait_for_all()
    finally:
        _stop_local_workers(worker_procs)
        coord.stop()

    # Override reward_config_file to the local path written above, then save results.
    all_runs = []
    for name, data in raw_runs:
        base_name, param_suffix = _split_grid_run_name(name)
        experiment_dir = adapter.experiment_dir(base_name, data.training_params, track_override)
        if param_suffix:
            experiment_dir = f"{experiment_dir}/{param_suffix}"
        data.reward_config_file = f"{experiment_dir}/reward_config.yaml"
        data.weights_file = f"{experiment_dir}/policy_weights.yaml"
        game_spec = adapter.build_game_spec(
            name,
            experiment_dir,
            data.weights_file,
            data.reward_config_file,
            data.training_params,
            track_override,
        )
        if game_spec.save_results_fn is not None:
            game_spec.save_results_fn(data, results_dir=f"{experiment_dir}/results")
        from framework.analytics import save_experiment_data_json

        save_experiment_data_json(data, results_dir=f"{experiment_dir}/results")
        all_runs.append((name, data))

    return all_runs


def _launch_local_workers(
    coordinator_host: str,
    coordinator_port: int,
    token: str,
    game_name: str,
    local_workers: int,
    no_interrupt: bool,
    re_initialize: bool,
    start_stagger_s: float = 5.0,
) -> list[subprocess.Popen]:
    """Start local worker subprocesses for distributed mode.

    ``coordinator_host`` is the host/IP embedded in the worker
    ``--coordinator`` URL.

    ``start_stagger_s`` introduces a cascading delay between consecutive
    worker launches (issue #254): the first worker starts immediately,
    the second waits ``start_stagger_s`` seconds, the third waits another
    ``start_stagger_s`` seconds, and so on.  This prevents the SC2 binaries
    spawned by each worker from racing to read the same ``.SC2Map`` file
    on disk at the same instant, which can fail with a "map not found"
    error when several PySC2 instances boot simultaneously.  Set to ``0``
    to disable.
    """
    if local_workers <= 0:
        return []

    coordinator_url = f"http://{coordinator_host}:{coordinator_port}"
    procs: list[subprocess.Popen] = []
    logger.info(
        "Launching %d local worker process(es) for game=%s at %s (stagger=%.1fs)",
        local_workers,
        game_name,
        coordinator_url,
        start_stagger_s,
    )
    try:
        for i in range(local_workers):
            cmd = [
                sys.executable,
                "-m",
                "distributed.worker",
                "--coordinator",
                coordinator_url,
                "--token",
                token,
                "--game",
                game_name,
                "--worker-id",
                f"local-{i + 1}",
            ]
            if no_interrupt:
                cmd.append("--no-interrupt")
            if re_initialize:
                cmd.append("--re-initialize")
            procs.append(subprocess.Popen(cmd))
            if i < local_workers - 1 and start_stagger_s > 0:
                sleep(start_stagger_s)
    except Exception:
        _stop_local_workers(procs)
        raise
    return procs


def _parse_non_negative_int(value: Any, source_name: str) -> int:
    """Parse *value* as an int >= 0, raising ValueError with source context."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source_name} must be an integer >= 0") from exc
    if parsed < 0:
        raise ValueError(f"{source_name} must be >= 0")
    return parsed


def _stop_local_workers(worker_procs: list[subprocess.Popen]) -> None:
    """Best-effort cleanup for local worker subprocesses."""
    for proc in worker_procs:
        try:
            if proc.poll() is not None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        except Exception as exc:
            logger.debug("Failed to stop local worker pid=%s: %s", proc.pid, exc)


def _consolidate(
    experiment_dirs: list[str],
    summary_name: str,
    summary_dir: str | None,
) -> None:
    """Load experiment data from *experiment_dirs* and produce a combined summary."""
    from framework.analytics import infer_varied_summary_keys, load_experiment_data

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

    varied_keys = infer_varied_summary_keys(all_runs)

    # Determine summary output dir
    if summary_dir is None:
        # Place summary next to the experiment dirs (common parent)
        parent = os.path.commonpath(experiment_dirs)
        summary_dir = os.path.join(parent, f"{summary_name}__summary")

    logger.info("Consolidating %d experiment(s) into %s", len(all_runs), summary_dir)
    from framework.analytics import save_grid_summary

    save_grid_summary(all_runs, varied_keys, summary_dir, summary_name)
    logger.info("Summary report: %s/summary.md", summary_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search over training/reward params (multi-game)")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to grid search YAML config (not needed with --consolidate)",
    )
    parser.add_argument(
        "--game",
        default=None,
        choices=["tmnf", "beamng", "car_racing", "torcs", "sc2", "rocket_league", "iracing", "atari"],
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
        "--live-gui",
        action="store_true",
        help=(
            "Show a live GUI window with per-step reward components (rolling avg "
            "of 5 steps) and observation values while each run trains."
        ),
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
        "python grid_search.py --consolidate "
        "experiments/tmnf/genetic/a03_centerline/gs/ms0.05 "
        "experiments/tmnf/genetic/a03_centerline/gs/ms0.1 "
        "--summary-name my_summary",
    )
    parser.add_argument(
        "--summary-name",
        default="consolidated",
        help="Base name for the consolidated summary (default: 'consolidated'). Used with --consolidate.",
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
        help="Coordinator HTTP port when --distribute is set (default: 5555, or value from config distribute.port)",
    )
    parser.add_argument(
        "--bind-host",
        default=None,
        help="Coordinator bind host/IP when --distribute is set "
        "(default: 0.0.0.0, or value from config distribute.bind_host).",
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
        "--local-workers",
        type=int,
        default=None,
        help="Spawn N local distributed.worker subprocesses automatically. "
        "Useful for running multiple SC2 instances on one machine "
        "(default: 1, or value from config distribute.local_workers).",
    )
    parser.add_argument(
        "--allow-non-lan",
        action="store_true",
        help="Allow non-LAN/public worker source IPs. By default the "
        "coordinator only accepts loopback/private/link-local clients.",
    )
    parser.add_argument(
        "--monitor-username",
        default=None,
        help="Username for the mobile run monitor (default: 'monitor', or "
        "value from config distribute.monitor_username).",
    )
    parser.add_argument(
        "--monitor-password",
        default=None,
        help="Password for the mobile run monitor (default: shared worker "
        "token, or value from config distribute.monitor_password).",
    )
    parser.add_argument(
        "--local-worker-stagger",
        type=float,
        default=None,
        metavar="S",
        help="Seconds to wait between launching consecutive local workers "
        "(default: 5.0, or value from config distribute.local_worker_stagger). "
        "Prevents parallel SC2 binaries from racing on the same .SC2Map file "
        "(issue #254). Set to 0 to disable.",
    )
    parser.add_argument(
        "--bc-warmstart-dir",
        default=None,
        metavar="DIR",
        help=(
            "Path to an existing BC-trained experiment directory containing "
            "policy_weights.yaml and bc_summary.json.  Every grid combo will be "
            "warm-started from those weights.  A policy-compatibility check is "
            "performed before any training starts.  Mutually exclusive with an "
            "inline 'bc:' section in the grid config YAML."
        ),
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
    logger.info("gamer-ai code version: %s", code_version())

    if args.consolidate:
        _consolidate(args.consolidate, args.summary_name, args.summary_dir)
        return

    if args.config is None:
        parser.error("config is required when not using --consolidate")

    if args.local_workers is not None:
        try:
            args.local_workers = _parse_non_negative_int(args.local_workers, "--local-workers")
        except ValueError as exc:
            parser.error(str(exc))
    if args.local_workers not in (None, 0) and not args.distribute:
        parser.error("--local-workers requires --distribute")

    base_name, game_name, track_override, training_spec, reward_spec, distribute_cfg, bc_cfg = _load_grid_config(
        args.config
    )

    # CLI --game / --track override YAML values
    game_name = getattr(args, "game", None) or game_name
    track_override = getattr(args, "track", None) or track_override

    adapter = GAME_ADAPTERS[game_name]()
    combos, varied_keys = _expand_grid(training_spec, reward_spec)
    if args.live_gui:
        for combo in combos:
            combo["training_params"]["live_gui"] = True

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

    # ------------------------------------------------------------------
    # BC warm-start: resolve warmstart dir, then validate compatibility.
    # ------------------------------------------------------------------
    bc_warmstart_dir: str | None = None
    if args.bc_warmstart_dir and bc_cfg:
        logger.warning(
            "--bc-warmstart-dir and an inline 'bc:' section are both present; "
            "--bc-warmstart-dir takes precedence and the 'bc:' section will be ignored."
        )
    if args.bc_warmstart_dir:
        bc_warmstart_dir = args.bc_warmstart_dir
    elif bc_cfg:
        bc_warmstart_dir = _run_inline_bc(bc_cfg, adapter, base_name, training_spec, track_override, game_name)

    if bc_warmstart_dir is not None:
        _validate_bc_warmstart_combos(bc_warmstart_dir, combos, names)
        logger.info("BC warm-start directory: %s", bc_warmstart_dir)

    if args.distribute:
        if args.local_workers is not None:
            local_workers = args.local_workers
        else:
            try:
                local_workers = _parse_non_negative_int(
                    distribute_cfg.get("local_workers", 1),
                    "distribute.local_workers",
                )
            except ValueError as exc:
                parser.error(str(exc))
        port = args.port or distribute_cfg.get("port", 5555)
        bind_host = args.bind_host or distribute_cfg.get("bind_host", "0.0.0.0")
        cfg_allow_non_lan = bool(distribute_cfg.get("allow_non_lan", False))
        allow_non_lan = args.allow_non_lan or cfg_allow_non_lan
        hb_timeout = args.heartbeat_timeout or distribute_cfg.get("heartbeat_timeout", 60.0)
        if args.local_worker_stagger is not None:
            local_worker_stagger = args.local_worker_stagger
        else:
            raw_stagger = distribute_cfg.get("local_worker_stagger", 5.0)
            try:
                local_worker_stagger = float(raw_stagger)
            except (TypeError, ValueError):
                parser.error(f"distribute.local_worker_stagger must be a non-negative number, got {raw_stagger!r}")
        if local_worker_stagger < 0:
            parser.error("--local-worker-stagger must be >= 0")
        token = args.token or os.environ.get("TMNF_GRID_TOKEN") or str(_uuid.uuid4())
        if not (args.token or os.environ.get("TMNF_GRID_TOKEN")):
            token_preview = f"{token[:8]}..." if len(token) > 8 else "[redacted]"
            logger.info(
                "Auto-generated token for distributed run (%s). Pass it to workers via --token or TMNF_GRID_TOKEN.",
                token_preview,
            )
        monitor_username = args.monitor_username or distribute_cfg.get("monitor_username") or "monitor"
        monitor_password = args.monitor_password or distribute_cfg.get("monitor_password") or token
        all_runs = _run_distributed(
            adapter,
            combos,
            names,
            track_override,
            token=token,
            port=port,
            heartbeat_timeout=hb_timeout,
            bind_host=bind_host,
            allow_non_lan=allow_non_lan,
            game_name=game_name,
            monitor_username=monitor_username,
            monitor_password=monitor_password,
            local_workers=local_workers,
            local_worker_start_stagger_s=local_worker_stagger,
            no_interrupt=args.no_interrupt,
            re_initialize=args.re_initialize,
            bc_warmstart_dir=bc_warmstart_dir,
        )
    else:
        all_runs = _run_local(
            adapter,
            combos,
            names,
            track_override=track_override,
            no_interrupt=args.no_interrupt,
            re_initialize=args.re_initialize,
            bc_warmstart_dir=bc_warmstart_dir,
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
    try:
        _analytics_mod = __import__(f"games.{game_name}.analytics", fromlist=["save_grid_summary"])
        _analytics_mod.save_grid_summary(all_runs, varied_keys, summary_dir, base_name)
    except (ImportError, AttributeError):
        logger.debug(
            "Game-specific save_grid_summary not available for %s; using framework fallback.",
            game_name,
        )
        from framework.analytics import save_grid_summary

        save_grid_summary(all_runs, varied_keys, summary_dir, base_name)
    logger.info("Summary report: %s/summary.md", summary_dir)


if __name__ == "__main__":
    main()
