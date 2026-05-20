"""Human-vs-AI interactive play mode for StarCraft 2.

Usage::

    python main.py <experiment_name> --game sc2 --play

Loads the champion policy from a completed experiment directory and
launches a two-player PySC2 session where a human controls one side via
the standard SC2 UI (keyboard + mouse) while the trained AI policy drives
the opponent side via PySC2.

No weight updates occur — this is pure inference for evaluation or fun.

At game end an episode summary is printed: score, game loop, and
win/loss/draw outcome.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import yaml

from games.sc2.client import SC2Client

if TYPE_CHECKING:
    import argparse
    import numpy as np

logger = logging.getLogger(__name__)

_HEAD_NAMES = ["fn_idx", "x", "y", "queue"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def play_sc2(experiment_name: str, args: argparse.Namespace) -> None:
    """Load champion weights and run a human-vs-AI interactive session.

    Parameters
    ----------
    experiment_name :
        Name of a previously trained experiment (must have ``policy_weights.yaml``).
    args :
        Parsed ``argparse.Namespace`` from ``main.py``.  Inspected for
        ``--track`` (map override) and ``--log-level``.
    """
    from games.sc2.adapter import SC2Adapter  # lazy — avoids pulling pysc2 at import time

    adapter = SC2Adapter()
    master_cfg = os.path.join(adapter.config_dir, "training_params.yaml")
    with open(master_cfg) as f:
        master_p = yaml.safe_load(f)

    track_override = getattr(args, "track", None)
    experiment_dir = adapter.experiment_dir(experiment_name, master_p, track_override)
    training_params_file = os.path.join(experiment_dir, "training_params.yaml")
    weights_file = os.path.join(experiment_dir, "policy_weights.yaml")

    if not os.path.isdir(experiment_dir):
        raise SystemExit(
            f"Experiment directory not found: {experiment_dir}\n"
            f"Train the agent first:  python main.py {experiment_name} --game sc2"
        )

    p = master_p
    if os.path.exists(training_params_file):
        with open(training_params_file) as f:
            p = yaml.safe_load(f)

    map_name    = track_override or p.get("map_name", "MoveToBeacon")
    step_mul    = p.get("step_mul", 1)
    screen_size = p.get("screen_size", 64)
    minimap_size = p.get("minimap_size", 64)
    agent_race  = p.get("agent_race", "random")

    policy = _load_champion_policy(weights_file, map_name)

    print()
    print("=" * 52)
    print("  SC2 Human-vs-AI Play Mode")
    print("=" * 52)
    print(f"  Map:    {map_name}")
    print(f"  Policy: {weights_file}")
    print(f"  You (Human) vs the trained AI agent")
    print(f"  Close the SC2 window or press Ctrl-C to quit")
    print("=" * 52)
    print()

    client = SC2Client(
        map_name=map_name,
        step_mul=step_mul,
        screen_size=screen_size,
        minimap_size=minimap_size,
        agent_race=agent_race,
        play_mode=True,
    )

    try:
        _run_episode(client, policy)
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Play loop
# ---------------------------------------------------------------------------

def _run_episode(client: SC2Client, policy) -> None:
    """Step through one episode and print the summary on termination."""
    obs, info = client.reset()

    if hasattr(policy, "on_episode_start"):
        policy.on_episode_start(info=info)

    step_count = 0

    try:
        while True:
            action = policy(obs)
            obs, _, done, info = client.step(action)
            step_count += 1

            if done:
                break
    except KeyboardInterrupt:
        print("\n[Play] Interrupted by user.")

    if hasattr(policy, "on_episode_end"):
        policy.on_episode_end()

    _print_summary(info, step_count)


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def _load_champion_policy(
    weights_file: str,
    map_name: str,
    obs_spec_preset: str | None = None,
    enable_belief: bool = False,
):
    """Return the champion policy loaded from *weights_file*.

    Detection order:

    1. Explicit ``policy_type`` key in the YAML file — dispatches to the
       matching policy class.
    2. No ``policy_type`` key — structural detection:

       * ``fn_idx_0_weights`` present → multi-head format →
         :class:`~games.sc2.sc2_policies.SC2MultiHeadLinearPolicy`
         (SC2GeneticPolicy / SC2CMAESPolicy champion files)
       * otherwise → per-head format →
         :class:`~games.sc2.policies.SC2LinearPolicy`
         (legacy CMA-ES champion files)
    """
    if not os.path.exists(weights_file):
        raise SystemExit(
            f"No champion weights found at: {weights_file}\n"
            "Train the agent first to generate champion weights."
        )

    from games.sc2.obs_spec import get_spec
    obs_spec = get_spec(map_name, preset=obs_spec_preset)

    if enable_belief:
        from pathlib import Path
        from games.sc2.belief_schema import load_belief_config, extend_obs_spec
        _cfg_path = Path(__file__).parent / "config" / "belief_config.yaml"
        belief_cfg = load_belief_config(_cfg_path)
        obs_spec = extend_obs_spec(obs_spec, belief_cfg)

    with open(weights_file) as f:
        cfg = yaml.safe_load(f) or {}

    policy_type = cfg.get("policy_type")
    logger.info("[Play] Loading champion policy (type=%s) from %s", policy_type, weights_file)

    if policy_type in ("sc2_genetic", "sc2_cmaes"):
        from games.sc2.sc2_policies import SC2MultiHeadLinearPolicy
        return SC2MultiHeadLinearPolicy.load(weights_file, obs_spec)

    if policy_type == "sc2_neural_net":
        from games.sc2.sc2_policies import SC2NeuralNetPolicy
        return SC2NeuralNetPolicy.from_cfg(cfg, obs_spec)

    if policy_type == "sc2_reinforce":
        from games.sc2.sc2_policies import SC2REINFORCEPolicy
        return SC2REINFORCEPolicy.from_cfg(cfg, obs_spec)

    if policy_type == "sc2_lstm":
        from games.sc2.sc2_policies import SC2LSTMPolicy
        return SC2LSTMPolicy.from_cfg(cfg, obs_spec)

    if policy_type == "sc2_neural_dqn":
        from games.sc2.sc2_policies import SC2NeuralDQNPolicy
        return SC2NeuralDQNPolicy.from_cfg(cfg, obs_spec)

    if policy_type == "sc2_cnn":
        # SC2CNNEvolutionPolicy saves its champion as a .npz file (not YAML).
        # The companion .npz path is weights_file with .yaml → .npz.
        npz_path = weights_file.replace(".yaml", ".npz")
        if not os.path.exists(npz_path):
            raise SystemExit(
                f"sc2_cnn champion not found at: {npz_path}\n"
                "Train the agent first to generate champion weights."
            )
        from games.sc2.cnn_policy import SC2CNNEvolutionPolicy
        import numpy as _np
        with _np.load(npz_path) as _data:
            n_channels = int(_data["n_channels"])
        policy = SC2CNNEvolutionPolicy(n_channels=n_channels, obs_spec=obs_spec)
        policy.load_champion(npz_path)
        return policy

    if policy_type == "neural_dqn":
        from games.sc2.sc2_policies import SC2NeuralDQNPolicy
        return SC2NeuralDQNPolicy.from_cfg(cfg, obs_spec)

    if policy_type == "reinforce":
        if (
            any(k in cfg for k in ("weights", "biases", "baseline_value"))
            and "trunk_weights" not in cfg
        ):
            raise SystemExit(
                "Unsupported legacy SC2 bare-name 'reinforce' weights format detected. "
                "This format was removed; retrain/save with policy_type 'sc2_reinforce'."
            )
        from games.sc2.sc2_policies import SC2REINFORCEPolicy
        return SC2REINFORCEPolicy.from_cfg(cfg, obs_spec)

    if policy_type == "lstm":
        if any(k in cfg for k in ("W_fn", "W_x", "W_y", "W_queue")):
            raise SystemExit(
                "Unsupported legacy SC2 bare-name 'lstm' weights format detected. "
                "This format was removed; retrain/save with policy_type 'sc2_lstm'."
            )
        from games.sc2.sc2_policies import SC2LSTMPolicy
        if "W_out" in cfg:
            return SC2LSTMPolicy.from_cfg(cfg, obs_spec)
        raise SystemExit(
            "Unsupported SC2 bare-name 'lstm' weights format. "
            "Use a champion saved with policy_type 'sc2_lstm'."
        )

    # No explicit policy_type — detect format by key structure.
    # SC2GeneticPolicy / SC2CMAESPolicy save via SC2MultiHeadLinearPolicy.save() →
    # fn_idx_0_weights keys, no policy_type tag.
    # Legacy CMA-ES saves via SC2LinearPolicy.save() → fn_idx_weights / x_weights.
    if "fn_idx_0_weights" in cfg:
        from games.sc2.sc2_policies import SC2MultiHeadLinearPolicy
        return SC2MultiHeadLinearPolicy.load(weights_file, obs_spec)

    from games.sc2.policies import SC2LinearPolicy
    return SC2LinearPolicy(obs_spec, _HEAD_NAMES, weights_file)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(info: dict, step_count: int) -> None:
    outcome = info.get("player_outcome")
    if outcome is not None and outcome > 0:
        result = "WIN  (AI)"
    elif outcome is not None and outcome < 0:
        result = "LOSS (AI)"
    else:
        result = "DRAW / INCONCLUSIVE"

    score     = info.get("score", 0.0)
    game_loop = info.get("game_loop", 0.0)

    print()
    print("=" * 52)
    print("  Episode Summary")
    print("=" * 52)
    print(f"  AI outcome:  {result}")
    print(f"  Score:       {score:.1f}")
    print(f"  Game loop:   {int(game_loop)}")
    print(f"  Steps:       {step_count}")
    print("=" * 52)
    print()
