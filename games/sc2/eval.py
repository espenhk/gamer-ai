"""SC2 evaluation mode — run trained champion against a bot, no human needed.

Usage::

    python main.py <experiment> --game sc2 --eval
    python main.py <experiment> --game sc2 --eval --num-episodes 5
    python main.py <experiment> --game sc2 --eval --num-episodes 3 --eval-speed 4
    python main.py <experiment> --game sc2 --eval --bot-difficulty hard

Loads the champion policy from a completed experiment and runs it for one or
more episodes against the configured built-in bot (ladder maps) or in the
standard minigame environment.  No weight updates occur.

Speed control
-------------
``--eval-speed N`` overrides ``step_mul``.  Lower N = more decisions per
second of real game time; higher N = fewer (but larger) jumps per decision.
Eval mode also enables PySC2's ``realtime`` flag so the game advances at
natural game pace instead of as fast as possible, making it watchable.

Action logging
--------------
Each step is logged at DEBUG level.  At the end of every episode a breakdown
of executed action types (with percentages and an ASCII bar) is printed to
stdout.  The same breakdown is repeated as an aggregate across all episodes.
"""

from __future__ import annotations

import collections
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import yaml

from games.sc2.actions import FUNCTION_IDS
from games.sc2.client import SC2Client

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)

# Ordered list of all action names for consistent display even when count=0.
_ALL_FN_NAMES: tuple[str, ...] = tuple(FUNCTION_IDS[k] for k in sorted(FUNCTION_IDS))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def eval_sc2(experiment_name: str, args: argparse.Namespace) -> None:
    """Load champion weights and run evaluation episodes against the bot.

    Parameters
    ----------
    experiment_name :
        Name of a previously trained experiment (must have
        ``policy_weights.yaml``).
    args :
        Parsed ``argparse.Namespace`` from ``main.py``.  Inspected for
        ``--track``, ``--num-episodes``, ``--bot-difficulty``,
        ``--eval-speed``, ``--log-level``.
    """
    from games.sc2.adapter import SC2Adapter
    from games.sc2.play import _load_champion_policy

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

    map_name = track_override or p.get("map_name", "MoveToBeacon")
    screen_size = p.get("screen_size", 64)
    minimap_size = p.get("minimap_size", 64)
    agent_race = p.get("agent_race", "random")

    # step_mul: --eval-speed overrides experiment config.
    config_step_mul: int = p.get("step_mul", 8)
    eval_speed = getattr(args, "eval_speed", None)
    step_mul: int = eval_speed if eval_speed is not None else config_step_mul

    # bot_difficulty: --bot-difficulty overrides experiment config.
    bot_difficulty: str = (
        getattr(args, "bot_difficulty", None)
        or p.get("bot_difficulty", "very_easy")
    )

    num_episodes: int = getattr(args, "num_episodes", 1)

    policy = _load_champion_policy(weights_file, map_name)

    speed_note = (
        f"{step_mul}  (overriding config={config_step_mul})"
        if eval_speed is not None
        else f"{step_mul}"
    )
    print()
    print("=" * 62)
    print("  SC2 Evaluation Mode")
    print("=" * 62)
    print(f"  Map:            {map_name}")
    print(f"  Weights:        {weights_file}")
    print(f"  Episodes:       {num_episodes}")
    print(f"  Bot difficulty: {bot_difficulty}")
    print(f"  Step mul:       {speed_note}")
    print(f"  Realtime:       yes  (game runs at natural game pace)")
    print("=" * 62)
    print()

    client = SC2Client(
        map_name=map_name,
        step_mul=step_mul,
        screen_size=screen_size,
        minimap_size=minimap_size,
        agent_race=agent_race,
        bot_difficulty=bot_difficulty,
        visualize=True,
        realtime=True,
        play_mode=False,
    )

    all_results: list[dict] = []
    all_action_counts: collections.Counter = collections.Counter()

    try:
        for ep_idx in range(num_episodes):
            result = _run_episode(client, policy, ep_idx + 1, num_episodes)
            all_results.append(result)
            all_action_counts.update(result["action_counts"])
    except KeyboardInterrupt:
        print("\n[Eval] Interrupted by user.")
    finally:
        client.close()

    if all_results:
        _print_aggregate_summary(all_results, all_action_counts)


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------

def _run_episode(
    client: SC2Client,
    policy,
    ep_idx: int,
    total_episodes: int,
) -> dict:
    """Run one episode and return a result dict."""
    obs, _info = client.reset()

    if hasattr(policy, "on_episode_start"):
        policy.on_episode_start()

    step_count = 0
    cumulative_score = 0.0
    action_counts: collections.Counter = collections.Counter()
    substitution_count = 0

    done = False
    while not done:
        action = policy(obs)

        requested_fn_name = FUNCTION_IDS.get(int(action[0]), "no_op")
        obs, score, done, info = client.step(action)
        executed_fn_name = FUNCTION_IDS.get(client.last_fn_idx, "no_op")

        action_counts[executed_fn_name] += 1
        if executed_fn_name != requested_fn_name:
            substitution_count += 1
        cumulative_score += score
        step_count += 1

        if logger.isEnabledFor(logging.DEBUG):
            x_norm = float(np.clip(action[1], 0.0, 1.0))
            y_norm = float(np.clip(action[2], 0.0, 1.0))
            if executed_fn_name != requested_fn_name:
                logger.debug(
                    "Step %4d | %-24s -> %-24s (subst.) | x=%.3f y=%.3f | score_delta=%+.1f",
                    step_count, requested_fn_name, executed_fn_name,
                    x_norm, y_norm, score,
                )
            else:
                logger.debug(
                    "Step %4d | %-24s            | x=%.3f y=%.3f | score_delta=%+.1f",
                    step_count, executed_fn_name, x_norm, y_norm, score,
                )

    if hasattr(policy, "on_episode_end"):
        policy.on_episode_end()

    outcome = info.get("player_outcome") or 0
    final_score = info.get("score", 0.0)
    game_loop = info.get("game_loop", 0)
    result_str = "WIN " if outcome > 0 else ("LOSS" if outcome < 0 else "DRAW")

    print(f"\n  Episode {ep_idx}/{total_episodes}  [{result_str}]"
          f"  score={final_score:.1f}  loop={int(game_loop)}"
          f"  steps={step_count}  cumulative_reward={cumulative_score:.1f}")

    _print_action_breakdown(action_counts, step_count, substitution_count,
                            label=f"Episode {ep_idx}")

    return {
        "outcome": outcome,
        "score": final_score,
        "game_loop": game_loop,
        "steps": step_count,
        "cumulative_reward": cumulative_score,
        "action_counts": action_counts,
        "substitution_count": substitution_count,
    }


# ---------------------------------------------------------------------------
# Summary printers
# ---------------------------------------------------------------------------

def _print_action_breakdown(
    action_counts: collections.Counter,
    total_steps: int,
    substitution_count: int = 0,
    label: str = "",
) -> None:
    """Print a per-action-type count and percentage table with an ASCII bar."""
    header = f"  Action breakdown ({label}):" if label else "  Action breakdown:"
    print(header)
    bar_width = 30
    for fn_name in _ALL_FN_NAMES:
        count = action_counts.get(fn_name, 0)
        pct = 100.0 * count / max(total_steps, 1)
        filled = int(round(pct / 100.0 * bar_width))
        bar = "#" * filled + "-" * (bar_width - filled)
        print(f"    {fn_name:<28} {count:5d} / {total_steps:5d}  ({pct:5.1f}%)  [{bar}]")
    if substitution_count > 0:
        sub_pct = 100.0 * substitution_count / max(total_steps, 1)
        print(f"    {'  (blocked → substituted)':<28} {substitution_count:5d} / {total_steps:5d}  ({sub_pct:5.1f}%)")


def _print_aggregate_summary(
    results: list[dict],
    all_action_counts: collections.Counter,
) -> None:
    n = len(results)
    outcomes = [r["outcome"] for r in results]
    scores = [r["score"] for r in results]
    game_loops = [r["game_loop"] for r in results]
    steps = [r["steps"] for r in results]
    rewards = [r["cumulative_reward"] for r in results]
    total_steps = sum(steps)
    total_subs = sum(r["substitution_count"] for r in results)

    wins = sum(1 for o in outcomes if o > 0)
    losses = sum(1 for o in outcomes if o < 0)
    draws = sum(1 for o in outcomes if o == 0)

    print()
    print("=" * 62)
    print("  Aggregate Evaluation Summary")
    print("=" * 62)
    print(f"  Episodes:                {n}")
    print(f"  Wins / Losses / Draws:   {wins} / {losses} / {draws}")
    if n > 0:
        print(f"  Win rate:                {100.0 * wins / n:.1f}%")
    print(f"  Score:                   {np.mean(scores):.1f}  (σ={np.std(scores):.1f})"
          f"  range=[{min(scores):.1f}, {max(scores):.1f}]")
    print(f"  Game loop:               {np.mean(game_loops):.0f}  (σ={np.std(game_loops):.0f})")
    print(f"  Steps per episode:       {np.mean(steps):.0f}  (σ={np.std(steps):.0f})")
    print(f"  Cumulative reward:       {np.mean(rewards):.1f}  (σ={np.std(rewards):.1f})")
    print()
    _print_action_breakdown(all_action_counts, total_steps, total_subs,
                            label="all episodes")
    print("=" * 62)
    print()
