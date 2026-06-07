"""Atari evaluation mode — load a champion policy and watch it play."""

from __future__ import annotations

import logging
import os

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def eval_atari(experiment_name: str, args) -> None:
    import games.atari.policies  # noqa: F401 — registers Atari policy types
    from framework.policies import POLICY_REGISTRY
    from games.atari.actions import DISCRETE_ACTIONS
    from games.atari.adapter import AtariAdapter
    from games.atari.env import make_env
    from games.atari.obs_spec import ATARI_OBS_SPEC

    adapter = AtariAdapter()
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
            "Run a training experiment first with:  python main.py <name> --game atari"
        )
    if not os.path.exists(weights_file):
        raise SystemExit(
            f"No champion weights found at: {weights_file}\nThe experiment may not have completed a greedy phase yet."
        )

    p = master_p
    if os.path.exists(training_params_file):
        with open(training_params_file) as f:
            p = yaml.safe_load(f)

    map_name = track_override or p.get("map_name", "Pong-v5")
    num_episodes: int = getattr(args, "num_episodes", 1)
    max_episode_time_s: float = float(p.get("in_game_episode_s", 60.0))
    policy_type: str = p.get("policy_type", "genetic")
    policy_params: dict = p.get("policy_params") or {}

    policy_cls = POLICY_REGISTRY.get(policy_type)
    if policy_cls is None:
        raise SystemExit(f"Unknown policy_type {policy_type!r} — was the experiment trained with a valid policy?")

    policy = policy_cls._construct_or_resume(
        obs_spec=ATARI_OBS_SPEC,
        head_names=["action"],
        discrete_actions=DISCRETE_ACTIONS,
        weights_file=weights_file,
        policy_params=policy_params,
        re_initialize=False,
    )

    print()
    print("=" * 56)
    print("  Atari Evaluation Mode")
    print("=" * 56)
    print(f"  Game:     {map_name}")
    print(f"  Policy:   {policy_type}")
    print(f"  Weights:  {weights_file}")
    print(f"  Episodes: {num_episodes}")
    print("=" * 56)
    print()

    env = make_env(
        experiment_dir=experiment_dir,
        map_name=map_name,
        max_episode_time_s=max_episode_time_s,
        render_mode="human",
    )

    scores: list[float] = []
    lengths: list[int] = []

    try:
        for ep in range(num_episodes):
            score, steps = _run_episode(env, policy, ep + 1, num_episodes)
            scores.append(score)
            lengths.append(steps)
    except KeyboardInterrupt:
        print("\n[Eval] Interrupted by user.")
    finally:
        env.close()

    if scores:
        _print_summary(scores, lengths)


def _run_episode(env, policy, ep_idx: int, total: int) -> tuple[float, int]:
    obs, _ = env.reset()

    cumulative_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        cumulative_reward += reward
        steps += 1

    print(f"  Episode {ep_idx}/{total}  score={cumulative_reward:.1f}  steps={steps}")
    return cumulative_reward, steps


def _print_summary(scores: list[float], lengths: list[int]) -> None:
    arr = np.array(scores, dtype=float)
    len_arr = np.array(lengths, dtype=float)
    print()
    print("=" * 56)
    print("  Aggregate Results")
    print("=" * 56)
    print(f"  Episodes : {len(scores)}")
    print(f"  Score    : mean={arr.mean():.1f}  σ={arr.std():.1f}  min={arr.min():.1f}  max={arr.max():.1f}")
    print(f"  Steps    : mean={len_arr.mean():.0f}  σ={len_arr.std():.0f}")
    print("=" * 56)
    print()
