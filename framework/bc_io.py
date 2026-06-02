"""I/O helpers for behaviour-cloning datasets and summaries.

The NPZ format here is byte-compatible with the one SC2 has been writing
since #351 (``games/sc2/replay_bc.build_dataset``), so existing ``demos.npz``
files load unchanged.

Dataset schema
==============

A BC dataset NPZ contains six keys::

    obs              float32[N, D]     — concatenated observations
    actions          float32[N, A]     — concatenated actions
    episode_starts   int64[E]          — start index of each episode in obs/actions
    episode_lengths  int64[E]          — length of each episode
    episode_id       int64[N]          — per-sample episode index ∈ [0, E)
    meta             0-d unicode str   — JSON-encoded metadata dict (numpy
                                          stores as `<U…`; ``np.load`` yields
                                          a 0-d ``str_``)

The exact shape of ``actions`` is game-specific (SC2 uses
``[fn_idx, x, y, queue]``; TMNF uses ``[steer, accel, brake]``).  The
framework loader is shape-agnostic.

Summary schema
==============

``bc_summary.json`` is written by :func:`save_summary` with at minimum::

    {
      "game": "sc2",
      "bc_target": "sc2_reinforce",
      "n_episodes": 12,
      "n_pairs": 8743,
      "bc_race": "terran",
      "final_bc_loss": 0.42,
      "extras": { ... game-specific stats ... }
    }
"""

from __future__ import annotations

import json
import pathlib

import numpy as np


def load_dataset(
    path: str | pathlib.Path,
    *,
    as_episodes: bool = False,
) -> dict | list[tuple[np.ndarray, np.ndarray]]:
    """Load a ``demos.npz`` dataset.

    Parameters
    ----------
    path :
        Path to the ``.npz`` file.
    as_episodes :
        ``False`` (default) — return a dict with keys ``obs``, ``actions``,
        ``episode_starts``, ``episode_lengths``, ``episode_id``, ``meta``.
        Suitable for memoryless policies that sample individual (obs, action)
        pairs.

        ``True`` — return a list of ``(obs_seq, act_seq)`` tuples, one per
        episode, preserving temporal order within each episode.  Suitable for
        recurrent policies that consume whole ordered sequences with
        hidden-state carry-over.
    """
    data = np.load(str(path), allow_pickle=False)
    obs = data["obs"]
    actions = data["actions"]
    episode_starts = data["episode_starts"]
    episode_lengths = data["episode_lengths"]
    episode_id = data["episode_id"]
    meta = json.loads(str(data["meta"]))

    if not as_episodes:
        return {
            "obs": obs,
            "actions": actions,
            "episode_starts": episode_starts,
            "episode_lengths": episode_lengths,
            "episode_id": episode_id,
            "meta": meta,
        }

    episodes: list[tuple[np.ndarray, np.ndarray]] = []
    for start, length in zip(episode_starts.tolist(), episode_lengths.tolist()):
        ep_obs = obs[start : start + length]
        ep_act = actions[start : start + length]
        episodes.append((ep_obs, ep_act))
    return episodes


def save_summary(experiment_dir: str | pathlib.Path, summary: dict) -> pathlib.Path:
    """Write *summary* to ``<experiment_dir>/bc_summary.json``.

    Returns the path written, for caller convenience.
    """
    experiment_dir = pathlib.Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    out = experiment_dir / "bc_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    return out
