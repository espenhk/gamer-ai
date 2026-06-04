"""TMNF behaviour-cloning adapter.

Implements :class:`framework.bc.BCAdapter` for TMNF.  Phase 3 (#395) ships
the *SimplePolicy* source — drive the canned hand-coded PD baseline
(:class:`games.tmnf.simple_policy.SimplePolicy`) in-game for a few laps
and least-squares-fit a :class:`games.tmnf.policies.WeightedLinearPolicy`
on the demos.  This is the same logic that the now-removed
``rl/pretrain.py`` ran under ``do_pretrain: true``; here it is exposed
through the framework BC pipeline instead.

Replay-file ingest (``*.Replay.Gbx``) is not currently supported.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np

from framework.obs_spec import ObsSpec

logger = logging.getLogger(__name__)

#: Default number of laps the SimplePolicy source records per BC run.
#: Matches the historical ``N_DEMO_LAPS = 3`` from ``rl/pretrain.py`` so a
#: vanilla ``--bc`` reproduces the old ``do_pretrain: true`` behaviour.
DEFAULT_N_DEMO_LAPS = 3


class TMNFBCAdapter:
    """TMNF implementation of the framework :class:`BCAdapter` Protocol.

    Source selection:

    * ``replay_dir=None`` — drive ``SimplePolicy`` live in a TMNF env and
      record ``DEFAULT_N_DEMO_LAPS`` (or ``training_params['bc_n_demo_laps']``)
      laps of (obs, action) demos.  This requires the TMNF env to be
      constructable, which in turn requires TMInterface + Trackmania to
      be reachable — i.e. running the BC mode binds to the live game
      just like a normal training run does.
    * ``replay_dir`` set to a directory of ``*.Replay.Gbx`` files —
      not currently supported.  Raises with a clear error message.
    """

    name = "tmnf"
    supported_targets: tuple[str, ...] = ("hill_climbing",)
    default_target = "hill_climbing"

    def validate_replay_dir(
        self,
        replay_dir: str | pathlib.Path | None,
        *,
        race: str | None = None,
    ) -> Any:
        if replay_dir is None:
            # SimplePolicy source — no replay files needed.  Return a
            # synthetic marker so logs read sensibly.
            return "<simple-policy-live-demos>"

        # .Replay.Gbx replay-file ingest is not currently implemented.
        path = pathlib.Path(replay_dir)
        gbx = sorted(path.glob("*.Replay.Gbx")) if path.is_dir() else []
        raise ValueError(
            f".Replay.Gbx ingest is not currently supported for TMNF BC.  "
            f"Found {len(gbx)} .Replay.Gbx file(s) in {replay_dir!r}.  "
            f"Run without --replay-dir to use the SimplePolicy live "
            f"demonstration source instead."
        )

    def build_dataset(
        self,
        replay_dir: str | pathlib.Path | None,
        save_path: str | pathlib.Path,
        *,
        obs_spec: ObsSpec,
        training_params: dict,
        race: str | None = None,
        max_replays: int | None = None,
    ) -> dict:
        if replay_dir is not None:
            # Caller skipped validate_replay_dir or framework re-entered with
            # a replay path — defend the same boundary.
            raise ValueError(
                ".Replay.Gbx ingest is not currently supported for TMNF BC; "
                "run without --replay-dir to use SimplePolicy demonstrations."
            )

        raw = training_params.get("bc_n_demo_laps")
        if raw is None:
            raw = max_replays if max_replays is not None else DEFAULT_N_DEMO_LAPS
        n_laps = int(raw)
        if n_laps <= 0:
            raise ValueError(f"bc_n_demo_laps must be > 0, got {n_laps}")

        env = _make_tmnf_env(training_params)
        try:
            obs_arr, act_arr, episode_lengths = collect_demos(env, n_laps)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("TMNF env.close() raised during BC: %s", exc)

        if obs_arr.shape[0] == 0:
            raise ValueError(f"SimplePolicy produced 0 demonstration steps over {n_laps} laps.")

        meta = {
            "source": "simple_policy",
            "n_episodes": int(len(episode_lengths)),
            "n_steps": int(obs_arr.shape[0]),
            "obs_dim": int(obs_arr.shape[1]),
            "bc_n_demo_laps": int(n_laps),
        }
        _save_dataset_npz(save_path, obs_arr, act_arr, episode_lengths, meta)
        return meta

    def fit_bc(
        self,
        dataset: dict,
        obs_spec: ObsSpec,
        *,
        target: str,
        training_params: dict,
    ) -> tuple[Any, float]:
        if target != "hill_climbing":
            # Defensive — the orchestrator already validates against
            # supported_targets, but keep the message clear if someone
            # bypasses it.
            raise ValueError(
                f"TMNF BC target {target!r} not supported; only 'hill_climbing' is available in phase 3 (#395)."
            )

        obs_arr = dataset["obs"]
        act_arr = dataset["actions"]
        policy = fit_weighted_linear(obs_arr, act_arr, obs_spec)
        loss = _bc_residual_loss(obs_arr, act_arr, obs_spec)
        return policy, loss


# ---------------------------------------------------------------------------
# Live-demo collection + lstsq fit  (lifted from the now-removed rl/pretrain.py)
# ---------------------------------------------------------------------------


def collect_demos(env, n_laps: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Drive *n_laps* with :class:`SimplePolicy`.

    Returns
    -------
    (obs, actions, episode_lengths)
        ``obs`` and ``actions`` are the concatenated trajectory arrays.
        ``episode_lengths`` is one entry per env episode between resets —
        which may be more than *n_laps* if the agent crashed or got
        truncated before finishing.  ``sum(episode_lengths) == len(obs)``.
    """
    from games.tmnf.simple_policy import SimplePolicy

    expert = SimplePolicy()
    obs, _ = env.reset()
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    episode_lengths: list[int] = []
    laps_done = 0
    steps_in_episode = 0
    while laps_done < n_laps:
        action = expert(obs)
        obs_list.append(obs.copy())
        act_list.append(action.copy())
        steps_in_episode += 1
        obs, _, terminated, truncated, info = env.step(action)
        if (terminated or truncated) and info.get("finished", False):
            laps_done += 1
        if terminated or truncated:
            episode_lengths.append(steps_in_episode)
            steps_in_episode = 0
            if laps_done < n_laps:
                obs, _ = env.reset()
    # Trailing in-flight partial episode (if any) — flush it.
    if steps_in_episode:
        episode_lengths.append(steps_in_episode)
    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(act_list, dtype=np.float32),
        episode_lengths,
    )


def fit_weighted_linear(obs_matrix, act_matrix, obs_spec: ObsSpec):
    """Fit steer/accel/brake heads via least-squares.  Returns a
    :class:`games.tmnf.policies.WeightedLinearPolicy`."""
    from games.tmnf.policies import WeightedLinearPolicy

    scales = obs_spec.scales.astype(float)
    norm_obs = obs_matrix / scales[np.newaxis, :]
    w_steer, *_ = np.linalg.lstsq(norm_obs, act_matrix[:, 0], rcond=None)
    w_accel, *_ = np.linalg.lstsq(norm_obs, act_matrix[:, 1], rcond=None)
    w_brake, *_ = np.linalg.lstsq(norm_obs, act_matrix[:, 2], rcond=None)
    n_lidar = sum(1 for d in obs_spec.dims if d.name.startswith("lidar_"))
    names = obs_spec.names
    return WeightedLinearPolicy.from_cfg(
        {
            "steer_weights": {name: float(w_steer[i]) for i, name in enumerate(names)},
            "accel_weights": {name: float(w_accel[i]) for i, name in enumerate(names)},
            "brake_weights": {name: float(w_brake[i]) for i, name in enumerate(names)},
        },
        n_lidar_rays=n_lidar,
    )


def _bc_residual_loss(obs_matrix, act_matrix, obs_spec: ObsSpec) -> float:
    """Mean-squared residual of the lstsq solution across the three heads.

    Reported as the framework's ``final_bc_loss`` so BC summaries carry a
    sensible numeric loss even though hill_climbing has no SGD step count.
    """
    scales = obs_spec.scales.astype(float)
    norm_obs = obs_matrix / scales[np.newaxis, :]
    total_sq = 0.0
    for col in range(min(3, act_matrix.shape[1])):
        w, *_ = np.linalg.lstsq(norm_obs, act_matrix[:, col], rcond=None)
        residual = norm_obs @ w - act_matrix[:, col]
        total_sq += float(np.mean(residual**2))
    return total_sq / max(1, min(3, act_matrix.shape[1]))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _make_tmnf_env(training_params: dict):
    """Build a TMNF env using the same factory the training loop uses.

    Requires the caller (typically ``main._run_bc``) to have stashed the
    resolved experiment directory in ``training_params['_bc_experiment_dir']``
    so :func:`games.tmnf.env.make_env` can find the experiment-local
    ``reward_config.yaml`` it loads at construction time.
    """
    experiment_dir = training_params.get("_bc_experiment_dir")
    if not experiment_dir:
        raise ValueError(
            "TMNF BC adapter needs the experiment directory in "
            "training_params['_bc_experiment_dir'] so make_env() can find "
            "reward_config.yaml.  This is normally set by main._run_bc — "
            "callers driving the adapter directly must set it themselves."
        )

    from games.tmnf.env import make_env  # deferred — needs TMInterface

    n_lidar_rays = training_params.get("n_lidar_rays", 0)
    return make_env(
        experiment_dir=experiment_dir,
        speed=training_params.get("speed", 1.0),
        in_game_episode_s=training_params.get("in_game_episode_s", 30.0),
        n_lidar_rays=n_lidar_rays,
        decision_offset_pct=training_params.get("decision_offset_pct", 0.75),
        action_window_ticks=training_params.get("action_window_ticks", 1),
    )


def _save_dataset_npz(
    save_path: str | pathlib.Path,
    obs_arr: np.ndarray,
    act_arr: np.ndarray,
    episode_lengths: list[int],
    meta: dict,
) -> None:
    """Write a ``demos.npz`` matching the framework BC dataset schema.

    ``episode_starts`` and ``episode_id`` are derived from
    ``episode_lengths`` so the on-disk file is internally consistent
    (``sum(episode_lengths) == len(obs)``) and round-trips through
    :func:`framework.bc_io.load_dataset` (including the
    ``as_episodes=True`` path used by recurrent BC targets).
    """
    import json

    ep_lengths_arr = np.asarray(episode_lengths, dtype=np.int64)
    ep_starts_arr = np.concatenate(([0], np.cumsum(ep_lengths_arr[:-1]))).astype(np.int64)
    ep_id_arr = np.repeat(np.arange(len(ep_lengths_arr), dtype=np.int64), ep_lengths_arr)
    np.savez_compressed(
        str(save_path),
        obs=obs_arr.astype(np.float32),
        actions=act_arr.astype(np.float32),
        episode_starts=ep_starts_arr,
        episode_lengths=ep_lengths_arr,
        episode_id=ep_id_arr,
        meta=np.array(json.dumps(meta)),
    )


def make_bc_adapter() -> TMNFBCAdapter:
    return TMNFBCAdapter()
