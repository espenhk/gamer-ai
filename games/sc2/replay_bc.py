"""Replay behaviour-cloning dataset builder and fitter for StarCraft II.

Issues #351 (dataset builder), #352 (validate_replay_dir), #353 (fit_bc + run).

Usage — build a dataset from replays::

    from games.sc2.obs_spec import get_spec
    from games.sc2.replay_bc import build_dataset, load_dataset

    spec = get_spec("MoveToBeacon")
    meta = build_dataset(
        "my_replays/",
        "demos.npz",
        obs_spec=spec,
        player_id="winner",
        race="terran",
        step_mul=1,
        screen_size=64,
        minimap_size=64,
    )

Usage — load a saved dataset::

    # Flat arrays (memoryless policies, random sampling)
    data = load_dataset("demos.npz")
    obs, actions = data["obs"], data["actions"]

    # Per-episode sequences (recurrent policies — ordered, hidden-state carry-over)
    episodes = load_dataset("demos.npz", as_episodes=True)
    for obs_seq, act_seq in episodes:
        ...  # obs_seq: [T, D], act_seq: [T, 4]

Usage — fit a BC policy from a dataset or replay directory::

    from games.sc2.replay_bc import fit_bc, run

    # Fit directly from a loaded dataset dict
    policy, loss = fit_bc(data, spec, target="sc2_reinforce", bc_epochs=20)

    # Full pipeline: replays → dataset → fit → save weights + summary
    summary = run(
        "my_replays/",
        "experiments/sc2/sc2_reinforce/MoveToBeacon/myrun/",
        obs_spec=spec,
        target="sc2_reinforce",
    )

PySC2 imports are lazy: this module can be imported without the SC2 binary or
PySC2 installed (matching the existing games/sc2 convention).
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

from framework.obs_spec import ObsSpec

logger = logging.getLogger(__name__)

_WINNER_SENTINEL = "winner"


class _TimestepAdapter:
    """Wraps a raw obs dict so it satisfies ``extract_flat_obs``'s protocol.

    ``extract_flat_obs`` expects ``timestep.observation`` to be a dict-like
    object (the same form returned by ``features.transform_obs``).
    """

    __slots__ = ("observation",)

    def __init__(self, observation: Any) -> None:
        self.observation = observation


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def iter_replays(folder: str | pathlib.Path) -> list[pathlib.Path]:
    """Return all ``*.SC2Replay`` paths under *folder*, sorted for reproducibility."""
    return sorted(pathlib.Path(folder).glob("*.SC2Replay"))


def validate_replay_dir(
    folder: str | pathlib.Path,
    *,
    race: str | None = None,
    version: str | None = None,
) -> list[pathlib.Path]:
    """Validate *folder* for use as a behaviour-cloning replay source.

    Checks that the directory exists, is non-empty, and contains at least one
    ``.SC2Replay`` file.  Emits :mod:`logging` warnings when *race* or
    *version* hints suggest the replay set may not match the experiment config.

    Parameters
    ----------
    folder:
        Directory to validate.
    race:
        Race filter the caller intends to apply (e.g. ``"terran"``).  When
        set to a non-empty, non-``"any"`` value, a ``WARNING`` is emitted
        reminding the user that cross-race replays will be silently skipped
        by :func:`build_dataset`.
    version:
        SC2 build version string to cross-check against replay filenames
        (e.g. ``"4.9.3"``).  Blizzard packs encode the build in the filename
        (e.g. ``"4.9.3.77379"``); a ``WARNING`` is emitted when one or more
        filenames do not contain the supplied string.  Does **not** reject
        any files.

    Returns
    -------
    list[Path]
        Sorted list of ``.SC2Replay`` paths found in *folder*.

    Raises
    ------
    ValueError
        If *folder* does not exist, is not a directory, or contains no
        ``.SC2Replay`` files.
    """
    folder = pathlib.Path(folder)
    if not folder.exists():
        raise ValueError(f"Replay directory does not exist: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Replay path is not a directory: {folder}")

    replays = sorted(folder.glob("*.SC2Replay"))
    if not replays:
        raise ValueError(
            f"No .SC2Replay files found in {folder!r}. "
            "Download Blizzard replay packs via the replay-api samples at "
            "https://github.com/Blizzard/s2client-proto and unzip them here."
        )

    logger.info("Found %d .SC2Replay file(s) in %s.", len(replays), folder)

    if race and race.lower() not in ("", "any"):
        logger.warning(
            "Race filter '%s' is active: replays whose cloned player is not "
            "%s will be silently skipped. Use race=None or race='any' to keep "
            "all races.",
            race,
            race,
        )

    if version:
        unmatched = [r for r in replays if version not in r.name]
        if unmatched:
            logger.warning(
                "%d/%d replay file(s) do not contain version string %r in "
                "their filename. Blizzard packs encode the SC2 build in the "
                "filename (e.g. '4.9.3.77379'). Replays from a different "
                "build may fail to load — ensure your SC2 binary version "
                "matches the replay pack.",
                len(unmatched),
                len(replays),
                version,
            )

    return replays


def _parse_replay_info(info: Any) -> tuple[int, dict[int, str]]:
    """Return ``(winner_player_id, {player_id: race_name})`` from a replay_info proto.

    *winner_player_id* is 0 when the result is undecided (tie / crash).
    Race strings are one of ``"terran"`` / ``"zerg"`` / ``"protoss"`` / ``"random"``.
    """
    _RACE_INT_TO_NAME: dict[int, str] = {
        1: "terran",
        2: "zerg",
        3: "protoss",
        4: "random",
    }
    winner_id = 0
    player_races: dict[int, str] = {}
    for pi in info.player_info:
        pid = int(pi.player_info.player_id)
        race = int(pi.player_info.race_actual)
        player_races[pid] = _RACE_INT_TO_NAME.get(race, "random")
        # result==1 → Victory
        if hasattr(pi, "player_result") and int(pi.player_result.result) == 1:
            winner_id = pid
    return winner_id, player_races


def _resolve_player_id(player_id: int | str, winner_id: int) -> int:
    """Convert *player_id* (``"winner"`` or a positive integer) to a concrete integer."""
    if player_id == _WINNER_SENTINEL:
        return winner_id if winner_id != 0 else 1
    pid = int(player_id)
    if pid < 1:
        raise ValueError(f"player_id must be a positive integer, got {player_id!r}")
    return pid


def _pick_best_action(function_calls: list[Any], strategy: str) -> Any | None:
    """Select one FunctionCall from a per-step list.

    Parameters
    ----------
    function_calls:
        Non-empty list of PySC2 ``FunctionCall`` objects for this step.
    strategy:
        ``"first_non_noop"`` — return the first non-no_op, falling back to
        the first element when all are no_op.
        ``"first"`` — always return the first element regardless of type.

    Returns ``None`` when *function_calls* is empty.
    """
    if not function_calls:
        return None
    if strategy == "first":
        return function_calls[0]
    # "first_non_noop" (default): skip no_op (PySC2 fn id 0)
    for fc in function_calls:
        if int(fc.function) != 0:
            return fc
    return function_calls[0]


# ---------------------------------------------------------------------------
# Core replay reader
# ---------------------------------------------------------------------------


def replay_observations(
    path: str | pathlib.Path,
    *,
    player_id: int | str,
    obs_spec: ObsSpec,
    step_mul: int = 1,
    screen_size: int = 64,
    minimap_size: int = 64,
    multi_action_strategy: str = "first_non_noop",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Open a single replay and yield ``(obs_vec, action_vec)`` pairs.

    Parameters
    ----------
    path:
        Path to a ``.SC2Replay`` file.
    player_id:
        Which player to observe: ``"winner"`` (resolved from ``replay_info``)
        or an explicit integer ``1`` / ``2``.
    obs_spec:
        Active observation spec — passed to :func:`extract_flat_obs`.
    step_mul:
        Game ticks advanced per loop step (coarsens replay playback).
    screen_size, minimap_size:
        Feature-layer resolutions; must match the experiment's config.
    multi_action_strategy:
        ``"first_non_noop"`` (default) keeps the first non-``no_op`` action
        when a step has multiple actions, falling back to the first action
        when all are ``no_op``.  ``"first"`` always returns the first action.

    Yields
    ------
    (obs_vec, action_vec):
        ``obs_vec`` is float32 of shape ``(obs_spec.n_dims,)``;
        ``action_vec`` is float32 of shape ``(4,)`` = ``[fn_idx, x, y, queue]``.
        Steps with no actions or only unknown fn_idx values are skipped.

    Notes
    -----
    PySC2 imports are lazy — callers without PySC2 installed can still import
    this module.  Raises ``ImportError`` at call-time if PySC2 is absent.
    """
    from pysc2 import run_configs
    from pysc2.lib import features as sc2_features
    from s2clientprotocol import sc2api_pb2 as sc2_api

    from games.sc2.actions import function_call_to_action
    from games.sc2.client import _ObsExtractState, extract_flat_obs

    path = pathlib.Path(path)
    run_config = run_configs.get()

    with run_config.start(want_rgb=False) as controller:
        replay_data = run_config.replay_data(str(path))
        info = controller.replay_info(replay_data)
        winner_id, _player_races = _parse_replay_info(info)
        resolved_pid = _resolve_player_id(player_id, winner_id)

        iface_options = sc2_api.InterfaceOptions(
            feature_layer=sc2_api.SpatialCameraSetup(
                width=24,
                resolution=sc2_api.Size2DI(x=screen_size, y=screen_size),
                minimap_resolution=sc2_api.Size2DI(x=minimap_size, y=minimap_size),
            ),
        )
        controller.start_replay(
            sc2_api.RequestStartReplay(
                replay_data=replay_data,
                options=iface_options,
                observed_player_id=resolved_pid,
                disable_fog=True,
            )
        )

        game_info = controller.game_info()
        feat = sc2_features.features_from_game_info(
            game_info=game_info,
            feature_dimensions=sc2_features.Dimensions(screen=screen_size, minimap=minimap_size),
            use_feature_units=True,
        )

        state = _ObsExtractState()
        skipped_unknown = 0

        while True:
            obs_proto = controller.observe()

            # player_result is a repeated field — truthy when the game is over.
            if obs_proto.player_result:
                break

            # Build flat obs vector via the shared extractor (issue #350).
            obs_dict = feat.transform_obs(obs_proto.observation)
            timestep = _TimestepAdapter(obs_dict)
            obs_vec, _feats = extract_flat_obs(timestep, obs_spec.names, state=state)

            # Extract actions issued during this step.
            raw_actions = list(obs_proto.actions)
            if not raw_actions:
                state.last_fn_idx = 0  # idle frame → no_op for next obs
                controller.step(step_mul)
                continue

            function_calls = [feat.reverse_action(a) for a in raw_actions]
            chosen_fc = _pick_best_action(function_calls, multi_action_strategy)
            if chosen_fc is None:
                state.last_fn_idx = 0
                controller.step(step_mul)
                continue

            action_vec = function_call_to_action(chosen_fc, screen_size=screen_size, minimap_size=minimap_size)
            if action_vec is None:
                skipped_unknown += 1
                state.last_fn_idx = 0
                logger.debug(
                    "Skipping step in %s: unknown PySC2 fn_id %d",
                    path.name,
                    int(chosen_fc.function),
                )
                controller.step(step_mul)
                continue

            # Update last-action state for next step's obs extraction.
            state.last_fn_idx = int(action_vec[0])
            yield obs_vec, action_vec
            controller.step(step_mul)

        if skipped_unknown:
            logger.info(
                "%s: skipped %d step(s) with unknown fn_idx",
                path.name,
                skipped_unknown,
            )


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def _read_one_replay(
    path: pathlib.Path,
    *,
    player_id: int | str,
    race_filter: str | None,
    obs_spec: ObsSpec,
    step_mul: int,
    screen_size: int,
    minimap_size: int,
    multi_action_strategy: str,
) -> tuple[bool, str, list[tuple[np.ndarray, np.ndarray]]]:
    """Open ONE SC2 process per replay: race-check first, then collect pairs.

    Returns
    -------
    (race_matched, player_race, pairs)
        *race_matched* is ``False`` (and *pairs* is empty) when the replay was
        dropped by the race filter.  *player_race* is the race string of the
        observed player regardless of filtering.
    """
    from pysc2 import run_configs
    from pysc2.lib import features as sc2_features
    from s2clientprotocol import sc2api_pb2 as sc2_api

    from games.sc2.actions import function_call_to_action
    from games.sc2.client import _ObsExtractState, extract_flat_obs

    run_config = run_configs.get()

    with run_config.start(want_rgb=False) as controller:
        replay_data = run_config.replay_data(str(path))
        info = controller.replay_info(replay_data)
        winner_id, player_races = _parse_replay_info(info)
        resolved_pid = _resolve_player_id(player_id, winner_id)
        player_race = player_races.get(resolved_pid, "random")

        # Apply race filter before starting the replay.
        if race_filter is not None and player_race != race_filter:
            return False, player_race, []

        iface_options = sc2_api.InterfaceOptions(
            feature_layer=sc2_api.SpatialCameraSetup(
                width=24,
                resolution=sc2_api.Size2DI(x=screen_size, y=screen_size),
                minimap_resolution=sc2_api.Size2DI(x=minimap_size, y=minimap_size),
            ),
        )
        controller.start_replay(
            sc2_api.RequestStartReplay(
                replay_data=replay_data,
                options=iface_options,
                observed_player_id=resolved_pid,
                disable_fog=True,
            )
        )

        game_info = controller.game_info()
        feat = sc2_features.features_from_game_info(
            game_info=game_info,
            feature_dimensions=sc2_features.Dimensions(screen=screen_size, minimap=minimap_size),
            use_feature_units=True,
        )

        state = _ObsExtractState()
        skipped_unknown = 0
        pairs: list[tuple[np.ndarray, np.ndarray]] = []

        while True:
            obs_proto = controller.observe()
            if obs_proto.player_result:
                break

            obs_dict = feat.transform_obs(obs_proto.observation)
            timestep = _TimestepAdapter(obs_dict)
            obs_vec, _feats = extract_flat_obs(timestep, obs_spec.names, state=state)

            raw_actions = list(obs_proto.actions)
            if not raw_actions:
                state.last_fn_idx = 0  # idle frame → no_op for next obs
                controller.step(step_mul)
                continue

            function_calls = [feat.reverse_action(a) for a in raw_actions]
            chosen_fc = _pick_best_action(function_calls, multi_action_strategy)
            if chosen_fc is None:
                state.last_fn_idx = 0
                controller.step(step_mul)
                continue

            action_vec = function_call_to_action(chosen_fc, screen_size=screen_size, minimap_size=minimap_size)
            if action_vec is None:
                skipped_unknown += 1
                state.last_fn_idx = 0
                logger.debug(
                    "Skipping step in %s: unknown PySC2 fn_id %d",
                    path.name,
                    int(chosen_fc.function),
                )
                controller.step(step_mul)
                continue

            state.last_fn_idx = int(action_vec[0])
            pairs.append((obs_vec, action_vec))
            controller.step(step_mul)

        if skipped_unknown:
            logger.info(
                "%s: skipped %d step(s) with unknown fn_idx",
                path.name,
                skipped_unknown,
            )

    return True, player_race, pairs


def build_dataset(
    folder: str | pathlib.Path,
    save_path: str | pathlib.Path,
    *,
    obs_spec: ObsSpec,
    player_id: int | str = _WINNER_SENTINEL,
    race: str | None = None,
    step_mul: int = 1,
    screen_size: int = 64,
    minimap_size: int = 64,
    multi_action_strategy: str = "first_non_noop",
    max_replays: int | None = None,
) -> dict:
    """Read replays in *folder*, optionally filter by race, and write ``demos.npz``.

    Parameters
    ----------
    folder:
        Directory containing ``.SC2Replay`` files.
    save_path:
        Output path for the compressed dataset (``demos.npz``).
    obs_spec:
        Active observation spec.  Determines the obs vector dimension.
    player_id:
        ``"winner"`` (default) or an explicit integer ``1`` / ``2``.
    race:
        When given (e.g. ``"terran"``), whole replays whose observed player
        is not that race are dropped.  ``None`` / ``"any"`` keeps all replays.
    step_mul, screen_size, minimap_size, multi_action_strategy:
        Forwarded to the per-replay collector.
    max_replays:
        When set, only the first *max_replays* files (sorted order) are
        processed.  ``None`` (default) processes all files.

    Returns
    -------
    dict
        Metadata dict; the same dict is embedded in the ``.npz`` as a JSON
        string under the ``meta`` key.

    Raises
    ------
    ValueError
        If no ``.SC2Replay`` files are found, or if the race filter removes
        every replay.
    """
    replay_paths = iter_replays(folder)
    if not replay_paths:
        raise ValueError(f"No .SC2Replay files found in {folder!r}")
    if max_replays is not None and max_replays > 0:
        replay_paths = replay_paths[:max_replays]
        logger.info("max_replays=%d: processing first %d replay(s)", max_replays, len(replay_paths))

    race_filter: str | None = None
    if race and race.lower() not in ("", "any"):
        race_filter = race.lower()

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    episode_starts: list[int] = []
    episode_lengths: list[int] = []
    episode_id_chunks: list[np.ndarray] = []
    source_names: list[str] = []
    kept = 0
    skipped_race = 0

    for replay_path in replay_paths:
        try:
            race_ok, player_race, pairs = _read_one_replay(
                replay_path,
                player_id=player_id,
                race_filter=race_filter,
                obs_spec=obs_spec,
                step_mul=step_mul,
                screen_size=screen_size,
                minimap_size=minimap_size,
                multi_action_strategy=multi_action_strategy,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to process %s: %s", replay_path.name, exc)
            continue

        if not race_ok:
            skipped_race += 1
            logger.debug(
                "Race filter '%s' dropped %s (player race: %s)",
                race_filter,
                replay_path.name,
                player_race,
            )
            continue

        if not pairs:
            logger.debug("Replay %s yielded no steps; skipping.", replay_path.name)
            continue

        start_idx = len(all_obs)
        ep_len = len(pairs)
        episode_starts.append(start_idx)
        episode_lengths.append(ep_len)
        all_obs.extend(obs for obs, _ in pairs)
        all_actions.extend(act for _, act in pairs)
        episode_id_chunks.append(np.full(ep_len, kept, dtype=np.int64))
        source_names.append(replay_path.name)
        kept += 1
        logger.info("Loaded %s (%d steps)", replay_path.name, ep_len)

    if kept == 0:
        if skipped_race > 0:
            raise ValueError(f"Race filter '{race}' removed all {skipped_race} replay(s) in {folder!r}.")
        raise ValueError(f"No usable replays found in {folder!r}.")

    obs_arr = np.stack(all_obs, axis=0).astype(np.float32)
    act_arr = np.stack(all_actions, axis=0).astype(np.float32)
    ep_starts_arr = np.array(episode_starts, dtype=np.int64)
    ep_lengths_arr = np.array(episode_lengths, dtype=np.int64)
    ep_id_arr = np.concatenate(episode_id_chunks).astype(np.int64)

    meta: dict = {
        "player_id": str(player_id),
        "race_filter": race_filter,
        "step_mul": step_mul,
        "screen_size": screen_size,
        "minimap_size": minimap_size,
        "source_filenames": source_names,
        "n_episodes": kept,
        "n_steps": int(obs_arr.shape[0]),
        "obs_dim": int(obs_arr.shape[1]),
        "n_replays_skipped_race": skipped_race,
    }

    logger.info(
        "Kept %d replay(s) (%d steps total; %d skipped by race filter).",
        kept,
        int(obs_arr.shape[0]),
        skipped_race,
    )

    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(save_path),
        obs=obs_arr,
        actions=act_arr,
        episode_starts=ep_starts_arr,
        episode_lengths=ep_lengths_arr,
        episode_id=ep_id_arr,
        meta=np.array(json.dumps(meta)),
    )
    logger.info("Saved dataset to %s", save_path)
    return meta


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def load_dataset(
    path: str | pathlib.Path,
    *,
    as_episodes: bool = False,
) -> dict | list[tuple[np.ndarray, np.ndarray]]:
    """Load a ``demos.npz`` dataset saved by :func:`build_dataset`.

    Parameters
    ----------
    path:
        Path to the ``.npz`` file.
    as_episodes:
        ``False`` (default) — return a dict with keys ``obs``, ``actions``,
        ``episode_starts``, ``episode_lengths``, ``episode_id``, and ``meta``
        (parsed JSON).  Suitable for memoryless policies that sample
        individual (obs, action) pairs.

        ``True`` — return a list of ``(obs_seq, act_seq)`` tuples, one per
        episode, preserving temporal order within each episode.  Suitable for
        recurrent policies that consume whole ordered sequences with
        hidden-state carry-over.

    Returns
    -------
    dict or list of (obs_seq, act_seq)
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


# ---------------------------------------------------------------------------
# BC fitting — private helpers
# ---------------------------------------------------------------------------


def _fit_bc_mlp(
    obs_arr: np.ndarray,
    act_arr: np.ndarray,
    obs_spec: ObsSpec,
    *,
    hidden_sizes: list[int],
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int | None,
) -> tuple[Any, float]:
    """Mini-batch gradient descent BC fit into an SC2REINFORCEPolicy.

    Loss = cross-entropy on the fn_idx softmax head
         + MSE on the sigmoid (x, y) spatial head for spatial actions only.

    Parameters are updated in-place on the policy via pure NumPy backward
    passes (no autograd dependency).

    Returns (policy, avg_loss_last_epoch).
    """
    from games.sc2.actions import SPATIAL_FN_IDS
    from games.sc2.sc2_policies import SC2REINFORCEPolicy

    policy = SC2REINFORCEPolicy(obs_spec, hidden_sizes=hidden_sizes, seed=seed)
    rng = np.random.default_rng(seed)

    fn_idx_arr = act_arr[:, 0].astype(int)
    xy_arr = act_arr[:, 1:3]
    n = len(obs_arr)
    scales = obs_spec.scales.astype(np.float64)
    spatial_mask_all = np.array([fi in SPATIAL_FN_IDS for fi in fn_idx_arr], dtype=bool)

    avg_loss = float("inf")
    for epoch in range(epochs):
        order = rng.permutation(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            obs_b = obs_arr[idx].astype(np.float64)
            fn_b = fn_idx_arr[idx]
            xy_b = xy_arr[idx].astype(np.float64)
            sp_mask = spatial_mask_all[idx]
            B = len(idx)

            # --- forward ---
            x = obs_b / scales  # (B, D)
            li: list[np.ndarray] = []  # layer inputs (for weight grads)
            pre: list[np.ndarray] = []  # pre-ReLU activations (for backprop)
            for w, b in zip(policy._trunk_w, policy._trunk_b):
                li.append(x.copy())
                z = x @ w.T.astype(np.float64) + b.astype(np.float64)
                pre.append(z.copy())
                x = np.maximum(0.0, z)
            h = x  # (B, H_last)

            # Capture pre-update weights for correct gradient computation.
            fn_w = policy._fn_w.astype(np.float64)
            sp_w = policy._sp_w.astype(np.float64)
            fn_b_vec = policy._fn_b.astype(np.float64)
            sp_b_vec = policy._sp_b.astype(np.float64)

            fn_logits = h @ fn_w.T + fn_b_vec  # (B, N_FN)
            sp_logits = h @ sp_w.T + sp_b_vec  # (B, 2)

            # --- fn cross-entropy ---
            fn_shift = fn_logits - fn_logits.max(axis=1, keepdims=True)
            exp_fn = np.exp(fn_shift)
            sm = exp_fn / exp_fn.sum(axis=1, keepdims=True)  # (B, N_FN)
            ce_loss = -float(np.mean(np.log(np.maximum(sm[np.arange(B), fn_b], 1e-10))))

            # --- sp MSE (spatial steps only) ---
            sp_sig = 1.0 / (1.0 + np.exp(-np.clip(sp_logits, -20.0, 20.0)))
            sp_loss = 0.0
            d_sp = np.zeros_like(sp_logits)
            if sp_mask.any():
                diff = sp_sig[sp_mask] - xy_b[sp_mask]  # (M, 2)
                sp_loss = float(np.mean(diff**2))
                # Gradient: ∂MSE/∂sp_logit = (2*(sig-y)/B) * sig*(1-sig)
                d_sp_sig = 2.0 * diff / B
                d_sp[sp_mask] = d_sp_sig * sp_sig[sp_mask] * (1.0 - sp_sig[sp_mask])

            total_loss += ce_loss + sp_loss
            n_batches += 1

            # --- fn CE gradient: ∂CE/∂fn_logits = (softmax - one_hot) / B ---
            d_fn = sm.copy()
            d_fn[np.arange(B), fn_b] -= 1.0
            d_fn /= B  # (B, N_FN)

            # --- backprop to h_last using pre-update head weights ---
            g = d_fn @ fn_w + d_sp @ sp_w  # (B, H_last)

            # --- collect trunk gradients (pre-update pass) ---
            trunk_dw: list[np.ndarray] = []
            trunk_db: list[np.ndarray] = []
            for i in range(len(policy._trunk_w) - 1, -1, -1):
                g = g * (pre[i] > 0)  # backprop through ReLU
                trunk_dw.insert(0, g.T @ li[i])  # (H_i, H_{i-1})
                trunk_db.insert(0, g.sum(axis=0))  # (H_i,)
                if i > 0:
                    g = g @ policy._trunk_w[i].astype(np.float64)

            # --- apply all updates ---
            policy._fn_w = (fn_w - lr * (d_fn.T @ h)).astype(np.float32)
            policy._fn_b = (fn_b_vec - lr * d_fn.sum(axis=0)).astype(np.float32)
            policy._sp_w = (sp_w - lr * (d_sp.T @ h)).astype(np.float32)
            policy._sp_b = (sp_b_vec - lr * d_sp.sum(axis=0)).astype(np.float32)
            for i, (dw, db) in enumerate(zip(trunk_dw, trunk_db)):
                policy._trunk_w[i] = (policy._trunk_w[i].astype(np.float64) - lr * dw).astype(np.float32)
                policy._trunk_b[i] = (policy._trunk_b[i].astype(np.float64) - lr * db).astype(np.float32)

        avg_loss = total_loss / max(n_batches, 1)
        logger.info("BC epoch %d/%d: avg_loss=%.6f", epoch + 1, epochs, avg_loss)

    return policy, avg_loss


def _fit_bc_linear(
    obs_arr: np.ndarray,
    act_arr: np.ndarray,
    obs_spec: ObsSpec,
) -> tuple[Any, float]:
    """Closed-form least-squares BC fit into an SC2MultiHeadLinearPolicy.

    fn head  : one-hot labels → lstsq over normalised obs.
    spatial head : logit-transformed (x, y) labels for spatial steps → lstsq.

    Returns (SC2MultiHeadLinearPolicy, normalised_residual).
    """
    from games.sc2.actions import SPATIAL_FN_IDS
    from games.sc2.sc2_policies import N_FUNCTION_IDS, N_SPATIAL_ROWS, SC2MultiHeadLinearPolicy

    scales = obs_spec.scales.astype(np.float64)
    X = obs_arr.astype(np.float64) / scales  # (N, D)
    fn_idx_arr = act_arr[:, 0].astype(int)
    xy_arr = act_arr[:, 1:3].astype(np.float64)
    N = len(X)

    # fn head: one-hot → lstsq → W_fn of shape (D, N_FN) → transpose to (N_FN, D)
    Y_fn = np.zeros((N, N_FUNCTION_IDS), dtype=np.float64)
    Y_fn[np.arange(N), fn_idx_arr] = 1.0
    W_fn, resid_fn, _, _ = np.linalg.lstsq(X, Y_fn, rcond=None)
    fn_weights = W_fn.T.astype(np.float32)  # (N_FN, D)

    # sp head: logit-transformed coords for spatial steps → lstsq → (D, 2) → (2, D)
    sp_mask = np.array([fi in SPATIAL_FN_IDS for fi in fn_idx_arr], dtype=bool)
    sp_weights = np.zeros((N_SPATIAL_ROWS, obs_spec.dim), dtype=np.float32)
    sp_resid = 0.0
    if sp_mask.any():
        X_sp = X[sp_mask]
        Y_sp_raw = np.clip(xy_arr[sp_mask], 1e-6, 1.0 - 1e-6)
        Y_logit = np.log(Y_sp_raw / (1.0 - Y_sp_raw))  # logit: (M, 2)
        W_sp, resid_sp, _, _ = np.linalg.lstsq(X_sp, Y_logit, rcond=None)
        sp_weights = W_sp.T.astype(np.float32)  # (2, D)
        sp_resid = float(resid_sp.mean()) if resid_sp.size > 0 else 0.0

    fn_resid = float(resid_fn.mean()) if resid_fn.size > 0 else 0.0
    final_loss = fn_resid / max(N, 1) + sp_resid / max(sp_mask.sum(), 1)

    policy = SC2MultiHeadLinearPolicy(obs_spec, fn_weights=fn_weights, spatial_weights=sp_weights)
    return policy, final_loss


# ---------------------------------------------------------------------------
# Public fit_bc entry point
# ---------------------------------------------------------------------------


def fit_bc(
    dataset: "dict | str | pathlib.Path",
    obs_spec: ObsSpec,
    *,
    target: str = "sc2_reinforce",
    hidden_sizes: list[int] | None = None,
    bc_epochs: int = 10,
    bc_learning_rate: float = 1e-3,
    bc_batch_size: int = 256,
    bc_ignore_noop: bool = True,
    seed: int | None = None,
) -> tuple[Any, float]:
    """Pre-train a policy on a demonstration dataset via behaviour cloning.

    Parameters
    ----------
    dataset :
        A dict returned by :func:`load_dataset`, or a path to a ``.npz``
        file written by :func:`build_dataset`.
    obs_spec :
        Observation spec used to normalise observations before fitting.
    target :
        Policy architecture to train.

        * ``"sc2_reinforce"`` (default) — two-head MLP
          (:class:`games.sc2.sc2_policies.SC2REINFORCEPolicy`) trained via
          mini-batch gradient descent.  Loss = cross-entropy on the fn_idx
          softmax head + MSE on the sigmoid ``(x, y)`` spatial head for
          spatial actions only.  Saved weights load via
          ``SC2REINFORCEPolicy.from_cfg`` so a subsequent ``sc2_reinforce``
          fine-tuning run resumes from them.

        * ``"sc2_genetic"`` — linear
          (:class:`games.sc2.sc2_policies.SC2MultiHeadLinearPolicy`) fitted
          via closed-form least squares.  Saved in ``SC2MultiHeadLinearPolicy``
          YAML format; compatible with ``sc2_genetic`` fine-tuning.

    hidden_sizes :
        MLP trunk hidden-layer widths (``"sc2_reinforce"`` only).
        Defaults to ``[128, 64]``.
    bc_epochs :
        Full passes over the dataset (``"sc2_reinforce"`` only).
    bc_learning_rate :
        Gradient step size (``"sc2_reinforce"`` only).
    bc_batch_size :
        Mini-batch size (``"sc2_reinforce"`` only).
    bc_ignore_noop :
        Drop steps where ``fn_idx == 0`` (no-op) before fitting so the fn
        head is not swamped by idle frames.
    seed :
        Optional RNG seed for reproducibility.

    Returns
    -------
    (policy, final_bc_loss)
        *policy* is the fitted policy object.  *final_bc_loss* is the
        average loss over the last epoch for MLP targets, or the normalised
        residual for the linear target.

    Raises
    ------
    ValueError
        If the dataset is empty after the optional no-op filter.
    """
    if not isinstance(dataset, dict):
        dataset = load_dataset(str(dataset))

    obs_arr = dataset["obs"].astype(np.float32)
    act_arr = dataset["actions"].astype(np.float32)

    if bc_ignore_noop:
        fn_idx_all = act_arr[:, 0].astype(int)
        keep = fn_idx_all != 0
        n_before = len(obs_arr)
        obs_arr = obs_arr[keep]
        act_arr = act_arr[keep]
        logger.info("bc_ignore_noop: kept %d / %d steps (dropped %d no-ops)", len(obs_arr), n_before, n_before - len(obs_arr))

    if len(obs_arr) == 0:
        raise ValueError("Empty dataset after filtering — no non-noop steps available for BC.")

    if target == "sc2_genetic":
        return _fit_bc_linear(obs_arr, act_arr, obs_spec)

    return _fit_bc_mlp(
        obs_arr,
        act_arr,
        obs_spec,
        hidden_sizes=hidden_sizes if hidden_sizes is not None else [128, 64],
        lr=bc_learning_rate,
        epochs=bc_epochs,
        batch_size=bc_batch_size,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# run — full pipeline: replays → dataset → fit → save
# ---------------------------------------------------------------------------


def run(
    replay_dir: "str | pathlib.Path",
    experiment_dir: "str | pathlib.Path",
    obs_spec: ObsSpec,
    *,
    target: str = "sc2_reinforce",
    player_id: "int | str" = "winner",
    race: str | None = None,
    max_replays: int | None = None,
    step_mul: int = 1,
    screen_size: int = 64,
    minimap_size: int = 64,
    hidden_sizes: list[int] | None = None,
    bc_epochs: int = 10,
    bc_learning_rate: float = 1e-3,
    bc_batch_size: int = 256,
    bc_ignore_noop: bool = True,
    seed: int | None = None,
) -> dict:
    """Orchestrate replay reading, BC fitting, and saving of weights + summary.

    Workflow:

    1. :func:`validate_replay_dir` and optionally cap at *max_replays*.
    2. :func:`build_dataset` — parse replays → NPZ demonstration dataset.
    3. :func:`fit_bc` — pre-train the policy on the dataset.
    4. Save ``policy_weights.yaml`` (+ ``trainer_state.npz`` for MLP targets)
       and ``bc_summary.json`` into *experiment_dir*.

    Parameters
    ----------
    replay_dir :
        Directory containing ``.SC2Replay`` files.
    experiment_dir :
        Destination directory for ``policy_weights.yaml``,
        ``trainer_state.npz``, and ``bc_summary.json``.
    obs_spec :
        Observation spec — built from ``training_params.yaml`` by the caller.
    target, player_id, race, max_replays, step_mul, screen_size,
    minimap_size, hidden_sizes, bc_epochs, bc_learning_rate,
    bc_batch_size, bc_ignore_noop, seed :
        Forwarded to :func:`build_dataset` and/or :func:`fit_bc`.

    Returns
    -------
    dict
        BC summary dict (same content written to ``bc_summary.json``).

    Raises
    ------
    ValueError
        If the replay directory is empty, race filter drops all replays, or
        the filtered dataset has no steps.
    """
    import os
    import tempfile

    from framework.policies import BasePolicy, trainer_state_path

    experiment_dir = pathlib.Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Normalise race: "any"/empty → None (no filter in build_dataset)
    race_filter: str | None = None
    if race and race.lower() not in ("", "any"):
        race_filter = race.lower()

    # Build dataset in a temporary file
    with tempfile.TemporaryDirectory() as tmp:
        demos_path = pathlib.Path(tmp) / "demos.npz"
        meta = build_dataset(
            replay_dir,
            demos_path,
            obs_spec=obs_spec,
            player_id=player_id,
            race=race_filter,
            step_mul=step_mul,
            screen_size=screen_size,
            minimap_size=minimap_size,
            max_replays=max_replays,
        )
        dataset = load_dataset(demos_path)

    # Fit BC
    policy, bc_loss = fit_bc(
        dataset,
        obs_spec,
        target=target,
        hidden_sizes=hidden_sizes,
        bc_epochs=bc_epochs,
        bc_learning_rate=bc_learning_rate,
        bc_batch_size=bc_batch_size,
        bc_ignore_noop=bc_ignore_noop,
        seed=seed,
    )

    # Save policy weights
    weights_path = str(experiment_dir / "policy_weights.yaml")
    policy.save(weights_path)
    logger.info("BC: saved policy weights → %s", weights_path)

    # Save trainer state for policies that carry one (e.g. SC2REINFORCEPolicy)
    if isinstance(policy, BasePolicy):
        ts_path = trainer_state_path(weights_path)
        policy.save_trainer_state(ts_path)
        logger.info("BC: saved trainer state → %s", ts_path)

    # Build fn_idx histogram from the raw (unfiltered) dataset
    act_arr_all = dataset["actions"]
    fn_idx_all = act_arr_all[:, 0].astype(int)
    unique_fns, counts = np.unique(fn_idx_all, return_counts=True)
    fn_histogram = {int(k): int(v) for k, v in zip(unique_fns, counts)}

    summary = {
        "n_replays_kept": int(meta["n_episodes"]),
        "n_replays_skipped_race": int(meta.get("n_replays_skipped_race", 0)),
        "n_episodes": int(meta["n_episodes"]),
        "n_pairs": int(meta["n_steps"]),
        "fn_idx_histogram": fn_histogram,
        "bc_player_id": str(player_id),
        "bc_race": race if race else "any",
        "bc_target": target,
        "final_bc_loss": float(bc_loss),
    }

    summary_path = experiment_dir / "bc_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("BC summary written → %s", summary_path)

    return summary
