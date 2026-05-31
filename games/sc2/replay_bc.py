"""Replay behaviour-cloning dataset builder for StarCraft II (issue #351).

Reads a folder of .SC2Replay files and produces a sequence-aware
``(obs, action)`` demonstration dataset for offline behaviour cloning.

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

PySC2 imports are lazy: this module can be imported without the SC2 binary or
PySC2 installed (matching the existing games/sc2 convention).
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Iterator

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
