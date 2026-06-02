"""StarCraft 2 BC adapter — wires :mod:`games.sc2.replay_bc` into the
framework BC orchestrator (:mod:`framework.bc`).

This is the SC2 implementation of :class:`framework.bc.BCAdapter`.  The
heavy lifting (replay parsing, dataset assembly, per-target fitting)
still lives in :mod:`games.sc2.replay_bc` — this module just owns the
SC2-specific config-key reading and forwards to it.
"""

from __future__ import annotations

import pathlib
from typing import Any

from framework.obs_spec import ObsSpec

SUPPORTED_TARGETS: tuple[str, ...] = (
    "sc2_reinforce",
    "sc2_genetic",
    "sc2_cmaes",
    "sc2_neural_net",
    "sc2_neural_dqn",
    "sc2_lstm",
    "sc2_cnn",
    "epsilon_greedy",
    "ucb_q",
)


class SC2BCAdapter:
    """SC2 implementation of the framework :class:`BCAdapter` Protocol."""

    name = "sc2"
    supported_targets: tuple[str, ...] = SUPPORTED_TARGETS
    default_target = "sc2_reinforce"

    def validate_replay_dir(
        self,
        replay_dir: str | pathlib.Path | None,
        *,
        race: str | None = None,
    ) -> Any:
        from games.sc2.replay_bc import validate_replay_dir as _validate

        if replay_dir is None:
            raise ValueError(
                "SC2 BC needs a directory of .SC2Replay files; "
                "pass --replay-dir or set bc_replay_dir in training_params.yaml."
            )
        return _validate(replay_dir, race=race)

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
        from games.sc2.replay_bc import build_dataset as _build

        if replay_dir is None:
            raise ValueError("SC2 BC requires a non-None replay directory.")

        player_id = _resolve_player_id(training_params)
        return _build(
            replay_dir,
            save_path,
            obs_spec=obs_spec,
            player_id=player_id,
            race=race,
            step_mul=training_params.get("bc_step_mul", training_params.get("step_mul", 1)),
            screen_size=training_params.get("screen_size", 64),
            minimap_size=training_params.get("minimap_size", 64),
            max_replays=max_replays,
        )

    def summary_extras(
        self,
        dataset: dict,
        meta: dict,
        *,
        target: str,
        training_params: dict,
    ) -> dict:
        """Per-game summary stats: fn_idx histogram + replay-counting fields."""
        import numpy as np

        fn_idx_all = dataset["actions"][:, 0].astype(int)
        unique_fns, counts = np.unique(fn_idx_all, return_counts=True)
        fn_histogram = {int(k): int(v) for k, v in zip(unique_fns, counts)}

        return {
            "n_replays_kept": int(meta.get("n_episodes", 0)),
            "n_replays_skipped_race": int(meta.get("n_replays_skipped_race", 0)),
            "fn_idx_histogram": fn_histogram,
            "bc_player_id": str(_resolve_player_id(training_params)),
        }

    def fit_bc(
        self,
        dataset: dict,
        obs_spec: ObsSpec,
        *,
        target: str,
        training_params: dict,
    ) -> tuple[Any, float]:
        from games.sc2.replay_bc import fit_bc as _fit

        return _fit(
            dataset,
            obs_spec,
            target=target,
            hidden_sizes=training_params.get("hidden_sizes"),
            bc_epochs=training_params.get("bc_epochs", 10),
            bc_learning_rate=training_params.get("bc_learning_rate", 1e-3),
            bc_batch_size=training_params.get("bc_batch_size", 256),
            bc_ignore_noop=training_params.get("bc_ignore_noop", True),
            seed=training_params.get("seed"),
            n_channels=training_params.get("_n_channels", 1),
            n_bins=training_params.get("n_bins", 3),
            bc_lstm_hidden_size=training_params.get("bc_lstm_hidden_size", 64),
        )


def _resolve_player_id(training_params: dict) -> int | str:
    """Read ``bc_player_id`` from *training_params*, accepting "winner"/"1"/"2"."""
    raw = training_params.get("bc_player_id", "winner")
    if raw in ("1", "2"):
        return int(raw)
    return raw


def make_bc_adapter() -> SC2BCAdapter:
    return SC2BCAdapter()
