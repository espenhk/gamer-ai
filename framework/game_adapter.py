"""Game adapter protocol and registry.

Each game implements a concrete adapter that knows how to build the config
bundles (GameSpec, RunConfig, ProbeSpec, WarmupSpec, PolicyExtras) from a
training-params dict.  The GAME_ADAPTERS registry maps game names to lazy
factories so loading one adapter never pulls in another game's heavy
dependencies (e.g. tminterface, pysc2).
"""

from __future__ import annotations

from typing import Any, Callable, Protocol


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class GameAdapter(Protocol):
    """Interface every per-game adapter module must satisfy."""

    name: str
    config_dir: str                     # e.g. "games/tmnf/config"

    def experiment_dir(
        self, experiment_name: str, training_params: dict,
        track_override: str | None,
    ) -> str:
        """Return the experiment directory path for this run."""
        ...

    def experiment_dir_root(
        self, training_params: dict, track_override: str | None,
    ) -> str:
        """Return the root directory under which all experiments live."""
        ...

    def track_label(
        self, training_params: dict, track_override: str | None,
    ) -> str:
        """Return the human-readable track/map label for dir naming."""
        ...

    def decorate_reward_cfg(
        self, reward_cfg: dict, training_params: dict,
        track_override: str | None,
    ) -> None:
        """Mutate *reward_cfg* with game-specific keys (e.g. centerline_path)."""
        ...

    def build_game_spec(
        self, experiment_name: str, experiment_dir: str,
        weights_file: str, reward_cfg_file: str,
        training_params: dict, track_override: str | None,
    ) -> Any:
        """Build a GameSpec for this experiment."""
        ...

    def build_probe(self, training_params: dict) -> Any | None:
        """Build a ProbeSpec, or None to skip probe/cold-start."""
        ...

    def build_warmup(self, training_params: dict) -> Any | None:
        """Build a WarmupSpec, or None to skip warmup."""
        ...

    def build_extras(
        self, weights_file: str, training_params: dict, re_initialize: bool,
    ) -> Any | None:
        """Build PolicyExtras, or None if no game-specific policy types."""
        ...


# ---------------------------------------------------------------------------
# Registry — lazy imports so loading one game never pulls in another
# ---------------------------------------------------------------------------

GAME_ADAPTERS: dict[str, Callable[[], GameAdapter]] = {
    "tmnf":       lambda: __import__("games.tmnf.adapter",       fromlist=["make_adapter"]).make_adapter(),
    "torcs":      lambda: __import__("games.torcs.adapter",      fromlist=["make_adapter"]).make_adapter(),
    "sc2":        lambda: __import__("games.sc2.adapter",        fromlist=["make_adapter"]).make_adapter(),
    "beamng":     lambda: __import__("games.beamng.adapter",     fromlist=["make_adapter"]).make_adapter(),
    "car_racing": lambda: __import__("games.car_racing.adapter", fromlist=["make_adapter"]).make_adapter(),
}
