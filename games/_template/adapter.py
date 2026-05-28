"""<GAME_TITLE> game adapter — builds config bundles for train_rl.

Copy this file into your new ``games/<name>/adapter.py`` and replace every
``<PLACEHOLDER>`` with real values.  The ``GameAdapter`` protocol is defined
in ``framework/game_adapter.py``; read ``docs/framework/game_adapter.md``
for a worked example.
"""

from __future__ import annotations

from framework.run_config import GameSpec, ProbeSpec, WarmupSpec


class _TemplateAdapter:
    """Adapter skeleton — rename to ``<Name>Adapter``."""

    # -- Required attributes ------------------------------------------------
    name = "<name>"  # must match the key in GAME_ADAPTERS
    config_dir = "games/<name>/config"

    # -- Required methods ---------------------------------------------------

    def experiment_dir(
        self,
        experiment_name: str,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        """Return the experiment directory path for this run."""
        raise NotImplementedError("Return e.g. f'experiments/<name>/{policy}/{track}/{experiment_name}'")

    def experiment_dir_root(
        self,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        """Return the root directory under which all experiments live."""
        raise NotImplementedError("Return e.g. f'experiments/<name>/{policy}/{track}'")

    def track_label(
        self,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        """Return the human-readable track/map label for dir naming."""
        raise NotImplementedError("Return track_override or a sensible default")

    def decorate_reward_cfg(
        self,
        reward_cfg: dict,
        training_params: dict,
        track_override: str | None,
    ) -> None:
        """Mutate *reward_cfg* with game-specific keys (e.g. centerline_path).

        Leave empty (``pass``) if the game needs no extra reward-config keys.
        """
        pass

    def build_game_spec(
        self,
        experiment_name: str,
        experiment_dir: str,
        weights_file: str,
        reward_cfg_file: str,
        training_params: dict,
        track_override: str | None,
    ) -> GameSpec:
        """Build a GameSpec for this experiment.

        Import your game's obs_spec, actions, analytics, and env here
        (lazy imports to avoid pulling in heavy SDKs at startup).
        """
        raise NotImplementedError(
            "Build and return a GameSpec — see games/car_racing/adapter.py"
        )

    def build_probe(self, training_params: dict) -> ProbeSpec | None:
        """Build a ProbeSpec, or None to skip probe/cold-start."""
        return None

    def build_warmup(self, training_params: dict) -> WarmupSpec | None:
        """Build a WarmupSpec, or None to skip warmup."""
        return None


def make_adapter() -> _TemplateAdapter:
    return _TemplateAdapter()
