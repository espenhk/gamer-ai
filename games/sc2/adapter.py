"""StarCraft 2 game adapter — builds config bundles for train_rl."""

from __future__ import annotations

import logging

from framework.run_config import GameSpec, ProbeSpec, WarmupSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Picklable env factory (used by framework.parallel_eval for issue #229)
# ---------------------------------------------------------------------------

class _SC2EnvFactory:
    """Zero-arg env factory that survives pickle (closures don't).

    Holds the kwargs needed to call ``games.sc2.env.make_env`` and defers
    the heavy SC2 import until ``__call__`` so importing this adapter
    module doesn't pull in PySC2.
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __call__(self):
        from games.sc2.env import make_env
        return make_env(**self._kwargs)



def _get_obs_spec(map_name: str, preset: str | None, enable_belief: bool):
    """Return the obs spec for *map_name*, extending with belief dims when requested."""
    from games.sc2.obs_spec import get_spec
    obs_spec = get_spec(map_name, preset=preset)
    if enable_belief:
        from pathlib import Path
        from games.sc2.belief_schema import (
            load_belief_config, extend_obs_spec as _extend_belief,
        )
        _bcfg = load_belief_config(
            Path(__file__).parent / "config" / "belief_config.yaml"
        )
        obs_spec = _extend_belief(obs_spec, _bcfg)
    return obs_spec


class SC2Adapter:
    name = "sc2"
    config_dir = "games/sc2/config"

    def _map_name(self, training_params: dict, track_override: str | None) -> str:
        if track_override:
            return track_override
        return training_params.get("map_name", "MoveToBeacon")

    def experiment_dir(
        self, experiment_name: str, training_params: dict,
        track_override: str | None,
    ) -> str:
        map_name = self._map_name(training_params, track_override)
        policy = training_params.get("policy_type", "sc2_genetic")
        return f"experiments/sc2/{policy}/{map_name}/{experiment_name}"

    def experiment_dir_root(
        self, training_params: dict, track_override: str | None,
    ) -> str:
        map_name = self._map_name(training_params, track_override)
        policy = training_params.get("policy_type", "sc2_genetic")
        return f"experiments/sc2/{policy}/{map_name}"

    def track_label(
        self, training_params: dict, track_override: str | None,
    ) -> str:
        map_name = self._map_name(training_params, track_override)
        return f"sc2_{map_name}"

    def decorate_reward_cfg(
        self, reward_cfg: dict, training_params: dict,
        track_override: str | None,
    ) -> None:
        pass

    def build_game_spec(
        self, experiment_name: str, experiment_dir: str,
        weights_file: str, reward_cfg_file: str,
        training_params: dict, track_override: str | None,
    ) -> GameSpec:
        from games.sc2.actions import DISCRETE_ACTIONS
        from games.sc2.analytics import save_experiment_results
        # Side-effect imports register the SC2 policy types in POLICY_REGISTRY
        # so train_rl._make_policy can resolve them.
        import games.sc2.sc2_policies  # noqa: F401 — sc2_genetic/reinforce/cmaes/lstm/neural_net/neural_dqn
        import games.sc2.cnn_policy    # noqa: F401 — sc2_cnn

        map_name = self._map_name(training_params, track_override)
        obs_spec_preset = training_params.get("obs_spec_preset")
        enable_belief = training_params.get("enable_belief", False)
        obs_spec = _get_obs_spec(map_name, obs_spec_preset, enable_belief)

        policy_type = training_params.get("policy_type", "sc2_genetic")
        # Spatial obs (dict observation space) is only supported by sc2_cnn.
        # All other SC2 policies operate on flat np.ndarray observations; if
        # the user accidentally left non-empty screen_layers in their config
        # we silently ignore them to avoid crashing those policies.
        if policy_type == "sc2_cnn":
            screen_layers  = training_params.get("screen_layers") or []
            minimap_layers = training_params.get("minimap_layers") or []
            n_channels = len(screen_layers) + len(minimap_layers)
            # Fail fast here (before the SC2 env is launched) on a misconfigured
            # sc2_cnn run with no spatial layers.
            if n_channels == 0:
                raise ValueError(
                    "sc2_cnn requires at least one spatial layer.  "
                    "Set screen_layers in training_params.yaml."
                )
            # The CNN policy needs the spatial channel count at construction
            # time, but train_rl only forwards policy_params to the registry's
            # make().  Inject it under a private key (ignored by param
            # validation, which skips leading-underscore keys).
            pp = training_params.get("policy_params")
            if not isinstance(pp, dict):
                pp = {}
                training_params["policy_params"] = pp
            pp["_n_channels"] = n_channels
        else:
            screen_layers  = []
            minimap_layers = []

        # Use a picklable factory class (not a closure) so the
        # multiprocessing.spawn workers in framework.parallel_eval can
        # reconstruct it after pickling.  Issue #229.
        make_env_fn = _SC2EnvFactory(
            experiment_dir=experiment_dir,
            map_name=map_name,
            max_episode_time_s=training_params["in_game_episode_s"],
            step_mul=training_params.get("step_mul", 1),
            screen_size=training_params.get("screen_size", 64),
            minimap_size=training_params.get("minimap_size", 64),
            agent_race=training_params.get("agent_race", "random"),
            bot_difficulty=training_params.get("bot_difficulty", "very_easy"),
            screen_layers=screen_layers,
            minimap_layers=minimap_layers,
            obs_spec_preset=obs_spec_preset,
            enable_belief=enable_belief,
            max_apm=training_params.get("max_apm", None),
            apm_burst_s=training_params.get("apm_burst_s", 2.0),
        )

        return GameSpec(
            experiment_name=experiment_name,
            track=self.track_label(training_params, track_override),
            make_env_fn=make_env_fn,
            obs_spec=obs_spec,
            head_names=["fn_idx", "x", "y", "queue"],
            discrete_actions=DISCRETE_ACTIONS,
            weights_file=weights_file,
            reward_config_file=reward_cfg_file,
            save_results_fn=save_experiment_results,
            game_name=self.name,
        )

    def build_probe(self, training_params: dict) -> ProbeSpec | None:
        return None

    def build_warmup(self, training_params: dict) -> WarmupSpec | None:
        return None



def make_adapter() -> SC2Adapter:
    return SC2Adapter()
