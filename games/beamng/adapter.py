"""BeamNG game adapter — builds config bundles for train_rl."""

from __future__ import annotations

from framework.run_config import GameSpec, ProbeSpec, WarmupSpec, PolicyExtras


class BeamNGAdapter:
    name = "beamng"
    config_dir = "games/beamng/config"

    def experiment_dir(
        self, experiment_name: str, training_params: dict,
        track_override: str | None,
    ) -> str:
        return f"experiments/beamng/{experiment_name}"

    def experiment_dir_root(
        self, training_params: dict, track_override: str | None,
    ) -> str:
        return "experiments/beamng"

    def track_label(
        self, training_params: dict, track_override: str | None,
    ) -> str:
        return track_override or "beamng"

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
        from games.beamng.obs_spec import BEAMNG_OBS_SPEC
        from games.beamng.actions import DISCRETE_ACTIONS
        from games.beamng.analytics import save_experiment_results

        def _make_env():
            from games.beamng.env import make_env
            return make_env(
                experiment_dir=experiment_dir,
                max_episode_time_s=training_params["in_game_episode_s"],
            )

        return GameSpec(
            experiment_name=experiment_name,
            track=self.track_label(training_params, track_override),
            make_env_fn=_make_env,
            obs_spec=BEAMNG_OBS_SPEC,
            head_names=["steer", "accel", "brake"],
            discrete_actions=DISCRETE_ACTIONS,
            weights_file=weights_file,
            reward_config_file=reward_cfg_file,
            save_results_fn=save_experiment_results,
        )

    def build_probe(self, training_params: dict) -> ProbeSpec | None:
        return None

    def build_warmup(self, training_params: dict) -> WarmupSpec | None:
        return None

    def build_extras(
        self, weights_file: str, training_params: dict, re_initialize: bool,
    ) -> PolicyExtras | None:
        return None


def make_adapter() -> BeamNGAdapter:
    return BeamNGAdapter()
