"""iRacing game adapter — builds config bundles for train_rl."""

from __future__ import annotations

from framework.run_config import GameSpec, ProbeSpec, WarmupSpec


class IRacingAdapter:
    name = "iracing"
    config_dir = "games/iracing/config"

    def experiment_dir(
        self, experiment_name: str, training_params: dict,
        track_override: str | None,
    ) -> str:
        policy = training_params.get("policy_type", "hill_climbing")
        track = self.track_label(training_params, track_override)
        return f"experiments/iracing/{policy}/{track}/{experiment_name}"

    def experiment_dir_root(
        self, training_params: dict, track_override: str | None,
    ) -> str:
        policy = training_params.get("policy_type", "hill_climbing")
        track = self.track_label(training_params, track_override)
        return f"experiments/iracing/{policy}/{track}"

    def track_label(
        self, training_params: dict, track_override: str | None,
    ) -> str:
        return track_override or training_params.get("track", "laguna_seca")

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
        from games.iracing.obs_spec import IRACING_OBS_SPEC
        from games.iracing.actions import DISCRETE_ACTIONS
        from games.iracing.analytics import save_experiment_results

        def _make_env():
            from games.iracing.env import make_env
            return make_env(
                experiment_dir=experiment_dir,
                max_episode_time_s=training_params["in_game_episode_s"],
                action_mode=training_params.get("action_mode", "telemetry_only"),
            )

        return GameSpec(
            experiment_name=experiment_name,
            track=self.track_label(training_params, track_override),
            make_env_fn=_make_env,
            obs_spec=IRACING_OBS_SPEC,
            head_names=["steer", "accel", "brake"],
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


def make_adapter() -> IRacingAdapter:
    return IRacingAdapter()
