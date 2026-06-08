"""Gymnasium classic-control game adapter."""

from __future__ import annotations

from framework.run_config import GameSpec, ProbeSpec, WarmupSpec


class GymClassicAdapter:
    name = "gym_classic"
    config_dir = "games/gym_classic/config"

    def experiment_dir(
        self,
        experiment_name: str,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        policy = training_params.get("policy_type", "genetic")
        track = self.track_label(training_params, track_override)
        return f"experiments/gym_classic/{policy}/{track}/{experiment_name}"

    def experiment_dir_root(
        self,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        policy = training_params.get("policy_type", "genetic")
        track = self.track_label(training_params, track_override)
        return f"experiments/gym_classic/{policy}/{track}"

    def track_label(
        self,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        return track_override or training_params.get("map_name", "CartPole-v1")

    def decorate_reward_cfg(
        self,
        reward_cfg: dict,
        training_params: dict,
        track_override: str | None,
    ) -> None:
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
        from games.gym_classic.actions import get_discrete_actions
        from games.gym_classic.analytics import save_experiment_results
        from games.gym_classic.env import make_env
        from games.gym_classic.obs_spec import get_obs_spec

        map_name = track_override or training_params.get("map_name", "CartPole-v1")
        render_mode = training_params.get("render_mode", None)
        obs_spec = get_obs_spec(map_name)
        discrete_actions = get_discrete_actions(map_name)

        def _make_env():
            return make_env(
                experiment_dir=experiment_dir,
                map_name=map_name,
                max_episode_time_s=training_params.get("in_game_episode_s", 10.0),
                render_mode=render_mode,
            )

        return GameSpec(
            experiment_name=experiment_name,
            track=self.track_label(training_params, track_override),
            make_env_fn=_make_env,
            obs_spec=obs_spec,
            head_names=["action"],
            discrete_actions=discrete_actions,
            weights_file=weights_file,
            reward_config_file=reward_cfg_file,
            save_results_fn=save_experiment_results,
            game_name=self.name,
        )

    def build_probe(self, training_params: dict) -> ProbeSpec | None:
        return None

    def build_warmup(self, training_params: dict) -> WarmupSpec | None:
        return None


def make_adapter() -> GymClassicAdapter:
    return GymClassicAdapter()
