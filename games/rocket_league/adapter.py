"""Rocket League game adapter — builds config bundles for train_rl."""

from __future__ import annotations

from framework.run_config import GameSpec, ProbeSpec, WarmupSpec


class RocketLeagueAdapter:
    name = "rocket_league"
    config_dir = "games/rocket_league/config"

    def experiment_dir(
        self, experiment_name: str, training_params: dict,
        track_override: str | None,
    ) -> str:
        policy = training_params.get("policy_type", "genetic")
        track = self.track_label(training_params, track_override)
        return f"experiments/rocket_league/{policy}/{track}/{experiment_name}"

    def experiment_dir_root(
        self, training_params: dict, track_override: str | None,
    ) -> str:
        policy = training_params.get("policy_type", "genetic")
        track = self.track_label(training_params, track_override)
        return f"experiments/rocket_league/{policy}/{track}"

    def track_label(
        self, training_params: dict, track_override: str | None,
    ) -> str:
        return track_override or "rocket_league"

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
        from games.rocket_league.obs_spec import ROCKET_LEAGUE_OBS_SPEC
        from games.rocket_league.actions import DISCRETE_ACTIONS
        from games.rocket_league.analytics import save_experiment_results

        tick_skip = training_params.get("tick_skip", 8)

        def _make_env():
            from games.rocket_league.env import make_env
            return make_env(
                experiment_dir=experiment_dir,
                max_episode_time_s=training_params["in_game_episode_s"],
                tick_skip=int(tick_skip),
            )

        return GameSpec(
            experiment_name=experiment_name,
            track=self.track_label(training_params, track_override),
            make_env_fn=_make_env,
            obs_spec=ROCKET_LEAGUE_OBS_SPEC,
            head_names=["throttle", "steer", "pitch", "yaw", "roll",
                        "jump", "boost", "handbrake"],
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


def make_adapter() -> RocketLeagueAdapter:
    return RocketLeagueAdapter()
