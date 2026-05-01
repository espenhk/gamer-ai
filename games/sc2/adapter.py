"""StarCraft 2 game adapter — builds config bundles for train_rl."""

from __future__ import annotations

from framework.run_config import GameSpec, ProbeSpec, WarmupSpec, PolicyExtras


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
        return f"experiments/sc2_{map_name}/{experiment_name}"

    def experiment_dir_root(
        self, training_params: dict, track_override: str | None,
    ) -> str:
        map_name = self._map_name(training_params, track_override)
        return f"experiments/sc2_{map_name}"

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
        from games.sc2.obs_spec import get_spec
        from games.sc2.actions import DISCRETE_ACTIONS
        from games.sc2.analytics import save_experiment_results

        map_name = self._map_name(training_params, track_override)
        obs_spec = get_spec(map_name)

        def _make_env():
            from games.sc2.env import make_env
            return make_env(
                experiment_dir=experiment_dir,
                map_name=map_name,
                max_episode_time_s=training_params["in_game_episode_s"],
                step_mul=training_params.get("step_mul", 8),
                screen_size=training_params.get("screen_size", 64),
                minimap_size=training_params.get("minimap_size", 64),
                agent_race=training_params.get("agent_race", "random"),
                bot_difficulty=training_params.get("bot_difficulty", "very_easy"),
            )

        return GameSpec(
            experiment_name=experiment_name,
            track=self.track_label(training_params, track_override),
            make_env_fn=_make_env,
            obs_spec=obs_spec,
            head_names=["fn_idx", "x", "y", "queue"],
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


def make_adapter() -> SC2Adapter:
    return SC2Adapter()
