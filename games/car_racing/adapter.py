"""CarRacing game adapter — builds config bundles for train_rl."""

from __future__ import annotations

from framework.run_config import GameSpec, ProbeSpec, WarmupSpec


class CarRacingAdapter:
    name = "car_racing"
    config_dir = "games/car_racing/config"

    def experiment_dir(
        self,
        experiment_name: str,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        policy = training_params.get("policy_type", "hill_climbing")
        track = self.track_label(training_params, track_override)
        return f"experiments/car_racing/{policy}/{track}/{experiment_name}"

    def experiment_dir_root(
        self,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        policy = training_params.get("policy_type", "hill_climbing")
        track = self.track_label(training_params, track_override)
        return f"experiments/car_racing/{policy}/{track}"

    def track_label(
        self,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        return track_override or "car_racing"

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
        from games.car_racing.actions import DISCRETE_ACTIONS
        from games.car_racing.analytics import save_experiment_results
        from games.car_racing.obs_spec import build_car_racing_obs_spec_from_steps, build_lookahead_steps

        # Configurable lookahead schedule (mirrors games/tmnf/adapter.py).
        # Leaving both keys unset reproduces the legacy 3-point schedule exactly.
        lookahead_steps = build_lookahead_steps(
            n_lookahead_points=training_params.get("n_lookahead_points"),
            lookahead_step_spacing=training_params.get("lookahead_step_spacing"),
        )
        obs_spec = build_car_racing_obs_spec_from_steps(lookahead_steps)

        def _make_env():
            from games.car_racing.env import make_env

            return make_env(
                experiment_dir=experiment_dir,
                max_episode_time_s=training_params["in_game_episode_s"],
                lookahead_steps=lookahead_steps,
            )

        return GameSpec(
            experiment_name=experiment_name,
            track=self.track_label(training_params, track_override),
            make_env_fn=_make_env,
            obs_spec=obs_spec,
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


def make_adapter() -> CarRacingAdapter:
    return CarRacingAdapter()
