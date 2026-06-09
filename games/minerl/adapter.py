"""MineRL game adapter — builds config bundles for train_rl."""

from __future__ import annotations

from framework.run_config import GameSpec, ProbeSpec, WarmupSpec


def _sanitize(name: str) -> str:
    """Make *name* safe for use as a directory component."""
    return name.replace("/", "_").replace(" ", "_")


class MineRLAdapter:
    name = "minerl"
    config_dir = "games/minerl/config"

    def experiment_dir(
        self,
        experiment_name: str,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        policy = training_params.get("policy_type", "genetic")
        track = self.track_label(training_params, track_override)
        return f"experiments/minerl/{policy}/{track}/{experiment_name}"

    def experiment_dir_root(
        self,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        policy = training_params.get("policy_type", "genetic")
        track = self.track_label(training_params, track_override)
        return f"experiments/minerl/{policy}/{track}"

    def track_label(
        self,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        raw = track_override or training_params.get("map_name", "MineRLNavigateDense-v0")
        return _sanitize(raw)

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
        from games.minerl.actions import DISCRETE_ACTIONS
        from games.minerl.analytics import save_experiment_results
        from games.minerl.obs_spec import MINERL_OBS_SPEC

        map_name = track_override or training_params.get("map_name", "MineRLNavigateDense-v0")

        def _make_env():
            from games.minerl.env import make_env

            return make_env(
                experiment_dir=experiment_dir,
                map_name=map_name,
                max_episode_time_s=training_params["in_game_episode_s"],
            )

        return GameSpec(
            experiment_name=experiment_name,
            track=self.track_label(training_params, track_override),
            make_env_fn=_make_env,
            obs_spec=MINERL_OBS_SPEC,
            head_names=["action"],
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


def make_adapter() -> MineRLAdapter:
    return MineRLAdapter()
