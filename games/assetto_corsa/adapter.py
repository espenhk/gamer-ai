"""Assetto Corsa game adapter — builds config bundles for train_rl."""

from __future__ import annotations

from framework.run_config import GameSpec, ProbeSpec, WarmupSpec


class AssettoCorsaAdapter:
    name = "assetto"
    config_dir = "games/assetto_corsa/config"

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def experiment_dir(
        self,
        experiment_name: str,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        policy = training_params.get("policy_type", "genetic")
        track = self.track_label(training_params, track_override)
        return f"experiments/assetto_corsa/{policy}/{track}/{experiment_name}"

    def experiment_dir_root(
        self,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        policy = training_params.get("policy_type", "genetic")
        track = self.track_label(training_params, track_override)
        return f"experiments/assetto_corsa/{policy}/{track}"

    def track_label(
        self,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        return track_override or training_params.get("track", "assetto_default")

    # ------------------------------------------------------------------
    # Reward config decoration
    # ------------------------------------------------------------------

    def decorate_reward_cfg(
        self,
        reward_cfg: dict,
        training_params: dict,
        track_override: str | None,
    ) -> None:
        pass  # AC doesn't load a track .npy; no extra keys needed

    # ------------------------------------------------------------------
    # GameSpec
    # ------------------------------------------------------------------

    def build_game_spec(
        self,
        experiment_name: str,
        experiment_dir: str,
        weights_file: str,
        reward_cfg_file: str,
        training_params: dict,
        track_override: str | None,
    ) -> GameSpec:
        from games.assetto_corsa.actions import DISCRETE_ACTIONS
        from games.assetto_corsa.analytics import save_experiment_results
        from games.assetto_corsa.obs_spec import with_vision

        n_vision = int(training_params.get("n_vision", 0))
        obs_spec = with_vision(n_vision)

        def _make_env():
            from games.assetto_corsa.env import make_env

            return make_env(
                experiment_dir=experiment_dir,
                speed=training_params.get("speed", 1.0),
                in_game_episode_s=training_params["in_game_episode_s"],
                n_vision=n_vision,
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

    # ------------------------------------------------------------------
    # Probe / warmup
    # ------------------------------------------------------------------

    def build_probe(self, training_params: dict) -> ProbeSpec | None:
        from games.assetto_corsa.actions import PROBE_ACTIONS

        return ProbeSpec(
            actions=PROBE_ACTIONS,
            probe_in_game_s=training_params.get("probe_s", 15.0),
            cold_start_restarts=training_params.get("cold_restarts", 5),
            cold_start_sims=training_params.get("cold_sims", 5),
        )

    def build_warmup(self, training_params: dict) -> WarmupSpec | None:
        from games.assetto_corsa.actions import WARMUP_ACTION

        return WarmupSpec(action=WARMUP_ACTION, steps=5)


def make_adapter() -> AssettoCorsaAdapter:
    return AssettoCorsaAdapter()
