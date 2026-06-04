"""TMNF game adapter — builds config bundles for train_rl."""

from __future__ import annotations

import logging

from framework.run_config import GameSpec, ProbeSpec, WarmupSpec

logger = logging.getLogger(__name__)


class TMNFAdapter:
    name = "tmnf"
    config_dir = "games/tmnf/config"

    def __init__(self) -> None:
        from games.tmnf.bc_adapter import TMNFBCAdapter

        self.bc = TMNFBCAdapter()

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def experiment_dir(
        self,
        experiment_name: str,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        track = self.track_label(training_params, track_override)
        policy = training_params.get("policy_type", "hill_climbing")
        return f"experiments/tmnf/{policy}/{track}/{experiment_name}"

    def experiment_dir_root(
        self,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        track = self.track_label(training_params, track_override)
        policy = training_params.get("policy_type", "hill_climbing")
        return f"experiments/tmnf/{policy}/{track}"

    def track_label(
        self,
        training_params: dict,
        track_override: str | None,
    ) -> str:
        return track_override or training_params.get("track", "a03_centerline")

    # ------------------------------------------------------------------
    # Reward config decoration
    # ------------------------------------------------------------------

    def decorate_reward_cfg(
        self,
        reward_cfg: dict,
        training_params: dict,
        track_override: str | None,
    ) -> None:
        track = self.track_label(training_params, track_override)
        reward_cfg["track_name"] = track
        reward_cfg["centerline_path"] = f"games/tmnf/tracks/{track}.npy"

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
        import games.tmnf.policies  # noqa: F401 — side-effect: registers TMNF policy types
        from games.tmnf.actions import DISCRETE_ACTIONS
        from games.tmnf.analytics import save_experiment_results
        from games.tmnf.obs_spec import TMNF_OBS_SPEC

        n_lidar_rays = training_params.get("n_lidar_rays", 0)
        obs_spec = TMNF_OBS_SPEC.with_lidar(n_lidar_rays)
        track = self.track_label(training_params, track_override)

        def _make_env():
            from games.tmnf.env import make_env

            return make_env(
                experiment_dir=experiment_dir,
                speed=training_params["speed"],
                in_game_episode_s=training_params["in_game_episode_s"],
                n_lidar_rays=n_lidar_rays,
                decision_offset_pct=training_params.get("decision_offset_pct", 0.75),
                action_window_ticks=training_params.get("action_window_ticks", 1),
            )

        return GameSpec(
            experiment_name=experiment_name,
            track=track,
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
        from games.tmnf.actions import PROBE_ACTIONS

        return ProbeSpec(
            actions=PROBE_ACTIONS,
            probe_in_game_s=training_params.get("probe_s", 15.0),
            cold_start_restarts=training_params.get("cold_restarts", 20),
            cold_start_sims=training_params.get("cold_sims", 5),
        )

    def build_warmup(self, training_params: dict) -> WarmupSpec | None:
        from games.tmnf.actions import WARMUP_ACTION

        return WarmupSpec(action=WARMUP_ACTION, steps=5)


def make_adapter() -> TMNFAdapter:
    return TMNFAdapter()
