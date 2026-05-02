"""TMNF game adapter — builds config bundles for train_rl.

Owns the probe, warmup, and policy-extras wiring previously duplicated
in main.py::_run_tmnf and grid_search.py::_build_tmnf_extras.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import yaml

from framework.run_config import GameSpec, ProbeSpec, WarmupSpec, PolicyExtras

logger = logging.getLogger(__name__)


class TMNFAdapter:
    name = "tmnf"
    config_dir = "games/tmnf/config"

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def experiment_dir(
        self, experiment_name: str, training_params: dict,
        track_override: str | None,
    ) -> str:
        track = self.track_label(training_params, track_override)
        return f"experiments/{track}/{experiment_name}"

    def experiment_dir_root(
        self, training_params: dict, track_override: str | None,
    ) -> str:
        track = self.track_label(training_params, track_override)
        return f"experiments/{track}"

    def track_label(
        self, training_params: dict, track_override: str | None,
    ) -> str:
        return track_override or training_params.get("track", "a03_centerline")

    # ------------------------------------------------------------------
    # Reward config decoration
    # ------------------------------------------------------------------

    def decorate_reward_cfg(
        self, reward_cfg: dict, training_params: dict,
        track_override: str | None,
    ) -> None:
        track = self.track_label(training_params, track_override)
        reward_cfg["track_name"] = track
        reward_cfg["centerline_path"] = f"games/tmnf/tracks/{track}.npy"

    # ------------------------------------------------------------------
    # GameSpec
    # ------------------------------------------------------------------

    def build_game_spec(
        self, experiment_name: str, experiment_dir: str,
        weights_file: str, reward_cfg_file: str,
        training_params: dict, track_override: str | None,
    ) -> GameSpec:
        from games.tmnf.obs_spec import TMNF_OBS_SPEC
        from games.tmnf.actions import DISCRETE_ACTIONS
        from games.tmnf.analytics import save_experiment_results

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
        )

    # ------------------------------------------------------------------
    # Probe / warmup / extras
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

    def build_extras(
        self, weights_file: str, training_params: dict, re_initialize: bool,
    ) -> PolicyExtras | None:
        from games.tmnf.policies import (
            CMAESPolicy,
            LSTMEvolutionPolicy,
            LSTMPolicy,
            NeuralDQNPolicy,
            REINFORCEPolicy,
            WeightedLinearPolicy,
        )

        n_lidar_rays = training_params.get("n_lidar_rays", 0)
        policy_params = training_params.get("policy_params") or {}
        trainer_state_file = os.path.join(
            os.path.dirname(os.path.abspath(weights_file)), "trainer_state.npz"
        )

        def _make_neural_dqn() -> NeuralDQNPolicy:
            if os.path.exists(weights_file) and not re_initialize:
                with open(weights_file) as _f:
                    _cfg = yaml.safe_load(_f)
                if isinstance(_cfg, dict) and _cfg.get("policy_type") == "neural_dqn":
                    policy = NeuralDQNPolicy.from_cfg(_cfg, n_lidar_rays=n_lidar_rays)
                    if os.path.exists(trainer_state_file):
                        try:
                            policy.load_trainer_state(trainer_state_file)
                            logger.info("[NeuralDQNPolicy] loaded trainer state from %s",
                                        trainer_state_file)
                        except (ValueError, KeyError) as exc:
                            logger.warning(
                                "[NeuralDQNPolicy] could not load trainer state from %s — %s; "
                                "continuing with default state.",
                                trainer_state_file, exc,
                            )
                    return policy
            return NeuralDQNPolicy(
                hidden_sizes=policy_params.get("hidden_sizes", [64, 64]),
                replay_buffer_size=policy_params.get("replay_buffer_size", 10000),
                batch_size=policy_params.get("batch_size", 64),
                min_replay_size=policy_params.get("min_replay_size", 500),
                target_update_freq=policy_params.get("target_update_freq", 200),
                learning_rate=policy_params.get("learning_rate", 0.001),
                epsilon_start=policy_params.get("epsilon_start", 1.0),
                epsilon_end=policy_params.get("epsilon_end", 0.05),
                epsilon_decay_steps=policy_params.get("epsilon_decay_steps", 5000),
                gamma=policy_params.get("gamma", 0.99),
                n_lidar_rays=n_lidar_rays,
            )

        def _make_cmaes() -> CMAESPolicy:
            policy = CMAESPolicy(
                population_size=policy_params.get("population_size", 20),
                initial_sigma=policy_params.get("initial_sigma", 0.3),
                n_lidar_rays=n_lidar_rays,
                eval_episodes=policy_params.get("eval_episodes", 1),
            )
            if os.path.exists(weights_file) and not re_initialize:
                with open(weights_file) as _f:
                    champion = WeightedLinearPolicy.from_cfg(
                        yaml.safe_load(_f) or {}, n_lidar_rays=n_lidar_rays
                    )
                policy.initialize_from_champion(champion)
                if os.path.exists(trainer_state_file):
                    try:
                        policy.load_trainer_state(trainer_state_file)
                        logger.info("[CMAESPolicy] loaded trainer state from %s",
                                    trainer_state_file)
                    except (ValueError, KeyError) as exc:
                        logger.warning(
                            "[CMAESPolicy] could not load trainer state from %s — %s; "
                            "continuing with champion weights and default distribution.",
                            trainer_state_file, exc,
                        )
            else:
                policy.initialize_random()
            return policy

        def _make_reinforce() -> REINFORCEPolicy:
            if os.path.exists(weights_file) and not re_initialize:
                with open(weights_file) as _f:
                    _cfg = yaml.safe_load(_f) or {}
                if isinstance(_cfg, dict) and _cfg.get("policy_type") == "reinforce":
                    policy = REINFORCEPolicy.from_cfg(_cfg, n_lidar_rays=n_lidar_rays)
                    if os.path.exists(trainer_state_file):
                        try:
                            policy.load_trainer_state(trainer_state_file)
                            logger.info("[REINFORCEPolicy] loaded trainer state from %s",
                                        trainer_state_file)
                        except (ValueError, KeyError) as exc:
                            logger.warning(
                                "[REINFORCEPolicy] could not load trainer state from %s — %s; "
                                "continuing with default state.",
                                trainer_state_file, exc,
                            )
                    return policy
            return REINFORCEPolicy(
                hidden_sizes=policy_params.get("hidden_sizes", [64, 64]),
                learning_rate=policy_params.get("learning_rate", 0.001),
                gamma=policy_params.get("gamma", 0.99),
                entropy_coeff=policy_params.get("entropy_coeff", 0.01),
                baseline=policy_params.get("baseline", "running_mean"),
                n_lidar_rays=n_lidar_rays,
            )

        def _make_lstm() -> LSTMEvolutionPolicy:
            hidden_size = policy_params.get("hidden_size", 32)
            policy = LSTMEvolutionPolicy(
                hidden_size=hidden_size,
                population_size=policy_params.get("population_size", 20),
                initial_sigma=policy_params.get("initial_sigma", 0.05),
                n_lidar_rays=n_lidar_rays,
            )
            if os.path.exists(weights_file) and not re_initialize:
                with open(weights_file) as _f:
                    _cfg = yaml.safe_load(_f) or {}
                if isinstance(_cfg, dict) and _cfg.get("policy_type") == "lstm":
                    saved_hidden = _cfg.get("hidden_size")
                    saved_lidar = _cfg.get("n_lidar_rays")
                    if saved_hidden is not None and saved_hidden != hidden_size:
                        raise ValueError(
                            "Saved LSTM champion hidden_size does not match current run: "
                            f"saved={saved_hidden}, current={hidden_size}"
                        )
                    if saved_lidar is not None and saved_lidar != n_lidar_rays:
                        raise ValueError(
                            "Saved LSTM champion n_lidar_rays does not match current run: "
                            f"saved={saved_lidar}, current={n_lidar_rays}"
                        )
                    policy.initialize_from_champion(LSTMPolicy.from_cfg(_cfg))
                    if os.path.exists(trainer_state_file):
                        try:
                            policy.load_trainer_state(trainer_state_file)
                            logger.info("[LSTMEvolutionPolicy] loaded trainer state from %s",
                                        trainer_state_file)
                        except (ValueError, KeyError) as exc:
                            logger.warning(
                                "[LSTMEvolutionPolicy] could not load trainer state from %s — %s; "
                                "continuing with champion weights and default distribution.",
                                trainer_state_file, exc,
                            )
            return policy

        return PolicyExtras(
            factories={
                "neural_dqn": _make_neural_dqn,
                "cmaes": _make_cmaes,
                "reinforce": _make_reinforce,
                "lstm": _make_lstm,
            },
            loop_dispatch={
                "neural_dqn": "q_learning",
                "cmaes": "cmaes",
                "reinforce": "q_learning",
                "lstm": "cmaes",
            },
        )


def make_adapter() -> TMNFAdapter:
    return TMNFAdapter()
