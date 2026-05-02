"""StarCraft 2 game adapter — builds config bundles for train_rl."""

from __future__ import annotations

import logging
import os

import yaml

from framework.run_config import GameSpec, ProbeSpec, WarmupSpec, PolicyExtras

logger = logging.getLogger(__name__)


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

        screen_layers   = training_params.get("screen_layers") or []
        minimap_layers  = training_params.get("minimap_layers") or []

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
                screen_layers=screen_layers,
                minimap_layers=minimap_layers,
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
        from games.sc2.obs_spec import get_spec
        from games.sc2.sc2_policies import SC2GeneticPolicy

        map_name = training_params.get("map_name", "MoveToBeacon")
        obs_spec = get_spec(map_name)
        policy_params = training_params.get("policy_params") or {}
        trainer_state_file = os.path.join(
            os.path.dirname(weights_file), "trainer_state.npz",
        )

        def _make_sc2_genetic() -> SC2GeneticPolicy:
            pop_size = policy_params.get("population_size", 30)
            elite_k = policy_params.get("elite_k", 5)
            policy = SC2GeneticPolicy(
                obs_spec=obs_spec,
                population_size=pop_size,
                elite_k=elite_k,
                mutation_scale=policy_params.get("mutation_scale", 0.1),
                mutation_share=policy_params.get("mutation_share", 0.3),
                eval_episodes=policy_params.get("eval_episodes", 2),
            )
            if os.path.exists(weights_file) and not re_initialize:
                policy.initialize_from_file(weights_file)
            else:
                policy.initialize_random()
                logger.info("[SC2GeneticPolicy] random population of %d", pop_size)
            return policy

        def _make_neural_dqn():
            from games.sc2.policies import NeuralDQNPolicy as SC2NeuralDQNPolicy
            if os.path.exists(weights_file) and not re_initialize:
                with open(weights_file) as _f:
                    _cfg = yaml.safe_load(_f) or {}
                if isinstance(_cfg, dict) and _cfg.get("policy_type") == "neural_dqn":
                    policy = SC2NeuralDQNPolicy.from_cfg(_cfg, obs_spec)
                    if os.path.exists(trainer_state_file):
                        try:
                            policy.load_trainer_state(trainer_state_file)
                            logger.info("[SC2 NeuralDQNPolicy] loaded trainer state from %s",
                                        trainer_state_file)
                        except (ValueError, KeyError) as exc:
                            logger.warning(
                                "[SC2 NeuralDQNPolicy] could not load trainer state — %s; "
                                "continuing with default state.", exc,
                            )
                    return policy
            return SC2NeuralDQNPolicy(
                obs_spec=obs_spec,
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
            )

        def _make_cmaes():
            from games.sc2.policies import (
                CMAESPolicy as SC2CMAESPolicy,
                SC2LinearPolicy,
            )
            _head_names = ["fn_idx", "x", "y", "queue"]
            pop_size = policy_params.get("population_size", 20)
            sigma = policy_params.get("initial_sigma", 0.3)
            policy = SC2CMAESPolicy(
                obs_spec=obs_spec,
                head_names=_head_names,
                population_size=pop_size,
                initial_sigma=sigma,
                eval_episodes=policy_params.get("eval_episodes", 1),
            )
            if os.path.exists(weights_file) and not re_initialize:
                champion = SC2LinearPolicy(obs_spec, _head_names, weights_file)
                policy.initialize_from_champion(champion)
                if os.path.exists(trainer_state_file):
                    try:
                        policy.load_trainer_state(trainer_state_file)
                        logger.info("[SC2 CMAESPolicy] loaded trainer state from %s",
                                    trainer_state_file)
                    except (ValueError, KeyError) as exc:
                        logger.warning(
                            "[SC2 CMAESPolicy] could not load trainer state — %s; "
                            "continuing with champion weights and default distribution.", exc,
                        )
            else:
                policy.initialize_random()
            return policy

        def _make_reinforce():
            from games.sc2.policies import REINFORCEPolicy as SC2REINFORCEPolicy
            if os.path.exists(weights_file) and not re_initialize:
                with open(weights_file) as _f:
                    _cfg = yaml.safe_load(_f) or {}
                if isinstance(_cfg, dict) and _cfg.get("policy_type") == "reinforce":
                    policy = SC2REINFORCEPolicy.from_cfg(_cfg, obs_spec)
                    if os.path.exists(trainer_state_file):
                        try:
                            policy.load_trainer_state(trainer_state_file)
                            logger.info("[SC2 REINFORCEPolicy] loaded trainer state from %s",
                                        trainer_state_file)
                        except (ValueError, KeyError) as exc:
                            logger.warning(
                                "[SC2 REINFORCEPolicy] could not load trainer state — %s; "
                                "continuing with default state.", exc,
                            )
                    return policy
            return SC2REINFORCEPolicy(
                obs_spec=obs_spec,
                hidden_sizes=policy_params.get("hidden_sizes", [64, 64]),
                learning_rate=policy_params.get("learning_rate", 0.001),
                gamma=policy_params.get("gamma", 0.99),
                entropy_coeff=policy_params.get("entropy_coeff", 0.01),
                baseline=policy_params.get("baseline", "running_mean"),
            )

        def _make_lstm():
            from games.sc2.policies import (
                LSTMPolicy as SC2LSTMPolicy,
                LSTMEvolutionPolicy as SC2LSTMEvolutionPolicy,
            )
            hidden_size = policy_params.get("hidden_size", 32)
            pop_size = policy_params.get("population_size", 20)
            sigma = policy_params.get("initial_sigma", 0.05)
            policy = SC2LSTMEvolutionPolicy(
                obs_spec=obs_spec,
                hidden_size=hidden_size,
                population_size=pop_size,
                initial_sigma=sigma,
            )
            if os.path.exists(weights_file) and not re_initialize:
                with open(weights_file) as _f:
                    _cfg = yaml.safe_load(_f) or {}
                if isinstance(_cfg, dict) and _cfg.get("policy_type") == "lstm":
                    champion = SC2LSTMPolicy.from_cfg(_cfg, obs_spec)
                    try:
                        policy.initialize_from_champion(champion)
                    except ValueError as exc:
                        logger.warning(
                            "[SC2 LSTMEvolutionPolicy] incompatible saved champion — %s; "
                            "starting from random.", exc,
                        )
                if os.path.exists(trainer_state_file):
                    try:
                        policy.load_trainer_state(trainer_state_file)
                        logger.info("[SC2 LSTMEvolutionPolicy] loaded trainer state from %s",
                                    trainer_state_file)
                    except (ValueError, KeyError) as exc:
                        logger.warning(
                            "[SC2 LSTMEvolutionPolicy] could not load trainer state — %s; "
                            "continuing.", exc,
                        )
            return policy

        screen_layers_extras  = training_params.get("screen_layers") or []
        minimap_layers_extras = training_params.get("minimap_layers") or []

        def _make_sc2_cnn():
            from games.sc2.cnn_policy import SC2CNNEvolutionPolicy
            n_channels = len(screen_layers_extras) + len(minimap_layers_extras)
            if n_channels == 0:
                raise ValueError(
                    "sc2_cnn requires at least one spatial layer.  "
                    "Set screen_layers in training_params.yaml."
                )
            pop_size = policy_params.get("population_size", 20)
            sigma    = policy_params.get("initial_sigma", 0.01)
            policy   = SC2CNNEvolutionPolicy(
                n_channels      = n_channels,
                obs_spec        = obs_spec,
                population_size = pop_size,
                initial_sigma   = sigma,
                eval_episodes   = policy_params.get("eval_episodes", 1),
            )
            champion_path = weights_file.replace(".yaml", ".npz")
            if os.path.exists(champion_path) and not re_initialize:
                try:
                    policy.load_champion(champion_path)
                    if os.path.exists(trainer_state_file):
                        policy.load_trainer_state(trainer_state_file)
                        logger.info("[SC2CNNEvolutionPolicy] loaded trainer state from %s",
                                    trainer_state_file)
                except (ValueError, KeyError) as exc:
                    logger.warning(
                        "[SC2CNNEvolutionPolicy] could not load saved state — %s; "
                        "starting from random.", exc,
                    )
            return policy

        return PolicyExtras(
            factories={
                "sc2_genetic": _make_sc2_genetic,
                "neural_dqn": _make_neural_dqn,
                "cmaes": _make_cmaes,
                "reinforce": _make_reinforce,
                "lstm": _make_lstm,
                "sc2_cnn": _make_sc2_cnn,
            },
            loop_dispatch={
                "sc2_genetic": "genetic",
                "neural_dqn": "q_learning",
                "cmaes": "cmaes",
                "reinforce": "q_learning",
                "lstm": "cmaes",
                "sc2_cnn": "cmaes",
            },
        )


def make_adapter() -> SC2Adapter:
    return SC2Adapter()
