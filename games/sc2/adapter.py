"""StarCraft 2 game adapter — builds config bundles for train_rl."""

from __future__ import annotations

import logging
import os

import yaml

from framework.run_config import GameSpec, ProbeSpec, WarmupSpec, PolicyExtras

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Picklable env factory (used by framework.parallel_eval for issue #229)
# ---------------------------------------------------------------------------

class _SC2EnvFactory:
    """Zero-arg env factory that survives pickle (closures don't).

    Holds the kwargs needed to call ``games.sc2.env.make_env`` and defers
    the heavy SC2 import until ``__call__`` so importing this adapter
    module doesn't pull in PySC2.
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __call__(self):
        from games.sc2.env import make_env
        return make_env(**self._kwargs)


# ---------------------------------------------------------------------------
# SC2-specific policy validation
# ---------------------------------------------------------------------------

# Policy types structurally incompatible with SC2's action encoding.
# The framework WeightedLinearPolicy clips fn_idx to [-1, 1] and thresholds
# x/y to binary — both wrong for SC2 where fn_idx ∈ [0, 5] and x, y ∈ [0, 1].
_SC2_INCOMPATIBLE_POLICY_TYPES: frozenset[str] = frozenset({
    "hill_climbing",
    "neural_net",
    "genetic",
})

# Migration hint for each incompatible type.
_SC2_MIGRATION_HINT: dict[str, str] = {
    "hill_climbing": "sc2_genetic",
    "neural_net":    "sc2_neural_net",
    "genetic":       "sc2_genetic",
}

# Valid policy_params keys for each SC2-registered policy type.
# All types handled either here or in the factories dict are listed so that
# unknown keys are caught early.  Note: epsilon_greedy and mcts also use
# .get() for each param and silently ignore unknown keys — they are included
# here to provide the same fail-fast guarantee as SC2-native types.
_SC2_VALID_POLICY_PARAMS: dict[str, frozenset[str]] = {
    "sc2_genetic":    frozenset({
        "population_size", "elite_k", "mutation_scale", "mutation_share", "eval_episodes",
    }),
    "neural_dqn":     frozenset({
        "hidden_sizes", "replay_buffer_size", "batch_size", "min_replay_size",
        "target_update_freq", "learning_rate", "epsilon_start", "epsilon_end",
        "epsilon_decay_steps", "gamma",
    }),
    "sc2_neural_dqn": frozenset({
        "hidden_sizes", "replay_buffer_size", "batch_size", "min_replay_size",
        "target_update_freq", "learning_rate", "epsilon_start", "epsilon_end",
        "epsilon_decay_steps", "gamma",
    }),
    "sc2_neural_net": frozenset({"hidden_sizes"}),
    "cmaes":          frozenset({"population_size", "initial_sigma", "eval_episodes"}),
    "reinforce":      frozenset({
        "hidden_sizes", "learning_rate", "gamma", "entropy_coeff", "baseline",
    }),
    "sc2_reinforce":  frozenset({
        "hidden_sizes", "learning_rate", "gamma", "entropy_coeff", "baseline",
    }),
    "lstm":           frozenset({"hidden_size", "population_size", "initial_sigma"}),
    "sc2_cnn":        frozenset({"population_size", "initial_sigma", "eval_episodes"}),
    "sc2_cmaes":      frozenset({"population_size", "initial_sigma", "eval_episodes"}),
    "sc2_lstm":       frozenset({
        "hidden_size", "population_size", "initial_sigma", "reset_on_episode",
    }),
    "epsilon_greedy": frozenset({
        "epsilon", "n_bins", "epsilon_decay", "epsilon_min", "alpha", "gamma",
    }),
    "mcts":           frozenset({"c", "alpha", "gamma", "n_bins"}),
}


def _validate_sc2_policy_config(policy_type: str, policy_params: dict) -> None:
    """Raise ValueError early for misconfigured SC2 policy settings.

    Checks two things:
    1. ``policy_type`` is not in the set of types that are structurally
       incompatible with SC2's action encoding (fn_idx, x, y, queue).
    2. ``policy_params`` contains only keys that are consumed by the chosen
       ``policy_type``; extra keys are silently ignored by each factory, so
       they would otherwise lead to silent misconfiguration.
    """
    if policy_type in _SC2_INCOMPATIBLE_POLICY_TYPES:
        hint = _SC2_MIGRATION_HINT.get(policy_type, "sc2_genetic")
        raise ValueError(
            f"policy_type={policy_type!r} is not compatible with SC2.  "
            f"The framework {policy_type!r} policy clips fn_idx to [-1, 1] "
            f"and thresholds x/y to binary, which is incorrect for SC2 "
            f"(fn_idx ∈ [0, 5], x/y ∈ [0, 1] continuous).  "
            f"Use {hint!r} instead — see CLAUDE.md 'Supported policies' under 'StarCraft 2'."
        )

    if policy_type in _SC2_VALID_POLICY_PARAMS and policy_params:
        valid_keys = _SC2_VALID_POLICY_PARAMS[policy_type]
        unknown_keys = sorted(set(policy_params) - valid_keys)
        if unknown_keys:
            raise ValueError(
                f"policy_params contains keys that have no effect for "
                f"policy_type={policy_type!r}: {unknown_keys}.  "
                f"Valid keys for {policy_type!r}: {sorted(valid_keys)}."
            )


def _get_obs_spec(map_name: str, preset: str | None, enable_belief: bool):
    """Return the obs spec for *map_name*, extending with belief dims when requested."""
    from games.sc2.obs_spec import get_spec
    obs_spec = get_spec(map_name, preset=preset)
    if enable_belief:
        from pathlib import Path
        from games.sc2.belief_schema import (
            load_belief_config, extend_obs_spec as _extend_belief,
        )
        _bcfg = load_belief_config(
            Path(__file__).parent / "config" / "belief_config.yaml"
        )
        obs_spec = _extend_belief(obs_spec, _bcfg)
    return obs_spec


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
        from games.sc2.actions import DISCRETE_ACTIONS
        from games.sc2.analytics import save_experiment_results

        map_name = self._map_name(training_params, track_override)
        obs_spec_preset = training_params.get("obs_spec_preset")
        enable_belief = training_params.get("enable_belief", False)
        obs_spec = _get_obs_spec(map_name, obs_spec_preset, enable_belief)

        policy_type = training_params.get("policy_type", "sc2_genetic")
        # Spatial obs (dict observation space) is only supported by sc2_cnn.
        # All other SC2 policies operate on flat np.ndarray observations; if
        # the user accidentally left non-empty screen_layers in their config
        # we silently ignore them to avoid crashing those policies.
        if policy_type == "sc2_cnn":
            screen_layers  = training_params.get("screen_layers") or []
            minimap_layers = training_params.get("minimap_layers") or []
        else:
            screen_layers  = []
            minimap_layers = []

        # Use a picklable factory class (not a closure) so the
        # multiprocessing.spawn workers in framework.parallel_eval can
        # reconstruct it after pickling.  Issue #229.
        make_env_fn = _SC2EnvFactory(
            experiment_dir=experiment_dir,
            map_name=map_name,
            max_episode_time_s=training_params["in_game_episode_s"],
            step_mul=training_params.get("step_mul", 1),
            screen_size=training_params.get("screen_size", 64),
            minimap_size=training_params.get("minimap_size", 64),
            agent_race=training_params.get("agent_race", "random"),
            bot_difficulty=training_params.get("bot_difficulty", "very_easy"),
            screen_layers=screen_layers,
            minimap_layers=minimap_layers,
            obs_spec_preset=obs_spec_preset,
            enable_belief=enable_belief,
            max_apm=training_params.get("max_apm", None),
            apm_burst_s=training_params.get("apm_burst_s", 2.0),
        )

        return GameSpec(
            experiment_name=experiment_name,
            track=self.track_label(training_params, track_override),
            make_env_fn=make_env_fn,
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
        policy_type = training_params.get("policy_type", "sc2_genetic")
        policy_params = training_params.get("policy_params") or {}
        _validate_sc2_policy_config(policy_type, policy_params)

        from games.sc2.sc2_policies import SC2GeneticPolicy

        map_name = training_params.get("map_name", "MoveToBeacon")
        obs_spec_preset = training_params.get("obs_spec_preset")
        enable_belief = training_params.get("enable_belief", False)
        obs_spec = _get_obs_spec(map_name, obs_spec_preset, enable_belief)
        trainer_state_file = os.path.join(
            os.path.dirname(weights_file), "trainer_state.npz",
        )

        agent_race = training_params.get("agent_race", "random")

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
                race=agent_race,
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

        def _make_sc2_neural_dqn():
            from games.sc2.policies import SC2NeuralDQNPolicy
            if os.path.exists(weights_file) and not re_initialize:
                with open(weights_file) as _f:
                    _cfg = yaml.safe_load(_f) or {}
                if isinstance(_cfg, dict) and _cfg.get("policy_type") == "sc2_neural_dqn":
                    policy = SC2NeuralDQNPolicy.from_cfg(_cfg, obs_spec)
                    if os.path.exists(trainer_state_file):
                        try:
                            policy.load_trainer_state(trainer_state_file)
                            logger.info("[SC2NeuralDQNPolicy] loaded trainer state from %s",
                                        trainer_state_file)
                        except (ValueError, KeyError) as exc:
                            logger.warning(
                                "[SC2NeuralDQNPolicy] could not load trainer state — %s; "
                                "continuing with default state.", exc,
                            )
                    return policy
            return SC2NeuralDQNPolicy(
                obs_spec=obs_spec,
                hidden_sizes=policy_params.get("hidden_sizes", [64, 64]),
                replay_buffer_size=policy_params.get("replay_buffer_size", 50000),
                batch_size=policy_params.get("batch_size", 64),
                min_replay_size=policy_params.get("min_replay_size", 2000),
                target_update_freq=policy_params.get("target_update_freq", 200),
                learning_rate=policy_params.get("learning_rate", 0.001),
                epsilon_start=policy_params.get("epsilon_start", 1.0),
                epsilon_end=policy_params.get("epsilon_end", 0.05),
                epsilon_decay_steps=policy_params.get("epsilon_decay_steps", 20000),
                gamma=policy_params.get("gamma", 0.995),
            )

        def _make_sc2_neural_net():
            from games.sc2.sc2_policies import SC2NeuralNetPolicy
            if os.path.exists(weights_file) and not re_initialize:
                with open(weights_file) as _f:
                    _cfg = yaml.safe_load(_f) or {}
                if isinstance(_cfg, dict) and _cfg.get("policy_type") == "sc2_neural_net":
                    return SC2NeuralNetPolicy.from_cfg(_cfg, obs_spec)
            return SC2NeuralNetPolicy(
                obs_spec=obs_spec,
                hidden_sizes=policy_params.get("hidden_sizes", [16, 16]),
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
            from games.sc2.policies import REINFORCEPolicy as _LegacyReinforce
            if os.path.exists(weights_file) and not re_initialize:
                with open(weights_file) as _f:
                    _cfg = yaml.safe_load(_f) or {}
                if isinstance(_cfg, dict) and _cfg.get("policy_type") == "reinforce":
                    policy = _LegacyReinforce.from_cfg(_cfg, obs_spec)
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
            return _LegacyReinforce(
                obs_spec=obs_spec,
                hidden_sizes=policy_params.get("hidden_sizes", [64, 64]),
                learning_rate=policy_params.get("learning_rate", 0.001),
                gamma=policy_params.get("gamma", 0.99),
                entropy_coeff=policy_params.get("entropy_coeff", 0.01),
                baseline=policy_params.get("baseline", "running_mean"),
            )

        def _make_sc2_reinforce():
            from games.sc2.sc2_policies import SC2REINFORCEPolicy
            if os.path.exists(weights_file) and not re_initialize:
                with open(weights_file) as _f:
                    _cfg = yaml.safe_load(_f) or {}
                if isinstance(_cfg, dict) and _cfg.get("policy_type") == "sc2_reinforce":
                    policy = SC2REINFORCEPolicy.from_cfg(_cfg, obs_spec)
                    if os.path.exists(trainer_state_file):
                        try:
                            policy.load_trainer_state(trainer_state_file)
                            logger.info("[SC2REINFORCEPolicy] loaded trainer state from %s",
                                        trainer_state_file)
                        except (ValueError, KeyError) as exc:
                            logger.warning(
                                "[SC2REINFORCEPolicy] could not load trainer state — %s; "
                                "continuing with default state.", exc,
                            )
                    return policy
            return SC2REINFORCEPolicy(
                obs_spec=obs_spec,
                hidden_sizes=policy_params.get("hidden_sizes", [128, 64]),
                learning_rate=policy_params.get("learning_rate", 0.0003),
                gamma=policy_params.get("gamma", 0.995),
                entropy_coeff=policy_params.get("entropy_coeff", 0.05),
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

        def _make_sc2_cmaes():
            from games.sc2.sc2_policies import (
                SC2CMAESPolicy,
                SC2MultiHeadLinearPolicy,
            )
            pop_size = policy_params.get("population_size", 30)
            sigma    = policy_params.get("initial_sigma", 0.5)
            policy   = SC2CMAESPolicy(
                obs_spec        = obs_spec,
                population_size = pop_size,
                initial_sigma   = sigma,
                eval_episodes   = policy_params.get("eval_episodes", 2),
                race            = agent_race,
            )
            if os.path.exists(weights_file) and not re_initialize:
                with open(weights_file) as _f:
                    _cfg = yaml.safe_load(_f) or {}
                if isinstance(_cfg, dict):
                    champion = SC2MultiHeadLinearPolicy.from_cfg(_cfg, obs_spec, race=agent_race)
                    policy.initialize_from_champion(champion)
                    if os.path.exists(trainer_state_file):
                        try:
                            policy.load_trainer_state(trainer_state_file)
                            logger.info("[SC2CMAESPolicy] loaded trainer state from %s",
                                        trainer_state_file)
                        except (ValueError, KeyError) as exc:
                            logger.warning(
                                "[SC2CMAESPolicy] could not load trainer state — %s; "
                                "continuing with champion weights.", exc,
                            )
            else:
                policy.initialize_random()
            return policy

        def _make_sc2_lstm():
            from games.sc2.sc2_policies import (
                SC2LSTMPolicy,
                SC2LSTMEvolutionPolicy,
            )
            hidden_size      = policy_params.get("hidden_size", 64)
            pop_size         = policy_params.get("population_size", 20)
            sigma            = policy_params.get("initial_sigma", 0.03)
            reset_on_episode = bool(policy_params.get("reset_on_episode", True))
            policy           = SC2LSTMEvolutionPolicy(
                obs_spec         = obs_spec,
                hidden_size      = hidden_size,
                population_size  = pop_size,
                initial_sigma    = sigma,
                reset_on_episode = reset_on_episode,
                race             = agent_race,
            )
            if os.path.exists(weights_file) and not re_initialize:
                with open(weights_file) as _f:
                    _cfg = yaml.safe_load(_f) or {}
                if isinstance(_cfg, dict) and _cfg.get("policy_type") == "sc2_lstm":
                    try:
                        champion = SC2LSTMPolicy.from_cfg(_cfg, obs_spec, race=agent_race)
                        policy.initialize_from_champion(champion)
                    except (ValueError, KeyError) as exc:
                        logger.warning(
                            "[SC2LSTMEvolutionPolicy] incompatible saved champion — %s; "
                            "starting from random.", exc,
                        )
                if os.path.exists(trainer_state_file):
                    try:
                        policy.load_trainer_state(trainer_state_file)
                        logger.info("[SC2LSTMEvolutionPolicy] loaded trainer state from %s",
                                    trainer_state_file)
                    except (ValueError, KeyError) as exc:
                        logger.warning(
                            "[SC2LSTMEvolutionPolicy] could not load trainer state — %s; "
                            "continuing.", exc,
                        )
            return policy

        return PolicyExtras(
            factories={
                "sc2_genetic":   _make_sc2_genetic,
                "neural_dqn":    _make_neural_dqn,
                "sc2_neural_dqn": _make_sc2_neural_dqn,
                "sc2_neural_net": _make_sc2_neural_net,
                "cmaes":         _make_cmaes,
                "reinforce":     _make_reinforce,
                "sc2_reinforce": _make_sc2_reinforce,
                "lstm":          _make_lstm,
                "sc2_cnn":       _make_sc2_cnn,
                "sc2_cmaes":     _make_sc2_cmaes,
                "sc2_lstm":      _make_sc2_lstm,
            },
            loop_dispatch={
                "sc2_genetic":   "genetic",
                "neural_dqn":    "q_learning",
                "sc2_neural_dqn": "q_learning",
                "cmaes":         "cmaes",
                "reinforce":     "q_learning",
                "sc2_reinforce": "q_learning",
                "lstm":          "cmaes",
                "sc2_cnn":       "cmaes",
                "sc2_cmaes":     "cmaes",
                "sc2_lstm":      "cmaes",
            },
        )


def make_adapter() -> SC2Adapter:
    return SC2Adapter()
