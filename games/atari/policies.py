"""Atari-specific policies: thin registered subclasses of framework algorithms."""

from __future__ import annotations

import logging
import os

import yaml

from framework.dqn import DQNPolicy as _FrameworkDQN
from framework.policies import POLICY_REGISTRY, register_policy, trainer_state_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NeuralDQNPolicy — Atari thin subclass of framework DQNPolicy
# ---------------------------------------------------------------------------

# Guard against duplicate registration when TMNF policies are also imported
# in the same process (e.g. during a full test suite run).
if "neural_dqn" not in POLICY_REGISTRY:

    @register_policy
    class NeuralDQNPolicy(_FrameworkDQN):  # type: ignore[no-redef]
        """Atari Deep Q-Network: 18-action discrete set, Atari RAM obs_spec."""

        POLICY_TYPE = "neural_dqn"
        LOOP_TYPE = "q_learning"
        VALID_POLICY_PARAMS = frozenset(
            {
                "hidden_sizes",
                "replay_buffer_size",
                "batch_size",
                "min_replay_size",
                "target_update_freq",
                "learning_rate",
                "epsilon_start",
                "epsilon_end",
                "epsilon_decay_steps",
                "gamma",
                "double_dqn",
                "dueling",
                "huber_loss",
                "huber_kappa",
                "max_grad_norm",
            }
        )

        @classmethod
        def compatible_with(cls, game_name: str) -> tuple[bool, str | None]:
            return True, None

        def to_cfg(self) -> dict:
            cfg = super().to_cfg()
            cfg["policy_type"] = "neural_dqn"
            return cfg

        @classmethod
        def _construct_or_resume(
            cls,
            *,
            obs_spec,
            head_names,
            discrete_actions,
            weights_file,
            policy_params,
            re_initialize,
        ):
            pp = policy_params
            if os.path.exists(weights_file) and not re_initialize:
                with open(weights_file) as _f:
                    cfg = yaml.safe_load(_f)
                if isinstance(cfg, dict) and cfg.get("policy_type") == "neural_dqn":
                    policy = cls.from_cfg(cfg, obs_spec, discrete_actions)
                    ts = trainer_state_path(weights_file)
                    if os.path.exists(ts):
                        try:
                            policy.load_trainer_state(ts)
                            logger.info("[NeuralDQNPolicy] loaded trainer state from %s", ts)
                        except (ValueError, KeyError) as exc:
                            logger.warning(
                                "[NeuralDQNPolicy] could not load trainer state — %s; continuing with default state.",
                                exc,
                            )
                    return policy
            return cls(
                obs_spec=obs_spec,
                discrete_actions=discrete_actions,
                hidden_sizes=pp.get("hidden_sizes", [64, 64]),
                replay_buffer_size=pp.get("replay_buffer_size", 10_000),
                batch_size=pp.get("batch_size", 64),
                min_replay_size=pp.get("min_replay_size", 500),
                target_update_freq=pp.get("target_update_freq", 200),
                learning_rate=pp.get("learning_rate", 0.001),
                epsilon_start=pp.get("epsilon_start", 1.0),
                epsilon_end=pp.get("epsilon_end", 0.05),
                epsilon_decay_steps=pp.get("epsilon_decay_steps", 5_000),
                gamma=pp.get("gamma", 0.99),
                double_dqn=pp.get("double_dqn", True),
                dueling=pp.get("dueling", False),
                huber_loss=pp.get("huber_loss", True),
                huber_kappa=pp.get("huber_kappa", 1.0),
                max_grad_norm=pp.get("max_grad_norm", 10.0),
            )
