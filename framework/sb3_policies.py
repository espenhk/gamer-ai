"""Stable-Baselines3-backed gradient deep-RL policies.

These wrap Stable-Baselines3 / SB3-Contrib algorithms behind the framework's
``BasePolicy`` contract so they are selectable via ``policy_type`` like any
other policy.  Unlike the pure-numpy policies, an SB3 algorithm owns its own
training loop, so all SB3 policies share ``LOOP_TYPE = "sb3"`` and are driven by
``framework.sb3_support.run_sb3_loop`` instead of the per-step ``update`` path.

Registered policy types
-----------------------
    ppo            — PPO (on-policy clipped-surrogate actor-critic)
    a2c            — A2C (synchronous advantage actor-critic)
    sac            — SAC (off-policy max-entropy actor-critic; continuous only)
    td3            — TD3 (deterministic twin-critic continuous control)
    qr_dqn         — QR-DQN (distributional value, SB3-Contrib); discrete actions
    recurrent_ppo  — RecurrentPPO (PPO with an LSTM policy, SB3-Contrib)

``stable-baselines3`` / ``sb3-contrib`` (and torch) are imported lazily inside
methods so importing this module — which happens at framework load so the
policies register — never pulls in the heavy deep-RL stack.  Install the deps
with ``poetry install --with deep_rl``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, ClassVar

import numpy as np

from framework.obs_spec import ObsSpec
from framework.policies import BasePolicy, register_policy

# Game whose multi-head [fn_idx, x, y, queue] action encoding the generic SB3
# Box/Discrete policies cannot drive — gated off below.
_SC2_GAME_NAME = "sc2"

logger = logging.getLogger(__name__)


def _model_zip_path(weights_file: str) -> str:
    """Canonical path for an SB3 policy's saved ``.zip`` model (best-reward snapshot)."""
    base, _ = os.path.splitext(os.path.abspath(weights_file))
    return base + "_sb3_model.zip"


def _checkpoint_zip_path(weights_file: str) -> str:
    """Canonical path for the periodic resume checkpoint (latest, not best-reward)."""
    base, _ = os.path.splitext(os.path.abspath(weights_file))
    return base + "_sb3_checkpoint.zip"


def _replay_buffer_path(weights_file: str) -> str:
    """Canonical path for an off-policy algo's persisted replay buffer."""
    base, _ = os.path.splitext(os.path.abspath(weights_file))
    return base + "_sb3_replay_buffer.pkl"


# --------------------------------------------------------------------------- #
# Base class                                                                   #
# --------------------------------------------------------------------------- #


class _SB3Policy(BasePolicy):
    """Shared plumbing for Stable-Baselines3-backed policies.

    The concrete SB3 model is built lazily by :meth:`build_model` once the
    environment exists (the framework constructs the policy before the env is
    handed to the training loop).  Subclasses declare:

    * ``POLICY_TYPE`` / ``SB3_ALGO`` (import path) / ``SB3_NET`` (network id)
    * ``REQUIRES_DISCRETE`` — wrap the env's continuous action space into a
      ``Discrete`` index over ``discrete_actions`` (DQN-family).
    * ``CONTINUOUS_ONLY`` — incompatible with games whose native action space
      is not a plain continuous ``Box`` (currently SC2).
    """

    LOOP_TYPE: ClassVar[str] = "sb3"

    SB3_ALGO: ClassVar[str] = ""  # "stable_baselines3:PPO" etc.
    SB3_NET: ClassVar[str] = "MlpPolicy"
    REQUIRES_DISCRETE: ClassVar[bool] = False
    CONTINUOUS_ONLY: ClassVar[bool] = False
    RECURRENT: ClassVar[bool] = False
    # Off-policy algos (replay buffer) get the buffer persisted/restored alongside
    # the model checkpoint; on-policy algos (rollout buffer) don't need it.
    OFF_POLICY: ClassVar[bool] = False

    # Hyperparameters every SB3 algo accepts plus run-budget knobs.
    _COMMON_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {
            "total_timesteps",
            "steps_per_sim",
            "learning_rate",
            "gamma",
            "hidden_sizes",
            "seed",
            "verbose",
            "checkpoint_freq",
        }
    )
    # Per-algo extra constructor kwargs (subclasses extend).
    _ALGO_PARAMS: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs) -> None:
        # Single source of truth for accepted params: common knobs + this
        # algo's extras.  Drives BasePolicy._validate_params.
        super().__init_subclass__(**kwargs)
        cls.VALID_POLICY_PARAMS = cls._COMMON_PARAMS | cls._ALGO_PARAMS

    def __init__(
        self,
        obs_spec: ObsSpec,
        discrete_actions: np.ndarray | None = None,
        *,
        weights_file: str | None = None,
        policy_params: dict | None = None,
    ) -> None:
        self._obs_spec = obs_spec
        self._discrete_actions = (
            np.asarray(discrete_actions, dtype=np.float32) if discrete_actions is not None else None
        )
        self._weights_file = weights_file
        self._params = dict(policy_params or {})
        self._model: Any = None
        self._resume: bool = False
        self._resume_path: str | None = None
        # Recurrent inference state (RecurrentPPO only).
        self._lstm_state: Any = None
        self._episode_start: bool = True

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    @classmethod
    def compatible_with(cls, game_name: str) -> tuple[bool, str | None]:
        # SC2's action space is a multi-head [fn_idx, x, y, queue] encoding that
        # SB3's generic Box/Discrete policies cannot drive correctly.  Gate it
        # off until a dedicated SC2 SB3 bridge exists.
        if game_name == _SC2_GAME_NAME:
            return False, (
                f"SB3 policy_type {cls.POLICY_TYPE!r} does not support SC2's multi-head "
                f"[fn_idx, x, y, queue] action encoding. Use an sc2_-prefixed policy "
                f"(e.g. 'sc2_genetic'); SC2 SB3 support is future work."
            )
        return True, None

    @classmethod
    def _construct_or_resume(
        cls, *, obs_spec, head_names, discrete_actions, weights_file, policy_params, re_initialize
    ) -> "_SB3Policy":
        params = {k: v for k, v in (policy_params or {}).items() if not k.startswith("_")}
        obj = cls(
            obs_spec=obs_spec,
            discrete_actions=discrete_actions,
            weights_file=weights_file,
            policy_params=params,
        )
        # Prefer the periodic resume checkpoint (latest training state, correct
        # cumulative timestep count) over the best-reward snapshot, which may
        # lag well behind the crash point on a long off-policy run.
        checkpoint_path = _checkpoint_zip_path(weights_file)
        best_path = _model_zip_path(weights_file)
        has_checkpoint = os.path.exists(checkpoint_path)
        if has_checkpoint:
            resume_path = checkpoint_path
        elif os.path.exists(best_path):
            resume_path = best_path
        else:
            resume_path = None

        # A crash-safe checkpoint exists specifically to be resumed after an
        # interruption — that's its entire purpose, distinct from the
        # best-reward "champion" snapshot every other policy's re_initialize
        # discards on purpose. Honouring re_initialize here would silently
        # throw away in-progress training on the routine relaunch that
        # follows any crash/restart (issue: --re-initialize logically
        # contradicts resuming a checkpoint that exists precisely to be
        # resumed). If you genuinely want to discard it, delete the
        # `*_sb3_checkpoint.zip` / `*_sb3_replay_buffer.pkl` files (or the
        # whole experiment directory) rather than relying on this flag.
        if re_initialize and has_checkpoint:
            logger.warning(
                "[%s] --re-initialize requested but a crash-safe checkpoint exists at %s; "
                "resuming from it instead. Delete the checkpoint (and replay buffer) files "
                "manually if you actually want to discard this run's progress.",
                cls.__name__,
                checkpoint_path,
            )
            re_initialize = False

        obj._resume = bool(resume_path and not re_initialize)
        obj._resume_path = resume_path if obj._resume else None
        if obj._resume:
            logger.info("[%s] will resume from saved model %s", cls.__name__, resume_path)
        return obj

    # ------------------------------------------------------------------
    # SB3 model lifecycle (called by run_sb3_loop)
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_algo(cls):
        module_name, _, attr = cls.SB3_ALGO.partition(":")
        mod = __import__(module_name, fromlist=[attr])
        return getattr(mod, attr)

    def total_timesteps(self, n_sims: int) -> int:
        explicit = self._params.get("total_timesteps")
        if explicit is not None:
            return int(explicit)
        steps_per_sim = int(self._params.get("steps_per_sim", 1000))
        return max(1, int(n_sims)) * steps_per_sim

    def _build_kwargs(self) -> dict:
        """Translate policy_params into SB3 constructor kwargs."""
        kw: dict[str, Any] = {}
        if "learning_rate" in self._params:
            kw["learning_rate"] = float(self._params["learning_rate"])
        if "gamma" in self._params:
            kw["gamma"] = float(self._params["gamma"])
        if "seed" in self._params:
            kw["seed"] = int(self._params["seed"])
        hidden = self._params.get("hidden_sizes")
        if hidden:
            kw["policy_kwargs"] = {"net_arch": list(hidden)}
        # Pass through recognised per-algo kwargs verbatim.
        for key in self._ALGO_PARAMS:
            if key in self._params:
                kw[key] = self._params[key]
        return kw

    def build_model(self, env) -> Any:
        """Construct (or load) the SB3 model bound to *env*. Returns the model."""
        algo = self._resolve_algo()
        verbose = int(self._params.get("verbose", 0))
        if getattr(self, "_resume", False):
            zip_path = self._resume_path or _model_zip_path(self._weights_file)
            self._model = algo.load(zip_path, env=env)
            logger.info(
                "[%s] loaded SB3 model from %s (resuming at %d timesteps)",
                type(self).__name__,
                zip_path,
                int(self._model.num_timesteps),
            )
            if self.OFF_POLICY:
                rb_path = _replay_buffer_path(self._weights_file)
                if os.path.exists(rb_path):
                    self._model.load_replay_buffer(rb_path)
                    logger.info(
                        "[%s] loaded replay buffer from %s (%d transitions)",
                        type(self).__name__,
                        rb_path,
                        self._model.replay_buffer.size(),
                    )
                else:
                    logger.info(
                        "[%s] no replay buffer found at %s; resuming with an empty buffer.",
                        type(self).__name__,
                        rb_path,
                    )
        else:
            self._model = algo(self.SB3_NET, env, verbose=verbose, **self._build_kwargs())
        return self._model

    def set_model(self, model: Any) -> None:
        self._model = model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def on_episode_start(self, **kwargs) -> None:
        self._lstm_state = None
        self._episode_start = True

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(
                f"{type(self).__name__}: no model — load a trained model or run the SB3 training loop before inference."
            )
        if self.RECURRENT:
            action, self._lstm_state = self._model.predict(
                obs,
                state=self._lstm_state,
                episode_start=np.array([self._episode_start]),
                deterministic=True,
            )
            self._episode_start = False
        else:
            action, _ = self._model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        return {
            "policy_type": self.POLICY_TYPE,
            "sb3_algo": self.SB3_ALGO,
            "sb3_net": self.SB3_NET,
            "requires_discrete": self.REQUIRES_DISCRETE,
            "recurrent": self.RECURRENT,
            "obs_dim": self._obs_spec.dim,
            "model_path": os.path.basename(_model_zip_path(self._weights_file)) if self._weights_file else None,
            "policy_params": self._params,
        }

    def save(self, path: str) -> None:
        # Write YAML metadata (for analytics / discovery) and the SB3 model zip.
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_cfg(), f, default_flow_style=False, sort_keys=False)
        if self._model is not None:
            zip_path = _model_zip_path(path)
            self._model.save(zip_path)
            logger.debug("[%s] saved SB3 model → %s", type(self).__name__, zip_path)


# --------------------------------------------------------------------------- #
# Concrete policies                                                            #
# --------------------------------------------------------------------------- #


@register_policy
class PPOPolicy(_SB3Policy):
    """Proximal Policy Optimization (on-policy, clipped surrogate + GAE)."""

    POLICY_TYPE = "ppo"
    SB3_ALGO = "stable_baselines3:PPO"
    _ALGO_PARAMS = frozenset({"n_steps", "batch_size", "n_epochs", "gae_lambda", "clip_range", "ent_coef", "vf_coef"})


@register_policy
class A2CPolicy(_SB3Policy):
    """Advantage Actor-Critic (synchronous on-policy)."""

    POLICY_TYPE = "a2c"
    SB3_ALGO = "stable_baselines3:A2C"
    _ALGO_PARAMS = frozenset({"n_steps", "gae_lambda", "ent_coef", "vf_coef"})


@register_policy
class SACPolicy(_SB3Policy):
    """Soft Actor-Critic (off-policy, max-entropy; continuous action only)."""

    POLICY_TYPE = "sac"
    SB3_ALGO = "stable_baselines3:SAC"
    CONTINUOUS_ONLY = True
    OFF_POLICY = True
    _ALGO_PARAMS = frozenset({"buffer_size", "batch_size", "tau", "train_freq", "learning_starts", "ent_coef"})


@register_policy
class TD3Policy(_SB3Policy):
    """Twin Delayed DDPG (off-policy deterministic; continuous action only)."""

    POLICY_TYPE = "td3"
    SB3_ALGO = "stable_baselines3:TD3"
    CONTINUOUS_ONLY = True
    OFF_POLICY = True
    _ALGO_PARAMS = frozenset({"buffer_size", "batch_size", "tau", "train_freq", "learning_starts", "policy_delay"})


@register_policy
class QRDQNPolicy(_SB3Policy):
    """Quantile-Regression DQN (distributional value; SB3-Contrib).

    Operates over a ``Discrete`` action space, built by wrapping the game's
    continuous action ``Box`` into an index over ``discrete_actions``.
    """

    POLICY_TYPE = "qr_dqn"
    SB3_ALGO = "sb3_contrib:QRDQN"
    REQUIRES_DISCRETE = True
    OFF_POLICY = True
    _ALGO_PARAMS = frozenset(
        {
            "buffer_size",
            "batch_size",
            "learning_starts",
            "target_update_interval",
            "train_freq",
            "exploration_fraction",
            "exploration_final_eps",
            "n_quantiles",
        }
    )

    def _build_kwargs(self) -> dict:
        kw = super()._build_kwargs()
        n_quantiles = self._params.get("n_quantiles")
        if n_quantiles is not None:
            pk = kw.setdefault("policy_kwargs", {})
            pk["n_quantiles"] = int(n_quantiles)
            kw.pop("n_quantiles", None)
        return kw


@register_policy
class RecurrentPPOPolicy(_SB3Policy):
    """PPO with an LSTM policy network (gradient-trained recurrence; SB3-Contrib).

    The field's recurrent agents (OpenAI Five, AlphaStar) gradient-train their
    LSTM cores; this is the gradient counterpart to the ES-trained ``lstm``.
    """

    POLICY_TYPE = "recurrent_ppo"
    SB3_ALGO = "sb3_contrib:RecurrentPPO"
    SB3_NET = "MlpLstmPolicy"
    RECURRENT = True
    _ALGO_PARAMS = frozenset({"n_steps", "batch_size", "n_epochs", "gae_lambda", "clip_range", "ent_coef"})
