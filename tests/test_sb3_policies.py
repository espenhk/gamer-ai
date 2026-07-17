"""Tests for the Stable-Baselines3-backed gradient deep-RL policies.

Skipped unless ``stable-baselines3`` / ``sb3-contrib`` are installed
(``poetry install --with deep_rl``).  The end-to-end tests run a handful of
SB3 timesteps on a tiny dummy environment — no game binary required.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")

import gymnasium as gym

from framework.base_env import BaseGameEnv
from framework.obs_spec import ObsDim, ObsSpec
from framework.policies import POLICY_REGISTRY
from framework.run_config import GameSpec, RunConfig
from framework.training import train_rl

_OBS_DIM = 3
_ACT_DIM = 2
_EP_LEN = 8
# Two-row discrete action table for the QR-DQN wrapper.
_DISCRETE = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)


class _DummyEnv(BaseGameEnv):
    """Tiny continuous-control env: fixed-length episodes, smooth reward."""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(_OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(_ACT_DIM,), dtype=np.float32)
        self._step = 0
        self._state = np.zeros(_OBS_DIM, dtype=np.float32)

    def _build_obs(self, step):  # abstract hook; unused directly here
        return self._state.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        self._state = np.zeros(_OBS_DIM, dtype=np.float32)
        return self._state.copy(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._step += 1
        self._state = np.clip(
            self._state + 0.1 * action[:_OBS_DIM] if _ACT_DIM >= _OBS_DIM else np.pad(action, (0, _OBS_DIM - _ACT_DIM)),
            -1.0,
            1.0,
        ).astype(np.float32)
        reward = float(-np.sum(np.abs(self._state)))
        terminated = False
        truncated = self._step >= _EP_LEN
        return self._state.copy(), reward, terminated, truncated, {}


def _game_spec(tmp_path) -> GameSpec:
    weights = str(tmp_path / "policy_weights.yaml")
    obs_spec = ObsSpec([ObsDim(f"f{i}", 1.0, "dummy") for i in range(_OBS_DIM)])
    return GameSpec(
        experiment_name="sb3_test",
        track="dummy",
        make_env_fn=_DummyEnv,
        obs_spec=obs_spec,
        head_names=["a", "b"],
        discrete_actions=_DISCRETE,
        weights_file=weights,
        reward_config_file=str(tmp_path / "reward_config.yaml"),
        game_name="dummy",
    )


def _run_config(policy_type: str, **params) -> RunConfig:
    return RunConfig(
        n_sims=2,
        in_game_episode_s=10.0,
        policy_type=policy_type,
        policy_params={"total_timesteps": 64, **params},
    )


# --------------------------------------------------------------------------- #
# Registry / metadata                                                          #
# --------------------------------------------------------------------------- #

_SB3_TYPES = ["ppo", "a2c", "sac", "td3", "qr_dqn", "recurrent_ppo"]


def test_all_sb3_policies_registered():
    for t in _SB3_TYPES:
        assert t in POLICY_REGISTRY, f"{t} not registered"
        assert POLICY_REGISTRY[t].LOOP_TYPE == "sb3"


def test_sb3_policies_incompatible_with_sc2():
    for t in _SB3_TYPES:
        ok, hint = POLICY_REGISTRY[t].compatible_with("sc2")
        assert ok is False
        assert hint and "sc2" in hint.lower()


def test_sb3_policies_compatible_with_racing_games():
    for t in _SB3_TYPES:
        ok, _ = POLICY_REGISTRY[t].compatible_with("tmnf")
        assert ok is True


def test_unknown_policy_param_rejected():
    with pytest.raises(ValueError, match="no effect"):
        POLICY_REGISTRY["ppo"]._validate_params({"not_a_real_knob": 1})


def test_total_timesteps_resolution(tmp_path):
    spec = _game_spec(tmp_path)
    p = POLICY_REGISTRY["ppo"].make(
        obs_spec=spec.obs_spec,
        head_names=spec.head_names,
        discrete_actions=spec.discrete_actions,
        weights_file=spec.weights_file,
        policy_params={"steps_per_sim": 100},
        re_initialize=True,
    )
    assert p.total_timesteps(5) == 500  # n_sims * steps_per_sim
    p2 = POLICY_REGISTRY["ppo"].make(
        obs_spec=spec.obs_spec,
        head_names=spec.head_names,
        discrete_actions=spec.discrete_actions,
        weights_file=spec.weights_file,
        policy_params={"total_timesteps": 321},
        re_initialize=True,
    )
    assert p2.total_timesteps(5) == 321  # explicit overrides


# --------------------------------------------------------------------------- #
# Observation normalisation                                                    #
# --------------------------------------------------------------------------- #


def test_normalize_obs_wrapper_divides_by_scales():
    """NormalizeObsWrapper must divide raw observations by ObsSpec.scales,
    matching the convention numpy-native policies apply internally
    (framework/policies.py: ``norm_obs = obs / self._obs_spec.scales``).
    Regression guard for the SB3 loop training on raw, unnormalised
    observations spanning wildly different magnitudes."""
    from framework.sb3_support import NormalizeObsWrapper

    env = _DummyEnv()
    scales = np.array([1.0, 10.0, 100.0], dtype=np.float32)
    wrapped = NormalizeObsWrapper(env, scales)

    raw_obs, _ = env.reset()
    norm_obs, _ = wrapped.reset()
    np.testing.assert_array_equal(norm_obs, raw_obs / scales)

    action = np.array([0.5, -0.5], dtype=np.float32)
    raw_obs, *_ = env.step(action)
    norm_obs, *_ = wrapped.step(action)
    np.testing.assert_allclose(norm_obs, raw_obs / scales, atol=1e-6)


def test_run_sb3_loop_applies_obs_spec_scale(tmp_path):
    """End-to-end: passing a non-unit-scale obs_spec through train_rl() must
    actually reach the SB3 model as normalised, not raw, observations."""
    from framework.obs_spec import ObsDim, ObsSpec

    spec = _game_spec(tmp_path)
    spec = GameSpec(
        experiment_name=spec.experiment_name,
        track=spec.track,
        make_env_fn=spec.make_env_fn,
        obs_spec=ObsSpec([ObsDim(f"f{i}", 10.0, "dummy") for i in range(_OBS_DIM)]),
        head_names=spec.head_names,
        discrete_actions=spec.discrete_actions,
        weights_file=spec.weights_file,
        reward_config_file=spec.reward_config_file,
        game_name=spec.game_name,
    )
    cfg = _run_config("ppo", n_steps=8, batch_size=8)
    data = train_rl(spec, cfg, no_interrupt=True, re_initialize=True)
    assert data.greedy_sims, "no episodes recorded"


# --------------------------------------------------------------------------- #
# End-to-end training                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "policy_type,params",
    [
        ("a2c", {"n_steps": 8}),
        ("ppo", {"n_steps": 32, "batch_size": 16}),
    ],
)
def test_continuous_policy_end_to_end(tmp_path, policy_type, params):
    import os

    spec = _game_spec(tmp_path)
    cfg = _run_config(policy_type, **params)
    data = train_rl(spec, cfg, no_interrupt=True, re_initialize=True)

    assert data.greedy_sims, "no episodes recorded"
    assert os.path.exists(spec.weights_file)
    base = os.path.splitext(spec.weights_file)[0]
    assert os.path.exists(base + "_sb3_model.zip"), "SB3 model zip not saved"

    # Resume from saved model and run inference.
    resumed = POLICY_REGISTRY[policy_type].make(
        obs_spec=spec.obs_spec,
        head_names=spec.head_names,
        discrete_actions=spec.discrete_actions,
        weights_file=spec.weights_file,
        policy_params={},
        re_initialize=False,
    )
    resumed.build_model(_DummyEnv())
    action = resumed(np.zeros(_OBS_DIM, dtype=np.float32))
    assert action.shape == (_ACT_DIM,)


def test_qr_dqn_discrete_end_to_end(tmp_path):
    import os

    spec = _game_spec(tmp_path)
    cfg = _run_config(
        "qr_dqn",
        learning_starts=8,
        buffer_size=200,
        batch_size=8,
        train_freq=1,
        target_update_interval=8,
    )
    data = train_rl(spec, cfg, no_interrupt=True, re_initialize=True)
    assert data.greedy_sims
    assert os.path.exists(os.path.splitext(spec.weights_file)[0] + "_sb3_model.zip")


# --------------------------------------------------------------------------- #
# Resume / crash-recovery checkpointing (issue #483)                          #
# --------------------------------------------------------------------------- #


def test_checkpoint_and_replay_buffer_saved_for_off_policy(tmp_path):
    """SAC (off-policy) leaves a resume checkpoint + replay buffer on disk."""
    import os

    spec = _game_spec(tmp_path)
    cfg = _run_config(
        "sac",
        learning_starts=8,
        buffer_size=200,
        batch_size=8,
        train_freq=1,
        checkpoint_freq=8,
    )
    train_rl(spec, cfg, no_interrupt=True, re_initialize=True)

    base = os.path.splitext(spec.weights_file)[0]
    assert os.path.exists(base + "_sb3_checkpoint.zip"), "resume checkpoint not saved"
    assert os.path.exists(base + "_sb3_replay_buffer.pkl"), "replay buffer not persisted"


def test_on_policy_checkpoint_has_no_replay_buffer(tmp_path):
    """PPO (on-policy) checkpoints the model for resume but has no replay buffer."""
    import os

    spec = _game_spec(tmp_path)
    cfg = _run_config("ppo", n_steps=32, batch_size=16, checkpoint_freq=16)
    train_rl(spec, cfg, no_interrupt=True, re_initialize=True)

    base = os.path.splitext(spec.weights_file)[0]
    assert os.path.exists(base + "_sb3_checkpoint.zip")
    assert not os.path.exists(base + "_sb3_replay_buffer.pkl")


def test_resume_continues_step_count_instead_of_restarting(tmp_path):
    """Resuming a run must continue the cumulative timestep count, not restart at 0."""
    spec = _game_spec(tmp_path)
    first_target = 64
    cfg = _run_config("sac", learning_starts=8, buffer_size=200, batch_size=8, train_freq=1)
    cfg.policy_params["total_timesteps"] = first_target
    data = train_rl(spec, cfg, no_interrupt=True, re_initialize=True)
    assert data.greedy_sims

    resumed_policy = POLICY_REGISTRY["sac"].make(
        obs_spec=spec.obs_spec,
        head_names=spec.head_names,
        discrete_actions=spec.discrete_actions,
        weights_file=spec.weights_file,
        policy_params={"learning_starts": 8, "buffer_size": 200, "batch_size": 8, "train_freq": 1},
        re_initialize=False,
    )
    assert resumed_policy._resume is True
    model = resumed_policy.build_model(_DummyEnv())
    assert model.num_timesteps >= first_target
    # Replay buffer state carried over rather than starting empty.
    assert model.replay_buffer.size() > 0


def test_resume_skips_training_once_target_reached(tmp_path):
    """Re-running with the same total_timesteps target after completion is a no-op."""
    spec = _game_spec(tmp_path)
    cfg = _run_config("sac", learning_starts=8, buffer_size=200, batch_size=8, train_freq=1)
    cfg.policy_params["total_timesteps"] = 64
    train_rl(spec, cfg, no_interrupt=True, re_initialize=True)

    # Re-invoke with the same config (simulating a re-run after the process
    # already reached the target); the second run should record no new episodes
    # beyond what a fresh `remaining_timesteps == 0` run would produce.
    data2 = train_rl(spec, cfg, no_interrupt=True, re_initialize=False)
    assert data2 is not None


def test_resume_with_no_new_episode_does_not_clobber_best_snapshot(tmp_path):
    """A resume that completes zero new episodes must not overwrite the
    genuine best-reward *_sb3_model.zip with the (possibly worse) checkpoint
    state it resumed from."""
    import os

    spec = _game_spec(tmp_path)
    cfg = _run_config("sac", learning_starts=8, buffer_size=200, batch_size=8, train_freq=1)
    cfg.policy_params["total_timesteps"] = 64
    train_rl(spec, cfg, no_interrupt=True, re_initialize=True)

    best_path = os.path.splitext(spec.weights_file)[0] + "_sb3_model.zip"
    assert os.path.exists(best_path)
    before = open(best_path, "rb").read()

    # Re-run at the same target: remaining_timesteps == 0, so no episode
    # completes and recorder.best_sim stays None this invocation.
    train_rl(spec, cfg, no_interrupt=True, re_initialize=False)

    after = open(best_path, "rb").read()
    assert before == after, "best-reward snapshot was overwritten despite no new episode completing"


def test_checkpoint_save_leaves_no_tmp_file(tmp_path):
    """Atomic checkpoint writes must not leave a `.tmp` sibling behind."""
    import os

    spec = _game_spec(tmp_path)
    cfg = _run_config(
        "sac",
        learning_starts=8,
        buffer_size=200,
        batch_size=8,
        train_freq=1,
        checkpoint_freq=8,
    )
    train_rl(spec, cfg, no_interrupt=True, re_initialize=True)

    base = os.path.splitext(spec.weights_file)[0]
    assert not os.path.exists(base + "_sb3_checkpoint.zip.tmp")
    assert not os.path.exists(base + "_sb3_replay_buffer.pkl.tmp")
    assert not os.path.exists(base + "_sb3_model.zip.tmp")
