"""Unit tests for patience-based early stopping in greedy training loops."""
import os
import sys
import unittest
import tempfile

import numpy as np

# Ensure project root is on path (conftest.py handles this for pytest,
# but we also need it when running this file directly).
_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

# framework.training has no top-level shim; import directly.
from framework.training import _greedy_loop, _greedy_loop_q_learning
# framework.policies.WeightedLinearPolicy is used directly because the TMNF shim
# in policies.py bakes in TMNF's obs_spec and a different from_cfg signature;
# these tests exercise framework internals with a custom 3-dim obs_spec.
from framework.policies import WeightedLinearPolicy
# obs_spec shim re-exports ObsSpec/ObsDim unchanged — use it for consistency.
from obs_spec import ObsSpec, ObsDim


# ---------------------------------------------------------------------------
# Minimal stub env that returns a fixed reward each episode
# ---------------------------------------------------------------------------

class _FixedRewardEnv:
    """Minimal env stub: every episode returns a fixed reward."""

    def __init__(self, reward: float) -> None:
        self._reward = reward

    def reset(self):
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        info = {"track_progress": 0.5, "laps_completed": 0, "pos_x": 0.0, "pos_z": 0.0}
        return np.zeros(3, dtype=np.float32), self._reward, True, False, info

    def get_episode_time_limit(self):
        return None

    def set_episode_time_limit(self, _):
        pass

    def close(self):
        pass


class _IncreasingRewardEnv:
    """Env that returns reward = call_count (always improving)."""

    def __init__(self) -> None:
        self._call = 0

    def reset(self):
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self._call += 1
        info = {"track_progress": 0.5, "laps_completed": 0, "pos_x": 0.0, "pos_z": 0.0}
        return np.zeros(3, dtype=np.float32), float(self._call), True, False, info

    def get_episode_time_limit(self):
        return None

    def set_episode_time_limit(self, _):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Minimal stub policy for Q-learning loop tests
# ---------------------------------------------------------------------------

class _StubQPolicy:
    """Minimal policy compatible with _greedy_loop_q_learning."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def update(self, *_) -> None:
        pass

    def on_episode_end(self) -> None:
        pass

    def save(self, path: str) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.dump({}, f)

    def to_cfg(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OBS_SPEC = ObsSpec([
    ObsDim("a", 1.0, ""),
    ObsDim("b", 1.0, ""),
    ObsDim("c", 1.0, ""),
])
_HEAD_NAMES = ["steer", "accel", "brake"]


def _make_wlp(weights_file: str) -> WeightedLinearPolicy:
    """Create a trivial WeightedLinearPolicy with zero weights."""
    cfg = {
        h + "_weights": {n: 0.0 for n in _OBS_SPEC.names}
        for h in _HEAD_NAMES
    }
    p = WeightedLinearPolicy.from_cfg(cfg, _OBS_SPEC, _HEAD_NAMES)
    p.save(weights_file)
    return p


# ---------------------------------------------------------------------------
# Tests — _greedy_loop (hill_climbing / neural_net)
# ---------------------------------------------------------------------------

class TestPatienceEarlyStopping(unittest.TestCase):

    def test_stops_after_patience_sims_no_improvement(self):
        """With a fixed-reward env and a high best_reward seed, early stopping fires
        after exactly `patience` sims (none of which improve)."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            wf = f.name
        try:
            policy = _make_wlp(wf)
            env    = _FixedRewardEnv(reward=10.0)
            n_sims   = 100
            patience = 5

            _, _, sims, early_stopped, early_stop_sim = _greedy_loop(
                env=env, policy=policy, n_sims=n_sims,
                mutation_scale=0.01, weights_file=wf,
                best_reward=100.0,  # higher than any candidate reward → never improves
                patience=patience, adaptive_mutation=False,
            )

            self.assertTrue(early_stopped)
            self.assertEqual(early_stop_sim, patience)
            self.assertEqual(len(sims), patience)
        finally:
            os.unlink(wf)

    def test_patience_zero_runs_all_sims(self):
        """patience=0 disables early stopping — all n_sims run."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            wf = f.name
        try:
            policy = _make_wlp(wf)
            env    = _FixedRewardEnv(reward=10.0)
            n_sims   = 10

            _, _, sims, early_stopped, early_stop_sim = _greedy_loop(
                env=env, policy=policy, n_sims=n_sims,
                mutation_scale=0.01, weights_file=wf,
                patience=0, adaptive_mutation=False,
            )

            self.assertFalse(early_stopped)
            self.assertIsNone(early_stop_sim)
            self.assertEqual(len(sims), n_sims)
        finally:
            os.unlink(wf)

    def test_streak_resets_on_improvement(self):
        """When reward improves mid-run, the no-improve streak resets."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            wf = f.name
        try:
            policy = _make_wlp(wf)
            env    = _IncreasingRewardEnv()
            n_sims   = 20
            patience = 5

            _, _, sims, early_stopped, early_stop_sim = _greedy_loop(
                env=env, policy=policy, n_sims=n_sims,
                mutation_scale=0.01, weights_file=wf,
                patience=patience, adaptive_mutation=False,
            )

            self.assertFalse(early_stopped)
            self.assertIsNone(early_stop_sim)
            self.assertEqual(len(sims), n_sims)
        finally:
            os.unlink(wf)

    def test_early_stop_sim_is_last_recorded_sim(self):
        """early_stop_sim matches the last sim index stored in greedy_sims."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            wf = f.name
        try:
            policy = _make_wlp(wf)
            env    = _FixedRewardEnv(reward=5.0)
            patience = 3

            _, _, sims, early_stopped, early_stop_sim = _greedy_loop(
                env=env, policy=policy, n_sims=50,
                mutation_scale=0.01, weights_file=wf,
                patience=patience, adaptive_mutation=False,
            )

            self.assertTrue(early_stopped)
            self.assertIsNotNone(early_stop_sim)
            self.assertEqual(sims[-1].sim, early_stop_sim)
        finally:
            os.unlink(wf)


# ---------------------------------------------------------------------------
# Tests — _greedy_loop_q_learning (epsilon_greedy / mcts)
# ---------------------------------------------------------------------------

class TestPatienceEarlyStoppingQLoop(unittest.TestCase):

    def test_q_loop_stops_after_patience_no_improvement(self):
        """Q-learning loop stops after exactly `patience` episodes when reward never improves."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            wf = f.name
        try:
            policy   = _StubQPolicy()
            env      = _FixedRewardEnv(reward=5.0)
            patience = 4

            # Seed best_reward above the fixed reward so nothing ever improves.
            _, _, sims, early_stopped, early_stop_sim = _greedy_loop_q_learning(
                env=env, policy=policy, n_episodes=100,
                weights_file=wf, patience=patience,
            )

            # The stub always returns reward=5.0; first episode sets best from -inf,
            # so streak starts counting from episode 2. Stop at episode 1+patience.
            self.assertTrue(early_stopped)
            self.assertEqual(len(sims), 1 + patience)
            self.assertEqual(sims[-1].sim, early_stop_sim)
        finally:
            os.unlink(wf)

    def test_q_loop_patience_zero_runs_all(self):
        """patience=0 in Q-loop disables early stopping."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            wf = f.name
        try:
            policy   = _StubQPolicy()
            env      = _FixedRewardEnv(reward=5.0)
            n_eps    = 8

            _, _, sims, early_stopped, early_stop_sim = _greedy_loop_q_learning(
                env=env, policy=policy, n_episodes=n_eps,
                weights_file=wf, patience=0,
            )

            self.assertFalse(early_stopped)
            self.assertIsNone(early_stop_sim)
            self.assertEqual(len(sims), n_eps)
        finally:
            os.unlink(wf)


if __name__ == "__main__":
    unittest.main()
