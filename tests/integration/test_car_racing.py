"""Integration tests for the CarRacing game.

These tests exercise the full stack — real gymnasium CarRacing-v3 environment,
actual Box2D physics, the CarRacingEnv wrapper, and the framework training loop
— without any mocking.

Requires ``gymnasium[box2d]``::

    pip install gymnasium[box2d]

Marked ``integration`` so they are excluded from the fast unit-test suite and
only run in the integration-tests workflow (on push to main) or on demand via::

    pytest tests/integration/ -m integration
"""

import math
import os
import tempfile
import unittest

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip the entire module when gymnasium[box2d] is not installed.
# ---------------------------------------------------------------------------

try:
    import gymnasium as gym  # noqa: F401
    from gymnasium.envs.box2d import CarRacing  # noqa: F401

    _BOX2D_AVAILABLE = True
except ImportError:
    _BOX2D_AVAILABLE = False

pytestmark = pytest.mark.integration

_skip_no_box2d = pytest.mark.skipif(
    not _BOX2D_AVAILABLE,
    reason="gymnasium[box2d] not installed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(max_episode_steps: int = 50):
    """Return a CarRacingEnv with a short episode limit for fast tests."""
    from games.car_racing.env import CarRacingEnv

    return CarRacingEnv(max_episode_steps=max_episode_steps)


# ---------------------------------------------------------------------------
# Environment reset / step / close
# ---------------------------------------------------------------------------


@_skip_no_box2d
class TestCarRacingEnvBasics(unittest.TestCase):
    """Smoke tests: env reset, step, and close against real Box2D physics."""

    def setUp(self):
        self.env = _make_env(max_episode_steps=30)

    def tearDown(self):
        self.env.close()

    def test_reset_returns_correct_obs_shape(self):
        """reset() returns a 1-D float32 array of the expected obs dimension."""
        from games.car_racing.obs_spec import BASE_OBS_DIM

        obs, info = self.env.reset(seed=0)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (BASE_OBS_DIM,))
        self.assertEqual(obs.dtype, np.float32)

    def test_reset_with_different_seeds_is_repeatable(self):
        """Resetting with the same seed twice produces the same initial obs."""
        obs_a, _ = self.env.reset(seed=7)
        obs_b, _ = self.env.reset(seed=7)
        np.testing.assert_array_equal(obs_a, obs_b)

    def test_step_returns_5_tuple(self):
        """step() returns (obs, reward, terminated, truncated, info)."""
        self.env.reset(seed=0)
        result = self.env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_step_reward_is_finite(self):
        """All rewards produced by step() are finite numbers."""
        self.env.reset(seed=42)
        action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for _ in range(10):
            _, reward, terminated, truncated, _ = self.env.step(action)
            self.assertTrue(math.isfinite(reward), f"reward {reward} is not finite")
            if terminated or truncated:
                break

    def test_step_obs_shape_consistent(self):
        """obs returned by step() has the same shape as from reset()."""
        from games.car_racing.obs_spec import BASE_OBS_DIM

        self.env.reset(seed=0)
        action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for _ in range(5):
            obs, _, terminated, truncated, _ = self.env.step(action)
            self.assertEqual(obs.shape, (BASE_OBS_DIM,))
            if terminated or truncated:
                break

    def test_info_has_termination_reason_after_truncation(self):
        """Episode truncated by step limit sets info['termination_reason']."""
        env = _make_env(max_episode_steps=3)
        try:
            obs, _ = env.reset(seed=0)
            action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            info = {}
            for _ in range(5):
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            # termination_reason may be "timeout" or "finish" depending on
            # the physics; the important thing is it's set.
            self.assertIn("termination_reason", info)
        finally:
            env.close()

    def test_native_reward_present_in_info(self):
        """info dict from step() contains 'native_reward' from CarRacing-v3."""
        self.env.reset(seed=0)
        _, _, _, _, info = self.env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        self.assertIn("native_reward", info)
        self.assertTrue(math.isfinite(info["native_reward"]))

    def test_close_is_idempotent(self):
        """Calling close() twice does not raise."""
        self.env.close()
        self.env.close()  # second call must not raise


# ---------------------------------------------------------------------------
# Full episode with a simple constant policy
# ---------------------------------------------------------------------------


@_skip_no_box2d
class TestCarRacingFullEpisode(unittest.TestCase):
    """Run complete episodes to verify that game physics terminates them."""

    def test_episode_terminates_within_step_limit(self):
        """A short-limit episode always terminates (truncated or finished)."""
        env = _make_env(max_episode_steps=25)
        try:
            obs, _ = env.reset(seed=0)
            action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated) and steps < 100:
                _, _, terminated, truncated, _ = env.step(action)
                steps += 1
            self.assertTrue(
                terminated or truncated,
                f"Episode did not end within {steps} steps",
            )
        finally:
            env.close()

    def test_episode_total_reward_finite(self):
        """Total accumulated reward over an episode is a finite number."""
        env = _make_env(max_episode_steps=20)
        try:
            env.reset(seed=1)
            action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            total = 0.0
            for _ in range(30):
                _, r, terminated, truncated, _ = env.step(action)
                total += r
                if terminated or truncated:
                    break
            self.assertTrue(math.isfinite(total), f"total reward {total} not finite")
        finally:
            env.close()

    def test_multiple_resets_do_not_leak_state(self):
        """Resetting mid-episode and starting fresh never causes an error."""
        env = _make_env(max_episode_steps=10)
        try:
            action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            for seed in range(3):
                env.reset(seed=seed)
                for _ in range(5):
                    _, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Training-loop integration
# ---------------------------------------------------------------------------


@_skip_no_box2d
class TestCarRacingTrainingLoop(unittest.TestCase):
    """Verify the framework training loop runs end-to-end on CarRacing."""

    def test_hill_climbing_one_sim(self):
        """_greedy_loop runs 1 simulation against the real CarRacing env."""
        from framework.policies import WeightedLinearPolicy
        from framework.training import _greedy_loop
        from games.car_racing.env import CarRacingEnv
        from games.car_racing.obs_spec import CAR_RACING_OBS_SPEC

        env = CarRacingEnv(max_episode_steps=15)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                weights_file = os.path.join(tmpdir, "policy_weights.yaml")
                policy = WeightedLinearPolicy(CAR_RACING_OBS_SPEC, ["steer", "accel", "brake"], weights_file)
                loop = _greedy_loop(
                    env,
                    policy,
                    n_sims=1,
                    mutation_scale=0.1,
                    weights_file=weights_file,
                    best_reward=float("-inf"),
                )
                self.assertEqual(len(loop.greedy_sims), 1)
                self.assertTrue(math.isfinite(loop.best_reward))
        finally:
            env.close()

    def test_genetic_policy_one_generation(self):
        """_greedy_loop_genetic runs 1 generation with a tiny population."""
        from framework.policies import GeneticPolicy
        from framework.training import _greedy_loop_genetic
        from games.car_racing.env import CarRacingEnv
        from games.car_racing.obs_spec import CAR_RACING_OBS_SPEC

        env = CarRacingEnv(max_episode_steps=10)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                weights_file = os.path.join(tmpdir, "policy_weights.yaml")
                policy = GeneticPolicy(
                    obs_spec=CAR_RACING_OBS_SPEC,
                    head_names=["steer", "accel", "brake"],
                    population_size=2,
                    elite_k=1,
                    mutation_scale=0.1,
                )
                policy.initialize_random()
                loop = _greedy_loop_genetic(
                    env,
                    policy,
                    n_generations=1,
                    weights_file=weights_file,
                )
                self.assertEqual(len(loop.greedy_sims), 1)
                self.assertTrue(math.isfinite(loop.best_reward))
        finally:
            env.close()

    def test_epsilon_greedy_one_episode(self):
        """_greedy_loop_q_learning runs 1 episode with ε-greedy policy."""
        from framework.policies import EpsilonGreedyPolicy
        from framework.training import _greedy_loop_q_learning
        from games.car_racing.actions import DISCRETE_ACTIONS
        from games.car_racing.env import CarRacingEnv
        from games.car_racing.obs_spec import CAR_RACING_OBS_SPEC

        env = CarRacingEnv(max_episode_steps=10)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                weights_file = os.path.join(tmpdir, "policy_weights.yaml")
                policy = EpsilonGreedyPolicy(
                    obs_spec=CAR_RACING_OBS_SPEC,
                    discrete_actions=DISCRETE_ACTIONS,
                    n_bins=2,
                    epsilon=1.0,
                )
                loop = _greedy_loop_q_learning(
                    env,
                    policy,
                    n_episodes=1,
                    weights_file=weights_file,
                )
                self.assertEqual(len(loop.greedy_sims), 1)
                self.assertTrue(math.isfinite(loop.best_reward))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
