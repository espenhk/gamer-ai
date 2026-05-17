"""Integration tests for StarCraft 2.

These tests exercise the full stack — real PySC2 client, actual Blizzard SC2
headless binary, the SC2Env wrapper, and the framework training loop — with
no mocking.

Requirements:

1. Blizzard SC2 headless Linux binary (see CLAUDE.md for download URL).
2. ``SC2PATH`` environment variable pointing to the install root.
3. PySC2 mini-game maps in ``$SC2PATH/Maps/mini_games/``.
4. Python extras: ``pip install pysc2 protobuf``

Marked ``integration`` so they are excluded from the fast unit-test suite and
only run in the ``integration-tests`` workflow (on push to main) or locally::

    pytest tests/integration/test_sc2.py -m integration -v

The tests use *MoveToBeacon* — the simplest PySC2 minigame — to keep episode
time short (~5–10 seconds per episode with ``step_mul=16``).
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Detect SC2 availability.
# ---------------------------------------------------------------------------

_SC2_AVAILABLE = False
_SC2_SKIP_REASON = "unknown"

try:
    import pysc2  # noqa: F401

    sc2path = os.environ.get("SC2PATH", "")
    if not sc2path or not os.path.isdir(sc2path):
        _SC2_SKIP_REASON = "SC2PATH not set or directory does not exist"
    else:
        _SC2_AVAILABLE = True
except ImportError:
    _SC2_SKIP_REASON = "pysc2 not installed"


pytestmark = pytest.mark.integration

_skip_no_sc2 = pytest.mark.skipif(
    not _SC2_AVAILABLE,
    reason=_SC2_SKIP_REASON,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Use step_mul=16 (instead of default 1) so episodes finish faster in CI.
_STEP_MUL = 16
_MAX_EPISODE_TIME_S = 30.0  # wall-clock truncation — short for CI
_MAP_NAME = "MoveToBeacon"


def _make_env():
    """Create an SC2Env pointed at MoveToBeacon with fast settings."""
    from games.sc2.env import SC2Env
    from games.sc2.reward import SC2RewardConfig

    return SC2Env(
        map_name=_MAP_NAME,
        reward_config=SC2RewardConfig(
            score_weight=1.0,
            step_penalty=-0.001,
        ),
        max_episode_time_s=_MAX_EPISODE_TIME_S,
        step_mul=_STEP_MUL,
        screen_size=64,
        minimap_size=64,
    )


# ---------------------------------------------------------------------------
# SC2Client low-level tests
# ---------------------------------------------------------------------------

@_skip_no_sc2
class TestSC2ClientBasics(unittest.TestCase):
    """Verify SC2Client reset/step against the real SC2 binary."""

    def test_client_reset_returns_obs_and_info(self):
        """client.reset() returns a numpy observation and an info dict."""
        from games.sc2.client import SC2Client

        client = SC2Client(
            map_name=_MAP_NAME,
            step_mul=_STEP_MUL,
            screen_size=64,
            minimap_size=64,
        )
        try:
            obs, info = client.reset()
            self.assertIsInstance(obs, np.ndarray)
            self.assertEqual(obs.dtype, np.float32)
            self.assertGreater(len(obs), 0)
            self.assertIsInstance(info, dict)
            # MoveToBeacon info should contain score
            self.assertIn("score", info)
        finally:
            client.close()

    def test_client_step_returns_four_tuple(self):
        """client.step() returns (obs, score, done, info)."""
        from games.sc2.client import SC2Client

        client = SC2Client(
            map_name=_MAP_NAME,
            step_mul=_STEP_MUL,
            screen_size=64,
            minimap_size=64,
        )
        try:
            client.reset()
            # Issue a no_op action: [fn_idx=0, x=0, y=0, queue=0]
            action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            result = client.step(action)
            self.assertEqual(len(result), 4)
            obs, score, done, info = result
            self.assertIsInstance(obs, np.ndarray)
            self.assertTrue(math.isfinite(score))
            self.assertIsInstance(done, bool)
            self.assertIsInstance(info, dict)
        finally:
            client.close()

    def test_client_step_select_army_then_move(self):
        """client can execute select_army then Move_screen without error."""
        from games.sc2.client import SC2Client

        client = SC2Client(
            map_name=_MAP_NAME,
            step_mul=_STEP_MUL,
            screen_size=64,
            minimap_size=64,
        )
        try:
            client.reset()
            # select_army
            act_select = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            client.step(act_select)
            # Move_screen to centre
            act_move = np.array([2.0, 0.5, 0.5, 0.0], dtype=np.float32)
            obs, score, done, info = client.step(act_move)
            self.assertIsInstance(obs, np.ndarray)
        finally:
            client.close()


# ---------------------------------------------------------------------------
# SC2Env lifecycle tests
# ---------------------------------------------------------------------------

@_skip_no_sc2
class TestSC2EnvLifecycle(unittest.TestCase):
    """Verify SC2Env reset/step/close against the real binary."""

    def test_reset_returns_obs_and_info(self):
        """reset() returns (obs, info) with correct obs dimension."""
        from games.sc2.obs_spec import BASE_OBS_DIM

        env = _make_env()
        try:
            obs, info = env.reset()
            self.assertIsInstance(obs, np.ndarray)
            self.assertEqual(obs.shape, (BASE_OBS_DIM,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertIsInstance(info, dict)
        finally:
            env.close()

    def test_step_returns_five_tuple(self):
        """step() returns the standard gymnasium 5-tuple."""
        env = _make_env()
        try:
            env.reset()
            action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            result = env.step(action)
            self.assertEqual(len(result), 5)
            obs, reward, terminated, truncated, info = result
            self.assertIsInstance(obs, np.ndarray)
            self.assertTrue(math.isfinite(reward))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)
        finally:
            env.close()

    def test_step_reward_is_finite(self):
        """All rewards from step() are finite numbers."""
        env = _make_env()
        try:
            env.reset()
            action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            for _ in range(5):
                _, reward, terminated, truncated, _ = env.step(action)
                self.assertTrue(
                    math.isfinite(reward), f"reward {reward} is not finite"
                )
                if terminated or truncated:
                    break
        finally:
            env.close()

    def test_episode_terminates(self):
        """A MoveToBeacon episode terminates or truncates within the time limit."""
        env = _make_env()
        try:
            env.reset()
            action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            done = False
            steps = 0
            while not done and steps < 500:
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1
            self.assertTrue(done, f"Episode did not end within {steps} steps")
        finally:
            env.close()

    def test_step_obs_shape_consistent(self):
        """obs returned by step() has the same shape as from reset()."""
        from games.sc2.obs_spec import BASE_OBS_DIM

        env = _make_env()
        try:
            obs_reset, _ = env.reset()
            action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            for _ in range(3):
                obs_step, _, terminated, truncated, _ = env.step(action)
                self.assertEqual(obs_step.shape, (BASE_OBS_DIM,))
                self.assertEqual(obs_step.shape, obs_reset.shape)
                if terminated or truncated:
                    break
        finally:
            env.close()

    def test_close_is_idempotent(self):
        """Calling close() twice does not raise."""
        env = _make_env()
        env.reset()
        env.close()
        env.close()  # must not raise

    def test_multiple_resets(self):
        """Resetting mid-episode works without errors."""
        env = _make_env()
        try:
            action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            for _ in range(3):
                env.reset()
                for _ in range(3):
                    _, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Full episode with action variety
# ---------------------------------------------------------------------------

@_skip_no_sc2
class TestSC2FullEpisode(unittest.TestCase):
    """Run a full episode with varied actions, validating obs and reward."""

    def test_full_episode_varied_actions(self):
        """Issue select_army + random Move_screen actions through a full episode."""
        env = _make_env()
        try:
            obs, _ = env.reset()
            rng = np.random.default_rng(42)
            total_reward = 0.0
            steps = 0

            while steps < 300:
                # Alternate between select_army and Move_screen to random spots
                if steps % 5 == 0:
                    action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                else:
                    x, y = rng.uniform(0, 1, size=2)
                    action = np.array([2.0, x, y, 0.0], dtype=np.float32)

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

                self.assertEqual(obs.dtype, np.float32)
                self.assertTrue(math.isfinite(reward))

                if terminated or truncated:
                    break

            self.assertTrue(
                terminated or truncated,
                f"Episode did not end within {steps} steps",
            )
            self.assertTrue(math.isfinite(total_reward))
        finally:
            env.close()

    def test_info_contains_score(self):
        """info dict from step() contains the cumulative score."""
        env = _make_env()
        try:
            env.reset()
            # select_army then move to force score
            env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
            _, _, _, _, info = env.step(
                np.array([2.0, 0.5, 0.5, 0.0], dtype=np.float32)
            )
            # Info should contain score related keys
            self.assertIn("score", info)
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Training loop integration
# ---------------------------------------------------------------------------

@_skip_no_sc2
class TestSC2TrainingLoop(unittest.TestCase):
    """Run framework training loops against the real SC2 binary."""

    def test_genetic_one_generation(self):
        """_greedy_loop_genetic runs 1 generation with a tiny population."""
        from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC
        from games.sc2.sc2_policies import SC2GeneticPolicy
        from framework.training import _greedy_loop_genetic

        env = _make_env()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                weights_file = os.path.join(tmpdir, "policy_weights.yaml")
                policy = SC2GeneticPolicy(
                    obs_spec=SC2_MINIGAME_OBS_SPEC,
                    population_size=2,
                    elite_k=1,
                    mutation_scale=0.1,
                    mutation_share=1.0,
                )
                policy.initialize_random()

                best_policy, best_reward, sims, _, _ = _greedy_loop_genetic(
                    env,
                    policy,
                    n_generations=1,
                    weights_file=weights_file,
                )
                self.assertEqual(len(sims), 1)
                self.assertTrue(math.isfinite(best_reward))
        finally:
            env.close()

    def test_epsilon_greedy_one_episode(self):
        """_greedy_loop_q_learning runs 1 episode with ε-greedy policy."""
        from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC
        from games.sc2.actions import DISCRETE_ACTIONS
        from framework.policies import EpsilonGreedyPolicy
        from framework.training import _greedy_loop_q_learning

        env = _make_env()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                weights_file = os.path.join(tmpdir, "policy_weights.yaml")
                policy = EpsilonGreedyPolicy(
                    obs_spec=SC2_MINIGAME_OBS_SPEC,
                    discrete_actions=DISCRETE_ACTIONS,
                    n_bins=2,
                    epsilon=1.0,
                )
                _, best_reward, sims, _, _ = _greedy_loop_q_learning(
                    env,
                    policy,
                    n_episodes=1,
                    weights_file=weights_file,
                )
                self.assertEqual(len(sims), 1)
                self.assertTrue(math.isfinite(best_reward))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
