"""Tests for the Atari env wrapper.

All gymnasium / ale-py interaction is mocked — the framework only needs to
verify that ``AtariEnv`` adapts the underlying Discrete-action / uint8-RAM
gym env into the framework's float32-array contract.

A separate real-binary integration test would belong under
``tests/integration/`` once ale-py is in the CI environment.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Test doubles for gymnasium.make() and ale-py registration
# ---------------------------------------------------------------------------


class _FakeGymEnv:
    """Minimal stand-in for a gymnasium Atari env in RAM obs mode."""

    def __init__(self, n_actions: int = 6, ram_size: int = 128) -> None:
        self.action_space = spaces.Discrete(n_actions)
        self._ram_size = ram_size
        self._step_count = 0
        self.last_action: int | None = None

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        return np.arange(self._ram_size, dtype=np.uint8), {"seed": seed}

    def step(self, action):
        self._step_count += 1
        self.last_action = int(action)
        obs = np.full(self._ram_size, self._step_count % 256, dtype=np.uint8)
        reward = 1.0 if action == 1 else 0.0
        terminated = self._step_count >= 3
        truncated = False
        info = {"ale": "fake", "step": self._step_count}
        return obs, reward, terminated, truncated, info

    def close(self):
        pass


def _patched_env_module(fake_env: _FakeGymEnv):
    """Return patchers for gym.make + ale_py import used by ``AtariEnv``."""
    import games.atari.env as atari_env  # noqa: PLC0415

    # ``import ale_py`` would normally fail; inject a stub.
    fake_ale_py = types.ModuleType("ale_py")
    sys.modules.setdefault("ale_py", fake_ale_py)

    # gym.make should return our fake env regardless of the env id passed.
    patcher = patch.object(atari_env.gym, "make", return_value=fake_env)
    return patcher


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAtariEnvBasics(unittest.TestCase):
    def test_reset_returns_float32_ram_vector(self):
        fake = _FakeGymEnv(n_actions=6)
        with _patched_env_module(fake):
            from games.atari.env import AtariEnv  # noqa: PLC0415

            env = AtariEnv(map_name="Pong-v5", max_episode_steps=10)
            obs, info = env.reset(seed=0)
            self.assertEqual(obs.shape, (128,))
            self.assertEqual(obs.dtype, np.float32)
            # values come from the fake env (0..127)
            np.testing.assert_array_equal(obs, np.arange(128, dtype=np.float32))
            self.assertIn("seed", info)

    def test_step_returns_5_tuple_with_native_reward(self):
        fake = _FakeGymEnv(n_actions=6)
        with _patched_env_module(fake):
            from games.atari.env import AtariEnv  # noqa: PLC0415

            env = AtariEnv(map_name="Pong-v5", max_episode_steps=10)
            env.reset(seed=0)
            obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
            self.assertEqual(obs.shape, (128,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIn("native_reward", info)
            self.assertIn("action_index", info)

    def test_episode_terminates_at_step_limit_of_fake_env(self):
        fake = _FakeGymEnv(n_actions=6)
        with _patched_env_module(fake):
            from games.atari.env import AtariEnv  # noqa: PLC0415

            env = AtariEnv(map_name="Pong-v5", max_episode_steps=10)
            env.reset(seed=0)
            done = False
            for _ in range(5):
                _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
                if terminated or truncated:
                    done = True
                    break
            self.assertTrue(done)

    def test_n_legal_actions_taken_from_underlying_env(self):
        fake = _FakeGymEnv(n_actions=4)
        with _patched_env_module(fake):
            from games.atari.env import AtariEnv  # noqa: PLC0415

            env = AtariEnv(map_name="Boxing-v5")
            self.assertEqual(env.n_legal_actions, 4)
            self.assertEqual(env.map_name, "Boxing-v5")


class TestAtariEnvActionMapping(unittest.TestCase):
    def test_continuous_near_minus_one_maps_to_first_action(self):
        fake = _FakeGymEnv(n_actions=6)
        with _patched_env_module(fake):
            from games.atari.env import AtariEnv  # noqa: PLC0415

            env = AtariEnv(map_name="Pong-v5")
            env.reset(seed=0)
            env.step(np.array([-1.0], dtype=np.float32))
            self.assertEqual(fake.last_action, 0)

    def test_continuous_near_plus_one_maps_to_last_legal_action(self):
        fake = _FakeGymEnv(n_actions=6)
        with _patched_env_module(fake):
            from games.atari.env import AtariEnv  # noqa: PLC0415

            env = AtariEnv(map_name="Pong-v5")
            env.reset(seed=0)
            # Use 0.999 — unambiguously continuous (not integer-like)
            env.step(np.array([0.999], dtype=np.float32))
            self.assertEqual(fake.last_action, 5)

    def test_continuous_near_zero_maps_to_middle_action(self):
        fake = _FakeGymEnv(n_actions=6)
        with _patched_env_module(fake):
            from games.atari.env import AtariEnv  # noqa: PLC0415

            env = AtariEnv(map_name="Pong-v5")
            env.reset(seed=0)
            # Use 0.001 — unambiguously continuous (not integer-like)
            env.step(np.array([0.001], dtype=np.float32))
            # 0.001 in [-1, 1] → (0.001+1)/2 * (6-1) ≈ 2.5 → round → 3
            self.assertIn(fake.last_action, (2, 3))

    def test_discrete_index_zero_passes_through(self):
        """Discrete index 0 (integer-like) must not be remapped by the continuous path."""
        fake = _FakeGymEnv(n_actions=6)
        with _patched_env_module(fake):
            from games.atari.env import AtariEnv  # noqa: PLC0415

            env = AtariEnv(map_name="Pong-v5")
            env.reset(seed=0)
            env.step(np.array([0.0], dtype=np.float32))
            self.assertEqual(fake.last_action, 0)

    def test_discrete_index_one_passes_through(self):
        """Discrete index 1 (integer-like) must not be remapped by the continuous path."""
        fake = _FakeGymEnv(n_actions=6)
        with _patched_env_module(fake):
            from games.atari.env import AtariEnv  # noqa: PLC0415

            env = AtariEnv(map_name="Pong-v5")
            env.reset(seed=0)
            env.step(np.array([1.0], dtype=np.float32))
            self.assertEqual(fake.last_action, 1)

    def test_out_of_range_discrete_index_clamps_to_noop(self):
        """Tabular policies on games with smaller legal sets may emit
        DISCRETE_ACTIONS row 17 even though the game only has 6 — clamp
        the resulting underlying action to NOOP rather than crashing."""
        fake = _FakeGymEnv(n_actions=6)
        with _patched_env_module(fake):
            from games.atari.env import AtariEnv  # noqa: PLC0415

            env = AtariEnv(map_name="Pong-v5")
            env.reset(seed=0)
            env.step(np.array([17.0], dtype=np.float32))
            self.assertEqual(fake.last_action, 0)

    def test_in_range_discrete_index_passes_through(self):
        """An integer-valued action in [0, n_legal) outside [-1, 1] must
        be treated as a discrete action index — not collapsed to NOOP."""
        fake = _FakeGymEnv(n_actions=6)
        with _patched_env_module(fake):
            from games.atari.env import AtariEnv  # noqa: PLC0415

            env = AtariEnv(map_name="Pong-v5")
            env.reset(seed=0)
            env.step(np.array([3.0], dtype=np.float32))
            self.assertEqual(fake.last_action, 3)


class TestAtariEnvIdResolution(unittest.TestCase):
    def test_bare_name_gets_ale_prefix(self):
        from games.atari.env import _resolve_env_id  # noqa: PLC0415

        self.assertEqual(_resolve_env_id("Pong-v5"), "ALE/Pong-v5")

    def test_already_qualified_id_unchanged(self):
        from games.atari.env import _resolve_env_id  # noqa: PLC0415

        self.assertEqual(_resolve_env_id("ALE/Breakout-v5"), "ALE/Breakout-v5")


if __name__ == "__main__":
    unittest.main()
