"""Unit tests for :class:`games.sc2.apm_limiter.ApmLimiter` and its
integration into :class:`games.sc2.env.SC2Env`.

All tests are pure-Python: no SC2 binary, no PySC2 import, no real env.
The env tests mock ``SC2Client`` as every other test_sc2_env test does.
"""
import unittest
from unittest.mock import patch

import numpy as np

from games.sc2.apm_limiter import ApmLimiter
from games.sc2.env import SC2Env
from games.sc2.obs_spec import BASE_OBS_DIM


# ---------------------------------------------------------------------------
# ApmLimiter unit tests
# ---------------------------------------------------------------------------

class TestApmLimiterConstruction(unittest.TestCase):

    def test_valid_construction(self):
        lim = ApmLimiter(max_apm=300)
        self.assertGreater(lim.max_tokens, 0)

    def test_invalid_max_apm_zero(self):
        with self.assertRaises(ValueError):
            ApmLimiter(max_apm=0)

    def test_invalid_max_apm_negative(self):
        with self.assertRaises(ValueError):
            ApmLimiter(max_apm=-100)

    def test_invalid_burst_s_zero(self):
        with self.assertRaises(ValueError):
            ApmLimiter(max_apm=300, burst_s=0)

    def test_invalid_burst_s_negative(self):
        with self.assertRaises(ValueError):
            ApmLimiter(max_apm=300, burst_s=-1.0)

    def test_max_tokens_formula(self):
        # max_tokens = (max_apm / 60) * burst_s
        lim = ApmLimiter(max_apm=60, burst_s=5.0)
        self.assertAlmostEqual(lim.max_tokens, 5.0)

    def test_tokens_start_full(self):
        lim = ApmLimiter(max_apm=300, burst_s=2.0)
        self.assertAlmostEqual(lim.tokens, lim.max_tokens)


class TestApmLimiterBasicBehaviour(unittest.TestCase):

    def _lim(self, max_apm=60, burst_s=1.0):
        """60 APM → 1 token/s; max 1 token with burst_s=1."""
        return ApmLimiter(max_apm=max_apm, burst_s=burst_s)

    def test_noop_always_allowed(self):
        """fn_idx=0 (no_op) must never be blocked regardless of token count."""
        lim = self._lim()
        lim.reset(0.0)
        # Drain all tokens.
        for _ in range(100):
            lim.allow(0.0, fn_idx=1)
        # No-op must still be allowed.
        self.assertTrue(lim.allow(0.0, fn_idx=0))

    def test_noop_does_not_consume_token(self):
        """A no-op call should leave the bucket unchanged."""
        lim = self._lim()
        lim.reset(0.0)
        tokens_before = lim.tokens
        lim.allow(0.0, fn_idx=0)
        # Tokens unchanged (no time elapsed, no action taken).
        self.assertAlmostEqual(lim.tokens, tokens_before)

    def test_first_action_allowed_when_full(self):
        lim = self._lim()
        lim.reset(0.0)
        self.assertTrue(lim.allow(0.0, fn_idx=1))

    def test_second_action_blocked_when_empty(self):
        """60 APM / burst=1s → 1 max token; first costs the token, second blocked."""
        lim = self._lim(max_apm=60, burst_s=1.0)
        lim.reset(0.0)
        lim.allow(0.0, fn_idx=1)   # consumes the token
        self.assertFalse(lim.allow(0.0, fn_idx=1))

    def test_token_refills_over_time(self):
        """After enough time passes the limiter allows another action."""
        lim = self._lim(max_apm=60, burst_s=1.0)
        lim.reset(0.0)
        lim.allow(0.0, fn_idx=1)         # drain
        self.assertFalse(lim.allow(0.0, fn_idx=1))  # still empty
        # 1 second later → +1 token refilled → should be allowed.
        self.assertTrue(lim.allow(1.0, fn_idx=1))

    def test_tokens_capped_at_max(self):
        """Waiting a very long time should not exceed max_tokens."""
        lim = self._lim(max_apm=60, burst_s=2.0)
        lim.reset(0.0)
        # Drain all tokens.
        lim.allow(0.0, fn_idx=1)
        lim.allow(0.0, fn_idx=1)
        # Wait 1000 seconds — tokens should cap at max_tokens.
        lim.allow(1000.0, fn_idx=1)    # refill + one action
        self.assertLessEqual(lim.tokens, lim.max_tokens)

    def test_reset_refills_bucket(self):
        lim = self._lim()
        lim.reset(0.0)
        lim.allow(0.0, fn_idx=1)       # drain
        lim.reset(10.0)
        self.assertAlmostEqual(lim.tokens, lim.max_tokens)

    def test_burst_capacity(self):
        """burst_s=3 at 60 APM → 3 tokens max; 3 immediate actions should succeed."""
        lim = ApmLimiter(max_apm=60, burst_s=3.0)
        lim.reset(0.0)
        for _ in range(3):
            self.assertTrue(lim.allow(0.0, fn_idx=1))
        # Fourth must be blocked.
        self.assertFalse(lim.allow(0.0, fn_idx=1))

    def test_high_apm_allows_many_actions_per_second(self):
        """300 APM with burst_s=2 → 10 immediate actions should all succeed."""
        lim = ApmLimiter(max_apm=300, burst_s=2.0)
        lim.reset(0.0)
        allowed = sum(1 for _ in range(15) if lim.allow(0.0, fn_idx=1))
        # max_tokens = (300/60)*2 = 10; so exactly 10 should pass.
        self.assertEqual(allowed, 10)

    def test_default_fn_idx_minus1_consumes_token(self):
        """Default fn_idx=-1 (unknown) should behave like a real action."""
        lim = ApmLimiter(max_apm=60, burst_s=1.0)
        lim.reset(0.0)
        self.assertTrue(lim.allow(0.0))      # fn_idx defaults to -1
        self.assertFalse(lim.allow(0.0))

    def test_protect_burst_budget_blocks_after_steady_capacity(self):
        """protect_burst_budget should keep burst headroom untouched."""
        lim = ApmLimiter(max_apm=300, burst_s=2.0)  # refill_rate=5, max_tokens=10
        lim.reset(0.0)
        allowed = sum(
            1 for _ in range(15)
            if lim.allow(0.0, fn_idx=1, protect_burst_budget=True)
        )
        self.assertEqual(allowed, 5)
        self.assertAlmostEqual(lim.tokens, 5.0)


class TestApmLimiterRollingBudget(unittest.TestCase):
    """Verify that the rolling budget prevents "all in first second" patterns."""

    def test_300_apm_spread_over_60_seconds(self):
        """At 300 APM the limiter should allow ~300 actions over 60 seconds,
        not 300 in the first second."""
        lim = ApmLimiter(max_apm=300, burst_s=2.0)
        lim.reset(0.0)

        allowed_first_second = 0
        total_allowed = 0
        # Simulate steps spaced 1/10 s apart over 60 s → 600 steps.
        for i in range(600):
            t = i * 0.1
            if lim.allow(t, fn_idx=1):
                total_allowed += 1
                if t < 1.0:
                    allowed_first_second += 1

        # At 300 APM = 5/s, over 60 s the total should be ~300 (within 5%).
        self.assertAlmostEqual(total_allowed, 300, delta=15)
        # The first second should allow at most burst_s * refill_rate = 10.
        self.assertLessEqual(allowed_first_second, 10)


# ---------------------------------------------------------------------------
# SC2Env integration tests
# ---------------------------------------------------------------------------

def _make_mock_env(max_apm=None, apm_burst_s=2.0):
    """Return a mocked SC2Env with optional APM limiting."""
    patcher = patch("games.sc2.env.SC2Client")
    mock_cls = patcher.start()
    mock_client = mock_cls.return_value
    mock_client.reset.return_value = (
        np.zeros(BASE_OBS_DIM, dtype=np.float32),
        {"score": 0.0, "minerals": 50.0, "vespene": 0.0,
         "food_used": 1.0, "food_cap": 15.0, "army_count": 0.0},
    )
    mock_client.step.return_value = (
        np.zeros(BASE_OBS_DIM, dtype=np.float32),
        0.0,
        False,
        {"score": 0.0, "minerals": 50.0, "vespene": 0.0,
         "food_used": 1.0, "food_cap": 15.0, "army_count": 0.0},
    )
    env = SC2Env(
        map_name="MoveToBeacon",
        max_apm=max_apm,
        apm_burst_s=apm_burst_s,
    )
    return env, mock_client, patcher


class TestSC2EnvApmLimiterDisabled(unittest.TestCase):
    """When max_apm is None the env behaves exactly as before."""

    def setUp(self):
        self.env, self.mock_client, self.patcher = _make_mock_env(max_apm=None)
        self.addCleanup(self.patcher.stop)

    def test_no_limiter_attribute_when_disabled(self):
        self.assertIsNone(self.env._apm_limiter)

    def test_step_passes_action_unchanged(self):
        self.env.reset()
        action = np.array([1.0, 0.5, 0.5, 0.0], dtype=np.float32)
        self.env.step(action)
        called_action = self.mock_client.step.call_args[0][0]
        np.testing.assert_array_equal(called_action, action)

    def test_apm_throttled_false_when_disabled(self):
        self.env.reset()
        _, _, _, _, info = self.env.step(np.array([1.0, 0.5, 0.5, 0.0], dtype=np.float32))
        self.assertFalse(info["apm_throttled"])

    def test_episode_apm_throttled_steps_zero(self):
        self.env.reset()
        _, _, _, _, info = self.env.step(np.array([1.0, 0.5, 0.5, 0.0], dtype=np.float32))
        self.assertEqual(info["episode_apm_throttled_steps"], 0)


class TestSC2EnvApmLimiterEnabled(unittest.TestCase):
    """When max_apm is set, actions are throttled once the bucket is empty."""

    def setUp(self):
        # 60 APM, burst_s=1 → exactly 1 max token; first action passes, rest blocked
        # until time advances.
        self.env, self.mock_client, self.patcher = _make_mock_env(
            max_apm=60, apm_burst_s=1.0
        )
        self.addCleanup(self.patcher.stop)

    def _do_reset(self):
        with patch("games.sc2.env.time") as mock_time:
            mock_time.monotonic.return_value = 0.0
            obs, info = self.env.reset()
        return obs, info

    def _do_step(self, action, now=0.0):
        with patch("games.sc2.env.time") as mock_time:
            mock_time.monotonic.return_value = now
            return self.env.step(action)

    def test_limiter_created(self):
        self.assertIsNotNone(self.env._apm_limiter)

    def test_first_action_passes(self):
        self._do_reset()
        select_army = np.array([1.0, 0.5, 0.5, 0.0], dtype=np.float32)
        _, _, _, _, info = self._do_step(select_army, now=0.0)
        self.assertFalse(info["apm_throttled"])
        called = self.mock_client.step.call_args[0][0]
        # fn_idx should still be 1 (select_army).
        self.assertEqual(int(called[0]), 1)

    def test_second_action_throttled_to_noop(self):
        self._do_reset()
        select_army = np.array([1.0, 0.5, 0.5, 0.0], dtype=np.float32)
        # First step at t=0 drains the token.
        self._do_step(select_army, now=0.0)
        # Second step at same time — bucket empty → should be replaced with no_op.
        _, _, _, _, info = self._do_step(select_army, now=0.0)
        self.assertTrue(info["apm_throttled"])
        called = self.mock_client.step.call_args[0][0]
        self.assertEqual(int(called[0]), 0)  # fn_idx 0 = no_op

    def test_noop_not_throttled(self):
        """fn_idx=0 (no_op) is always allowed and never sets apm_throttled."""
        self._do_reset()
        # Drain the token.
        select_army = np.array([1.0, 0.5, 0.5, 0.0], dtype=np.float32)
        self._do_step(select_army, now=0.0)
        # Now issue a no_op — should not be throttled.
        no_op = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32)
        _, _, _, _, info = self._do_step(no_op, now=0.0)
        self.assertFalse(info["apm_throttled"])

    def test_episode_apm_throttled_steps_accumulates(self):
        self._do_reset()
        select_army = np.array([1.0, 0.5, 0.5, 0.0], dtype=np.float32)
        self._do_step(select_army, now=0.0)   # passes (1 token consumed)
        self._do_step(select_army, now=0.0)   # throttled → count = 1
        self._do_step(select_army, now=0.0)   # throttled → count = 2
        _, _, _, _, info = self._do_step(select_army, now=0.0)
        self.assertEqual(info["episode_apm_throttled_steps"], 3)

    def test_episode_apm_throttled_steps_reset_on_new_episode(self):
        self._do_reset()
        select_army = np.array([1.0, 0.5, 0.5, 0.0], dtype=np.float32)
        self._do_step(select_army, now=0.0)
        self._do_step(select_army, now=0.0)
        # Start a new episode.
        self._do_reset()
        _, _, _, _, info = self._do_step(select_army, now=0.0)
        self.assertEqual(info["episode_apm_throttled_steps"], 0)

    def test_action_passes_after_refill(self):
        self._do_reset()
        select_army = np.array([1.0, 0.5, 0.5, 0.0], dtype=np.float32)
        self._do_step(select_army, now=0.0)          # drains token
        self._do_step(select_army, now=0.0)          # throttled
        # Advance 1 second → refill → should pass again.
        _, _, _, _, info = self._do_step(select_army, now=1.0)
        self.assertFalse(info["apm_throttled"])

    def test_burst_budget_always_protected(self):
        env, mock_client, patcher = _make_mock_env(max_apm=300, apm_burst_s=2.0)
        self.addCleanup(patcher.stop)

        with patch("games.sc2.env.time") as mock_time:
            mock_time.monotonic.return_value = 0.0
            env.reset()
            select_army = np.array([1.0, 0.5, 0.5, 0.0], dtype=np.float32)
            allowed = 0
            for _ in range(10):
                _, _, _, _, info = env.step(select_army)
                if not info["apm_throttled"]:
                    allowed += 1

        self.assertEqual(allowed, 5)
        called = mock_client.step.call_args[0][0]
        self.assertEqual(int(called[0]), 0)


if __name__ == "__main__":
    unittest.main()
