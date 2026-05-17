"""Tests for games/sc2/eval.py.

SC2Env and SC2Client are mocked throughout — no PySC2 installation needed.
"""

import argparse
import collections
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from games.sc2.obs_spec import BASE_OBS_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(dim: int = BASE_OBS_DIM) -> np.ndarray:
    return np.zeros(dim, dtype=np.float32)


def _make_info(outcome: float | None = None, game_loop: float = 500.0) -> dict:
    return {
        "score": 10.0,
        "player_outcome": outcome,
        "game_loop": game_loop,
        "available_fn_ids": [0, 1, 2],
    }


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "track": None,
        "num_episodes": 1,
        "bot_difficulty": None,
        "eval_speed": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# CLI argument validation
# ---------------------------------------------------------------------------

class TestArgValidation(unittest.TestCase):
    """--num-episodes and --eval-speed must be ≥ 1; exercises the real main.py parser."""

    def _parser(self):
        from main import _build_arg_parser
        return _build_arg_parser()

    def test_num_episodes_zero_rejected(self):
        with self.assertRaises(SystemExit):
            self._parser().parse_args(["myexp", "--num-episodes", "0"])

    def test_num_episodes_negative_rejected(self):
        with self.assertRaises(SystemExit):
            self._parser().parse_args(["myexp", "--num-episodes", "-3"])

    def test_num_episodes_one_accepted(self):
        ns = self._parser().parse_args(["myexp", "--num-episodes", "1"])
        self.assertEqual(ns.num_episodes, 1)

    def test_eval_speed_zero_rejected(self):
        with self.assertRaises(SystemExit):
            self._parser().parse_args(["myexp", "--eval-speed", "0"])

    def test_eval_speed_positive_accepted(self):
        ns = self._parser().parse_args(["myexp", "--eval-speed", "4"])
        self.assertEqual(ns.eval_speed, 4)

    def test_play_and_eval_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            self._parser().parse_args(["myexp", "--play", "--eval"])

    def test_bot_difficulty_invalid_name_rejected(self):
        """cheater_easy / cheater_hard / elite are not real PySC2 Difficulty names."""
        for bad in ("cheater_easy", "cheater_hard", "elite"):
            with self.subTest(name=bad), self.assertRaises(SystemExit):
                self._parser().parse_args(["myexp", "--bot-difficulty", bad])

    def test_bot_difficulty_valid_names_accepted(self):
        for name in ("very_easy", "hard", "cheat_insane"):
            with self.subTest(name=name):
                ns = self._parser().parse_args(["myexp", "--bot-difficulty", name])
                self.assertEqual(ns.bot_difficulty, name)


# ---------------------------------------------------------------------------
# _print_action_breakdown
# ---------------------------------------------------------------------------

class TestPrintActionBreakdown(unittest.TestCase):

    def _capture(self, counts, total, subs=0, label="test"):
        from games.sc2.eval import _print_action_breakdown
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _print_action_breakdown(counts, total, subs, label=label)
        return buf.getvalue()

    def test_output_contains_fn_names(self):
        counts = collections.Counter({"no_op": 10, "Move_screen": 5})
        out = self._capture(counts, 15)
        self.assertIn("no_op", out)
        self.assertIn("Move_screen", out)

    def test_percentages_sum_to_100_when_complete(self):
        counts = collections.Counter({"no_op": 7, "select_army": 3})
        out = self._capture(counts, 10)
        # Both percentages appear in the output.
        self.assertIn("70.0", out)
        self.assertIn("30.0", out)

    def test_substitution_line_shown_when_nonzero(self):
        counts = collections.Counter({"no_op": 5})
        out = self._capture(counts, 10, subs=3)
        self.assertIn("substituted", out)

    def test_substitution_line_hidden_when_zero(self):
        counts = collections.Counter({"no_op": 10})
        out = self._capture(counts, 10, subs=0)
        self.assertNotIn("substituted", out)

    def test_zero_total_steps_does_not_divide_by_zero(self):
        counts: collections.Counter = collections.Counter()
        # Should not raise.
        self._capture(counts, 0)


# ---------------------------------------------------------------------------
# _print_aggregate_summary
# ---------------------------------------------------------------------------

class TestPrintAggregateSummary(unittest.TestCase):

    def _make_result(self, outcome, score=10.0, game_loop=500, steps=20,
                     reward=5.0, subs=0):
        return {
            "outcome": outcome,
            "score": score,
            "game_loop": game_loop,
            "steps": steps,
            "cumulative_reward": reward,
            "action_counts": collections.Counter({"no_op": steps}),
            "substitution_count": subs,
        }

    def _capture(self, results, counts):
        from games.sc2.eval import _print_aggregate_summary
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _print_aggregate_summary(results, counts)
        return buf.getvalue()

    def test_win_rate_correct(self):
        results = [
            self._make_result(1),   # win
            self._make_result(-1),  # loss
            self._make_result(1),   # win
        ]
        all_counts = sum((r["action_counts"] for r in results), collections.Counter())
        out = self._capture(results, all_counts)
        self.assertIn("66.7%", out)

    def test_all_wins(self):
        results = [self._make_result(1) for _ in range(3)]
        all_counts = sum((r["action_counts"] for r in results), collections.Counter())
        out = self._capture(results, all_counts)
        self.assertIn("100.0%", out)

    def test_single_episode_no_crash(self):
        results = [self._make_result(0)]
        all_counts = results[0]["action_counts"].copy()
        # Should not raise.
        self._capture(results, all_counts)


# ---------------------------------------------------------------------------
# _run_episode
# ---------------------------------------------------------------------------

class TestRunEpisode(unittest.TestCase):
    """Episode loop: policy called, actions logged, lifecycle hooks invoked."""

    def _make_env(self, n_steps=3, obs_dim=BASE_OBS_DIM, outcome=None):
        """Return a mock SC2Env that terminates after n_steps."""
        env = MagicMock()
        env.reset.return_value = (_make_obs(obs_dim), _make_info())

        fake_client = MagicMock()
        fake_client.last_fn_idx = 1  # select_army
        env._client = fake_client

        steps = []
        for i in range(n_steps):
            done = i == n_steps - 1
            steps.append((_make_obs(obs_dim), 1.0, done, False,
                          _make_info(outcome=outcome if done else None)))
        env.step.side_effect = steps
        return env

    def _make_policy(self, obs_dim=BASE_OBS_DIM):
        policy = MagicMock()
        policy.return_value = np.zeros(4, dtype=np.float32)
        return policy

    def test_policy_called_each_step(self):
        from games.sc2.eval import _run_episode
        env = self._make_env(n_steps=4)
        policy = self._make_policy()
        _run_episode(env, policy, ep_idx=1, total_episodes=1)
        self.assertEqual(policy.call_count, 4)

    def test_result_step_count_matches_env_steps(self):
        from games.sc2.eval import _run_episode
        env = self._make_env(n_steps=5)
        policy = self._make_policy()
        result = _run_episode(env, policy, ep_idx=1, total_episodes=1)
        self.assertEqual(result["steps"], 5)

    def test_on_episode_start_called_with_info_kwarg(self):
        """on_episode_start(info=reset_info) — info dict passed as 'info' kwarg,
        not spread as top-level kwargs, so policies can read available_fn_ids."""
        from games.sc2.eval import _run_episode
        env = self._make_env(n_steps=1)
        policy = self._make_policy()
        policy.on_episode_start = MagicMock()
        _run_episode(env, policy, ep_idx=1, total_episodes=1)
        policy.on_episode_start.assert_called_once()
        kwargs = policy.on_episode_start.call_args[1]
        # Must be passed as info=<dict>, not as **info.
        self.assertIn("info", kwargs, "on_episode_start must receive info=<dict>")
        self.assertIn("available_fn_ids", kwargs["info"])

    def test_policy_update_receives_correct_signature(self):
        """update(prev_obs, action, reward, next_obs, done, info=info) —
        verify positional and keyword args match the policy contract."""
        from games.sc2.eval import _run_episode
        obs_dim = BASE_OBS_DIM

        received_calls = []

        class _FakePolicy:
            def __call__(self, obs):
                return np.array([1, 0.0, 0.0, 0], dtype=np.float32)
            def update(self, obs, action, reward, next_obs, done, **kwargs):
                received_calls.append({
                    "obs": obs, "action": action, "reward": reward,
                    "next_obs": next_obs, "done": done,
                    "info": kwargs.get("info"),
                })

        env = self._make_env(n_steps=2, obs_dim=obs_dim)
        _run_episode(env, _FakePolicy(), ep_idx=1, total_episodes=1)

        self.assertEqual(len(received_calls), 2)
        for i, call in enumerate(received_calls):
            self.assertEqual(call["obs"].shape, (obs_dim,), f"step {i}: obs wrong shape")
            self.assertEqual(call["next_obs"].shape, (obs_dim,), f"step {i}: next_obs wrong shape")
            self.assertIsNotNone(call["info"], f"step {i}: info must be passed")
            self.assertIn("available_fn_ids", call["info"], f"step {i}: available_fn_ids missing")
        # done=True on the last step.
        self.assertTrue(received_calls[-1]["done"])

    def test_result_outcome_from_terminal_info(self):
        from games.sc2.eval import _run_episode
        env = self._make_env(n_steps=2, outcome=1)
        policy = self._make_policy()
        result = _run_episode(env, policy, ep_idx=1, total_episodes=1)
        self.assertEqual(result["outcome"], 1)

    def test_substitution_counted_when_executed_differs_from_requested(self):
        """When the client substitutes a different action, substitution_count
        should increase."""
        from games.sc2.eval import _run_episode

        env = self._make_env(n_steps=2)
        # Policy requests fn_idx=2 (Move_screen) but client executes fn_idx=1
        # (select_army).
        env._client.last_fn_idx = 1
        policy = self._make_policy()
        policy.return_value = np.array([2, 0.5, 0.5, 0], dtype=np.float32)

        result = _run_episode(env, policy, ep_idx=1, total_episodes=1)
        self.assertEqual(result["substitution_count"], 2)

    def test_no_substitution_when_executed_matches_requested(self):
        from games.sc2.eval import _run_episode

        env = self._make_env(n_steps=3)
        # Policy requests fn_idx=1 and client also executes fn_idx=1.
        env._client.last_fn_idx = 1
        policy = self._make_policy()
        policy.return_value = np.array([1, 0.0, 0.0, 0], dtype=np.float32)

        result = _run_episode(env, policy, ep_idx=1, total_episodes=1)
        self.assertEqual(result["substitution_count"], 0)

    def test_cumulative_reward_summed(self):
        from games.sc2.eval import _run_episode
        env = self._make_env(n_steps=4)
        policy = self._make_policy()
        result = _run_episode(env, policy, ep_idx=1, total_episodes=1)
        self.assertAlmostEqual(result["cumulative_reward"], 4.0)


if __name__ == "__main__":
    unittest.main()
