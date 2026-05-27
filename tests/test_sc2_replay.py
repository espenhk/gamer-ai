"""Tests for SC2 replay saving on new-best events.

Covers:
  - SC2Client.save_replay       — delegates to pysc2 env; makedirs inside try;
                                   handles None and exceptions
  - SC2Env.save_replay          — thin delegation to the client
  - _try_save_replay            — single-episode loop helper (no-op for non-SC2;
                                   sequential naming; candidate files excluded from count)
  - _save_candidate_replay      — speculative save before next reset
  - _finalize_candidate_replay  — rename candidate to sequential best-N name
  - _discard_candidate_replay   — deletes temp candidate file; no-op when missing
"""

import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from framework.training import (
    _discard_candidate_replay,
    _finalize_candidate_replay,
    _save_candidate_replay,
    _try_save_replay,
)
from games.sc2.client import SC2Client
from games.sc2.env import SC2Env

# ---------------------------------------------------------------------------
# SC2Client.save_replay
# ---------------------------------------------------------------------------


class TestSC2ClientSaveReplay(unittest.TestCase):
    def _make_client(self):
        return SC2Client(map_name="MoveToBeacon")

    def test_returns_none_when_no_sc2_env(self):
        client = self._make_client()
        self.assertIsNone(client._sc2_env)
        result = client.save_replay("/tmp/replays", "myrun_best-01")
        self.assertIsNone(result)

    def test_delegates_to_pysc2_env(self):
        client = self._make_client()
        mock_env = MagicMock()
        mock_env.save_replay.return_value = "/tmp/replays/myrun_best-01.SC2Replay"
        client._sc2_env = mock_env

        with patch("games.sc2.client.os.makedirs"):
            result = client.save_replay("/tmp/replays", "myrun_best-01")

        mock_env.save_replay.assert_called_once_with("/tmp/replays", prefix="myrun_best-01")
        self.assertEqual(result, "/tmp/replays/myrun_best-01.SC2Replay")

    def test_makedirs_failure_swallowed(self):
        """os.makedirs failure is inside the try block and must not propagate."""
        client = self._make_client()
        mock_env = MagicMock()
        client._sc2_env = mock_env

        with patch("games.sc2.client.os.makedirs", side_effect=OSError("no space")):
            result = client.save_replay("/tmp/replays", "myrun_best-01")

        self.assertIsNone(result)
        mock_env.save_replay.assert_not_called()

    def test_swallows_exception_and_returns_none(self):
        client = self._make_client()
        mock_env = MagicMock()
        mock_env.save_replay.side_effect = RuntimeError("SC2 crashed")
        client._sc2_env = mock_env

        with patch("games.sc2.client.os.makedirs"):
            result = client.save_replay("/tmp/replays", "myrun_best-01")

        self.assertIsNone(result)

    def test_save_timeout_returns_none(self):
        client = self._make_client()
        mock_env = MagicMock()

        def _slow_save(*args, **kwargs):
            time.sleep(0.2)
            return "/tmp/replays/myrun_best-01.SC2Replay"

        mock_env.save_replay.side_effect = _slow_save
        client._sc2_env = mock_env

        with (
            patch("games.sc2.client.os.makedirs"),
            patch("games.sc2.client._SAVE_REPLAY_TIMEOUT_S", 0.01),
        ):
            start = time.monotonic()
            result = client.save_replay("/tmp/replays", "myrun_best-01")
            elapsed = time.monotonic() - start

        self.assertIsNone(result)
        self.assertLess(elapsed, 0.15)


# ---------------------------------------------------------------------------
# SC2Env.save_replay
# ---------------------------------------------------------------------------


class TestSC2EnvSaveReplay(unittest.TestCase):
    def test_delegates_to_client(self):
        with patch("games.sc2.env.SC2Client"):
            env = SC2Env(map_name="MoveToBeacon")

        env._client = MagicMock()
        env._client.save_replay.return_value = "/tmp/replays/run_best-01.SC2Replay"

        result = env.save_replay("/tmp/replays", "run_best-01")

        env._client.save_replay.assert_called_once_with("/tmp/replays", prefix="run_best-01")
        self.assertEqual(result, "/tmp/replays/run_best-01.SC2Replay")


# ---------------------------------------------------------------------------
# _try_save_replay  (single-episode-per-sim loops)
# ---------------------------------------------------------------------------


class TestTrySaveReplay(unittest.TestCase):
    def test_noop_for_non_sc2_env(self):
        plain_env = object()
        _try_save_replay(plain_env, "/some/exp/policy_weights.yaml")

    def test_first_replay_numbered_01(self):
        import tempfile

        with tempfile.TemporaryDirectory() as experiment_dir:
            weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
            experiment_name = os.path.basename(experiment_dir)

            env = MagicMock()
            env.save_replay.return_value = None

            _try_save_replay(env, weights_file)

            replay_dir = os.path.join(experiment_dir, "replays")
            env.save_replay.assert_called_once_with(replay_dir, prefix=f"{experiment_name}_best-01")

    def test_second_replay_numbered_02(self):
        import tempfile

        with tempfile.TemporaryDirectory() as experiment_dir:
            weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
            experiment_name = os.path.basename(experiment_dir)

            replay_dir = os.path.join(experiment_dir, "replays")
            os.makedirs(replay_dir)
            open(os.path.join(replay_dir, f"{experiment_name}_best-01.SC2Replay"), "w").close()

            env = MagicMock()
            env.save_replay.return_value = None

            _try_save_replay(env, weights_file)

            env.save_replay.assert_called_once_with(replay_dir, prefix=f"{experiment_name}_best-02")

    def test_candidate_files_excluded_from_count(self):
        """Candidate files (starting with _) must not be counted as confirmed bests."""
        import tempfile

        with tempfile.TemporaryDirectory() as experiment_dir:
            weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
            experiment_name = os.path.basename(experiment_dir)

            replay_dir = os.path.join(experiment_dir, "replays")
            os.makedirs(replay_dir)
            # A leftover candidate file and one confirmed best.
            open(os.path.join(replay_dir, "_candidate.SC2Replay"), "w").close()
            open(os.path.join(replay_dir, f"{experiment_name}_best-01.SC2Replay"), "w").close()

            env = MagicMock()
            env.save_replay.return_value = None

            _try_save_replay(env, weights_file)

            # Only the confirmed best counts → next is _best-02.
            env.save_replay.assert_called_once_with(replay_dir, prefix=f"{experiment_name}_best-02")

    def test_exception_from_save_replay_is_swallowed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as experiment_dir:
            weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
            env = MagicMock()
            env.save_replay.side_effect = RuntimeError("save failed")
            _try_save_replay(env, weights_file)

    def test_replay_dir_is_inside_experiment_dir(self):
        import tempfile

        with tempfile.TemporaryDirectory() as experiment_dir:
            weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
            env = MagicMock()
            env.save_replay.return_value = None

            _try_save_replay(env, weights_file)

            replay_dir_used = env.save_replay.call_args[0][0]
            self.assertEqual(replay_dir_used, os.path.join(experiment_dir, "replays"))

    def test_non_replay_files_not_counted_for_numbering(self):
        """Files that don't match the {experiment}_best-N naming pattern are ignored."""
        import tempfile

        with tempfile.TemporaryDirectory() as experiment_dir:
            weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
            experiment_name = os.path.basename(experiment_dir)

            replay_dir = os.path.join(experiment_dir, "replays")
            os.makedirs(replay_dir)
            open(os.path.join(replay_dir, "some_notes.txt"), "w").close()
            open(os.path.join(replay_dir, "config.yaml"), "w").close()

            env = MagicMock()
            env.save_replay.return_value = None

            _try_save_replay(env, weights_file)

            env.save_replay.assert_called_once_with(replay_dir, prefix=f"{experiment_name}_best-01")

    def test_prefix_sharing_non_numeric_files_not_counted(self):
        """Files sharing the {experiment}_best- prefix but without a digit suffix are ignored."""
        import tempfile

        with tempfile.TemporaryDirectory() as experiment_dir:
            weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
            experiment_name = os.path.basename(experiment_dir)

            replay_dir = os.path.join(experiment_dir, "replays")
            os.makedirs(replay_dir)
            # These share the prefix but are not numbered replays.
            open(os.path.join(replay_dir, f"{experiment_name}_best-notes.txt"), "w").close()
            open(os.path.join(replay_dir, f"{experiment_name}_best-01.SC2Replay"), "w").close()

            env = MagicMock()
            env.save_replay.return_value = None

            _try_save_replay(env, weights_file)

            # Only the numbered best-01 counts; notes.txt must not inflate the count.
            env.save_replay.assert_called_once_with(replay_dir, prefix=f"{experiment_name}_best-02")


# ---------------------------------------------------------------------------
# _save_candidate_replay
# ---------------------------------------------------------------------------


class TestSaveCandidateReplay(unittest.TestCase):
    def test_noop_for_non_sc2_env(self):
        result = _save_candidate_replay(object(), "/exp/weights.yaml")
        self.assertIsNone(result)

    def test_calls_save_replay_with_candidate_prefix(self):
        import tempfile

        with tempfile.TemporaryDirectory() as experiment_dir:
            weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
            replay_dir = os.path.join(experiment_dir, "replays")
            candidate_path = os.path.join(replay_dir, "_candidate.SC2Replay")

            env = MagicMock()
            env.save_replay.return_value = candidate_path

            result = _save_candidate_replay(env, weights_file)

            env.save_replay.assert_called_once_with(replay_dir, prefix="_candidate")
            self.assertEqual(result, candidate_path)

    def test_swallows_exception_returns_none(self):
        import tempfile

        with tempfile.TemporaryDirectory() as experiment_dir:
            weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
            env = MagicMock()
            env.save_replay.side_effect = RuntimeError("SC2 down")
            result = _save_candidate_replay(env, weights_file)
            self.assertIsNone(result)


# ---------------------------------------------------------------------------
# _finalize_candidate_replay
# ---------------------------------------------------------------------------


class TestFinalizeCandidateReplay(unittest.TestCase):
    def test_noop_when_candidate_path_is_none(self):
        _finalize_candidate_replay(None, "/exp/weights.yaml")

    def test_noop_when_file_missing(self):
        _finalize_candidate_replay("/nonexistent/_candidate.SC2Replay", "/exp/weights.yaml")

    def test_renames_to_best_01_when_no_confirmed_replays(self):
        import tempfile

        with tempfile.TemporaryDirectory() as experiment_dir:
            weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
            experiment_name = os.path.basename(experiment_dir)

            replay_dir = os.path.join(experiment_dir, "replays")
            os.makedirs(replay_dir)
            candidate = os.path.join(replay_dir, "_candidate.SC2Replay")
            open(candidate, "w").close()

            _finalize_candidate_replay(candidate, weights_file)

            expected = os.path.join(replay_dir, f"{experiment_name}_best-01.SC2Replay")
            self.assertTrue(os.path.exists(expected))
            self.assertFalse(os.path.exists(candidate))

    def test_renames_to_best_02_when_one_confirmed_exists(self):
        import tempfile

        with tempfile.TemporaryDirectory() as experiment_dir:
            weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
            experiment_name = os.path.basename(experiment_dir)

            replay_dir = os.path.join(experiment_dir, "replays")
            os.makedirs(replay_dir)
            open(os.path.join(replay_dir, f"{experiment_name}_best-01.SC2Replay"), "w").close()
            candidate = os.path.join(replay_dir, "_candidate.SC2Replay")
            open(candidate, "w").close()

            _finalize_candidate_replay(candidate, weights_file)

            expected = os.path.join(replay_dir, f"{experiment_name}_best-02.SC2Replay")
            self.assertTrue(os.path.exists(expected))
            self.assertFalse(os.path.exists(candidate))

    def test_candidate_not_counted_in_numbering(self):
        """A leftover _candidate.SC2Replay must not inflate the count."""
        import tempfile

        with tempfile.TemporaryDirectory() as experiment_dir:
            weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
            experiment_name = os.path.basename(experiment_dir)

            replay_dir = os.path.join(experiment_dir, "replays")
            os.makedirs(replay_dir)
            # Simulate a leftover candidate (should not count).
            open(os.path.join(replay_dir, "_candidate.SC2Replay"), "w").close()
            # The real candidate we're finalizing.
            candidate = os.path.join(replay_dir, "_candidate_new.SC2Replay")
            open(candidate, "w").close()

            _finalize_candidate_replay(candidate, weights_file)

            # No confirmed replays → should be _best-01.
            expected = os.path.join(replay_dir, f"{experiment_name}_best-01.SC2Replay")
            self.assertTrue(os.path.exists(expected))


# ---------------------------------------------------------------------------
# _discard_candidate_replay
# ---------------------------------------------------------------------------


class TestDiscardCandidateReplay(unittest.TestCase):
    def test_noop_when_none(self):
        _discard_candidate_replay(None)

    def test_noop_when_file_missing(self):
        _discard_candidate_replay("/nonexistent/_candidate.SC2Replay")

    def test_deletes_existing_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "_candidate.SC2Replay")
            open(path, "w").close()
            self.assertTrue(os.path.exists(path))
            _discard_candidate_replay(path)
            self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
