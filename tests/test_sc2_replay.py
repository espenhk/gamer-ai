"""Tests for SC2 replay saving on new-best events.

Covers:
  - SC2Client.save_replay  — delegates to pysc2 env; handles None and exceptions
  - SC2Env.save_replay     — thin delegation to the client
  - _try_save_replay       — no-op for non-SC2 envs; sequential naming; exception safety
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from games.sc2.client import SC2Client
from games.sc2.env import SC2Env
from framework.training import _try_save_replay


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

        with patch("os.makedirs"):
            result = client.save_replay("/tmp/replays", "myrun_best-01")

        mock_env.save_replay.assert_called_once_with("/tmp/replays", prefix="myrun_best-01")
        self.assertEqual(result, "/tmp/replays/myrun_best-01.SC2Replay")

    def test_swallows_exception_and_returns_none(self):
        client = self._make_client()
        mock_env = MagicMock()
        mock_env.save_replay.side_effect = RuntimeError("SC2 crashed")
        client._sc2_env = mock_env

        with patch("os.makedirs"):
            result = client.save_replay("/tmp/replays", "myrun_best-01")

        self.assertIsNone(result)


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
# _try_save_replay
# ---------------------------------------------------------------------------

class TestTrySaveReplay(unittest.TestCase):

    def test_noop_for_non_sc2_env(self):
        """An env without save_replay is silently skipped."""
        plain_env = object()
        # Should complete without error and not attempt to call anything.
        _try_save_replay(plain_env, "/some/exp/policy_weights.yaml")

    def test_first_replay_numbered_01(self, tmp_path=None):
        """First new-best → _best-01 prefix when no replays exist yet."""
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as experiment_dir:
            weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
            experiment_name = os.path.basename(experiment_dir)

            env = MagicMock()
            env.save_replay.return_value = os.path.join(
                experiment_dir, "replays", f"{experiment_name}_best-01.SC2Replay"
            )

            _try_save_replay(env, weights_file)

            replay_dir = os.path.join(experiment_dir, "replays")
            expected_prefix = f"{experiment_name}_best-01"
            env.save_replay.assert_called_once_with(replay_dir, prefix=expected_prefix)

    def test_second_replay_numbered_02(self):
        """Second new-best → _best-02 when one .SC2Replay file already exists."""
        import tempfile
        with tempfile.TemporaryDirectory() as experiment_dir:
            weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
            experiment_name = os.path.basename(experiment_dir)

            # Pre-create a replay dir with one existing replay.
            replay_dir = os.path.join(experiment_dir, "replays")
            os.makedirs(replay_dir)
            open(os.path.join(replay_dir, f"{experiment_name}_best-01.SC2Replay"), "w").close()

            env = MagicMock()
            env.save_replay.return_value = None

            _try_save_replay(env, weights_file)

            expected_prefix = f"{experiment_name}_best-02"
            env.save_replay.assert_called_once_with(replay_dir, prefix=expected_prefix)

    def test_exception_from_save_replay_is_swallowed(self):
        """A failing save_replay must not propagate and abort training."""
        import tempfile
        with tempfile.TemporaryDirectory() as experiment_dir:
            weights_file = os.path.join(experiment_dir, "policy_weights.yaml")

            env = MagicMock()
            env.save_replay.side_effect = RuntimeError("save failed")

            # Should not raise.
            _try_save_replay(env, weights_file)

    def test_replay_dir_is_inside_experiment_dir(self):
        """replay_dir is always <experiment_dir>/replays/."""
        import tempfile
        with tempfile.TemporaryDirectory() as experiment_dir:
            weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
            env = MagicMock()
            env.save_replay.return_value = None

            _try_save_replay(env, weights_file)

            call_args = env.save_replay.call_args
            replay_dir_used = call_args[0][0]
            self.assertEqual(replay_dir_used, os.path.join(experiment_dir, "replays"))

    def test_only_sc2replay_files_counted_for_numbering(self):
        """Non-.SC2Replay files in the replay dir are not counted."""
        import tempfile
        with tempfile.TemporaryDirectory() as experiment_dir:
            weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
            experiment_name = os.path.basename(experiment_dir)

            replay_dir = os.path.join(experiment_dir, "replays")
            os.makedirs(replay_dir)
            # A txt file and a .yaml — neither should count.
            open(os.path.join(replay_dir, "some_notes.txt"), "w").close()
            open(os.path.join(replay_dir, "config.yaml"), "w").close()

            env = MagicMock()
            env.save_replay.return_value = None

            _try_save_replay(env, weights_file)

            # Still _best-01 because no .SC2Replay files exist.
            expected_prefix = f"{experiment_name}_best-01"
            env.save_replay.assert_called_once_with(replay_dir, prefix=expected_prefix)


if __name__ == "__main__":
    unittest.main()
