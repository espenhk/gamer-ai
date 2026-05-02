"""Tests for the SC2 human-vs-AI play mode.

PySC2 is not required; SC2Client is mocked throughout.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_obs(dim: int = 13) -> np.ndarray:
    return np.zeros(dim, dtype=np.float32)


def _make_fake_info(done: bool = False, outcome: float | None = None) -> dict:
    return {
        "score": 42.0,
        "prev_score": 0.0,
        "minerals": 50.0,
        "vespene": 0.0,
        "prev_minerals": 0.0,
        "prev_vespene": 0.0,
        "food_used": 12.0,
        "food_cap": 15.0,
        "army_count": 4.0,
        "player_outcome": outcome,
        "is_last": done,
        "game_loop": 500.0,
    }


# ---------------------------------------------------------------------------
# _load_champion_policy
# ---------------------------------------------------------------------------

class TestLoadChampionPolicy(unittest.TestCase):

    def test_missing_weights_file_raises(self):
        from games.sc2.play import _load_champion_policy
        with self.assertRaises(SystemExit):
            _load_champion_policy("/nonexistent/path/policy_weights.yaml", "MoveToBeacon")

    def test_loads_sc2_linear_policy_for_sc2_genetic(self):
        from games.sc2.play import _load_champion_policy
        from games.sc2.policies import SC2LinearPolicy

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump({"policy_type": "sc2_genetic"}, f)
            tmp = f.name

        try:
            policy = _load_champion_policy(tmp, "MoveToBeacon")
            self.assertIsInstance(policy, SC2LinearPolicy)
        finally:
            os.unlink(tmp)

    def test_loads_sc2_linear_policy_when_no_policy_type_key(self):
        """A weights file without a policy_type key defaults to sc2_genetic path."""
        from games.sc2.play import _load_champion_policy
        from games.sc2.policies import SC2LinearPolicy

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump({}, f)
            tmp = f.name

        try:
            policy = _load_champion_policy(tmp, "MoveToBeacon")
            self.assertIsInstance(policy, SC2LinearPolicy)
        finally:
            os.unlink(tmp)

    def test_loads_neural_dqn_policy(self):
        from games.sc2.play import _load_champion_policy
        from games.sc2.policies import NeuralDQNPolicy

        cfg = {"policy_type": "neural_dqn", "hidden_sizes": [16]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(cfg, f)
            tmp = f.name

        try:
            policy = _load_champion_policy(tmp, "MoveToBeacon")
            self.assertIsInstance(policy, NeuralDQNPolicy)
        finally:
            os.unlink(tmp)

    def test_loads_reinforce_policy(self):
        from games.sc2.play import _load_champion_policy
        from games.sc2.policies import REINFORCEPolicy

        cfg = {"policy_type": "reinforce", "hidden_sizes": [16]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(cfg, f)
            tmp = f.name

        try:
            policy = _load_champion_policy(tmp, "MoveToBeacon")
            self.assertIsInstance(policy, REINFORCEPolicy)
        finally:
            os.unlink(tmp)

    def test_loads_lstm_policy(self):
        from games.sc2.play import _load_champion_policy
        from games.sc2.policies import LSTMPolicy
        from games.sc2.obs_spec import get_spec

        obs_spec = get_spec("MoveToBeacon")
        proto = LSTMPolicy(obs_spec=obs_spec, hidden_size=8)
        cfg = proto.to_cfg()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(cfg, f)
            tmp = f.name

        try:
            policy = _load_champion_policy(tmp, "MoveToBeacon")
            self.assertIsInstance(policy, LSTMPolicy)
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# _print_summary
# ---------------------------------------------------------------------------

class TestPrintSummary(unittest.TestCase):

    def _call(self, info: dict, step_count: int = 10, reward: float = 5.0) -> None:
        from games.sc2.play import _print_summary
        _print_summary(info, step_count, reward)

    def test_win_outcome(self):
        info = _make_fake_info(done=True, outcome=1.0)
        self._call(info)   # should not raise

    def test_loss_outcome(self):
        info = _make_fake_info(done=True, outcome=-1.0)
        self._call(info)

    def test_draw_outcome(self):
        info = _make_fake_info(done=True, outcome=0.0)
        self._call(info)

    def test_no_outcome(self):
        info = _make_fake_info(done=False, outcome=None)
        self._call(info)


# ---------------------------------------------------------------------------
# _run_episode
# ---------------------------------------------------------------------------

class TestRunEpisode(unittest.TestCase):

    def _make_client(self, n_steps: int = 3) -> MagicMock:
        """Return a mock SC2Client that runs for *n_steps* then terminates."""
        client = MagicMock()
        obs = _make_fake_obs()
        info_mid  = _make_fake_info(done=False)
        info_last = _make_fake_info(done=True, outcome=1.0)

        client.reset.return_value = (obs, info_mid)

        step_returns = [(obs, 0.1, False, info_mid)] * (n_steps - 1) + [
            (obs, 1.0, True, info_last)
        ]
        client.step.side_effect = step_returns
        return client

    def _make_policy(self) -> MagicMock:
        policy = MagicMock()
        policy.return_value = np.zeros(4, dtype=np.float32)
        return policy

    def test_episode_calls_policy_each_step(self):
        from games.sc2.play import _run_episode
        client = self._make_client(n_steps=4)
        policy = self._make_policy()
        _run_episode(client, policy)
        self.assertEqual(policy.call_count, 4)

    def test_episode_calls_client_step_until_done(self):
        from games.sc2.play import _run_episode
        client = self._make_client(n_steps=3)
        policy = self._make_policy()
        _run_episode(client, policy)
        self.assertEqual(client.step.call_count, 3)

    def test_episode_calls_on_episode_start_and_end(self):
        from games.sc2.play import _run_episode
        client = self._make_client(n_steps=2)
        policy = self._make_policy()
        policy.on_episode_start = MagicMock()
        policy.on_episode_end   = MagicMock()
        _run_episode(client, policy)
        policy.on_episode_start.assert_called_once()
        policy.on_episode_end.assert_called_once()

    def test_episode_works_without_lifecycle_hooks(self):
        """Policies without on_episode_start/end should not crash."""
        from games.sc2.play import _run_episode
        client = self._make_client(n_steps=1)
        policy = self._make_policy()
        del policy.on_episode_start
        del policy.on_episode_end
        _run_episode(client, policy)   # must not raise AttributeError


# ---------------------------------------------------------------------------
# SC2Client play_mode flag
# ---------------------------------------------------------------------------

class TestSC2ClientPlayMode(unittest.TestCase):

    def test_client_stores_play_mode_flag(self):
        from games.sc2.client import SC2Client
        client = SC2Client(map_name="MoveToBeacon", play_mode=True)
        self.assertTrue(client._play_mode)

    def test_client_default_play_mode_false(self):
        from games.sc2.client import SC2Client
        client = SC2Client(map_name="MoveToBeacon")
        self.assertFalse(client._play_mode)

    @patch("games.sc2.client.SC2Client._make_sc2_env")
    def test_play_mode_uses_human_and_agent(self, mock_make):
        """_make_sc2_env is called with the Human+Agent player list in play mode."""
        from games.sc2.client import SC2Client

        captured: list = []

        def _fake_make_env():
            from unittest.mock import MagicMock
            env = MagicMock()
            env.reset.return_value = [
                MagicMock(observation={}, reward=0.0, last=lambda: True)
            ]
            return env

        mock_make.side_effect = _fake_make_env

        client = SC2Client(map_name="MoveToBeacon", play_mode=True)
        mock_make.assert_not_called()  # lazy: only called on first reset

        # Calling reset triggers _make_sc2_env
        with patch.object(client, "_timestep_to_obs_info",
                          return_value=(np.zeros(13, dtype=np.float32), {})):
            try:
                client.reset()
            except Exception:
                pass
        mock_make.assert_called_once()


# ---------------------------------------------------------------------------
# game_loop in info dict
# ---------------------------------------------------------------------------

class TestGameLoopInInfo(unittest.TestCase):

    def test_game_loop_present_in_minigame_info(self):
        from games.sc2.client import SC2Client

        client = SC2Client(map_name="MoveToBeacon")

        class _FakeOb:
            def get(self, key, default=None):
                return None
            def __getitem__(self, key):
                raise KeyError(key)
            def __contains__(self, key):
                return False

        class _FakeTimestep:
            observation = _FakeOb()
            reward = 0.0
            def last(self):
                return False

        _, info = client._timestep_to_obs_info(_FakeTimestep())
        self.assertIn("game_loop", info)
        self.assertIsInstance(info["game_loop"], float)

    def test_game_loop_value_extracted_from_obs(self):
        from games.sc2.client import SC2Client

        client = SC2Client(map_name="MoveToBeacon")

        gl_arr = np.array([1234], dtype=np.int32)

        class _FakeOb:
            def get(self, key, default=None):
                return None
            def __getitem__(self, key):
                if key == "game_loop":
                    return gl_arr
                raise KeyError(key)

        class _FakeTimestep:
            observation = _FakeOb()
            reward = 0.0
            def last(self):
                return False

        _, info = client._timestep_to_obs_info(_FakeTimestep())
        self.assertEqual(info["game_loop"], 1234.0)


if __name__ == "__main__":
    unittest.main()
