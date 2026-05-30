"""Tests for the SC2 self-play mechanism.

Validates that SC2Client and SC2Env correctly support self-play mode,
where two AI agents play against each other instead of one agent vs a bot.
PySC2 is not required — all PySC2 interactions are mocked.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from games.sc2.client import SC2Client
from games.sc2.obs_spec import BASE_OBS_DIM, LADDER_OBS_DIM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_obs(dim: int = BASE_OBS_DIM) -> np.ndarray:
    return np.zeros(dim, dtype=np.float32)


def _make_fake_info() -> dict:
    return {
        "minerals": 50.0,
        "vespene": 0.0,
        "score": 0.0,
        "army_count": 0.0,
        "game_loop": 100.0,
        "available_fn_ids": [0, 1, 2],
    }


class _FakeTimeStep:
    """Minimal stand-in for PySC2 TimeStep."""

    def __init__(self, reward=0.0, last=False):
        self.observation = {
            "player": MagicMock(
                __getitem__=lambda s, k: 0.0,
                get=lambda k, d=None: d,
            ),
        }
        self.reward = reward
        self._last = last

    def last(self) -> bool:
        return self._last


# ---------------------------------------------------------------------------
# SC2Client self-play mode
# ---------------------------------------------------------------------------


class TestSC2ClientSelfPlayInit(unittest.TestCase):
    """SC2Client constructor accepts self_play and opponent_policy params."""

    def test_self_play_flag_stored(self):
        client = SC2Client(map_name="Simple64", self_play=True)
        self.assertTrue(client._self_play)

    def test_self_play_default_false(self):
        client = SC2Client(map_name="Simple64")
        self.assertFalse(client._self_play)

    def test_opponent_policy_stored(self):
        opp = MagicMock()
        client = SC2Client(map_name="Simple64", self_play=True, opponent_policy=opp)
        self.assertIs(client._opponent_policy, opp)

    def test_opponent_obs_initially_none(self):
        client = SC2Client(map_name="Simple64", self_play=True)
        self.assertIsNone(client._opponent_obs)


class TestSC2ClientSelfPlayMakeEnv(unittest.TestCase):
    """_make_sc2_env creates two Agent players in self-play mode."""

    def test_two_agent_players_in_self_play(self):
        """When self_play=True, _make_sc2_env creates two sc2_env.Agent slots
        (not a Bot)."""
        # Instead of mocking PySC2 internals, verify the player-list logic
        # by inspecting what SC2Env would receive.  We patch only the imports
        # and capture the players list via a side-effect.
        captured = {}

        class _FakeSC2Env:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        class _FakeAgent:
            def __init__(self, race, name=""):
                self.race = race
                self.name = name

        mock_sc2_env = MagicMock()
        mock_sc2_env.SC2Env = _FakeSC2Env
        mock_sc2_env.Agent = _FakeAgent
        mock_sc2_env.Race.random = "random"

        mock_features = MagicMock()
        mock_absl = MagicMock()
        mock_absl.FLAGS.is_parsed.return_value = True

        # Build a module dict where `from pysc2.env import sc2_env` resolves
        # to our mock_sc2_env.
        mock_pysc2_env_pkg = MagicMock()
        mock_pysc2_env_pkg.sc2_env = mock_sc2_env

        with patch.dict(
            "sys.modules",
            {
                "pysc2": MagicMock(),
                "pysc2.env": mock_pysc2_env_pkg,
                "pysc2.env.sc2_env": mock_sc2_env,
                "pysc2.lib": MagicMock(),
                "pysc2.lib.features": mock_features,
                "absl": mock_absl,
                "absl.flags": mock_absl,
            },
        ):
            with patch("games.sc2.map_access_gate.acquire_map_access_slot"):
                client = SC2Client(map_name="Simple64", self_play=True)
                client._make_sc2_env()

        players = captured.get("players", [])
        self.assertEqual(len(players), 2)
        self.assertIsInstance(players[0], _FakeAgent)
        self.assertIsInstance(players[1], _FakeAgent)


class TestSC2ClientSelfPlayReset(unittest.TestCase):
    """reset() caches opponent observation in self-play mode."""

    def _make_client(self):
        opp = MagicMock(return_value=np.array([0, 0.5, 0.5, 0], dtype=np.float32))
        client = SC2Client(map_name="Simple64", self_play=True, opponent_policy=opp)
        return client, opp

    def test_reset_caches_opponent_obs(self):
        client, opp = self._make_client()

        agent_obs = _make_fake_obs(LADDER_OBS_DIM)
        opp_obs = _make_fake_obs(LADDER_OBS_DIM) + 1.0

        # Mock internal env
        fake_ts_agent = _FakeTimeStep()
        fake_ts_opp = _FakeTimeStep()
        fake_env = MagicMock()
        fake_env.reset.return_value = [fake_ts_agent, fake_ts_opp]

        client._sc2_env = fake_env
        # In reset(), _timestep_to_obs_info is called for opponent (timesteps[1])
        # first, then for the primary agent (timesteps[0]).
        client._timestep_to_obs_info = MagicMock(
            side_effect=[(opp_obs, _make_fake_info()), (agent_obs, _make_fake_info())]
        )

        obs, info = client.reset()
        np.testing.assert_array_equal(obs, agent_obs)
        # Opponent obs should be cached.
        self.assertIsNotNone(client._opponent_obs)
        np.testing.assert_array_equal(client._opponent_obs, opp_obs)


class TestSC2ClientSelfPlayStep(unittest.TestCase):
    """step() uses opponent policy for the second agent in self-play."""

    def _make_client_with_env(self, done=False):
        opp = MagicMock(return_value=np.array([0, 0.5, 0.5, 0], dtype=np.float32))
        client = SC2Client(map_name="Simple64", self_play=True, opponent_policy=opp)

        agent_obs = _make_fake_obs(LADDER_OBS_DIM)
        opp_obs = _make_fake_obs(LADDER_OBS_DIM) + 1.0

        fake_ts_agent = _FakeTimeStep(last=done)
        fake_ts_opp = _FakeTimeStep(last=done)
        fake_env = MagicMock()
        fake_env.step.return_value = [fake_ts_agent, fake_ts_opp]

        client._sc2_env = fake_env
        client._opponent_obs = _make_fake_obs(LADDER_OBS_DIM) + 0.5
        # Mock _timestep_to_obs_info: first call = agent, second = opponent.
        client._timestep_to_obs_info = MagicMock(
            side_effect=[(agent_obs, _make_fake_info()), (opp_obs, _make_fake_info())]
        )
        # Mock _action_to_call to avoid pysc2 import.
        client._action_to_call = MagicMock(return_value=MagicMock())
        # Disable proactive selection (no pysc2 available).
        client._available_actions = None
        client._selected_count = 1.0

        return client, opp, fake_env

    def test_step_passes_two_actions(self):
        """In self-play, step() sends two fn_calls to the PySC2 env."""
        client, opp, fake_env = self._make_client_with_env()

        action = np.array([0, 0.5, 0.5, 0], dtype=np.float32)
        client.step(action)

        # PySC2's step should have received a list of 2 fn_calls.
        call_args = fake_env.step.call_args[0][0]
        self.assertEqual(len(call_args), 2)

    def test_opponent_policy_called_with_cached_obs(self):
        """Opponent policy is called with the cached opponent observation."""
        client, opp, fake_env = self._make_client_with_env()
        cached_opp_obs = client._opponent_obs.copy()

        action = np.array([0, 0.5, 0.5, 0], dtype=np.float32)
        client.step(action)

        opp.assert_called_once()
        np.testing.assert_array_equal(opp.call_args[0][0], cached_opp_obs)

    def test_opponent_obs_updated_after_step(self):
        """After step(), opponent_obs is updated from the new timestep."""
        client, opp, fake_env = self._make_client_with_env()
        old_opp_obs = client._opponent_obs.copy()

        action = np.array([0, 0.5, 0.5, 0], dtype=np.float32)
        client.step(action)

        # Opponent obs should have been updated to the new observation.
        self.assertIsNotNone(client._opponent_obs)
        # The new obs is from the second call to _timestep_to_obs_info (opp_obs + 1.0).
        self.assertFalse(np.array_equal(client._opponent_obs, old_opp_obs))


class TestSC2ClientNonSelfPlayUnchanged(unittest.TestCase):
    """When self_play=False, step() sends only one action (existing behaviour)."""

    def test_single_action_when_not_self_play(self):
        client = SC2Client(map_name="MoveToBeacon")
        fake_ts = _FakeTimeStep()
        fake_env = MagicMock()
        fake_env.step.return_value = [fake_ts]

        client._sc2_env = fake_env
        client._selected_count = 1.0
        client._action_to_call = MagicMock(return_value=MagicMock())
        client._timestep_to_obs_info = MagicMock(return_value=(_make_fake_obs(), _make_fake_info()))

        action = np.array([0, 0.5, 0.5, 0], dtype=np.float32)
        client.step(action)

        call_args = fake_env.step.call_args[0][0]
        self.assertEqual(len(call_args), 1)


# ---------------------------------------------------------------------------
# SC2Env self-play integration
# ---------------------------------------------------------------------------


class TestSC2EnvSelfPlay(unittest.TestCase):
    """SC2Env forwards self_play and opponent_policy to the client."""

    @patch("games.sc2.env.SC2Client")
    def test_client_receives_self_play_params(self, MockClient):
        """SC2Env passes self_play and opponent_policy to SC2Client."""
        from games.sc2.env import SC2Env

        opp = MagicMock()
        SC2Env(
            map_name="Simple64",
            self_play=True,
            opponent_policy=opp,
        )
        # Check the SC2Client constructor was called with self_play and opponent_policy.
        call_kwargs = MockClient.call_args[1]
        self.assertTrue(call_kwargs["self_play"])
        self.assertIs(call_kwargs["opponent_policy"], opp)

    @patch("games.sc2.env.SC2Client")
    def test_set_opponent_policy(self, MockClient):
        """set_opponent_policy updates the client's opponent."""
        from games.sc2.env import SC2Env

        env = SC2Env(map_name="Simple64", self_play=True)
        new_opp = MagicMock()
        env.set_opponent_policy(new_opp)
        self.assertIs(env._client._opponent_policy, new_opp)


# ---------------------------------------------------------------------------
# make_env factory
# ---------------------------------------------------------------------------


class TestMakeEnvSelfPlay(unittest.TestCase):
    """make_env passes self_play through to SC2Env."""

    @patch("games.sc2.env.SC2Client")
    def test_make_env_self_play_kwarg(self, MockClient):
        from games.sc2.env import make_env

        with tempfile.TemporaryDirectory() as tmpdir:
            make_env(
                experiment_dir=tmpdir,
                map_name="Simple64",
                self_play=True,
            )
        call_kwargs = MockClient.call_args[1]
        self.assertTrue(call_kwargs["self_play"])


# ---------------------------------------------------------------------------
# Adapter wiring
# ---------------------------------------------------------------------------


class TestSC2AdapterSelfPlay(unittest.TestCase):
    """SC2Adapter.build_game_spec passes self_play from training_params."""

    def test_self_play_in_env_factory_kwargs(self):
        from games.sc2.adapter import SC2Adapter

        adapter = SC2Adapter()
        training_params = {
            "map_name": "Simple64",
            "in_game_episode_s": 60.0,
            "policy_type": "sc2_genetic",
            "n_sims": 10,
            "self_play": True,
            "initial_extreme_random_fraction": 0,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = adapter.build_game_spec(
                experiment_name="test",
                experiment_dir=tmpdir,
                weights_file=os.path.join(tmpdir, "w.yaml"),
                reward_cfg_file="",
                training_params=training_params,
                track_override=None,
            )
        # The env factory should contain self_play=True in its kwargs.
        self.assertTrue(spec.make_env_fn._kwargs["self_play"])

    def test_self_play_default_false(self):
        from games.sc2.adapter import SC2Adapter

        adapter = SC2Adapter()
        training_params = {
            "map_name": "Simple64",
            "in_game_episode_s": 60.0,
            "policy_type": "sc2_genetic",
            "n_sims": 10,
            "initial_extreme_random_fraction": 0,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = adapter.build_game_spec(
                experiment_name="test",
                experiment_dir=tmpdir,
                weights_file=os.path.join(tmpdir, "w.yaml"),
                reward_cfg_file="",
                training_params=training_params,
                track_override=None,
            )
        self.assertFalse(spec.make_env_fn._kwargs["self_play"])


# ---------------------------------------------------------------------------
# train_rl self-play opponent wiring
# ---------------------------------------------------------------------------


class TestTrainRLSelfPlayWiring(unittest.TestCase):
    """train_rl sets the opponent policy on the env when self_play is enabled."""

    @patch("framework.training._greedy_loop_genetic")
    @patch("framework.training._maybe_build_evaluator", return_value=None)
    @patch("framework.training.make_live_monitor", return_value=None)
    def test_opponent_set_when_self_play_enabled(self, _mock_lm, _mock_eval, mock_loop):
        """When training_params has self_play=True and env has set_opponent_policy,
        train_rl should call set_opponent_policy with a deepcopy of the policy."""

        from framework.run_config import GameSpec, RunConfig
        from framework.training import train_rl
        from games.sc2.actions import DISCRETE_ACTIONS
        from games.sc2.obs_spec import get_spec

        obs_spec = get_spec("Simple64")

        # Mock env with set_opponent_policy
        mock_env = MagicMock()
        mock_env.observation_space = MagicMock()
        mock_env.action_space = MagicMock()

        training_params = {
            "policy_type": "sc2_genetic",
            "n_sims": 1,
            "speed": 1.0,
            "in_game_episode_s": 10.0,
            "mutation_scale": 0.05,
            "mutation_share": 1.0,
            "adaptive_mutation": False,
            "patience": 0,
            "log_stats_every_n_sims": 0,
            "self_play": True,
            "policy_params": {
                "population_size": 2,
                "elite_k": 1,
                "eval_episodes": 1,
                "mutation_scale": 0.1,
                "mutation_share": 0.3,
            },
        }

        # Mock the loop to return a result immediately
        from framework.training import GreedyLoopResult

        mock_loop.return_value = GreedyLoopResult(
            policy=MagicMock(),
            best_reward=0.0,
            greedy_sims=[],
            early_stopped=False,
            early_stop_sim=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_file = os.path.join(tmpdir, "w.yaml")
            game = GameSpec(
                experiment_name="test_sp",
                track="sc2_Simple64",
                make_env_fn=lambda: mock_env,
                obs_spec=obs_spec,
                head_names=["fn_idx", "x", "y", "queue"],
                discrete_actions=DISCRETE_ACTIONS,
                weights_file=weights_file,
                reward_config_file="",
                save_results_fn=lambda *a, **kw: None,
                game_name="sc2",
            )
            config = RunConfig.from_training_params(training_params)

            train_rl(game, config, no_interrupt=True)

        # set_opponent_policy should have been called once.
        mock_env.set_opponent_policy.assert_called_once()
        # The argument should be a policy object (deepcopy of the best_policy).
        opp_arg = mock_env.set_opponent_policy.call_args[0][0]
        self.assertIsNotNone(opp_arg)


# ---------------------------------------------------------------------------
# SelfPlayManager unit tests
# ---------------------------------------------------------------------------


class _CallablePolicy:
    """Minimal stand-in for a policy with champion_reward and mutated()."""

    def __init__(self, reward: float = 0.0):
        self.champion_reward = reward
        self._champion = self  # policy is its own champion for simplicity
        self.mutated_called = False

    def __call__(self, obs):
        return np.zeros(4, dtype=np.float32)

    def mutated(self, scale: float = 0.1, share: float = 1.0) -> "_CallablePolicy":
        self.mutated_called = True
        child = _CallablePolicy(self.champion_reward)
        return child


class TestSelfPlayManagerExactMode(unittest.TestCase):
    """SelfPlayManager in 'exact' mode always returns a snapshot of champion."""

    def test_invalid_mode_raises(self):
        from framework.self_play import SelfPlayManager

        with self.assertRaises(ValueError):
            SelfPlayManager(mode="unknown")

    def test_build_initial_opponent_returns_callable(self):
        from framework.self_play import SelfPlayManager

        manager = SelfPlayManager(mode="exact")
        policy = _CallablePolicy(reward=10.0)
        opp = manager.build_initial_opponent(policy)
        self.assertIsNotNone(opp)
        self.assertTrue(callable(opp))

    def test_step_always_returns_new_snapshot(self):
        from framework.self_play import SelfPlayManager

        manager = SelfPlayManager(mode="exact")
        policy = _CallablePolicy(reward=5.0)
        opp1 = manager.step(policy, improved=False)
        opp2 = manager.step(policy, improved=True)
        # Both should be callable; they are independent copies, not the same object.
        self.assertIsNotNone(opp1)
        self.assertIsNotNone(opp2)
        self.assertIsNot(opp1, opp2)

    def test_step_no_improvement_still_refreshes(self):
        from framework.self_play import SelfPlayManager

        manager = SelfPlayManager(mode="exact")
        policy = _CallablePolicy(reward=3.0)
        opp = manager.step(policy, improved=False)
        self.assertIsNotNone(opp)


class TestSelfPlayManagerMutatedMode(unittest.TestCase):
    """SelfPlayManager in 'mutated' mode calls champion.mutated()."""

    def test_mutated_mode_calls_mutated(self):
        from framework.self_play import SelfPlayManager

        manager = SelfPlayManager(mode="mutated", mutation_scale=0.1)
        policy = _CallablePolicy(reward=1.0)
        opp = manager.step(policy, improved=True)
        # _CallablePolicy.mutated() sets mutated_called on the champion.
        self.assertTrue(policy._champion.mutated_called)
        self.assertIsNotNone(opp)

    def test_mutated_fallback_when_no_mutated_method(self):
        from framework.self_play import SelfPlayManager

        manager = SelfPlayManager(mode="mutated", mutation_scale=0.1)

        class _NoMutated:
            champion_reward = 2.0

            def __call__(self, obs):
                return np.zeros(4)

        policy = _NoMutated()
        # Should not raise; falls back to deepcopy.
        opp = manager.step(policy, improved=True)
        self.assertIsNotNone(opp)


class TestSelfPlayManagerTopNMode(unittest.TestCase):
    """SelfPlayManager in 'top_n' mode maintains a pool of top champions."""

    def test_pool_grows_on_each_improvement(self):
        from framework.self_play import SelfPlayManager

        manager = SelfPlayManager(mode="top_n", top_n=3, seed=0)
        policy = _CallablePolicy(reward=1.0)
        manager.build_initial_opponent(policy)
        self.assertEqual(len(manager._pool), 1)

        policy.champion_reward = 2.0
        manager.step(policy, improved=True)
        self.assertEqual(len(manager._pool), 2)

        policy.champion_reward = 3.0
        manager.step(policy, improved=True)
        self.assertEqual(len(manager._pool), 3)

    def test_pool_capped_at_top_n(self):
        from framework.self_play import SelfPlayManager

        manager = SelfPlayManager(mode="top_n", top_n=2, seed=0)
        policy = _CallablePolicy(reward=1.0)
        manager.build_initial_opponent(policy)
        policy.champion_reward = 2.0
        manager.step(policy, improved=True)
        # Pool at capacity now.
        policy.champion_reward = 3.0
        manager.step(policy, improved=True)
        self.assertEqual(len(manager._pool), 2)

    def test_weakest_replaced_on_new_champion(self):
        from framework.self_play import SelfPlayManager

        manager = SelfPlayManager(mode="top_n", top_n=2, seed=0)
        policy = _CallablePolicy(reward=1.0)
        manager.build_initial_opponent(policy)  # pool: [(1.0, …)]
        policy.champion_reward = 2.0
        manager.step(policy, improved=True)  # pool: [(1.0, …), (2.0, …)]
        # Now a stronger champion arrives.
        policy.champion_reward = 5.0
        manager.step(policy, improved=True)
        # The weakest (score=1.0) should have been evicted.
        scores = [score for score, _ in manager._pool]
        self.assertNotIn(1.0, scores)
        self.assertIn(5.0, scores)

    def test_no_update_when_not_improved_and_pool_full(self):
        from framework.self_play import SelfPlayManager

        manager = SelfPlayManager(mode="top_n", top_n=2, seed=0)
        policy = _CallablePolicy(reward=1.0)
        manager.build_initial_opponent(policy)
        policy.champion_reward = 2.0
        manager.step(policy, improved=True)  # pool full
        # No improvement; pool should stay the same.
        initial_scores = [s for s, _ in manager._pool]
        policy.champion_reward = 2.0
        manager.step(policy, improved=False)
        final_scores = [s for s, _ in manager._pool]
        self.assertEqual(initial_scores, final_scores)

    def test_step_returns_callable_from_pool(self):
        from framework.self_play import SelfPlayManager

        manager = SelfPlayManager(mode="top_n", top_n=3, seed=42)
        policy = _CallablePolicy(reward=1.0)
        manager.build_initial_opponent(policy)
        opp = manager.step(policy, improved=False)
        self.assertIsNotNone(opp)
        self.assertTrue(callable(opp))

    def test_step_returns_fresh_copy_from_pool(self):
        from framework.self_play import SelfPlayManager

        manager = SelfPlayManager(mode="top_n", top_n=3, seed=0)
        policy = _CallablePolicy(reward=1.0)
        manager.build_initial_opponent(policy)
        opp = manager.step(policy, improved=False)
        self.assertIsNot(opp, manager._pool[0][1])

    def test_weak_champion_does_not_replace_stronger_pool_entry(self):
        from framework.self_play import SelfPlayManager

        manager = SelfPlayManager(mode="top_n", top_n=2, seed=0)
        policy = _CallablePolicy(reward=10.0)
        manager.build_initial_opponent(policy)
        policy.champion_reward = 9.0
        manager.step(policy, improved=True)  # pool: [(10, …), (9, …)]
        # A weaker "improved" champion (e.g. score=1.0) should NOT evict
        # any existing entry since it is not stronger than any pool member.
        policy.champion_reward = 1.0
        manager.step(policy, improved=True)
        scores = [s for s, _ in manager._pool]
        self.assertNotIn(1.0, scores)


class TestGreedyLoopCmaesWithSelfPlay(unittest.TestCase):
    """_greedy_loop_cmaes calls self_play_manager.step() each generation."""

    def _make_mock_cmaes_policy(self, champion_reward=0.0):
        """Return a minimal CMA-ES-like policy stub."""
        from unittest.mock import MagicMock

        import numpy as np

        policy = MagicMock()
        policy.champion_reward = champion_reward
        policy.sigma = 0.1
        # sample_population returns two individuals
        ind = MagicMock()
        ind.__call__ = MagicMock(return_value=np.zeros(4, dtype=np.float32))
        policy.sample_population.return_value = [ind, ind]
        # update_distribution returns True (improved) on first call
        policy.update_distribution.side_effect = [True, False, False]
        return policy

    def test_self_play_manager_step_called_each_gen(self):
        """Manager.step() is called once per generation and env.set_opponent_policy updated."""
        import os
        import tempfile
        from unittest.mock import MagicMock

        import numpy as np

        from framework.self_play import SelfPlayManager
        from framework.training import _greedy_loop_cmaes

        policy = self._make_mock_cmaes_policy(champion_reward=1.0)

        # Minimal env stub
        obs = np.zeros(15, dtype=np.float32)
        mock_env = MagicMock()
        mock_env.get_episode_time_limit.return_value = None
        mock_env.reset.return_value = (obs, {})
        mock_env.step.return_value = (obs, 1.0, True, False, {"episode_reward_components": {}})

        manager = MagicMock(spec=SelfPlayManager)
        new_opp = MagicMock()
        manager.step.return_value = new_opp

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_file = os.path.join(tmpdir, "w.yaml")
            _greedy_loop_cmaes(
                env=mock_env,
                policy=policy,
                n_generations=3,
                weights_file=weights_file,
                self_play_manager=manager,
            )

        # step() called once per generation
        self.assertEqual(manager.step.call_count, 3)
        # env.set_opponent_policy updated each time with the returned opponent
        self.assertEqual(mock_env.set_opponent_policy.call_count, 3)
        for c in mock_env.set_opponent_policy.call_args_list:
            self.assertIs(c[0][0], new_opp)

    def test_no_self_play_manager_no_set_opponent(self):
        """Without a manager, set_opponent_policy is never called."""
        import os
        import tempfile
        from unittest.mock import MagicMock

        import numpy as np

        from framework.training import _greedy_loop_cmaes

        policy = self._make_mock_cmaes_policy()
        obs = np.zeros(15, dtype=np.float32)
        mock_env = MagicMock()
        mock_env.get_episode_time_limit.return_value = None
        mock_env.reset.return_value = (obs, {})
        mock_env.step.return_value = (obs, 0.5, True, False, {})

        with tempfile.TemporaryDirectory() as tmpdir:
            _greedy_loop_cmaes(
                env=mock_env,
                policy=policy,
                n_generations=2,
                weights_file=os.path.join(tmpdir, "w.yaml"),
            )

        mock_env.set_opponent_policy.assert_not_called()


class TestTrainRLSelfPlayModes(unittest.TestCase):
    """train_rl passes SelfPlayManager to the genetic loop when self_play is set."""

    @patch("framework.training._greedy_loop_genetic")
    @patch("framework.training._maybe_build_evaluator", return_value=None)
    @patch("framework.training.make_live_monitor", return_value=None)
    def _run_with_mode(self, mode, _mock_lm, _mock_eval, mock_loop):
        from framework.run_config import GameSpec, RunConfig
        from framework.training import GreedyLoopResult, train_rl
        from games.sc2.actions import DISCRETE_ACTIONS
        from games.sc2.obs_spec import get_spec

        obs_spec = get_spec("Simple64")
        mock_env = MagicMock()
        mock_loop.return_value = GreedyLoopResult(
            policy=MagicMock(),
            best_reward=0.0,
            greedy_sims=[],
            early_stopped=False,
            early_stop_sim=None,
        )
        training_params = {
            "policy_type": "sc2_genetic",
            "n_sims": 1,
            "speed": 1.0,
            "in_game_episode_s": 10.0,
            "mutation_scale": 0.05,
            "mutation_share": 1.0,
            "adaptive_mutation": False,
            "patience": 0,
            "log_stats_every_n_sims": 0,
            "self_play": True,
            "self_play_mode": mode,
            "policy_params": {
                "population_size": 2,
                "elite_k": 1,
                "eval_episodes": 1,
                "mutation_scale": 0.1,
                "mutation_share": 0.3,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            game = GameSpec(
                experiment_name="test_mode",
                track="sc2_Simple64",
                make_env_fn=lambda: mock_env,
                obs_spec=obs_spec,
                head_names=["fn_idx", "x", "y", "queue"],
                discrete_actions=DISCRETE_ACTIONS,
                weights_file=os.path.join(tmpdir, "w.yaml"),
                reward_config_file="",
                save_results_fn=lambda *a, **kw: None,
                game_name="sc2",
            )
            config = RunConfig.from_training_params(training_params)
            train_rl(game, config, no_interrupt=True)

        # set_opponent_policy called for initial opponent.
        mock_env.set_opponent_policy.assert_called()
        # The genetic loop received a self_play_manager kwarg.
        call_kwargs = mock_loop.call_args[1]
        self.assertIn("self_play_manager", call_kwargs)
        manager = call_kwargs["self_play_manager"]
        self.assertIsNotNone(manager)
        self.assertEqual(manager.mode, mode)

    def test_exact_mode_wired(self):
        self._run_with_mode("exact")

    def test_mutated_mode_wired(self):
        self._run_with_mode("mutated")

    def test_top_n_mode_wired(self):
        self._run_with_mode("top_n")


if __name__ == "__main__":
    unittest.main()
