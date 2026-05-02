"""Smoke tests for the Simple64 1v1 ladder training loop.

Verifies end-to-end integration of SC2Env, the reward calculator, and all
supported policy types against the 21-dim ladder observation space.  PySC2 is
not required — the SC2Client is mocked throughout.
"""
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from games.sc2.env import SC2Env
from games.sc2.obs_spec import LADDER_OBS_DIM, SC2_LADDER_OBS_SPEC
from games.sc2.reward import SC2RewardConfig
from games.sc2.policies import (
    SC2LinearPolicy,
    SC2GeneticPolicy,
    NeuralDQNPolicy,
    CMAESPolicy,
    REINFORCEPolicy,
    LSTMPolicy,
    LSTMEvolutionPolicy,
)
from framework.policies import (
    EpsilonGreedyPolicy,
    MCTSPolicy,
)
from framework.training import (
    _greedy_loop_genetic,
    _greedy_loop_cmaes,
    _greedy_loop_q_learning,
)
from games.sc2.actions import DISCRETE_ACTIONS

_OBS_SPEC = SC2_LADDER_OBS_SPEC
_HEAD_NAMES = ["fn_idx", "x", "y", "queue"]
_OBS_DIM = LADDER_OBS_DIM


def _make_mock_env(done_after: int = 5) -> SC2Env:
    """Return an SC2Env(Simple64) with a mocked SC2Client.

    The mocked client returns LADDER_OBS_DIM-dim observations.  After
    *done_after* steps it signals a win (player_outcome=1.0, done=True).
    """
    patcher = patch("games.sc2.env.SC2Client")
    mock_cls = patcher.start()
    env = SC2Env(
        map_name="Simple64",
        reward_config=SC2RewardConfig(
            score_weight=0.0,
            win_bonus=100.0,
            loss_penalty=-100.0,
            step_penalty=-0.001,
            economy_weight=0.001,
        ),
        max_episode_time_s=60.0,
    )
    patcher.stop()

    mock_client = mock_cls.return_value

    def _ladder_obs():
        return np.zeros(_OBS_DIM, dtype=np.float32)

    def _reset():
        _reset.step_count = 0
        return (
            _ladder_obs(),
            {"score": 0.0, "minerals": 50.0, "vespene": 0.0,
             "food_used": 6.0, "food_cap": 15.0, "army_count": 0.0},
        )
    _reset.step_count = 0

    def _step(action):
        _reset.step_count += 1
        is_done = _reset.step_count >= done_after
        info = {
            "score": float(_reset.step_count),
            "minerals": 60.0,
            "vespene": 0.0,
            "food_used": 6.0,
            "food_cap": 15.0,
            "army_count": 1.0,
            "player_outcome": 1.0 if is_done else None,
            "is_last": is_done,
        }
        return _ladder_obs(), 1.0 if is_done else 0.0, is_done, info

    mock_client.reset.side_effect = _reset
    mock_client.step.side_effect  = _step
    mock_client.close.return_value = None
    env._client = mock_client
    return env


# ---------------------------------------------------------------------------
# SC2Env ladder smoke tests
# ---------------------------------------------------------------------------

class TestSC2Simple64EnvIntegration(unittest.TestCase):
    """Verify SC2Env step/reset works end-to-end on Simple64."""

    def setUp(self):
        self.env = _make_mock_env(done_after=3)

    def tearDown(self):
        self.env.close()

    def test_observation_space_is_21_dim(self):
        self.assertEqual(self.env.observation_space.shape, (_OBS_DIM,))

    def test_reset_returns_correct_obs_shape(self):
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, (_OBS_DIM,))

    def test_step_returns_five_tuple(self):
        self.env.reset()
        action = DISCRETE_ACTIONS[0]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertEqual(obs.shape, (_OBS_DIM,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)

    def test_win_terminates_with_win_bonus(self):
        """Episode terminates with win bonus on player_outcome=1."""
        cfg = SC2RewardConfig(score_weight=0.0, win_bonus=200.0,
                              loss_penalty=-200.0, step_penalty=0.0)
        with patch("games.sc2.env.SC2Client") as mock_cls:
            env = SC2Env(map_name="Simple64", reward_config=cfg)
        mock_client = mock_cls.return_value
        mock_client.reset.return_value = (
            np.zeros(_OBS_DIM, dtype=np.float32),
            {"score": 0.0, "minerals": 50.0, "vespene": 0.0,
             "food_used": 0.0, "food_cap": 0.0, "army_count": 0.0},
        )
        mock_client.step.return_value = (
            np.zeros(_OBS_DIM, dtype=np.float32),
            1.0, True,
            {"score": 0.0, "minerals": 50.0, "vespene": 0.0,
             "food_used": 0.0, "food_cap": 0.0, "army_count": 0.0,
             "player_outcome": 1.0, "is_last": True},
        )
        env._client = mock_client
        env.reset()
        _, reward, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertAlmostEqual(reward, 200.0)
        self.assertEqual(info["termination_reason"], "win")
        env.close()

    def test_loss_terminates_with_penalty(self):
        """Episode terminates with loss penalty on player_outcome=-1."""
        cfg = SC2RewardConfig(score_weight=0.0, win_bonus=200.0,
                              loss_penalty=-200.0, step_penalty=0.0)
        with patch("games.sc2.env.SC2Client") as mock_cls:
            env = SC2Env(map_name="Simple64", reward_config=cfg)
        mock_client = mock_cls.return_value
        mock_client.reset.return_value = (
            np.zeros(_OBS_DIM, dtype=np.float32),
            {"score": 0.0, "minerals": 50.0, "vespene": 0.0,
             "food_used": 0.0, "food_cap": 0.0, "army_count": 0.0},
        )
        mock_client.step.return_value = (
            np.zeros(_OBS_DIM, dtype=np.float32),
            -1.0, True,
            {"score": 0.0, "minerals": 50.0, "vespene": 0.0,
             "food_used": 0.0, "food_cap": 0.0, "army_count": 0.0,
             "player_outcome": -1.0, "is_last": True},
        )
        env._client = mock_client
        env.reset()
        _, reward, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertAlmostEqual(reward, -200.0)
        self.assertEqual(info["termination_reason"], "loss")
        env.close()

    def test_economy_reward_flows_through(self):
        """Economy weight produces non-zero reward on mineral delta."""
        cfg = SC2RewardConfig(score_weight=0.0, win_bonus=0.0, loss_penalty=0.0,
                              step_penalty=0.0, economy_weight=1.0)
        with patch("games.sc2.env.SC2Client") as mock_cls:
            env = SC2Env(map_name="Simple64", reward_config=cfg)
        mock_client = mock_cls.return_value
        mock_client.reset.return_value = (
            np.zeros(_OBS_DIM, dtype=np.float32),
            {"score": 0.0, "minerals": 50.0, "vespene": 0.0,
             "food_used": 0.0, "food_cap": 0.0, "army_count": 0.0},
        )
        # +25 minerals → economy reward = 1.0 * 25 = 25
        mock_client.step.return_value = (
            np.zeros(_OBS_DIM, dtype=np.float32),
            0.0, False,
            {"score": 0.0, "minerals": 75.0, "vespene": 0.0,
             "food_used": 0.0, "food_cap": 0.0, "army_count": 0.0,
             "player_outcome": None, "is_last": False},
        )
        env._client = mock_client
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        self.assertAlmostEqual(reward, 25.0)
        env.close()


# ---------------------------------------------------------------------------
# SC2LinearPolicy action encoding verification
# ---------------------------------------------------------------------------

class TestSC2LinearPolicyActionEncoding(unittest.TestCase):
    """Verify SC2LinearPolicy uses sigmoid-based output, not clip/binary."""

    def _obs(self) -> np.ndarray:
        return np.ones(_OBS_DIM, dtype=np.float32)

    def test_fn_idx_in_valid_range(self):
        """fn_idx must be in [0, N_FUNCS-1] = [0, 5], not clipped to [-1,1]."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            policy = SC2LinearPolicy(_OBS_SPEC, _HEAD_NAMES)
            # Set large positive weights so sigmoid output is near 1 → fn_idx ≈ 5
            for head in _HEAD_NAMES:
                policy._weights[head] = rng.standard_normal(_OBS_DIM).astype(np.float32) * 5.0
            action = policy(self._obs())
            self.assertGreaterEqual(float(action[0]), 0.0)
            self.assertLessEqual(float(action[0]), 5.0)

    def test_x_y_are_continuous_not_binary(self):
        """x and y must be in (0, 1), not forced to {0, 1} extremes."""
        # With moderate weights, sigmoid gives interior values
        policy = SC2LinearPolicy(_OBS_SPEC, _HEAD_NAMES)
        n_interior = 0
        for seed in range(30):
            rng = np.random.default_rng(seed)
            for head in _HEAD_NAMES:
                policy._weights[head] = rng.standard_normal(_OBS_DIM).astype(np.float32) * 0.5
            action = policy(self._obs())
            # x and y should not be stuck at exactly 0.0 or 1.0
            if 0.01 < float(action[1]) < 0.99 and 0.01 < float(action[2]) < 0.99:
                n_interior += 1
        self.assertGreater(n_interior, 10,
                           "SC2LinearPolicy x/y should produce interior values, not just 0 or 1")

    def test_queue_is_binary(self):
        """queue must be 0.0 or 1.0 (binary)."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            policy = SC2LinearPolicy(_OBS_SPEC, _HEAD_NAMES)
            for head in _HEAD_NAMES:
                policy._weights[head] = rng.standard_normal(_OBS_DIM).astype(np.float32)
            action = policy(self._obs())
            self.assertIn(float(action[3]), (0.0, 1.0))

    def test_action_shape(self):
        policy = SC2LinearPolicy(_OBS_SPEC, _HEAD_NAMES)
        self.assertEqual(policy(self._obs()).shape, (4,))

    def test_sc2_genetic_population_uses_sc2_linear(self):
        """SC2GeneticPolicy population members must be SC2LinearPolicy instances."""
        policy = SC2GeneticPolicy(
            obs_spec=_OBS_SPEC, head_names=_HEAD_NAMES,
            population_size=4, elite_k=2,
        )
        policy.initialize_random()
        for member in policy.population:
            self.assertIsInstance(member, SC2LinearPolicy)

    def test_cmaes_offspring_use_sc2_linear(self):
        """CMAESPolicy offspring must be SC2LinearPolicy instances."""
        policy = CMAESPolicy(obs_spec=_OBS_SPEC, head_names=_HEAD_NAMES, population_size=4)
        offspring = policy.sample_population()
        for ind in offspring:
            self.assertIsInstance(ind, SC2LinearPolicy)
            # Also verify the fn_idx range is valid
            action = ind(self._obs())
            self.assertGreaterEqual(float(action[0]), 0.0)
            self.assertLessEqual(float(action[0]), 5.0)


# ---------------------------------------------------------------------------
# SC2-specific advanced policy compatibility on 21-dim ladder obs
# ---------------------------------------------------------------------------

class TestSC2PoliciesOnLadderObs(unittest.TestCase):
    """Verify SC2-specific advanced policies work with the 21-dim ladder obs."""

    def _obs(self) -> np.ndarray:
        return np.ones(_OBS_DIM, dtype=np.float32)

    def test_sc2_linear_policy_action_shape(self):
        policy = SC2LinearPolicy(_OBS_SPEC, _HEAD_NAMES)
        action = policy(self._obs())
        self.assertEqual(action.shape, (4,))

    def test_sc2_genetic_policy_action_shape(self):
        policy = SC2GeneticPolicy(
            obs_spec=_OBS_SPEC, head_names=_HEAD_NAMES,
            population_size=4, elite_k=2,
        )
        policy.initialize_random()
        action = policy(self._obs())
        self.assertEqual(action.shape, (4,))

    def test_epsilon_greedy_policy(self):
        policy = EpsilonGreedyPolicy(
            obs_spec=_OBS_SPEC,
            discrete_actions=DISCRETE_ACTIONS,
            n_bins=2,
            epsilon=1.0,
        )
        action = policy(self._obs())
        self.assertEqual(action.shape, (4,))

    def test_mcts_policy(self):
        policy = MCTSPolicy(
            obs_spec=_OBS_SPEC,
            discrete_actions=DISCRETE_ACTIONS,
            n_bins=2,
        )
        action = policy(self._obs())
        self.assertEqual(action.shape, (4,))

    def test_neural_dqn_policy_action_shape(self):
        policy = NeuralDQNPolicy(obs_spec=_OBS_SPEC, hidden_sizes=[16])
        action = policy(self._obs())
        self.assertEqual(action.shape, (4,))

    def test_neural_dqn_policy_update_does_not_crash(self):
        policy = NeuralDQNPolicy(
            obs_spec=_OBS_SPEC, hidden_sizes=[16],
            min_replay_size=2, batch_size=2,
        )
        obs = self._obs()
        action = policy(obs)
        policy.update(obs, action, 1.0, self._obs(), False)
        policy.update(obs, action, 1.0, self._obs(), True)
        # gradient step fires once min_replay_size reached
        policy.update(obs, action, 1.0, self._obs(), False)

    def test_cmaes_policy_sample_shape(self):
        policy = CMAESPolicy(
            obs_spec=_OBS_SPEC,
            head_names=_HEAD_NAMES,
            population_size=4,
        )
        offspring = policy.sample_population()
        self.assertEqual(len(offspring), 4)
        self.assertEqual(offspring[0](self._obs()).shape, (4,))

    def test_cmaes_policy_update_distribution(self):
        policy = CMAESPolicy(
            obs_spec=_OBS_SPEC,
            head_names=_HEAD_NAMES,
            population_size=4,
        )
        policy.sample_population()
        improved = policy.update_distribution([10.0, 5.0, 8.0, 3.0])
        self.assertTrue(improved)
        self.assertAlmostEqual(policy.champion_reward, 10.0)

    def test_reinforce_policy_action_shape(self):
        policy = REINFORCEPolicy(obs_spec=_OBS_SPEC, hidden_sizes=[16])
        action = policy(self._obs())
        self.assertEqual(action.shape, (4,))

    def test_reinforce_policy_episode(self):
        policy = REINFORCEPolicy(
            obs_spec=_OBS_SPEC, hidden_sizes=[8],
            learning_rate=0.01,
        )
        obs = self._obs()
        action = policy(obs)
        policy.update(obs, action, 1.0, self._obs(), False)
        action2 = policy(obs)
        policy.update(obs, action2, 2.0, self._obs(), True)
        policy.on_episode_end()  # triggers gradient update

    def test_lstm_policy_action_shape(self):
        policy = LSTMPolicy(obs_spec=_OBS_SPEC, hidden_size=8)
        policy.on_episode_start()
        action = policy(self._obs())
        self.assertEqual(action.shape, (4,))

    def test_lstm_evolution_policy_sample_and_update(self):
        policy = LSTMEvolutionPolicy(
            obs_spec=_OBS_SPEC,
            hidden_size=8,
            population_size=4,
        )
        offspring = policy.sample_population()
        self.assertEqual(len(offspring), 4)
        improved = policy.update_distribution([10.0, 5.0, 8.0, 3.0])
        self.assertTrue(improved)
        action = policy(self._obs())
        self.assertEqual(action.shape, (4,))


# ---------------------------------------------------------------------------
# Training loop integration: 2 generations with mocked env
# ---------------------------------------------------------------------------

class TestSimple64TrainingLoopSmoke(unittest.TestCase):
    """Run 2 generations of each supported policy type against a mocked Simple64 env."""

    def _env(self) -> SC2Env:
        return _make_mock_env(done_after=3)

    def _tmpfile(self) -> str:
        f = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        f.close()
        return f.name

    def tearDown(self):
        # clean up any leftover weight files
        pass

    def test_sc2_genetic_training_loop(self):
        env = self._env()
        policy = SC2GeneticPolicy(
            obs_spec=_OBS_SPEC,
            head_names=_HEAD_NAMES,
            population_size=4,
            elite_k=2,
        )
        policy.initialize_random()
        wf = self._tmpfile()
        try:
            best_policy, best_reward, sims, _, _ = _greedy_loop_genetic(
                env=env, policy=policy, n_generations=2, weights_file=wf,
            )
            self.assertIsNotNone(best_policy)
            self.assertIsInstance(best_reward, float)
            self.assertEqual(len(sims), 2)
            # Champion must be an SC2LinearPolicy (correct action encoding)
            self.assertIsInstance(best_policy._champion, SC2LinearPolicy)
        finally:
            os.unlink(wf)
            env.close()

    def test_cmaes_training_loop(self):
        env = self._env()
        policy = CMAESPolicy(
            obs_spec=_OBS_SPEC,
            head_names=_HEAD_NAMES,
            population_size=4,
        )
        policy.initialize_random()
        wf = self._tmpfile()
        try:
            best_policy, best_reward, sims, _, _ = _greedy_loop_cmaes(
                env=env, policy=policy, n_generations=2, weights_file=wf,
            )
            self.assertIsNotNone(best_policy)
            self.assertIsInstance(best_reward, float)
            self.assertEqual(len(sims), 2)
        finally:
            os.unlink(wf)
            env.close()

    def test_neural_dqn_training_loop(self):
        env = self._env()
        policy = NeuralDQNPolicy(
            obs_spec=_OBS_SPEC,
            hidden_sizes=[16],
            min_replay_size=2,
            batch_size=2,
            epsilon_decay_steps=10,
        )
        wf = self._tmpfile()
        try:
            best_policy, best_reward, sims, _, _ = _greedy_loop_q_learning(
                env=env, policy=policy, n_episodes=2, weights_file=wf,
            )
            self.assertIsNotNone(best_policy)
            self.assertIsInstance(best_reward, float)
            self.assertEqual(len(sims), 2)
        finally:
            os.unlink(wf)
            env.close()

    def test_reinforce_training_loop(self):
        env = self._env()
        policy = REINFORCEPolicy(
            obs_spec=_OBS_SPEC,
            hidden_sizes=[16],
            learning_rate=0.01,
        )
        wf = self._tmpfile()
        try:
            best_policy, best_reward, sims, _, _ = _greedy_loop_q_learning(
                env=env, policy=policy, n_episodes=2, weights_file=wf,
            )
            self.assertIsNotNone(best_policy)
            self.assertIsInstance(best_reward, float)
            self.assertEqual(len(sims), 2)
        finally:
            os.unlink(wf)
            env.close()

    def test_lstm_training_loop(self):
        env = self._env()
        policy = LSTMEvolutionPolicy(
            obs_spec=_OBS_SPEC,
            hidden_size=8,
            population_size=4,
        )
        wf = self._tmpfile()
        try:
            best_policy, best_reward, sims, _, _ = _greedy_loop_cmaes(
                env=env, policy=policy, n_generations=2, weights_file=wf,
            )
            self.assertIsNotNone(best_policy)
            self.assertIsInstance(best_reward, float)
            self.assertEqual(len(sims), 2)
        finally:
            os.unlink(wf)
            env.close()


# ---------------------------------------------------------------------------
# SC2-specific policy serialisation round-trips
# ---------------------------------------------------------------------------

class TestSC2PolicySaveLoad(unittest.TestCase):
    """Verify that save_trainer_state / load_trainer_state round-trips correctly."""

    def test_cmaes_trainer_state_roundtrip(self):
        policy = CMAESPolicy(
            obs_spec=_OBS_SPEC,
            head_names=_HEAD_NAMES,
            population_size=4,
        )
        policy.sample_population()
        policy.update_distribution([5.0, 3.0, 4.0, 2.0])
        original_sigma = policy.sigma
        original_gen   = policy._gen

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            policy.save_trainer_state(path)
            policy2 = CMAESPolicy(
                obs_spec=_OBS_SPEC,
                head_names=_HEAD_NAMES,
                population_size=4,
            )
            policy2.load_trainer_state(path)
            self.assertAlmostEqual(policy2.sigma, original_sigma, places=6)
            self.assertEqual(policy2._gen, original_gen)
        finally:
            os.unlink(path)

    def test_neural_dqn_trainer_state_roundtrip(self):
        policy = NeuralDQNPolicy(
            obs_spec=_OBS_SPEC,
            hidden_sizes=[16],
            min_replay_size=2,
            batch_size=2,
        )
        obs = np.ones(_OBS_DIM, dtype=np.float32)
        action = policy(obs)
        policy.update(obs, action, 1.0, obs, False)
        policy.update(obs, action, 2.0, obs, True)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            policy.save_trainer_state(path)
            policy2 = NeuralDQNPolicy(
                obs_spec=_OBS_SPEC,
                hidden_sizes=[16],
                min_replay_size=2,
                batch_size=2,
            )
            policy2.load_trainer_state(path)
            self.assertEqual(policy2._total_steps, policy._total_steps)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
