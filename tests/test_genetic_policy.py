"""Tests for GeneticPolicy in tmnf/policies.py."""
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from helpers import make_wlp
from games.tmnf.policies import GeneticPolicy, WeightedLinearPolicy
from framework.training import _greedy_loop_genetic


def _make_genetic(pop=6, elite=2, eval_episodes=1) -> GeneticPolicy:
    return GeneticPolicy(population_size=pop, elite_k=elite, mutation_scale=0.1,
                         eval_episodes=eval_episodes)


class TestGeneticPolicy(unittest.TestCase):

    def test_initialize_random_population_size(self):
        gp = _make_genetic(pop=6)
        gp.initialize_random()
        self.assertEqual(len(gp._population), 6)

    def test_champion_set_after_initialize_random(self):
        gp = _make_genetic()
        gp.initialize_random()
        self.assertIsNotNone(gp._champion)

    def test_initialize_from_champion_seeds_population(self):
        gp = _make_genetic(pop=5)
        champion = make_wlp()
        gp.initialize_from_champion(champion)
        self.assertEqual(len(gp._population), 5)
        self.assertIs(gp._champion, champion)

    def test_evaluate_and_evolve_updates_champion_reward(self):
        gp = _make_genetic(pop=4, elite=2)
        gp.initialize_random()
        gp.evaluate_and_evolve([10.0, 20.0, 5.0, 1.0])
        self.assertAlmostEqual(gp._champion_reward, 20.0)

    def test_evaluate_and_evolve_returns_true_when_improved(self):
        gp = _make_genetic(pop=4)
        gp.initialize_random()
        improved = gp.evaluate_and_evolve([10.0, 20.0, 5.0, 1.0])
        self.assertTrue(improved)

    def test_evaluate_and_evolve_returns_false_when_no_improvement(self):
        gp = _make_genetic(pop=4)
        gp.initialize_random()
        gp.evaluate_and_evolve([100.0, 90.0, 80.0, 70.0])   # sets champion_reward = 100
        improved = gp.evaluate_and_evolve([50.0, 40.0, 30.0, 20.0])
        self.assertFalse(improved)

    def test_crossover_draws_from_both_parents(self):
        names = WeightedLinearPolicy.OBS_NAMES
        cfg1 = {
            "steer_threshold": 0.5, "throttle_threshold": 0.5,
            "steer_weights":    {n:  1.0 for n in names},
            "throttle_weights": {n:  0.0 for n in names},
        }
        cfg2 = {
            "steer_threshold": 0.5, "throttle_threshold": 0.5,
            "steer_weights":    {n: -1.0 for n in names},
            "throttle_weights": {n:  0.0 for n in names},
        }
        child_cfg = GeneticPolicy._crossover(cfg1, cfg2)
        sw = list(child_cfg["steer_weights"].values())
        self.assertIn( 1.0, sw)
        self.assertIn(-1.0, sw)

    def test_population_replaced_after_evolution(self):
        gp = _make_genetic(pop=4, elite=2)
        gp.initialize_random()
        original_ids = [id(ind) for ind in gp._population]
        gp.evaluate_and_evolve([1.0, 2.0, 3.0, 4.0])
        new_ids = [id(ind) for ind in gp._population]
        # At least some individuals should be new objects
        self.assertFalse(original_ids == new_ids)

    def test_eval_episodes_default_is_one(self):
        gp = _make_genetic()
        self.assertEqual(gp._eval_episodes, 1)

    def test_eval_episodes_stored(self):
        gp = _make_genetic(eval_episodes=3)
        self.assertEqual(gp._eval_episodes, 3)

    def test_eval_episodes_in_to_cfg(self):
        gp = _make_genetic(eval_episodes=4)
        gp.initialize_random()
        cfg = gp.to_cfg()
        self.assertIn("eval_episodes", cfg)
        self.assertEqual(cfg["eval_episodes"], 4)

    def test_eval_episodes_from_cfg_roundtrip(self):
        gp = GeneticPolicy.from_cfg({"eval_episodes": 5})
        self.assertEqual(gp._eval_episodes, 5)

    def test_eval_episodes_from_cfg_default(self):
        gp = GeneticPolicy.from_cfg({})
        self.assertEqual(gp._eval_episodes, 1)


# ---------------------------------------------------------------------------
# Minimal stub env for training loop tests
# ---------------------------------------------------------------------------

class _SequentialRewardEnv:
    """Env that returns rewards from a preset list on successive episodes."""

    def __init__(self, rewards: list[float]) -> None:
        self._rewards = rewards
        self._idx = 0

    def reset(self):
        from games.tmnf.obs_spec import BASE_OBS_DIM
        return np.zeros(BASE_OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        from games.tmnf.obs_spec import BASE_OBS_DIM
        info = {"track_progress": 0.5, "laps_completed": 0, "pos_x": 0.0, "pos_z": 0.0}
        reward = self._rewards[self._idx % len(self._rewards)]
        self._idx += 1
        return np.zeros(BASE_OBS_DIM, dtype=np.float32), reward, True, False, info

    def get_episode_time_limit(self):
        return None

    def set_episode_time_limit(self, _):
        pass

    def close(self):
        pass


class TestGeneticEvalEpisodes(unittest.TestCase):
    """Tests for multi-episode fitness averaging in _greedy_loop_genetic."""

    def _run_one_gen(self, pop_size, eval_episodes, rewards_per_episode):
        """Run one generation and capture the rewards passed to evaluate_and_evolve."""
        gp = _make_genetic(pop=pop_size, elite=1, eval_episodes=eval_episodes)
        gp.initialize_random()

        env = _SequentialRewardEnv(rewards_per_episode)
        captured = []
        original_fn = gp.evaluate_and_evolve

        def _capture(rewards):
            captured.append(list(rewards))
            return original_fn(rewards)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            wf = f.name
        try:
            with patch.object(gp, "evaluate_and_evolve", side_effect=_capture):
                _greedy_loop_genetic(env=env, policy=gp, n_generations=1,
                                     weights_file=wf)
        finally:
            if os.path.exists(wf):
                os.unlink(wf)

        return captured

    def test_eval_episodes_1_passes_single_reward(self):
        """eval_episodes=1 passes single episode rewards unchanged."""
        # 4 individuals, 1 episode each → rewards = [10, 20, 30, 40]
        captured = self._run_one_gen(
            pop_size=4, eval_episodes=1,
            rewards_per_episode=[10.0, 20.0, 30.0, 40.0],
        )
        self.assertEqual(len(captured), 1)
        self.assertEqual(len(captured[0]), 4)
        np.testing.assert_allclose(captured[0], [10.0, 20.0, 30.0, 40.0])

    def test_eval_episodes_3_averages_rewards(self):
        """eval_episodes=3 passes the mean of 3 episode rewards per individual."""
        # 2 individuals, 3 episodes each
        # Rewards in order: ind0ep0=10, ind0ep1=20, ind0ep2=30,
        #                   ind1ep0=40, ind1ep1=50, ind1ep2=60
        # Mean ind0 = 20.0, mean ind1 = 50.0
        captured = self._run_one_gen(
            pop_size=2, eval_episodes=3,
            rewards_per_episode=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )
        self.assertEqual(len(captured), 1)
        self.assertEqual(len(captured[0]), 2)
        np.testing.assert_allclose(captured[0], [20.0, 50.0])

    def test_eval_episodes_2_correct_env_reset_calls(self):
        """eval_episodes=2 resets env pop_size * 2 times per generation."""
        from games.tmnf.obs_spec import BASE_OBS_DIM
        pop_size = 3
        eval_episodes = 2
        gp = _make_genetic(pop=pop_size, elite=1, eval_episodes=eval_episodes)
        gp.initialize_random()

        reset_count = [0]
        rewards_iter = iter([float(i) for i in range(100)])

        class _CountingEnv:
            def reset(self):
                reset_count[0] += 1
                return np.zeros(BASE_OBS_DIM, dtype=np.float32), {}

            def step(self, action):
                info = {"track_progress": 0.5, "laps_completed": 0,
                        "pos_x": 0.0, "pos_z": 0.0}
                return np.zeros(BASE_OBS_DIM, dtype=np.float32), next(rewards_iter), True, False, info

            def get_episode_time_limit(self):
                return None

            def set_episode_time_limit(self, _):
                pass

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            wf = f.name
        try:
            _greedy_loop_genetic(env=_CountingEnv(), policy=gp, n_generations=1,
                                 weights_file=wf)
        finally:
            if os.path.exists(wf):
                os.unlink(wf)

        self.assertEqual(reset_count[0], pop_size * eval_episodes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
