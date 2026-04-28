"""Tests for GeneticPolicy in tmnf/policies.py."""
import unittest

import numpy as np

from helpers import make_wlp
from games.tmnf.policies import GeneticPolicy, WeightedLinearPolicy


def _make_genetic(pop=6, elite=2) -> GeneticPolicy:
    return GeneticPolicy(population_size=pop, elite_k=elite, mutation_scale=0.1)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
