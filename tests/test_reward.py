"""Tests for RewardCalculator and RewardConfig in tmnf/rl/reward.py."""
import unittest

from helpers import make_state_data
from rl.reward import RewardCalculator, RewardConfig


class TestRewardConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = RewardConfig()
        self.assertEqual(cfg.finish_bonus, 100.0)
        self.assertEqual(cfg.progress_weight, 10.0)
        self.assertLess(cfg.step_penalty, 0.0)

    def test_custom_values(self):
        cfg = RewardConfig(finish_bonus=50.0, progress_weight=5.0)
        self.assertEqual(cfg.finish_bonus, 50.0)
        self.assertEqual(cfg.progress_weight, 5.0)


class TestRewardCalculator(unittest.TestCase):

    def setUp(self):
        self.cfg  = RewardConfig()
        self.calc = RewardCalculator(self.cfg)

    def _r(self, prev, curr, finished=False, elapsed_s=0.0, accelerating=False):
        return self.calc.compute(prev, curr, finished, elapsed_s, accelerating)

    # --- Progress ---

    def test_progress_reward(self):
        prev = make_state_data(track_progress=0.0)
        curr = make_state_data(track_progress=0.1, speed=(0.0, 0.0, 0.0))
        reward = self._r(prev, curr)
        # progress contribution: 0.1 * 10.0 = 1.0  (plus tiny step_penalty)
        self.assertAlmostEqual(reward, 1.0 + self.cfg.step_penalty, places=4)

    def test_no_progress_no_progress_reward(self):
        state = make_state_data(track_progress=0.5, speed=(0.0, 0.0, 0.0), lateral_offset=0.0)
        reward = self._r(state, state)
        self.assertAlmostEqual(reward, self.cfg.step_penalty, places=4)

    # --- Centerline ---

    def test_centerline_penalty_quadratic(self):
        prev = make_state_data(track_progress=0.5, lateral_offset=0.0)
        curr = make_state_data(track_progress=0.5, lateral_offset=2.0, speed=(0.0, 0.0, 0.0))
        reward = self._r(prev, curr)
        # -0.5 * 2^2 = -2.0
        self.assertAlmostEqual(reward, -2.0 + self.cfg.step_penalty, places=4)

    def test_centerline_on_center_no_penalty(self):
        state = make_state_data(track_progress=0.5, lateral_offset=0.0, speed=(0.0, 0.0, 0.0))
        reward = self._r(state, state)
        self.assertAlmostEqual(reward, self.cfg.step_penalty, places=4)

    # --- Finish ---

    def test_finish_bonus_present(self):
        prev = make_state_data(track_progress=0.9)
        curr = make_state_data(track_progress=1.0, speed=(0.0, 0.0, 0.0))
        reward = self._r(prev, curr, finished=True, elapsed_s=self.cfg.par_time_s)
        self.assertGreater(reward, self.cfg.finish_bonus * 0.9)

    def test_finish_time_penalty_over_par(self):
        prev = make_state_data(track_progress=1.0)
        curr = make_state_data(track_progress=1.0, speed=(0.0, 0.0, 0.0))
        on_par   = self._r(prev, curr, finished=True, elapsed_s=60.0)
        over_par = self._r(prev, curr, finished=True, elapsed_s=70.0)
        # 10 s over par → extra -0.1 * 10 = -1.0
        self.assertAlmostEqual(on_par - over_par, 1.0, places=4)

    def test_finish_bonus_not_given_when_not_finished(self):
        state = make_state_data(track_progress=1.0, speed=(0.0, 0.0, 0.0))
        reward = self._r(state, state, finished=False)
        self.assertLess(reward, self.cfg.finish_bonus)

    # --- Acceleration ---

    def test_accel_bonus(self):
        state = make_state_data(track_progress=0.5, speed=(0.0, 0.0, 0.0))
        r_accel = self._r(state, state, accelerating=True)
        r_coast  = self._r(state, state, accelerating=False)
        self.assertAlmostEqual(r_accel - r_coast, self.cfg.accel_bonus, places=4)

    # --- Step penalty ---

    def test_step_penalty_always_applied(self):
        state = make_state_data(track_progress=0.5, speed=(0.0, 0.0, 0.0))
        reward = self._r(state, state)
        self.assertLessEqual(reward, 0.0)

    # --- Airborne penalty ---

    def test_airborne_penalty_when_off_ground(self):
        # ≤1 wheel contact AND vertical_offset ≤ 0 → penalty applied
        prev = make_state_data(track_progress=0.5, vertical_offset=-1.0,
                               wheel_contacts=(True, False, False, False))
        curr = make_state_data(track_progress=0.5, vertical_offset=-1.0,
                               speed=(0.0, 0.0, 0.0),
                               wheel_contacts=(True, False, False, False))
        reward_air  = self._r(prev, curr)
        reward_land = self._r(
            prev,
            make_state_data(track_progress=0.5, vertical_offset=-1.0,
                            speed=(0.0, 0.0, 0.0),
                            wheel_contacts=(True, True, True, True)),
        )
        self.assertLess(reward_air, reward_land)

    def test_airborne_penalty_not_applied_above_centreline(self):
        # vertical_offset > 0 → legitimate jump → no airborne penalty
        state = make_state_data(track_progress=0.5, vertical_offset=1.0,
                                speed=(0.0, 0.0, 0.0),
                                wheel_contacts=(False, False, False, False))
        reward = self._r(state, state)
        # Should NOT have airborne penalty — only step penalty (≈ -0.01)
        self.assertGreater(reward, self.cfg.airborne_penalty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
