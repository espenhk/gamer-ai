"""Tests for RewardCalculator and RewardConfig in tmnf/rl/reward.py."""
import tempfile
import os
import unittest

import numpy as np

from helpers import make_state_data
from games.tmnf.curiosity import ICM, RND, make_curiosity
from rl.reward import RewardCalculator, RewardConfig


def _write_yaml(content: str) -> str:
    """Write content to a temp YAML file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(content)
    f.close()
    return f.name


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

    def test_from_yaml_unknown_key_raises(self):
        path = _write_yaml("ceterline_weight: -1.0\n")
        try:
            with self.assertRaises(ValueError) as ctx:
                RewardConfig.from_yaml(path)
            self.assertIn("ceterline_weight", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_from_yaml_valid_keys_no_exception(self):
        path = _write_yaml("progress_weight: 5.0\nfinish_bonus: 200.0\n")
        try:
            cfg = RewardConfig.from_yaml(path)
            self.assertEqual(cfg.progress_weight, 5.0)
            self.assertEqual(cfg.finish_bonus, 200.0)
        finally:
            os.unlink(path)

    def test_from_yaml_partial_keys_uses_defaults(self):
        path = _write_yaml("finish_bonus: 42.0\n")
        try:
            cfg = RewardConfig.from_yaml(path)
            self.assertEqual(cfg.finish_bonus, 42.0)
            self.assertEqual(cfg.progress_weight, RewardConfig().progress_weight)
        finally:
            os.unlink(path)


class TestRewardCalculator(unittest.TestCase):

    def setUp(self):
        self.cfg  = RewardConfig()
        self.calc = RewardCalculator(self.cfg)

    def _r(self, prev, curr, finished=False, elapsed_s=0.0, accelerating=False,
           n_ticks=1):
        return self.calc.compute(prev, curr, finished, elapsed_s,
                                 info={"accelerating": accelerating},
                                 n_ticks=n_ticks)

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


class TestNTicksScaling(unittest.TestCase):
    """Per-tick reward components must scale linearly with n_ticks."""

    def setUp(self):
        self.cfg  = RewardConfig()
        self.calc = RewardCalculator(self.cfg)

    def _r(self, prev, curr, n_ticks=1, finished=False, elapsed_s=0.0,
           accelerating=False):
        return self.calc.compute(prev, curr, finished, elapsed_s,
                                 info={"accelerating": accelerating},
                                 n_ticks=n_ticks)

    def test_centerline_scales_with_n_ticks(self):
        # lateral_offset=2, progress=0, speed=0 → only centerline + step_penalty vary
        prev = make_state_data(track_progress=0.5, lateral_offset=0.0, speed=(0.0, 0.0, 0.0))
        curr = make_state_data(track_progress=0.5, lateral_offset=2.0, speed=(0.0, 0.0, 0.0))
        r1 = self._r(prev, curr, n_ticks=1)
        r3 = self._r(prev, curr, n_ticks=3)
        # centerline contribution: cfg.centerline_weight * 2^2 = -2.0 per tick
        # step_penalty also scales, so r3 - r1 == 2 * (centerline + step_penalty)
        expected_diff = 2.0 * (self.cfg.centerline_weight * 2.0 ** self.cfg.centerline_exp
                               + self.cfg.step_penalty)
        self.assertAlmostEqual(r3 - r1, expected_diff, places=5)

    def test_speed_scales_with_n_ticks(self):
        # speed=(5,0,0) → velocity.magnitude()=5; no lateral offset, no progress change
        state = make_state_data(track_progress=0.5, lateral_offset=0.0, speed=(5.0, 0.0, 0.0))
        r1 = self._r(state, state, n_ticks=1)
        r3 = self._r(state, state, n_ticks=3)
        # (speed + step_penalty) scales; diff = 2 * (speed_weight*5 + step_penalty)
        expected_diff = 2.0 * (self.cfg.speed_weight * 5.0 + self.cfg.step_penalty)
        self.assertAlmostEqual(r3 - r1, expected_diff, places=5)

    def test_airborne_penalty_scales_with_n_ticks(self):
        prev = make_state_data(track_progress=0.5, vertical_offset=-1.0,
                               speed=(0.0, 0.0, 0.0),
                               wheel_contacts=(True, False, False, False))
        curr = make_state_data(track_progress=0.5, vertical_offset=-1.0,
                               speed=(0.0, 0.0, 0.0),
                               wheel_contacts=(True, False, False, False))
        r1 = self._r(prev, curr, n_ticks=1)
        r3 = self._r(prev, curr, n_ticks=3)
        # airborne + step_penalty scale; diff = 2 * (airborne_penalty + step_penalty)
        expected_diff = 2.0 * (self.cfg.airborne_penalty + self.cfg.step_penalty)
        self.assertAlmostEqual(r3 - r1, expected_diff, places=5)

    def test_accel_bonus_scales_with_n_ticks(self):
        state = make_state_data(track_progress=0.5, lateral_offset=0.0, speed=(0.0, 0.0, 0.0))
        r1 = self._r(state, state, n_ticks=1, accelerating=True)
        r3 = self._r(state, state, n_ticks=3, accelerating=True)
        expected_diff = 2.0 * (self.cfg.accel_bonus + self.cfg.step_penalty)
        self.assertAlmostEqual(r3 - r1, expected_diff, places=5)

    def test_finish_bonus_does_not_scale_with_n_ticks(self):
        prev = make_state_data(track_progress=0.9, speed=(0.0, 0.0, 0.0))
        curr = make_state_data(track_progress=1.0, speed=(0.0, 0.0, 0.0))
        r1 = self._r(prev, curr, n_ticks=1, finished=True, elapsed_s=self.cfg.par_time_s)
        r3 = self._r(prev, curr, n_ticks=3, finished=True, elapsed_s=self.cfg.par_time_s)
        # finish_bonus and progress are one-time in this setup; with zero speed,
        # no lateral offset, and no acceleration, only the per-tick step penalty scales.
        per_tick_diff = r3 - r1
        expected_diff = 2.0 * self.cfg.step_penalty
        self.assertAlmostEqual(per_tick_diff, expected_diff, places=5)

    def test_progress_reward_does_not_scale_with_n_ticks(self):
        # progress delta is inherently multi-tick; doubling n_ticks should not double it
        prev = make_state_data(track_progress=0.0, speed=(0.0, 0.0, 0.0))
        curr = make_state_data(track_progress=0.1, speed=(0.0, 0.0, 0.0))
        r1 = self._r(prev, curr, n_ticks=1)
        r3 = self._r(prev, curr, n_ticks=3)
        # progress contribution is identical in both; only per-tick terms scale
        per_tick_diff = r3 - r1
        expected_diff = 2.0 * self.cfg.step_penalty
        self.assertAlmostEqual(per_tick_diff, expected_diff, places=5)


class TestRewardConfigMultiTrack(unittest.TestCase):
    """RewardConfig fields added for multi-track support (issue #14)."""

    def test_default_track_name(self):
        cfg = RewardConfig()
        self.assertEqual(cfg.track_name, "a03")

    def test_default_centerline_path(self):
        cfg = RewardConfig()
        self.assertEqual(cfg.centerline_path, "tracks/a03_centerline.npy")

    def test_custom_track_fields(self):
        cfg = RewardConfig(track_name="b05", centerline_path="tracks/b05_centerline.npy")
        self.assertEqual(cfg.track_name, "b05")
        self.assertEqual(cfg.centerline_path, "tracks/b05_centerline.npy")

    def test_from_yaml_reads_track_fields(self):
        import tempfile, os
        yaml_content = (
            "progress_weight: 10.0\n"
            "track_name: b05\n"
            "centerline_path: tracks/b05_centerline.npy\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            tmp = f.name
        try:
            cfg = RewardConfig.from_yaml(tmp)
            self.assertEqual(cfg.track_name, "b05")
            self.assertEqual(cfg.centerline_path, "tracks/b05_centerline.npy")
        finally:
            os.unlink(tmp)

    def test_from_yaml_backward_compat_missing_fields(self):
        """Old YAML files without track_name/centerline_path fall back to defaults."""
        import tempfile, os
        yaml_content = "progress_weight: 10.0\nfinish_bonus: 100.0\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            tmp = f.name
        try:
            cfg = RewardConfig.from_yaml(tmp)
            self.assertEqual(cfg.track_name, "a03")
            self.assertEqual(cfg.centerline_path, "tracks/a03_centerline.npy")
        finally:
            os.unlink(tmp)


class TestRewardCuriosityIntegration(unittest.TestCase):
    """RewardCalculator + curiosity: defaults stay backward compatible."""

    def _states(self):
        prev = make_state_data(track_progress=0.5, lateral_offset=0.0,
                               speed=(0.0, 0.0, 0.0))
        curr = make_state_data(track_progress=0.5, lateral_offset=0.0,
                               speed=(0.0, 0.0, 0.0))
        return prev, curr

    def test_default_config_is_backward_compatible(self):
        cfg = RewardConfig()
        self.assertEqual(cfg.curiosity_type, "none")
        self.assertEqual(cfg.curiosity_weight, 0.0)
        # No curiosity attached -> bit-for-bit unchanged behaviour.
        calc_plain = RewardCalculator(cfg)
        prev, curr = self._states()
        r_plain = calc_plain.compute(prev, curr, finished=False, elapsed_s=0.0,
                                     info={"accelerating": True})
        # Even if a module is somehow attached, weight=0 still skips it.
        icm = ICM(obs_dim=4, action_dim=3, seed=0)
        calc_off = RewardCalculator(cfg, curiosity=icm)
        r_off = calc_off.compute(prev, curr, finished=False, elapsed_s=0.0,
                                 info={"accelerating": True,
                                       "prev_obs": np.zeros(4, dtype=np.float32),
                                       "curr_obs": np.ones(4, dtype=np.float32),
                                       "action":   np.array([0.5, 1.0, 0.0],
                                                            dtype=np.float32)})
        self.assertAlmostEqual(r_plain, r_off, places=6)

    def test_icm_adds_positive_intrinsic_reward(self):
        cfg = RewardConfig(curiosity_type="icm", curiosity_weight=10.0)
        icm = make_curiosity("icm", obs_dim=4, action_dim=3,
                             feature_dim=4, hidden_size=8, seed=1)
        calc = RewardCalculator(cfg, curiosity=icm)
        prev, curr = self._states()

        rng = np.random.default_rng(9)
        prev_obs = rng.standard_normal(4).astype(np.float32)
        curr_obs = rng.standard_normal(4).astype(np.float32)
        action   = rng.standard_normal(3).astype(np.float32)
        info = {"accelerating": False,
                "prev_obs": prev_obs, "curr_obs": curr_obs, "action": action}

        r_with = calc.compute(prev, curr, finished=False, elapsed_s=0.0, info=info)

        cfg_off = RewardConfig()
        r_without = RewardCalculator(cfg_off).compute(
            prev, curr, finished=False, elapsed_s=0.0,
            info={"accelerating": False},
        )
        self.assertGreater(r_with, r_without)

    def test_intrinsic_reward_scales_with_n_ticks(self):
        # Intrinsic bonus should scale linearly with n_ticks, like the other
        # per-tick reward components, so the intrinsic-vs-extrinsic ratio is
        # invariant to skip-event frequency.
        cfg = RewardConfig(curiosity_type="rnd", curiosity_weight=10.0,
                           accel_bonus=0.0, step_penalty=0.0)
        rnd = make_curiosity("rnd", obs_dim=4, action_dim=3,
                             feature_dim=4, hidden_size=8, seed=11)
        calc = RewardCalculator(cfg, curiosity=rnd)
        prev, curr = self._states()
        rng = np.random.default_rng(13)
        prev_obs = rng.standard_normal(4).astype(np.float32)
        curr_obs = rng.standard_normal(4).astype(np.float32)
        action   = rng.standard_normal(3).astype(np.float32)
        # Two separate calculators with identical seeds so the underlying
        # intrinsic value matches; only n_ticks differs.
        rnd1 = make_curiosity("rnd", obs_dim=4, action_dim=3,
                              feature_dim=4, hidden_size=8, seed=11)
        calc1 = RewardCalculator(cfg, curiosity=rnd1)
        info = {"accelerating": False,
                "prev_obs": prev_obs, "curr_obs": curr_obs, "action": action}
        r1 = calc1.compute(prev, curr, finished=False, elapsed_s=0.0,
                           info=info, n_ticks=1)

        rnd3 = make_curiosity("rnd", obs_dim=4, action_dim=3,
                              feature_dim=4, hidden_size=8, seed=11)
        calc3 = RewardCalculator(cfg, curiosity=rnd3)
        r3 = calc3.compute(prev, curr, finished=False, elapsed_s=0.0,
                           info=info, n_ticks=3)
        self.assertAlmostEqual(r3, 3.0 * r1, places=4)

    def test_curiosity_skipped_when_obs_missing(self):
        # If the env forgets to supply obs/action, curiosity is silently skipped
        # rather than crashing — extrinsic reward is unaffected.
        cfg = RewardConfig(curiosity_type="icm", curiosity_weight=10.0)
        icm = make_curiosity("icm", obs_dim=4, action_dim=3, seed=2)
        calc = RewardCalculator(cfg, curiosity=icm)
        prev, curr = self._states()
        r = calc.compute(prev, curr, finished=False, elapsed_s=0.0,
                         info={"accelerating": False})
        # With zero progress / speed and no curiosity inputs, only step penalty.
        self.assertAlmostEqual(r, cfg.step_penalty, places=4)

    def test_reset_propagates_to_curiosity(self):
        cfg = RewardConfig(curiosity_type="rnd", curiosity_weight=1.0)
        rnd = make_curiosity("rnd", obs_dim=4, action_dim=3, seed=3)
        calls = {"n": 0}
        original = rnd.reset_episode

        def _spy():
            calls["n"] += 1
            original()

        rnd.reset_episode = _spy  # type: ignore[assignment]
        calc = RewardCalculator(cfg, curiosity=rnd)
        calc.reset()
        calc.reset()
        self.assertEqual(calls["n"], 2)

    def test_from_yaml_accepts_new_curiosity_keys(self):
        yaml_content = (
            "curiosity_type: icm\n"
            "curiosity_weight: 0.05\n"
            "curiosity_feature_dim: 16\n"
            "curiosity_hidden_size: 64\n"
            "curiosity_lr: 0.005\n"
            "curiosity_beta: 0.1\n"
            "curiosity_seed: 42\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            tmp = f.name
        try:
            cfg = RewardConfig.from_yaml(tmp)
            self.assertEqual(cfg.curiosity_type, "icm")
            self.assertAlmostEqual(cfg.curiosity_weight, 0.05)
            self.assertEqual(cfg.curiosity_feature_dim, 16)
            self.assertEqual(cfg.curiosity_hidden_size, 64)
            self.assertAlmostEqual(cfg.curiosity_lr, 0.005)
            self.assertAlmostEqual(cfg.curiosity_beta, 0.1)
            self.assertEqual(cfg.curiosity_seed, 42)
        finally:
            os.unlink(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
