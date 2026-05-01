"""Tests for the TORCS reward calculator."""
import tempfile
import os
import unittest

from games.torcs.reward import TorcsRewardConfig, TorcsRewardCalculator


def _write_yaml(content: str) -> str:
    """Write content to a temp YAML file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(content)
    f.close()
    return f.name


class TestTorcsRewardConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = TorcsRewardConfig()
        self.assertEqual(cfg.finish_bonus, 100.0)
        self.assertEqual(cfg.progress_weight, 10.0)
        self.assertLess(cfg.step_penalty, 0.0)
        self.assertEqual(cfg.crash_threshold_m, 8.0)

    def test_custom_values(self):
        cfg = TorcsRewardConfig(finish_bonus=50.0, progress_weight=5.0)
        self.assertEqual(cfg.finish_bonus, 50.0)
        self.assertEqual(cfg.progress_weight, 5.0)

    def test_from_yaml(self):
        path = _write_yaml("progress_weight: 20.0\nfinish_bonus: 200.0\n")
        try:
            cfg = TorcsRewardConfig.from_yaml(path)
            self.assertEqual(cfg.progress_weight, 20.0)
            self.assertEqual(cfg.finish_bonus, 200.0)
            # Other fields keep defaults
            self.assertEqual(cfg.speed_weight, 0.05)
        finally:
            os.unlink(path)

    def test_from_yaml_unknown_key_raises(self):
        path = _write_yaml("unknown_key: 1.0\n")
        try:
            with self.assertRaises(ValueError) as ctx:
                TorcsRewardConfig.from_yaml(path)
            self.assertIn("unknown_key", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_from_yaml_loads_torcs_config(self):
        """Ensure the bundled TORCS reward config loads without error."""
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "games", "torcs", "config", "reward_config.yaml"
        )
        cfg = TorcsRewardConfig.from_yaml(cfg_path)
        self.assertIsInstance(cfg.progress_weight, float)


class TestTorcsRewardCalculator(unittest.TestCase):

    def _make_calc(self, **kwargs):
        return TorcsRewardCalculator(TorcsRewardConfig(**kwargs))

    def test_progress_reward(self):
        calc = self._make_calc(progress_weight=100.0, speed_weight=0.0,
                               step_penalty=0.0, centerline_weight=0.0,
                               accel_bonus=0.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info={"prev_progress": 0.1, "track_progress": 0.2},
        )
        self.assertAlmostEqual(r, 10.0)  # 0.1 * 100

    def test_centerline_penalty(self):
        calc = self._make_calc(progress_weight=0.0, speed_weight=0.0,
                               step_penalty=0.0, centerline_weight=-1.0,
                               centerline_exp=2.0, accel_bonus=0.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info={"lateral_offset": 2.0, "prev_progress": 0.0, "track_progress": 0.0},
        )
        self.assertAlmostEqual(r, -4.0)  # -1.0 * 2^2

    def test_speed_reward(self):
        calc = self._make_calc(progress_weight=0.0, speed_weight=0.1,
                               step_penalty=0.0, centerline_weight=0.0,
                               accel_bonus=0.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info={"speed_ms": 50.0, "prev_progress": 0.0, "track_progress": 0.0},
        )
        self.assertAlmostEqual(r, 5.0)  # 0.1 * 50

    def test_finish_bonus(self):
        calc = self._make_calc(progress_weight=0.0, speed_weight=0.0,
                               step_penalty=0.0, centerline_weight=0.0,
                               accel_bonus=0.0, finish_bonus=500.0,
                               finish_time_weight=-1.0, par_time_s=60.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=True,
            elapsed_s=70.0,
            info={"prev_progress": 0.99, "track_progress": 1.0},
        )
        # finish_bonus + finish_time_weight * (70 - 60) + progress delta
        self.assertAlmostEqual(r, 500.0 + (-1.0 * 10.0))

    def test_accel_bonus(self):
        calc = self._make_calc(progress_weight=0.0, speed_weight=0.0,
                               step_penalty=0.0, centerline_weight=0.0,
                               accel_bonus=1.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info={"accelerating": True, "prev_progress": 0.0, "track_progress": 0.0},
        )
        self.assertAlmostEqual(r, 1.0)

    def test_step_penalty(self):
        calc = self._make_calc(progress_weight=0.0, speed_weight=0.0,
                               step_penalty=-0.5, centerline_weight=0.0,
                               accel_bonus=0.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info={"prev_progress": 0.0, "track_progress": 0.0},
        )
        self.assertAlmostEqual(r, -0.5)

    def test_n_ticks_scaling(self):
        calc = self._make_calc(progress_weight=0.0, speed_weight=1.0,
                               step_penalty=-0.1, centerline_weight=0.0,
                               accel_bonus=0.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info={"speed_ms": 10.0, "prev_progress": 0.0, "track_progress": 0.0},
            n_ticks=3,
        )
        # speed_weight * speed * n_ticks + step_penalty * n_ticks
        self.assertAlmostEqual(r, 1.0 * 10.0 * 3 + (-0.1 * 3))

    def test_lap_wraparound(self):
        """Progress going from ~1 back to ~0 should be a positive delta."""
        calc = self._make_calc(progress_weight=100.0, speed_weight=0.0,
                               step_penalty=0.0, centerline_weight=0.0,
                               accel_bonus=0.0)
        r = calc.compute(
            prev_state=None, curr_state=None, finished=False,
            elapsed_s=1.0,
            info={"prev_progress": 0.95, "track_progress": 0.05},
        )
        # delta = 0.05 - 0.95 = -0.9, but wraparound adds 1.0 → 0.1
        self.assertAlmostEqual(r, 10.0)  # 0.1 * 100


if __name__ == "__main__":
    unittest.main()
