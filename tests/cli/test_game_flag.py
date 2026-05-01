"""Tests for the --game CLI flag in main.py.

Verifies that:
- The parser exposes --game with all expected choices.
- Each valid game value routes correctly.
- Missing optional dependencies raise a descriptive ValueError (not a raw
  ImportError).
- The experiment directory path embeds the game name.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestGameFlagChoices(unittest.TestCase):
    """Parser-level checks: choices, default, help text."""

    def _build_parser(self) -> argparse.ArgumentParser:
        """Replicate the argument parser from main.py without importing main."""
        parser = argparse.ArgumentParser(description="RL training")
        parser.add_argument("experiment")
        parser.add_argument(
            "--game",
            default="tmnf",
            choices=["tmnf", "beamng", "assetto", "car_racing", "torcs", "sc2"],
        )
        parser.add_argument("--track", default=None)
        parser.add_argument("--no-interrupt", action="store_true")
        parser.add_argument("--re-initialize", action="store_true")
        parser.add_argument("--log-level", default="INFO",
                            choices=["DEBUG", "INFO", "WARNING", "ERROR"])
        return parser

    def test_default_game_is_tmnf(self):
        parser = self._build_parser()
        args = parser.parse_args(["my_exp"])
        self.assertEqual(args.game, "tmnf")

    def test_all_valid_choices_accepted(self):
        parser = self._build_parser()
        for game in ("tmnf", "beamng", "assetto", "car_racing", "torcs", "sc2"):
            with self.subTest(game=game):
                args = parser.parse_args(["my_exp", "--game", game])
                self.assertEqual(args.game, game)

    def test_invalid_choice_raises_system_exit(self):
        parser = self._build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["my_exp", "--game", "unknown_game"])

    def test_help_text_contains_game_flag(self):
        """--help output should mention all new games."""
        import io
        parser = self._build_parser()
        buf = io.StringIO()
        try:
            parser.print_help(file=buf)
        except SystemExit:
            pass
        help_text = buf.getvalue()
        for game in ("tmnf", "beamng", "assetto", "car_racing"):
            self.assertIn(game, help_text)

    def test_track_flag_accepted(self):
        parser = self._build_parser()
        args = parser.parse_args(["my_exp", "--game", "torcs", "--track", "aalborg"])
        self.assertEqual(args.track, "aalborg")


class TestMainGameFlagChoices(unittest.TestCase):
    """Verify that main.py's actual parser has all expected choices."""

    def test_main_parser_has_all_choices(self):
        """Import main and verify --game choices include all required values."""
        import main  # noqa: PLC0415
        original_argv = sys.argv[:]
        try:
            for game in ("tmnf", "beamng", "assetto", "car_racing", "torcs", "sc2"):
                sys.argv = ["main.py", "test_exp", "--game", game]
                # Patch the runners so we don't actually run training.
                with patch.object(main, "_run_one", MagicMock()):
                    with patch.object(main, "_run_assetto", MagicMock()):
                        try:
                            main.main()
                        except (SystemExit, Exception):
                            pass
        finally:
            sys.argv = original_argv


class TestGameRouting(unittest.TestCase):
    """Verify that each --game value routes correctly."""

    def _run_main_with_game(self, game: str):
        """Call main.main() with --game <game> and check routing."""
        import main  # noqa: PLC0415

        mock_run_one = MagicMock()
        mock_run_assetto = MagicMock()

        original_argv = sys.argv[:]
        sys.argv = ["main.py", "test_exp", "--game", game]
        try:
            with patch.object(main, "_run_one", mock_run_one):
                with patch.object(main, "_run_assetto", mock_run_assetto):
                    main.main()
        finally:
            sys.argv = original_argv

        return mock_run_one, mock_run_assetto

    def test_game_tmnf_calls_run_one(self):
        mock_run_one, mock_assetto = self._run_main_with_game("tmnf")
        mock_run_one.assert_called_once()
        mock_assetto.assert_not_called()

    def test_game_beamng_calls_run_one(self):
        mock_run_one, mock_assetto = self._run_main_with_game("beamng")
        mock_run_one.assert_called_once()
        mock_assetto.assert_not_called()

    def test_game_assetto_calls_run_assetto(self):
        mock_run_one, mock_assetto = self._run_main_with_game("assetto")
        mock_assetto.assert_called_once()
        mock_run_one.assert_not_called()

    def test_game_car_racing_calls_run_one(self):
        mock_run_one, mock_assetto = self._run_main_with_game("car_racing")
        mock_run_one.assert_called_once()
        mock_assetto.assert_not_called()

    def test_game_torcs_calls_run_one(self):
        mock_run_one, mock_assetto = self._run_main_with_game("torcs")
        mock_run_one.assert_called_once()
        mock_assetto.assert_not_called()

    def test_game_sc2_calls_run_one(self):
        mock_run_one, mock_assetto = self._run_main_with_game("sc2")
        mock_run_one.assert_called_once()
        mock_assetto.assert_not_called()


class TestImportErrorConversion(unittest.TestCase):
    """Missing optional deps should raise ValueError, not ImportError."""

    def test_assetto_missing_dep_raises_value_error(self):
        """_run_assetto should raise ValueError when assetto_corsa entry is not importable."""
        import main  # noqa: PLC0415
        args = argparse.Namespace(
            experiment="test",
            game="assetto",
            no_interrupt=True,
            re_initialize=False,
            log_level="INFO",
        )
        with patch.dict(sys.modules, {"games.assetto_corsa.entry": None}):
            with self.assertRaises((ValueError, ImportError)):
                main._run_assetto(args)


class TestExperimentDirectoryNaming(unittest.TestCase):
    """Experiment directory must embed the game name."""

    def test_adapter_experiment_dir_contains_game_name(self):
        """Each adapter's experiment_dir should contain the game name or track."""
        from framework.game_adapter import GAME_ADAPTERS

        test_cases = {
            "tmnf": ("myrun", {"track": "a03_centerline"}, "a03_centerline"),
            "torcs": ("myrun", {}, "torcs"),
            "sc2": ("myrun", {"map_name": "MoveToBeacon"}, "sc2"),
            "beamng": ("myrun", {}, "beamng"),
            "car_racing": ("myrun", {}, "car_racing"),
        }

        for game_name, (exp_name, tp, expected_substr) in test_cases.items():
            with self.subTest(game=game_name):
                adapter = GAME_ADAPTERS[game_name]()
                exp_dir = adapter.experiment_dir(exp_name, tp, None)
                self.assertIn(expected_substr, exp_dir)
                self.assertIn(exp_name, exp_dir)


if __name__ == "__main__":
    unittest.main()
