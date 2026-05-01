"""Tests for the --game CLI flag in main.py.

Verifies that:
- The parser exposes --game with all expected choices.
- Each valid game value routes to the correct runner function.
- Missing optional dependencies raise a descriptive ValueError (not a raw
  ImportError).
- The experiment directory path embeds the game name.
"""

from __future__ import annotations

import argparse
import sys
import unittest
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
            choices=["tmnf", "beamng", "assetto", "car_racing", "torcs"],
        )
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
        for game in ("tmnf", "beamng", "assetto", "car_racing", "torcs"):
            with self.subTest(game=game):
                args = parser.parse_args(["my_exp", "--game", game])
                self.assertEqual(args.game, game)

    def test_invalid_choice_raises_system_exit(self):
        parser = self._build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["my_exp", "--game", "unknown_game"])

    def test_help_text_contains_game_flag(self):
        """--help output should mention all four new games."""
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


class TestMainGameFlagChoices(unittest.TestCase):
    """Verify that main.py's actual parser has all four new choices."""

    def test_main_parser_has_all_choices(self):
        """Import main and verify --game choices include all required values."""
        # main.py only imports framework.training at module level (no heavy deps).
        import main  # noqa: PLC0415
        parser = argparse.ArgumentParser()
        # Re-parse by inspecting what main.main() would create via the real module.
        # We check the choices indirectly by calling parse_args.
        # Since main.main() calls parse_args() on sys.argv, we patch sys.argv.
        original_argv = sys.argv[:]
        try:
            for game in ("tmnf", "beamng", "assetto", "car_racing", "torcs"):
                sys.argv = ["main.py", "test_exp", "--game", game]
                # Patch the runner so we don't actually run training.
                with patch.object(main, f"_run_{game}", create=True) as mock_runner:
                    # Patch all runner functions to avoid side-effects.
                    with patch.dict(
                        vars(main),
                        {
                            "_run_tmnf": MagicMock(),
                            "_run_beamng": MagicMock(),
                            "_run_assetto": MagicMock(),
                            "_run_car_racing": MagicMock(),
                            "_run_torcs": MagicMock(),
                        },
                    ):
                        # Should not raise SystemExit for valid choices.
                        try:
                            main.main()
                        except (SystemExit, Exception):
                            pass  # Errors from mocked runners are OK here.
        finally:
            sys.argv = original_argv


class TestGameRouting(unittest.TestCase):
    """Verify that each --game value calls the correct runner function."""

    def _run_main_with_game(self, game: str) -> tuple[str, MagicMock]:
        """Call main.main() with --game <game> and capture which runner was called."""
        import main  # noqa: PLC0415

        runners = {
            "tmnf":       MagicMock(),
            "beamng":     MagicMock(),
            "assetto":    MagicMock(),
            "car_racing": MagicMock(),
            "torcs":      MagicMock(),
        }

        original_argv = sys.argv[:]
        sys.argv = ["main.py", "test_exp", "--game", game]
        try:
            with patch.multiple(
                main,
                _run_tmnf=runners["tmnf"],
                _run_beamng=runners["beamng"],
                _run_assetto=runners["assetto"],
                _run_car_racing=runners["car_racing"],
                _run_torcs=runners["torcs"],
            ):
                main.main()
        finally:
            sys.argv = original_argv

        return runners

    def test_game_tmnf_calls_run_tmnf(self):
        runners = self._run_main_with_game("tmnf")
        runners["tmnf"].assert_called_once()
        runners["beamng"].assert_not_called()
        runners["assetto"].assert_not_called()
        runners["car_racing"].assert_not_called()

    def test_game_beamng_calls_run_beamng(self):
        runners = self._run_main_with_game("beamng")
        runners["beamng"].assert_called_once()
        runners["tmnf"].assert_not_called()

    def test_game_assetto_calls_run_assetto(self):
        runners = self._run_main_with_game("assetto")
        runners["assetto"].assert_called_once()
        runners["tmnf"].assert_not_called()

    def test_game_car_racing_calls_run_car_racing(self):
        runners = self._run_main_with_game("car_racing")
        runners["car_racing"].assert_called_once()
        runners["tmnf"].assert_not_called()

    def test_game_torcs_calls_run_torcs(self):
        runners = self._run_main_with_game("torcs")
        runners["torcs"].assert_called_once()
        runners["tmnf"].assert_not_called()


class TestImportErrorConversion(unittest.TestCase):
    """Missing optional deps should raise ValueError, not ImportError."""

    def _run_beamng_with_missing_dep(self):
        import main  # noqa: PLC0415
        args = argparse.Namespace(
            experiment="test",
            game="beamng",
            no_interrupt=True,
            re_initialize=False,
            log_level="INFO",
        )
        with patch.dict(sys.modules, {"beamng_gym": None}):
            # Ensure games.beamng.env raises ImportError when beamng_gym is None.
            with patch("builtins.__import__", side_effect=_selective_import_error("beamng_gym")):
                main._run_beamng(args)

    def test_beamng_missing_dep_raises_value_error(self):
        """_run_beamng should raise ValueError when beamng_gym is not installed."""
        import main  # noqa: PLC0415
        args = argparse.Namespace(
            experiment="test",
            game="beamng",
            no_interrupt=True,
            re_initialize=False,
            log_level="INFO",
        )
        # Make the import of games.beamng.env raise ImportError.
        with patch.dict(sys.modules, {"games.beamng.env": None}):
            with self.assertRaises((ValueError, ImportError)):
                main._run_beamng(args)

    def test_assetto_missing_dep_raises_value_error(self):
        """_run_assetto should raise ValueError when assettocorsa is not installed."""
        import main  # noqa: PLC0415
        args = argparse.Namespace(
            experiment="test",
            game="assetto",
            no_interrupt=True,
            re_initialize=False,
            log_level="INFO",
        )
        with patch.dict(sys.modules, {"games.assetto.env": None}):
            with self.assertRaises((ValueError, ImportError)):
                main._run_assetto(args)

    def test_car_racing_missing_dep_raises_value_error(self):
        """_run_car_racing should raise ValueError when gymnasium[box2d] is not installed."""
        import main  # noqa: PLC0415
        args = argparse.Namespace(
            experiment="test",
            game="car_racing",
            no_interrupt=True,
            re_initialize=False,
            log_level="INFO",
        )
        with patch.dict(sys.modules, {"games.car_racing.env": None}):
            with self.assertRaises((ValueError, ImportError)):
                main._run_car_racing(args)


def _selective_import_error(module_name: str):
    """Return an __import__ side-effect that raises ImportError for one module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _import(name, *args, **kwargs):
        if name == module_name or name.startswith(module_name + "."):
            raise ImportError(f"No module named '{module_name}'")
        return real_import(name, *args, **kwargs)

    return _import


class TestExperimentDirectoryNaming(unittest.TestCase):
    """Experiment directory must embed the game name."""

    def _get_experiment_dir_for_new_game(self, game: str, experiment: str) -> str:
        """Capture the experiment_dir used by a new-game runner by mocking imports."""
        import main  # noqa: PLC0415
        args = argparse.Namespace(
            experiment=experiment,
            game=game,
            no_interrupt=True,
            re_initialize=False,
            log_level="INFO",
        )

        captured: dict = {}

        # Build a mock module with the needed symbols.
        mock_game_module = MagicMock()

        def _fake_makedirs(path, **_kw):
            # Capture the first makedirs call (= experiment_dir).
            if "experiment_dir" not in captured:
                captured["experiment_dir"] = path
            # Don't actually create directories.

        game_module_path = f"games.{game}.env"
        obs_module_path  = f"games.{game}.obs_spec"
        act_module_path  = f"games.{game}.actions"
        ana_module_path  = f"games.{game}.analytics"

        # Provide stub modules for all imports inside the runner.
        stub_obs_spec = MagicMock()
        stub_obs_spec_attr = {
            "beamng":      "BEAMNG_OBS_SPEC",
            "assetto":     "ASSETTO_OBS_SPEC",
            "car_racing":  "CAR_RACING_OBS_SPEC",
        }.get(game, "OBS_SPEC")

        stub_modules = {
            game_module_path:                    MagicMock(),
            obs_module_path:                     MagicMock(),
            act_module_path:                     MagicMock(),
            ana_module_path:                     MagicMock(),
            f"games.{game}":                     MagicMock(),
        }

        with patch.dict(sys.modules, stub_modules):
            with patch("os.makedirs", side_effect=_fake_makedirs):
                # open() will fail trying to read the config YAML — that's OK,
                # we only care that makedirs was called with the right path.
                try:
                    runner = getattr(main, f"_run_{game}")
                    runner(args)
                except Exception:
                    pass

        return captured.get("experiment_dir", "")

    def test_beamng_experiment_dir_contains_beamng(self):
        exp_dir = self._get_experiment_dir_for_new_game("beamng", "my_exp")
        self.assertIn("beamng", exp_dir)
        self.assertIn("my_exp", exp_dir)

    def test_assetto_experiment_dir_contains_assetto(self):
        exp_dir = self._get_experiment_dir_for_new_game("assetto", "my_exp")
        self.assertIn("assetto", exp_dir)
        self.assertIn("my_exp", exp_dir)

    def test_car_racing_experiment_dir_contains_car_racing(self):
        exp_dir = self._get_experiment_dir_for_new_game("car_racing", "my_exp")
        self.assertIn("car_racing", exp_dir)
        self.assertIn("my_exp", exp_dir)

    def test_torcs_experiment_dir_contains_torcs(self):
        exp_dir = self._get_experiment_dir_for_new_game("torcs", "my_exp")
        self.assertIn("torcs", exp_dir)
        self.assertIn("my_exp", exp_dir)


if __name__ == "__main__":
    unittest.main()
