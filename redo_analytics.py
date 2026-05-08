#!/usr/bin/env python
"""Re-generate analytics for one or more completed experiments.

Each experiment folder must contain ``results/experiment_data.json``.

Usage
-----
Regenerate plots and ``results.md`` for a single experiment::

    python redo_analytics.py experiments/a03_centerline/my_experiment

Regenerate individual results for several experiments and produce a
combined summary report (same format as a grid-search summary)::

    python redo_analytics.py experiments/a03/exp1 experiments/a03/exp2 \\
        --summary-name my_summary

Only write the combined summary without re-running individual plots::

    python redo_analytics.py experiments/a03/exp1 experiments/a03/exp2 \\
        --summary-name combined --no-individual

Always write a combined summary even for a single experiment::

    python redo_analytics.py experiments/a03/exp1 \\
        --summary-name single_summary
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from framework.analytics import ExperimentData


# ---------------------------------------------------------------------------
# Game detection
# ---------------------------------------------------------------------------

_SC2_KEYS = {"map_name", "agent_race", "step_mul", "screen_size"}
_GAME_ALIASES = {
    "assetto_corsa": "assetto",
    "assetto-corsa": "assetto",
}
_ANALYTICS_MODULES = {
    "assetto": "assetto_corsa",
}


def _normalize_game_name(game: str | None) -> str:
    if game is None:
        return ""
    game = str(game).strip().lower()
    game = _GAME_ALIASES.get(game, game)
    return game


def _detect_game(training_params: dict) -> str:
    """Infer game from training-param keys; defaults to 'tmnf'."""
    game = _normalize_game_name(training_params.get("game", ""))
    if game:
        return game
    if _SC2_KEYS & set(training_params):
        return "sc2"
    return "tmnf"


# ---------------------------------------------------------------------------
# Analytics module loader
# ---------------------------------------------------------------------------

def _load_analytics_fns(game: str):
    """Return ``(save_experiment_results, save_grid_summary)`` for *game*.

    Falls back to the framework-level ``save_grid_summary`` when the
    game-specific module does not export one.
    """
    save_exp = save_grid = None
    game = _normalize_game_name(game)
    module_game = _ANALYTICS_MODULES.get(game, game)
    try:
        mod = __import__(
            f"games.{module_game}.analytics",
            fromlist=["save_experiment_results", "save_grid_summary"],
        )
        save_exp = getattr(mod, "save_experiment_results", None)
        save_grid = getattr(mod, "save_grid_summary", None)
    except ImportError:
        pass

    if save_exp is None:
        from games.tmnf.analytics import save_experiment_results as _fn
        save_exp = _fn

    if save_grid is None:
        from framework.analytics import save_grid_summary as _fn
        save_grid = _fn

    return save_exp, save_grid

# ---------------------------------------------------------------------------
# Core function (importable for testing)
# ---------------------------------------------------------------------------

def redo_analytics(
    experiment_dirs: list[str],
    game: str | None = None,
    summary_name: str | None = None,
    summary_dir: str | None = None,
    no_individual: bool = False,
) -> None:
    """Re-generate analytics for one or more completed experiments.

    Parameters
    ----------
    experiment_dirs:
        Paths to experiment folders; each must contain
        ``results/experiment_data.json``.
    game:
        Game identifier (``'tmnf'``, ``'sc2'``, ``'torcs'`` …).  Auto-detected
        from the first successfully loaded experiment's ``training_params`` when
        *None*.
    summary_name:
        Base name for the combined summary.  When *None* and multiple
        experiments are given, defaults to ``'combined'``.  Pass any non-None
        value to force a summary even for a single experiment.
    summary_dir:
        Explicit output directory for the combined summary.  When *None*,
        inferred as ``<common-parent>/<summary_name>__summary/``.
    no_individual:
        Skip regenerating individual experiment results; only write the
        combined summary (requires *summary_name* or multiple experiments).
    """
    from framework.analytics import infer_varied_summary_keys, load_experiment_data

    # Load all experiments.
    loaded: list[tuple[str, "ExperimentData"]] = []  # (experiment_dir, ExperimentData)
    effective_game: str | None = game

    for d in experiment_dirs:
        try:
            data = load_experiment_data(d)
        except FileNotFoundError:
            logger.error(
                "No experiment_data.json found in %s/results/ — skipping.", d
            )
            continue

        if effective_game is None:
            effective_game = _detect_game(data.training_params)
            logger.info("Auto-detected game: %s", effective_game)

        # Normalize path fields: if the stored path doesn't exist (e.g. because
        # the experiment was moved or generated on another machine), fall back to
        # a same-named file inside the experiment directory itself.
        for attr, filename in (
            ("weights_file", "policy_weights.yaml"),
            ("reward_config_file", "reward_config.yaml"),
        ):
            stored = getattr(data, attr)
            if stored and not os.path.exists(stored):
                candidate = os.path.join(d, filename)
                if os.path.exists(candidate):
                    setattr(data, attr, candidate)
                    logger.debug("Remapped %s → %s", attr, candidate)

        loaded.append((d, data))
        best = max((s.reward for s in data.greedy_sims), default=float("-inf"))
        logger.info("  Loaded %-50s  best_reward=%+.1f", data.experiment_name, best)

    if not loaded:
        logger.error("No experiment data loaded — nothing to do.")
        return

    effective_game = effective_game or "tmnf"
    save_experiment_results, save_grid_summary = _load_analytics_fns(effective_game)

    # Regenerate individual results.
    if not no_individual:
        for d, data in loaded:
            results_dir = os.path.join(d, "results")
            logger.info("Regenerating %s → %s/", data.experiment_name, results_dir)
            save_experiment_results(data, results_dir)

    # Combined summary when requested or when multiple experiments are present.
    do_summary = len(loaded) > 1 or summary_name is not None
    if not do_summary:
        if no_individual:
            raise ValueError(
                "--no-individual was set but no summary will be written: "
                "pass --summary-name or provide multiple experiments."
            )
        return

    name = summary_name or "combined"
    if summary_dir is None:
        parent = os.path.commonpath([d for d, _ in loaded])
        summary_dir = os.path.join(parent, f"{name}__summary")

    all_runs = [(data.experiment_name, data) for _, data in loaded]

    varied_keys = infer_varied_summary_keys(all_runs)

    logger.info("Writing summary (%d experiment(s)) → %s/summary.md", len(all_runs), summary_dir)
    save_grid_summary(all_runs, varied_keys, summary_dir, name)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-generate analytics for one or more completed experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        metavar="DIR",
        help="Path(s) to experiment folder(s) containing results/experiment_data.json",
    )
    parser.add_argument(
        "--game",
        default=None,
        choices=["tmnf", "sc2", "torcs", "beamng", "car_racing", "assetto"],
        help=(
            "Game analytics module to use. "
            "Auto-detected from training_params when not given "
            "(uses training_params['game'] when available, "
            "otherwise SC2 is inferred from 'map_name'/'agent_race' keys)."
        ),
    )
    parser.add_argument(
        "--summary-name",
        default=None,
        metavar="NAME",
        help=(
            "Base name for the combined summary report. "
            "If given, a summary is written even for a single experiment. "
            "Defaults to 'combined' when multiple experiments are provided."
        ),
    )
    parser.add_argument(
        "--summary-dir",
        default=None,
        metavar="PATH",
        help=(
            "Output directory for the combined summary. "
            "Defaults to <common-parent>/<summary-name>__summary/"
        ),
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Skip regenerating individual experiment results; only write the combined summary.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    redo_analytics(
        experiment_dirs=args.experiment_dirs,
        game=args.game,
        summary_name=args.summary_name,
        summary_dir=args.summary_dir,
        no_individual=args.no_individual,
    )


if __name__ == "__main__":
    main()
