"""Common analytics data classes and generic plots.

TMNF-specific plots and combined entry points live in games.tmnf.analytics.

CLI usage — consolidate multiple grid-search runs into one summary:

    python analytics.py path/to/exp1 path/to/exp2 ... [options]

Each path should be an experiment folder containing results/experiment_data.json.
"""
from __future__ import annotations

from framework.analytics import (  # noqa: F401
    RunTrace,
    ProbeResult,
    ColdStartSimResult,
    ColdStartRestartResult,
    GreedySimResult,
    ExperimentData,
    plot_probe_rewards,
    plot_cold_start_rewards,
    plot_greedy_rewards,
    plot_reward_trajectory,
    _probe_table_md,
    _cold_start_table_md,
    _greedy_table_md,
    _timings_md,
    _summary_md,
    save_grid_summary,
)


def _load_experiment(folder: str) -> "ExperimentData":
    """Load an ExperimentData from an experiment folder's results/experiment_data.json."""
    import json
    import os
    from distributed.protocol import experiment_from_dict

    json_path = os.path.join(folder, "results", "experiment_data.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"No experiment_data.json found at {json_path!r}.\n"
            "Re-run the experiment to generate it, or run it with the updated code."
        )
    with open(json_path, encoding="utf-8") as f:
        return experiment_from_dict(json.loads(f.read()))


def _infer_varied_keys(runs: "list[tuple[str, ExperimentData]]") -> "list[str]":
    """Return training-param keys whose values differ across experiments."""
    import yaml
    import os

    all_params: list[dict] = []
    for _name, data in runs:
        params = dict(data.training_params)
        # Merge reward config if present
        if os.path.exists(data.reward_config_file):
            with open(data.reward_config_file, encoding="utf-8") as f:
                params.update(yaml.safe_load(f) or {})
        all_params.append(params)

    if not all_params:
        return []

    all_keys = set().union(*all_params)
    varied = [
        k for k in sorted(all_keys)
        if len({str(p.get(k)) for p in all_params}) > 1
    ]
    return varied


def consolidate_summary(
    folders: "list[str]",
    summary_dir: str,
    base_name: str,
) -> None:
    """Load ExperimentData from each folder and generate a consolidated grid summary.

    Parameters
    ----------
    folders:     List of experiment root folders (each must contain results/experiment_data.json).
    summary_dir: Directory to write the summary report and plots into.
    base_name:   Title / base name for the summary report.
    """
    import os
    from games.tmnf.analytics import save_grid_summary as _tmnf_save_grid_summary

    runs: list[tuple[str, ExperimentData]] = []
    for folder in folders:
        folder = os.path.normpath(folder)
        name = os.path.basename(folder)
        data = _load_experiment(folder)
        runs.append((name, data))

    varied_keys = _infer_varied_keys(runs)
    _tmnf_save_grid_summary(runs, varied_keys, summary_dir, base_name)
    print(f"Summary written to {summary_dir}/summary.md  ({len(runs)} experiments)")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description=(
            "Consolidate multiple grid-search experiment folders into a single summary.\n\n"
            "Each FOLDER must contain results/experiment_data.json "
            "(produced automatically by save_experiment_results)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "folders",
        nargs="+",
        metavar="FOLDER",
        help="Experiment folder(s) to include (e.g. games/tmnf/experiments/a03/my_run)",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="DIR",
        default=None,
        help=(
            "Directory to write summary into. "
            "Defaults to <parent of first folder>/<base_name>__summary."
        ),
    )
    parser.add_argument(
        "--name", "-n",
        metavar="NAME",
        default=None,
        help="Base name / title for the summary report (defaults to common prefix of folder names).",
    )

    args = parser.parse_args()

    # Derive base_name from common prefix or first folder name
    folder_names = [os.path.basename(os.path.normpath(f)) for f in args.folders]
    if args.name:
        base_name = args.name
    else:
        # Find longest common prefix among folder names
        if len(folder_names) == 1:
            base_name = folder_names[0]
        else:
            prefix = os.path.commonprefix(folder_names).rstrip("_-")
            base_name = prefix if prefix else "consolidated"

    # Derive output dir
    if args.output:
        summary_dir = args.output
    else:
        parent = os.path.dirname(os.path.normpath(args.folders[0]))
        summary_dir = os.path.join(parent, f"{base_name}__summary")

    consolidate_summary(args.folders, summary_dir, base_name)

