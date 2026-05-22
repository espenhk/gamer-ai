#!/usr/bin/env python
"""Collate multiple grid-search summaries under a policy/version tree.

The script scans a root folder recursively for grid-search summary directories
(``*__summary/summary.md``) whose parent folders follow the repository's
``<policy>/vX/`` layout. For each discovered grid search it:

1. loads the underlying experiment_data.json files,
2. copies the summary bundle into the output directory,
3. rewrites copied summary links so they still point to the original runs, and
4. writes a cross-grid ``summary.md`` comparing the discovered searches.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import shutil
from dataclasses import dataclass
from statistics import fmean

import yaml

logger = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)


@dataclass
class GridSearchFamily:
    policy_name: str
    version_name: str
    version_dir: str
    summary_dir: str
    summary_name: str
    copied_summary_dir: str | None = None
    copied_summary_rel: str | None = None
    runs: list | None = None
    best_run_name: str | None = None
    best_reward: float | None = None
    avg_best_reward: float | None = None
    best_generation: int | None = None
    avg_generation_to_best: float | None = None
    population_sizes: list[str] | None = None
    hidden_layer_sizes: list[str] | None = None
    varied_keys: list[str] | None = None
    best_run_params: dict | None = None


def _read_reward_config(path: str | None) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            parsed = yaml.safe_load(f) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        logger.warning("Failed to read reward config: %s", path)
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _is_subpath(path: str, parent: str) -> bool:
    normalized_path = os.path.normcase(os.path.abspath(path))
    normalized_parent = os.path.normcase(os.path.abspath(parent))
    if normalized_path == normalized_parent:
        return True
    return normalized_path.startswith(normalized_parent + os.sep)


def _format_value(value) -> str:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:g}"
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    return str(value)


def _format_reward(value: float | None) -> str:
    return f"{value:+.1f}" if value is not None else "—"


def _format_generation(value: float | int | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.1f}"
    return str(int(value))


def _list_label(values: list[str] | None) -> str:
    return ", ".join(values) if values else "—"


def _best_reward_and_generation(data) -> tuple[float | None, int | None]:
    if not data.greedy_sims:
        return None, None
    best_reward = max(s.reward for s in data.greedy_sims)
    ordered = sorted(data.greedy_sims, key=lambda s: s.sim)
    best_generation = next((s.sim for s in ordered if s.reward == best_reward), None)
    return best_reward, best_generation


def _extract_population_size(training_params: dict) -> str | None:
    policy_params = training_params.get("policy_params")
    if isinstance(policy_params, dict) and "population_size" in policy_params:
        return _format_value(policy_params["population_size"])
    if "population_size" in training_params:
        return _format_value(training_params["population_size"])
    return None


def _extract_hidden_sizes(training_params: dict) -> str | None:
    policy_params = training_params.get("policy_params")
    if isinstance(policy_params, dict):
        if "hidden_sizes" in policy_params:
            return _format_value(policy_params["hidden_sizes"])
        if "hidden_size" in policy_params:
            return _format_value(policy_params["hidden_size"])
    if "hidden_sizes" in training_params:
        return _format_value(training_params["hidden_sizes"])
    if "hidden_size" in training_params:
        return _format_value(training_params["hidden_size"])
    return None


def _remap_data_paths(experiment_dir: str, data) -> None:
    for attr, filename in (
        ("weights_file", "policy_weights.yaml"),
        ("reward_config_file", "reward_config.yaml"),
    ):
        stored = getattr(data, attr, None)
        if stored and os.path.exists(stored):
            continue
        candidate = os.path.join(experiment_dir, filename)
        if os.path.exists(candidate):
            setattr(data, attr, candidate)


def _flatten_run_params(training_params: dict, reward_cfg: dict) -> dict:
    params = {key: value for key, value in training_params.items() if key != "policy_params"}
    policy_params = training_params.get("policy_params")
    if isinstance(policy_params, dict):
        for key, value in policy_params.items():
            if key == "c" and "mcts_c" in params:
                continue
            params.setdefault(key, value)
    params.update(reward_cfg)
    return params


def _infer_flattened_varied_keys(runs: list[tuple[str, object]]) -> list[str]:
    flattened_by_run: list[dict] = []
    all_keys: set[str] = set()
    for _, data in runs:
        reward_cfg = _read_reward_config(data.reward_config_file)
        flattened = _flatten_run_params(data.training_params, reward_cfg)
        flattened_by_run.append(flattened)
        all_keys.update(flattened.keys())
    return sorted(key for key in all_keys if len({str(flattened.get(key)) for flattened in flattened_by_run}) > 1)


def discover_grid_search_families(
    root_dir: str,
    exclude_dirs: list[str] | None = None,
) -> list[GridSearchFamily]:
    families: list[GridSearchFamily] = []
    exclude_roots = [os.path.abspath(path) for path in (exclude_dirs or [])]
    seen_version_dirs: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        abs_dirpath = os.path.abspath(dirpath)
        child_dirs = {name: os.path.join(abs_dirpath, name) for name in dirnames}
        dirnames[:] = [
            name
            for name, child_dir in child_dirs.items()
            if not any(_is_subpath(child_dir, exclude_root) for exclude_root in exclude_roots)
        ]
        if any(_is_subpath(abs_dirpath, exclude_root) for exclude_root in exclude_roots):
            continue
        if "summary.md" not in filenames or not dirpath.endswith("__summary"):
            continue
        version_dir = os.path.dirname(dirpath)
        if version_dir in seen_version_dirs:
            continue
        version_name = os.path.basename(version_dir)
        if not _VERSION_RE.fullmatch(version_name):
            continue
        policy_dir = os.path.dirname(version_dir)
        policy_name = os.path.basename(policy_dir)
        seen_version_dirs.add(version_dir)
        families.append(
            GridSearchFamily(
                policy_name=policy_name,
                version_name=version_name,
                version_dir=version_dir,
                summary_dir=dirpath,
                summary_name=os.path.basename(dirpath),
            )
        )
    families.sort(key=lambda f: (f.policy_name, f.version_name, f.summary_name))
    return families


def _rewrite_copied_summary_links(
    source_summary_dir: str,
    copied_summary_dir: str,
) -> None:
    summary_md = os.path.join(copied_summary_dir, "summary.md")
    if not os.path.exists(summary_md):
        return
    with open(summary_md, encoding="utf-8") as f:
        content = f.read()

    def repl(match: re.Match[str]) -> str:
        rel_path = match.group(1)
        target_abs = os.path.normpath(os.path.join(source_summary_dir, rel_path))
        new_rel = os.path.relpath(target_abs, copied_summary_dir)
        return "(" + new_rel.replace(os.sep, "/") + ")"

    rewritten = re.sub(r"\((\.\./[^)]+)\)", repl, content)
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(rewritten)


def copy_grid_search_summary(
    family: GridSearchFamily,
    output_dir: str,
) -> tuple[str, str]:
    copied_summary_dir = os.path.join(
        output_dir,
        "grid_summaries",
        family.policy_name,
        family.version_name,
        family.summary_name,
    )
    parent = os.path.dirname(copied_summary_dir)
    os.makedirs(parent, exist_ok=True)
    try:
        if os.path.samefile(family.summary_dir, copied_summary_dir):
            copied_summary_rel = os.path.relpath(
                os.path.join(copied_summary_dir, "summary.md"),
                output_dir,
            ).replace(os.sep, "/")
            return copied_summary_dir, copied_summary_rel
    except FileNotFoundError:
        pass
    if os.path.exists(copied_summary_dir):
        shutil.rmtree(copied_summary_dir)
    shutil.copytree(family.summary_dir, copied_summary_dir)
    _rewrite_copied_summary_links(family.summary_dir, copied_summary_dir)
    copied_summary_rel = os.path.relpath(
        os.path.join(copied_summary_dir, "summary.md"),
        output_dir,
    ).replace(os.sep, "/")
    return copied_summary_dir, copied_summary_rel


def _load_family_metrics(family: GridSearchFamily) -> GridSearchFamily | None:
    from framework.analytics import load_experiment_data

    runs: list[tuple[str, object]] = []
    for name in sorted(os.listdir(family.version_dir)):
        experiment_dir = os.path.join(family.version_dir, name)
        if not os.path.isdir(experiment_dir) or experiment_dir == family.summary_dir:
            continue
        experiment_data_path = os.path.join(experiment_dir, "results", "experiment_data.json")
        if not os.path.exists(experiment_data_path):
            continue
        try:
            data = load_experiment_data(experiment_dir)
        except FileNotFoundError:
            continue
        _remap_data_paths(experiment_dir, data)
        runs.append((data.experiment_name, data))

    if not runs:
        logger.warning("No experiment data found for %s", family.version_dir)
        return None

    varied_keys = _infer_flattened_varied_keys(runs)
    per_run_best: list[float] = []
    per_run_generation: list[int] = []
    population_sizes: set[str] = set()
    hidden_sizes: set[str] = set()
    best_pair: tuple[str, object] | None = None
    best_reward: float | None = None
    best_generation: int | None = None

    for run_name, data in runs:
        run_best_reward, run_best_generation = _best_reward_and_generation(data)
        if run_best_reward is not None:
            per_run_best.append(run_best_reward)
            if best_reward is None or run_best_reward > best_reward:
                best_pair = (run_name, data)
                best_reward = run_best_reward
                best_generation = run_best_generation
        if run_best_generation is not None:
            per_run_generation.append(run_best_generation)
        pop_size = _extract_population_size(data.training_params)
        if pop_size is not None:
            population_sizes.add(pop_size)
        hidden = _extract_hidden_sizes(data.training_params)
        if hidden is not None:
            hidden_sizes.add(hidden)

    best_run_params: dict = {}
    best_run_name: str | None = None
    if best_pair is not None:
        best_run_name, best_data = best_pair
        reward_cfg = _read_reward_config(best_data.reward_config_file)
        all_params = _flatten_run_params(best_data.training_params, reward_cfg)
        keys = varied_keys or sorted(all_params)
        best_run_params = {k: all_params.get(k, "?") for k in keys}

    family.runs = [data for _, data in runs]
    family.best_run_name = best_run_name
    family.best_reward = best_reward
    family.avg_best_reward = fmean(per_run_best) if per_run_best else None
    family.best_generation = best_generation
    family.avg_generation_to_best = fmean(per_run_generation) if per_run_generation else None
    family.population_sizes = sorted(population_sizes)
    family.hidden_layer_sizes = sorted(hidden_sizes)
    family.varied_keys = varied_keys
    family.best_run_params = best_run_params
    return family


def write_cross_grid_summary(
    families: list[GridSearchFamily],
    root_dir: str,
    output_dir: str,
    summary_name: str,
) -> str:
    ranked = sorted(
        families,
        key=lambda f: (
            f.best_reward is None,
            0.0 if f.best_reward is None else -f.best_reward,
            f.avg_best_reward is None,
            0.0 if f.avg_best_reward is None else -f.avg_best_reward,
            f.policy_name,
            f.version_name,
        ),
    )

    lines = [
        f"# Cross-Grid Search Summary: {summary_name}\n\n",
        f"Scanned `{root_dir}` and found {len(ranked)} grid-search summaries.\n\n",
        "## Grid search comparison\n\n",
        "| Rank | Policy | Version | Runs | Best Reward | Avg Best Reward | Best Run | Best Gen | Avg Gen to Best | Population Sizes | Hidden Sizes | Summary |\n",
        "|------|--------|---------|------|-------------|-----------------|----------|----------|-----------------|------------------|--------------|---------|\n",
    ]
    for rank, family in enumerate(ranked, 1):
        run_count = len(family.runs or [])
        summary_link = f"[copied summary]({family.copied_summary_rel})" if family.copied_summary_rel else "—"
        lines.append(
            f"| {rank} | {family.policy_name} | {family.version_name} | {run_count} "
            f"| {_format_reward(family.best_reward)} | {_format_reward(family.avg_best_reward)} "
            f"| {family.best_run_name or '—'} | {_format_generation(family.best_generation)} "
            f"| {_format_generation(family.avg_generation_to_best)} | {_list_label(family.population_sizes)} "
            f"| {_list_label(family.hidden_layer_sizes)} | {summary_link} |\n"
        )
    lines.append("\n")

    for rank, family in enumerate(ranked, 1):
        lines.append(f"---\n\n## {rank}. {family.policy_name} / {family.version_name}\n\n")
        lines.append(f"- Source summary: `{family.summary_dir}`\n")
        if family.copied_summary_rel:
            lines.append(f"- Copied summary: [{family.summary_name}]({family.copied_summary_rel})\n")
        lines.append(f"- Runs discovered: {len(family.runs or [])}\n")
        lines.append(f"- Best run: `{family.best_run_name or '—'}` ({_format_reward(family.best_reward)})\n")
        lines.append(f"- Average best reward: {_format_reward(family.avg_best_reward)}\n")
        lines.append(f"- Best generation to best reward: {_format_generation(family.best_generation)}\n")
        lines.append(f"- Average generation to best reward: {_format_generation(family.avg_generation_to_best)}\n")
        lines.append(f"- Population sizes: {_list_label(family.population_sizes)}\n")
        lines.append(f"- Hidden layer sizes: {_list_label(family.hidden_layer_sizes)}\n\n")
        if family.best_run_params:
            lines.append("### Best-run parameter choices\n\n")
            lines.append("| Param | Value |\n|---|---|\n")
            for key, value in family.best_run_params.items():
                lines.append(f"| `{key}` | {_format_value(value)} |\n")
            lines.append("\n")

    report_path = os.path.join(output_dir, "summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(lines).rstrip("\n") + "\n")
    logger.info("Saved cross-grid summary → %s", report_path)
    return report_path


def build_cross_grid_report(
    root_dir: str,
    output_dir: str | None = None,
    summary_name: str = "cross_grid",
) -> str | None:
    root_dir = os.path.abspath(root_dir)
    output_dir = os.path.abspath(output_dir or os.path.join(root_dir, f"{summary_name}__summary"))
    os.makedirs(output_dir, exist_ok=True)

    families = discover_grid_search_families(root_dir, exclude_dirs=[output_dir])
    if not families:
        logger.error("No grid-search summaries found under %s", root_dir)
        return None

    loaded_families: list[GridSearchFamily] = []
    for family in families:
        copied_summary_dir, copied_summary_rel = copy_grid_search_summary(family, output_dir)
        family.copied_summary_dir = copied_summary_dir
        family.copied_summary_rel = copied_summary_rel
        loaded = _load_family_metrics(family)
        if loaded is not None:
            loaded_families.append(loaded)

    if not loaded_families:
        logger.error("No experiment data found for discovered summaries under %s", root_dir)
        return None

    return write_cross_grid_summary(loaded_families, root_dir, output_dir, summary_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare grid-search summaries found under a policy/version tree.",
    )
    parser.add_argument(
        "root_dir",
        help="Root directory to scan recursively for <policy>/vX/*__summary/summary.md",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write the copied summaries and cross-grid summary into.",
    )
    parser.add_argument(
        "--summary-name",
        default="cross_grid",
        help="Base name used in the generated report title and default output dir.",
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
    build_cross_grid_report(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        summary_name=args.summary_name,
    )


if __name__ == "__main__":
    main()
