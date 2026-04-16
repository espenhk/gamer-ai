"""Manual parameter exploration tracker. Add, list, remove, and plot entries."""

import argparse
import json
import os
import sys
from datetime import datetime

_DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "param_exploration.json")

_EXCLUDED_LOAD_KEYS = {"track_name", "centerline_path"}


def _load_db(path):
    if not os.path.exists(path):
        return {"next_id": 1, "entries": []}
    with open(path) as f:
        return json.load(f)


def _save_db(db, path):
    with open(path, "w") as f:
        json.dump(db, f, indent=2)
        f.write("\n")


def _parse_kv_args(extras):
    params = {}
    for item in extras:
        if "=" not in item:
            sys.exit(f"Error: expected key=value, got: {item!r}")
        k, v = item.split("=", 1)
        try:
            params[k] = float(v)
        except ValueError:
            params[k] = v
    return params


def cmd_add(args, extras, db_path):
    params = _parse_kv_args(extras)
    if not params:
        sys.exit("Error: provide at least one key=value param (e.g. mutation_scale=0.05)")
    db = _load_db(db_path)
    entry = {
        "id": db["next_id"],
        "score": float(args.score),
        "note": args.note or "",
        "params": params,
        "added": datetime.now().isoformat(timespec="seconds"),
    }
    db["entries"].append(entry)
    db["next_id"] += 1
    _save_db(db, db_path)
    print(f"Added entry #{entry['id']} (score={entry['score']}, {len(params)} param(s))")


def cmd_load(args, db_path):
    exp_dir = args.experiment_dir
    if not os.path.isdir(exp_dir):
        sys.exit(f"Error: not a directory: {exp_dir}")

    params = {}
    for fname in ("training_params.yaml", "reward_config.yaml"):
        fpath = os.path.join(exp_dir, fname)
        if os.path.exists(fpath):
            import yaml
            with open(fpath) as f:
                data = yaml.safe_load(f) or {}
            for k, v in data.items():
                if k in _EXCLUDED_LOAD_KEYS:
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    params[k] = float(v)
                elif isinstance(v, str):
                    params[k] = v
                # skip dicts (policy_params nested block)

    if not params:
        sys.exit(f"Error: no usable params found in {exp_dir}")

    db = _load_db(db_path)
    entry = {
        "id": db["next_id"],
        "score": float(args.score),
        "note": args.note or "",
        "params": params,
        "added": datetime.now().isoformat(timespec="seconds"),
    }
    db["entries"].append(entry)
    db["next_id"] += 1
    _save_db(db, db_path)
    print(f"Loaded {len(params)} params from {exp_dir}, added as entry #{entry['id']} (score={entry['score']})")


def cmd_list(args, db_path):
    db = _load_db(db_path)
    entries = db["entries"]
    if not entries:
        print("No entries yet.")
        return

    if args.sort == "score":
        entries = sorted(entries, key=lambda e: e["score"], reverse=True)

    # Build rows
    rows = []
    for e in entries:
        param_str = ", ".join(f"{k}={v}" for k, v in e["params"].items())
        if len(param_str) > 60:
            param_str = param_str[:57] + "..."
        note = e["note"][:30] + ("..." if len(e["note"]) > 30 else "")
        rows.append((str(e["id"]), f"{e['score']:.1f}", note, param_str))

    id_w = max(2, max(len(r[0]) for r in rows))
    sc_w = max(5, max(len(r[1]) for r in rows))
    no_w = max(4, max(len(r[2]) for r in rows))

    header = f"{'ID'.ljust(id_w)}  {'Score'.ljust(sc_w)}  {'Note'.ljust(no_w)}  Params"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r[0].ljust(id_w)}  {r[1].ljust(sc_w)}  {r[2].ljust(no_w)}  {r[3]}")


def cmd_plot(args, db_path):
    try:
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
    except ImportError:
        sys.exit("Error: matplotlib and numpy are required for the plot command")

    x_param, y_param = args.x_param, args.y_param
    db = _load_db(db_path)
    visible = [e for e in db["entries"] if x_param in e["params"] and y_param in e["params"]]

    if not visible:
        print(f"No entries with both '{x_param}' and '{y_param}' — nothing to plot.")
        return

    raw_xs = [e["params"][x_param] for e in visible]
    raw_ys = [e["params"][y_param] for e in visible]
    scores = [e["score"] for e in visible]

    # Handle categorical axes
    def _encode(vals):
        if all(isinstance(v, (int, float)) for v in vals):
            return [float(v) for v in vals], None
        labels = sorted(set(vals), key=str)
        mapping = {v: i for i, v in enumerate(labels)}
        return [float(mapping[v]) for v in vals], labels

    xs, x_labels = _encode(raw_xs)
    ys, y_labels = _encode(raw_ys)

    # Jitter overlapping points
    np.random.seed(42)
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    x_range = max(xs) - min(xs) if len(xs) > 1 else 1.0
    y_range = max(ys) - min(ys) if len(ys) > 1 else 1.0
    from collections import Counter
    coord_counts = Counter(zip(xs.tolist(), ys.tolist()))
    jx = np.zeros(len(xs))
    jy = np.zeros(len(ys))
    for i, (x, y) in enumerate(zip(xs.tolist(), ys.tolist())):
        if coord_counts[(x, y)] > 1:
            jx[i] = np.random.normal(0, 0.02 * (x_range or 1))
            jy[i] = np.random.normal(0, 0.02 * (y_range or 1))
    xs = xs + jx
    ys = ys + jy

    # Color normalization
    mn, mx = min(scores), max(scores)
    if mn == mx:
        mn = mx - 1
    norm = mcolors.Normalize(vmin=mn, vmax=mx)
    colors = [cm.RdYlGn(norm(s)) for s in scores]

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(xs, ys, c=scores, cmap="RdYlGn", norm=norm, s=80,
                    edgecolors="#333333", linewidths=0.5, zorder=3)
    fig.colorbar(sc, ax=ax, label="Score")

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f"Parameter Exploration: {x_param} vs {y_param}  (n={len(visible)})")
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

    if args.annotate:
        for x, y, s in zip(xs, ys, scores):
            ax.annotate(f"{s:.0f}", (x, y), fontsize=7, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points")

    out = args.out or f"param_plot_{x_param}_vs_{y_param}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out}")


def cmd_remove(args, db_path):
    db = _load_db(db_path)
    target_id = args.id
    for i, e in enumerate(db["entries"]):
        if e["id"] == target_id:
            removed = db["entries"].pop(i)
            _save_db(db, db_path)
            params_summary = ", ".join(f"{k}={v}" for k, v in removed["params"].items())
            print(f"Removed entry #{target_id} (score={removed['score']}, params: {params_summary})")
            return
    print(f"No entry with ID {target_id}.")


def main():
    parser = argparse.ArgumentParser(description="Manual parameter exploration tracker")
    parser.add_argument("--db", default=_DEFAULT_DB, metavar="PATH",
                        help="Path to the JSON database (default: param_exploration.json)")
    sub = parser.add_subparsers(dest="command", required=True)

    add_p = sub.add_parser("add", help="Add an entry with key=value params")
    add_p.add_argument("--score", type=float, required=True, help="Quality score for this run")
    add_p.add_argument("--note", default="", help="Optional free-text note")

    load_p = sub.add_parser("load", help="Load params from an experiment directory")
    load_p.add_argument("experiment_dir", help="Path to experiment directory")
    load_p.add_argument("--score", type=float, required=True, help="Quality score for this run")
    load_p.add_argument("--note", default="", help="Optional free-text note")

    list_p = sub.add_parser("list", help="List all tracked entries")
    list_p.add_argument("--sort", choices=["id", "score"], default="id")

    plot_p = sub.add_parser("plot", help="Scatter plot of two parameters colored by score")
    plot_p.add_argument("x_param", help="Parameter name for X axis")
    plot_p.add_argument("y_param", help="Parameter name for Y axis")
    plot_p.add_argument("--out", default=None, help="Output PNG path")
    plot_p.add_argument("--annotate", action="store_true", help="Label each dot with its score")

    remove_p = sub.add_parser("remove", help="Remove an entry by ID")
    remove_p.add_argument("id", type=int, help="Entry ID to remove")

    args, extras = parser.parse_known_args()

    if extras and args.command != "add":
        print(f"Warning: unexpected arguments ignored: {extras}", file=sys.stderr)

    db_path = args.db

    if args.command == "add":
        cmd_add(args, extras, db_path)
    elif args.command == "load":
        cmd_load(args, db_path)
    elif args.command == "list":
        cmd_list(args, db_path)
    elif args.command == "plot":
        cmd_plot(args, db_path)
    elif args.command == "remove":
        cmd_remove(args, db_path)


if __name__ == "__main__":
    main()
