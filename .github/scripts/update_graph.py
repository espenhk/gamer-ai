"""
AST-only knowledge graph update for CI.

Detects which code files changed since the last graph build (via manifest.json),
re-runs AST extraction on those files, merges into the existing graph, re-clusters,
and writes updated graph.json, GRAPH_REPORT.md, and manifest.json.

No LLM calls — safe to run in CI without an API key.
"""

import json
import sys
from pathlib import Path

from graphify.analyze import god_nodes, suggest_questions, surprising_connections
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.detect import detect_incremental, save_manifest
from graphify.export import to_json
from graphify.extract import collect_files, extract
from graphify.report import generate
from networkx.readwrite import json_graph

REPO_ROOT = Path(__file__).parent.parent.parent
GRAPH_DIR = REPO_ROOT / "graphify-out"
GRAPH_JSON = GRAPH_DIR / "graph.json"
MANIFEST = GRAPH_DIR / "manifest.json"

CODE_EXTENSIONS = {
    ".py",
    ".ts",
    ".js",
    ".go",
    ".rs",
    ".java",
    ".cpp",
    ".c",
    ".rb",
    ".swift",
    ".kt",
    ".cs",
    ".scala",
    ".php",
}


def main() -> None:
    GRAPH_DIR.mkdir(exist_ok=True)

    if not GRAPH_JSON.exists():
        print("No graph.json found — skipping update (run /graphify locally first).")
        sys.exit(0)

    if not MANIFEST.exists():
        print("No manifest.json found — skipping update (run /graphify locally first).")
        sys.exit(0)

    # Find changed files since last graph build
    incremental = detect_incremental(REPO_ROOT)
    new_files = incremental.get("new_files", {})
    deleted = set(incremental.get("deleted_files", []))

    all_changed = [f for files in new_files.values() for f in files]
    code_changed = [f for f in all_changed if Path(f).suffix.lower() in CODE_EXTENSIONS]

    print(f"Changed files: {len(all_changed)} total, {len(code_changed)} code, {len(deleted)} deleted")

    if not code_changed and not deleted:
        print("No code changes detected — graph is up to date.")
        save_manifest(incremental["files"])
        sys.exit(0)

    # Load existing graph
    existing_data = json.loads(GRAPH_JSON.read_text(encoding="utf-8"))
    G = json_graph.node_link_graph(existing_data, edges="links")
    print(f"Loaded existing graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Prune nodes from deleted/changed files (changed files will be re-extracted)
    stale_sources = deleted | set(code_changed)
    stale_nodes = [
        n
        for n, d in G.nodes(data=True)
        if d.get("source_file") in stale_sources or str(REPO_ROOT / d.get("source_file", "")) in stale_sources
    ]
    if stale_nodes:
        G.remove_nodes_from(stale_nodes)
        print(f"Pruned {len(stale_nodes)} stale nodes")

    # Re-run AST on changed code files
    if code_changed:
        code_paths = []
        for f in code_changed:
            p = Path(f)
            if not p.is_absolute():
                p = REPO_ROOT / p
            if p.exists():
                code_paths.extend(collect_files(p) if p.is_dir() else [p])

        if code_paths:
            print(f"Running AST extraction on {len(code_paths)} file(s)...")
            ast_result = extract(code_paths)
            G_new = build_from_json(ast_result)
            G.update(G_new)
            print(f"Added {G_new.number_of_nodes()} nodes, {G_new.number_of_edges()} edges from AST")

    print(f"Merged graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Re-cluster
    communities = cluster(G)
    cohesion = score_all(G, communities)
    print(f"Re-clustered: {len(communities)} communities")

    # Generate labels from highest-degree node per community
    labels = {}
    for cid, node_ids in communities.items():
        ranked = sorted(node_ids, key=lambda n: G.degree(n), reverse=True)
        for nid in ranked[:5]:
            raw = G.nodes[nid].get("label", "")
            if raw and len(raw) <= 60 and not raw.endswith((".py", ".yaml")):
                labels[cid] = raw[:40]
                break
        if cid not in labels:
            labels[cid] = G.nodes[ranked[0]].get("label", f"Community {cid}")[:40]

    # Regenerate report
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    questions = suggest_questions(G, communities, labels)
    tokens = {"input": 0, "output": 0}
    detection = {
        "total_files": len(incremental.get("files", {}).get("code", [])),
        "total_words": 0,
        "needs_graph": True,
        "warning": None,
        "files": incremental.get("files", {"code": [], "document": [], "paper": []}),
    }
    report = generate(
        G,
        communities,
        cohesion,
        labels,
        gods,
        surprises,
        detection,
        tokens,
        str(REPO_ROOT),
        suggested_questions=questions,
    )
    (GRAPH_DIR / "GRAPH_REPORT.md").write_text(report, encoding="utf-8")

    # Save outputs
    to_json(G, communities, str(GRAPH_JSON))
    save_manifest(incremental["files"])

    # Update cost.json (AST runs are free — record with 0 tokens)
    cost_path = GRAPH_DIR / "cost.json"
    cost = (
        json.loads(cost_path.read_text(encoding="utf-8"))
        if cost_path.exists()
        else {
            "runs": [],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }
    )
    from datetime import datetime, timezone

    cost["runs"].append(
        {
            "date": datetime.now(timezone.utc).isoformat(),
            "input_tokens": 0,
            "output_tokens": 0,
            "files": len(code_changed),
            "note": "CI AST-only update",
        }
    )
    cost_path.write_text(json.dumps(cost, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Graph updated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(communities)} communities")
    print("Wrote: graph.json, GRAPH_REPORT.md, manifest.json, cost.json")


if __name__ == "__main__":
    main()
