#!/usr/bin/env python3
"""Render PNG figures from pre-computed graph JSON files."""

import argparse
import json
import shutil
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

PHASE_COLORS = {
    "localization": "#C5B3F0",
    "patch":        "#FCC9B0",
    "validation":   "#A8E6F0",
    "general":      "#CFE0F6",
}

MODELS = ["claude-sonnet-4", "deepseek-v3", "deepseek-r1-0528", "devstral-small"]
DEFAULT_DPI = 800

MODEL_TRAJ_DIRS = {
    "claude-sonnet-4":  "claude-sonnet-4_maxiter_100_N_v0.40.0-no-hint-run_1",
    "deepseek-v3":      "deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1",
    "deepseek-r1-0528": "deepseek-r1-0528_maxiter_100_N_v0.40.0-no-hint-run_1",
    "devstral-small":   "devstral-small_maxiter_100_N_v0.40.0-no-hint-run_1",
}
TRAJ_DIR_TO_MODEL = {v: k for k, v in MODEL_TRAJ_DIRS.items()}

PHASE_TYPE_COLORS = {
    **PHASE_COLORS,
    "patching": PHASE_COLORS["patch"],
}

NODE_TYPE_COLORS = {
    "start":        "#E8E8E8",
    "phase":        None,  # resolved from phase_type
    "code_block":   "#FFF9C4",
    "termination":  "#FFCDD2",
}


def load_resolution_sets(trajectories_dir: Path, models: list[str]) -> dict[str, dict[str, set[str]]]:
    """Return {model: {"success": set, "failed": set}} from eval reports."""
    result: dict[str, dict[str, set[str]]] = {}
    for model in models:
        report_path = trajectories_dir / MODEL_TRAJ_DIRS[model] / "report.json"
        if not report_path.is_file():
            print(f"[WARN] Report not found: {report_path}", file=sys.stderr)
            result[model] = {"success": set(), "failed": set()}
            continue
        with open(report_path) as f:
            report = json.load(f)
        resolved = set(report.get("resolved_ids", []))
        unresolved = set(report.get("unresolved_ids", []))
        result[model] = {"success": resolved, "failed": unresolved}
    return result


def outcome_for(model: str, instance_id: str, resolution_sets: dict) -> str:
    if instance_id in resolution_sets[model]["success"]:
        return "success"
    return "failed"


def outcome_from_json(json_path: Path) -> str:
    with open(json_path) as f:
        status = json.load(f).get("graph", {}).get("resolution_status", "")
    return "success" if status == "resolved" else "failed"


def _node_label(nd: dict, node_id: str, label_len: int) -> str:
    if nd.get("label"):
        return nd["label"][:label_len]
    node_type = nd.get("node_type")
    if node_type == "phase":
        phase = nd.get("phase_type", "general")
        return f"{phase} [{nd.get('start_step', '?')}-{nd.get('end_step', '?')}]"[:label_len]
    if node_type == "code_block":
        path = nd.get("file_path", "")
        if "\n" in path or len(path) > 120:
            name = "code_block"
        else:
            name = path.rsplit("/", 1)[-1] if path else "?"
        sl, el = nd.get("start_line"), nd.get("end_line")
        if sl is not None and el is not None:
            return f"{name}:{sl}-{el}"[:label_len]
        return name[:label_len]
    if node_type == "termination":
        return (nd.get("termination_type") or "termination")[:label_len]
    if node_type == "start":
        return "start"
    return str(node_id)[:label_len]


def _node_color(nd: dict) -> str:
    node_type = nd.get("node_type")
    if node_type == "phase":
        phase = nd.get("phase_type", "general")
        return PHASE_TYPE_COLORS.get(phase, PHASE_COLORS["general"])
    if node_type in NODE_TYPE_COLORS and NODE_TYPE_COLORS[node_type]:
        return NODE_TYPE_COLORS[node_type]
    phases = nd.get("phases") or ["general"]
    return PHASE_COLORS.get(phases[0], PHASE_COLORS["general"])


def _graph_for_render(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return a structure-only copy; styling attrs are set explicitly after to_agraph."""
    R = nx.MultiDiGraph()
    R.add_nodes_from(G.nodes)
    for u, v, key in G.edges(keys=True):
        R.add_edge(u, v, key=key)
    return R


def _escape_dot_label(label: str) -> str:
    """Quote and escape a label for Graphviz DOT syntax."""
    escaped = label.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _build_agraph(G, splines="ortho", label_len=40, fontsize=8, dpi=DEFAULT_DPI):
    R = _graph_for_render(G)
    A = nx.nx_agraph.to_agraph(R)
    A.graph_attr.update(rankdir="TB", splines=splines, bgcolor="white", pad="0.5", dpi=str(dpi))
    A.node_attr.update(shape="box", style="filled,rounded", fontsize=fontsize, fontname="DejaVu Sans")
    A.edge_attr.update(fontsize=fontsize - 1, arrowsize=0.5)
    for n in A.nodes():
        nd = G.nodes[n]
        A.get_node(n).attr["fillcolor"] = _node_color(nd)
        A.get_node(n).attr["label"] = _escape_dot_label(_node_label(nd, n, label_len))
    return A


def render_png(json_path: str, png_path: str, dpi: int = DEFAULT_DPI, force: bool = False) -> str:
    """Render a single graph JSON file to PNG. Returns instance_id on success."""
    json_p = Path(json_path)
    png_p = Path(png_path)
    if png_p.exists() and not force:
        return json_p.parent.name

    with open(json_p) as f:
        G = json_graph.node_link_graph(json.load(f), edges="edges", directed=True, multigraph=True)

    png_p.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for splines, label_len, fontsize in [("ortho", 40, 8), ("true", 30, 7)]:
            try:
                A = _build_agraph(G, splines, label_len, fontsize, dpi=dpi)
                A.draw(str(png_p), prog="dot", format="png")
                break
            except Exception:
                if splines == "true":
                    raise

    return json_p.parent.name


def collect_jobs(
    graphs_dir: Path,
    output_dir: Path,
    models: list[str],
    resolution_sets: dict,
    *,
    auto_discover: bool = False,
    use_json_resolution: bool = False,
) -> list[tuple[str, str]]:
    jobs = []
    if auto_discover:
        model_dirs = sorted(p for p in graphs_dir.iterdir() if p.is_dir())
    else:
        model_dirs = [graphs_dir / model for model in models]

    for model_dir in model_dirs:
        if not model_dir.is_dir():
            print(f"[WARN] Model dir not found: {model_dir}", file=sys.stderr)
            continue
        model = TRAJ_DIR_TO_MODEL.get(model_dir.name, model_dir.name)
        for json_path in sorted(model_dir.glob("*/*.json")):
            iid = json_path.parent.name
            if use_json_resolution:
                outcome = outcome_from_json(json_path)
            else:
                outcome = outcome_for(model, iid, resolution_sets)
            png_path = output_dir / outcome / model / f"{iid}.png"
            jobs.append((str(json_path), str(png_path)))
    return jobs


def reorganize_existing(output_dir: Path, models: list[str], resolution_sets: dict) -> None:
    """Move flat {model}/{instance}.png files into success/ or failed/ subfolders."""
    moved = 0
    for model in models:
        model_dir = output_dir / model
        if not model_dir.is_dir():
            continue
        for png in model_dir.glob("*.png"):
            outcome = outcome_for(model, png.stem, resolution_sets)
            dest = output_dir / outcome / model / png.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(png), str(dest))
            moved += 1
        if model_dir.is_dir() and not any(model_dir.iterdir()):
            model_dir.rmdir()
    if moved:
        print(f"Reorganized {moved} existing figures into success/ and failed/")


def main():
    parser = argparse.ArgumentParser(description="Render PNG figures from graph JSON files")
    parser.add_argument("--graphs_dir", type=Path,
                        default=Path("data/OpenHands/graphs"),
                        help="Root directory containing {model}/{instance}/{instance}.json")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("figures/openhands"),
                        help="Output directory for PNG figures")
    parser.add_argument("--trajectories_dir", type=Path,
                        default=Path("/home/yiboz7/data/raw_trajectories/OpenHands"),
                        help="Directory with per-model report.json files")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="Model subdirectories to process")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                        help=f"Output PNG resolution (default: {DEFAULT_DPI})")
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if PNG already exists")
    parser.add_argument("--auto_discover", action="store_true",
                        help="Scan graphs_dir for model subdirectories (supports traj dir names)")
    parser.add_argument("--use_json_resolution", action="store_true",
                        help="Use resolution_status from graph JSON instead of eval reports")
    args = parser.parse_args()

    resolution_sets = load_resolution_sets(args.trajectories_dir, args.models)
    if not args.auto_discover:
        reorganize_existing(args.output_dir, args.models, resolution_sets)

    jobs = collect_jobs(
        args.graphs_dir,
        args.output_dir,
        args.models,
        resolution_sets,
        auto_discover=args.auto_discover,
        use_json_resolution=args.use_json_resolution,
    )
    if not args.force:
        jobs = [(j, p) for j, p in jobs if not Path(p).exists()]

    total = len(jobs)
    if total == 0:
        print("Nothing to render (all figures up to date).")
    else:
        print(f"Rendering {total} figures at {args.dpi} dpi with {args.workers} workers → {args.output_dir}/")
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(render_png, j, p, args.dpi, args.force): p for j, p in jobs}
            for future in as_completed(futures):
                done += 1
                if done % 100 == 0 or done == total:
                    print(f"  [{done}/{total}]")
                try:
                    future.result()
                except Exception as e:
                    print(f"  [ERROR] {futures[future]}: {e}", file=sys.stderr)
        print(f"Done. {done} figures written.")

    # Summary
    for outcome in ("success", "failed"):
        count = sum(
            len(list((args.output_dir / outcome / model).glob("*.png")))
            for model in args.models
            if (args.output_dir / outcome / model).is_dir()
        )
        print(f"  {outcome}: {count} figures")


if __name__ == "__main__":
    main()
