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


def _build_agraph(G, splines="ortho", label_len=40, fontsize=8, dpi=DEFAULT_DPI):
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update(rankdir="TB", splines=splines, bgcolor="white", pad="0.5", dpi=str(dpi))
    A.node_attr.update(shape="box", style="filled,rounded", fontsize=fontsize, fontname="DejaVu Sans")
    A.edge_attr.update(fontsize=fontsize - 1, arrowsize=0.5)
    for n in A.nodes():
        nd = G.nodes[n]
        phases = nd.get("phases") or ["general"]
        A.get_node(n).attr["fillcolor"] = PHASE_COLORS.get(phases[0], PHASE_COLORS["general"])
        A.get_node(n).attr["label"] = (nd.get("label") or n)[:label_len].replace("\n", "\\n")
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
) -> list[tuple[str, str]]:
    jobs = []
    for model in models:
        model_dir = graphs_dir / model
        if not model_dir.is_dir():
            print(f"[WARN] Model dir not found: {model_dir}", file=sys.stderr)
            continue
        for json_path in sorted(model_dir.glob("*/*.json")):
            iid = json_path.parent.name
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
    args = parser.parse_args()

    resolution_sets = load_resolution_sets(args.trajectories_dir, args.models)
    reorganize_existing(args.output_dir, args.models, resolution_sets)

    jobs = collect_jobs(args.graphs_dir, args.output_dir, args.models, resolution_sets)
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
