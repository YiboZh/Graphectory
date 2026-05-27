#!/usr/bin/env python3
"""Render interactive Plotly HTML figures from pre-computed graph JSON files."""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_REPO / "graph_construction"), str(_REPO / "scripts"), str(_REPO)]

from plotly_renderer import render_graph_html  # noqa: E402
from render_graph_figures import (  # noqa: E402
    MODELS,
    TRAJ_DIR_TO_MODEL,
    collect_jobs,
    load_resolution_sets,
    outcome_for,
    outcome_from_json,
)


def _render_job(json_path: str, html_path: str, force: bool = False) -> str:
    out = Path(html_path)
    if out.exists() and not force:
        return Path(json_path).parent.name
    render_graph_html(json_path, html_path)
    return Path(json_path).parent.name


def collect_html_jobs(
    graphs_dir: Path,
    output_dir: Path,
    models: list[str],
    resolution_sets: dict,
    *,
    auto_discover: bool = False,
    use_json_resolution: bool = False,
) -> list[tuple[str, str]]:
    png_jobs = collect_jobs(
        graphs_dir,
        output_dir,
        models,
        resolution_sets,
        auto_discover=auto_discover,
        use_json_resolution=use_json_resolution,
    )
    return [(j, str(Path(p).with_suffix(".html"))) for j, p in png_jobs]


def main():
    parser = argparse.ArgumentParser(
        description="Render interactive Plotly HTML graphs with node/edge hover tooltips",
    )
    parser.add_argument(
        "--graphs_dir",
        type=Path,
        required=True,
        help="Root directory containing {model}/{instance}/{instance}.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for HTML figures",
    )
    parser.add_argument(
        "--trajectories_dir",
        type=Path,
        default=Path("/home/yiboz7/data/raw_trajectories/OpenHands"),
        help="Directory with per-model report.json files",
    )
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--auto_discover", action="store_true")
    parser.add_argument("--use_json_resolution", action="store_true")
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Render a single instance_id only (for quick testing)",
    )
    args = parser.parse_args()

    resolution_sets = load_resolution_sets(args.trajectories_dir, args.models)

    if args.instance:
        found = False
        for json_path in sorted(args.graphs_dir.rglob(f"{args.instance}/{args.instance}.json")):
            found = True
            model_dir = json_path.parents[1].name
            model = TRAJ_DIR_TO_MODEL.get(model_dir, model_dir)
            if args.use_json_resolution:
                outcome = outcome_from_json(json_path)
            else:
                outcome = outcome_for(model, args.instance, resolution_sets)
            html_path = args.output_dir / outcome / model / f"{args.instance}.html"
            render_graph_html(json_path, html_path)
            print(f"Wrote {html_path}")
        if not found:
            print(f"[ERROR] Instance not found: {args.instance}", file=sys.stderr)
            sys.exit(1)
        return

    jobs = collect_html_jobs(
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
        print("Nothing to render (all HTML files up to date).")
        return

    print(f"Rendering {total} HTML graphs with {args.workers} workers → {args.output_dir}/")
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_render_job, j, p, args.force): p for j, p in jobs}
        for future in as_completed(futures):
            done += 1
            if done % 100 == 0 or done == total:
                print(f"  [{done}/{total}]")
            try:
                future.result()
            except Exception as e:
                print(f"  [ERROR] {futures[future]}: {e}", file=sys.stderr)
    print(f"Done. {done} HTML files written.")


if __name__ == "__main__":
    main()
