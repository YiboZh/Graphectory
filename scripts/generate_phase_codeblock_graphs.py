#!/usr/bin/env python3
"""
Generate phase-centric + code-block graphs for OpenHands trajectories.

Walks all model run directories under a raw-trajectories root, finds each
output.jsonl, loads trajectories line-by-line, and saves a graph JSON per
instance to the specified output directory.

Two graph variants are supported via --graph_version:

  v1 (default, version 1)
      Full graph.  Code-block nodes are keyed by (file_path, start_line, end_line);
      all files touched by the agent (viewed, searched, or edited) are included.
      Output default: /home/yiboz7/data/processed_graphs/phase_codeblock_openhands

  v2 (version 2, ablation)
      Modified-files-only graph.  Code-block nodes are keyed by file_path only
      (no line numbers stored on nodes).  Files that were only viewed or searched
      but never edited/created/deleted are removed from the graph.
      Output default: /home/yiboz7/data/processed_graphs/phase_codeblock_openhands_v2

Use --html to also render an interactive Plotly HTML figure for each graph.
HTML files are saved alongside the JSON: {instance_id}.html in the same directory.

Default raw input:  /home/yiboz7/data/raw_trajectories/OpenHands

Usage
-----
# Version 1 (default) — all models
python scripts/generate_phase_codeblock_graphs.py

# Version 2 ablation — all models, also render HTML
python scripts/generate_phase_codeblock_graphs.py --graph_version v2 --html

# Override paths / options
python scripts/generate_phase_codeblock_graphs.py \\
    --graph_version v2 \\
    --trajs_root  /path/to/raw_trajectories/OpenHands \\
    --output_root /path/to/processed_graphs/phase_codeblock_openhands_v2 \\
    --workers 8 --html

# Process only specific model directories
python scripts/generate_phase_codeblock_graphs.py \\
    --graph_version v2 --html \\
    --model_dirs claude-sonnet-4_maxiter_100_N_v0.40.0-no-hint-run_1 \\
                 deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1

# Pass an eval report to include resolution_status
python scripts/generate_phase_codeblock_graphs.py \\
    --eval_report /path/to/report.json

Output structure
----------------
{output_root}/{model_dir}/{instance_id}/{instance_id}.json
{output_root}/{model_dir}/{instance_id}/{instance_id}.html  (when --html)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure graph_construction is importable when run from repo root or scripts/
_REPO_ROOT = Path(__file__).resolve().parent.parent
_GC_DIR = _REPO_ROOT / "graph_construction"
for _p in [str(_REPO_ROOT), str(_GC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phaseCodeBlockGraph import (
    build_phase_codeblock_graph_from_oh_trajectory,
    build_phase_codeblock_graph_v2_from_oh_trajectory,
    build_phase_codeblock_graph_v3_from_oh_trajectory,
    build_phase_codeblock_graph_v4_from_oh_trajectory,
    build_phase_codeblock_graph_v5_from_oh_trajectory,
    build_phase_codeblock_graph_v6_from_oh_trajectory,
)

_BUILDER_BY_VERSION = {
    "v2": build_phase_codeblock_graph_v2_from_oh_trajectory,
    "v3": build_phase_codeblock_graph_v3_from_oh_trajectory,
    "v4": build_phase_codeblock_graph_v4_from_oh_trajectory,
    "v5": build_phase_codeblock_graph_v5_from_oh_trajectory,
    "v6": build_phase_codeblock_graph_v6_from_oh_trajectory,
}
from plotly_renderer import render_graph_html

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Default paths ─────────────────────────────────────────────────────────────

DEFAULT_TRAJS_ROOT = "/home/yiboz7/data/raw_trajectories/OpenHands"
DEFAULT_OUTPUT_ROOT_V1 = "/home/yiboz7/data/processed_graphs/phase_codeblock_openhands"
DEFAULT_OUTPUT_ROOT_V2 = "/home/yiboz7/data/processed_graphs/phase_codeblock_openhands_v2"
DEFAULT_OUTPUT_ROOT_V3 = "/home/yiboz7/data/processed_graphs/phase_codeblock_openhands_v3"
DEFAULT_OUTPUT_ROOT_V4 = "/home/yiboz7/data/processed_graphs/phase_codeblock_openhands_v4"
DEFAULT_OUTPUT_ROOT_V5 = "/home/yiboz7/data/processed_graphs/phase_codeblock_openhands_v5"
DEFAULT_OUTPUT_ROOT_V6 = "/home/yiboz7/data/processed_graphs/phase_codeblock_openhands_v6"
_OUTPUT_ROOT_BY_VERSION = {
    "v1": DEFAULT_OUTPUT_ROOT_V1, "v2": DEFAULT_OUTPUT_ROOT_V2,
    "v3": DEFAULT_OUTPUT_ROOT_V3, "v4": DEFAULT_OUTPUT_ROOT_V4, "v5": DEFAULT_OUTPUT_ROOT_V5,
    "v6": DEFAULT_OUTPUT_ROOT_V6,
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ModelRunSpec:
    """One model-run directory to process."""
    model_dir_name: str       # e.g. "claude-sonnet-4_maxiter_100_N_v0.40.0-no-hint-run_1"
    jsonl_path: Path          # path to output.jsonl
    eval_report_path: Optional[Path]  # path to report.json (may be None)
    output_dir: Path          # where graphs for this model run are saved


@dataclass
class InstanceResult:
    """Result of processing one trajectory instance."""
    model_dir_name: str
    instance_id: str
    status: str               # "success" | "skip" | "error"
    json_path: Optional[str] = None
    html_path: Optional[str] = None
    reason: Optional[str] = None


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_model_runs(
    trajs_root: Path,
    output_root: Path,
    model_dir_filter: Optional[List[str]] = None,
    eval_report_path: Optional[Path] = None,
) -> List[ModelRunSpec]:
    """Discover all model-run directories that contain an output.jsonl."""
    if not trajs_root.is_dir():
        logger.error("Trajectories root not found: %s", trajs_root)
        sys.exit(1)

    specs: List[ModelRunSpec] = []
    for entry in sorted(trajs_root.iterdir()):
        if not entry.is_dir():
            continue
        if model_dir_filter and entry.name not in model_dir_filter:
            continue

        jsonl = entry / "output.jsonl"
        if not jsonl.exists():
            logger.warning("SKIP  %s  (no output.jsonl)", entry.name)
            continue

        # Prefer a report.json next to the jsonl (if no global report given)
        local_report = entry / "report.json"
        effective_report: Optional[Path] = None
        if eval_report_path and eval_report_path.exists():
            effective_report = eval_report_path
        elif local_report.exists():
            effective_report = local_report

        out_dir = output_root / entry.name
        specs.append(
            ModelRunSpec(
                model_dir_name=entry.name,
                jsonl_path=jsonl,
                eval_report_path=effective_report,
                output_dir=out_dir,
            )
        )

    return specs


# ── JSONL loading ─────────────────────────────────────────────────────────────

def load_trajectories(jsonl_path: Path) -> List[Tuple[str, Dict[str, Any], str]]:
    """Load all trajectory dicts from an output.jsonl.

    Returns:
        List of (instance_id, traj_data, skip_reason) tuples.
        skip_reason is '' for valid trajectories, non-empty for skipped ones.
    """
    results: List[Tuple[str, Dict[str, Any], str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                traj = json.loads(line)
            except json.JSONDecodeError as exc:
                results.append(("", {}, f"line {line_num}: JSON parse error: {exc}"))
                continue

            instance_id = traj.get("instance_id", "")
            if not instance_id:
                results.append(("", traj, f"line {line_num}: missing instance_id"))
                continue

            history = traj.get("history")
            if not isinstance(history, list):
                results.append((instance_id, traj, "history field missing or not a list"))
                continue

            results.append((instance_id, traj, ""))

    return results


# ── Per-instance processing ───────────────────────────────────────────────────

def _process_one(
    model_dir_name: str,
    instance_id: str,
    traj_data: Dict[str, Any],
    output_dir: str,
    eval_report_path: Optional[str],
    graph_version: str = "v1",
    generate_html: bool = False,
) -> InstanceResult:
    """Top-level worker function (safe to call in a subprocess)."""
    try:
        builder_fn = _BUILDER_BY_VERSION.get(graph_version, build_phase_codeblock_graph_from_oh_trajectory)
        json_path = builder_fn(
            traj_data=traj_data,
            instance_id=instance_id,
            output_dir=output_dir,
            eval_report_path=eval_report_path,
        )

        html_path: Optional[str] = None
        if generate_html and json_path:
            html_path = str(Path(json_path).with_suffix(".html"))
            render_graph_html(json_path, html_path)

        return InstanceResult(
            model_dir_name=model_dir_name,
            instance_id=instance_id,
            status="success",
            json_path=json_path,
            html_path=html_path,
        )
    except Exception as exc:
        return InstanceResult(
            model_dir_name=model_dir_name,
            instance_id=instance_id,
            status="error",
            reason=str(exc),
        )


def process_model_run(
    spec: ModelRunSpec,
    max_workers: int = 8,
    graph_version: str = "v1",
    generate_html: bool = False,
) -> Dict[str, List[InstanceResult]]:
    """Process all trajectories in one model-run directory."""
    logger.info("=" * 70)
    logger.info("Model run   : %s", spec.model_dir_name)
    logger.info("JSONL       : %s", spec.jsonl_path)
    logger.info("Output      : %s", spec.output_dir)
    logger.info("Report      : %s", spec.eval_report_path or "(none)")
    logger.info("Version     : %s", graph_version)
    logger.info("HTML output : %s", "yes" if generate_html else "no")

    spec.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_trajectories(spec.jsonl_path)
    logger.info("Loaded %d lines from JSONL", len(raw))

    stats: Dict[str, List[InstanceResult]] = {
        "success": [],
        "skip": [],
        "error": [],
    }

    # Separate valid from skipped-before-processing
    to_process: List[Tuple[str, Dict[str, Any]]] = []
    for instance_id, traj_data, skip_reason in raw:
        if skip_reason:
            r = InstanceResult(
                model_dir_name=spec.model_dir_name,
                instance_id=instance_id or "(unknown)",
                status="skip",
                reason=skip_reason,
            )
            stats["skip"].append(r)
            logger.debug("SKIP %s  reason=%s", instance_id, skip_reason)
        else:
            to_process.append((instance_id, traj_data))

    logger.info(
        "Processing %d instances (%d skipped before processing) with %d workers",
        len(to_process),
        len(stats["skip"]),
        max_workers,
    )

    eval_report_str = str(spec.eval_report_path) if spec.eval_report_path else None
    output_dir_str = str(spec.output_dir)

    t0 = time.perf_counter()
    completed = 0
    total = len(to_process)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _process_one,
                spec.model_dir_name,
                iid,
                tdata,
                output_dir_str,
                eval_report_str,
                graph_version,
                generate_html,
            ): iid
            for iid, tdata in to_process
        }
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result.status == "success":
                stats["success"].append(result)
                logger.info("[%d/%d] OK   %s", completed, total, result.instance_id)
            else:
                stats["error"].append(result)
                logger.warning(
                    "[%d/%d] ERR  %s  %s",
                    completed, total, result.instance_id, result.reason,
                )

    elapsed = time.perf_counter() - t0
    html_count = sum(1 for r in stats["success"] if r.html_path)
    logger.info(
        "Model run done in %.1fs — success=%d  error=%d  skip=%d  html=%d",
        elapsed,
        len(stats["success"]),
        len(stats["error"]),
        len(stats["skip"]),
        html_count,
    )
    return stats


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(
    all_stats: Dict[str, Dict[str, List[InstanceResult]]],
    output_root: Path,
    graph_version: str = "v1",
    generate_html: bool = False,
) -> None:
    """Print an aggregated processing summary across all model runs."""
    total_success = 0
    total_error = 0
    total_skip = 0
    total_html = 0

    print()
    print("=" * 70)
    print("PHASE-CODEBLOCK GRAPH GENERATION SUMMARY")
    print("=" * 70)
    print(f"  Graph version : {graph_version}")
    print(f"  HTML figures  : {'yes' if generate_html else 'no'}")
    print()

    for model_dir, stats in sorted(all_stats.items()):
        n_ok = len(stats["success"])
        n_err = len(stats["error"])
        n_skip = len(stats["skip"])
        n_html = sum(1 for r in stats["success"] if r.html_path)
        total_success += n_ok
        total_error += n_err
        total_skip += n_skip
        total_html += n_html
        print(f"  {model_dir}")
        print(f"    success={n_ok}  error={n_err}  skip={n_skip}  html={n_html}")
        if stats["error"]:
            for r in stats["error"][:5]:
                print(f"    ERR  {r.instance_id}: {r.reason}")
            if len(stats["error"]) > 5:
                print(f"    ... and {len(stats['error']) - 5} more errors")
        if stats["skip"]:
            for r in stats["skip"][:3]:
                print(f"    SKIP {r.instance_id}: {r.reason}")
            if len(stats["skip"]) > 3:
                print(f"    ... and {len(stats['skip']) - 3} more skipped")
        print()

    print("-" * 70)
    print(f"  TOTAL  trajectories processed : {total_success + total_error + total_skip}")
    print(f"  TOTAL  graphs generated       : {total_success}")
    print(f"  TOTAL  HTML figures           : {total_html}")
    print(f"  TOTAL  failed trajectories    : {total_error}")
    print(f"  TOTAL  skipped trajectories   : {total_skip}")
    print(f"  Output root                   : {output_root}")
    print("=" * 70)
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate phase-centric + code-block graphs for OpenHands trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--trajs_root",
        type=str,
        default=DEFAULT_TRAJS_ROOT,
        help=f"Root directory containing model-run subdirs (default: {DEFAULT_TRAJS_ROOT})",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Root directory for output graphs. "
            f"Defaults to {DEFAULT_OUTPUT_ROOT_V1} for v1 and {DEFAULT_OUTPUT_ROOT_V2} for v2."
        ),
    )
    parser.add_argument(
        "--graph_version",
        type=str,
        default="v1",
        choices=["v1", "v2", "v3", "v4", "v5", "v6"],
        help=(
            "Graph variant to generate. "
            "v1: full graph (line-range code-block nodes). "
            "v2: modified-file nodes only, no line numbers. "
            "v3: clean full (v2 minus cognitive/IO/latency, raw filename, termination feature fields). "
            "v4: FS-pruned (selected phase fields, structural code_block nodes, no op-edge attrs). "
            "v5: phase-only (no code_block nodes / phase_code_operation edges). "
            "v6: same schema as v2 but with the #39 corrected phase labels (edit-op => patch)."
        ),
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help=(
            "Also generate interactive Plotly HTML figures alongside each graph JSON. "
            "Each HTML is saved as {instance_id}.html next to the JSON file."
        ),
    )
    parser.add_argument(
        "--model_dirs",
        nargs="*",
        default=None,
        help=(
            "Restrict to specific model-run directory names. "
            "If omitted, all subdirs with an output.jsonl are processed."
        ),
    )
    parser.add_argument(
        "--eval_report",
        type=str,
        default=None,
        help=(
            "Path to a global report.json for resolution_status lookup. "
            "Falls back to report.json inside each model-run directory."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker processes per model run (default: 8)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    trajs_root = Path(args.trajs_root)
    graph_version: str = args.graph_version
    generate_html: bool = args.html

    # Resolve output root: explicit flag > version-specific default
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = Path(_OUTPUT_ROOT_BY_VERSION.get(graph_version, DEFAULT_OUTPUT_ROOT_V1))

    eval_report = Path(args.eval_report) if args.eval_report else None

    if eval_report and not eval_report.exists():
        logger.error("--eval_report path does not exist: %s", eval_report)
        sys.exit(1)

    logger.info("Graph version : %s", graph_version)
    logger.info("Output root   : %s", output_root)
    logger.info("HTML figures  : %s", "yes" if generate_html else "no")

    specs = discover_model_runs(
        trajs_root=trajs_root,
        output_root=output_root,
        model_dir_filter=args.model_dirs,
        eval_report_path=eval_report,
    )

    if not specs:
        logger.error("No model-run directories found under %s", trajs_root)
        sys.exit(1)

    logger.info("Found %d model-run director%s to process", len(specs), "y" if len(specs) == 1 else "ies")

    all_stats: Dict[str, Dict[str, List[InstanceResult]]] = {}

    overall_t0 = time.perf_counter()
    for spec in specs:
        stats = process_model_run(
            spec,
            max_workers=args.workers,
            graph_version=graph_version,
            generate_html=generate_html,
        )
        all_stats[spec.model_dir_name] = stats

    overall_elapsed = time.perf_counter() - overall_t0
    logger.info("All model runs completed in %.1fs", overall_elapsed)

    print_summary(all_stats, output_root, graph_version=graph_version, generate_html=generate_html)

    # Exit with non-zero if any errors occurred
    total_errors = sum(len(s["error"]) for s in all_stats.values())
    sys.exit(1 if total_errors else 0)


if __name__ == "__main__":
    main()
