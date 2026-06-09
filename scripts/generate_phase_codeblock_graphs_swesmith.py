#!/usr/bin/env python3
"""
Generate phase-codeblock v2 graphs for the SWE-smith / SWE-agent trajectory
dataset (``SWE-bench/SWE-smith-trajectories``).

The dataset ships three prompt/parse variants as parquet splits — ``tool``,
``xml``, ``ticks`` — plus a ``train`` split that is byte-identical to ``xml``
(same trajectories, ``messages`` re-encoded as a structured list).  ``train`` is
therefore **excluded** to avoid duplicating the xml trajectories.

Each variant is treated as a distinct **model group** (the prompt-format is the
model-heldout key for the outcome-prediction study; the underlying LLM is a mix
of Claude 3.5 / Claude 3.7 / GPT-4o across all formats).  Group slugs:

    swesmith-tool  →  output dir  swesmith-tool_swesmith/
    swesmith-xml   →  output dir  swesmith-xml_swesmith/
    swesmith-ticks →  output dir  swesmith-ticks_swesmith/

Per-graph identity is the row's unique ``traj_id`` (instance_id is NOT unique —
SWE-smith records multiple rollouts per instance).  Duplicate ``traj_id`` rows
within a variant (byte-identical; xml only) are de-duplicated, keeping the first.

Layout
------
    {output_root}/{MODEL}_swesmith/{traj_id}/{traj_id}.json

Usage
-----
    HF_HOME=/srv/local/scratch/yiboz7/hf_cache \
    /home/yiboz7/code/Graphectory/.venv/bin/python \
        scripts/generate_phase_codeblock_graphs_swesmith.py \
        --raw_root  /home/yiboz7/data/raw_trajectories/swesmith/data \
        --output_root /home/yiboz7/data/processed_graphs/phase_codeblock_swesmith_v2 \
        --workers 16
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GC_DIR = _REPO_ROOT / "graph_construction"
for _p in (str(_REPO_ROOT), str(_GC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from swesmithBuilder import (  # noqa: E402
    build_phase_codeblock_graph_v2_from_swesmith_trajectory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_RAW_ROOT = "/home/yiboz7/data/raw_trajectories/swesmith/data"
DEFAULT_OUTPUT_ROOT = "/home/yiboz7/data/processed_graphs/phase_codeblock_swesmith_v2"

# Prompt-format variants to build (train == xml, excluded).
VARIANTS = ("tool", "xml", "ticks")
MODEL_SLUG = {"tool": "swesmith-tool", "xml": "swesmith-xml", "ticks": "swesmith-ticks"}


@dataclass
class Result:
    variant: str
    traj_id: str
    status: str            # "success" | "skip" | "error"
    json_path: Optional[str] = None
    reason: Optional[str] = None


def _process_one(
    messages_json: str,
    instance_id: str,
    traj_id: str,
    resolved: Optional[bool],
    model: Optional[str],
    patch: Optional[str],
    variant: str,
    output_dir: str,
) -> Result:
    """Worker: parse messages JSON and build one graph (safe in subprocess)."""
    try:
        messages = json.loads(messages_json)
        if not isinstance(messages, list) or not messages:
            return Result(variant, traj_id, "skip", reason="empty/invalid messages")
        # Need at least one assistant turn to have any phase.
        if not any((m or {}).get("role") == "assistant" for m in messages):
            return Result(variant, traj_id, "skip", reason="no assistant messages")
        json_path = build_phase_codeblock_graph_v2_from_swesmith_trajectory(
            messages=messages,
            instance_id=traj_id,            # per-graph id = unique traj_id
            output_dir=output_dir,
            resolved=resolved,
            model=model,
            patch=patch,
            variant=variant,
        )
        return Result(variant, traj_id, "success", json_path=json_path)
    except Exception as exc:  # noqa: BLE001
        return Result(variant, traj_id, "error", reason=f"{type(exc).__name__}: {exc}")


def iter_rows(raw_root: Path, variant: str):
    """Yield (instance_id, traj_id, resolved, model, patch, messages_json) per row.

    De-duplicates byte-identical duplicate traj_ids within the variant.
    """
    files = sorted(glob.glob(str(raw_root / f"{variant}-*.parquet")))
    seen: set = set()
    for f in files:
        tbl = pq.read_table(
            f, columns=["messages", "instance_id", "resolved", "model", "traj_id", "patch"]
        )
        msgs = tbl.column("messages").to_pylist()
        iids = tbl.column("instance_id").to_pylist()
        reso = tbl.column("resolved").to_pylist()
        mdl = tbl.column("model").to_pylist()
        traj = tbl.column("traj_id").to_pylist()
        patch = tbl.column("patch").to_pylist()
        for i in range(len(msgs)):
            tid = traj[i]
            if tid in seen:
                continue
            seen.add(tid)
            yield iids[i], tid, reso[i], mdl[i], patch[i], msgs[i]


def process_variant(
    raw_root: Path, output_root: Path, variant: str, workers: int, limit: Optional[int]
) -> List[Result]:
    model_slug = MODEL_SLUG[variant]
    out_dir = output_root / f"{model_slug}_swesmith"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Variant     : %s  (group slug: %s)", variant, model_slug)
    logger.info("Output dir  : %s", out_dir)

    rows = list(iter_rows(raw_root, variant))
    if limit:
        rows = rows[:limit]
    logger.info("Rows (deduped by traj_id): %d", len(rows))

    results: List[Result] = []
    t0 = time.perf_counter()
    completed = 0
    total = len(rows)
    out_dir_str = str(out_dir)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                _process_one,
                msgs_json, iid, tid, resolved, model, patch, variant, out_dir_str,
            ): tid
            for (iid, tid, resolved, model, patch, msgs_json) in rows
        }
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            completed += 1
            if r.status == "error":
                logger.warning("[%d/%d] ERR  %s  %s", completed, total, r.traj_id, r.reason)
            if completed % 1000 == 0:
                logger.info("[%d/%d] %s …", completed, total, variant)

    n_ok = sum(1 for r in results if r.status == "success")
    n_skip = sum(1 for r in results if r.status == "skip")
    n_err = sum(1 for r in results if r.status == "error")
    logger.info(
        "Variant %s done in %.1fs — success=%d skip=%d error=%d",
        variant, time.perf_counter() - t0, n_ok, n_skip, n_err,
    )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    ap.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="Cap rows per variant (debug)")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, List[Result]] = {}
    overall_t0 = time.perf_counter()
    for variant in args.variants:
        all_results[variant] = process_variant(
            raw_root, output_root, variant, args.workers, args.limit
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SWE-SMITH PHASE-CODEBLOCK v2 GRAPH GENERATION SUMMARY")
    print("=" * 70)
    g_ok = g_skip = g_err = 0
    for variant in args.variants:
        rs = all_results[variant]
        n_ok = sum(1 for r in rs if r.status == "success")
        n_skip = sum(1 for r in rs if r.status == "skip")
        n_err = sum(1 for r in rs if r.status == "error")
        g_ok += n_ok; g_skip += n_skip; g_err += n_err
        print(f"  {MODEL_SLUG[variant]:18s} success={n_ok}  skip={n_skip}  error={n_err}")
        for r in [r for r in rs if r.status == "error"][:5]:
            print(f"      ERR  {r.traj_id}: {r.reason}")
        for r in [r for r in rs if r.status == "skip"][:3]:
            print(f"      SKIP {r.traj_id}: {r.reason}")
    print("-" * 70)
    print(f"  TOTAL success={g_ok}  skip={g_skip}  error={g_err}")
    print(f"  Output root: {output_root}")
    print(f"  Elapsed    : {time.perf_counter() - overall_t0:.1f}s")
    print("=" * 70)

    sys.exit(1 if g_err else 0)


if __name__ == "__main__":
    main()
