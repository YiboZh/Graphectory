#!/usr/bin/env python3
"""
Generate phase-codeblock **v2** graphs for the SWE-rebench-OpenHands dataset.

Reads ``nebius/SWE-rebench-openhands-trajectories`` (a single parquet of
OpenAI-style chat trajectories), adapts each row to the OpenHands ``traj_data``
shape, and builds a v2 phase-codeblock graph that is feature-identical to the
reference OpenHands v2 graphs (see :mod:`swerebenchBuilder`).

The dataset uses a SINGLE bootstrapping model
(``Qwen/Qwen3-Coder-480B-A35B-Instruct``), so every graph lands under one model
slug (``qwen3-coder-480b``).  Because each SWE-rebench ``instance_id`` is run
multiple times (≈10×), graphs are keyed by a UNIQUE id
``{instance_id}__{traj_short}`` to avoid collisions, while the original
``instance_id`` and ``repo_id`` are preserved for the project-safe split.

Output layout
-------------
    {output_root}/{MODEL}_swerebench/{unique_id}/{unique_id}.json

A project-safe split CSV is written to
``{output_root}/split_projectsafe.csv`` (GroupShuffleSplit on repo_id,
test_size=0.2, random_state=42).

Usage
-----
    python scripts/generate_phase_codeblock_graphs_swerebench.py \
        --parquet /home/yiboz7/data/raw_trajectories/swerebench_openhands/trajectories.parquet \
        --output_root /home/yiboz7/data/processed_graphs/phase_codeblock_swerebench_openhands_v2 \
        --workers 16
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GC_DIR = _REPO_ROOT / "graph_construction"
for _p in (str(_REPO_ROOT), str(_GC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from swerebenchBuilder import build_phase_codeblock_graph_v2_from_swerebench_trajectory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Single model for the whole dataset (per dataset README / config.toml).
MODEL_SLUG = "qwen3-coder-480b"

DEFAULT_PARQUET = "/home/yiboz7/data/raw_trajectories/swerebench_openhands/trajectories.parquet"
DEFAULT_OUTPUT_ROOT = "/home/yiboz7/data/processed_graphs/phase_codeblock_swerebench_openhands_v2"

# Split / CSV constants (fixed by the task contract).
RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_PREFIX_K = 40
BENCHMARK_SETTING = "full_benchmark"
MIN_TRAJ_STEPS = 0
SPLIT_STRATEGY = "projectsafe_by_repo_id"

_TRAILING_NUM_RE = re.compile(r"-\d+$")

# Columns we actually need from the parquet (avoid loading the huge `tools` col).
_NEEDED_COLS = [
    "trajectory_id",
    "instance_id",
    "repo",
    "trajectory",
    "model_patch",
    "exit_status",
    "resolved",
]


def _traj_short(trajectory_id: str) -> str:
    """A short, filesystem-safe suffix from the unique trajectory_id."""
    t = trajectory_id.replace("chatcmpl-", "")
    return re.sub(r"[^A-Za-z0-9]", "", t)[:16]


def _repo_id_from(instance_id: str, repo: Optional[str]) -> str:
    """repo_id = instance_id with a trailing ``-<number>`` stripped.

    Falls back to the ``repo`` column (owner/name → owner__name) when available,
    which is equivalent and more robust.
    """
    if repo:
        return repo.replace("/", "__")
    return _TRAILING_NUM_RE.sub("", instance_id)


# ── Worker ────────────────────────────────────────────────────────────────────

def _process_one(
    row: Dict[str, Any],
    unique_id: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Build one graph; return a result dict (safe for subprocess)."""
    try:
        json_path = build_phase_codeblock_graph_v2_from_swerebench_trajectory(
            row=row,
            instance_id=unique_id,
            output_dir=output_dir,
            resolved=row.get("resolved"),
        )
        # Read back termination_type + total_steps for the split CSV.
        with open(json_path, "r") as fh:
            g = json.load(fh)
        term_type = ""
        total_steps = 0
        for n in g["nodes"]:
            if n.get("node_type") == "termination":
                term_type = n.get("termination_type", "")
                total_steps = int(n.get("final_step_idx", 0)) + 1
                break
        resolution = g.get("graph", {}).get("resolution_status", "unknown")
        return {
            "status": "success",
            "unique_id": unique_id,
            "json_path": json_path,
            "termination_type": term_type,
            "total_steps": total_steps,
            "resolution_status": resolution,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "unique_id": unique_id, "reason": repr(exc)}


# ── GroupShuffleSplit (sklearn-equivalent) ────────────────────────────────────

def group_shuffle_split(
    groups: List[str], test_size: float, random_state: int
) -> Dict[str, str]:
    """Replicate ``sklearn.model_selection.GroupShuffleSplit`` (1 split).

    Returns a mapping ``group -> "train"|"test"``.  Mirrors sklearn: shuffle the
    *unique* groups with ``RandomState(random_state).permutation`` and take the
    first ``ceil(test_size * n_groups)`` as the test groups.
    """
    classes, _ = np.unique(groups, return_inverse=True)
    n_groups = len(classes)
    n_test = int(math.ceil(test_size * n_groups))
    rng = np.random.RandomState(random_state)
    permutation = rng.permutation(n_groups)
    test_group_idx = permutation[:n_test]
    test_groups = set(classes[test_group_idx].tolist())
    return {str(g): ("test" if g in test_groups else "train") for g in classes.tolist()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="Cap rows (0 = all; for debugging)")
    ap.add_argument("--batch_size", type=int, default=2000, help="Parquet read batch size")
    args = ap.parse_args()

    parquet = Path(args.parquet)
    output_root = Path(args.output_root)
    model_dir = output_root / f"{MODEL_SLUG}_swerebench"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(model_dir)

    if not parquet.exists():
        logger.error("Parquet not found: %s", parquet)
        sys.exit(1)

    pf = pq.ParquetFile(str(parquet))
    total_rows = pf.metadata.num_rows
    logger.info("Parquet: %s  rows=%d  model_slug=%s", parquet, total_rows, MODEL_SLUG)
    logger.info("Output dir: %s  workers=%d", output_dir_str, args.workers)

    # Per-graph metadata accumulated for the split CSV.
    csv_rows: List[Dict[str, Any]] = []
    n_success = 0
    n_error = 0
    errors: List[Tuple[str, str]] = []
    seen_unique: set[str] = set()

    t0 = time.perf_counter()
    processed = 0
    stop = False

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=_NEEDED_COLS):
            rows = batch.to_pylist()
            futures = {}
            for r in rows:
                if args.limit and processed >= args.limit:
                    stop = True
                    break
                processed += 1
                iid = r.get("instance_id", "") or "unknown"
                tid = r.get("trajectory_id", "") or str(processed)
                unique_id = f"{iid}__{_traj_short(tid)}"
                # Guard against the (extremely unlikely) collision of short ids.
                if unique_id in seen_unique:
                    unique_id = f"{iid}__{_traj_short(tid)}_{processed}"
                seen_unique.add(unique_id)
                repo_id = _repo_id_from(iid, r.get("repo"))
                meta = {
                    "instance_id": iid,
                    "unique_id": unique_id,
                    "repo_id": repo_id,
                    "resolved": int(r.get("resolved") or 0),
                }
                fut = pool.submit(_process_one, r, unique_id, output_dir_str)
                futures[fut] = meta

            for fut in as_completed(futures):
                meta = futures[fut]
                res = fut.result()
                if res["status"] == "success":
                    n_success += 1
                    label = 1 if res["resolution_status"] == "resolved" else 0
                    csv_rows.append(
                        {
                            "example_id": f"{MODEL_SLUG}:{meta['unique_id']}",
                            "model": MODEL_SLUG,
                            "instance_id": meta["unique_id"],
                            "repo_id": meta["repo_id"],
                            "label": label,
                            "total_steps": res["total_steps"],
                            "termination_type": res["termination_type"],
                            "source_path": str(parquet),
                        }
                    )
                else:
                    n_error += 1
                    if len(errors) < 50:
                        errors.append((res["unique_id"], res.get("reason", "")))

            done = n_success + n_error
            if done and done % args.batch_size < len(rows):
                logger.info(
                    "[%d/%d] success=%d error=%d  (%.0fs)",
                    done, total_rows if not args.limit else args.limit,
                    n_success, n_error, time.perf_counter() - t0,
                )
            if stop:
                break

    elapsed = time.perf_counter() - t0
    logger.info(
        "Build done in %.1fs — success=%d error=%d (of %d processed)",
        elapsed, n_success, n_error, processed,
    )
    if errors:
        logger.warning("First errors:")
        for uid, reason in errors[:10]:
            logger.warning("  %s : %s", uid, reason)

    # ── Project-safe split ───────────────────────────────────────────────────
    repo_ids = [r["repo_id"] for r in csv_rows]
    split_map = group_shuffle_split(repo_ids, TEST_SIZE, RANDOM_SEED)
    for r in csv_rows:
        r["split"] = split_map[r["repo_id"]]

    # ── Write split CSV ──────────────────────────────────────────────────────
    csv_path = output_root / "split_projectsafe.csv"
    fieldnames = [
        "split", "example_id", "model", "instance_id", "repo_id", "label",
        "total_steps", "source_path", "random_seed", "test_size", "max_prefix_k",
        "benchmark_setting", "min_trajectory_steps", "split_strategy", "termination_type",
    ]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in csv_rows:
            w.writerow(
                {
                    "split": r["split"],
                    "example_id": r["example_id"],
                    "model": r["model"],
                    "instance_id": r["instance_id"],
                    "repo_id": r["repo_id"],
                    "label": r["label"],
                    "total_steps": r["total_steps"],
                    "source_path": r["source_path"],
                    "random_seed": RANDOM_SEED,
                    "test_size": TEST_SIZE,
                    "max_prefix_k": MAX_PREFIX_K,
                    "benchmark_setting": BENCHMARK_SETTING,
                    "min_trajectory_steps": MIN_TRAJ_STEPS,
                    "split_strategy": SPLIT_STRATEGY,
                    "termination_type": r["termination_type"],
                }
            )

    n_train = sum(1 for r in csv_rows if r["split"] == "train")
    n_test = sum(1 for r in csv_rows if r["split"] == "test")
    n_pos = sum(1 for r in csv_rows if r["label"] == 1)
    logger.info("Split CSV written: %s", csv_path)
    logger.info("  rows=%d  train=%d  test=%d", len(csv_rows), n_train, n_test)
    logger.info(
        "  label balance: positive=%d (%.1f%%)  negative=%d",
        n_pos, 100.0 * n_pos / len(csv_rows) if csv_rows else 0.0, len(csv_rows) - n_pos,
    )

    # Machine-readable summary for the report step.
    summary = {
        "total_rows": total_rows,
        "processed": processed,
        "built": n_success,
        "failed": n_error,
        "train": n_train,
        "test": n_test,
        "positive": n_pos,
        "negative": len(csv_rows) - n_pos,
        "csv_path": str(csv_path),
        "output_dir": output_dir_str,
        "model_slug": MODEL_SLUG,
        "elapsed_sec": round(elapsed, 1),
        "error_samples": errors[:10],
    }
    (output_root / "_build_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Summary written: %s", output_root / "_build_summary.json")


if __name__ == "__main__":
    main()
