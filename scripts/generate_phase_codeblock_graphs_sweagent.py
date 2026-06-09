#!/usr/bin/env python3
"""
Generate phase-codeblock **v2** graphs for the SWE-agent (nebius/SWE-agent-trajectories)
dataset, with the IDENTICAL node/edge feature schema as the OpenHands v2 graphs.

Each parquet row is a single SWE-agent rollout.  The dataset contains MANY rollouts
per (model, instance_id) pair (multi-attempt sampling), so every row is a distinct
trajectory.  To keep them all on disk without collision, each row gets a unique
on-disk instance id ``{instance_id}__a{n}`` (n = attempt index within its
(model, instance_id) group).  ``repo_id`` for project-safe splitting is derived from
the ORIGINAL instance id (strip the ``__a<n>`` suffix, then strip trailing ``-<num>``).

Output layout (matches the downstream driver's ``model_dir.name.startswith(model+"_")``):
    {output_root}/{model}_sweagent/{unique_instance_id}/{unique_instance_id}.json

Also emits a project-safe split CSV at
    {output_root}/split_projectsafe.csv

Usage
-----
    python scripts/generate_phase_codeblock_graphs_sweagent.py \
        --raw_dir   /home/yiboz7/data/raw_trajectories/sweagent_nebius \
        --output_root /home/yiboz7/data/processed_graphs/phase_codeblock_sweagent_nebius_v2 \
        --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GC_DIR = _REPO_ROOT / "graph_construction"
for _p in (str(_REPO_ROOT), str(_GC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sweagentBuilder import build_phase_codeblock_graph_v2_from_sweagent_trajectory  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = "/home/yiboz7/data/raw_trajectories/sweagent_nebius"
DEFAULT_OUTPUT_ROOT = "/home/yiboz7/data/processed_graphs/phase_codeblock_sweagent_nebius_v2"

# Split CSV constants (per task contract)
RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_PREFIX_K = 40
BENCHMARK_SETTING = "full_benchmark"
MIN_TRAJECTORY_STEPS = 0
SPLIT_STRATEGY = "projectsafe_by_repo_id"

_ATTEMPT_SUFFIX_RE = re.compile(r"__a\d+$")
_TRAILING_NUM_RE = re.compile(r"-\d+$")


def repo_id_of(orig_instance_id: str) -> str:
    """repo_id = original instance_id minus trailing ``-<num>``."""
    return _TRAILING_NUM_RE.sub("", orig_instance_id)


def _is_na(x: Any) -> bool:
    try:
        import math

        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return x is None


def _row_to_dict(row: pd.Series) -> Dict[str, Any]:
    return {k: row[k] for k in row.index}


# ── Per-row worker ────────────────────────────────────────────────────────────


def _process_one(
    row_dict: Dict[str, Any],
    model: str,
    unique_iid: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Build one graph; return a result record (also used to fill the split CSV)."""
    try:
        json_path = build_phase_codeblock_graph_v2_from_sweagent_trajectory(
            row=row_dict,
            output_dir=output_dir,
            instance_id=unique_iid,
        )
        g = json.loads(Path(json_path).read_text())
        term = next(
            (n for n in g["nodes"] if n.get("node_type") == "termination"), {}
        )
        term_type = term.get("termination_type", "unknown")
        final_step_idx = term.get("final_step_idx", 0)
        total_steps = int(final_step_idx) + 1 if g.get("nodes") else 0
        res = g.get("graph", {}).get("resolution_status", "unknown")
        return {
            "status": "success",
            "json_path": json_path,
            "unique_iid": unique_iid,
            "model": model,
            "termination_type": term_type,
            "total_steps": total_steps,
            "resolution_status": res,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "status": "error",
            "unique_iid": unique_iid,
            "model": model,
            "reason": f"{type(exc).__name__}: {exc}",
        }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", default=DEFAULT_RAW_DIR)
    ap.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap number of rows (0 = all)")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    parquets = sorted(raw_dir.glob("*.parquet"))
    if not parquets:
        logger.error("No parquet files under %s", raw_dir)
        sys.exit(1)
    logger.info("Found %d parquet shards", len(parquets))

    # ── Pass 1: assemble all rows + assign unique attempt-indexed instance ids ─
    tasks: List[Tuple[Dict[str, Any], str, str, str, str]] = []
    # (row_dict, model, unique_iid, orig_iid, source_path)
    attempt_counter: Dict[Tuple[str, str], int] = defaultdict(int)
    n_rows = 0
    n_skip = 0
    skip_reasons: Counter = Counter()

    for pq in parquets:
        df = pd.read_parquet(pq)
        for _, row in df.iterrows():
            n_rows += 1
            orig_iid = "" if _is_na(row.get("instance_id")) else str(row["instance_id"])
            model = "" if _is_na(row.get("model_name")) else str(row["model_name"])
            traj = row.get("trajectory")
            if not orig_iid or not model:
                n_skip += 1
                skip_reasons["missing instance_id/model_name"] += 1
                continue
            if traj is None or len(traj) == 0:
                n_skip += 1
                skip_reasons["empty trajectory"] += 1
                continue
            # any ai step?
            has_ai = any(getattr(st, "get", lambda k: None)("role") == "ai" for st in traj)
            if not has_ai:
                n_skip += 1
                skip_reasons["no ai/action steps"] += 1
                continue
            k = (model, orig_iid)
            idx = attempt_counter[k]
            attempt_counter[k] += 1
            unique_iid = f"{orig_iid}__a{idx}"
            tasks.append((_row_to_dict(row), model, unique_iid, orig_iid, str(pq)))
            if args.limit and len(tasks) >= args.limit:
                break
        if args.limit and len(tasks) >= args.limit:
            break

    logger.info("Rows=%d  buildable=%d  pre-skip=%d  %s",
                n_rows, len(tasks), n_skip, dict(skip_reasons))

    # ── Pass 2: build graphs (multiprocessing) ────────────────────────────────
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    total = len(tasks)
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for row_dict, model, unique_iid, orig_iid, src in tasks:
            out_dir = str(output_root / f"{model}_sweagent")
            fut = pool.submit(_process_one, row_dict, model, unique_iid, out_dir)
            futures[fut] = (orig_iid, src)
        for fut in as_completed(futures):
            orig_iid, src = futures[fut]
            r = fut.result()
            completed += 1
            if r["status"] == "success":
                r["orig_iid"] = orig_iid
                r["source_path"] = src
                results.append(r)
            else:
                errors.append(r)
                logger.warning("[%d/%d] ERR %s : %s", completed, total, r["unique_iid"], r.get("reason"))
            if completed % 2000 == 0:
                logger.info("[%d/%d] built (%.0fs)", completed, total, time.perf_counter() - t0)

    elapsed = time.perf_counter() - t0
    logger.info("Build done in %.0fs  success=%d  error=%d  skip=%d",
                elapsed, len(results), len(errors), n_skip)

    # ── Split CSV (project-safe GroupShuffleSplit by repo_id) ─────────────────
    write_split_csv(results, output_root)

    # ── Label / model summary ─────────────────────────────────────────────────
    print_summary(results, errors, n_skip, skip_reasons, output_root)


def _group_shuffle_split(groups, test_size: float, random_state: int):
    """Faithful reproduction of sklearn's GroupShuffleSplit(n_splits=1).

    Mirrors sklearn: permute the unique groups with RandomState(random_state),
    take the first ceil(test_size * n_groups) groups as the test set.  Avoids a
    hard sklearn dependency (not installed in this venv).
    """
    import numpy as np

    groups = np.asarray(groups)
    _classes, group_indices = np.unique(groups, return_inverse=True)
    n_groups = _classes.shape[0]
    n_test = int(np.ceil(test_size * n_groups))
    n_train = n_groups - n_test
    rng = np.random.RandomState(random_state)
    permutation = rng.permutation(n_groups)
    ind_test = permutation[:n_test]
    ind_train = permutation[n_test:n_test + n_train]
    train = np.flatnonzero(np.isin(group_indices, ind_train))
    test = np.flatnonzero(np.isin(group_indices, ind_test))
    return train, test


def write_split_csv(results: List[Dict[str, Any]], output_root: Path) -> None:
    rows = []
    for r in results:
        model = r["model"]
        unique_iid = r["unique_iid"]
        orig_iid = r["orig_iid"]
        rid = repo_id_of(orig_iid)
        label = 1 if r["resolution_status"] == "resolved" else 0
        rows.append({
            "example_id": f"{model}:{unique_iid}",
            "model": model,
            "instance_id": unique_iid,
            "repo_id": rid,
            "label": label,
            "total_steps": r["total_steps"],
            "source_path": r["source_path"],
            "termination_type": r["termination_type"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No successful graphs; split CSV will be empty.")
    else:
        train_idx, test_idx = _group_shuffle_split(
            df["repo_id"].values, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
        df["split"] = "train"
        df.iloc[test_idx, df.columns.get_loc("split")] = "test"

    # constants
    df["random_seed"] = RANDOM_SEED
    df["test_size"] = TEST_SIZE
    df["max_prefix_k"] = MAX_PREFIX_K
    df["benchmark_setting"] = BENCHMARK_SETTING
    df["min_trajectory_steps"] = MIN_TRAJECTORY_STEPS
    df["split_strategy"] = SPLIT_STRATEGY

    cols = [
        "split", "example_id", "model", "instance_id", "repo_id", "label",
        "total_steps", "source_path", "random_seed", "test_size", "max_prefix_k",
        "benchmark_setting", "min_trajectory_steps", "split_strategy", "termination_type",
    ]
    df = df[cols]
    out = output_root / "split_projectsafe.csv"
    df.to_csv(out, index=False)
    n_train = int((df["split"] == "train").sum())
    n_test = int((df["split"] == "test").sum())
    logger.info("Wrote split CSV %s  train=%d  test=%d", out, n_train, n_test)


def print_summary(results, errors, n_skip, skip_reasons, output_root) -> None:
    by_model = defaultdict(lambda: {"n": 0, "resolved": 0})
    overall_resolved = 0
    for r in results:
        m = r["model"]
        by_model[m]["n"] += 1
        if r["resolution_status"] == "resolved":
            by_model[m]["resolved"] += 1
            overall_resolved += 1
    n = len(results)
    print("\n" + "=" * 70)
    print("SWE-AGENT (nebius) PHASE-CODEBLOCK v2 GRAPH BUILD SUMMARY")
    print("=" * 70)
    print(f"  output_root : {output_root}")
    print(f"  built       : {n}")
    print(f"  failed      : {len(errors)}")
    print(f"  skipped     : {n_skip}  {dict(skip_reasons)}")
    if n:
        print(f"  label balance overall : resolved={overall_resolved} ({overall_resolved/n:.1%})  "
              f"unresolved={n-overall_resolved} ({(n-overall_resolved)/n:.1%})")
    print("  per-model:")
    for m, d in sorted(by_model.items()):
        res = d["resolved"]; tot = d["n"]
        print(f"    {m}_sweagent : n={tot}  resolved={res} ({res/tot:.1%})  unresolved={tot-res}")
    if errors:
        print("  first errors:")
        for e in errors[:10]:
            print(f"    {e['unique_iid']}: {e.get('reason')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
