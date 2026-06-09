#!/usr/bin/env python3
"""Build split_projectsafe.csv for the SWE-smith v2 graphs.

GroupShuffleSplit(group=repo_id, test_size=0.2, random_state=42), implemented to
match sklearn's algorithm (seeded permutation of unique groups, first ~test_size
fraction of *samples* assigned to test by group). repo_id = SWE-smith owner__repo
prefix (substring before the first '.'); SWE-smith instance_ids carry no trailing
'-<num>', so the rule 'instance_id minus trailing -<num>' is a no-op there and we
fall back to the actual source-repo prefix for project-safe grouping.
"""
from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

OUT_ROOT = Path("/home/yiboz7/data/processed_graphs/phase_codeblock_swesmith_v2")
RAW_ROOT = Path("/home/yiboz7/data/raw_trajectories/swesmith/data")
CSV_PATH = OUT_ROOT / "split_projectsafe.csv"

MODEL_FOR_DIR = {
    "swesmith-tool_swesmith": "swesmith-tool",
    "swesmith-xml_swesmith": "swesmith-xml",
    "swesmith-ticks_swesmith": "swesmith-ticks",
}
VARIANT_FOR_MODEL = {"swesmith-tool": "tool", "swesmith-xml": "xml", "swesmith-ticks": "ticks"}

RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_PREFIX_K = 40
BENCHMARK_SETTING = "full_benchmark"
MIN_TRAJ_STEPS = 0
SPLIT_STRATEGY = "projectsafe_by_repo_id"

_TRAIL_NUM = re.compile(r"-\d+$")


def repo_id_from_instance(instance_id: str) -> str:
    """repo_id = instance_id minus trailing -<num>; for SWE-smith ids (no such
    suffix) this yields the owner__repo source-repo prefix before the first '.'."""
    stripped = _TRAIL_NUM.sub("", instance_id)
    return stripped.split(".")[0]


def graph_meta(path: Path):
    g = json.loads(path.read_text())
    gg = g.get("graph", {})
    res = gg.get("resolution_status", "unknown")
    label = 1 if res == "resolved" else 0
    term_type = "unknown"
    total_steps = 0
    for n in g["nodes"]:
        if n.get("node_type") == "termination":
            term_type = n.get("termination_type", "unknown")
            total_steps = int(n.get("final_step_idx", 0))
    return label, term_type, total_steps


def main() -> None:
    rows = []
    for dir_name, model in MODEL_FOR_DIR.items():
        variant = VARIANT_FOR_MODEL[model]
        gdir = OUT_ROOT / dir_name
        files = sorted(gdir.glob("*/*.json"))
        files = [f for f in files if f.parent.name == f.stem]
        for f in files:
            traj_id = f.stem
            label, term_type, total_steps = graph_meta(f)
            repo_id = repo_id_from_instance(traj_id)
            rows.append(
                dict(
                    example_id=f"{model}:{traj_id}",
                    model=model,
                    instance_id=traj_id,
                    repo_id=repo_id,
                    label=label,
                    total_steps=total_steps,
                    source_path=str(RAW_ROOT / f"{variant}-*.parquet"),
                    termination_type=term_type,
                )
            )

    n = len(rows)
    print(f"Collected {n} graph rows")

    # GroupShuffleSplit (sklearn-equivalent): seeded permutation of unique groups,
    # accumulate groups into the test set until ~test_size of *samples* is reached.
    groups = np.array([r["repo_id"] for r in rows])
    classes, group_indices = np.unique(groups, return_inverse=True)
    n_groups = len(classes)
    rng = np.random.RandomState(RANDOM_SEED)
    n_test = int(np.ceil(TEST_SIZE * n))

    # sklearn GroupShuffleSplit replicates ShuffleSplit over group *counts*.
    group_counts = np.bincount(group_indices)
    found = False
    for _ in range(100):
        permutation = rng.permutation(n_groups)
        cum = np.cumsum(group_counts[permutation])
        idx = np.searchsorted(cum, n_test, side="right")
        test_groups = permutation[: idx + 1]
        n_test_actual = cum[idx] if idx < len(cum) else cum[-1]
        if 0 < n_test_actual < n:
            found = True
            break
    if not found:
        raise RuntimeError("GroupShuffleSplit failed to find a valid split")

    test_group_set = set(test_groups.tolist())
    test_mask = np.array([gi in test_group_set for gi in group_indices])

    n_train = int((~test_mask).sum())
    n_test_final = int(test_mask.sum())
    print(f"Groups (repo_id): {n_groups}  train_rows={n_train}  test_rows={n_test_final}")

    cols = [
        "split", "example_id", "model", "instance_id", "repo_id", "label",
        "total_steps", "source_path", "random_seed", "test_size", "max_prefix_k",
        "benchmark_setting", "min_trajectory_steps", "split_strategy", "termination_type",
    ]
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow(
                dict(
                    split="test" if test_mask[i] else "train",
                    example_id=r["example_id"],
                    model=r["model"],
                    instance_id=r["instance_id"],
                    repo_id=r["repo_id"],
                    label=r["label"],
                    total_steps=r["total_steps"],
                    source_path=r["source_path"],
                    random_seed=RANDOM_SEED,
                    test_size=TEST_SIZE,
                    max_prefix_k=MAX_PREFIX_K,
                    benchmark_setting=BENCHMARK_SETTING,
                    min_trajectory_steps=MIN_TRAJ_STEPS,
                    split_strategy=SPLIT_STRATEGY,
                    termination_type=r["termination_type"],
                )
            )
    print(f"Wrote {CSV_PATH}")

    # Stats for the report.
    from collections import Counter, defaultdict
    by_model = defaultdict(lambda: [0, 0])
    term_overall = Counter()
    for i, r in enumerate(rows):
        by_model[r["model"]][r["label"]] += 1
        term_overall[r["termination_type"]] += 1
    print("== label balance per group ==")
    for m, (neg, pos) in sorted(by_model.items()):
        tot = neg + pos
        print(f"  {m}: n={tot} resolved={pos} ({pos/tot*100:.1f}%) unresolved={neg}")
    tot_pos = sum(v[1] for v in by_model.values())
    print(f"  OVERALL: n={n} resolved={tot_pos} ({tot_pos/n*100:.1f}%)")
    print("== termination types overall ==")
    for k, v in term_overall.most_common():
        print(f"  {k}: {v}")
    # test split label balance
    test_pos = sum(rows[i]["label"] for i in range(n) if test_mask[i])
    print(f"== test split: n={n_test_final} resolved={test_pos} ({test_pos/n_test_final*100:.1f}%) ==")
    print(f"== train split: n={n_train} ==")


if __name__ == "__main__":
    main()
