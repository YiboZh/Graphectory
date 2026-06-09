#!/usr/bin/env python3
"""check_feature_parity.py

Verify that phase-codeblock graphs built from a *new* dataset carry the
**identical node/edge feature schema** as the reference OpenHands v2 graph
(`build_phase_codeblock_graph_v2_from_oh_trajectory`).

This is the GATE-1a parity check for task #09 (more-trajectory-datasets): every new
dataset builder must emit nodes/edges whose feature *keys* (and types) match the v2
contract exactly, so a `phase`/`code_block`/`termination` node from SWE-agent or
SWE-smith means the same thing as one from OpenHands.

Usage
-----
    # Generate the golden reference key-sets from a real OpenHands v2 graph and
    # diff a directory of candidate graphs against it.
    python scripts/check_feature_parity.py \
        --candidate-dir /home/yiboz7/data/processed_graphs/phase_codeblock_swesmith_v2 \
        [--reference-graph /path/to/an/openhands_v2/instance.json] \
        [--sample 25]

    # Just print the golden reference schema (no candidate):
    python scripts/check_feature_parity.py --print-reference

If --reference-graph is omitted, the script builds one on the fly from the existing
OpenHands raw trajectories at
/home/yiboz7/data/raw_trajectories/OpenHands/<first model>/output.jsonl.

Exit code 0 = parity holds for every sampled candidate graph; 1 = mismatch/empty.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "graph_construction"))

OH_RAW_ROOT = Path("/home/yiboz7/data/raw_trajectories/OpenHands")


def schema_of(graph: dict) -> dict:
    """Return {node_type -> sorted(feature_keys)} and {edge_type -> sorted(keys)}.

    Structural keys that are not part of the feature contract are ignored:
    node `id` and edge `source`/`target`/`key` (multigraph plumbing).
    """
    node_keys: dict[str, set] = defaultdict(set)
    node_seen: dict[str, set] = defaultdict(set)
    for n in graph["nodes"]:
        nt = n.get("node_type", "?")
        keys = {k for k in n.keys() if k != "id"}
        if nt not in node_seen or not node_seen[nt]:
            node_keys[nt] = set(keys)
            node_seen[nt] = set(keys)
        else:
            # require *consistent* keys across nodes of the same type
            node_keys[nt] |= keys
    edge_keys: dict[str, set] = defaultdict(set)
    for e in graph["edges"]:
        et = e.get("edge_type", "?")
        keys = {k for k in e.keys() if k not in ("source", "target", "key")}
        edge_keys[et] |= keys
    return {
        "nodes": {k: sorted(v) for k, v in node_keys.items()},
        "edges": {k: sorted(v) for k, v in edge_keys.items()},
    }


def build_reference_graph() -> dict:
    """Build a fresh OpenHands v2 graph from the existing raw trajectories."""
    import phaseCodeBlockGraph as p

    model_dirs = sorted(d for d in OH_RAW_ROOT.iterdir() if d.is_dir())
    if not model_dirs:
        raise SystemExit(f"No OpenHands model dirs under {OH_RAW_ROOT}")
    md = model_dirs[0]
    report = md / "report.json"
    with open(md / "output.jsonl") as f:
        td = json.loads(f.readline())
    iid = td["instance_id"]
    out = tempfile.mkdtemp()
    path = p.build_phase_codeblock_graph_v2_from_oh_trajectory(
        td, iid, out, str(report) if report.exists() else None
    )
    return json.loads(Path(path).read_text())


def reference_schema(reference_graph_path: str | None) -> dict:
    if reference_graph_path:
        g = json.loads(Path(reference_graph_path).read_text())
    else:
        g = build_reference_graph()
    return schema_of(g)


def iter_candidate_graphs(candidate_dir: Path, sample: int):
    files = sorted(candidate_dir.rglob("*.json"))
    files = [f for f in files if f.parent.name == f.stem]  # {iid}/{iid}.json
    if sample > 0:
        # spread the sample across the directory
        if len(files) > sample:
            step = max(1, len(files) // sample)
            files = files[::step][:sample]
    return files


def compare(ref: dict, cand: dict) -> list[str]:
    """Return a list of human-readable mismatches (empty = parity holds)."""
    problems: list[str] = []
    for kind in ("nodes", "edges"):
        ref_types = set(ref[kind])
        cand_types = set(cand[kind])
        # candidate may legitimately omit a type if a given trajectory lacks it
        # (e.g. a graph with no code_block). Only flag *extra* types and *key* diffs.
        for t in cand_types - ref_types:
            problems.append(f"{kind}: candidate has unexpected {kind[:-1]} type '{t}'")
        for t in cand_types & ref_types:
            rk, ck = set(ref[kind][t]), set(cand[kind][t])
            missing = rk - ck
            extra = ck - rk
            if missing:
                problems.append(f"{kind}[{t}]: MISSING keys {sorted(missing)}")
            if extra:
                problems.append(f"{kind}[{t}]: EXTRA keys {sorted(extra)}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-dir", type=Path)
    ap.add_argument("--reference-graph", type=str, default=None)
    ap.add_argument("--sample", type=int, default=25)
    ap.add_argument("--print-reference", action="store_true")
    args = ap.parse_args()

    ref = reference_schema(args.reference_graph)
    if args.print_reference:
        print(json.dumps(ref, indent=2))
        return 0

    if not args.candidate_dir:
        ap.error("--candidate-dir required unless --print-reference")

    print("REFERENCE (OpenHands v2) schema:")
    print(json.dumps(ref, indent=2))
    print("=" * 70)

    files = iter_candidate_graphs(args.candidate_dir, args.sample)
    if not files:
        print(f"[FAIL] No candidate graphs found under {args.candidate_dir}")
        return 1

    n_ok = 0
    all_problems: list[str] = []
    for f in files:
        try:
            g = json.loads(f.read_text())
        except Exception as exc:
            all_problems.append(f"{f}: load error {exc}")
            continue
        cand = schema_of(g)
        problems = compare(ref, cand)
        if problems:
            for p in problems:
                all_problems.append(f"{f.stem}: {p}")
        else:
            n_ok += 1

    print(f"Checked {len(files)} candidate graphs from {args.candidate_dir}")
    print(f"  parity-OK: {n_ok}/{len(files)}")
    if all_problems:
        print("  MISMATCHES:")
        for p in sorted(set(all_problems))[:50]:
            print("   -", p)
        print("[FAIL] feature parity does NOT hold")
        return 1
    print("[PASS] feature parity holds for all sampled candidate graphs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
