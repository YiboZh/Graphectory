# Phase-CodeBlock Graph v2 (Ablation) — Features Reference

This document describes **version 2** of the phase-codeblock graph representation, an
ablation variant of the [v1 graph](PHASE_CODEBLOCK_GRAPH_FEATURES.md) designed to study the
impact of including only the files the agent actually modified.

**Implementation files:**
- `graph_construction/phaseCodeBlockGraph.py` — builder class (shared with v1; controlled by
  `graph_version="v2"`)
- `scripts/generate_phase_codeblock_graphs.py` — batch CLI (`--graph_version v2 --html`)
- `docs/PHASE_CODEBLOCK_GRAPH_V2_FEATURES.md` — this file

**On-disk format:** NetworkX **node-link JSON** (`directed`, `multigraph`, `graph`, `nodes`,
`edges`) — identical container format to v1.

---

## Table of contents

1. [What changed from v1](#what-changed-from-v1)
2. [How to construct v2 graphs](#how-to-construct-v2-graphs)
3. [Graph-level metadata](#graph-level-metadata)
4. [Node types and features](#node-types-and-features)
   - [Start Node](#start-node)
   - [Phase Node](#phase-node)
   - [Code Block Node (v2)](#code-block-node-v2)
   - [Termination Node](#termination-node)
5. [Edge types](#edge-types)
6. [Ablation design rationale](#ablation-design-rationale)
7. [Known limitations](#known-limitations)
8. [v1 vs v2 comparison table](#v1-vs-v2-comparison-table)

---

## What changed from v1

Version 2 introduces two modifications relative to v1, both motivated by focusing the graph
on the files that matter for the agent's actual repair work:

### Change 1 — Modified-files-only (view-only nodes removed)

In v1, a code-block node is created for *every* file or code region the agent touches:
viewed, searched, edited, created, or deleted.

In **v2**, after the full graph is built, any code-block node whose `num_edits == 0`
(i.e. the agent only viewed or searched that file, never patched it) is **removed** along
with all its incident edges.

**Effect:** The graph retains only the files the agent actually modified.  Files consulted
for context during localization but left unchanged are excluded.

### Change 2 — No line numbers (file-name only labels)

In v1, code-block nodes are deduplicated by `(file_path, start_line, end_line)`.  A single
file viewed at ranges 1–50 and 100–200 produces **two** separate nodes.

In **v2**, code-block nodes are deduplicated by `file_path` only.  All operations on the
same file (regardless of line range) collapse into a single node.  Line numbers are not
stored on the node.  The visual label shows only the file name (the last path component),
with no `:start-end` suffix.

---

## How to construct v2 graphs

### Prerequisites

```bash
source /home/yiboz7/code/Graphectory/.venv/bin/activate
```

### Generate v2 graphs (with HTML figures)

```bash
# All OpenHands models — v2 graphs + HTML
python scripts/generate_phase_codeblock_graphs.py --graph_version v2 --html
```

Default paths:
- **Input:**  `/home/yiboz7/data/raw_trajectories/OpenHands/`
- **Output:** `/home/yiboz7/data/processed_graphs/phase_codeblock_openhands_v2/`

### Common options

```bash
# Custom paths
python scripts/generate_phase_codeblock_graphs.py \
    --graph_version v2 --html \
    --trajs_root  /path/to/raw_trajectories/OpenHands \
    --output_root /path/to/processed_graphs/phase_codeblock_openhands_v2

# Specific model runs only
python scripts/generate_phase_codeblock_graphs.py \
    --graph_version v2 --html \
    --model_dirs claude-sonnet-4_maxiter_100_N_v0.40.0-no-hint-run_1

# Without HTML (JSON only)
python scripts/generate_phase_codeblock_graphs.py --graph_version v2

# v1 is still the default (unchanged behaviour)
python scripts/generate_phase_codeblock_graphs.py
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--graph_version` | `v1` | Graph variant: `v1` (full) or `v2` (ablation) |
| `--html` | off | Also render interactive Plotly HTML alongside each JSON |
| `--trajs_root` | `/home/yiboz7/data/raw_trajectories/OpenHands` | Root directory containing model-run subdirectories |
| `--output_root` | *(version-specific default)* | Root directory for output graphs |
| `--model_dirs` | *(all)* | Restrict to specific model-run directory names |
| `--eval_report` | *(per-model `report.json`)* | Path to a shared eval report JSON |
| `--workers` | `8` | Number of parallel worker processes per model run |
| `--verbose` | off | Enable DEBUG-level logging |

### Output structure

```
{output_root}/
└── {model_dir_name}/
    └── {instance_id}/
        ├── {instance_id}.json      ← NetworkX node-link JSON graph
        └── {instance_id}.html      ← Plotly HTML figure (when --html)
```

Example:
```
/home/yiboz7/data/processed_graphs/phase_codeblock_openhands_v2/
├── claude-sonnet-4_maxiter_100_N_v0.40.0-no-hint-run_1/
│   ├── astropy__astropy-12907/
│   │   ├── astropy__astropy-12907.json
│   │   └── astropy__astropy-12907.html
│   └── django__django-12345/
│       ├── django__django-12345.json
│       └── django__django-12345.html
└── ...
```

### Using the builder programmatically

```python
import json, sys
sys.path.insert(0, "graph_construction")
from phaseCodeBlockGraph import build_phase_codeblock_graph_v2_from_oh_trajectory

with open("output.jsonl") as f:
    traj_data = json.loads(f.readline())

json_path = build_phase_codeblock_graph_v2_from_oh_trajectory(
    traj_data=traj_data,
    instance_id=traj_data["instance_id"],
    output_dir="/path/to/output",
    eval_report_path="/path/to/report.json",   # optional
)
print("Graph saved to:", json_path)
```

---

## Graph-level metadata

| Field | Type | v1 value | v2 value |
|-------|------|----------|----------|
| `resolution_status` | string | same | same |
| `instance_name` | string | same | same |
| `graph_type` | string | `"phase_codeblock"` | `"phase_codeblock_v2"` |
| `graph_version` | string | `"v1"` | `"v2"` |
| `num_phases` | int | same | same |
| `num_code_blocks` | int | count of all touched file/range nodes | count of **modified** file nodes only |

---

## Node types and features

### Start Node

Identical to v1.

| Feature | Type | Description |
|---------|------|-------------|
| `node_type` | string | Always `"start"` |
| `step_idx` | int | Always `0` |

---

### Phase Node

Identical to v1.  All phase-level aggregates (views, edits, searches, etc.) are computed
before the view-only filter is applied, so phase stats remain consistent even when some
code-block nodes are removed.

See [v1 Phase Node](PHASE_CODEBLOCK_GRAPH_FEATURES.md#phase-node) for the full feature list.

---

### Code Block Node (v2)

Represents one unique **file** (not file + line range) touched by the agent.  Only files
that were modified (edit, create, or delete operations) survive the ablation filter.

| Feature | Type | Description | Construction method |
|---------|------|-------------|---------------------|
| `node_type` | string | Always `"code_block"` | **Constant** |
| `file_path` | string | Full file path as it appears in the tool call args | **Direct copy** |
| `file_path_hash` | string | MD5 hex digest of `file_path` | **Structural computation** |
| `num_views` | int | Total "view" operations on this file across all phases | **Derived aggregation** |
| `num_search_hits` | int | Total "search_hit" operations on this file across all phases | **Derived aggregation** |
| `num_edits` | int | Total edit+create+delete operations on this file across all phases | **Derived aggregation** — always ≥ 1 (filter guarantee) |

**Differences from v1:**
- `start_line` and `end_line` are **not present** on v2 nodes.
- Deduplication key is `(file_path,)` instead of `(file_path, start_line, end_line)`.
- Only nodes with `num_edits ≥ 1` are retained.

**Node key pattern:** `"{counter}:code_block:{file_path_hash_prefix}"`

---

### Termination Node

Identical to v1.

| Feature | Type | Description |
|---------|------|-------------|
| `node_type` | string | Always `"termination"` |
| `termination_type` | string | `"submit"`, `"max_step"`, `"no_submit"`, `"error_stop"` |
| `final_step_idx` | int | Step index of the last processed step |
| `last_phase` | string | `phase_type` of the last Phase Node |
| `last_observation_outcome` | string | `"success"`, `"failure"`, or `"neutral"` |

---

## Edge types

All edge types are identical to v1:

| Edge type | Description |
|-----------|-------------|
| `start_to_phase` | Start → first Phase |
| `phase_transition` | Phase_i → Phase_{i+1} |
| `phase_code_operation` | Phase → CodeBlock (one per op-type per phase; view-only target nodes removed in v2) |
| `phase_to_termination` | last Phase → Termination |
| `start_to_termination` | Start → Termination (empty trajectory) |

**Note:** `phase_code_operation` edges whose target code-block node was removed by the
view-only filter are automatically removed when NetworkX removes the node.  This means
phases that only touched files they never modified will have no outgoing
`phase_code_operation` edges in the v2 graph.

---

## Ablation design rationale

The v1 graph includes all files the agent **looked at**, which captures the full
localization search trajectory.  This is useful for understanding how the agent explored
the codebase but introduces many nodes that are irrelevant to the actual repair.

The v2 ablation asks: *does restricting the graph to only modified files change what we
can predict about agent success?*

Two hypotheses motivate this ablation:

1. **Signal concentration**: Modified-file nodes carry stronger signal about repair quality
   than viewed-file nodes.  Removing the latter may reduce noise.
2. **Graph size reduction**: Fewer nodes and edges make the graph easier to reason about
   for downstream ML models and human analysts.

By comparing model performance on v1 vs v2 graphs, researchers can quantify the marginal
value of the localization-phase file-view information.

---

## Known limitations

All [v1 limitations](PHASE_CODEBLOCK_GRAPH_FEATURES.md#known-limitations-of-this-first-version)
apply.  Additional v2-specific limitations:

1. **Edit detection is operation-based.** A file is considered "modified" only if an
   `edit`, `create`, or `delete` operation is recorded for it.  Files edited through
   complex pipelines (e.g. `awk`, `python -c`, heredoc redirections) that were not
   parsed may be incorrectly classified as view-only and removed.

2. **Phase stats are not recomputed after filtering.** Phase node attributes such as
   `num_unique_files` and `num_unique_code_blocks` reflect the original counts before
   view-only removal.  They are not updated to match the post-filter graph structure.

3. **No line-range resolution.** Because all operations on a file merge into one node,
   it is no longer possible to distinguish whether the agent viewed lines 1–50 vs
   1–500 before making an edit.  This granularity is only available in v1.

4. **Same-file multiple views collapse.** If the agent viewed the same file at many
   different ranges (common for large files), all those views accumulate into one
   `num_views` count on the single merged node.  The distribution of view ranges is lost.

---

## v1 vs v2 comparison table

| Dimension | v1 | v2 |
|-----------|----|----|
| **Code-block dedup key** | `(file_path, start_line, end_line)` | `(file_path,)` |
| **Line numbers on nodes** | Yes — `start_line`, `end_line` | No |
| **Node label** | `filename.py:start-end` | `filename.py` |
| **Files included** | All touched (viewed + edited) | Modified only (`num_edits ≥ 1`) |
| **View-only nodes** | Included | Removed |
| **Graph type metadata** | `"phase_codeblock"` | `"phase_codeblock_v2"` |
| **Typical node count** | Higher (many view-only nodes) | Lower |
| **Phase stats** | Reflects all operations | Reflects all operations (unchanged) |
| **Default output dir** | `phase_codeblock_openhands` | `phase_codeblock_openhands_v2` |
| **HTML generation** | Optional (`--html`) | Optional (`--html`) |
