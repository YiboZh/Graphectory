# Phase-CodeBlock Graph Features Reference

This document describes every attribute on **nodes**, **edges**, and **graph-level metadata** in
the `PhaseCodeBlockGraphBuilder` graph representation, how each value is constructed, how
OpenHands raw trajectories are parsed, and how this representation differs from the existing
Graphectory graph.

**Implementation files:**
- `graph_construction/phaseCodeBlockGraph.py` — builder class + module-level entry function
- `scripts/generate_phase_codeblock_graphs.py` — batch CLI for OpenHands trajectories
- `docs/PHASE_CODEBLOCK_GRAPH_FEATURES.md` — this file

**On-disk format:** NetworkX **node-link JSON** (`directed`, `multigraph`, `graph`, `nodes`, `edges`)
— identical container format to the existing Graphectory graphs.

---

## Table of contents

1. [How to construct graphs](#how-to-construct-graphs)
2. [Motivation and design overview](#motivation-and-design-overview)
3. [Graph-level metadata](#graph-level-metadata)
4. [Node types and features](#node-types-and-features)
   - [Start Node](#start-node)
   - [Phase Node](#phase-node)
   - [Code Block Node](#code-block-node)
   - [Termination Node](#termination-node)
5. [Edge types and features](#edge-types-and-features)
   - [start_to_phase](#start_to_phase)
   - [phase_transition](#phase_transition)
   - [phase_code_operation](#phase_code_operation)
   - [phase_to_termination / start_to_termination](#phase_to_termination--start_to_termination)
6. [How each feature is constructed](#how-each-feature-is-constructed)
7. [How OpenHands raw trajectories are parsed](#how-openhands-raw-trajectories-are-parsed)
8. [Phase classification rules](#phase-classification-rules)
9. [Code-block operation extraction](#code-block-operation-extraction)
10. [Termination type detection](#termination-type-detection)
11. [Known limitations of this first version](#known-limitations-of-this-first-version)
12. [How this graph differs from the existing Graphectory graph](#how-this-graph-differs-from-the-existing-graphectory-graph)
13. [Collection-method legend](#collection-method-legend)

---

## How to construct graphs

All graph construction runs from the repo root using the project's virtual environment.

### Prerequisites

```bash
# Activate the virtual environment (all dependencies are already installed)
source /home/yiboz7/code/Graphectory/.venv/bin/activate
```

### Generate graphs for all OpenHands models (default)

```bash
python scripts/generate_phase_codeblock_graphs.py
```

This uses the default paths:
- **Input:**  `/home/yiboz7/data/raw_trajectories/OpenHands/`
- **Output:** `/home/yiboz7/data/processed_graphs/phase_codeblock_openhands/`

Each model-run subdirectory containing an `output.jsonl` is discovered automatically.
The local `report.json` in each model directory is used for `resolution_status` lookup.

### Common options

```bash
# Custom input/output paths
python scripts/generate_phase_codeblock_graphs.py \
    --trajs_root  /path/to/raw_trajectories/OpenHands \
    --output_root /path/to/processed_graphs/phase_codeblock_openhands

# Process only specific model runs
python scripts/generate_phase_codeblock_graphs.py \
    --model_dirs claude-sonnet-4_maxiter_100_N_v0.40.0-no-hint-run_1 \
                 deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1

# Use a single shared eval report for all models
python scripts/generate_phase_codeblock_graphs.py \
    --eval_report /path/to/report.json

# Control parallelism (default: 8 workers)
python scripts/generate_phase_codeblock_graphs.py --workers 16

# Enable debug logging
python scripts/generate_phase_codeblock_graphs.py --verbose
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--trajs_root` | `/home/yiboz7/data/raw_trajectories/OpenHands` | Root directory containing model-run subdirectories |
| `--output_root` | `/home/yiboz7/data/processed_graphs/phase_codeblock_openhands` | Root directory for output graphs |
| `--model_dirs` | *(all)* | Restrict to specific model-run directory names (space-separated) |
| `--eval_report` | *(per-model `report.json`)* | Path to a shared eval report JSON for `resolution_status` lookup |
| `--workers` | `8` | Number of parallel worker processes per model run |
| `--verbose` | off | Enable DEBUG-level logging |

### Output structure

```
{output_root}/
└── {model_dir_name}/
    └── {instance_id}/
        └── {instance_id}.json      ← NetworkX node-link JSON graph
```

Example:
```
/home/yiboz7/data/processed_graphs/phase_codeblock_openhands/
├── claude-sonnet-4_maxiter_100_N_v0.40.0-no-hint-run_1/
│   ├── astropy__astropy-12907/
│   │   └── astropy__astropy-12907.json
│   └── django__django-12345/
│       └── django__django-12345.json
├── deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1/
│   └── ...
└── ...
```

### Logging and exit codes

The script logs a per-instance `OK` / `ERR` line and prints a summary table at the end:

```
======================================================================
PHASE-CODEBLOCK GRAPH GENERATION SUMMARY
======================================================================
  claude-sonnet-4_maxiter_100_N_v0.40.0-no-hint-run_1
    success=500  error=0  skip=0
  ...
----------------------------------------------------------------------
  TOTAL  trajectories processed : 1974
  TOTAL  graphs generated       : 1974
  TOTAL  failed trajectories    : 0
  TOTAL  skipped trajectories   : 0
  Output root                   : /home/yiboz7/data/processed_graphs/phase_codeblock_openhands
======================================================================
```

Exit code is `0` when all trajectories succeed, `1` if any errors occurred.

### Using the builder programmatically

```python
import json, sys
sys.path.insert(0, "graph_construction")   # or install the package
from phaseCodeBlockGraph import build_phase_codeblock_graph_from_oh_trajectory

with open("output.jsonl") as f:
    traj_data = json.loads(f.readline())

json_path = build_phase_codeblock_graph_from_oh_trajectory(
    traj_data=traj_data,
    instance_id=traj_data["instance_id"],
    output_dir="/path/to/output",
    eval_report_path="/path/to/report.json",   # optional
)
print("Graph saved to:", json_path)
```

The returned JSON can be loaded back into NetworkX with:

```python
import json, networkx as nx
from networkx.readwrite import json_graph

with open(json_path) as f:
    data = json.load(f)

G = nx.node_link_graph(data, edges="edges")
```

---

## Motivation and design overview

The existing Graphectory graph is an **action-level** graph: every distinct tool invocation
(grep, str_replace_editor view, pytest, …) becomes a node, and repeated calls merge into the
same node via signature deduplication. This representation is rich but makes it hard to directly
read two high-level failure signals:

1. **Repeated operations on the same code region** — symptom of an agent stuck in a loop.
2. **Missing validation after patching** — agent edits code and terminates without running tests.

The phase-codeblock graph lifts the representation to two coarser levels:

- **Phase Nodes** group consecutive trajectory steps that share the same high-level workflow
  phase (`localization`, `patching`, `validation`, `general`). Think/reasoning steps are
  absorbed into the adjacent phase rather than splitting it.
- **Code Block Nodes** represent file regions (file path + optional line range). They are
  shared across phases and aggregated globally, so a single code-block node accumulates all
  operations across the entire trajectory.

Graph structure example:

```
Start → loc_1 → patch_1 → val_1 → patch_2 → val_2 → Termination
                  │                   │
                  └──► foo.py (edit)   └──► test_foo.py (view/search_hit)
```

The Phase→CodeBlock edges record *how many times* each file/region was touched per operation
type, making repeated-operation patterns immediately visible as high-count edges.

---

## Graph-level metadata

Stored on the root `graph` object (not on individual nodes).

| Field | Type | Description | Construction method |
|-------|------|-------------|---------------------|
| `resolution_status` | string | Whether the SWE-bench instance was solved: `resolved`, `unresolved`, `unsubmitted`, or `unknown` | **Set lookup** — `instance_id` in `eval_report["resolved_ids"]` / `"unresolved_ids"`; falls back to `"unknown"` when report unavailable |
| `instance_name` | string | SWE-bench instance ID (e.g. `django__django-12345`) | **Direct copy** from caller argument |
| `graph_type` | string | Always `"phase_codeblock"` | **Constant** |
| `num_phases` | int | Total number of Phase Nodes in the graph | **Structural computation** — `len(phases)` |
| `num_code_blocks` | int | Total number of Code Block Nodes in the graph | **Structural computation** — count nodes with `node_type == "code_block"` |

---

## Node types and features

### Start Node

One per graph; represents the beginning of the trajectory.

| Feature | Type | Description | Construction method |
|---------|------|-------------|---------------------|
| `node_type` | string | Always `"start"` | **Constant** |
| `step_idx` | int | Always `0` | **Constant** |

**Node key pattern:** `"0:start"`

---

### Phase Node

One per contiguous segment of consecutive steps with the same workflow phase.
Phase nodes are **not** deduplicated — if the agent transitions localization → patching →
localization, three Phase Nodes are created (loc_0, patch_1, loc_2).

| Feature | Type | Description | Construction method |
|---------|------|-------------|---------------------|
| `node_type` | string | Always `"phase"` | **Constant** |
| `phase_type` | string | `"localization"`, `"patching"`, `"validation"`, or `"general"` | **Rule-based classification** via `mapPhase.get_phase()` (see [Phase classification rules](#phase-classification-rules)) |
| `phase_index` | int | 0-based temporal index of this phase within the trajectory | **Structural computation** — monotonically incremented counter |
| `start_step` | int | Step index of the first step in this phase | **Direct copy** from first StepInfo in the accumulator |
| `end_step` | int | Step index of the last step in this phase | **Direct copy** from last StepInfo in the accumulator |
| `num_steps` | int | Total number of trajectory steps (including think) in this phase | **Derived aggregation** — `len(steps)` |
| `num_actions` | int | Number of non-think steps (actual tool invocations) | **Derived aggregation** — count steps where `is_think_only == False` |
| `num_unique_files` | int | Number of distinct file paths touched in this phase | **Derived aggregation** — `len({op.file_path for op in all_ops})` |
| `num_unique_code_blocks` | int | Number of distinct (file_path, start_line, end_line) tuples touched | **Derived aggregation** — `len({(fp, sl, el) for op in all_ops})` |
| `num_views` | int | Count of "view" code-block operations in this phase | **Derived aggregation** — count ops with `op_type == "view"` |
| `num_searches` | int | Count of "search_hit" operations (grep, rg, find, …) | **Derived aggregation** — count ops with `op_type == "search_hit"` |
| `num_edits` | int | Count of "edit" operations (str_replace, insert, sed -i, …) | **Derived aggregation** — count ops with `op_type == "edit"` |
| `num_creates` | int | Count of "create" operations (str_replace_editor create, touch) | **Derived aggregation** — count ops with `op_type == "create"` |
| `num_deletes` | int | Count of "delete" operations (rm, str_replace_editor delete) | **Derived aggregation** — count ops with `op_type == "delete"` |
| `num_tests` | int | Count of steps that executed a test command | **Derived aggregation** — count steps where `is_test_step == True` |
| `num_failed_actions` | int | Count of steps where the observation outcome was "failure" | **Derived aggregation** — count steps where `observation_outcome == "failure"` |
| `has_error_observation` | bool | True if any step in this phase had a traceback or error in its observation | **Substring match** — checks for "traceback (most recent call last)", "error:", "exception:", etc. |
| `has_patch` | bool | True if any edit/create/delete operation targeted a non-test file | **Rule-based** — check `op.op_type in (edit, create, delete)` AND `not any(test_hint in file_path.lower())` |
| `has_validation` | bool | True if any test command was executed in this phase | **Rule-based** — `any(s.is_test_step for s in steps)` |
| `is_after_first_patch` | bool | True if at least one earlier Phase Node had `has_patch == True` | **Structural computation** — running boolean `first_patch_seen` updated as phases are built |
| `thought_length_sum` | int | Total character count of all thought texts in this phase | **Derived aggregation** — `sum(s.thought_len_raw for s in steps)` |
| `thought_length_mean` | float | Mean thought character count per step | **Derived aggregation** — `thought_length_sum / num_steps` |
| `observation_length_sum` | int | Total character count of all observation texts in this phase | **Derived aggregation** — `sum(s.observation_length for s in steps)` |
| `observation_length_mean` | float | Mean observation character count per step | **Derived aggregation** — `observation_length_sum / num_steps` |
| `latency_sum` | float or null | Sum of response latencies (seconds) for steps in this phase | **Derived aggregation** — sum of available latencies; `null` if none available |
| `latency_mean` | float or null | Mean response latency per step | **Derived aggregation** — `latency_sum / count(non-null latencies)` |

**Node key pattern:** `"{counter}:phase_{phase_index}"`

---

### Code Block Node

One per unique `(file_path, start_line, end_line)` tuple across the entire trajectory.
Shared between all phases; global op counts accumulate across all phases.

| Feature | Type | Description | Construction method |
|---------|------|-------------|---------------------|
| `node_type` | string | Always `"code_block"` | **Constant** |
| `file_path` | string | Full file path as it appears in the tool call args | **Direct copy** from `str_replace_editor` `path` arg or regex extraction from bash command |
| `file_path_hash` | string | MD5 hex digest of `file_path` | **Structural computation** — `hashlib.md5(file_path.encode()).hexdigest()` |
| `start_line` | int or null | Start of the line range (1-based), or null if no range info | **Direct copy** from `view_range[0]` (str_replace_editor view), `insert_line` (insert), or `null` |
| `end_line` | int or null | End of the line range (1-based), or null if no range info | **Direct copy** from `view_range[1]` or same as `start_line` for point insertions |
| `num_views` | int | Total "view" operations on this code block across all phases | **Derived aggregation** — global sum of view ops for this (fp, sl, el) key |
| `num_search_hits` | int | Total "search_hit" operations on this code block across all phases | **Derived aggregation** — global sum of search_hit ops |
| `num_edits` | int | Total edit+create+delete operations on this code block across all phases | **Derived aggregation** — sum of edit + create + delete ops |

**Node key pattern:** `"{counter}:code_block:{file_path_hash_prefix}:{start_line}:{end_line}"`

**Deduplication key:** `(file_path, start_line, end_line)` — exact string match on file path, exact
integer match on line numbers. Different view ranges of the same file produce separate nodes.
A whole-file operation (`start_line=None, end_line=None`) and a ranged operation on the same file
produce separate nodes.

---

### Termination Node

One per graph; represents the end of the trajectory.

| Feature | Type | Description | Construction method |
|---------|------|-------------|---------------------|
| `node_type` | string | Always `"termination"` | **Constant** |
| `termination_type` | string | How the trajectory ended; one of `"submit"`, `"max_step"`, `"no_submit"`, `"error_stop"`, `"unknown"` | **Rule-based classification** — see [Termination type detection](#termination-type-detection) |
| `final_step_idx` | int | Step index of the last processed step (0-based) | **Direct copy** from last StepInfo |
| `last_phase` | string | `phase_type` of the last Phase Node, or `"none"` for empty trajectories | **Structural computation** — `phases[-1].phase_type` |
| `last_observation_outcome` | string | Observation outcome of the last processed step: `"success"`, `"failure"`, or `"neutral"` | **Substring match** — `detect_observation_outcome()` from `buildGraph.py` |

**Node key pattern:** `"{counter}:termination"`

---

## Edge types and features

### start_to_phase

Connects the Start Node to the first Phase Node.

| Feature | Type | Description |
|---------|------|-------------|
| `edge_type` | string | Always `"start_to_phase"` |

---

### phase_transition

Connects Phase_i to Phase_{i+1}. Added for every consecutive pair of Phase Nodes.

| Feature | Type | Description | Construction method |
|---------|------|-------------|---------------------|
| `edge_type` | string | Always `"phase_transition"` | **Constant** |
| `from_phase` | string | `phase_type` of the source Phase Node | **Direct copy** from source node |
| `to_phase` | string | `phase_type` of the target Phase Node | **Direct copy** from target node |
| `step_gap` | int | Number of steps between the end of the source phase and the start of the target phase | **Structural computation** — `target.start_step - source.end_step` (typically 1) |

---

### phase_code_operation

Connects a Phase Node to a Code Block Node. One edge per unique `(phase, code_block, op_type)`
triple — so a single phase can have multiple edges to the same code block (one per operation type).

| Feature | Type | Description | Construction method |
|---------|------|-------------|---------------------|
| `edge_type` | string | Always `"phase_code_operation"` | **Constant** |
| `operation_type` | string | `"view"`, `"search_hit"`, `"edit"`, `"create"`, or `"delete"` | **Rule-based classification** — see [Code-block operation extraction](#code-block-operation-extraction) |
| `num_actions` | int | How many times this operation type occurred on this code block within this phase | **Derived aggregation** — count of matching ops in the phase accumulator |

---

### phase_to_termination / start_to_termination

Connects the last Phase Node (or Start Node for empty trajectories) to the Termination Node.

| Feature | Type | Description |
|---------|------|-------------|
| `edge_type` | string | `"phase_to_termination"` or `"start_to_termination"` |

---

## How each feature is constructed

This section maps every feature to one of the construction methods in the
[legend](#collection-method-legend).

### Phase Node features

| Feature | Method | Source |
|---------|--------|--------|
| `phase_type` | Rule-based classification | `mapPhase.get_phase(tool, subcommand, command, args, prev_phases, flags)` |
| `phase_index` | Structural computation | Monotonic counter, reset to 0 per trajectory |
| `start_step`, `end_step` | Direct copy | `StepInfo.step_idx` of first/last step in phase |
| `num_steps` | Derived aggregation | `len(PhaseAccumulator.steps)` |
| `num_actions` | Derived aggregation | `sum(1 for s in steps if not s.is_think_only)` |
| `num_unique_files` | Derived aggregation | `len({op.file_path for op in all_ops})` |
| `num_unique_code_blocks` | Derived aggregation | `len({(op.file_path, op.start_line, op.end_line) for ...})` |
| `num_views` … `num_deletes` | Derived aggregation | Count of `CodeBlockOp` with matching `op_type` |
| `num_tests` | Rule-based + derived aggregation | `_is_test_execution()` per step; then sum |
| `num_failed_actions` | Derived aggregation | `sum(1 for s in steps if s.is_failed)` |
| `has_error_observation` | Substring match | Scan observation for traceback / error keywords |
| `has_patch` | Rule-based | op_type ∈ {edit, create, delete} AND file_path not test-related |
| `has_validation` | Rule-based | `any(s.is_test_step)` |
| `is_after_first_patch` | Structural computation | Running `first_patch_seen` boolean across phases |
| `thought_length_sum/mean` | Derived aggregation | Sum/mean of `thought_len_raw` across steps |
| `observation_length_sum/mean` | Derived aggregation | Sum/mean of `len(step.content)` |
| `latency_sum/mean` | Derived aggregation | Sum/mean of matched latency values |

### Code Block Node features

| Feature | Method | Source |
|---------|--------|--------|
| `file_path` | Direct copy | `str_replace_editor.path` arg or regex from bash command |
| `file_path_hash` | Structural computation | `hashlib.md5(file_path.encode()).hexdigest()` |
| `start_line`, `end_line` | Direct copy | `view_range[0/1]`, `insert_line`, or `None` |
| `num_views` | Derived aggregation | Global sum of view ops across all phases |
| `num_search_hits` | Derived aggregation | Global sum of search_hit ops |
| `num_edits` | Derived aggregation | Global sum of edit + create + delete ops |

### Termination Node features

| Feature | Method | Source |
|---------|--------|--------|
| `termination_type` | Rule-based classification | See [Termination type detection](#termination-type-detection) |
| `final_step_idx` | Direct copy | `steps[-1].step_idx` |
| `last_phase` | Direct copy | `phases[-1].phase_type` |
| `last_observation_outcome` | Substring match | `detect_observation_outcome(last_observation_text)` |

---

## How OpenHands raw trajectories are parsed

### Input format

Each line of an OH `output.jsonl` file is a JSON object with keys:
`instance_id`, `history`, `metrics`, `metadata`, `error`, `report`, `test_result`, etc.

`history` is a list of step dicts. Steps come in two forms:

1. **Action steps** (`action=X, observation=None`): the agent's tool call request.
   Contains `tool_call_metadata` with the model's response (thought + tool calls).
2. **Result steps** (`observation=X, action=None`): the execution result.
   Contains `content` (observation text) and the same `tool_call_metadata` as the action step.

The builder processes **only result steps** (`observation` is not None and not in `{"system",
"message", "recall"}`), exactly as the existing `build_graph_from_oh_trajectory` does.

### Per-step extraction

For each result step:

1. **Thought text** — from `tool_call_metadata.model_response.choices[0].message.content`
   (string or list of `{type: "text", text: "..."}` blocks).
2. **Observation text** — from `step["content"]`.
3. **Tool calls** — walk `choices → message → tool_calls → function → {name, arguments}`.
4. **Execute-bash parsing** — if `function.name == "execute_bash"`, the `command` string is
   passed to `CommandParser.parse()` (same parser as the existing OH builder, no YAML tool
   configs loaded). The returned list of parsed-command dicts is used for phase classification
   and code-block op extraction.
5. **Other tool calls** — `{tool: function.name, subcommand: args["command"], args: remaining_args}`.

### Latency matching

`metrics.response_latencies` is a list of `{latency, model, response_id}` dicts, one per model
call, in temporal order. The builder assigns `latencies[i]` to the `i`-th result step that has
`tool_call_metadata` present. Steps without a model response (recall, etc.) do not consume a
latency entry.

### Phase grouping

After parsing all steps into `StepInfo` objects, consecutive same-phase steps are grouped into
`PhaseAccumulator` objects:

- If a step is `is_think_only = True` (only tool call is `think`), it is **absorbed into the
  current phase** rather than triggering a phase transition. This avoids spurious `general` phase
  nodes between substantive steps.
- A phase transition only occurs when a non-think step has a different `phase_type` from the
  current phase.

---

## Phase classification rules

Phase is determined per result step by calling `mapPhase.get_phase(tool, subcommand, command,
args, prev_phases_set, flags)`. The step's phase is computed from the **first non-think tool
call** in that step.

Summary of rules (see `mapPhase.py` for full detail):

| Condition | Phase |
|-----------|-------|
| `str_replace_editor view` on a non-test file | `localization` |
| `str_replace_editor view` on a test file AFTER a prior patch | `validation` |
| `str_replace_editor str_replace / create / insert / undo_edit` on a non-test file | `patching` |
| `str_replace_editor` op on a test file AFTER a prior patch | `validation` |
| `str_replace_editor` op on a test file BEFORE first patch | `localization` |
| `grep / find / cat / …` (read-only) | `localization` |
| `grep / …` on test file AFTER a prior patch | `validation` |
| `grep / …` with output redirection to non-test file | `patching` |
| `pytest`, `python -m pytest` AFTER a prior patch | `validation` |
| `pytest`, `python -m pytest` BEFORE first patch | `localization` |
| `think` tool call | inherits current phase (or `general` if first step) |
| Anything else | `general` |

The `prev_phases_set` tracks all phases seen so far (like the existing builder). The key
distinction is whether `"patching"` has appeared in this set (`_has_prior_patch`).

Note: the existing Graphectory builder uses the phase string `"patch"` internally; this new
builder uses `"patching"` as the `phase_type` value on Phase Nodes for clarity, but passes the
same `prev_phases_set` (which uses `"patch"` strings) to `get_phase()` unchanged.

Wait — actually `get_phase()` returns `"patch"` not `"patching"`. The Phase Node `phase_type`
stores exactly what `get_phase()` returns, so `phase_type` values are:

- `"localization"`
- `"patch"` (not `"patching"` — matches `get_phase()` return values)
- `"validation"`
- `"general"`

---

## Code-block operation extraction

Operation type is determined per tool call:

### str_replace_editor

| Subcommand | Operation type | Line range |
|------------|----------------|------------|
| `view` | `view` | `view_range[0:2]` if present, else `(None, None)` |
| `str_replace` | `edit` | `(None, None)` |
| `create` | `create` | `(None, None)` |
| `insert` | `edit` | `(insert_line, insert_line)` |
| `undo_edit` | `edit` | `(None, None)` |
| `delete` | `delete` | `(None, None)` |

File path is read from `args["path"]`.

### execute_bash (after CommandParser parsing)

| Command | Operation type | Path source |
|---------|----------------|-------------|
| `grep`, `rg`, `ack`, `ag`, `find`, `locate` | `search_hit` | Parsed args + regex fallback |
| `cat`, `head`, `tail`, `nl`, `less`, `more`, `bat` | `view` | Parsed args + regex fallback |
| `touch`, `tee` | `create` | Parsed args + regex fallback |
| `rm`, `unlink`, `rmdir` | `delete` | Parsed args + regex fallback |
| `sed -i`, `perl -i` | `edit` | Parsed args + regex fallback |
| `pytest`, `python -m pytest`, `python test_*.py` | (sets `is_test_step=True` on StepInfo; adds `view` ops for recognized test file paths) | |

**Path extraction strategy for bash commands:**

1. From structured `args` dict/list: values that match `_looks_like_path()` heuristic
   (starts with `/`, `./`, `../`, `~/`, contains `/`, or matches `*.ext`).
2. Fallback: regex `_PATH_RE` over the raw command string for any path-like tokens.

This is a heuristic extraction and may miss some paths or include false positives. See
[Known limitations](#known-limitations-of-this-first-version).

---

## Termination type detection

Scans the full `history` list (including action-only steps) in priority order:

| Priority | Condition | `termination_type` |
|----------|-----------|-------------------|
| 1 | Any step has `action == "finish"` or `observation == "finish"` | `"submit"` |
| 2 | `traj_data["error"]` is not None | `"error_stop"` |
| 3 | Count of non-think actionable steps ≥ `metadata["max_iterations"]`, OR count of model responses ≥ `max_iterations` | `"max_step"` |
| 4 | None of the above | `"no_submit"` |

The `"unknown"` value is reserved for the Termination Node's `last_observation_outcome` field
when no steps were processed (empty trajectory), and is not a `termination_type`.

---

## Known limitations of this first version

1. **Path extraction from bash commands is heuristic.** The `_PATH_RE` regex and
   `_looks_like_path()` may miss paths without extensions, or match flags and environment
   variable names. Paths embedded in complex pipelines or heredocs may not be extracted.

2. **Line ranges are only available for `str_replace_editor view`.** For all other operations
   (`str_replace`, `create`, `grep`, etc.) the line range is `(None, None)`. This means that a
   view of lines 100–200 and an edit to the same file produce separate Code Block Nodes (with
   and without line ranges), which inflates the code-block count.

3. **Think-step absorption is unconditional.** All think steps are absorbed into the current
   phase, even long reasoning segments that semantically belong to a new phase. This may cause
   some phases to contain many think steps if the agent alternates reasoning and tool calls.

4. **Phase classification inherits existing heuristics.** Since `get_phase()` is used unchanged,
   all its limitations apply — e.g. complex piped commands may be misclassified, and ambiguous
   python scripts are classified by whether test hints appear in their argument paths.

5. **`has_patch` is file-path based.** It checks whether the edited file contains test hint
   substrings. This may misclassify edits to files that happen to have "test" in their path for
   unrelated reasons.

6. **Latency matching is positional.** Latencies from `metrics.response_latencies` are matched
   in order to result steps with `tool_call_metadata`. If a model produces multiple tool calls
   in a single turn, or if some steps do not correspond to distinct model calls, the latency
   assignments may be off by one.

7. **No support for agents other than OpenHands** in this first version. SWE-agent and
   mini-swe-agent graph builders are not implemented here.

8. **No difficulty lookup.** The existing Graphectory graph includes `debug_difficulty` from
   the SWE-bench Verified dataset. This graph does not include that field.

9. **File path normalization.** Paths are stored verbatim (e.g. `/workspace/django__django__3.2/
   django/db/models/query.py`). The `/workspace/{instance_id}/` prefix is not stripped. When
   comparing paths across trajectories for the same instance, callers should normalize paths
   themselves.

---

## How this graph differs from the existing Graphectory graph

| Dimension | Existing Graphectory graph | Phase-CodeBlock graph |
|-----------|---------------------------|----------------------|
| **Node granularity** | Per distinct tool invocation (action-level) | Per contiguous phase segment + per touched file/region |
| **Node deduplication** | Yes — same `(label, args, flags)` signature → one node across the entire trajectory | Phase nodes: no dedup (preserve temporal order). Code block nodes: dedup by `(file_path, start_line, end_line)` |
| **Edge semantics** | `exec` = sequential execution flow; `hier` = file-path containment hierarchy | `phase_transition` = temporal workflow flow; `phase_code_operation` = file/region access |
| **Multi-edges** | Yes — multiple exec edges between the same pair of action nodes | Yes — multiple phase_code_operation edges between same (phase, code_block) for different op types |
| **Think nodes** | Explicit `think` nodes with their own edges | No dedicated think nodes; think steps absorbed into current phase |
| **Code region tracking** | None | Core feature: Code Block Nodes with line-range support |
| **Repeated-operation signal** | Implicit in node revisit count (`step_indices` length) | Explicit in `phase_code_operation.num_actions` and `code_block.num_edits` |
| **Validation-after-patch signal** | Must be inferred from phase labels on adjacent edges | Explicit in `phase.has_patch`, `phase.has_validation`, `phase.is_after_first_patch` |
| **Termination** | No termination node; graph ends at the last action node | Explicit Termination Node with `termination_type`, `last_phase`, `last_observation_outcome` |
| **Start** | No start node | Explicit Start Node |
| **Hierarchical edges** | Yes — `hier` edges between file/path view nodes | None |
| **Supports SA / MSA** | Yes | OpenHands only (v1) |
| **Output path** | `{output_dir}/{Agent}/graphs/{model}/{instance_id}/{instance_id}.json` | `{output_root}/{model_dir}/{instance_id}/{instance_id}.json` |
| **Graph metadata** | `resolution_status`, `instance_name`, `debug_difficulty` | `resolution_status`, `instance_name`, `graph_type`, `num_phases`, `num_code_blocks` |
| **Thought lengths** | Per action node: `thought_len_raw`, `thought_len_clean` | Per phase (sum and mean): `thought_length_sum`, `thought_length_mean` |
| **Observation outcome** | Per last-action node in step: `observation_outcome` | Per phase (aggregated): `num_failed_actions`, `has_error_observation` |
| **Latency** | Not stored | Per phase (sum and mean): `latency_sum`, `latency_mean` |

### When to use each graph

- Use the **Graphectory graph** for action-sequence modeling, fine-grained command pattern
  analysis, or when you need per-action details like exact arguments and flags.
- Use the **Phase-CodeBlock graph** for coarse-grained workflow analysis, repeated-operation
  detection, validation-gap detection, or when you need interpretable phase-level features for
  trajectory outcome prediction.

---

## Collection-method legend

| Method | Description |
|--------|-------------|
| **Direct copy** | Value is read verbatim from the trajectory JSON without transformation |
| **Constant** | A fixed hard-coded value (e.g. `"start"`, `"phase_codeblock"`) |
| **Substring match** | Value is determined by scanning a text field for known keyword substrings |
| **Regex** | Value is extracted or matched using a compiled regular expression |
| **Rule-based classification** | Value is assigned by a deterministic decision tree or rule set (e.g. `get_phase()`) |
| **Structural computation** | Value is derived from the graph's own structure (node count, index, ordering) |
| **Derived aggregation** | Value is computed by summing, counting, or averaging over a list of sub-values (step ops, step lengths, etc.) |
| **Set lookup** | Value is found by testing membership in a known set (resolved/unresolved IDs from eval report) |
| **External dataset** | Value is looked up from an external data source (only used in the existing Graphectory graph for difficulty) |
