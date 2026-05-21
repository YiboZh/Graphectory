# Graphectory Graph Features Reference

This document describes every attribute stored on **nodes**, **edges**, and **graph-level metadata** in Graphectory trajectory graphs, and **how each value is collected** (regex, substring match, rule-based logic, structural computation, external lookup, etc.).

Graphs are built in `graph_construction/buildGraph.py` (batch JSON via `generatejson.py`) and `graph_construction/server/graph_builder.py` (live browser viewer). Unless noted, behavior refers to the **batch** pipeline; differences for the **live server** are called out in [Batch vs live server](#batch-vs-live-server).

**On-disk format:** NetworkX **node-link JSON** (`directed`, `multigraph`, `graph`, `nodes`, `edges`).

---

## Table of contents

1. [Graph-level metadata](#graph-level-metadata)
2. [Node features](#node-features)
3. [Edge features](#edge-features)
4. [How phases are assigned](#how-phases-are-assigned)
5. [Command parsing pipeline](#command-parsing-pipeline)
6. [Batch vs live server](#batch-vs-live-server)
7. [Collection method legend](#collection-method-legend)

---

## Graph-level metadata

Stored on the root `graph` object in the JSON file (not on individual nodes).

| Field | Type | Description | Collection method |
|-------|------|-------------|-------------------|
| `resolution_status` | string | Whether the SWE-bench instance was solved: `resolved`, `unresolved`, or `unsubmitted` | **Set lookup** — `instance_id in eval_report["resolved_ids"]` or `unresolved_ids` (`determine_resolution_status` in `buildGraph.py`) |
| `instance_name` | string | SWE-bench instance ID (e.g. `django__django-10973`) | **Direct copy** from trajectory processing argument |
| `debug_difficulty` | string | Human-readable difficulty from SWE-bench Verified | **External dataset** — `datasets.load_dataset("princeton-nlp/SWE-bench_Verified")` → `difficulty_lookup[instance_id]`, default `"unknown"` |

---

## Node features

Nodes represent **deduplicated actions**: the same `(label, args, flags)` signature merges into one node; each revisit appends to list fields.

### Node identity

| Field | Type | Description | Collection method |
|-------|------|-------------|-------------------|
| `id` | string | Unique node key, e.g. `"3:str_replace_editor: view"` | **Counter + label** — `f"{len(self.G.nodes)}:{node_label}"` at first creation (`add_or_update_node`) |

### Action identity & parsing

| Field | Type | Description | Collection method |
|-------|------|-------------|-------------------|
| `label` | string | Human-readable action name, e.g. `"str_replace_editor: view"`, `"grep"`, `"think"` | **String formatting** — `f"{tool}: {subcommand}"` if tool present, else command or raw action string |
| `tool` | string \| null | SWE-agent / tool-registry name | **CommandParser** or **JSON tool call** (OpenHands `function.name`; MSA function args) |
| `subcommand` | string \| null | Tool subcommand (e.g. `view`, `str_replace`) | **CommandParser** / tool YAML enum / OpenHands args `command` key |
| `command` | string | Shell verb when no tool wrapper | **bashlex AST** + **shlex** (`CommandParser`); or first token fallback for unparsed MSA commands |
| `args` | object | Parsed arguments (paths, `old_str`, `new_str`, `view_range`, etc.) | **Tool YAML specs** (`ToolDefinition.parse`), **bashlex** walk, **heredoc regex** extraction; may include injected `edit_status` |
| `flags` | object | CLI flags (e.g. `sed -n` → `{n: true}`) | **CommandParser** token scan (`--flag`, short flags) |

**Dedup signature** (not stored on node, used internally): MD5 of sorted JSON `{"label", "args", "flags"}` (`hash_node_signature`).

### Temporal & phase (per visit)

| Field | Type | Description | Collection method |
|-------|------|-------------|-------------------|
| `step_indices` | list[int] | Trajectory step index for each time this node was executed | **Incrementing counter** per agent step loop; **append** on revisit |
| `phases` | list[string] | Phase label per visit, aligned with `step_indices` | **Inferred** by `get_phase()` in `mapPhase.py` — **not** read from raw trajectory (see [How phases are assigned](#how-phases-are-assigned)) |
| `thought_lengths` | list[int] | Raw thought character count per visit | **`len(thought)`** (`compute_thought_length_raw`) |

### Reasoning length (scalars on node)

| Field | Type | Description | Collection method |
|-------|------|-------------|-------------------|
| `thought_len_raw` | int | Character count of thought text for this step | **`len(thought)`** |
| `thought_len_clean` | int | Thought length after stripping quoted / backtick blocks | **Regex removal** then length — `re.sub` for ` ```…``` `, `` `…` ``, `"…"`, `'…'` (`compute_thought_length_clean`) |

### Observation feedback (usually on last node of a step)

| Field | Type | Description | Collection method |
|-------|------|-------------|-------------------|
| `observation_length` | int | Character length of tool observation text | **`len(observation)`** on the **last** node created in that step |
| `observation_outcome` | string | `"success"`, `"failure"`, or `"neutral"` | **Substring match** on lowercased observation (`detect_observation_outcome`) — see table below |

**`detect_observation_outcome` substring rules** (`buildGraph.py`):

| Outcome | Substrings searched (case-insensitive) |
|---------|----------------------------------------|
| `failure` | `traceback (most recent call last)`, `error:`, `exception:`, `failed`, `failure`, `assertion`, `syntaxerror`, `nameerror`, `typeerror` |
| `success` | `success`, `passed`, `has been edited`, `created successfully` |
| `neutral` | (default if no match) |

Failure is checked **before** success (first match wins).

### Edit outcome (inside `args`)

| Field | Type | Description | Collection method |
|-------|------|-------------|-------------------|
| `args.edit_status` | string \| absent | Result of `str_replace_editor: str_replace` only | **Exact substring match** on observation (`check_edit_status`) |

| Value | Observation substring / condition |
|-------|-----------------------------------|
| `"success"` | `"has been edited."` |
| `"failure: not found"` | `"did not appear verbatim"` |
| `"failure: multiple occurrences"` | `"Multiple occurrences of old_str"` |
| `"failure: no change"` | `"old_str"` and `"is the same as new_str"` both present |
| `"failure: unknown"` | Edit tool but no known pattern |
| (absent) | Not a `str_replace` edit |

### Shell / step context

| Field | Type | Description | Collection method |
|-------|------|-------------|-------------------|
| `has_cd` | bool | Step included a `cd` command before this sub-action | **Command name equality** — `command.lower() == "cd"` seen earlier in same step; flag set on subsequent nodes in that step |

### Think-only nodes

| Field | Type | Description | Collection method |
|-------|------|-------------|-------------------|
| `args.thought_len` | int | (think nodes only) Duplicate of raw thought length | **`len(thought)`** when `action` is blank (SA) or tool is `think` (OH) |

### Live-server-only node fields

Present when graphs are built via `live_graph_server.py` / `server/graph_builder.py`, **not** in typical batch JSON from `generatejson.py`:

| Field | Type | Description | Collection method |
|-------|------|-------------|-------------------|
| `step_data` | list[object] | Per-visit `{step_idx, thought, action, observation}` for UI sidebar | **Direct copy** from trajectory fields (`_accumulate_step_data`) |
| `observation_lengths` | list[int] | Length per revisit | **Append** `len(observation)` (`_accumulate_observation`) |

### Common `args` keys (from parsing, not separate node fields)

These live inside `args` and vary by tool/command:

| Key | Typical source |
|-----|----------------|
| `path` | Tool YAML / bashlex positional args |
| `view_range` | `[start, end]` integers for `str_replace_editor view` |
| `old_str`, `new_str` | Edit commands |
| `file_text` | `create` commands |
| `_raw` | Unparsed MSA command string fallback |
| `edit_status` | Injected by `check_edit_status` (substring) |

---

## Edge features

Graphectory uses a **`MultiDiGraph`**: parallel edges between the same pair are allowed (`key` in JSON).

### Edge type: `exec` (execution flow)

Links the previous action node to the current one along the trajectory spine.

| Field | Type | Description | Collection method |
|-------|------|-------------|-------------------|
| `type` | string | Always `"exec"` | **Constant** |
| `label` | string | Step index as string | **`str(step_idx)`** |
| `is_first_in_step` | bool | First sub-command in a multi-command (`&&`) step | **Boolean flag** in step loop |
| `thought_length_raw` | int | Thought length carried on this edge | **Copied from step** if `is_first_in_step`, else `0` |
| `thought_length_clean` | int | Cleaned thought length on edge | Same as above, from `compute_thought_length_clean` |

**Creation:** `GraphBuilder.add_execution_edge()` chains `previous_node → current node` after each parsed sub-command.

### Edge type: `hier` (localization hierarchy)

Links `str_replace_editor: view` nodes when one view is structurally “inside” another.

| Field | Type | Description | Collection method |
|-------|------|-------------|-------------------|
| `type` | string | Always `"hier"` | **Constant** |
| `label` | string | Usually `""` | **Constant** |

**Creation:** `build_hierarchical_edges()` at finalize time — **structural / geometric**, no text regex:

1. **Directory containment** — `pathlib.Path` prefix: parent path parts ⊆ child path parts; connect to **closest** parent only (transitive reduction).
2. **Line-range nesting** — For same file, `[b1,b2]` nested in `[a1,a2]` if `b1 >= a1` and `b2 <= a2`; immediate parent only.
3. **Whole-file → range** — Path-only view nodes linked to outermost range views.

Requires `args.path` and optional integer `args.view_range` of length 2.

### Edge type: `thought` (batch code path — currently unused in output)

| Field | Type | Description | Collection method |
|-------|------|-------------|-------------------|
| `type` | string | `"thought"` | Would be added by `GraphBuilder.track_thought()` |
| `label` | string | `""` | **Constant** |

**Collection:** **`str.startswith`** — if `thought_text.startswith(prev_thought)` (`track_thought` in `buildGraph.py`).

> **Note:** `track_thought()` is **never called** in the batch builders today, so **batch JSON graphs do not contain `type: "thought"` edges**. Precomputed samples confirm this.

### Live-server thought continuation (on `exec` edges)

Instead of separate `thought` edges, the live server may set:

| Field | Type | Description | Collection method |
|-------|------|-------------|-------------------|
| `is_thought_continuation` | bool | Exec edge marks reused/extended reasoning | **Substring:** `prev_thought in curr_thought` (both non-empty) (`_mark_thought_continuation` in `server/graph_builder.py`) |

---

## How phases are assigned

**Phases are not fields in raw trajectories.** They are computed at graph-build time by `get_phase(tool, subcommand, command, args, prev_phases, flags)` in `mapPhase.py`, using:

- Parsed command structure from `CommandParser`
- **`prev_phases`**: a **set** of phase labels seen so far in the trajectory (used for the “key rule”: test-related work before any `patch` → `localization`; after `patch` → `validation`)

### Phase values

| Phase | Meaning |
|-------|---------|
| `localization` | Reading/searching/exploring before patching (or test work before first patch) |
| `patch` | Creating/editing non-test files |
| `validation` | Tests or test-file work after a patch has occurred |
| `general` | Think steps, unknown commands, fallbacks |

### `get_phase` collection methods (by mechanism)

| Mechanism | Used for |
|-----------|----------|
| **Tool/subcommand tables** | `str_replace_editor` + `view` → localization; `create`/`str_replace`/… → patch |
| **Command name set** | `READONLY_CMDS`, `PY_CMDS`, `EDIT_CMDS` membership |
| **Substring (`TEST_HINTS`)** | `test_`, `_test`, `/tests/`, `reproduc`, `debug`, etc. in paths |
| **Regex** | `_PATHISH = re.compile(r"(^[/~.]|/|\.py$)")` to extract path-like tokens from args |
| **Token / substring redirection** | `>`, `>>`, `<<`, `tee`, embedded `" >>"` in tokens (`_contains_redirection`) |
| **`str.startswith` on tokens** | `>`, `>>`, `1>`, `2>` |
| **Flag inspection** | `sed -n` without `-i` → read-only; `perl -i` → edit |
| **Pipe detection** | `\|` in tokens for piped read-only ops |
| **AST (`ast.parse`)** | Detect file writes in `python -c` / heredoc inline code (`_extract_edited_files_from_python_code`) |
| **Substring code sniff** | Heredoc body treated as code if `'Path(' in item`, `'open(' in item`, `'write' in item`, etc. |
| **Prior-phase set** | `_has_prior_patch(prev_phases)` |

Blank SWE-agent actions and explicit `think` tool steps are assigned **`general`** in the builders (not via `get_phase`).

---

## Command parsing pipeline

Most node fields (`tool`, `command`, `args`, `flags`, `label`) depend on **`CommandParser`** (`commandParser.py`).

| Stage | Method | Technology |
|-------|--------|------------|
| Primary parse | `bashlex.parse()` | **AST** — respects quotes, nesting |
| Fallback | `shlex.split` | Whitespace / quoting |
| Tool commands | `ToolDefinition` + YAML | **Schema-driven** (`data/SWE-agent/tools/*/config.yaml`) |
| Heredoc detection | `_is_simple_heredoc` | **Regex** `<<-?\s*([\'"]?)\w+\1`; control-word regex; compound `&&`/`;` checks |
| Heredoc body split | `_parse_heredoc` | **Regex** delimiter + `re.match(r'>\s*([^\n]+)')` for redirect |
| Env prefix | `_parse_with_env` | **Regex** `^((?:\w+=...)+)(.+)?` |
| Complex commands | `is_complex` | **bashlex** + top-level control structure detection |

### Agent-specific trajectory → step extraction

| Agent | Trajectory keys | Thought / action / observation extraction |
|-------|-----------------|-------------------------------------------|
| **SWE-agent** | `trajectory[]` → `action`, `thought`, `observation` | **Direct JSON fields** |
| **OpenHands** | `history[]` | Thought: `content`; action: `tool_call_metadata` → `execute_bash` parsed or other tools as single node; observation: `content` |
| **mini-swe-agent** | `messages[]` | Thought: nested `output[].content`; action: `function_call.arguments.command`; observation: next message `output` or `extra.raw_output` |
| **MSA v1.0** | `messages[]` | **Regex:** `THOUGHT:\s*(.*?)`, ` ```bash\s*(.*?)``` `, `<output>(.*?)</output>` |

---

## Batch vs live server

| Feature | Batch (`buildGraph.py` + `generatejson.py`) | Live server (`server/graph_builder.py`) |
|---------|---------------------------------------------|----------------------------------------|
| `observation_outcome` | `detect_observation_outcome` (substring only) | Same helper + optional **`check_command_outcome`** (pytest **regex**, traceback strings) when building in memory — **not** written to batch JSON |
| Thought continuation | `track_thought()` exists but **unused** | `is_thought_continuation` on **exec** edges (`prev_thought in curr_thought`) |
| `step_data`, `observation_lengths` | Absent | Present for UI |
| `args.command_outcome` | Not set in batch | May be set via `check_command_outcome` in live path |

**Pytest regex (live only)** — `server/graph_builder.py`:

```python
RE_PYTEST_FAIL  = re.compile(r"\b(\d+)\s+failed\b",  re.IGNORECASE)
RE_PYTEST_ERROR = re.compile(r"\b(\d+)\s+errors?\b", re.IGNORECASE)
RE_PYTEST_PASS  = re.compile(r"\b(\d+)\s+passed\b",  re.IGNORECASE)
# Also: "FAILURES", "ERRORS", "INTERNALERROR" in obs (substring)
```

---

## Collection method legend

| Symbol | Meaning |
|--------|---------|
| **Direct copy** | Value taken verbatim from trajectory JSON |
| **Counter / append** | Deterministic integer or list growth |
| **String formatting** | Concatenation / template of parsed fields |
| **Set lookup** | Membership in eval report or dataset |
| **Substring match** | `needle in haystack.lower()` or exact phrase match |
| **Regex** | `re.search`, `re.sub`, `re.match`, `re.compile` |
| **`str.startswith` / `in`** | Prefix or containment on strings |
| **Rule / table** | Fixed command lists and tool subcommand maps |
| **bashlex / AST** | Shell or Python abstract syntax |
| **Structural** | Path parts, integer ranges, graph topology |
| **External dataset** | HuggingFace `datasets` load |
| **Inferred** | Derived from other fields + history (`get_phase`) |
| **Unused in batch** | Implemented but not invoked in batch export path |

---

## Quick reference: what uses regex vs substring?

| Feature | Regex | Substring | Other |
|---------|:-----:|:---------:|-------|
| `thought_len_clean` | ✓ (strip blocks) | | length |
| `observation_outcome` | | ✓ | |
| `args.edit_status` | | ✓ (exact phrases) | |
| `phases` | partial (paths) | ✓ (`TEST_HINTS`) | rules + AST |
| `args` / `command` / `flags` | ✓ (heredoc, env) | | bashlex, YAML |
| MSA v1.0 thought/action/obs | ✓ | | |
| `hier` edges | | | Path + range geometry |
| `exec` edges | | | Step chaining |
| Thought continuation (live) | | ✓ (`in`) | |
| Graph `resolution_status` | | | JSON list lookup |
| Graph `debug_difficulty` | | | HF dataset |

---

## Source files

| File | Role |
|------|------|
| `graph_construction/buildGraph.py` | `GraphBuilder`, node/edge creation, observation/edit helpers, hier edges, SA/OH/MSA builders |
| `graph_construction/mapPhase.py` | `get_phase()` classification |
| `graph_construction/commandParser.py` | Bash / tool parsing |
| `graph_construction/generatejson.py` | Batch export orchestration |
| `graph_construction/server/graph_builder.py` | Live graph build, pytest outcome, `step_data` |
| `graph_construction/server/graph_renderer.py` | Display-only label heuristics (not stored in batch JSON) |
| `graph_analysis/analyzer.py` | Reads stored features; computes derived metrics (no regex on raw text) |

---

## Related reading

- [graph_construction/README.md](../graph_construction/README.md) — live server UI and visual semantics
- [README.md](../README.md) — project overview and CLI examples
