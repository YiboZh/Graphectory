"""
server/graph_builder.py

Responsible for:
  - Scanning the trajectories directory (or JSONL file) for available instances
  - Loading individual trajectories (SWE-agent .traj OR OpenHands output.jsonl)
  - Building a NetworkX graph from a trajectory (with optional cd filtering)

OpenHands output.jsonl format
------------------------------
Each line is a JSON object for one instance. The actual structure is:

  {
    "instance_id": "astropy__astropy-12907",
    "test_result":  { "git_patch": "..." },
    "metadata":     { "agent_class": "OpenHands", ... },
    "history": [
      {
        "id": 1,
        "observation": "think",           <- type of action/observation
        "content": "Your thought...",     <- tool result / observation text
        "tool_call_metadata": {
          "function_name": "think",
          "tool_call_id":  "toolu_...",
          "model_response": {
            "model": "claude-4-sonnet",
            "choices": [{
              "message": {
                "content": "I'll help you...",   <- assistant preamble / thought
                "role": "assistant",
                "tool_calls": [{
                  "index": 1,
                  "function": {
                    "name": "str_replace_editor",
                    "arguments": "{\"command\": \"view\", \"path\": \"/workspace\"}"
                  },
                  "type": "function"
                }]
              }
            }]
          }
        }
      },
      ...
    ]
  }

Each history entry represents one agent action. We normalise it to:
  {"thought": str, "action": str, "observation": str}

Where:
  - thought     = choices[0].message.content  (text preamble before the tool call)
  - action      = function_name(arguments)    (synthesised from function name + args)
  - observation = content field of the entry  (the tool result)

OpenHands report.json format
-----------------------------
Uses standard SWE-bench keys:
  {
    "resolved_ids":   ["id1", "id2", ...],
    "unresolved_ids": ["id3", ...]
  }
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

# Ensure parent directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from buildGraph import (
    GraphBuilder as _GraphBuilderBase,
    determine_resolution_status,
    check_edit_status,
    compute_thought_length_raw,
    compute_thought_length_clean,
    detect_observation_outcome,
    build_hierarchical_edges,
)

import networkx as nx


# ── Test-outcome helpers ─────────────────────────────────────────────────────

TEST_COMMANDS = {"python", "python2", "python3", "pytest", "unittest", "nosetests", "tox"}
RE_PYTEST_FAIL  = re.compile(r"\b(\d+)\s+failed\b",  re.IGNORECASE)
RE_PYTEST_ERROR = re.compile(r"\b(\d+)\s+errors?\b", re.IGNORECASE)
RE_PYTEST_PASS  = re.compile(r"\b(\d+)\s+passed\b",  re.IGNORECASE)
EXCEPTION_SIGNS = ["Traceback (most recent call last):"]


def check_command_outcome(command: str, observation: str,
                          tool: str = None, subcommand: str = None,
                          args: dict = None) -> str | None:
    """Return 'success', 'failure', or None for a command + its observation."""
    obs = observation or ""

    # Edit-status from str_replace_editor takes priority
    if tool and subcommand:
        edit_status = check_edit_status(tool, subcommand, args or {}, observation)
        if edit_status and str(edit_status).startswith("failure"):
            return "failure"
        if edit_status == "success":
            return "success"

    for sig in EXCEPTION_SIGNS:
        if sig in obs:
            return "failure"

    if RE_PYTEST_FAIL.search(obs) or RE_PYTEST_ERROR.search(obs):
        return "failure"
    if RE_PYTEST_PASS.search(obs):
        return "success"
    if "FAILURES" in obs or "ERRORS" in obs or "INTERNALERROR" in obs:
        return "failure"

    return None


# ── Extended GraphBuilder ────────────────────────────────────────────────────

class GraphBuilder(_GraphBuilderBase):
    """Extends the base GraphBuilder - no overrides needed; inherits everything."""
    pass


# ==============================================================================
# OpenHands JSONL helpers
# ==============================================================================

def _oh_extract_thought(entry: dict) -> str:
    """Extract the assistant's preamble / thought text from a history entry.

    The thought is in:
      tool_call_metadata -> model_response -> choices[0] -> message -> content

    This is the plain-text portion the model wrote before issuing the tool call.
    It may be an empty string if the model went straight to the tool call.
    """
    try:
        choices = (
            entry
            .get("tool_call_metadata", {})
            .get("model_response", {})
            .get("choices", [])
        )
        if choices:
            return choices[0].get("message", {}).get("content", "") or ""
    except Exception:
        pass
    return ""


def _oh_extract_action(entry: dict) -> str:
    """Build a synthetic action string from a history entry.

    We use the function name and its JSON-encoded arguments to reconstruct
    something the CommandParser can at least partially understand.

    Examples of what this produces:
      str_replace_editor(command="view", path="/workspace/foo.py")
      execute_bash(command="pytest tests/")
      think(thought="Let me analyse...")
    """
    try:
        tool_meta  = entry.get("tool_call_metadata", {})
        func_name  = tool_meta.get("function_name", "")

        # The arguments live in the first tool_call of the model response
        choices = (
            tool_meta
            .get("model_response", {})
            .get("choices", [])
        )
        raw_args: dict = {}
        if choices:
            tool_calls = choices[0].get("message", {}).get("tool_calls", [])
            if tool_calls:
                raw_args_str = tool_calls[0].get("function", {}).get("arguments", "{}")
                try:
                    raw_args = json.loads(raw_args_str)
                except (json.JSONDecodeError, TypeError):
                    raw_args = {}

        if not func_name:
            return ""

        if not raw_args:
            return func_name

        # Build a readable representation that the parser can work with.
        kv = ", ".join(
            f'{k}="{v}"' if isinstance(v, str) else f"{k}={json.dumps(v)}"
            for k, v in raw_args.items()
            if v is not None
        )
        return f"{func_name}({kv})"

    except Exception:
        return entry.get("tool_call_metadata", {}).get("function_name", "") or ""


def _oh_extract_observation(entry: dict) -> str:
    """Extract the observation (tool result) from a history entry.

    The result is in the top-level ``content`` field of the history entry.
    It may be a string or, occasionally, a list of content blocks.
    """
    content = entry.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", "") or str(block))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(content) if content else ""


def _normalise_openhands_history(history: list) -> list:
    """Convert an OpenHands ``history`` list to canonical step dicts.

    Each history entry with a tool_call_metadata -> one step dict:
      {"thought": str, "action": str, "observation": str}

    Entries without tool_call_metadata (bare observation entries with no
    corresponding action) are skipped.
    """
    normalised: list[dict] = []

    for entry in history:
        if not isinstance(entry, dict):
            continue

        tool_meta = entry.get("tool_call_metadata")
        if not tool_meta:
            # No action associated - skip
            continue

        thought     = _oh_extract_thought(entry)
        action      = _oh_extract_action(entry)
        observation = _oh_extract_observation(entry)

        normalised.append({
            "thought":     thought,
            "action":      action,
            "observation": observation,
        })

    return normalised


def _load_openhands_jsonl(jsonl_path: Path) -> dict[str, dict]:
    """Parse an OpenHands output.jsonl file.

    Returns a dict mapping instance_id -> traj_data where traj_data has the
    standard {"trajectory": [...]} structure expected by build_graph().
    """
    instances: dict[str, dict] = {}
    with open(jsonl_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [OH] Skipping malformed line {lineno} in {jsonl_path.name}: {exc}")
                continue

            instance_id = record.get("instance_id", "")
            if not instance_id:
                continue

            # OpenHands uses the "history" key for the list of agent steps.
            # Fall back to "trajectory" or "messages" for forward-compatibility.
            raw_history = (
                record.get("history")
                or record.get("trajectory")
                or record.get("messages")
                or []
            )

            normalised = _normalise_openhands_history(raw_history)
            instances[instance_id] = {
                "trajectory": normalised,
                "_source":    "openhands",
            }

    return instances


def _load_openhands_report(report_path: str) -> tuple[set, set]:
    """Parse an OpenHands report.json, returning (resolved_set, unresolved_set).

    Handles both the standard SWE-bench format (resolved_ids / unresolved_ids)
    and alternative OpenHands-specific key names.
    """
    resolved:   set[str] = set()
    unresolved: set[str] = set()
    try:
        with open(report_path) as f:
            report = json.load(f)

        # Standard SWE-bench keys (also used by OpenHands reports)
        resolved   = set(report.get("resolved_ids",   report.get("resolved_instances",   [])))
        unresolved = set(report.get("unresolved_ids", report.get("unresolved_instances", [])))

        # Some OH reports use a flat dict of {instance_id: bool}
        if not resolved and not unresolved:
            for k, v in report.items():
                if isinstance(v, bool):
                    (resolved if v else unresolved).add(k)

    except Exception as exc:
        print(f"  [OH] Could not parse report {report_path}: {exc}")

    return resolved, unresolved


# ==============================================================================
# Directory scanning  (supports both SWE-agent dirs and OpenHands JSONL)
# ==============================================================================

def _is_openhands_source(graphs_dir: Path) -> bool:
    """True when graphs_dir points at an OpenHands JSONL file or a dir of them."""
    if graphs_dir.is_file() and graphs_dir.suffix == ".jsonl":
        return True
    if graphs_dir.is_dir():
        has_jsonl = any(graphs_dir.rglob("*.jsonl"))
        has_traj  = any(graphs_dir.rglob("*.traj"))
        return has_jsonl and not has_traj
    return False


def _collect_openhands_jsonl_files(graphs_dir: Path) -> list[Path]:
    """Return the primary OpenHands JSONL file(s) to scan.

    OpenHands run directories contain several JSONL files:
      output.jsonl          <- full trajectories  (what we want)
      output.swebench.jsonl <- SWE-bench patches only
      patch_metrics.jsonl   <- per-patch metrics

    We prefer output.jsonl; fall back to any .jsonl that is not a known
    auxiliary file.
    """
    if graphs_dir.is_file():
        return [graphs_dir]

    # Prefer the canonical output.jsonl in any subdirectory
    primary = sorted(graphs_dir.rglob("output.jsonl"))
    if primary:
        return primary

    # Fallback: any .jsonl that doesn't look like an auxiliary file
    _SKIP_PATTERNS = {"swebench", "patch_metrics", "metrics"}
    fallback = [
        p for p in sorted(graphs_dir.rglob("*.jsonl"))
        if not any(skip in p.stem for skip in _SKIP_PATTERNS)
    ]
    return fallback


def scan_trajectories(graphs_dir: Path,
                      eval_report_path: str | None = None) -> list[dict]:
    """Return a sorted list of trajectory metadata dicts.

    Each dict has: instance_id, status, difficulty, step_count.
    Supports both SWE-agent (.traj) and OpenHands (.jsonl) sources.
    """
    if _is_openhands_source(graphs_dir):
        return _scan_openhands(graphs_dir, eval_report_path)
    return _scan_swe_agent(graphs_dir, eval_report_path)


def _scan_swe_agent(graphs_dir: Path,
                    eval_report_path: str | None) -> list[dict]:
    """Original SWE-agent scan logic."""
    resolved_set:   set[str] = set()
    unresolved_set: set[str] = set()
    if eval_report_path:
        try:
            with open(eval_report_path) as f:
                report = json.load(f)
            resolved_set   = set(report.get("resolved_ids",   []))
            unresolved_set = set(report.get("unresolved_ids", []))
        except Exception:
            pass

    results = []

    for traj_file in sorted(graphs_dir.rglob("*.traj")):
        instance_id = traj_file.stem

        if instance_id in resolved_set:
            status = "resolved"
        elif instance_id in unresolved_set:
            status = "unresolved"
        else:
            status = "unsubmitted"
            json_file = traj_file.with_suffix(".json")
            if json_file.exists():
                try:
                    with open(json_file) as f:
                        meta = json.load(f)
                    s = meta.get("graph", {}).get("resolution_status", "")
                    if s in ("resolved", "unresolved", "unsubmitted"):
                        status = s
                except Exception:
                    pass

        difficulty = "unknown"
        json_file = traj_file.with_suffix(".json")
        if json_file.exists():
            try:
                with open(json_file) as f:
                    meta = json.load(f)
                difficulty = meta.get("graph", {}).get("debug_difficulty", "unknown")
            except Exception:
                pass

        step_count = 0
        try:
            with open(traj_file) as f:
                traj = json.load(f)
            step_count = len(traj.get("trajectory", []))
        except Exception:
            pass

        results.append({
            "instance_id": instance_id,
            "status":      status,
            "difficulty":  difficulty,
            "step_count":  step_count,
        })

    return results


def _scan_openhands(graphs_dir: Path,
                    eval_report_path: str | None) -> list[dict]:
    """Scan OpenHands JSONL file(s) and return metadata list.

    Uses a lightweight scan: reads each line to get instance_id and history
    length without fully normalising the trajectory (fast for 500+ instances).
    """
    resolved_set:   set[str] = set()
    unresolved_set: set[str] = set()
    if eval_report_path:
        resolved_set, unresolved_set = _load_openhands_report(eval_report_path)

    results = []
    for jsonl_file in _collect_openhands_jsonl_files(graphs_dir):
        print(f"  [OH] Scanning {jsonl_file.name} ...")
        with open(jsonl_file, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                instance_id = record.get("instance_id", "")
                if not instance_id:
                    continue

                if instance_id in resolved_set:
                    status = "resolved"
                elif instance_id in unresolved_set:
                    status = "unresolved"
                else:
                    status = "unsubmitted"

                # Count history entries that have a tool_call_metadata (= actual
                # agent action steps) without fully normalising the trajectory.
                raw_history = (
                    record.get("history")
                    or record.get("trajectory")
                    or record.get("messages")
                    or []
                )
                step_count = sum(
                    1 for e in raw_history
                    if isinstance(e, dict) and e.get("tool_call_metadata")
                )

                results.append({
                    "instance_id": instance_id,
                    "status":      status,
                    "difficulty":  "unknown",
                    "step_count":  step_count,
                })

    results.sort(key=lambda x: x["instance_id"])
    return results


# ── Trajectory loading ──────────────────────────────────────────────────────

# Module-level cache for OpenHands JSONL data so we don't re-parse the
# (potentially 100 MB+) file on every graph request.
_OH_CACHE: dict[str, dict[str, dict]] = {}   # jsonl_path_str -> {instance_id: traj_data}


def load_trajectory(graphs_dir: Path, instance_id: str) -> dict:
    """Load and return raw trajectory data for *instance_id*.

    Supports both SWE-agent .traj files and OpenHands output.jsonl.
    Raises FileNotFoundError if the trajectory cannot be found.
    """
    if _is_openhands_source(graphs_dir):
        return _load_openhands_trajectory(graphs_dir, instance_id)
    return _load_swe_agent_trajectory(graphs_dir, instance_id)


def _load_swe_agent_trajectory(graphs_dir: Path, instance_id: str) -> dict:
    for traj_file in graphs_dir.rglob(f"{instance_id}.traj"):
        with open(traj_file) as f:
            return json.load(f)
    raise FileNotFoundError(
        f"No .traj file found for '{instance_id}' under {graphs_dir}"
    )


def _load_openhands_trajectory(graphs_dir: Path, instance_id: str) -> dict:
    global _OH_CACHE
    for jsonl_file in _collect_openhands_jsonl_files(graphs_dir):
        key = str(jsonl_file)
        if key not in _OH_CACHE:
            print(f"  [OH] Parsing {jsonl_file.name} ...")
            _OH_CACHE[key] = _load_openhands_jsonl(jsonl_file)
        instances = _OH_CACHE[key]
        if instance_id in instances:
            return instances[instance_id]
    raise FileNotFoundError(
        f"No OpenHands trajectory found for '{instance_id}' under {graphs_dir}"
    )


def _find_instance_config(graphs_dir: Path, instance_id: str) -> Path | None:
    """Locate the config YAML for a given instance (SWE-agent only)."""
    if graphs_dir.is_file():
        return None
    canonical = graphs_dir / instance_id / f"{instance_id}.config.yaml"
    if canonical.exists():
        return canonical
    for match in graphs_dir.rglob(f"{instance_id}.config.yaml"):
        return match
    return None


def _make_parser_for_instance(base_parser, graphs_dir: Path, instance_id: str):
    """Return a CommandParser loaded with the instance's tool config."""
    import copy
    from commandParser import CommandParser

    parser = CommandParser()
    parser.tool_map = copy.deepcopy(base_parser.tool_map)

    config_path = _find_instance_config(graphs_dir, instance_id)
    if config_path:
        parser.load_tool_yaml_files([str(config_path)])
        print(f"  [config] Loaded {config_path.name}")
    else:
        print(f"  [config] No config YAML found for '{instance_id}' - using base parser")

    return parser


def _accumulate_step_data(node_data: dict, step_idx: int,
                           thought: str, action: str, observation: str) -> None:
    """Append the full text of this step visit to the node's step_data list."""
    if "step_data" not in node_data:
        node_data["step_data"] = []
    node_data["step_data"].append({
        "step_idx":    step_idx,
        "thought":     thought or "",
        "action":      action  or "",
        "observation": observation or "",
    })


def _accumulate_observation(node_data: dict, observation: str) -> None:
    """Append the observation length for this step visit to the node's running list."""
    length  = len(observation)
    outcome = detect_observation_outcome(observation)

    if "observation_lengths" not in node_data:
        node_data["observation_lengths"] = []
    node_data["observation_lengths"].append(length)

    node_data["observation_length"]  = length
    node_data["observation_outcome"] = outcome


# ── Thought-continuation helper ─────────────────────────────────────────────

def _mark_thought_continuation(
    G,
    src_node: str | None,
    dst_node: str,
    prev_thought: str,
    curr_thought: str,
) -> None:
    if not src_node or not prev_thought or not curr_thought:
        return
    if prev_thought not in curr_thought:
        return
    edges = G.get_edge_data(src_node, dst_node)
    if not edges:
        return
    last_key = max(edges.keys())
    if edges[last_key].get("type") == "exec":
        edges[last_key]["is_thought_continuation"] = True


# ── Graph construction ──────────────────────────────────────────────────────

def build_graph(traj_data: dict, instance_id: str,
                eval_report_path: str, cmd_parser,
                graphs_dir: Path | None = None,
                filter_cd: bool = True):
    """Build and return a NetworkX MultiDiGraph from *traj_data*.

    Works for both SWE-agent and OpenHands trajectories.  OpenHands trajectories
    are pre-normalised by load_trajectory() into the standard
    {thought, action, observation} format, so the graph construction loop is
    identical for both sources.

    Args:
        traj_data:        Raw trajectory dict (from .traj or normalised JSONL).
        instance_id:      Instance identifier.
        eval_report_path: Path to the evaluation report JSON.
        cmd_parser:       Base CommandParser instance.
        graphs_dir:       Root directory / file for trajectory discovery.
        filter_cd:        Strip leading ``cd`` commands and mark nodes with triangle.

    Raises:
        ValueError: if cmd_parser is None.
    """
    if cmd_parser is None:
        raise ValueError(
            "cmd_parser must be a CommandParser instance. "
            "Pass a configured CommandParser from live_graph_server.setup_cmd_parser()."
        )

    # Build a per-instance parser (for SWE-agent config YAMLs; no-op for OH)
    if graphs_dir is not None and not _is_openhands_source(graphs_dir):
        instance_parser = _make_parser_for_instance(cmd_parser, graphs_dir, instance_id)
    else:
        instance_parser = cmd_parser

    try:
        from mapPhase import get_phase
    except ImportError:
        def get_phase(*_args, **_kwargs):
            return "general"

    builder    = GraphBuilder()
    trajectory = traj_data.get("trajectory", [])
    prev_phases_list: list[str] = []

    prev_thought: str = ""
    prev_step_first_node: str | None = None

    for step_idx, step in enumerate(trajectory):
        action_str  = step.get("action", "")
        thought     = step.get("thought", "") or ""
        observation = step.get("observation", "") or ""

        thought_len_raw   = compute_thought_length_raw(thought)
        thought_len_clean = compute_thought_length_clean(thought)

        # ── Pure-think steps (blank action) ────────────────────────────
        if not action_str.strip():
            node_key = builder.add_or_update_node(
                node_label         = "think",
                args               = {"thought_len": thought_len_raw},
                flags              = {},
                phase              = "general",
                step_idx           = step_idx,
                tool               = None,
                command            = None,
                subcommand         = None,
                thought_length     = thought_len_raw,
                has_cd             = False,
            )
            builder.G.nodes[node_key]["thought_len_raw"]   = thought_len_raw
            builder.G.nodes[node_key]["thought_len_clean"] = thought_len_clean
            _accumulate_observation(builder.G.nodes[node_key], observation)
            _accumulate_step_data(builder.G.nodes[node_key], step_idx,
                                  thought, action_str, observation)

            builder.add_execution_edge(
                node_key, step_idx,
                is_first_in_step=True,
                thought_length_raw=thought_len_raw,
                thought_length_clean=thought_len_clean,
            )
            _mark_thought_continuation(
                builder.G, prev_step_first_node, node_key,
                prev_thought, thought,
            )

            builder.update_previous_node(node_key)
            prev_phases_list.append("general")
            builder.prev_phases.add("general")
            prev_thought = thought
            prev_step_first_node = node_key
            continue

        # ── Parse action string ────────────────────────────────────────
        parsed_commands = instance_parser.parse(action_str)

        if not parsed_commands:
            continue

        # ── Optional cd filtering ──────────────────────────────────────
        has_cd = False
        if filter_cd and len(parsed_commands) > 1:
            first = parsed_commands[0]
            if (first.get("command") or "").strip().lower() == "cd":
                has_cd          = True
                parsed_commands = parsed_commands[1:]

        # ── Create nodes / edges ───────────────────────────────────────
        is_first_in_step  = True
        node_keys_in_step = []
        step_first_node: str | None = None

        for parsed in parsed_commands:
            tool       = (parsed.get("tool")       or "").strip()
            subcommand = (parsed.get("subcommand") or "").strip()
            command    = (parsed.get("command")    or "").strip()
            args       = parsed.get("args",  {})
            flags      = parsed.get("flags", {})

            if tool:
                node_label = f"{tool}: {subcommand}" if subcommand else tool
            else:
                node_label = command.strip() or action_str.strip()

            phase = get_phase(tool, subcommand, command, args, prev_phases_list)

            outcome = check_command_outcome(
                command=command, observation=observation,
                tool=tool, subcommand=subcommand,
                args=args if isinstance(args, dict) else {},
            )
            edit_status = check_edit_status(tool, subcommand, args, observation)
            if edit_status and isinstance(args, dict):
                args["edit_status"] = edit_status
            if outcome and isinstance(args, dict):
                args.setdefault("command_outcome", outcome)

            node_key = builder.add_or_update_node(
                node_label     = node_label,
                args           = args,
                flags          = flags,
                phase          = phase,
                step_idx       = step_idx,
                tool           = tool,
                command        = command,
                subcommand     = subcommand,
                thought_length = thought_len_raw,
                has_cd         = has_cd,
            )

            builder.G.nodes[node_key]["thought_len_raw"]   = thought_len_raw
            builder.G.nodes[node_key]["thought_len_clean"] = thought_len_clean
            _accumulate_step_data(builder.G.nodes[node_key], step_idx,
                                  thought, action_str, observation)

            node_keys_in_step.append(node_key)
            if step_first_node is None:
                step_first_node = node_key

            builder.add_execution_edge(
                node_key, step_idx,
                is_first_in_step=is_first_in_step,
                thought_length_raw=thought_len_raw if is_first_in_step else 0,
                thought_length_clean=thought_len_clean if is_first_in_step else 0,
            )

            if is_first_in_step:
                _mark_thought_continuation(
                    builder.G, prev_step_first_node, node_key,
                    prev_thought, thought,
                )

            builder.update_previous_node(node_key)
            prev_phases_list.append(phase)
            builder.prev_phases.add(phase)

            is_first_in_step = False

        # ── Mark last node of this step with observation info ─────────
        if node_keys_in_step:
            last_node = node_keys_in_step[-1]
            _accumulate_observation(builder.G.nodes[last_node], observation)

        prev_thought = thought
        prev_step_first_node = step_first_node

    # ── Post-processing ────────────────────────────────────────────────
    build_hierarchical_edges(builder.G, builder.localization_nodes)

    resolution_status = determine_resolution_status(instance_id, eval_report_path)
    builder.G.graph["resolution_status"] = resolution_status
    builder.G.graph["instance_name"]     = instance_id

    try:
        from buildGraph import difficulty_lookup
        builder.G.graph["debug_difficulty"] = difficulty_lookup.get(instance_id, "unknown")
    except Exception:
        builder.G.graph["debug_difficulty"] = "unknown"

    return builder.G