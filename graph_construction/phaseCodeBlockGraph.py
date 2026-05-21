"""
Phase-centric + Code-Block Graph Builder for OpenHands Trajectories.

Builds graphs that capture two key failure signals:
  1. Repeated operations over the same files / code blocks.
  2. Whether edited code blocks are followed by relevant validation before termination.

Graph structure:
    Start → Phase_1 → Phase_2 → ... → Phase_N → Termination
    Phase_i → CodeBlock_j  (phase_code_operation edges, one per op type per phase)

New class : PhaseCodeBlockGraphBuilder
New entry : build_phase_codeblock_graph_from_oh_trajectory()

This module lives alongside the existing GraphBuilder in buildGraph.py.
It does NOT modify or replace any existing code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from networkx.readwrite import json_graph

from buildGraph import (
    check_edit_status,
    compute_thought_length_clean,
    compute_thought_length_raw,
    detect_observation_outcome,
    determine_resolution_status,
)
from commandParser import CommandParser
from mapPhase import get_phase

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

_SEARCH_CMDS: frozenset = frozenset(["grep", "rg", "ack", "ag", "find", "locate"])
_VIEW_CMDS: frozenset = frozenset(["cat", "head", "tail", "nl", "less", "more", "bat"])
_CREATE_CMDS: frozenset = frozenset(["touch", "tee"])
_DELETE_CMDS: frozenset = frozenset(["rm", "unlink", "rmdir"])
_TEST_CMDS: frozenset = frozenset(["pytest", "python", "python3", "python2", "pylint"])

# Substrings that indicate a path is test-related
_TEST_HINTS: Tuple[str, ...] = ("test_", "reproduc", "debug", "_test", "/tests/", "/test/")

# Regex: file-path-like strings extracted from raw bash commands
_PATH_RE = re.compile(
    r"(?:^|\s)"
    r"(/[^\s'\"*{}()\[\]<>,;!|&$`\\]+"          # absolute path
    r"|[a-zA-Z0-9_][a-zA-Z0-9_./\-]*/[^\s'\"*{}()\[\]<>,;!|&$`\\]*"  # relative with slash
    r"|\./[^\s'\"*{}()\[\]<>,;!|&$`\\]+"         # ./relative
    r"|[a-zA-Z0-9_][a-zA-Z0-9_\-]*\.[a-zA-Z]{1,6})"  # filename.ext
    r"(?:\s|$|:|,|;)"
)

# Obs types to skip (system-generated, not agent actions)
_SKIP_OBS: frozenset = frozenset([None, "system", "message", "recall"])


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CodeBlockOp:
    """A single operation on a file / code region."""
    file_path: str
    start_line: Optional[int]
    end_line: Optional[int]
    op_type: str  # "view" | "search_hit" | "edit" | "create" | "delete"


@dataclass
class StepInfo:
    """Parsed information from one actionable trajectory step (result step)."""
    step_idx: int
    obs_type: str                   # "run", "think", "edit", "read/write", …
    thought: str
    thought_len_raw: int
    thought_len_clean: int
    observation: str                # raw observation / output text
    observation_length: int
    observation_outcome: str        # "success" | "failure" | "neutral"
    phase: str                      # classified phase for this step
    is_think_only: bool             # True ↔ only tool call is "think"
    code_block_ops: List[CodeBlockOp]
    is_test_step: bool              # True ↔ step executes a test command
    is_failed: bool                 # True ↔ observation_outcome == "failure"
    has_error_obs: bool             # True ↔ traceback / error: detected
    latency: Optional[float]        # response latency (seconds), None if unavailable


@dataclass
class PhaseAccumulator:
    """Accumulates stats for a contiguous phase segment."""
    phase_type: str
    phase_index: int
    start_step: int
    end_step: int
    steps: List[StepInfo] = field(default_factory=list)

    # ── Aggregated properties ────────────────────────────────────────────────

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def num_actions(self) -> int:
        return sum(1 for s in self.steps if not s.is_think_only)

    @property
    def _all_ops(self) -> List[CodeBlockOp]:
        return [op for s in self.steps for op in s.code_block_ops]

    @property
    def num_unique_files(self) -> int:
        return len({op.file_path for op in self._all_ops})

    @property
    def num_unique_code_blocks(self) -> int:
        return len({(op.file_path, op.start_line, op.end_line) for op in self._all_ops})

    @property
    def num_views(self) -> int:
        return sum(1 for op in self._all_ops if op.op_type == "view")

    @property
    def num_searches(self) -> int:
        return sum(1 for op in self._all_ops if op.op_type == "search_hit")

    @property
    def num_edits(self) -> int:
        return sum(1 for op in self._all_ops if op.op_type == "edit")

    @property
    def num_creates(self) -> int:
        return sum(1 for op in self._all_ops if op.op_type == "create")

    @property
    def num_deletes(self) -> int:
        return sum(1 for op in self._all_ops if op.op_type == "delete")

    @property
    def num_tests(self) -> int:
        return sum(1 for s in self.steps if s.is_test_step)

    @property
    def num_failed_actions(self) -> int:
        return sum(1 for s in self.steps if s.is_failed)

    @property
    def has_error_observation(self) -> bool:
        return any(s.has_error_obs for s in self.steps)

    @property
    def has_patch(self) -> bool:
        """True if any edit/create/delete op on a non-test file occurred."""
        for op in self._all_ops:
            if op.op_type in ("edit", "create", "delete"):
                fp = op.file_path.lower()
                if not any(h in fp for h in _TEST_HINTS):
                    return True
        return False

    @property
    def has_validation(self) -> bool:
        return any(s.is_test_step for s in self.steps)

    @property
    def thought_length_sum(self) -> int:
        return sum(s.thought_len_raw for s in self.steps)

    @property
    def thought_length_mean(self) -> float:
        n = len(self.steps)
        return self.thought_length_sum / n if n else 0.0

    @property
    def observation_length_sum(self) -> int:
        return sum(s.observation_length for s in self.steps)

    @property
    def observation_length_mean(self) -> float:
        n = len(self.steps)
        return self.observation_length_sum / n if n else 0.0

    @property
    def latency_sum(self) -> Optional[float]:
        lats = [s.latency for s in self.steps if s.latency is not None]
        return sum(lats) if lats else None

    @property
    def latency_mean(self) -> Optional[float]:
        lats = [s.latency for s in self.steps if s.latency is not None]
        return sum(lats) / len(lats) if lats else None


# ── Main builder class ────────────────────────────────────────────────────────

class PhaseCodeBlockGraphBuilder:
    """Build phase-centric + code-block graphs from OpenHands trajectories.

    Designed to sit alongside the existing GraphBuilder in buildGraph.py.
    No existing code is modified.

    Graph structure (node types):
        start       – single start sentinel
        phase       – one node per contiguous same-phase segment
        code_block  – one node per unique (file_path, start_line, end_line)
        termination – single termination sentinel

    Edge types:
        start_to_phase          – Start → first Phase
        phase_transition        – Phase_i → Phase_{i+1}
        phase_code_operation    – Phase → CodeBlock (one per op-type per phase)
        phase_to_termination    – last Phase → Termination
        start_to_termination    – Start → Termination (empty trajectory)
    """

    def __init__(self) -> None:
        self._cmd_parser = CommandParser()
        # No YAML tool configs loaded; OH uses JSON function calls

    # ── Public API ────────────────────────────────────────────────────────────

    def build_from_oh_trajectory(
        self,
        traj_data: Dict[str, Any],
        instance_id: str,
        output_dir: str,
        eval_report_path: Optional[str] = None,
    ) -> str:
        """Build and persist a phase-codeblock graph for one OH trajectory.

        Args:
            traj_data:         Parsed dict from one line of output.jsonl
            instance_id:       SWE-bench instance ID
            output_dir:        Directory under which {instance_id}/{instance_id}.json is written
            eval_report_path:  Optional path to eval report JSON (for resolution_status)

        Returns:
            str: Absolute path to the saved graph JSON file
        """
        history = traj_data.get("history", [])
        latencies = self._extract_latencies(traj_data)

        steps = self._parse_oh_steps(history, latencies)
        phases = self._group_into_phases(steps)
        term_type = self._detect_termination_type(traj_data, steps)
        last_obs_outcome = steps[-1].observation_outcome if steps else "unknown"

        G = self._build_graph(phases, steps, term_type, last_obs_outcome)

        # Graph-level metadata
        resolution_status = "unknown"
        if eval_report_path and Path(eval_report_path).exists():
            try:
                resolution_status = determine_resolution_status(instance_id, eval_report_path)
            except Exception as exc:
                logger.debug("Could not determine resolution status for %s: %s", instance_id, exc)

        G.graph["resolution_status"] = resolution_status
        G.graph["instance_name"] = instance_id
        G.graph["graph_type"] = "phase_codeblock"
        G.graph["num_phases"] = len(phases)
        G.graph["num_code_blocks"] = sum(
            1 for _, d in G.nodes(data=True) if d.get("node_type") == "code_block"
        )

        self._validate_graph(G, phases)

        instance_dir = Path(output_dir) / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        json_path = instance_dir / f"{instance_id}.json"

        with open(json_path, "w") as fh:
            json.dump(json_graph.node_link_data(G, edges="edges"), fh, indent=2)

        return str(json_path)

    # ── Latency extraction ────────────────────────────────────────────────────

    @staticmethod
    def _extract_latencies(traj_data: Dict[str, Any]) -> List[Optional[float]]:
        """Return response latencies in temporal order (one per model call)."""
        metrics = traj_data.get("metrics", {})
        entries = metrics.get("response_latencies", [])
        return [entry.get("latency") for entry in entries]

    # ── Step parsing ──────────────────────────────────────────────────────────

    def _parse_oh_steps(
        self,
        history: List[Dict[str, Any]],
        latencies: List[Optional[float]],
    ) -> List[StepInfo]:
        """Parse actionable (result) steps from OH history into StepInfo objects."""
        steps: List[StepInfo] = []
        lat_idx = 0
        prev_phases_set: Set[str] = set()

        for raw_step in history:
            obs_type = raw_step.get("observation")
            if obs_type in _SKIP_OBS:
                continue

            thought = self._extract_thought(raw_step)
            thought_len_raw = compute_thought_length_raw(thought)
            thought_len_clean = compute_thought_length_clean(thought)

            observation = raw_step.get("content", "") or ""
            observation_length = len(observation)
            observation_outcome = detect_observation_outcome(observation)
            is_failed = observation_outcome == "failure"
            has_error_obs = _has_error_observation(observation)

            # Extract and parse tool calls
            raw_calls = self._extract_raw_tool_calls(raw_step)
            all_parsed: List[Dict[str, Any]] = []
            for fn_name, args_raw in raw_calls:
                all_parsed.extend(self._parse_tool_call(fn_name, args_raw))

            # Determine if this is a think-only step
            is_think_only = bool(all_parsed) and all(
                p.get("tool") == "think" for p in all_parsed
            )
            if not all_parsed and obs_type == "think":
                is_think_only = True

            # Classify phase — skip think and cd-only preamble commands
            if is_think_only:
                phase = steps[-1].phase if steps else "general"
            elif all_parsed:
                phase = "general"
                for parsed in all_parsed:
                    t = parsed.get("tool", "")
                    cmd = (parsed.get("command") or "").lower().strip()
                    # Skip think and bare cd commands (preamble navigation)
                    if t == "think" or cmd == "cd":
                        continue
                    phase = get_phase(
                        t,
                        parsed.get("subcommand", ""),
                        parsed.get("command", ""),
                        parsed.get("args", {}),
                        prev_phases_set,
                        parsed.get("flags", {}),
                    )
                    break
            else:
                phase = "general"

            # Collect code-block operations
            code_block_ops: List[CodeBlockOp] = []
            is_test_step = False
            for parsed in all_parsed:
                ops, is_test = self._extract_code_block_ops(
                    tool=parsed.get("tool", ""),
                    subcommand=parsed.get("subcommand", ""),
                    command=parsed.get("command", ""),
                    args=parsed.get("args", {}),
                    flags=parsed.get("flags", {}),
                    raw_cmd=parsed.get("_raw_cmd", ""),
                )
                code_block_ops.extend(ops)
                if is_test:
                    is_test_step = True

            # Match latency: advance index for each step with a model response
            latency: Optional[float] = None
            has_model_response = bool(raw_calls)
            if has_model_response:
                latency = latencies[lat_idx] if lat_idx < len(latencies) else None
                lat_idx += 1

            step_info = StepInfo(
                step_idx=len(steps),
                obs_type=obs_type,
                thought=thought,
                thought_len_raw=thought_len_raw,
                thought_len_clean=thought_len_clean,
                observation=observation,
                observation_length=observation_length,
                observation_outcome=observation_outcome,
                phase=phase,
                is_think_only=is_think_only,
                code_block_ops=code_block_ops,
                is_test_step=is_test_step,
                is_failed=is_failed,
                has_error_obs=has_error_obs,
                latency=latency,
            )
            steps.append(step_info)
            prev_phases_set.add(phase)

        return steps

    # ── Tool-call extraction ──────────────────────────────────────────────────

    @staticmethod
    def _extract_thought(raw_step: Dict[str, Any]) -> str:
        """Extract model thought text from a history step."""
        tcm = raw_step.get("tool_call_metadata") or {}
        choices = tcm.get("model_response", {}).get("choices", [])
        for choice in choices:
            msg = choice.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            return text
        return ""

    @staticmethod
    def _extract_raw_tool_calls(raw_step: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Extract (function_name, arguments_json_str) pairs from a history step."""
        tcm = raw_step.get("tool_call_metadata") or {}
        if not tcm:
            return []

        result: List[Tuple[str, str]] = []
        choices = tcm.get("model_response", {}).get("choices", [])
        for choice in choices:
            msg = choice.get("message", {})
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_raw = fn.get("arguments", "{}")
                if name:
                    result.append((name, args_raw))

        # Some traces store tool call at top level
        if not result and tcm.get("function_name"):
            result.append((tcm["function_name"], "{}"))

        return result

    def _parse_tool_call(
        self, fn_name: str, args_raw: str
    ) -> List[Dict[str, Any]]:
        """Parse a single tool call into a list of parsed-command dicts."""
        try:
            args_loaded = json.loads(args_raw)
        except (json.JSONDecodeError, TypeError):
            args_loaded = {}

        if fn_name == "execute_bash":
            cmd = (args_loaded.get("command") or "").strip()
            if not cmd:
                return []
            try:
                parsed_list = self._cmd_parser.parse(cmd)
            except Exception:
                parsed_list = [
                    {
                        "tool": "",
                        "subcommand": "",
                        "command": "complex_command",
                        "args": {"_raw": cmd},
                        "flags": {},
                    }
                ]
            for p in parsed_list:
                p["_raw_cmd"] = cmd
            return parsed_list
        else:
            subcommand = args_loaded.pop("command", None) if isinstance(args_loaded, dict) else None
            return [
                {
                    "tool": fn_name,
                    "subcommand": subcommand or "",
                    "command": "",
                    "args": args_loaded if isinstance(args_loaded, dict) else {},
                    "flags": {},
                    "_raw_cmd": "",
                }
            ]

    # ── Code-block operation extraction ──────────────────────────────────────

    def _extract_code_block_ops(
        self,
        tool: str,
        subcommand: str,
        command: str,
        args: Any,
        flags: Dict[str, Any],
        raw_cmd: str,
    ) -> Tuple[List[CodeBlockOp], bool]:
        """Extract code-block operations from one parsed command.

        Returns:
            (ops, is_test_step)
        """
        ops: List[CodeBlockOp] = []
        is_test = False
        sub = (subcommand or "").lower()
        cmd = (command or "").lower()

        # ── str_replace_editor ────────────────────────────────────────────────
        if (tool or "").lower() == "str_replace_editor":
            path = _safe_get_str(args, "path")
            if not path:
                return ops, is_test

            if sub == "view":
                vr = _safe_get(args, "view_range")
                start, end = None, None
                if isinstance(vr, (list, tuple)) and len(vr) == 2:
                    try:
                        start, end = int(vr[0]), int(vr[1])
                    except (ValueError, TypeError):
                        pass
                ops.append(CodeBlockOp(path, start, end, "view"))

            elif sub == "str_replace":
                ops.append(CodeBlockOp(path, None, None, "edit"))

            elif sub == "create":
                ops.append(CodeBlockOp(path, None, None, "create"))

            elif sub == "insert":
                il = _safe_get(args, "insert_line")
                line = int(il) if il is not None else None
                ops.append(CodeBlockOp(path, line, line, "edit"))

            elif sub in ("undo_edit",):
                ops.append(CodeBlockOp(path, None, None, "edit"))

            elif sub == "delete":
                ops.append(CodeBlockOp(path, None, None, "delete"))

            return ops, is_test

        # ── think tool ────────────────────────────────────────────────────────
        if (tool or "").lower() == "think":
            return ops, is_test

        # ── bash commands ─────────────────────────────────────────────────────
        if not cmd:
            return ops, is_test

        paths = self._extract_paths_from_bash(cmd, args, flags, raw_cmd)

        if cmd in _SEARCH_CMDS:
            for p in paths:
                ops.append(CodeBlockOp(p, None, None, "search_hit"))

        elif cmd in _VIEW_CMDS:
            for p in paths:
                ops.append(CodeBlockOp(p, None, None, "view"))

        elif cmd in _CREATE_CMDS:
            for p in paths:
                ops.append(CodeBlockOp(p, None, None, "create"))

        elif cmd in _DELETE_CMDS:
            for p in paths:
                ops.append(CodeBlockOp(p, None, None, "delete"))

        elif cmd == "sed" and flags.get("i"):
            for p in paths:
                ops.append(CodeBlockOp(p, None, None, "edit"))

        elif cmd == "perl" and flags.get("i"):
            for p in paths:
                ops.append(CodeBlockOp(p, None, None, "edit"))

        elif cmd in _TEST_CMDS:
            # Determine if this is actually a test execution vs arbitrary python
            is_test = _is_test_execution(cmd, raw_cmd, flags)
            if is_test:
                for p in paths:
                    if any(h in p.lower() for h in _TEST_HINTS):
                        ops.append(CodeBlockOp(p, None, None, "view"))

        return ops, is_test

    def _extract_paths_from_bash(
        self,
        cmd: str,
        args: Any,
        flags: Dict[str, Any],
        raw_cmd: str,
    ) -> List[str]:
        """Extract file paths from a parsed bash command."""
        paths: List[str] = []

        # From structured args
        if isinstance(args, dict):
            for v in args.values():
                if isinstance(v, str) and _looks_like_path(v):
                    paths.append(v)
                elif isinstance(v, (list, tuple)):
                    for item in v:
                        if isinstance(item, str) and _looks_like_path(item):
                            paths.append(item)
        elif isinstance(args, (list, tuple)):
            for item in args:
                if isinstance(item, str) and _looks_like_path(item):
                    paths.append(item)
        elif isinstance(args, str) and _looks_like_path(args):
            paths.append(args)

        # Fallback: regex over raw command string
        if not paths and raw_cmd:
            for m in _PATH_RE.finditer(raw_cmd + " "):
                p = m.group(1).rstrip(":/,;")
                if _looks_like_path(p) and not p.startswith("-"):
                    paths.append(p)

        # Deduplicate while preserving order
        seen: Set[str] = set()
        result: List[str] = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                result.append(p)
        return result

    # ── Phase grouping ────────────────────────────────────────────────────────

    @staticmethod
    def _group_into_phases(steps: List[StepInfo]) -> List[PhaseAccumulator]:
        """Group consecutive same-phase steps into PhaseAccumulator objects.

        Think-only steps are absorbed into the current phase rather than
        causing transitions, since they are reasoning steps within a workflow
        phase rather than a distinct activity.
        """
        if not steps:
            return []

        phases: List[PhaseAccumulator] = []
        current: Optional[PhaseAccumulator] = None
        phase_idx = 0

        for step in steps:
            if current is None:
                current = PhaseAccumulator(
                    phase_type=step.phase,
                    phase_index=phase_idx,
                    start_step=step.step_idx,
                    end_step=step.step_idx,
                )
                phase_idx += 1
                current.steps.append(step)

            elif step.is_think_only:
                # Absorb into current phase
                current.steps.append(step)
                current.end_step = step.step_idx

            elif step.phase == current.phase_type:
                current.steps.append(step)
                current.end_step = step.step_idx

            else:
                # Phase transition
                phases.append(current)
                current = PhaseAccumulator(
                    phase_type=step.phase,
                    phase_index=phase_idx,
                    start_step=step.step_idx,
                    end_step=step.step_idx,
                )
                phase_idx += 1
                current.steps.append(step)

        if current is not None:
            phases.append(current)

        return phases

    # ── Termination detection ─────────────────────────────────────────────────

    @staticmethod
    def _detect_termination_type(
        traj_data: Dict[str, Any],
        steps: List[StepInfo],
    ) -> str:
        """Classify how the trajectory ended.

        Priority: submit > error_stop > max_step > no_submit
        """
        history = traj_data.get("history", [])

        for raw_step in history:
            if raw_step.get("action") == "finish":
                return "submit"
            if raw_step.get("observation") == "finish":
                return "submit"

        error = traj_data.get("error")
        if error:
            return "error_stop"

        metadata = traj_data.get("metadata", {})
        max_iter = metadata.get("max_iterations")
        metrics = traj_data.get("metrics", {})
        n_responses = len(metrics.get("response_latencies", []))

        if max_iter:
            n_actions = sum(1 for s in steps if not s.is_think_only)
            if n_actions >= max_iter or n_responses >= max_iter:
                return "max_step"

        return "no_submit"

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(
        self,
        phases: List[PhaseAccumulator],
        steps: List[StepInfo],
        term_type: str,
        last_obs_outcome: str,
    ) -> nx.MultiDiGraph:
        """Assemble the full NetworkX graph from parsed phase and step data."""
        G = nx.MultiDiGraph()
        counter = _NodeCounter()

        # ── Start Node ────────────────────────────────────────────────────────
        start_key = counter.next("start")
        G.add_node(start_key, node_type="start", step_idx=0)

        prev_key = start_key

        # Global code-block tracking
        cb_key_map: Dict[Tuple, str] = {}           # (fp, sl, el) → node_key
        cb_stats: Dict[str, Dict[str, int]] = {}    # node_key → {op_type: count}

        # ── Phase + CodeBlock Nodes ───────────────────────────────────────────
        first_patch_seen = False

        for phase_acc in phases:
            is_after_first_patch = first_patch_seen
            if phase_acc.has_patch:
                first_patch_seen = True

            phase_key = counter.next(f"phase_{phase_acc.phase_index}")
            G.add_node(
                phase_key,
                node_type="phase",
                phase_type=phase_acc.phase_type,
                phase_index=phase_acc.phase_index,
                start_step=phase_acc.start_step,
                end_step=phase_acc.end_step,
                num_steps=phase_acc.num_steps,
                num_actions=phase_acc.num_actions,
                num_unique_files=phase_acc.num_unique_files,
                num_unique_code_blocks=phase_acc.num_unique_code_blocks,
                num_views=phase_acc.num_views,
                num_searches=phase_acc.num_searches,
                num_edits=phase_acc.num_edits,
                num_creates=phase_acc.num_creates,
                num_deletes=phase_acc.num_deletes,
                num_tests=phase_acc.num_tests,
                num_failed_actions=phase_acc.num_failed_actions,
                has_error_observation=phase_acc.has_error_observation,
                has_patch=phase_acc.has_patch,
                has_validation=phase_acc.has_validation,
                is_after_first_patch=is_after_first_patch,
                thought_length_sum=phase_acc.thought_length_sum,
                thought_length_mean=phase_acc.thought_length_mean,
                observation_length_sum=phase_acc.observation_length_sum,
                observation_length_mean=phase_acc.observation_length_mean,
                latency_sum=phase_acc.latency_sum,
                latency_mean=phase_acc.latency_mean,
            )

            # Edge: previous node → this phase
            if prev_key == start_key:
                G.add_edge(start_key, phase_key, edge_type="start_to_phase")
            else:
                prev_data = G.nodes[prev_key]
                G.add_edge(
                    prev_key,
                    phase_key,
                    edge_type="phase_transition",
                    from_phase=prev_data.get("phase_type", ""),
                    to_phase=phase_acc.phase_type,
                    step_gap=phase_acc.start_step - prev_data.get("end_step", 0),
                )

            prev_key = phase_key

            # Ensure code-block nodes exist and accumulate global op stats
            for op in phase_acc._all_ops:
                cb_tuple = (op.file_path, op.start_line, op.end_line)
                if cb_tuple not in cb_key_map:
                    fp_hash = hashlib.md5(op.file_path.encode()).hexdigest()
                    tag = f"{fp_hash[:8]}:{op.start_line}:{op.end_line}"
                    cb_node_key = counter.next(f"code_block:{tag}")
                    G.add_node(
                        cb_node_key,
                        node_type="code_block",
                        file_path=op.file_path,
                        file_path_hash=fp_hash,
                        start_line=op.start_line,
                        end_line=op.end_line,
                        num_views=0,
                        num_search_hits=0,
                        num_edits=0,
                    )
                    cb_key_map[cb_tuple] = cb_node_key
                    cb_stats[cb_node_key] = defaultdict(int)

                cb_node_key = cb_key_map[cb_tuple]
                cb_stats[cb_node_key][op.op_type] += 1

            # Phase → CodeBlock edges: one edge per (phase, code_block, op_type)
            # Aggregate op counts across all ops in this phase
            edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)
            for op in phase_acc._all_ops:
                cb_tuple = (op.file_path, op.start_line, op.end_line)
                cb_node_key = cb_key_map[cb_tuple]
                edge_counts[(cb_node_key, op.op_type)] += 1

            for (cb_node_key, op_type), num_actions in edge_counts.items():
                G.add_edge(
                    phase_key,
                    cb_node_key,
                    edge_type="phase_code_operation",
                    operation_type=op_type,
                    num_actions=num_actions,
                )

        # Update code-block node aggregated stats
        for cb_node_key, stats in cb_stats.items():
            G.nodes[cb_node_key]["num_views"] = stats.get("view", 0)
            G.nodes[cb_node_key]["num_search_hits"] = stats.get("search_hit", 0)
            G.nodes[cb_node_key]["num_edits"] = (
                stats.get("edit", 0) + stats.get("create", 0) + stats.get("delete", 0)
            )

        # ── Termination Node ──────────────────────────────────────────────────
        last_phase_type = phases[-1].phase_type if phases else "none"
        final_step_idx = steps[-1].step_idx if steps else 0

        term_key = counter.next("termination")
        G.add_node(
            term_key,
            node_type="termination",
            termination_type=term_type,
            final_step_idx=final_step_idx,
            last_phase=last_phase_type,
            last_observation_outcome=last_obs_outcome,
        )

        edge_type = "phase_to_termination" if prev_key != start_key else "start_to_termination"
        G.add_edge(prev_key, term_key, edge_type=edge_type)

        return G

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_graph(
        G: nx.MultiDiGraph, phases: List[PhaseAccumulator]
    ) -> None:
        """Lightweight structural validation. Raises ValueError on failure."""
        nodes = dict(G.nodes(data=True))

        start_nodes = [k for k, v in nodes.items() if v.get("node_type") == "start"]
        term_nodes = [k for k, v in nodes.items() if v.get("node_type") == "termination"]
        phase_nodes = [k for k, v in nodes.items() if v.get("node_type") == "phase"]

        if len(start_nodes) != 1:
            raise ValueError(
                f"Expected exactly 1 start node, found {len(start_nodes)}"
            )
        if len(term_nodes) != 1:
            raise ValueError(
                f"Expected exactly 1 termination node, found {len(term_nodes)}"
            )
        if phases and not phase_nodes:
            raise ValueError("Trajectory has phases but no phase nodes were added")

        # Check temporal ordering
        if len(phase_nodes) > 1:
            pdata = sorted(
                (G.nodes[k]["phase_index"], G.nodes[k]["start_step"], G.nodes[k]["end_step"])
                for k in phase_nodes
            )
            for i, (pi, ss, _) in enumerate(pdata):
                if pi != i:
                    raise ValueError(
                        f"Phase indices not sequential: {[p[0] for p in pdata]}"
                    )
                if i > 0 and ss < pdata[i - 1][2]:
                    raise ValueError(
                        f"Phase {pi} start_step ({ss}) overlaps previous end_step ({pdata[i-1][2]})"
                    )


# ── Module-level builder function ─────────────────────────────────────────────

def build_phase_codeblock_graph_from_oh_trajectory(
    traj_data: Dict[str, Any],
    instance_id: str,
    output_dir: str,
    eval_report_path: Optional[str] = None,
) -> str:
    """Build and save a phase-codeblock graph from one OpenHands trajectory.

    This is the primary entry point for external callers.

    Args:
        traj_data:         Parsed dict from one line of an OH output.jsonl file.
        instance_id:       SWE-bench instance ID (e.g. "django__django-12345").
        output_dir:        Directory where {instance_id}/{instance_id}.json will be written.
        eval_report_path:  Optional path to report.json for resolution_status lookup.

    Returns:
        str: Path to the saved graph JSON file.
    """
    builder = PhaseCodeBlockGraphBuilder()
    return builder.build_from_oh_trajectory(
        traj_data=traj_data,
        instance_id=instance_id,
        output_dir=output_dir,
        eval_report_path=eval_report_path,
    )


# ── Private helpers ───────────────────────────────────────────────────────────

class _NodeCounter:
    """Simple counter for generating unique node keys."""

    def __init__(self) -> None:
        self._count = 0

    def next(self, label: str) -> str:
        key = f"{self._count}:{label}"
        self._count += 1
        return key


def _safe_get(d: Any, key: str) -> Any:
    """Return d[key] if d is a dict, else None."""
    return d.get(key) if isinstance(d, dict) else None


def _safe_get_str(d: Any, key: str) -> str:
    """Return d[key] as string if d is a dict, else ''."""
    v = _safe_get(d, key)
    return v if isinstance(v, str) else ""


def _looks_like_path(s: str) -> bool:
    """Heuristic: return True if s looks like a file or directory path."""
    if not s or len(s) < 2:
        return False
    if s.startswith(("-", "=", "#", "@", "%", "'")):
        return False
    if s.startswith(("/", "./", "../", "~/")):
        return True
    if "/" in s and not s.startswith("-"):
        return True
    if re.search(r"\.[a-zA-Z]{1,6}$", s) and "/" not in s and not s.startswith("-"):
        # single filename.ext (no directory component)
        return len(s) > 4
    return False


def _has_error_observation(observation: str) -> bool:
    """Return True if observation text indicates an error."""
    if not observation:
        return False
    obs_lower = observation.lower()
    return any(
        sign in obs_lower
        for sign in (
            "traceback (most recent call last)",
            "error:",
            "exception:",
            "syntaxerror",
            "nameerror",
            "typeerror",
            "attributeerror",
            "importerror",
            "modulenotfounderror",
        )
    )


def _is_test_execution(cmd: str, raw_cmd: str, flags: Dict[str, Any]) -> bool:
    """Return True if the command appears to be a test execution."""
    if cmd == "pytest":
        return True
    if cmd in ("python", "python3", "python2"):
        raw_lower = raw_cmd.lower()
        # python -m pytest ...
        if "-m" in raw_lower and "pytest" in raw_lower:
            return True
        # python test_*.py
        if re.search(r"test_\w+\.py", raw_lower):
            return True
        # pytest via flags
        m_flag = flags.get("m", "")
        if isinstance(m_flag, str) and "pytest" in m_flag.lower():
            return True
    return False
