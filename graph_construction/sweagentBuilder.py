"""
SWE-agent (nebius/SWE-agent-trajectories) → phase-codeblock v2 graph builder.

This module adapts the SWE-agent chat-style trajectory format onto the SAME
phase-codeblock pipeline used for OpenHands v2 graphs.  It does so by *translating*
each SWE-agent (thought / action / observation) turn into the OpenHands ``traj_data``
shape that ``PhaseCodeBlockGraphBuilder.build_from_oh_trajectory(graph_version="v2")``
already consumes, then calling that builder unchanged.  This guarantees identical
node/edge feature schema (feature parity) with the reference OH v2 graph.

SWE-agent trajectory format (one row of the parquet)
---------------------------------------------------
A row carries:
  * ``instance_id``     – SWE-bench instance id (e.g. "AnalogJ__lexicon-336")
  * ``model_name``      – agent/model slug (e.g. "swe-agent-llama-70b")
  * ``target``          – bool, True iff the gold patch resolved the issue
  * ``trajectory``      – array of message dicts, each with keys
                          {cutoff_date, mask, role, system_prompt, text}
                          roles alternate system / user / ai / user / ai / ...
                          - ``ai``   step: thought text + a fenced ``` command block ```
                          - ``user`` step: the command's observation (terminal output)
  * ``exit_status``     – how the run ended (submitted, exit_context, early_exit, ...)
  * ``generated_patch`` – final git diff produced by the agent
  * ``eval_logs``       – patch-apply / eval harness logs

Action mapping (SWE-agent → OpenHands tool-call shape)
------------------------------------------------------
Each ``ai`` step's executed command is the LAST fenced code block in its text
(SWE-agent runs the last code block).  Its first token selects the mapping:

  * ``open <path> [line]``                → str_replace_editor / view  (path)
  * ``goto <line>`` / ``scroll_up`` /
    ``scroll_down`` / ``next_match``      → str_replace_editor / view  (current open file)
  * ``create <path>``                     → str_replace_editor / create (path)
  * ``edit <a>:<b> ... end_of_edit``      → str_replace_editor / str_replace (current open file)
  * ``search_file <q> [file]``            → execute_bash  grep <q> <file/openfile>
  * ``search_dir  <q> [dirs...]``         → execute_bash  grep -r <q> <dir>
  * ``find_file   <q> [dir]``             → execute_bash  find <dir> -name <q>
  * ``submit``                            → step ``action == "finish"`` (no code op)
  * anything else (python, pytest, ls,
    grep, find, cat, rm, flake8, asv, …)  → execute_bash {command: "<raw cmd>"}

The path targeted by ``edit`` / ``create`` / ``open`` / ``goto`` / ``scroll`` is NOT
inline in the command for edit/goto/scroll; SWE-agent operates on the "currently
open file".  We resolve it from the observation that follows the command (every
file-window command re-renders a ``[File: <path> (...)]`` header / ``(Open file: <path>)``
footer).  We also track the open-file across the trajectory as a fallback.

Termination-semantics mapping (SWE-agent exit_status → OH termination_type)
---------------------------------------------------------------------------
We set the adapted dict's ``error`` / ``action=="finish"`` / ``test_result.git_patch``
fields so the unchanged OH ``_detect_termination_type`` yields:

  exit_status                          → termination_type   (how we drive it)
  -----------------------------------    -----------------    ------------------------------
  submitted                            → submit             (last actionable step action="finish")
  submitted (exit_context) /
    exit_context                       → max_step           (error="...reached maximum iteration...")
  submitted (exit_format) / exit_format→ error_stop         (error="exit_format ...")
  early_exit                           → error_stop         (error="early_exit ...")
  submitted_no_patch                   → no_submit          (no finish / no error / no patch)
  (anything else, unknown)             → no_submit          (no finish / no error / no patch)

Rationale: context/format/early exits are *not* clean submissions.  Context-window
exhaustion is the SWE-agent analogue of OpenHands' iteration cap (a budget/step
limit), so it maps to ``max_step``.  Format errors and early aborts are crash-like
→ ``error_stop``.  A "submitted_no_patch" run reached the end without producing any
work, i.e. a voluntary stop with nothing to show → ``no_submit``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure sibling modules import cleanly whether run from repo root or scripts/.
_GC_DIR = Path(__file__).resolve().parent
if str(_GC_DIR) not in sys.path:
    sys.path.insert(0, str(_GC_DIR))

from phaseCodeBlockGraph import PhaseCodeBlockGraphBuilder  # noqa: E402


# ── Regexes for parsing SWE-agent text ────────────────────────────────────────

# Fenced code block (the command). ``bash`` language hint optional.
_FENCE_RE = re.compile(r"```(?:bash)?\s*\n?(.*?)```", re.DOTALL)
# File header in observations: "[File: /lexicon/reproduce.py (1 lines total)]"
_FILE_HDR_RE = re.compile(r"\[File:\s*(.*?)\s*\(")
# Footer: "(Open file: /lexicon/reproduce.py)"
_OPEN_FILE_RE = re.compile(r"\(Open file:\s*(.*?)\)")
# edit <start>:<end> ... end_of_edit
_EDIT_RE = re.compile(r"^edit\s+\d+:\d+", re.IGNORECASE)

# The OpenHands max-iteration sentinel that ``_detect_termination_type`` matches
# via ``reached maximum iteration``.  We reuse the exact phrasing so context-window
# exits are classified as ``max_step``.
_MAXITER_ERROR = (
    "Agent reached maximum iteration in headless mode. "
    "Current iteration: {n}, max iteration: {n}. (context window exhausted)"
)

# SWE-agent special interface commands that target the *currently open file*.
_VIEW_NAV_CMDS = frozenset(["goto", "scroll_up", "scroll_down", "next_match"])


def _is_na(x: Any) -> bool:
    if x is None:
        return True
    try:
        # pandas / numpy NaN floats
        import math

        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def _s(x: Any) -> str:
    return "" if _is_na(x) else str(x)


def _last_command_block(ai_text: str) -> str:
    """Return the executed command (last fenced block) from an ai-step text."""
    blocks = _FENCE_RE.findall(ai_text or "")
    for b in reversed(blocks):
        if b.strip():
            return b.strip()
    return ""


def _thought_of(ai_text: str) -> str:
    """The model reasoning = ai text with the command fence(s) removed."""
    return _FENCE_RE.sub("", ai_text or "").strip()


def _resolve_open_file(observation: str) -> Optional[str]:
    """Pull the open-file path from an observation's File header / footer."""
    if not observation:
        return None
    m = _FILE_HDR_RE.search(observation)
    if m:
        p = m.group(1).strip()
        if p and p.lower() != "n/a":
            return p
    m = _OPEN_FILE_RE.search(observation)
    if m:
        p = m.group(1).strip()
        if p and p.lower() != "n/a":
            return p
    return None


def _shquote(s: str) -> str:
    s = s.replace('"', '\\"')
    return f'"{s}"'


def _build_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Build an OpenHands-style tool_call entry (function name + JSON args)."""
    import json as _json

    return {
        "function": {
            "name": name,
            "arguments": _json.dumps(arguments),
        }
    }


def _map_action_to_tool_call(
    command: str,
    open_file: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Map one SWE-agent command string → an OH tool_call dict.

    Returns ``(tool_call_or_None, is_submit)``.  ``open_file`` is the
    currently-open file (already resolved from this step's observation when
    possible, else the tracked previous open file).
    """
    cmd = command.strip()
    if not cmd:
        return None, False

    first_line = cmd.splitlines()[0].strip()
    tok = first_line.split()[0] if first_line.split() else ""
    tok_l = tok.lower()

    # ── submit ────────────────────────────────────────────────────────────────
    if tok_l == "submit":
        return None, True

    # ── str_replace_editor: create ────────────────────────────────────────────
    if tok_l == "create":
        parts = first_line.split(None, 1)
        path = parts[1].strip() if len(parts) > 1 else (open_file or "")
        if path:
            return _build_tool_call(
                "str_replace_editor", {"command": "create", "path": path}
            ), False
        return None, False

    # ── str_replace_editor: open (view) ───────────────────────────────────────
    if tok_l == "open":
        parts = first_line.split()
        # open <path> [line] — strip trailing numeric line arg
        path = open_file or ""
        if len(parts) >= 2:
            cand = parts[1]
            path = cand
        if path:
            return _build_tool_call(
                "str_replace_editor", {"command": "view", "path": path}
            ), False
        return None, False

    # ── str_replace_editor: edit (str_replace on open file) ───────────────────
    if _EDIT_RE.match(cmd):
        if open_file:
            return _build_tool_call(
                "str_replace_editor", {"command": "str_replace", "path": open_file}
            ), False
        return None, False

    # ── view-navigation on the open file (goto/scroll/next_match) ─────────────
    if tok_l in _VIEW_NAV_CMDS:
        if open_file:
            return _build_tool_call(
                "str_replace_editor", {"command": "view", "path": open_file}
            ), False
        # no open file → treat as a no-op general step
        return _build_tool_call("execute_bash", {"command": ":"}), False

    # ── search ops → equivalent bash so they count as localization/searches ───
    if tok_l == "search_file":
        # search_file "<query>" [file]
        rest = first_line[len(tok):].strip()
        m = re.match(r'"(.*?)"\s*(.*)$', rest) or re.match(r"(\S+)\s*(.*)$", rest)
        target = ""
        if m:
            target = m.group(2).strip() or (open_file or "")
        bash = f"grep -n {_shquote(_extract_query(rest))} {target}".strip()
        return _build_tool_call("execute_bash", {"command": bash}), False

    if tok_l == "search_dir":
        rest = first_line[len(tok):].strip()
        dirs = _split_after_query(rest)
        dir_target = dirs[0] if dirs else "."
        bash = f"grep -rn {_shquote(_extract_query(rest))} {dir_target}".strip()
        return _build_tool_call("execute_bash", {"command": bash}), False

    if tok_l == "find_file":
        rest = first_line[len(tok):].strip()
        dirs = _split_after_query(rest)
        dir_target = dirs[0] if dirs else "."
        bash = f"find {dir_target} -name {_shquote(_extract_query(rest))}".strip()
        return _build_tool_call("execute_bash", {"command": bash}), False

    # ── default: raw bash command (use full multi-line command) ───────────────
    return _build_tool_call("execute_bash", {"command": cmd}), False


def _extract_query(rest: str) -> str:
    """First quoted/bare token after a search command name."""
    rest = rest.strip()
    m = re.match(r'"(.*?)"', rest)
    if m:
        return m.group(1)
    m = re.match(r"'(.*?)'", rest)
    if m:
        return m.group(1)
    parts = rest.split()
    return parts[0] if parts else ""


def _split_after_query(rest: str) -> List[str]:
    """Tokens after the (possibly quoted) query token."""
    rest = rest.strip()
    m = re.match(r'"(.*?)"\s*(.*)$', rest) or re.match(r"'(.*?)'\s*(.*)$", rest)
    if m:
        tail = m.group(2)
    else:
        parts = rest.split(None, 1)
        tail = parts[1] if len(parts) > 1 else ""
    return tail.split()


# ── SWE-agent row → OpenHands traj_data adapter ───────────────────────────────


def adapt_sweagent_to_oh(row: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str]:
    """Translate one SWE-agent parquet row into an OpenHands-style ``traj_data``.

    Returns ``(traj_data, instance_id, resolution_status)``.
    """
    instance_id = _s(row.get("instance_id"))
    trajectory = row.get("trajectory")
    if trajectory is None:
        trajectory = []
    # numpy arrays / lists both iterable
    steps_in = list(trajectory)

    exit_status = _s(row.get("exit_status")).strip().lower()
    generated_patch = _s(row.get("generated_patch")).strip()
    target = bool(row.get("target"))

    history: List[Dict[str, Any]] = []
    # iterate ai steps with their following user observation
    open_file: Optional[str] = None
    n_ai = 0
    submit_seen = False

    # Pair each ai step with the immediately-following user observation.
    for i, st in enumerate(steps_in):
        if st.get("role") != "ai":
            continue
        n_ai += 1
        ai_text = _s(st.get("text"))
        thought = _thought_of(ai_text)
        command = _last_command_block(ai_text)

        # observation = the next user step's text (if any)
        observation = ""
        for j in range(i + 1, len(steps_in)):
            nxt = steps_in[j]
            if nxt.get("role") == "user":
                observation = _s(nxt.get("text"))
                break
            if nxt.get("role") == "ai":
                break

        # Resolve the open file from THIS observation first (file-window cmds
        # re-render the header/footer), else keep tracked open_file.
        resolved = _resolve_open_file(observation)
        eff_open = resolved or open_file

        tool_call, is_submit = _map_action_to_tool_call(command, eff_open)
        if is_submit:
            submit_seen = True

        # Update tracked open-file for subsequent steps.
        if resolved:
            open_file = resolved

        # Build the OH-style history step.
        step: Dict[str, Any] = {
            "observation": "run",  # actionable obs type (not in _SKIP_OBS)
            "content": observation,
        }
        if is_submit:
            step["action"] = "finish"

        tool_calls = [tool_call] if tool_call is not None else []
        step["tool_call_metadata"] = {
            "model_response": {
                "choices": [
                    {
                        "message": {
                            "content": thought,
                            "tool_calls": tool_calls,
                        }
                    }
                ]
            }
        }
        history.append(step)

    # ── Synthesize metrics.response_latencies (one per model-response step) ────
    # SWE-agent provides no per-step latencies → latency stays None, so
    # latency_sum/mean become None like sparse OH cases (keys still present).
    response_latencies = [{"latency": None} for _ in range(n_ai)]

    # ── Drive termination_type via OH detector ────────────────────────────────
    # Priority in OH detector: finish-action > error(maxiter→max_step) > error(→error_stop)
    #                          > max_iterations count > git_patch(→submit) > no_submit
    error: Optional[str] = None
    test_result: Dict[str, Any] = {}
    # Reconcile exit_status onto OH semantics.
    if exit_status in ("submitted_no_patch",):
        # voluntary stop with no work → no_submit. Suppress finish + patch.
        submit_seen = False
        generated_patch = ""
        for hs in history:
            hs.pop("action", None)
    elif "exit_context" in exit_status or exit_status == "exit_context":
        # context-window exhaustion ≈ iteration/step cap → max_step
        error = _MAXITER_ERROR.format(n=n_ai)
        # ensure the finish action / patch don't pre-empt max_step
        submit_seen = False
        for hs in history:
            hs.pop("action", None)
    elif "exit_format" in exit_status or exit_status == "exit_format":
        error = f"early agent format failure: exit_format ({exit_status})"
        submit_seen = False
        for hs in history:
            hs.pop("action", None)
    elif exit_status == "early_exit":
        error = f"agent aborted early: early_exit"
        submit_seen = False
        for hs in history:
            hs.pop("action", None)
    elif exit_status.startswith("submitted"):
        # clean submit (already has finish action if submit cmd present); if the
        # agent stopped at a step limit without an explicit submit but left a
        # patch, the OH detector's git_patch branch yields submit too.
        if generated_patch:
            test_result["git_patch"] = generated_patch
    # else: unknown → leave as no_submit (no finish, no error, no patch)

    if not submit_seen:
        # Make sure no stray finish action remains unless we intend submit.
        if error is not None:
            for hs in history:
                hs.pop("action", None)

    if generated_patch and "git_patch" not in test_result and error is None and not submit_seen:
        # Patch exists but no explicit submit and no error: let OH infer submit.
        test_result["git_patch"] = generated_patch

    traj_data: Dict[str, Any] = {
        "instance_id": instance_id,
        "history": history,
        "metrics": {"response_latencies": response_latencies},
        "metadata": {},  # no configured max_iterations in this dataset
        "test_result": test_result,
    }
    if error is not None:
        traj_data["error"] = error

    resolution_status = "resolved" if target else "unresolved"
    return traj_data, instance_id, resolution_status


def build_phase_codeblock_graph_v2_from_sweagent_trajectory(
    row: Dict[str, Any],
    output_dir: str,
    instance_id: Optional[str] = None,
) -> str:
    """Build & save a phase-codeblock **v2** graph from one SWE-agent row.

    The row is adapted to the OpenHands ``traj_data`` shape, then handed to the
    unchanged ``PhaseCodeBlockGraphBuilder.build_from_oh_trajectory(graph_version="v2")``
    so the output carries the identical v2 feature schema.  The graph's
    ``resolution_status`` is set from the SWE-agent ``target`` label.

    Args:
        row:          One SWE-agent parquet row as a dict.
        output_dir:   Directory under which {instance_id}/{instance_id}.json is written.
        instance_id:  Optional override; defaults to row["instance_id"].

    Returns:
        str: Path to the saved graph JSON.
    """
    traj_data, iid, resolution_status = adapt_sweagent_to_oh(row)
    iid = instance_id or iid

    builder = PhaseCodeBlockGraphBuilder()
    # eval_report_path=None → builder sets resolution_status="unknown"; we override after.
    json_path = builder.build_from_oh_trajectory(
        traj_data=traj_data,
        instance_id=iid,
        output_dir=output_dir,
        eval_report_path=None,
        graph_version="v2",
    )

    # Stamp the correct resolution status (label) onto the saved graph, in the
    # same place the OH builder stores it: under the top-level "graph" attrs dict.
    import json as _json

    g = _json.loads(Path(json_path).read_text())
    g.setdefault("graph", {})["resolution_status"] = resolution_status
    Path(json_path).write_text(_json.dumps(g, indent=2))
    return json_path
