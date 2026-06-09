"""
SWE-smith / SWE-agent trajectory adapter → phase-codeblock v2 graph builder.

The HuggingFace dataset ``SWE-bench/SWE-smith-trajectories`` stores agent
trajectories as **chat messages** (system / user / assistant / tool) produced by
running SWE-agent + Claude/GPT on SWE-smith task instances.  Three prompt/parse
styles are shipped as parquet splits:

  * ``tool``   – assistant messages carry OpenAI-style ``tool_calls`` (functions
                 ``bash`` / ``str_replace_editor`` / ``submit``); observations are
                 the following ``role == "tool"`` message.
  * ``xml``    – the assistant encodes the action inside its ``content`` using
                 ``<function=NAME><parameter=KEY>VALUE</parameter></function>``
                 tags; observations are the following ``role == "user"`` message.
  * ``ticks``  – the assistant encodes the action in a trailing triple-backtick
                 fenced block, either a bare bash command or a CLI-style
                 ``str_replace_editor <subcommand> <path> [--view_range A B]
                 [--file_text '...'] [--old_str '...'] [--new_str '...']`` line;
                 observations are the following ``role == "user"`` message.

(The ``train`` split is byte-identical to ``xml`` except ``messages`` is a
structured list instead of a JSON string — it is **not** a distinct trajectory
set and is excluded from graph building; see the build report.)

This module converts EACH variant's message list into the **same** ``traj_data``
dict that the OpenHands builder consumes, then delegates to
``build_phase_codeblock_graph_v2_from_oh_trajectory`` so the emitted graphs share
the *identical* node/edge feature schema as the OpenHands v2 reference (feature
parity is a hard gate).  Nothing in the existing OpenHands pipeline is modified.

Action → OpenHands tool-call mapping (so ``get_phase`` / ``_extract_code_block_ops``
classify them correctly):

  * ``bash`` / shell / test / search commands → ``execute_bash`` {command}
  * ``str_replace_editor`` (view / str_replace / create / insert / undo_edit) →
    ``str_replace_editor`` with the OpenHands arg shape {command: <sub>, path,
    view_range, ...}
  * ``submit`` → a synthetic history step with ``action == "finish"`` (clean
    submit) so termination is classified as ``submit``.

Reasoning text (assistant ``thought`` / prose ``content``) → the model-response
``message.content`` thought.  One synthesised latency entry (value ``None``) is
recorded per assistant message so the latency machinery and the response-count
based max-step fallback behave exactly as for OpenHands.

Public API
----------
    build_phase_codeblock_graph_v2_from_swesmith_trajectory(
        messages, instance_id, output_dir, resolved=None, model=None,
        patch=None, variant=None, max_iterations=None)
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

from networkx.readwrite import json_graph

from phaseCodeBlockGraph import build_phase_codeblock_graph_v2_from_oh_trajectory

# SWE-agent / SWE-smith default per-run iteration cap (the harness stops at this
# many model turns).  Used only as the count-based max-step fallback signal.
DEFAULT_MAX_ITERATIONS = 75

# Function/tool names emitted across all three variants.
_BASH_NAMES = frozenset({"bash", "execute_bash"})
_EDITOR_NAMES = frozenset({"str_replace_editor", "edit"})
_SUBMIT_NAMES = frozenset({"submit", "finish"})

# Editor subcommands recognised by the OpenHands pipeline.
_EDITOR_SUBCMDS = frozenset(
    {"view", "create", "str_replace", "insert", "undo_edit", "delete"}
)

# XML action encoding: <function=NAME> ... </function> with nested
# <parameter=KEY>VALUE</parameter> blocks.
_XML_FN_RE = re.compile(r"<function=([a-zA-Z0-9_]+)>(.*?)</function>", re.S)
_XML_PARAM_RE = re.compile(r"<parameter=([a-zA-Z0-9_]+)>(.*?)</parameter>", re.S)

# ticks action encoding: trailing ```...``` fenced block.
_TICKS_BLOCK_RE = re.compile(r"```(?:[\w.+-]*)\n(.*?)```", re.S)


# ── Action containers ─────────────────────────────────────────────────────────


def _bash_tool_call(command: str) -> Dict[str, Any]:
    """OpenHands ``execute_bash`` tool-call dict."""
    return {
        "function": {
            "name": "execute_bash",
            "arguments": json.dumps({"command": command}),
        }
    }


def _editor_tool_call(subcommand: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """OpenHands ``str_replace_editor`` tool-call dict.

    The OpenHands ``_parse_tool_call`` pops ``command`` as the subcommand and
    keeps the remaining keys (``path``, ``view_range``, ``insert_line``, …) as
    args, so we re-assemble the args dict in exactly that shape.
    """
    payload: Dict[str, Any] = {"command": subcommand}
    payload.update(args)
    return {
        "function": {
            "name": "str_replace_editor",
            "arguments": json.dumps(payload),
        }
    }


# ── Per-variant action parsers ────────────────────────────────────────────────


def _parse_tool_message(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse a `tool`-variant assistant message → OpenHands tool-call dicts.

    Returns (tool_calls, is_submit).  The OpenAI-style ``tool_calls`` already
    use the OpenHands function shape (``bash`` → rename to ``execute_bash``;
    ``str_replace_editor`` already correct), so this is mostly a passthrough.
    """
    calls: List[Dict[str, Any]] = []
    submit = False
    for tc in msg.get("tool_calls") or []:
        fn = (tc or {}).get("function", {}) or {}
        name = fn.get("name", "") or ""
        args_raw = fn.get("arguments", "{}")
        if name in _SUBMIT_NAMES:
            submit = True
            continue
        if name in _BASH_NAMES:
            try:
                a = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except (json.JSONDecodeError, TypeError):
                a = {}
            cmd = (a.get("command") or "").strip()
            if cmd:
                calls.append(_bash_tool_call(cmd))
        elif name in _EDITOR_NAMES:
            try:
                a = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except (json.JSONDecodeError, TypeError):
                a = {}
            if not isinstance(a, dict):
                a = {}
            sub = a.pop("command", "") or ""
            calls.append(_editor_tool_call(sub, a))
    return calls, submit


def _parse_xml_content(content: str) -> List[Dict[str, Any]]:
    """Parse an `xml`-variant assistant content → tool-call dicts + submit flag."""
    calls: List[Dict[str, Any]] = []
    submit = False
    for fm in _XML_FN_RE.finditer(content or ""):
        name = fm.group(1)
        body = fm.group(2)
        params = {p.group(1): p.group(2) for p in _XML_PARAM_RE.finditer(body)}
        if name in _SUBMIT_NAMES:
            submit = True
            continue
        if name in _BASH_NAMES:
            cmd = (params.get("command") or "").strip()
            if cmd:
                calls.append(_bash_tool_call(cmd))
        elif name in _EDITOR_NAMES:
            sub = (params.get("command") or "").strip()
            args: Dict[str, Any] = {}
            if "path" in params:
                args["path"] = params["path"].strip()
            vr = params.get("view_range")
            if vr:
                rng = _parse_view_range(vr)
                if rng:
                    args["view_range"] = rng
            il = params.get("insert_line")
            if il:
                try:
                    args["insert_line"] = int(str(il).strip())
                except ValueError:
                    pass
            calls.append(_editor_tool_call(sub, args))
    return calls, submit


def _parse_ticks_content(content: str) -> List[Dict[str, Any]]:
    """Parse a `ticks`-variant assistant content → tool-call dicts + submit flag.

    The action is the *last* triple-backtick fenced block.  Its first token is
    either ``str_replace_editor`` (CLI form) or a shell command.
    """
    blocks = _TICKS_BLOCK_RE.findall(content or "")
    if not blocks:
        return [], False
    body = blocks[-1].strip()
    if not body:
        return [], False

    first_tok = body.split(None, 1)[0]
    if first_tok in _SUBMIT_NAMES:
        return [], True

    if first_tok == "str_replace_editor":
        call = _parse_ticks_editor(body)
        return ([call] if call else []), False

    # Otherwise a bash command (possibly multi-line). Submit may appear bare.
    if body in _SUBMIT_NAMES:
        return [], True
    return [_bash_tool_call(body)], False


def _parse_ticks_editor(body: str) -> Optional[Dict[str, Any]]:
    """Parse a ticks ``str_replace_editor <sub> <path> [--flags ...]`` line.

    Only the subcommand, path, view_range and insert_line are needed for phase
    classification and code-block op extraction; large ``--file_text`` /
    ``--old_str`` / ``--new_str`` payloads are intentionally ignored.
    """
    # Tokenise just the header (subcommand + path + flags) robustly. The body may
    # contain a huge multi-line --file_text/--old_str payload, so we parse the
    # first line for the subcommand/path then regex out the optional numeric args.
    head_line = body.split("\n", 1)[0]
    try:
        toks = shlex.split(head_line)
    except ValueError:
        toks = head_line.split()
    if len(toks) < 2:
        return None
    sub = toks[1] if len(toks) > 1 else ""
    if sub not in _EDITOR_SUBCMDS:
        return None
    path = ""
    if len(toks) > 2 and not toks[2].startswith("--"):
        path = toks[2]
    args: Dict[str, Any] = {}
    if path:
        args["path"] = path
    m = re.search(r"--view_range\s+(\d+)\s+(\d+)", body)
    if m:
        args["view_range"] = [int(m.group(1)), int(m.group(2))]
    m = re.search(r"--insert_line\s+(\d+)", body)
    if m:
        args["insert_line"] = int(m.group(1))
    return _editor_tool_call(sub, args)


def _parse_view_range(vr: str) -> Optional[List[int]]:
    """Parse a view_range string/list into [start, end] ints."""
    if isinstance(vr, (list, tuple)) and len(vr) == 2:
        try:
            return [int(vr[0]), int(vr[1])]
        except (ValueError, TypeError):
            return None
    nums = re.findall(r"-?\d+", str(vr))
    if len(nums) >= 2:
        return [int(nums[0]), int(nums[1])]
    return None


# ── Observation text extraction ───────────────────────────────────────────────


def _observation_text(msg: Optional[Dict[str, Any]]) -> str:
    """Return the observation text from a following tool/user message."""
    if not msg:
        return ""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _assistant_thought(msg: Dict[str, Any], variant: str) -> str:
    """Extract reasoning text from an assistant message."""
    # `tool` variant carries an explicit `thought` field.
    thought = msg.get("thought")
    if isinstance(thought, str) and thought:
        return thought
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                return block["text"]
    return ""


# ── Messages → OpenHands traj_data adapter ────────────────────────────────────


def adapt_swesmith_to_oh(
    messages: List[Dict[str, Any]],
    variant: str,
    patch: Optional[str] = None,
    max_iterations: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert one SWE-smith trajectory's messages into an OpenHands traj_data dict."""
    history: List[Dict[str, Any]] = []
    latencies: List[Dict[str, Any]] = []

    n = len(messages)
    for i, msg in enumerate(messages):
        if (msg.get("role") or "") != "assistant":
            continue

        if variant == "tool":
            calls, is_submit = _parse_tool_message(msg)
        elif variant == "xml":
            calls, is_submit = _parse_xml_content(msg.get("content", ""))
        elif variant == "ticks":
            calls, is_submit = _parse_ticks_content(msg.get("content", ""))
        else:
            raise ValueError(f"Unknown variant: {variant!r}")

        thought = _assistant_thought(msg, variant)

        # Observation text comes from the immediately-following tool/user message.
        obs_text = ""
        if i + 1 < n:
            nxt = messages[i + 1]
            if (nxt.get("role") or "") in ("tool", "user"):
                obs_text = _observation_text(nxt)

        step: Dict[str, Any] = {
            "observation": "run",
            "content": obs_text,
            "tool_call_metadata": {
                "model_response": {
                    "choices": [
                        {"message": {"content": thought, "tool_calls": calls}}
                    ]
                }
            },
        }
        if is_submit:
            step["action"] = "finish"
        history.append(step)

        # One synthesised latency entry per assistant model call (value None ok).
        latencies.append({"latency": None})

    traj_data: Dict[str, Any] = {
        "history": history,
        "metrics": {"response_latencies": latencies},
        "metadata": {"max_iterations": max_iterations or DEFAULT_MAX_ITERATIONS},
        "test_result": {"git_patch": (patch or "")},
    }
    return traj_data


# ── Public builder ────────────────────────────────────────────────────────────


def build_phase_codeblock_graph_v2_from_swesmith_trajectory(
    messages: List[Dict[str, Any]],
    instance_id: str,
    output_dir: str,
    resolved: Optional[bool] = None,
    model: Optional[str] = None,
    patch: Optional[str] = None,
    variant: Optional[str] = None,
    max_iterations: Optional[int] = None,
) -> str:
    """Build + persist a phase-codeblock v2 graph for one SWE-smith trajectory.

    Args:
        messages:     The trajectory chat-message list (parsed from the parquet
                      ``messages`` column).
        instance_id:  SWE-smith instance id (used for the on-disk layout).
        output_dir:   Directory under which ``{instance_id}/{instance_id}.json``
                      is written.
        resolved:     Whether the trajectory resolved the task; sets
                      ``resolution_status`` (resolved/unresolved).
        model:        Underlying model name (recorded on the graph metadata).
        patch:        The final git patch text (used for termination inference).
        variant:      One of ``tool`` / ``xml`` / ``ticks``.
        max_iterations: Iteration cap for max-step termination (default 75).

    Returns:
        str: Absolute path to the saved graph JSON file.
    """
    if variant not in ("tool", "xml", "ticks"):
        raise ValueError(f"variant must be tool/xml/ticks, got {variant!r}")

    traj_data = adapt_swesmith_to_oh(
        messages, variant=variant, patch=patch, max_iterations=max_iterations
    )

    json_path = build_phase_codeblock_graph_v2_from_oh_trajectory(
        traj_data=traj_data,
        instance_id=instance_id,
        output_dir=output_dir,
        eval_report_path=None,
    )

    # Patch graph-level metadata: SWE-smith ships an authoritative `resolved`
    # boolean per trajectory, so we set resolution_status directly instead of
    # consulting an eval report.
    g = json.loads(Path(json_path).read_text())
    if resolved is True:
        g["graph"]["resolution_status"] = "resolved"
    elif resolved is False:
        g["graph"]["resolution_status"] = "unresolved"
    if model:
        g["graph"]["model"] = model
    if variant:
        g["graph"]["swesmith_variant"] = variant
    with open(json_path, "w") as fh:
        json.dump(g, fh, indent=2)

    return json_path
