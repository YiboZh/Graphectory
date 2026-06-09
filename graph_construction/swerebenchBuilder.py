"""
SWE-rebench-OpenHands trajectory → phase-codeblock v2 graph builder.

The ``nebius/SWE-rebench-openhands-trajectories`` dataset stores each run as a
flat list of OpenAI-style **chat messages** (``role`` ∈ {system, user,
assistant, tool}) rather than the native OpenHands ``history`` list that the
reference builder (:mod:`phaseCodeBlockGraph`) consumes.

This module provides a thin *adapter* that re-packages each chat trajectory into
the exact ``traj_data`` shape the reference v2 builder expects, then delegates to
``PhaseCodeBlockGraphBuilder.build_from_oh_trajectory(..., graph_version="v2")``.
Because all aggregation / phase / code-block logic is reused unchanged, the
emitted graphs carry the IDENTICAL node/edge feature schema as the OpenHands v2
reference (feature parity holds by construction).

Adapter mapping (chat message → OH history step)
------------------------------------------------
The chat list is a strict alternation of::

    assistant(content=thought, tool_calls=[{function:{name,arguments}}])
    tool(name=<tool fn name>, content=<observation text>, tool_call_id=...)

Each such (assistant, tool) pair becomes one OH history step::

    {
      "observation": <tool fn name>,           # OH obs-type marker
      "content":     <tool message content>,   # observation text
      "tool_call_metadata": {
        "model_response": {
          "choices": [{"message": {
            "content":    <assistant thought>,
            "tool_calls": <assistant tool_calls>,   # passed through verbatim
          }}]
        }
      }
    }

The terminal ``finish`` tool-call (which has no following tool message) is
mapped to an OH step with ``action == "finish"`` so the reference
``_detect_termination_type`` classifies it as a ``submit``.

Labels / termination
--------------------
* ``resolved`` (int 0/1) → ``G.graph["resolution_status"]`` ∈ {resolved, unresolved}.
* ``exit_status`` carries the harness outcome.  A max-iteration RuntimeError is
  surfaced as the top-level ``error`` string so the reference detector routes it
  to ``max_step`` (same regex as OpenHands); other non-``submit`` exit statuses
  become ``error_stop``.  ``model_patch`` is forwarded as
  ``test_result.git_patch`` so patch-based submit inference matches OpenHands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from phaseCodeBlockGraph import PhaseCodeBlockGraphBuilder

# Roles that carry an agent action (assistant) or an observation (tool).
_FINISH_FN = "finish"

# exit_status values that mean "the agent voluntarily submitted".
_SUBMIT_EXIT = "submit"
# Substring marking the harness iteration-cap RuntimeError (→ max_step via the
# reference _MAXITER_ERROR_RE which matches "reached maximum iteration").
_MAXITER_SUBSTR = "reached maximum iteration"


def _build_oh_history(trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Re-package a SWE-rebench chat trajectory into an OH ``history`` list.

    Walks the message list, pairing each ``assistant`` message (which holds the
    thought + tool_calls) with the immediately-following ``tool`` message (the
    observation).  A trailing ``finish`` tool-call with no tool response is
    emitted as an ``action=="finish"`` step.
    """
    history: List[Dict[str, Any]] = []
    i = 0
    n = len(trajectory)
    while i < n:
        msg = trajectory[i] or {}
        role = msg.get("role")

        if role != "assistant":
            # system / user / orphan tool messages are not actionable steps.
            i += 1
            continue

        tool_calls = msg.get("tool_calls") or []
        thought = msg.get("content") or ""

        # Detect a finish tool-call → submit signal (no observation follows).
        is_finish = any(
            (tc.get("function") or {}).get("name") == _FINISH_FN for tc in tool_calls
        )

        # The observation is the next tool-role message (if present).
        obs_msg: Optional[Dict[str, Any]] = None
        if i + 1 < n and (trajectory[i + 1] or {}).get("role") == "tool":
            obs_msg = trajectory[i + 1]

        # OH message object reconstructed for the reference extractors.
        oh_message: Dict[str, Any] = {
            "content": thought,
            "tool_calls": tool_calls,
        }
        tcm = {"model_response": {"choices": [{"message": oh_message}]}}

        if is_finish:
            # Submit step: carries the finish action; observation (if any) is the
            # finish message text.
            step: Dict[str, Any] = {
                "action": "finish",
                "observation": "finish",
                "content": (obs_msg.get("content") if obs_msg else "") or "",
                "tool_call_metadata": tcm,
            }
            history.append(step)
            i += 2 if obs_msg is not None else 1
            continue

        # Regular action step.  The obs-type marker is the tool function name
        # (execute_bash / str_replace_editor / think / ...), matching how
        # OpenHands records the observation type, and content is the tool output.
        if tool_calls:
            obs_type = (tool_calls[0].get("function") or {}).get("name") or "run"
        elif obs_msg is not None:
            obs_type = obs_msg.get("name") or "run"
        else:
            obs_type = "run"

        content = (obs_msg.get("content") if obs_msg else "") or ""
        step = {
            "observation": obs_type,
            "content": content,
            "tool_call_metadata": tcm,
        }
        history.append(step)
        i += 2 if obs_msg is not None else 1

    return history


def adapt_row_to_traj_data(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map one SWE-rebench parquet row into the OH ``traj_data`` dict shape.

    Args:
        row: A parquet row with keys ``trajectory``, ``instance_id``,
            ``model_patch``, ``exit_status``, ``resolved`` (and others).

    Returns:
        A ``traj_data`` dict consumable by
        ``PhaseCodeBlockGraphBuilder.build_from_oh_trajectory``.
    """
    trajectory = row.get("trajectory") or []
    history = _build_oh_history(trajectory)

    exit_status = (row.get("exit_status") or "").strip()
    patch = row.get("model_patch") or ""

    # Surface the harness exit status as the OH top-level ``error`` so the
    # reference termination detector classifies it identically to OpenHands:
    #   - "...reached maximum iteration..." → max_step
    #   - any other non-submit status       → error_stop
    #   - "submit"                          → no error (submit via finish/patch)
    error: Optional[str] = None
    if exit_status and exit_status != _SUBMIT_EXIT:
        error = exit_status

    traj_data: Dict[str, Any] = {
        "instance_id": row.get("instance_id", ""),
        "history": history,
        # No per-step latency data in this dataset → latencies stay None.
        "metrics": {"response_latencies": []},
        "metadata": {"max_iterations": 100},  # OpenHands v0.54.0 cap for this run
        "error": error,
        "test_result": {"git_patch": patch},
    }
    return traj_data


def build_phase_codeblock_graph_v2_from_swerebench_trajectory(
    row: Dict[str, Any],
    instance_id: str,
    output_dir: str,
    resolved: Optional[int] = None,
    eval_report_path: Optional[str] = None,
) -> str:
    """Build & persist a phase-codeblock **v2** graph from one SWE-rebench row.

    Reuses the reference :class:`PhaseCodeBlockGraphBuilder` (graph_version="v2")
    via an adapter, guaranteeing feature parity with the OpenHands v2 graphs.

    Args:
        row:           One SWE-rebench parquet row (dict).
        instance_id:   Unique id used as the graph dir/file stem (e.g.
                       ``<instance>__<trajshort>``); written as
                       ``{output_dir}/{instance_id}/{instance_id}.json``.
        output_dir:    Model-run output directory.
        resolved:      0/1 success flag.  If given, overrides the graph's
                       ``resolution_status`` (1 → "resolved", 0 → "unresolved").
                       Falls back to ``row['resolved']`` when None.
        eval_report_path: Unused (kept for signature symmetry); resolution is set
                       directly from ``resolved``.

    Returns:
        str: Path to the saved graph JSON file.
    """
    traj_data = adapt_row_to_traj_data(row)

    builder = PhaseCodeBlockGraphBuilder()
    json_path = builder.build_from_oh_trajectory(
        traj_data=traj_data,
        instance_id=instance_id,
        output_dir=output_dir,
        eval_report_path=None,  # resolution set explicitly below
        graph_version="v2",
    )

    # Set resolution_status from the dataset label and re-save.
    if resolved is None:
        resolved = row.get("resolved")
    status = "resolved" if (resolved == 1 or resolved is True) else "unresolved"

    # node_link_data nests graph-level attrs under the top-level "graph" key;
    # downstream reads json["graph"]["resolution_status"], so override there.
    with open(json_path, "r") as fh:
        g = json.load(fh)
    g.setdefault("graph", {})["resolution_status"] = status
    with open(json_path, "w") as fh:
        json.dump(g, fh, indent=2)

    return json_path
