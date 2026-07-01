#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase classifier for agent actions (robust to dict/sequence `command`, heredocs, and shell None tool).

Phases:
  - "localization" : gathering info, searching, reading, or generating a *new* test/repro
                     script *before* any patch
  - "patch"        : editing existing code (any file, incl. test files that are the repair
                     target) or creating a new non-test source file
  - "validation"   : (re-)running tests or test-like commands *after* a patch; viewing test
                     assets *after* a patch; creating a new test asset *after* a patch
  - "general"      : everything else

#39 fix (2026-06-30) — two parts (see CHANGES.md in the #39 experiment folder):
  (1) PATH-COMPONENT test detection. `_path_is_test` anchors the TEST_HINTS to real path
      structure (a whole dir component `tests`/`test`/`testing`, or a `test_*` / `*_test`
      basename, or a REPRO hint) instead of a raw substring. The old substring match treated
      the SWE-bench workspace dir `pytest-dev__pytest__X.Y` as test-related (it contains
      "test_"), so EVERY path in a pytest trajectory — including `src/_pytest/*.py` source —
      looked test-related and no edit ever flipped `has_patch`. This is the dominant cause.
  (2) EDIT-vs-CREATE rule (option (a), refined):
      • EDITING AN EXISTING FILE (str_replace / insert / undo_edit / sed -i / perl -i):
          - source file → "patch";
          - genuine test file → "patch" if it is the repair target (no prior patch, the
            pytest-dev case), else "validation" (post-patch verification, django/sympy);
          - reproduction/debug script → key rule (localization before a patch, else validation).
      • CREATING A NEW FILE (create / touch / `>` redirection / heredoc / tee) keeps the
        test-path key rule: a new test/repro file → "localization"/"validation" (test
        generation / understanding); a new non-test file → "patch".

Key rule (test generation & execution) — applies to READ / EXECUTE / CREATE-NEW-TEST only:
  • If test generation/execution happens with NO prior "patch" in the phase history → "localization".
  • If it happens AFTER a "patch" → "validation".

Other bash commands:
  • grep/find/cat/nl WITHOUT redirection (>, >>) → "localization" or ("validation" if test-related after patch).
  • Piped read-only operations (e.g., nl file.py | sed -n '10,20p') → "localization" (or "validation" if test-related after patch).
  • In-place edits (sed -i / perl -i) → "patch" (edit of existing code), repro-script exception aside.
  • Redirection/heredoc/tee that CREATE files → test-path key rule if target is test/repro, else "patch".

Function:
  get_phase(tool, subcommand, command, args, prev_phases=None, flags)
"""

from __future__ import annotations
import ast
import re
from typing import Iterable, List, Tuple, Any, Optional, Dict

# --------------------------- Configurable Heuristics ---------------------------

# Tokens/paths hinting that something is test-related.
TEST_HINTS: Tuple[str, ...] = (
    "test_", "reproduc", "debug", "_test", "/tests/", "/test/",
)

# Paths hinting at the agent's own reproduction/debug script (a subset of TEST_HINTS).
# Editing/creating one of these is *understanding*, not patching the codebase, so it is
# excluded from the "edit-op ⇒ patch" rule and from has_patch. Kept deliberately narrow
# ("test_" is NOT here) so that test_* files that are the *repair target* still count as
# patches in testing-framework repos (the #39 pytest-dev fix).
REPRO_HINTS: Tuple[str, ...] = ("reproduc", "debug")

# Commands that typically *read/search* only; with redirection they can become edits.
READONLY_CMDS: Tuple[str, ...] = ("grep", "find", "cat", "ls", "head", "tail", "awk", "nl")

# Commands that are clearly *editing* or *creating* content.
# Note: sed and perl handled explicitly below based on their flags
EDIT_CMDS: Tuple[str, ...] = ("touch",)

# str_replace_editor subcommands that indicate edits vs reads.
SRE_EDIT_SUBCMDS: Tuple[str, ...] = ("create", "str_replace", "insert", "undo_edit")
SRE_READONLY_SUBCMDS: Tuple[str, ...] = ("view",)

# Python commands that usually execute code/tests.
PY_CMDS: Tuple[str, ...] = ("python", "python3", "python2", "pytest", "pylint")

# --------------------------- Utilities ---------------------------

def _flatten_args(args: Any) -> List[str]:
    """Normalize args into a flat list of lowercase string tokens."""
    tokens: List[str] = []
    if isinstance(args, dict):
        for v in args.values():
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                tokens.extend(str(x) for x in v)
            else:
                tokens.append(str(v))
    elif isinstance(args, (list, tuple)):
        tokens = [str(x) for x in args]
    elif isinstance(args, str):
        tokens = [args]
    return [t.lower() for t in tokens]

_PATHISH = re.compile(r"(^[/~.]|/|\.py$)")

def _extract_paths(args: Any) -> List[str]:
    """Extract path-like strings from args."""
    tokens = _flatten_args(args)
    return [t for t in tokens if _PATHISH.search(t)]

def _has_prior_patch(prev_phases: Optional[Iterable[str]]) -> bool:
    return any(p == "patch" for p in (prev_phases or []))

def _contains_redirection(tokens: List[str]) -> bool:
    """
    Detect shell redirection/heredoc/tee implying writes/edits.
    Handles both separated tokens (">", ">>", "<<") and embedded heredocs like "cat <<'EOF' > file".
    """
    if not tokens:
        return False
    # Exact tokens / prefixed tokens
    redir_ops = {">", ">>", "1>", "2>", ">|", "<<<", "<<", "<>", ">&", "2>&1"}
    if any(t in redir_ops or t.startswith((">", ">>", "1>", "2>")) for t in tokens):
        return True
    # Embedded operators (e.g., "cat << 'EOF' > file", or script blobs)
    embedded_ops = (" <<", "<<", " >>", ">>", " 1>", " 2>", " >"," >|","<>", ">&", "2>&1")
    if any(any(op in t for op in embedded_ops) for t in tokens):
        return True
    # 'tee' writes to files via pipe
    return any("tee" == t or " tee " in t for t in tokens)

def _is_piped_readonly_operation(cmd: str, tokens: List[str]) -> bool:
    """
    Detect if this is a piped read-only operation (e.g., nl file.py | sed -n '10,20p').
    Returns True if:
      - The command is a read-only command (nl, cat, grep, etc.)
      - There's a pipe (|) in the tokens
      - There's no output redirection (>, >>, tee)
    This indicates the command is for viewing/filtering only, not editing.
    """
    if cmd not in READONLY_CMDS:
        return False
    has_pipe = "|" in tokens or any("|" in t for t in tokens)
    has_output_redir = _contains_redirection(tokens)
    return has_pipe and not has_output_redir

def _path_is_test(path: str) -> bool:
    """Path-component-aware test-file detection (#39 fix).

    The old check ``any(hint in path for hint in TEST_HINTS)`` was a raw *substring* match,
    so the SWE-bench workspace/instance directory ``pytest-dev__pytest__X.Y`` matched
    ``"test_"`` (inside "py**test__**X") and made **every** path in a pytest-dev trajectory —
    including ``src/_pytest/*.py`` source files — look test-related. That poisoned the whole
    trajectory's phase labels and stopped ``has_patch`` from ever flipping.

    This version anchors the hints to real path structure: a path is test-related iff a
    directory *component* is a conventional test dir (``tests`` / ``test`` / ``testing``), OR
    the *basename* is a test/repro file (``test_*``, ``*_test``, or contains a REPRO hint).
    ``pytest-dev__pytest__4.6`` is not a whole component named test/tests and is not a
    basename, so source files are no longer mislabelled.
    """
    p = (path or "").replace("\\", "/")
    comps = [c for c in p.split("/") if c]
    if not comps:
        return False
    base = comps[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else base
    if any(h in base for h in REPRO_HINTS):
        return True
    if any(c in ("tests", "test", "testing") for c in comps[:-1]):
        return True
    if base.startswith("test_") or stem.endswith("_test"):
        return True
    return False

def _is_test_related(tokens: List[str], paths: List[str]) -> bool:
    """Test-related if any extracted path is a test/repro file (path-component aware)."""
    return any(_path_is_test(s) for s in paths)

def _is_repro_related(paths: List[str]) -> bool:
    """True if any path looks like a reproduction/debug script (REPRO_HINTS)."""
    return any(any(h in s for h in REPRO_HINTS) for s in paths)

def _edit_existing_phase(paths: List[str], has_patch: bool) -> str:
    """Phase for an in-place edit of an EXISTING file (#39 option (a), refined).

    - The agent's own reproduction/debug script → understanding (key rule: localization
      before a patch, validation after).
    - A genuine **test file**: the repair target when edited *before* any patch (this is the
      pytest-dev case → ``patch``, flips ``has_patch``); verification of an existing fix when
      edited *after* a patch (django/sympy test tweaks → ``validation``). This split is what
      fixes pytest without relabelling legitimate post-patch verification edits.
    - Any other existing file (source) → ``patch``.
    """
    if _is_repro_related(paths):
        return "validation" if has_patch else "localization"
    if any(_path_is_test(p) for p in paths):
        return "validation" if has_patch else "patch"
    return "patch"

def _sre_phase(subcommand: Optional[str]) -> str:
    sub = (subcommand or "").lower()
    if sub in SRE_EDIT_SUBCMDS:
        return "patch"
    if sub in SRE_READONLY_SUBCMDS:
        return "localization"
    return "general"

def _normalize_command_and_merge_args(command: Any, args: Any) -> Tuple[str, List[str], List[str]]:
    """
    Normalize `command` into a lowercase command string (may be empty if not a simple str)
    and merge any command-embedded arguments into the args token/path sets.

    Returns: (cmd_str, merged_tokens, merged_paths)
    """
    # Determine command string if possible
    if isinstance(command, str) or command is None:
        cmd_str = (command or "").lower().strip()
        cmd_tokens = []
    else:
        # If command is dict/list/tuple, treat its contents as additional tokens/paths.
        cmd_str = ""
        cmd_tokens = _flatten_args(command)

    arg_tokens = _flatten_args(args)
    merged_tokens = arg_tokens + cmd_tokens
    merged_paths  = _extract_paths(args) + _extract_paths(command)
    return cmd_str, merged_tokens, merged_paths

def _extract_edited_files_from_python_code(code: str) -> List[str]:
    """
    Analyze Python code via AST to extract file paths being edited/created.
    Looks for patterns like:
    - Path('file.py').write_text(...)
    - open('file.py', 'w').write(...)
    - with open('file.py', 'w') as f: ...
    Returns list of file paths found.
    """
    if not code or not isinstance(code, str):
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If code doesn't parse, fall back to empty
        return []

    # First pass: collect all variable assignments
    path_vars: Dict[str, str] = {}
    string_vars: Dict[str, str] = {}

    class VariableCollector(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign):
            # Track assignments like: var = Path('file.py') or var = 'file.py'
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name) and node.value.func.id == 'Path':
                    if node.value.args and isinstance(node.value.args[0], ast.Constant):
                        filepath = node.value.args[0].value
                        if isinstance(filepath, str):
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    path_vars[target.id] = filepath
            elif isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                # Track simple string assignments: var = 'file.py'
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        string_vars[target.id] = node.value.value
            self.generic_visit(node)

    # Collect variables first
    var_collector = VariableCollector()
    var_collector.visit(tree)

    # Second pass: detect file edits using collected variables
    edited_files: List[str] = []
    with_files: set = set()  # Track files in 'with' to avoid duplicates

    class FileEditVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            # Pattern 1: Path('file.py').write_text(...) or Path('file.py').write_bytes(...)
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ('write_text', 'write_bytes'):
                    # Check if calling on Path(...) directly
                    if isinstance(node.func.value, ast.Call):
                        if isinstance(node.func.value.func, ast.Name) and node.func.value.func.id == 'Path':
                            if node.func.value.args and isinstance(node.func.value.args[0], ast.Constant):
                                filepath = node.func.value.args[0].value
                                if isinstance(filepath, str):
                                    edited_files.append(filepath)
                    # Check if calling on a variable that was assigned Path(...)
                    elif isinstance(node.func.value, ast.Name):
                        var_name = node.func.value.id
                        if var_name in path_vars:
                            edited_files.append(path_vars[var_name])

            # Pattern 2: open('file.py', 'w') or open(variable, 'w') - check for write modes
            if isinstance(node.func, ast.Name) and node.func.id == 'open':
                if len(node.args) >= 2:
                    filename = None

                    # First arg can be a constant string or a variable
                    if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        filename = node.args[0].value
                    elif isinstance(node.args[0], ast.Name):
                        # Variable reference - check if it was assigned a string
                        var_name = node.args[0].id
                        if var_name in string_vars:
                            filename = string_vars[var_name]

                    if filename:
                        # Skip if already handled by visit_With
                        if filename in with_files:
                            self.generic_visit(node)
                            return
                        # Second arg is mode
                        if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                            mode = node.args[1].value
                            # Check for write/append/exclusive modes
                            if any(m in mode for m in ['w', 'a', 'x']):
                                edited_files.append(filename)

            self.generic_visit(node)

        def visit_With(self, node: ast.With):
            # Pattern 3: with open('file.py', 'w') as f: ... or with open(variable, 'w') as f: ...
            for item in node.items:
                if isinstance(item.context_expr, ast.Call):
                    call = item.context_expr
                    if isinstance(call.func, ast.Name) and call.func.id == 'open':
                        if len(call.args) >= 2:
                            filename = None

                            # First arg can be constant or variable
                            if isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
                                filename = call.args[0].value
                            elif isinstance(call.args[0], ast.Name):
                                var_name = call.args[0].id
                                if var_name in string_vars:
                                    filename = string_vars[var_name]

                            if filename and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str):
                                mode = call.args[1].value
                                if any(m in mode for m in ['w', 'a', 'x']):
                                    edited_files.append(filename)
                                    with_files.add(filename)  # Mark as handled
            self.generic_visit(node)

    visitor = FileEditVisitor()
    visitor.visit(tree)

    return edited_files

# --------------------------- Core classification ---------------------------

def get_phase(
    tool: Optional[str],
    subcommand: Optional[str],
    command: Optional[str | dict | list | tuple],
    args: Any,
    prev_phases: Optional[Iterable[str]] = None,
    flags: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Map a (tool, subcommand, command, args, prev_phases, flags) to a phase:
        "localization" | "patch" | "validation" | "general"

    flags:
        Optional dict for additional context, e.g. {"c": "assert ..."} for python -c inline code,
        or {"__heredoc__": True} for heredoc/stdin input
    """
    flags = flags or {}
    cmd, tokens, paths = _normalize_command_and_merge_args(command, args)
    has_patch = _has_prior_patch(prev_phases)

    # 1) str_replace_editor decisions (tool-specific)
    if (tool or "").lower() == "str_replace_editor":
        sub = (subcommand or "").lower()
        if sub in SRE_EDIT_SUBCMDS:
            if sub == "create":
                # Creating a NEW file: test/repro generation → key rule (loc before first
                # patch, validation after); a new non-test source file → patch.
                if _is_test_related(tokens, paths):
                    return "validation" if has_patch else "localization"
                return "patch"
            # str_replace / insert / undo_edit EDIT AN EXISTING FILE → patch even on a
            # test-named path (the #39 fix for testing-framework repair targets).
            return _edit_existing_phase(paths, has_patch)

        # 'view' (read-only) remains localization unless it's test-related AFTER a patch → validation
        if sub in SRE_READONLY_SUBCMDS:
            if _is_test_related(tokens, paths) and has_patch:
                return "validation"
            return "localization"

        return "general"

    # 2) Python / pytest / pylint
    #    - Execution: apply key rule regardless of file hints.
    #    - If command line includes redirection (creating/editing files), treat as edit-like and use heuristics.
    #    - If inline code (heredoc, -c flag) is editing files, classify based on target files.
    if cmd in PY_CMDS:
        # Check for inline code execution (heredoc, -c flag)
        is_heredoc = flags.get("__heredoc__", False)

        # For heredocs, check inline code first before treating as redirection
        # (heredocs contain << which looks like redirection but needs code analysis first)
        if _contains_redirection(tokens) and not is_heredoc:
            # Edit-like via redirection (e.g., python -c '...' > tests/test_x.py)
            return ("validation" if has_patch else "localization") if _is_test_related(tokens, paths) else "patch"
        code_content = None

        # Source 1: heredoc (stdin)
        if is_heredoc and args:
            args_list = args if isinstance(args, (list, tuple)) else [args]
            for item in args_list:
                if isinstance(item, str):
                    # Check if this looks like Python code
                    is_code = (
                        len(item) > 20 or
                        '\n' in item or
                        'Path(' in item or
                        'open(' in item or
                        'write' in item
                    )
                    if is_code and item not in ['-', '>']:
                        code_content = item
                        break

        # Source 2: -c flag (python -c 'code')
        if not code_content and flags:
            c_code = flags.get('c')
            if c_code and isinstance(c_code, str) and len(c_code) > 5:
                code_content = c_code

        # For inline code (heredoc, -c), check if editing files
        edited_files_from_code: List[str] = []
        if code_content:
            edited_files_from_code = _extract_edited_files_from_python_code(code_content)

        # If inline code is editing files, classify based on what files are being edited
        if edited_files_from_code:
            test_files_edited = [f for f in edited_files_from_code if _is_test_related([], [f.lower()])]
            if test_files_edited:
                # Editing/creating test files
                return "validation" if has_patch else "localization"
            else:
                # Editing non-test files → patching
                return "patch"

        # Default: test/code execution → key rule
        return "validation" if has_patch else "localization"

    # 3) Read-only commands (grep/find/cat/ls/head/tail/awk/echo/nl/sed -n/perl -n/-p without -i)
    is_sed_readonly = (cmd == "sed" and "i" not in flags and "n" in flags)
    # perl -n/-p without -i and with file args = readonly viewing
    is_perl_readonly = (cmd == "perl" and "i" not in flags and
                        ("n" in flags or "p" in flags) and paths)
    if cmd in READONLY_CMDS or is_sed_readonly or is_perl_readonly:
        # Piped operations without output redirection (e.g., nl file.py | sed -n '10,20p') are read-only
        if _is_piped_readonly_operation(cmd, tokens):
            # Viewing content: test-related AFTER patch → validation; otherwise → localization
            if _is_test_related(tokens, paths) and has_patch:
                return "validation"
            return "localization"

        if _contains_redirection(tokens):
            # These become edits when redirecting to files or using tee/heredoc
            return ("validation" if has_patch else "localization") if _is_test_related(tokens, paths) else "patch"

        # read-only, test-related AFTER a prior patch counts as validation; otherwise localization
        if _is_test_related(tokens, paths) and has_patch:
            return "validation"
        return "localization"

    # 3.5) perl test execution (perl script.pl where script is test-related)
    if cmd == "perl" and "i" not in flags:
        # Not in-place editing, not readonly viewing (already handled)
        # Check if executing test-related scripts
        if _is_test_related(tokens, paths):
            return "validation" if has_patch else "localization"

    # 4) Edit/creation commands (sed/touch/perl -i)
    is_perl_edit = (cmd == "perl" and "i" in flags)
    if (cmd == "sed" and "i" in flags) or is_perl_edit:
        # sed -i / perl -i EDIT AN EXISTING FILE in place → patch even on a test-named path
        # (#39 fix), except the agent's own reproduction/debug script before any patch.
        return _edit_existing_phase(paths, has_patch)
    if cmd in EDIT_CMDS:
        # touch CREATES a (new, empty) file → keep the test-path key rule for new test/repro
        # placeholders; a new non-test file → patch.
        return ("validation" if has_patch else "localization") if _is_test_related(tokens, paths) else "patch"

    # 5) Fallbacks:
    #    If any redirection is present (even embedded), treat as edit-like.
    if _contains_redirection(tokens):
        return ("validation" if has_patch else "localization") if _is_test_related(tokens, paths) else "patch"

    #    Otherwise, unknown → general.
    return "general"


# --------------------------- Self-checks ---------------------------
if __name__ == "__main__":
    # Simple tests
    test_cases = [
        # (tool, subcommand, command, args, prev_phases, expected_phase)
        (None, None, "grep", ["def foo():", "file.py"], None, None, "localization"),
        (None, None, "grep", ["def foo():", "test_file.py"], ["patch"], None, "validation"),
        (None, None, "grep", ["def foo():", "file.py", ">", "out.txt"], None, None, "patch"),
        (None, None, "grep", ["def test_foo():", "file.py", ">", "tests/test_file.py"], None, None, "localization"),
        (None, None, "grep", ["def test_foo():", "file.py", ">", "tests/test_file.py"], ["patch"], None, "validation"),
        (None, None, "sed", ["s/foo/bar/g", "file.py"], None, {'i': True}, "patch"),
        (None, None, "python", ["script.py"], None, None, "localization"),
        (None, None, "python", ["script.py"], ["patch"], None, "validation"),
        (None, None, "python", ["-c", "'print(42)'", ">", "out.txt"], None, None, "patch"),
        (None, None, "python", ["-c", "'print(42)'", ">", "tests/test_out.py"], None, None, "localization"),
        (None, None, "python", ["-c", "'print(42)'", ">", "tests/test_out.py"], ["patch"], None, "validation"),
        # str_replace_editor where `command` may be a dict (observed in traces)
        ("str_replace_editor", "create", {"path": "file.py"}, None, None, None, "patch"),
        ("str_replace_editor", "create", {"path": "tests/test_file.py"}, None, None, None, "localization"),
        ("str_replace_editor", "create", {"path": "tests/test_file.py"}, None, ["patch"], None, "validation"),
        ("str_replace_editor", "view", {"path": "test_file.py"}, None, ["patch"], None, "validation"),
        # ── #39 fix: editing an EXISTING test-named file = the repair target = patch ──
        # pytest-dev family: the file under repair *is* a test_* / testing/ file.
        ("str_replace_editor", "str_replace", {"path": "testing/test_assertion.py"}, None, None, None, "patch"),
        ("str_replace_editor", "str_replace", {"path": "testing/python/test_fixtures.py"}, None, None, None, "patch"),
        ("str_replace_editor", "insert", {"path": "/testbed/testing/test_cacheprovider.py", "insert_line": 10}, None, None, None, "patch"),
        ("str_replace_editor", "str_replace", {"path": "src/_pytest/python.py"}, None, None, None, "patch"),
        # str_replace on a test file AFTER a patch = verification of the fix → validation
        # (the repair-target case only applies before the first patch).
        ("str_replace_editor", "str_replace", {"path": "testing/test_assertion.py"}, None, ["patch"], None, "validation"),
        # source file (src/_pytest/*) is NOT test-related under path-component detection,
        # even though the workspace dir "pytest__X" contains the substring "test_".
        ("str_replace_editor", "str_replace", {"path": "/workspace/pytest-dev__pytest__4.6/src/_pytest/pastebin.py"}, None, None, None, "patch"),
        # Repro/debug script exception: editing the agent's own repro script before any
        # patch stays localization (understanding), not a patch.
        ("str_replace_editor", "str_replace", {"path": "/workspace/reproduce_bug.py"}, None, None, None, "localization"),
        ("str_replace_editor", "str_replace", {"path": "/workspace/debug_script.py"}, None, None, None, "localization"),
        # sed -i / perl -i editing an existing test file → patch (in-place edit of code).
        (None, None, "sed", ["s/foo/bar/g", "testing/test_assertion.py"], None, {'i': True}, "patch"),
        ("str_replace_editor", "str_replace", {"path": "/testbed/django/db/backends/postgresql/client.py", "new_str": "        temp_pgpass = None\n        sigint_handler = signal.getsignal(signal.SIGINT)\n        try:\n            print(f\"DEBUG: passwd = '{passwd}'\")  # DEBUG\n            if passwd:\n                print(\"DEBUG: Creating temporary .pgpass file\")  # DEBUG\n                # Create temporary .pgpass file.\n                temp_pgpass = NamedTemporaryFile(mode='w+')}"}, None, None, None, "patch"),
        # Heredoc embedded in a single token (should be detected as redirection → edit-like).
        # Target is test-related and no prior patch → localization (test generation).
        (None, None, "complex_command",
         ["cat << 'EOF' > /workspace/test_hstack_fix.py\nprint('hi')\nEOF"], None, None, "localization"),

        # nl piped commands (read-only viewing operations)
        # nl file.py | sed -n '10,20p' - viewing regular file before patch
        (None, None, "nl", ['filename.py', '|', 'sed', '10,20p'], None, {'b': True, 'a': True, 'n': True}, "localization"),
        # nl test_file.py | sed -n '10,20p' - viewing test file before patch
        (None, None, "nl", ["test_file.py", "|", "sed", '10,20p'], None, {'b': True, 'a': True, 'n': True}, "localization"),
        # nl test_file.py | sed -n '10,20p' - viewing test file AFTER patch
        (None, None, "nl", ["test_file.py", "|", "sed", '10,20p'], ["patch"], {'b': True, 'a': True, 'n': True}, "validation"),
        # nl file.py | sed -n '10,20p' - viewing regular file AFTER patch
        (None, None, "nl", ["filename.py", "|", "sed", '10,20p'], ["patch"], {'b': True, 'a': True, 'n': True}, "localization"),

        # nl with output redirection (becomes an edit operation)
        # nl file.py > output.txt - creating/editing non-test file
        (None, None, "nl", ["file.py", ">", "output.txt"], None, None, "patch"),
        # nl file.py > test_output.py - creating/editing test file before patch
        (None, None, "nl", ["file.py", ">", "test_output.py"], None, None, "localization"),
        # nl file.py > test_output.py - creating/editing test file AFTER patch
        (None, None, "nl", ["file.py", ">", "test_output.py"], ["patch"], None, "validation"),

        (None, None, "sed", ["/testbed/seaborn/_core/scales.py"], None, {"n": "1,440p"}, "localization"),
        # #39: perl -i on a test file is an in-place EDIT of existing code → patch
        # (was "localization" pre-fix; updated to reflect the edit-op ⇒ patch rule).
        (None, None, "perl", ['s/old/new/g', 'test_file.py'], None, {'i': True, 'p': True, 'e': True}, "patch"),

        (None, None, "python", ["-", "from pathlib import Path\np=Path('/testbed/seaborn/_core/scales.py')\ns=p.read_text()\nold=''' if prop.legend:\n axis.set_view_interval(vmin, vmax)\n locs = axis.major.locator()\n locs = locs[(vmin <= locs) & (locs <= vmax)]\n labels = axis.major.formatter.format_ticks(locs)\n new._legend = list(locs), list(labels)\n\n return new\n'''\nif old in s:\n new=''' if prop.legend:\n axis.set_view_interval(vmin, vmax)\n locs = axis.major.locator()\n locs = locs[(vmin <= locs) & (locs <= vmax)]\n formatter = axis.major.formatter\n labels = formatter.format_ticks(locs)\n # Attempt to capture any multiplicative offset used by the formatter\n offset = None\n try:\n # Many formatters (e.g. ScalarFormatter) provide a get_offset() method\n off = formatter.get_offset()\n except Exception:\n off = None\n if off:\n offset = str(off)\n new._legend = list(locs), list(labels) if offset is None else (list(locs), list(labels), offset)\n\n return new\n'''\n s=s.replace(old,new)\n p.write_text(s)\n print('patched')\nelse:\n print('pattern not found')"], None, {"__heredoc__": True}, "patch"),
    ]

    for i, (tool, subcmd, cmd, args, prev, flag, expected) in enumerate(test_cases, 1):
        result = get_phase(tool, subcmd, cmd, args, prev, flag)
        assert result == expected, f"Test case {i} failed: got {result}, expected {expected}"
    print("All test cases passed.")