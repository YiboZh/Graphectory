import os
import json
import sys
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict, Counter
from statistics import mean
from gsppy.gsp import GSP
from write_summary import write_summary_file

# --------------------------- Graph Analyzer ---------------------------
class TrajectoryGraphAnalyzer:
    def __init__(self, graph_data):
        self.raw_data = graph_data
        self.graph = self._load_graph()
        self._exec_graph = None
        self._hier_graph = None

    def _load_graph(self):
        return nx.node_link_graph(self.raw_data, edges="edges")

    def get_metric_dict(self):
        mf_node = self.get_most_frequent_node()
        return {
            "node_count": self.graph.number_of_nodes(),
            "exec_edge_count": len(self.get_exec_edges()),
            "hier_edge_count": len(self.get_hier_edges()),
            "step_count": self.get_step_count(),
            "longest_path": self.get_longest_simple_path(),
            "avg_out_degree": self.get_avg_out_degree(),
            "loop_count": self.get_loop_count(),
            "avg_loop_length": self.get_avg_loop_length(),
            "backedge_fraction": self.get_backedge_fraction(),
            "max_loop_length": self.get_max_loop_length(),
            "min_loop_length": self.get_min_loop_length(),
            "edge_count": self.graph.number_of_edges(),
            "most_freq_node": mf_node.replace("\n", " ") if mf_node else None,
            "most_freq_node_freq": self.get_frequency(mf_node),
        }

    def get_exec_edges(self):
        return [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("type") == "exec"]

    def get_exec_graph(self):
        if self._exec_graph is not None:
            return self._exec_graph
        G_exec = nx.DiGraph()
        G_exec.add_nodes_from(self.graph.nodes(data=True))
        for u, v, d in self.graph.edges(data=True):
            if d.get("type") == "exec":
                G_exec.add_edge(u, v)
        self._exec_graph = G_exec
        return G_exec

    def get_hier_edges(self):
        return [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("type") == "hier"]

    def get_hier_graph(self):
        if self._hier_graph is not None:
            return self._hier_graph
        G_hier = nx.DiGraph()
        for node, data in self.graph.nodes(data=True):
            if data.get("label") == "str_replace_editor: view":
                G_hier.add_node(node, **data)
        for u, v, d in self.graph.edges(data=True):
            if d.get("type") == "hier" and u in G_hier and v in G_hier:
                G_hier.add_edge(u, v)
        self._hier_graph = G_hier
        return G_hier
    
    def get_step_count(self) -> int:
        max_idx = -1
        for _, data in self.graph.nodes(data=True):
            steps = data.get("step_indices", [])
            if steps:
                try:
                    smax = max(steps)
                except TypeError:
                    numeric = [s for s in steps if isinstance(s, (int, float))]
                    if not numeric:
                        continue
                    smax = max(numeric)
                if smax > max_idx:
                    max_idx = int(smax)
        return 0 if max_idx < 0 else max_idx + 1

    def get_loop_count(self):
        return sum(1 for _ in nx.simple_cycles(self.get_exec_graph()))

    def get_loop_lengths(self):
        return [len(cycle) for cycle in nx.simple_cycles(self.get_exec_graph())]

    def get_max_loop_length(self):
        lengths = self.get_loop_lengths()
        return max(lengths) if lengths else 0

    def get_min_loop_length(self):
        lengths = self.get_loop_lengths()
        return min(lengths) if lengths else 0

    def get_avg_loop_length(self):
        lengths = self.get_loop_lengths()
        return mean(lengths) if lengths else 0

    def get_avg_out_degree(self, only_exec: bool = True):
        """
        Average out-degree over nodes. By default, computed on the exec-only subgraph
        to align with loop/longest-path metrics.
        """
        G = self.get_exec_graph() if only_exec else self.graph
        n = G.number_of_nodes()
        if n == 0:
            return 0
        return G.number_of_edges() / n

    def get_avg_degree(self):
        degrees = [deg for _, deg in self.graph.degree()]
        return mean(degrees) if degrees else 0

    def get_backedge_fraction(self, only_exec: bool = True) -> float:
        """
        Fraction of edges that point backward in time relative to the earliest step index
        at which each node appears. Self-loops are also counted as back edges.

        Back edge definition used here:
        - (u, v) is a back edge iff:
            * u == v  (self-loop), OR
            * first_step(v) < first_step(u)

        By default, uses exec-only edges to match other execution metrics.
        """
        edges = self.get_exec_edges() if only_exec else list(self.graph.edges())
        m = len(edges)
        if m == 0:
            return 0.0

        # Earliest step per node
        first_step = {}
        for n, data in self.graph.nodes(data=True):
            steps = data.get("step_indices", [])
            first_step[n] = min(steps) if steps else float("inf")

        back = 0
        for u, v in edges:
            if u == v:
                back += 1
            elif first_step.get(v, float("inf")) < first_step.get(u, float("inf")):
                back += 1

        return back / m


    def get_frequency(self, node):
        return len(self.graph.nodes[node].get("step_indices", [])) if node else 0

    def get_in_degree(self, node):
        return self.graph.in_degree(node) if node else 0

    def get_out_degree(self, node):
        return self.graph.out_degree(node) if node else 0

    def get_most_frequent_node(self):
        return max(
            self.graph.nodes,
            key=lambda n: len(self.graph.nodes[n].get("step_indices", [])),
            default=None
        )

    def get_longest_simple_path(self):
        G = self.get_exec_graph()

        if G is None or G.number_of_nodes() == 0:
            return 0

        if nx.is_directed_acyclic_graph(G):
            path = nx.dag_longest_path(G) if G.number_of_nodes() > 0 else []
            return max(len(path) - 1, 0)

        C = nx.condensation(G)
        if C.number_of_nodes() == 0:
            return 0
        path = nx.dag_longest_path(C) if C.number_of_nodes() > 0 else []
        return max(len(path) - 1, 0)

    def extract_phase_sequence(self):
        step_sequence = []
        for node in self.graph.nodes(data=True):
            for idx in node[1].get("step_indices", []):
                step_sequence.append((idx, node[1]))
        step_sequence.sort(key=lambda x: x[0])
        seq, prev = [], None
        for _, node in step_sequence:
            curr = node.get("phase")
            if curr and curr != "general" and curr != prev:
                seq.append(curr)
                prev = curr
        return seq

    def extract_label_sequence(self):
        step_sequence = []
        for node in self.graph.nodes(data=True):
            for idx in node[1].get("step_indices", []):
                step_sequence.append((idx, node[1]))
        step_sequence.sort(key=lambda x: x[0])
        return [n.get("label", "Unknown").replace('\n', ': ').strip() for _, n in step_sequence]

    def extract_step_label_pairs(self):
        """Return sorted (step_index, label) pairs using true step indices."""
        pairs = []
        for node in self.graph.nodes(data=True):
            label = node[1].get("label", "Unknown").replace('\n', ': ').strip()
            for idx in node[1].get("step_indices", []):
                pairs.append((idx, label))
        pairs.sort(key=lambda x: x[0])
        return pairs
    
    def get_localization_summary(self):
        """
        Returns localization behavior metrics:
        - loc_focus_ratio: percent of actions that are localization
        - loc_dominant_zone: early/middle/late/mixed/none
        - loc_num_clusters: number of contiguous localization clusters
        - loc_avg_node_freq: mean revisit frequency of unique localization nodes
        - repeated_view: True if any localization node is revisited
        - max_view_depth: deepest hierarchical level visited
        - avg_view_depth: average view depth
        - max_view_span: largest number of nodes viewed at same hierarchical level
        - avg_view_span: average number of nodes viewed per level
        - scroll_behavior: True if overlapping views within same file
        - num_deep_zooms_without_edit: count of leaf nodes explored without edits
        - back_and_forth_switch: True if zigzag pattern detected in view hierarchy
        """
        steps = []
        loc_nodes_freq = []
        loc_ranges_by_path = defaultdict(list)

        # -- patch nodes
        patch_paths = set()
        for _, data in self.graph.nodes(data=True):
            label = data.get("label", "")
            if label in {"str_replace_editor: str_replace", "str_replace_editor: create", "str_replace_editor: insert", "ast_editor: str_replace"}:
                path = data.get("args", {}).get("path")
                if path:
                    patch_paths.add(path)

        # -- gather step and localization info
        for node_id, data in self.graph.nodes(data=True):
            phase = data.get("phase", "")
            freq = len(data.get("step_indices", []))
            if phase == "localization":
                loc_nodes_freq.append(freq)
                view_range = data.get("args", {}).get("view_range") if isinstance(data.get("args", {}), dict) else None
                path = data.get("args", {}).get("path") if isinstance(data.get("args", {}), dict) else None
                if isinstance(view_range, (list, tuple)) and len(view_range) == 2 and path:
                    loc_ranges_by_path[path].append(tuple(view_range))
            for idx in data.get("step_indices", []):
                steps.append((idx, phase, node_id))
        steps.sort(key=lambda x: x[0])
        phases = [p for _, p, _ in steps]

        total_actions = len(phases)
        if total_actions == 0:
            return {
                "loc_focus_ratio": 0, "loc_dominant_zone": "none", "loc_num_clusters": 0,
                "loc_avg_node_freq": 0, "repeated_view": False,
                "max_view_depth": 0, "avg_view_depth": 0,
                "max_view_span": 0, "avg_view_span": 0,
                "scroll_behavior": False, "num_deep_zooms_without_edit": 0
            }

        # -- zone and cluster analysis
        bins = [0, 0, 0]
        clusters, current = [], 0
        for i, p in enumerate(phases):
            if p == "localization":
                if i < total_actions // 3: bins[0] += 1
                elif i < 2 * total_actions // 3: bins[1] += 1
                else: bins[2] += 1
                current += 1
            else:
                if current > 0: clusters.append(current)
                current = 0
        if current > 0: clusters.append(current)

        total_loc = sum(bins)
        ratio = round(total_loc / total_actions, 2)
        dominant = (
            "none" if total_loc == 0 else
            ["early", "middle", "late"][bins.index(max(bins))] if max(bins) / total_loc >= 0.5
            else "mixed"
        )

        loc_avg_freq = round(mean(loc_nodes_freq), 2) if loc_nodes_freq else 0
        repeated_view = any(freq > 1 for freq in loc_nodes_freq)

        # --- Hierarchical structure
        hier_graph = self.get_hier_graph()
        prefix_map = {}
        node_path_map = {}

        seen = set()
        def dfs(node, prefix):
            if node in seen:   
                return
            seen.add(node)     
            prefix_map[node] = prefix
            for i, child in enumerate(list(hier_graph.successors(node))):
                dfs(child, f"{prefix}-{i}" if prefix else str(i))

        roots = [n for n in hier_graph.nodes if hier_graph.in_degree(n) == 0]
        for i, root in enumerate(roots):
            dfs(root, str(i))

        for node_id, data in hier_graph.nodes(data=True):
            path = data.get("args", {}).get("path")
            if path:
                node_path_map[node_id] = path

        # --- View depth and span analysis ---
        exec_prefixes = [(nid, prefix_map.get(nid, "")) for _, phase, nid in steps if phase == "localization"]
        level_counts = Counter()
        leaf_depths = []

        for nid, prefix in exec_prefixes:
            if not prefix:
                continue
            level = prefix.count("-")
            level_counts[level] += 1

            # Check if it's a leaf in the hierarchy
            if hier_graph.out_degree(nid) == 0:
                leaf_depths.append(level + 1)  # depth = number of segments

        max_view_span = max(level_counts.values(), default=0)
        avg_view_span = round(mean(level_counts.values()), 2) if level_counts else 0
        max_view_depth = max(leaf_depths, default=0)
        avg_view_depth = round(mean(leaf_depths), 2) if leaf_depths else 0

        # --- scroll behavior
        scroll_behavior = False
        for path, ranges in loc_ranges_by_path.items():
            if len(ranges) <= 1:
                continue
            ranges.sort()
            for i in range(1, len(ranges)):
                if ranges[i][0] <= ranges[i - 1][1]:
                    scroll_behavior = True
                    break
            if scroll_behavior:
                break

        # --- deep zoom without edit
        leaf_nodes = [n for n in hier_graph.nodes if hier_graph.out_degree(n) == 0]
        leaf_paths = {
            node_path_map[n] for n in leaf_nodes
            if n in node_path_map and n in prefix_map
        }
        deep_zooms_without_edit = [
            p for p in leaf_paths if not any(p in patch for patch in patch_paths)
        ]
        num_deep_zooms_without_edit = len(deep_zooms_without_edit)

        # --- back-and-forth switch detection (formal L3 across ALL occurrences) ---
        def _cp_len_from_code(code1: str, code2: str) -> int:
            """Common prefix length of two DFS path codes like '0-2-1'."""
            s1, s2 = code1.split("-"), code2.split("-")
            i = 0
            while i < min(len(s1), len(s2)) and s1[i] == s2[i]:
                i += 1
            return i

        # Build occurrence-level localization code sequence (execution order, duplicates kept)
        loc_codes: list[str] = []
        for _, phase, nid in steps:            # 'steps' is already sorted by step index
            if phase == "localization":
                code = prefix_map.get(nid, "")
                if code:                        # only keep if we have a hierarchy code
                    loc_codes.append(code)

        back_and_forth_switch = False
        m = len(loc_codes)
        if m >= 3:
            # Precompute cp(i,j) for i<j
            cp = [[0] * m for _ in range(m)]
            for i in range(m):
                ci = loc_codes[i]
                for j in range(i + 1, m):
                    cp[i][j] = _cp_len_from_code(ci, loc_codes[j])

            # Search for any i<j<k satisfying:
            # cp(v1,v3) >= max(cp(v1,v2), cp(v2,v3)) and cp(v1,v2) != cp(v2,v3)
            for i in range(m - 2):
                for j in range(i + 1, m - 1):
                    cp12 = cp[i][j]
                    for k in range(j + 1, m):
                        cp23 = cp[j][k]
                        cp13 = cp[i][k]
                        if cp13 >= max(cp12, cp23) and cp12 != cp23:
                            back_and_forth_switch = True
                            break
                    if back_and_forth_switch:
                        break

        return {
            "loc_focus_ratio": ratio,
            "loc_dominant_zone": dominant,
            "loc_num_clusters": len(clusters),
            "loc_avg_node_freq": loc_avg_freq,
            "repeated_view": repeated_view,
            "max_view_depth": max_view_depth,
            "avg_view_depth": avg_view_depth,
            "max_view_span": max_view_span,
            "avg_view_span": avg_view_span,
            "scroll_behavior": scroll_behavior,
            "num_deep_zooms_without_edit": num_deep_zooms_without_edit,
            "back_and_forth_switch": back_and_forth_switch
        }

    def get_patch_summary(self):
        """
        Analyze patching behavior from a trajectory graph.

        Returns:
            dict: A summary of patch-related metrics, including:
                - patch_total: total number of patch attempts.
                - patch_success: count of successful patch attempts.
                - fail_types: breakdown of all failure types encountered.
                - fail_streaks: dict with max, average, and count of consecutive failed patch attempts.
                - flip_flop: True if an edit is undone by a reverse change.
                - repeat_failed_edit: True if a previously failed patch is attempted and failed again.
                - abandonment: True if there exists a file with ≥1 attempts and no success on that file).
                - fail_to_success_patterns: common reasoning phase transitions from a failed to successful patch.
        """
        patch_nodes = []
        step_node_map = {}

        # First, extract all patch nodes and build a step-to-node map
        for node_id, data in self.graph.nodes(data=True):
            phase = data.get("phase", "")
            if phase == "patch":
                for step in data.get("step_indices", []):
                    patch_nodes.append((step, node_id, data))
            for step in data.get("step_indices", []):
                step_node_map[step] = (node_id, data)

        # Sort the patch_nodes based on step indices
        patch_nodes.sort(key=lambda x: x[0])
        patch_steps = sorted(step_node_map.keys())

        patch_total = len(patch_nodes)
        patch_success = 0
        fail_types = Counter()
        fail_streaks = []
        seen_edits = set()
        flip_flop = False
        repeat_failed_edit = False
        abandonment = False
        edit_history = []
        current_streak = 0
        reasoning_between_patches = []
        fail_to_success_phases = []
        reasoning_transitions = Counter()
        fail_success_transitions = Counter()

        previous_status = None
        previous_edit = None
        previous_step = None

        # --- track success/failure per file path to detect "abandonment on some path" ---
        file_attempts = Counter()     # path -> #attempts
        file_has_success = set()      # paths that ever succeeded

        # Reasoning span detection between patches
        for i in range(len(patch_steps) - 1):
            span = list(range(patch_steps[i] + 1, patch_steps[i + 1]))
            phases = []
            for s in span:
                _, node_data = step_node_map.get(s, (None, {}))
                if node_data:
                    phase = node_data.get("phase", "unknown")
                    if phase != "patch":
                        phases.append(phase)
            if phases:
                reasoning_between_patches.append(phases)
                deduped = [p for i, p in enumerate(phases) if i == 0 or p != phases[i-1]]
                reasoning_transitions[tuple(deduped)] += 1

        for step, node_id, data in patch_nodes:
            args = data.get("args", {})
            path = args.get("path", "")
            old_str = args.get("old_str", "")
            new_str = args.get("new_str", "")
            status_raw = args.get("edit_status", "")
            edit_key = (path, old_str, new_str)

            # Count per-file attempts
            if path:
                file_attempts[path] += 1

            # Normalize status
            if isinstance(status_raw, str) and status_raw.startswith("failure:"):
                status = status_raw.replace("failure: ", "").strip()
                current_streak += 1
                fail_types[status] += 1
                if edit_key in seen_edits:
                    repeat_failed_edit = True
            else:
                status = "success"
                patch_success += 1
                if path:
                    file_has_success.add(path)
                if current_streak > 0:
                    fail_streaks.append(current_streak)
                    current_streak = 0

            # Flip-flop check (undo)
            if edit_history and edit_key == (edit_history[-1][0], edit_history[-1][2], edit_history[-1][1]):
                flip_flop = True

            # Reasoning transitions from fail -> success
            if previous_status != "success" and status == "success":
                if previous_step is not None:
                    span = list(range(previous_step + 1, step))
                    inter_phases = []
                    for s in span:
                        _, node_data = step_node_map.get(s, (None, {}))
                        if node_data:
                            phase = node_data.get("phase", "unknown")
                            if phase != "patch":
                                inter_phases.append(phase)
                    if inter_phases:
                        fail_to_success_phases.append(inter_phases)
                        deduped = [p for i, p in enumerate(inter_phases) if i == 0 or p != inter_phases[i-1]]
                        fail_success_transitions[tuple(deduped)] += 1

            seen_edits.add(edit_key)
            edit_history.append(edit_key)
            previous_edit = edit_key
            previous_status = status
            previous_step = step

        if current_streak > 0:
            fail_streaks.append(current_streak)

        # --- abandonment rule ---
        # True iff ∃ file with ≥1 patch attempts and NO success on that file.
        # This captures "all patches on some path (focused on that file) fail", even if other paths succeed.
        if any(file_attempts[p] > 0 and p not in file_has_success for p in file_attempts):
            abandonment = True

        max_fail_streak = max(fail_streaks) if fail_streaks else 0
        avg_fail_streak = round(mean(fail_streaks), 2) if fail_streaks else 0
        num_fail_streaks = len(fail_streaks)

        full_fail_types = {
            "not found": 0,
            "no change": 0,
            "multiple occurrences": 0,
            "unknown": 0,
            "invalid code": 0,
        }
        full_fail_types.update(fail_types)

        return {
            "patch_total": patch_total,
            "patch_success": patch_success,
            "fail_types": dict(full_fail_types),
            "fail_streaks": {
                "max": max_fail_streak,
                "avg": avg_fail_streak,
                "count": num_fail_streaks
            },
            "flip_flop": flip_flop,
            "repeat_failed_edit": repeat_failed_edit,
            "abandonment": abandonment,
            "fail_to_success_patterns": fail_success_transitions.most_common(3),
        }


# --------------------------- Main Analysis ---------------------------
def graphs_analyzer(instance_dir):
    out_dir = os.path.join(instance_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    difficulty_rename = {
        "<15 min fix": "under15min",
        "15 min - 1 hour": "under1h",
        "1-4 hours": "under4h",
        ">4 hours": "over4h"
    }

    categories = defaultdict(lambda: {
        "phases": [],
        "labels": [],
        "metrics": [],
        "localization": [],
        "patches": [],
        "freq_nodes": Counter(),
        "loc_dominant_zone": [],
        # "top_loc_info": [],
        "start_actions": Counter()
    })
    rows = []

    for root, _, files in os.walk(instance_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(root, fname)) as f:
                data = json.load(f)

            analyzer = TrajectoryGraphAnalyzer(data)
            step_label_pairs = analyzer.extract_step_label_pairs()

            labels_seq = [lbl for _, lbl in step_label_pairs]
            starting_label = next((lbl for lbl in labels_seq if lbl and lbl.strip().lower() != "empty action"), None)
            starting_label = starting_label.split(":")[-1].strip() if starting_label else None

            # resolution status for split counters
            resolution = data.get("graph", {}).get("resolution_status", "unknown")

            metrics = {}
            debug_difficulty_raw = data.get("graph", {}).get("debug_difficulty", "unknown")
            debug_difficulty = difficulty_rename.get(debug_difficulty_raw, debug_difficulty_raw)
            golden_patch_difficulty = data.get("graph", {}).get("golden_patch_difficulty", "unknown")
            patch_difficulty = data.get("graph", {}).get("patch_difficulty", "unknown")
            files_changed_num = data.get("graph", {}).get("files_change", 0)
            files_changed = (
                "none" if files_changed_num == 0
                else "single" if files_changed_num == 1
                else "multiple"
            )
            inst_id = data.get("graph", {}).get("instance_name")
            metrics.update({
                "instance": inst_id,
                "resolution": resolution,
                "debug_difficulty": debug_difficulty,
                "golden_patch_difficulty": golden_patch_difficulty,
                "patch_difficulty": patch_difficulty,
                "files_changed_num": files_changed_num,
            })
            metrics.update(analyzer.get_metric_dict())
            localization_stats = analyzer.get_localization_summary()
            patch_stats = analyzer.get_patch_summary()

            flat_patch_stats = patch_stats.copy()
            fs = flat_patch_stats.pop("fail_streaks", {})
            flat_patch_stats["fail_streak_max"] = fs.get("max", 0)
            flat_patch_stats["fail_streak_avg"] = fs.get("avg", 0)
            flat_patch_stats["fail_streak_count"] = fs.get("count", 0)

            fail_types = flat_patch_stats.pop("fail_types", {})
            for kf, vf in fail_types.items():
                flat_patch_stats[f"fail_type_{kf}"] = vf

            for k in ["fail_to_success_patterns"]:
                val = flat_patch_stats.get(k, [])
                flat_patch_stats[k] = str(val[0][0]) if val else "N/A"

            metrics.update(localization_stats)
            metrics.update(flat_patch_stats)
            rows.append(metrics)

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "trajectory_metrics.csv"), index=False)

def _usage_and_exit():
    print("Usage: python analyze_graph.py <agent> <model> [config]\n"
          "  agent: SWE-agent | OpenHands\n"
          "  model: e.g., deepseek/deepseek-chat or openrouter/mistralai/devstral-small\n"
          "  config (SWE-agent only, optional): default 'anthropic_filemap'",
          file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        _usage_and_exit()

    agent = sys.argv[1]
    model = sys.argv[2]

    if agent == "SWE-agent":
        config = sys.argv[3] if len(sys.argv) == 4 else "anthropic_filemap"
        model_format = model.replace("/", "--")
        graphs_dir = os.path.join(
            os.path.dirname(__file__),
            "../SWE-agent/graphs",
            f"{config}__{model_format}__t-0.00__p-1.00__c-2.00___swe_bench_verified_test",
        )

    elif agent == "OpenHands":
        model_name = model.split("/")[-1]
        graphs_dir = os.path.join(
            os.path.dirname(__file__),
            "../OpenHands/graphs/",
            "princeton-nlp__SWE-bench_Verified-test/CodeActAgent",
            f"{model_name}_maxiter_100_N_v0.40.0-no-hint-run_1/",
        )

    else:
        print("Error: agent must be 'SWE-agent' or 'OpenHands'.", file=sys.stderr)
        _usage_and_exit()

    graphs_analyzer(graphs_dir)
