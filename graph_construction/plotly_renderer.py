"""Interactive Plotly HTML renderer for NetworkX node-link graph JSON files."""

from __future__ import annotations

import html
import json
import warnings
from pathlib import Path
from typing import Any

import networkx as nx
import plotly.graph_objects as go
from networkx.readwrite import json_graph

PHASE_COLORS = {
    "localization": "#C5B3F0",
    "patch":        "#FCC9B0",
    "patching":     "#FCC9B0",
    "validation":   "#A8E6F0",
    "general":      "#CFE0F6",
}

NODE_TYPE_COLORS = {
    "start":       "#E8E8E8",
    "phase":       None,
    "code_block":  "#FFF9C4",
    "termination": "#FFCDD2",
}

EDGE_TYPE_COLORS = {
    "phase_transition":     "#666666",
    "phase_code_operation": "#4A90D9",
    "start_to_phase":       "#999999",
    "phase_to_termination": "#999999",
    "start_to_termination": "#999999",
    "exec":                 "#888888",
    "hier":                 "#2E8B57",
    "intra":                "#4A90D9",
}

_HOVER_SKIP = frozenset({"id"})


def load_graph(json_path: str | Path) -> nx.MultiDiGraph:
    with open(json_path) as f:
        data = json.load(f)
    return json_graph.node_link_graph(data, edges="edges", directed=True, multigraph=True)


def _truncate(value: str, limit: int = 400) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def attrs_to_hover(attrs: dict[str, Any], *, title: str | None = None) -> str:
    """Format node/edge attributes as Plotly hover HTML."""
    lines: list[str] = []
    if title:
        lines.append(f"<b>{html.escape(title)}</b>")
    for key in sorted(attrs):
        if key in _HOVER_SKIP:
            continue
        val = attrs[key]
        if isinstance(val, (dict, list)):
            text = _truncate(json.dumps(val, ensure_ascii=False))
        elif val is None:
            text = "null"
        else:
            text = _truncate(str(val))
        lines.append(f"<b>{html.escape(str(key))}</b>: {html.escape(text)}")
    return "<br>".join(lines)


def _node_color(nd: dict[str, Any]) -> str:
    node_type = nd.get("node_type")
    if node_type == "phase":
        return PHASE_COLORS.get(nd.get("phase_type", "general"), PHASE_COLORS["general"])
    if node_type in NODE_TYPE_COLORS and NODE_TYPE_COLORS[node_type]:
        return NODE_TYPE_COLORS[node_type]
    phases = nd.get("phases") or ["general"]
    return PHASE_COLORS.get(phases[0], PHASE_COLORS["general"])


def _node_label(nd: dict[str, Any], node_id: str) -> str:
    if nd.get("label"):
        return str(nd["label"])[:48]
    node_type = nd.get("node_type")
    if node_type == "phase":
        phase = nd.get("phase_type", "general")
        return f"{phase} [{nd.get('start_step', '?')}-{nd.get('end_step', '?')}]"
    if node_type == "code_block":
        path = nd.get("file_path", "")
        if "\n" in path or len(path) > 120:
            name = "code_block"
        else:
            name = path.rsplit("/", 1)[-1] if path else "?"
        sl, el = nd.get("start_line"), nd.get("end_line")
        if sl is not None and el is not None:
            return f"{name}:{sl}-{el}"[:48]
        return name[:48]
    if node_type == "termination":
        return nd.get("termination_type") or "termination"
    if node_type == "start":
        return "start"
    base = nd.get("command") or nd.get("subcommand") or nd.get("label") or node_id
    return str(base)[:48]


def _edge_color(ed: dict[str, Any]) -> str:
    etype = ed.get("edge_type") or ed.get("type") or "exec"
    return EDGE_TYPE_COLORS.get(etype, "#888888")


def _edge_dash(ed: dict[str, Any]) -> str:
    etype = ed.get("edge_type") or ed.get("type") or "exec"
    if etype in ("hier", "intra"):
        return "dash"
    return "solid"


def _compute_layout(G: nx.MultiDiGraph) -> dict[str, tuple[float, float]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")
        except Exception:
            pass
        try:
            return nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            pass
    return nx.spring_layout(G, seed=42, k=2.0 / max(1, G.number_of_nodes()))


def build_plotly_figure(G: nx.MultiDiGraph) -> go.Figure:
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="Empty graph")
        return fig

    pos = _compute_layout(G)
    xs = {n: float(pos[n][0]) for n in G.nodes}
    ys = {n: float(pos[n][1]) for n in G.nodes}

    # Scale layout to a readable coordinate range
    if xs:
        min_x, max_x = min(xs.values()), max(xs.values())
        min_y, max_y = min(ys.values()), max(ys.values())
        span = max(max_x - min_x, max_y - min_y, 1.0)
        xs = {n: (xs[n] - min_x) / span for n in xs}
        ys = {n: (ys[n] - min_y) / span for n in ys}

    fig = go.Figure()

    # Edge line segments grouped by style (no hover — midpoint markers handle that)
    style_segments: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
    for u, v, _key, ed in G.edges(keys=True, data=True):
        style = (_edge_color(ed), _edge_dash(ed))
        sx, sy = style_segments.setdefault(style, ([], []))
        sx.extend([xs[u], xs[v], None])
        sy.extend([ys[u], ys[v], None])

    for (color, dash), (seg_x, seg_y) in style_segments.items():
        fig.add_trace(
            go.Scatter(
                x=seg_x,
                y=seg_y,
                mode="lines",
                line=dict(color=color, width=1.5, dash=dash),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Edge hover targets at midpoints
    edge_x, edge_y, edge_hover = [], [], []
    for u, v, key, ed in G.edges(keys=True, data=True):
        edge_x.append((xs[u] + xs[v]) / 2)
        edge_y.append((ys[u] + ys[v]) / 2)
        title = f"edge {u} → {v}" + (f" (key={key})" if G.number_of_edges(u, v) > 1 else "")
        hover = attrs_to_hover(ed, title=title)
        edge_hover.append(hover)

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="markers",
            marker=dict(size=14, color="rgba(0,0,0,0.01)", line=dict(width=0)),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=edge_hover,
            name="edges",
            showlegend=False,
        )
    )

    # Nodes
    node_x, node_y, node_text, node_colors, node_hover = [], [], [], [], []
    for n, nd in G.nodes(data=True):
        node_x.append(xs[n])
        node_y.append(ys[n])
        node_text.append(_node_label(nd, n))
        node_colors.append(_node_color(nd))
        node_hover.append(attrs_to_hover(dict(nd), title=f"node {n}"))

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                size=28,
                color=node_colors,
                line=dict(color="#333333", width=1),
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=node_hover,
            name="nodes",
            showlegend=False,
        )
    )

    meta = G.graph or {}
    instance = meta.get("instance_name", "graph")
    graph_type = meta.get("graph_type", "action")
    resolution = meta.get("resolution_status", "unknown")
    title = (
        f"{instance}  ·  {graph_type}  ·  {resolution}"
        f"  ·  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
        dragmode="pan",
    )
    return fig


def render_graph_html(json_path: str | Path, html_path: str | Path) -> None:
    """Load a graph JSON file and write a self-contained interactive Plotly HTML page."""
    G = load_graph(json_path)
    fig = build_plotly_figure(G)
    out = Path(html_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(out),
        include_plotlyjs="cdn",
        full_html=True,
        config={"scrollZoom": True, "displayModeBar": True},
    )
