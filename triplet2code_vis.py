#!/usr/bin/env python3
"""
Load and visualize triplet2code mappings: triplet string -> { "code": "..." }.

Triplet keys are parsed as either:
  - "subject. relation, object"  (period after subject, comma before object)
  - "subject, relation, object"   (split on first and last comma; same as mem_graph_data_with_code.json)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Any

import networkx as nx
from pyvis.network import Network

# Component colors (no matplotlib dependency)
_TABLEAU_LIKE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)
_EXTRA_COLORS = (
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
    "#e6550d",
    "#31a354",
    "#756bb1",
    "#636363",
)


def parse_triplet_key(key: str) -> tuple[str, str, str]:
    """
    Parse a triplet string into (subject, relation, object).

    Supports:
    - "<subject>. <relation>, <object>"
    - "<subject>, <relation>, <object>" (first comma / last comma; middle is relation)
    """
    key = key.strip()
    if not key:
        raise ValueError("Empty triplet key")

    m = re.match(r"^(.+?)\.\s+(.+?),\s*(.+)$", key, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

    first = key.find(",")
    last = key.rfind(",")
    if first != -1 and last != -1 and first < last:
        subj = key[:first].strip()
        rel = key[first + 1 : last].strip()
        obj = key[last + 1 :].strip()
        if subj and rel and obj:
            return subj, rel, obj

    raise ValueError(f"Cannot parse triplet key: {key!r}")


def extract_code(value: Any) -> str:
    """Normalize triplet2code values to a code string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        c = value.get("code")
        return c if isinstance(c, str) else ""
    return ""


def load_triplet2code_from_json(path: str | Path) -> dict[str, Any]:
    """Load triplet2code from a mem_graph_data_with_code.json-style file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    t2c = data.get("triplet2code")
    if not isinstance(t2c, dict):
        raise ValueError(f"No 'triplet2code' object in {path}")
    return t2c


def triplet2code_to_digraph(triplet2code: dict[str, Any]) -> nx.DiGraph:
    """
    Build a directed graph: subject -relation-> object for each key,
    plus optional code-snippet nodes linked to subject and object.
    """
    G = nx.DiGraph()
    for i, (triplet_key, meta) in enumerate(triplet2code.items()):
        subj, rel, obj = parse_triplet_key(triplet_key)
        code = extract_code(meta)
        G.add_edge(subj, obj, label=rel, triplet_key=triplet_key)

        if code.strip():
            cid = f"__code_snippet_{i}__"
            G.add_node(cid, node_type="code", code=code, triplet_key=triplet_key)
            G.add_edge(subj, cid, label="", edge_kind="to_code")
            G.add_edge(cid, obj, label="", edge_kind="from_code")
    return G


def _format_code_label(code: str, max_lines: int = 10, line_width: int = 40) -> str:
    """Short multi-line label for on-canvas display."""
    lines = code.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n")
    out: list[str] = []
    for line in lines[:max_lines]:
        if len(line) > line_width:
            out.extend(textwrap.wrap(line, width=line_width, break_long_words=True))
        else:
            out.append(line)
    truncated = len(lines) > max_lines or len(out) >= max_lines * line_width
    if truncated and out:
        out.append("…")
    elif truncated:
        out = ["…"]
    return "\n".join(out) if out else "(empty snippet)"


def _code_tooltip(code: str, max_chars: int = 12000) -> str:
    """Plain-text hover tooltip (pyvis escapes HTML in titles)."""
    if len(code) > max_chars:
        return code[:max_chars] + "\n\n… [truncated]"
    return code


def vis_triplet2code_graph(
    triplet2code: dict[str, Any],
    exp_path: str | Path,
    save_as: str = "triplet2code_graph",
    *,
    code_node_color: str = "#fff3cd",
) -> str:
    """
    Visualize triplets (and linked code snippets) like vis_net, with code visible on dedicated nodes.

    Returns path to the written HTML file.
    """
    exp_path = Path(exp_path)
    exp_path.mkdir(parents=True, exist_ok=True)

    subgraph = triplet2code_to_digraph(triplet2code)
    components = list(nx.weakly_connected_components(subgraph))
    num_comps = len(components)

    net = Network(
        height="1000px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
    )

    colors = list(_TABLEAU_LIKE) + list(_EXTRA_COLORS)
    while len(colors) < len(components):
        colors = colors * 2

    component_map: dict[str, dict[str, Any]] = {}
    for i, component in enumerate(components):
        color = colors[i % len(colors)]
        for node in component:
            component_map[node] = {"component_id": i, "color": color}

    all_positions: dict[str, tuple[float, float]] = {}
    component_spacing = 2200
    grid_cols = max(1, math.ceil(math.sqrt(num_comps)))

    for idx, component in enumerate(components):
        comp_subgraph = subgraph.subgraph(component)
        n = len(component)
        if n < 30:
            pos = nx.spring_layout(comp_subgraph, k=2.5, iterations=80, scale=550)
        elif n < 100:
            pos = nx.spring_layout(comp_subgraph, k=1.8, iterations=80, scale=850)
        else:
            try:
                pos = nx.kamada_kawai_layout(comp_subgraph, scale=1000)
            except Exception:
                pos = nx.spring_layout(comp_subgraph, k=1, iterations=50, scale=1000)

        row = idx // grid_cols
        col = idx % grid_cols
        offset_x = col * component_spacing
        offset_y = row * component_spacing
        for node, (x, y) in pos.items():
            all_positions[node] = (x + offset_x, y + offset_y)

    for node in subgraph.nodes():
        x, y = all_positions[node]
        ndata = subgraph.nodes[node]
        is_code = ndata.get("node_type") == "code"
        base_color = component_map[node]["color"]

        if is_code:
            code = ndata.get("code", "")
            label = _format_code_label(code)
            title = _code_tooltip(code)
            net.add_node(
                node,
                label=label,
                title=title,
                color=code_node_color,
                shape="box",
                borderWidth=2,
                font={"size": 11, "face": "monospace"},
                x=x,
                y=y,
                physics=False,
            )
        else:
            net.add_node(
                node,
                label=node,
                title=node,
                color=base_color,
                shape="dot",
                size=12,
                x=x,
                y=y,
                physics=False,
            )

    for s, o, edge_data in subgraph.edges(data=True):
        kind = edge_data.get("edge_kind")
        lbl = edge_data.get("label", "")
        if kind == "to_code":
            net.add_edge(
                s,
                o,
                label="",
                title="code snippet for this triplet",
                color={"color": "#adb5bd", "highlight": "#6c757d"},
                dashes=True,
                width=1,
            )
        elif kind == "from_code":
            net.add_edge(
                s,
                o,
                label="",
                title="",
                color={"color": "#adb5bd", "highlight": "#6c757d"},
                dashes=True,
                width=1,
            )
        else:
            net.add_edge(
                s,
                o,
                label=lbl,
                title=lbl or edge_data.get("triplet_key", ""),
            )

    net.toggle_physics(False)
    net.set_options(
        """
    {
      "nodes": {
        "font": { "size": 12 }
      },
      "edges": {
        "color": { "inherit": false, "color": "#848484" },
        "smooth": { "type": "continuous" },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } },
        "font": { "size": 10, "align": "middle" }
      },
      "physics": { "enabled": false },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """
    )

    out = exp_path / f"{save_as}.html"
    net.save_graph(str(out))
    return str(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize triplet2code JSON as an interactive graph.")
    p.add_argument("json_path", type=Path, help="Path to mem_graph_data_with_code.json (or JSON with triplet2code)")
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("/home/asagirova/arigraph/graph_outputs"),
        help="Directory for the HTML file",
    )
    p.add_argument("--save-as", default="triplet2code_graph", help="Basename without .html")
    args = p.parse_args()

    t2c = load_triplet2code_from_json(args.json_path)
    html_path = vis_triplet2code_graph(t2c, args.output_dir, save_as=args.save_as)
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
