from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .schemas import BaseCard, MemoryCard, SymbolAsset, TypedGraph


INTENT_RELATION_HINTS = {
    "configuration": "config",
    "procedure": "procedure",
    "evaluation": "evaluation",
    "mapping": "implementation",
    "artifact": "artifact",
    "diagnostics": "diagnostics",
    "performance": "performance",
}


def _estimate_tokens(text: str) -> int:
    # Fast deterministic approximation used in tests and budget prechecks.
    if not text:
        return 0
    return max(1, len(text) // 4)


def _node_label_by_id(graph: TypedGraph) -> Dict[str, str]:
    return {node.id: node.label for node in graph.nodes}


def _triplet_lookup(graph: TypedGraph) -> Dict[str, Tuple[str, str, str]]:
    return {t.id: (t.subject, t.relation, t.object) for t in graph.triplets}


def _cluster_triplets_by_anchor(graph: TypedGraph) -> Dict[str, List[str]]:
    clusters: Dict[str, List[str]] = defaultdict(list)
    for edge in graph.edges:
        clusters[edge.source].append(edge.id)
    return clusters


def _triplet_ids_for_edge_ids(graph: TypedGraph, edge_ids: Sequence[str]) -> List[str]:
    edge_by_id = {e.id: e for e in graph.edges}
    result: List[str] = []
    for t in graph.triplets:
        for edge_id in edge_ids:
            edge = edge_by_id.get(edge_id)
            if not edge:
                continue
            if t.subject == _node_label(edge.source, graph) and t.object == _node_label(edge.target, graph):
                if t.id not in result:
                    result.append(t.id)
    return result


def _node_label(node_id: str, graph: TypedGraph) -> str:
    for node in graph.nodes:
        if node.id == node_id:
            return node.label
    return node_id


def _collect_obs_ids(graph: TypedGraph, triplet_ids: Iterable[str]) -> List[str]:
    obs_ids: List[str] = []
    tri_by_id = {t.id: t for t in graph.triplets}
    for tid in triplet_ids:
        triplet = tri_by_id.get(tid)
        if not triplet:
            continue
        for obs_id in triplet.source_obs_ids:
            if obs_id not in obs_ids:
                obs_ids.append(obs_id)
    return obs_ids


def _summary_from_triplets(graph: TypedGraph, triplet_ids: Sequence[str]) -> str:
    tri = _triplet_lookup(graph)
    lines: List[str] = []
    for tid in triplet_ids:
        if tid not in tri:
            continue
        s, r, o = tri[tid]
        lines.append(f"- {s} --{r}--> {o}")
    return "\n".join(lines)


def build_paper_cards(graph: TypedGraph) -> List[BaseCard]:
    clusters = _cluster_triplets_by_anchor(graph)
    cards: List[BaseCard] = []
    for idx, (anchor_id, edge_ids) in enumerate(clusters.items()):
        triplet_ids = _triplet_ids_for_edge_ids(graph, edge_ids)
        if not triplet_ids:
            continue
        intents = {INTENT_RELATION_HINTS.get(e.relation_family, "implementation") for e in graph.edges if e.id in edge_ids}
        intent = sorted(intents)[0] if intents else "implementation"
        summary = _summary_from_triplets(graph, triplet_ids)
        retrieval_text = f"Intent: {intent}\nAnchor: {_node_label(anchor_id, graph)}\n{summary}"
        card = BaseCard(
            id=f"paper_card_{idx:04d}",
            card_type="paper",
            anchor_nodes=[anchor_id],
            member_triplet_ids=triplet_ids,
            source_obs_ids=_collect_obs_ids(graph, triplet_ids),
            intent=intent,
            priority=max(1, min(5, len(triplet_ids))),
            retrieval_text=retrieval_text,
            generation_summary=summary,
            token_estimate=_estimate_tokens(retrieval_text),
        )
        cards.append(card)
    return cards


def build_memory_cards(
    graph: TypedGraph,
    symbol_assets: Sequence[SymbolAsset] | None = None,
    triplet_to_asset_id: Dict[str, List[str]] | None = None,
) -> List[MemoryCard]:
    asset_map = {asset.id: asset for asset in (symbol_assets or [])}
    triplet_to_asset_id = triplet_to_asset_id or {}
    clusters = _cluster_triplets_by_anchor(graph)
    cards: List[MemoryCard] = []
    tri_lookup = _triplet_lookup(graph)

    for idx, (anchor_id, edge_ids) in enumerate(clusters.items()):
        triplet_ids = _triplet_ids_for_edge_ids(graph, edge_ids)
        if not triplet_ids:
            continue

        linked_assets: List[str] = []
        for tid in triplet_ids:
            triple = tri_lookup.get(tid)
            if not triple:
                continue
            key = f"{triple[0]}, {triple[1]}, {triple[2]}"
            for aid in triplet_to_asset_id.get(key, []):
                if aid in asset_map and aid not in linked_assets:
                    linked_assets.append(aid)

        summary = _summary_from_triplets(graph, triplet_ids)
        asset_summaries = []
        for aid in linked_assets:
            asset = asset_map.get(aid)
            if not asset:
                continue
            asset_summaries.append(f"* {asset.symbol} [{asset.path}:{asset.line_start}] - {asset.summary}")
        retrieval_text = (
            f"Intent: implementation\nAnchor: {_node_label(anchor_id, graph)}\n"
            f"{summary}\n"
            f"{chr(10).join(asset_summaries)}"
        )
        cards.append(
            MemoryCard(
                id=f"memory_card_{idx:04d}",
                card_type="memory",
                anchor_nodes=[anchor_id],
                member_triplet_ids=triplet_ids,
                source_obs_ids=_collect_obs_ids(graph, triplet_ids),
                intent="implementation",
                priority=max(1, min(5, len(triplet_ids) + len(linked_assets))),
                retrieval_text=retrieval_text,
                generation_summary=summary,
                token_estimate=_estimate_tokens(retrieval_text),
                linked_asset_ids=linked_assets,
            )
        )
    return cards


def cards_to_dict(cards: Sequence[BaseCard]) -> List[Dict[str, object]]:
    return [asdict(card) for card in cards]
