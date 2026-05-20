from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from ontology.cookbook_ontology import (
    infer_node_type,
    normalize_entity,
    normalize_relation,
    relation_family,
)

from .schemas import EdgeRecord, NodeRecord, TripletRecord, TypedGraph


def _triplet_key(subject: str, relation: str, object_: str) -> str:
    return f"{subject}|{relation}|{object_}"


def _build_triplet_provenance(
    obs_episodic: Dict[str, Tuple[List[str], object]] | Dict[str, List[object]]
) -> Dict[str, List[str]]:
    triplet_to_obs: Dict[str, List[str]] = defaultdict(list)
    for idx, (obs, payload) in enumerate(obs_episodic.items()):
        if not payload:
            continue
        obs_triplets = payload[0] if isinstance(payload, (list, tuple)) and payload else []
        obs_id = f"obs_{idx:04d}"
        for t in obs_triplets:
            triplet_to_obs[t].append(obs_id)
    return triplet_to_obs


def materialize_typed_graph(
    graph_kind: str,
    raw_triplets: Iterable[List[object]],
    obs_episodic: Dict[str, object],
) -> TypedGraph:
    """Build typed graph sidecar with provenance and basic graph features."""
    triplet_provenance = _build_triplet_provenance(obs_episodic)
    triplets: List[TripletRecord] = []
    nodes: Dict[str, NodeRecord] = {}
    edges: Dict[str, EdgeRecord] = {}
    node_in_degree: Dict[str, int] = defaultdict(int)
    node_out_degree: Dict[str, int] = defaultdict(int)

    for i, row in enumerate(raw_triplets):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        subject = normalize_entity(str(row[0]))
        object_ = normalize_entity(str(row[1]))
        rel_data = row[2]
        relation = rel_data.get("label") if isinstance(rel_data, dict) else rel_data
        relation = normalize_relation(str(relation))
        if not subject or not object_ or not relation:
            continue

        t_key = _triplet_key(subject, relation, object_)
        t_id = f"triplet_{i:05d}"
        source_obs_ids = triplet_provenance.get(f"{subject}, {relation}, {object_}", [])
        triplets.append(
            TripletRecord(
                id=t_id,
                subject=subject,
                relation=relation,
                object=object_,
                source_obs_ids=source_obs_ids,
            )
        )

        for entity, role in ((subject, "subject"), (object_, "object")):
            if entity not in nodes:
                nodes[entity] = NodeRecord(
                    id=f"node_{len(nodes):05d}",
                    label=entity,
                    node_type=infer_node_type(entity, relation, role=role),
                    aliases=[],
                    support_count=0,
                    source_obs_ids=[],
                )
            nodes[entity].support_count += 1
            for obs_id in source_obs_ids:
                if obs_id not in nodes[entity].source_obs_ids:
                    nodes[entity].source_obs_ids.append(obs_id)

        edge_key = _triplet_key(subject, relation, object_)
        if edge_key not in edges:
            edges[edge_key] = EdgeRecord(
                id=f"edge_{len(edges):05d}",
                source=nodes[subject].id,
                target=nodes[object_].id,
                relation=relation,
                relation_family=relation_family(relation),
                source_obs_ids=[],
                support_count=0,
            )
        edges[edge_key].support_count += 1
        for obs_id in source_obs_ids:
            if obs_id not in edges[edge_key].source_obs_ids:
                edges[edge_key].source_obs_ids.append(obs_id)

        node_out_degree[nodes[subject].id] += 1
        node_in_degree[nodes[object_].id] += 1

    features: Dict[str, Dict[str, float]] = {}
    for node in nodes.values():
        indeg = float(node_in_degree.get(node.id, 0))
        outdeg = float(node_out_degree.get(node.id, 0))
        total = indeg + outdeg
        bridge = indeg * outdeg
        features[node.id] = {
            "in_degree": indeg,
            "out_degree": outdeg,
            "degree": total,
            "bridge_score": bridge,
            "support_count": float(node.support_count),
        }

    provenance = {}
    for idx, obs in enumerate(obs_episodic.keys()):
        provenance[f"obs_{idx:04d}"] = {"observation": obs}

    return TypedGraph(
        graph_kind=graph_kind,
        nodes=list(nodes.values()),
        edges=list(edges.values()),
        triplets=triplets,
        features=features,
        provenance=provenance,
    )
