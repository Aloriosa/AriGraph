from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class TripletRecord:
    id: str
    subject: str
    relation: str
    object: str
    source_obs_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NodeRecord:
    id: str
    label: str
    node_type: str
    aliases: List[str] = field(default_factory=list)
    support_count: int = 0
    source_obs_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EdgeRecord:
    id: str
    source: str
    target: str
    relation: str
    relation_family: str
    source_obs_ids: List[str] = field(default_factory=list)
    support_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TypedGraph:
    graph_kind: str
    nodes: List[NodeRecord]
    edges: List[EdgeRecord]
    triplets: List[TripletRecord]
    features: Dict[str, Dict[str, float]]
    provenance: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_kind": self.graph_kind,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "triplets": [t.to_dict() for t in self.triplets],
            "features": self.features,
            "provenance": self.provenance,
        }


@dataclass
class SymbolAsset:
    id: str
    symbol: str
    path: str
    line_start: int
    line_end: int
    role_tags: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    api_surface: str = ""
    summary: str = ""
    code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaseCard:
    id: str
    card_type: str
    anchor_nodes: List[str]
    member_triplet_ids: List[str]
    source_obs_ids: List[str]
    intent: str
    priority: int
    retrieval_text: str
    generation_summary: str
    token_estimate: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryCard(BaseCard):
    linked_asset_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
