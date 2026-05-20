from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .schemas import BaseCard, MemoryCard


def _token_overlap_score(a: str, b: str) -> float:
    a_tokens = {t for t in a.lower().split() if len(t) > 2}
    b_tokens = {t for t in b.lower().split() if len(t) > 2}
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0


def _intent_score(paper_card: BaseCard, memory_card: MemoryCard) -> float:
    if paper_card.intent == memory_card.intent:
        return 1.0
    if paper_card.intent in memory_card.retrieval_text:
        return 0.7
    return 0.3


@dataclass
class RetrievalConfig:
    per_paper_card: int = 4
    total_budget: int = 32
    min_score: float = 0.05


def retrieve_memory_cards(
    paper_cards: Sequence[BaseCard],
    memory_cards: Sequence[MemoryCard],
    config: RetrievalConfig | None = None,
) -> List[Tuple[str, str, float]]:
    """Retrieve memory cards for each paper card with intent and lexical scoring."""
    config = config or RetrievalConfig()
    selections: List[Tuple[str, str, float]] = []
    selected_memory_ids = set()

    for pcard in sorted(paper_cards, key=lambda c: (-c.priority, c.id)):
        ranked: List[Tuple[MemoryCard, float]] = []
        for mcard in memory_cards:
            dense_proxy = _token_overlap_score(pcard.retrieval_text, mcard.retrieval_text)
            intent = _intent_score(pcard, mcard)
            # Favor concise reusable cards for prompt budget.
            compactness = 1.0 / max(1.0, mcard.token_estimate / 800.0)
            score = 0.55 * dense_proxy + 0.30 * intent + 0.15 * compactness
            if score >= config.min_score:
                ranked.append((mcard, score))
        ranked.sort(key=lambda x: x[1], reverse=True)

        chosen = 0
        used_anchors = set()
        for mcard, score in ranked:
            if chosen >= config.per_paper_card:
                break
            if len(selections) >= config.total_budget:
                break
            # diversification: avoid cards from same anchor in a query set
            anchor = mcard.anchor_nodes[0] if mcard.anchor_nodes else mcard.id
            if anchor in used_anchors:
                continue
            used_anchors.add(anchor)
            selections.append((pcard.id, mcard.id, round(score, 4)))
            selected_memory_ids.add(mcard.id)
            chosen += 1

    # Additional global dedupe budget guard.
    unique: List[Tuple[str, str, float]] = []
    seen_pairs = set()
    for row in selections:
        key = (row[0], row[1])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        unique.append(row)
        if len(unique) >= config.total_budget:
            break
    return unique
