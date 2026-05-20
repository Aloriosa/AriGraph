from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional, Callable

from .schemas import BaseCard, MemoryCard
import numpy as np


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


Embedder = Callable[[Sequence[str]], object]
"""Callable that maps a list of strings to a 2D array-like of embeddings.

Accepted return shapes: numpy.ndarray (N, dim), torch.Tensor (N, dim), or
list of N vectors. The retrieval code coerces all of them to numpy."""

@dataclass
class RetrievalConfig:
    per_paper_card: int = 4
    total_budget: int = 32
    min_score: float = 0.05
    # Optional dense scorer. When provided, replaces the Jaccard-based "dense_proxy"
    # with true embedding cosine. Falls back to Jaccard if embedding fails.
    embedder: Optional[Embedder] = None
    # Weight given to embedding cosine in the final score; the rest is split between
    # intent and compactness. Lexical Jaccard becomes a small tiebreaker when embeddings
    # are present (helps disambiguate near-duplicate vectors).
    dense_weight: float = 0.55
    lex_weight: float = 0.15
    intent_weight: float = 0.25
    compactness_weight: float = 0.15


def _to_numpy_2d(value) -> np.ndarray:
    """Coerce embedder output to a (N, dim) numpy array."""
    try:
        import torch  # noqa: WPS433 - optional import
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-8)


def _safe_embed_matrix(embedder: Embedder, texts: Sequence[str]) -> Optional[np.ndarray]:
    if not texts:
        return None
    try:
        raw = embedder(list(texts))
    except Exception:
        return None
    try:
        arr = _to_numpy_2d(raw)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] != len(texts):
        return None
    return _l2_normalize(arr.astype(np.float32, copy=False))



def retrieve_memory_cards(
    paper_cards: Sequence[BaseCard],
    memory_cards: Sequence[MemoryCard],
    config: RetrievalConfig | None = None,
) -> List[Tuple[str, str, float]]:
    """Retrieve memory cards for each paper card."""
    config = config or RetrievalConfig()
    selections: List[Tuple[str, str, float]] = []
    selected_memory_ids = set()


    # Precompute memory-card embeddings once (independent of which paper card we score).
    mem_matrix: Optional[np.ndarray] = None
    if config.embedder is not None and memory_cards:
        mem_matrix = _safe_embed_matrix(
            config.embedder, [mc.retrieval_text for mc in memory_cards]
        )

    for pcard in sorted(paper_cards, key=lambda c: (-c.priority, c.id)):

        #queries = _build_queries(pcard)
        # Per-paper-card query embeddings (small batch).
        query_matrix: Optional[np.ndarray] = None
        if mem_matrix is not None:
            query_matrix = _safe_embed_matrix(config.embedder, [pcard.retrieval_text])


        ranked: List[Tuple[MemoryCard, float]] = []
        for j, mcard in enumerate(memory_cards):

            if query_matrix is not None and mem_matrix is not None:
                # Cosine == dot for L2-normalized rows; max over query variants.
                sims = query_matrix @ mem_matrix[j]
                # Clamp to [0, 1] — Contriever similarities can dip slightly negative for unrelated text.
                dense = float(max(0.0, np.max(sims)))
            else:
                dense = _token_overlap_score(pcard.retrieval_text, mcard.retrieval_text)

            # Lexical tiebreaker (kept even with embeddings: catches exact symbol matches like 'PPO').
            lex = _token_overlap_score(pcard.retrieval_text, mcard.retrieval_text)

            intent = _intent_score(pcard, mcard)
            # Favor concise reusable cards for prompt budget.
            compactness = 1.0 / max(1.0, mcard.token_estimate / 800.0)
            #score = 0.55 * dense + 0.30 * intent + 0.15 * compactness
            score = (
                config.dense_weight * dense
                + config.lex_weight * lex
                + config.intent_weight * intent
                + config.compactness_weight * compactness
            )
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
