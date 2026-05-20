"""
Weighting and aggregation methods:
  - baseline majority vote
  - Centroid Proximity Weighting (CPW)
  - Semantic Consensus Weighting (SCW)
"""

import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict


def baseline_vote(answers: List[str]) -> str:
    """Return the most frequent answer."""
    counter = Counter(answers)
    return counter.most_common(1)[0][0]


def cpw_weight(
    embeddings: np.ndarray,
    answers: List[str],
) -> str:
    """
    Centroid Proximity Weighting.
    embeddings: (n_samples, d)
    answers: list of answer strings corresponding to embeddings
    """
    centroid = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    # avoid division by zero
    distances += 1e-9
    norm_dist = distances / distances.sum()
    weights = 1.0 / norm_dist
    # Sum weights per unique answer
    answer_weights: Dict[str, float] = {}
    for w, ans in zip(weights, answers):
        answer_weights[ans] = answer_weights.get(ans, 0.0) + w
    # Pick answer with highest total weight
    return max(answer_weights.items(), key=lambda x: x[1])[0]


def scw_weight(
    embeddings: np.ndarray,
    answers: List[str],
) -> str:
    """
    Semantic Consensus Weighting: sum of cosine similarities with all others.
    """
    sims = cosine_similarity(embeddings)  # (n, n)
    # Sum similarities per sample
    sim_scores = sims.sum(axis=1)
    # Aggregate by answer
    answer_scores: Dict[str, float] = {}
    for score, ans in zip(sim_scores, answers):
        answer_scores[ans] = answer_scores.get(ans, 0.0) + score
    return max(answer_scores.items(), key=lambda x: x[1])[0]