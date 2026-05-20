import re
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, set_seed

# --------------------------------------------------------------------------- #
# 1. Parsing helpers
# --------------------------------------------------------------------------- #
def extract_answer_from_text(text: str) -> str:
    """
    Very simple heuristic to extract the final answer from a CoT rationale.
    Looks for patterns like 'The answer is X' or 'Answer: X'.
    Falls back to the last token if nothing else is found.
    """
    # Lowercase for case‑insensitive search
    lower = text.lower()

    # Look for "answer is" pattern
    match = re.search(r'answer\s+is\s+([^\s]+)', lower)
    if match:
        return match.group(1).strip().strip('.').strip('\'"')

    # Look for "answer:" pattern
    match = re.search(r'answer\s*[:\-]\s*([^\s]+)', lower)
    if match:
        return match.group(1).strip().strip('.').strip('\'"')

    # Fallback: take last token
    tokens = text.strip().split()
    if not tokens:
        return ""
    return tokens[-1].strip().strip('.').strip('\'"')

# --------------------------------------------------------------------------- #
# 2. Embedding helpers
# --------------------------------------------------------------------------- #
def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Embed a list of strings using the provided SentenceTransformer.
    Returns a 2‑D numpy array of shape (len(texts), hidden_size).
    """
    embeddings = model.encode(texts, batch_size=8, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

# --------------------------------------------------------------------------- #
# 3. Weighting helpers
# --------------------------------------------------------------------------- #
def centroid_proximity_weighting(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute inverse‑distance weights based on the centroid.
    Returns a 1‑D array of weights for each embedding.
    """
    centroid = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    normalized = distances / distances.sum()
    # avoid division by zero
    eps = 1e-8
    weights = 1.0 / (normalized + eps)
    return weights

def semantic_consensus_weighting(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute similarity‑based weights: each embedding gets the sum of its cosine
    similarities to all other embeddings.
    Returns a 1‑D array of weights for each embedding.
    """
    # Cosine similarity matrix (already normalized embeddings)
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, 0)  # exclude self similarity
    weights = sims.sum(axis=1)
    return weights

# --------------------------------------------------------------------------- #
# 4. Aggregation helpers
# --------------------------------------------------------------------------- #
def aggregate_by_weight(answers: List[str], weights: np.ndarray) -> Tuple[str, float]:
    """
    Sum weights per unique answer and return the answer with the highest total weight.
    """
    answer_to_weight: Dict[str, float] = {}
    for ans, w in zip(answers, weights):
        answer_to_weight[ans] = answer_to_weight.get(ans, 0.0) + w
    best_answer = max(answer_to_weight, key=answer_to_weight.get)
    best_score = answer_to_weight[best_answer]
    return best_answer, best_score