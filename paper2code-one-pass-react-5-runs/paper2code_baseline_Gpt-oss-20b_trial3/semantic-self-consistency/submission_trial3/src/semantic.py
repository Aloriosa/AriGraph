import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import re


# Simple regex to extract the last numeric answer or last word
ANSWER_REGEX = re.compile(r"(?P<answer>\d+\.?\d*|[A-Za-z]+)$", re.IGNORECASE)


def extract_answer(text: str) -> str:
    """
    Extract the final answer token from a CoT completion.
    Returns an empty string if nothing matches.
    """
    match = ANSWER_REGEX.search(text.strip())
    return match.group("answer") if match else ""


def embed_texts(
    texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Embed a list of texts using a sentence‑transformer model.
    """
    embedder = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute a cosine similarity matrix for a set of embeddings.
    """
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    normalized = embeddings / norm
    sim_matrix = np.dot(normalized, normalized.T)
    return sim_matrix


def semantic_consensus_weighting(
    answers: List[str], embeddings: np.ndarray
) -> str:
    """
    Apply Semantic Consensus Weighting (SCW).
    For each sample, sum its cosine similarity with all others.
    Then aggregate these weights per unique answer and pick the answer
    with the highest total weight.
    """
    sim_matrix = cosine_similarity_matrix(embeddings)
    # Sum similarity for each sample (excluding self‑similarity)
    weights = np.sum(sim_matrix, axis=1) - 1.0
    # Aggregate weights per answer
    answer_weights: Dict[str, float] = {}
    for ans, w in zip(answers, weights):
        answer_weights[ans] = answer_weights.get(ans, 0.0) + w
    # Return answer with maximum weight
    best_answer = max(answer_weights.items(), key=lambda kv: kv[1])[0]
    return best_answer