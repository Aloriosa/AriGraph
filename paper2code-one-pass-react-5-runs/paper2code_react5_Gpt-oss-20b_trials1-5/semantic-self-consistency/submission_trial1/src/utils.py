"""
Utility functions used by the experiment.
"""

import re
from typing import List, Tuple, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def parse_answer(text: str) -> str:
    """
    Extract the final answer from a chain‑of‑thought generation.
    The function is deliberately permissive to handle the
    various “Answer is X”, “The answer is X”, “Answer: X” patterns
    used in the datasets.
    """
    # Common patterns
    patterns = [
        r"Answer is ([^\s]+)",
        r"The answer is ([^\s]+)",
        r"Answer: ([^\s]+)",
        r"Answer\s*—\s*([^\s]+)",
        r"Answer\s*:\s*([^\s]+)",
        r"Answer:\s*([^\s]+)",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Fallback: take the last token that looks like a number or a letter
    tokens = text.strip().split()
    for token in reversed(tokens):
        if re.match(r"^[A-Za-z0-9\.\-]+$", token):
            return token.strip()
    # If nothing found, return empty string
    return ""


def load_dataset_by_name(name: str):
    """
    Load the test split of the dataset specified by *name*.
    The function is tolerant to a few common aliases.
    """
    from datasets import load_dataset

    name = name.lower()
    if name in ["aquarate", "aqua-rat", "aqua_rat", "aquarat"]:
        # AQuA‑RAT
        ds = load_dataset("AQuA-RAT", split="test")
    elif name in ["svamp", "svamp-0.2"]:
        ds = load_dataset("SVAMP", split="test")
    elif name in ["strategyqa", "strategyqa-2022"]:
        ds = load_dataset("strategyqa", split="test")
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    # Normalise fields: the datasets use different key names for the answer.
    # We create a unified `answer` field.
    if "answer" not in ds.column_names:
        if "answers" in ds.column_names:
            ds = ds.rename_column("answers", "answer")
        elif "label" in ds.column_names:
            ds = ds.rename_column("label", "answer")
        else:
            # As a last resort, try to infer from the question text
            pass
    return ds


def compute_cpw_weights(embeddings: np.ndarray) -> np.ndarray:
    """
    Centroid Proximity Weighting:
        1. Compute centroid of embeddings.
        2. Compute Euclidean distance of each point to centroid.
        3. Normalise distances.
        4. Weight = 1 / normalized_distance.
    """
    centroid = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    norm_dist = distances / np.sum(distances)
    # Avoid division by zero
    norm_dist[norm_dist == 0] = 1e-12
    weights = 1.0 / norm_dist
    return weights


def compute_scw_weights(embeddings: np.ndarray) -> np.ndarray:
    """
    Semantic Consensus Weighting:
        1. Compute pairwise cosine similarity.
        2. For each sample, sum its similarities to all others.
    """
    sims = cosine_similarity(embeddings)
    # Sum across rows (exclude self‑similarity if desired)
    scores = np.sum(sims, axis=1)
    return scores