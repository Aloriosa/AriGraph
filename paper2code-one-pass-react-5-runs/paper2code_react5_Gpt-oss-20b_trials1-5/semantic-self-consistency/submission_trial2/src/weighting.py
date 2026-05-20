import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_cpw(embeddings):
    """
    Centroid Proximity Weighting (CPW).
    """
    centroid = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    norm_dist = distances / distances.sum()
    # Avoid division by zero
    norm_dist = np.where(norm_dist == 0, 1e-9, norm_dist)
    weights = 1.0 / norm_dist
    return weights

def compute_scw(embeddings):
    """
    Semantic Consensus Weighting (SCW) using pairwise cosine similarity.
    """
    sim_matrix = cosine_similarity(embeddings)
    scores = np.sum(sim_matrix, axis=1)
    return scores