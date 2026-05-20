"""
Utilities to construct a latent class hierarchy via hierarchical
K‑means clustering and to compute LCA distances with respect to this
hierarchy.
"""

import numpy as np
from sklearn.cluster import KMeans


def build_latent_hierarchy(class_features, max_levels=9):
    """
    Build a hierarchical K‑means clustering for the given class features.

    Parameters
    ----------
    class_features : np.ndarray
        Array of shape (num_classes, feature_dim).
    max_levels : int
        Number of binary splits (2**max_levels < num_classes).

    Returns
    -------
    list[np.ndarray]
        List of cluster assignment arrays, one per level.
        assignment[level][cls_idx] gives the cluster id at that level.
    """
    assignments = []
    X = class_features
    for level in range(1, max_levels + 1):
        n_clusters = 2 ** level
        if n_clusters >= X.shape[0]:
            # Stop if we cannot split further
            break
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        assignments.append(labels)
    return assignments


def pair_lca_level(assignments, i, j):
    """
    Find the lowest common ancestor level for two classes i and j
    given the assignments list.

    Parameters
    ----------
    assignments : list[np.ndarray]
        List of cluster assignment arrays per level.
    i, j : int
        Indices of the two classes.

    Returns
    -------
    int
        Index of the lowest level where the two classes share the same cluster.
    """
    for level in reversed(range(len(assignments))):
        if assignments[level][i] == assignments[level][j]:
            return level
    return 0  # root (implicit)


def compute_latent_lca_distance(i, j, assignments):
    """
    Compute LCA distance between two classes in the latent hierarchy.
    Distance is defined as the number of levels above the LCA.

    Parameters
    ----------
    i, j : int
        Indices of the two classes.
    assignments : list[np.ndarray]
        List of cluster assignment arrays per level.

    Returns
    -------
    float
        LCA distance.
    """
    lca_level = pair_lca_level(assignments, i, j)
    depth = len(assignments)
    return float(depth - lca_level - 1)


def build_lca_matrix(assignments):
    """
    Build a full pairwise LCA distance matrix from the hierarchical
    cluster assignments.

    Parameters
    ----------
    assignments : list[np.ndarray]
        List of cluster assignment arrays per level.

    Returns
    -------
    np.ndarray
        Matrix where mat[i, j] is the LCA distance between class i and j.
    """
    n = len(assignments[0])
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_latent_lca_distance(i, j, assignments)
            mat[i, j] = mat[j, i] = d
    return mat