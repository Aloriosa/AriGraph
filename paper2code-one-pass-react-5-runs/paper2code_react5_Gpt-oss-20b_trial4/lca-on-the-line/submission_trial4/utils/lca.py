"""
Utilities for computing the Lowest Common Ancestor (LCA) distance
between two ImageNet classes using WordNet and for computing
information‑content based LCA distances.  The module also provides
simple correlation utilities that are used by the evaluation script.
"""

import math
import numpy as np
from nltk.corpus import wordnet as wn
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# ------------------------------------------------------------------
# Helper functions for taxonomy traversal
# ------------------------------------------------------------------
def _compute_leaf_counts(synsets):
    """
    Recursively count the number of leaf descendants for each synset
    *within the provided ImageNet synset set*.

    A leaf is a synset that has no hyponyms that are also ImageNet
    classes.  This implementation only counts descendants that are
    present in ``synsets``; other WordNet hyponyms are ignored.

    Parameters
    ----------
    synsets : list[wn.synset]
        List of ImageNet synsets.

    Returns
    -------
    dict[wn.synset, int]
        Mapping from synset to the number of ImageNet leaf descendants.
    """
    synset_set = set(synsets)
    memo = {}

    def count(s):
        if s in memo:
            return memo[s]
        # Only consider hyponyms that belong to the ImageNet set
        hypos = [h for h in s.hyponyms() if h in synset_set]
        if not hypos:
            memo[s] = 1  # treat as leaf
        else:
            memo[s] = sum(count(h) for h in hypos)
        return memo[s]

    leaf_counts = {s: count(s) for s in synsets}
    return leaf_counts


# ------------------------------------------------------------------
# Information‑content based LCA distance
# ------------------------------------------------------------------
def compute_info_content(synsets):
    """
    Compute the information content (IC) for each synset in the
    provided list of ImageNet synsets.

    IC(s) = -log2(P(s)),  P(s) = (# leaf descendants of s) / (# total leaves)

    Parameters
    ----------
    synsets : list[wn.synset]
        List of WordNet synsets corresponding to the ImageNet classes.

    Returns
    -------
    dict[wn.synset, float]
        Mapping from synset to its IC value.
    """
    leaf_counts = _compute_leaf_counts(synsets)
    total_leaves = len(synsets)  # The ImageNet validation set has exactly 1000 leaf classes
    info_content = {s: -math.log2(c / total_leaves) for s, c in leaf_counts.items()}
    return info_content


def compute_lca_distance(target_idx, pred_idx, synsets, info_content_dict):
    """
    Compute the LCA distance D_LCA(y', y) = IC(y) - IC(LCA(y, y'))
    where y is the ground‑truth class and y′ is the predicted class.

    Parameters
    ----------
    target_idx : int
        Index of the ground‑truth class.
    pred_idx : int
        Index of the predicted class.
    synsets : list[wn.synset]
        List mapping class indices to synsets.
    info_content_dict : dict[wn.synset, float]
        Mapping from synset to its IC value.

    Returns
    -------
    float
        LCA distance (0.0 if the prediction is correct or no LCA found).
    """
    if target_idx == pred_idx:
        return 0.0

    try:
        target_syn = synsets[target_idx]
        pred_syn = synsets[pred_idx]
    except Exception:
        return 0.0

    lcas = target_syn.lowest_common_hypernyms(pred_syn)
    if not lcas:
        return 0.0

    lca = lcas[0]
    f_target = info_content_dict.get(target_syn, 0.0)
    f_lca = info_content_dict.get(lca, 0.0)
    return f_target - f_lca


def compute_average_lca(preds, targets, synsets, info_content_dict):
    """
    Compute the average LCA distance over all mispredictions.

    Parameters
    ----------
    preds : list[int] or np.ndarray
        Predicted class indices.
    targets : list[int] or np.ndarray
        Ground‑truth class indices.
    synsets : list[wn.synset]
        List mapping class indices to synsets.
    info_content_dict : dict[wn.synset, float]
        Mapping from synset to its IC value.

    Returns
    -------
    float
        Mean LCA distance over all mispredictions. 0.0 if no mispredictions.
    """
    total = 0.0
    count = 0
    for p, t in zip(preds, targets):
        if p != t:
            total += compute_lca_distance(t, p, synsets, info_content_dict)
            count += 1
    return total / count if count > 0 else 0.0


def compute_average_lca_matrix(preds, targets, lca_matrix):
    """
    Compute the average LCA distance using a pre‑computed pairwise LCA
    distance matrix.

    Parameters
    ----------
    preds : list[int] or np.ndarray
        Predicted class indices.
    targets : list[int] or np.ndarray
        Ground‑truth class indices.
    lca_matrix : np.ndarray
        Matrix where lca_matrix[i, j] is the LCA distance between class i
        and class j.

    Returns
    -------
    float
        Mean LCA distance over all mispredictions. 0.0 if no mispredictions.
    """
    total = 0.0
    count = 0
    for p, t in zip(preds, targets):
        if p != t:
            total += lca_matrix[t, p]
            count += 1
    return total / count if count > 0 else 0.0


# ------------------------------------------------------------------
# Correlation utilities
# ------------------------------------------------------------------
def compute_pearson_r2(x, y):
    """
    Compute Pearson correlation coefficient and R² between two arrays.

    Parameters
    ----------
    x, y : array‑like
        Two numeric sequences of the same length.

    Returns
    -------
    pearson : float
        Pearson correlation coefficient.
    r2 : float
        Coefficient of determination.
    """
    pearson, _ = pearsonr(x, y)
    r2 = r2_score(x, y)
    return pearson, r2


# ------------------------------------------------------------------
# Soft‑label alignment loss (skeleton)
# ------------------------------------------------------------------
def lca_alignment_loss(logits, targets, alignment_mode, LCA_matrix, lambda_weight=0.03):
    """
    Compute a soft‑label alignment loss based on the LCA matrix.
    This is a utility skeleton that can be plugged into a training loop.

    Parameters
    ----------
    logits : torch.Tensor
        Logits of shape (B, num_classes).
    targets : torch.Tensor
        LongTensor of ground‑truth class indices.
    alignment_mode : str
        Either 'BCE' or 'CE'.
    LCA_matrix : np.ndarray
        Pairwise LCA distance matrix (already normalised to [0,1]).
    lambda_weight : float
        Weight for the standard cross‑entropy loss.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    import torch.nn.functional as F
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
    ce_loss = - (one_hot * probs.log()).sum(dim=1).mean()

    # Reverse LCA matrix: higher value -> more similar
    rev_lca = 1.0 - LCA_matrix
    soft_targets = torch.from_numpy(rev_lca[targets]).to(logits.device).float()

    if alignment_mode == 'BCE':
        bcet = F.binary_cross_entropy_with_logits(logits, soft_targets, reduction='mean')
        loss = lambda_weight * ce_loss + bcet
    else:  # 'CE'
        loss = lambda_weight * ce_loss + (- (soft_targets * probs.log()).sum(dim=1).mean())

    return loss