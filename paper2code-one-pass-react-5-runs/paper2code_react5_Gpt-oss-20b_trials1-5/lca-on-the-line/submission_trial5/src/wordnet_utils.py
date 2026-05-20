"""
Utilities for working with WordNet and ImageNet synset IDs.
"""

import nltk
from nltk.corpus import wordnet as wn


def synset_from_id(syn_id: str) -> wn.synset:
    """
    Convert an ImageNet synset ID (e.g. 'n01440764') to an NLTK WordNet Synset.
    """
    return wn.synset(syn_id)


def lca_distance(syn_true: wn.synset, syn_pred: wn.synset) -> int:
    """
    Compute the Lowest Common Ancestor (LCA) distance between two synsets.
    According to the paper, f(y) is the depth of the true synset.
    The distance is depth(true) - depth(lca).
    """
    # Find all lowest common hypernyms and pick the one with maximum depth
    lcas = syn_true.lowest_common_hypernyms(syn_pred)
    if not lcas:
        # Fallback if WordNet fails to find a common ancestor
        return syn_true.min_depth()
    lca = max(lcas, key=lambda s: s.min_depth())
    depth_true = syn_true.min_depth()
    depth_lca = lca.min_depth()
    return depth_true - depth_lca


def batch_lca_distance(true_ids, pred_ids):
    """
    Compute the mean LCA distance for a batch.
    true_ids/pred_ids are lists of synset IDs (strings).
    """
    total = 0
    for t, p in zip(true_ids, pred_ids):
        total += lca_distance(synset_from_id(t), synset_from_id(p))
    return total / len(true_ids)