"""
Functions to compute the LCA distance between two WordNet synsets.
"""

import numpy as np
from typing import Dict
from nltk.corpus import wordnet as wn
from lca.hierarchy import SynsetMapper


def _lowest_common_ancestor(syn1: wn.synset, syn2: wn.synset) -> wn.synset:
    """
    Find the lowest common ancestor (deepest shared hypernym) of two synsets.
    """
    paths1 = syn1.hypernym_paths()[0]
    paths2 = syn2.hypernym_paths()[0]
    # Find common synsets
    common = set(paths1).intersection(paths2)
    if not common:
        return wn.synset('entity.n.01')  # root
    # Return the deepest common synset
    return max(common, key=lambda s: s.min_depth())


def compute_lca_distance(
    true_idx: int,
    pred_idx: int,
    mapper: SynsetMapper,
) -> float:
    """
    Compute the LCA distance between the true and predicted class indices.

    The distance is defined as:
        depth(true) + depth(pred) - 2 * depth(LCA)

    Parameters
    ----------
    true_idx : int
        Ground‑truth class index.
    pred_idx : int
        Predicted class index.
    mapper : SynsetMapper
        Utility that maps indices to synsets and depths.

    Returns
    -------
    float
        LCA distance.
    """
    syn_true = mapper.get_synset(true_idx)
    syn_pred = mapper.get_synset(pred_idx)

    lca = _lowest_common_ancestor(syn_true, syn_pred)

    depth_true = syn_true.min_depth()
    depth_pred = syn_pred.min_depth()
    depth_lca = lca.min_depth()

    distance = depth_true + depth_pred - 2 * depth_lca
    return float(distance)