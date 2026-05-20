# lca.py – Lowest Common Ancestor distance implementation
# --------------------------------------------------------

import nltk
from nltk.corpus import wordnet as wn
import numpy as np

# Ensure the WordNet data is downloaded
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Map ImageNet synset IDs (e.g. 'n01440764') to WordNet Synset objects
def load_imagenet_synsets():
    """
    Returns a dict: imagenet_id -> wn.synset
    """
    mapping = {}
    for syn in wn.all_synsets():
        # WordNet synset name format: 'n01440764.n.01'
        # ImageNet ID is the first 8 chars
        im_id = syn.name().split(".")[0]
        if im_id.isdigit():
            mapping[im_id] = syn
    return mapping

IMGNET_SYNSETS = load_imagenet_synsets()

def depth(syn):
    """Return the depth of a synset in the hypernym graph."""
    return syn.min_depth()

def lca(syn1, syn2):
    """Return the lowest common ancestor synset of two synsets."""
    # Use the built‑in lowest_common_hypernyms
    lcas = syn1.lowest_common_hypernyms(syn2)
    if lcas:
        # There can be multiple; pick the one with minimal depth
        return min(lcas, key=lambda s: s.min_depth())
    # Fallback to root
    return wn.synset("entity.n.01")

def lca_distance(pred_id, gt_id):
    """
    Compute the LCA distance between predicted and ground‑truth ImageNet class.
    Uses the information‑content definition from the paper:
        D_LCA = depth(gt) + depth(pred) - 2 * depth(lca)
    """
    pred_syn = IMGNET_SYNSETS.get(pred_id)
    gt_syn = IMGNET_SYNSETS.get(gt_id)
    if pred_syn is None or gt_syn is None:
        # Unknown synset – treat as maximum distance
        return float("inf")
    d_gt = depth(gt_syn)
    d_pred = depth(pred_syn)
    lca_syn = lca(pred_syn, gt_syn)
    d_lca = depth(lca_syn)
    return d_gt + d_pred - 2 * d_lca

def elca_probability_distribution(probs, gt_id):
    """
    Expected LCA distance for a single sample given probability distribution.
    probs: numpy array of shape (num_classes,)
    gt_id: ground‑truth ImageNet ID (string)
    """
    gt_syn = IMGNET_SYNSETS.get(gt_id)
    if gt_syn is None:
        return float("inf")
    d_gt = depth(gt_syn)
    total = 0.0
    for idx, p in enumerate(probs):
        pred_id = str(idx).zfill(8)  # ImageNet ID is zero‑padded 8 digits
        pred_syn = IMGNET_SYNSETS.get(pred_id)
        if pred_syn is None:
            continue
        d_pred = depth(pred_syn)
        lca_syn = lca(pred_syn, gt_syn)
        d_lca = depth(lca_syn)
        total += p * (d_gt + d_pred - 2 * d_lca)
    return total