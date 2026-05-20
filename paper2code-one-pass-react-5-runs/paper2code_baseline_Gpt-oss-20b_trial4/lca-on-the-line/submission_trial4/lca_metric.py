"""
Utilities for computing the Lowest Common Ancestor (LCA) distance
between two ImageNet class indices using the WordNet hierarchy.

The implementation follows the depth‑based definition:
    D_LCA(y', y) = depth(y) + depth(y') - 2 * depth(LCA(y, y'))
"""

import os
import json
from collections import defaultdict
from typing import Dict, Tuple

import nltk
from nltk.corpus import wordnet as wn

# -------------------------------------------------------------
# Mapping from ImageNet class index (0‑999) to WordNet synset ID
# -------------------------------------------------------------
#  The mapping file is part of the torchvision ImageNet class mapping.
#  For the purpose of this demo we ship a tiny mapping file
#  (`imagenet_synset_to_idx.json`) that covers the 200 images in the
#  sample dataset.  In a full‑scale experiment you would use the
#  official mapping from the ImageNet website.
# -------------------------------------------------------------
IMAGENET_MAP_FILE = os.path.join(
    os.path.dirname(__file__), "imagenet_synset_to_idx.json"
)

def load_imagenet_mapping() -> Dict[int, str]:
    """
    Returns a dictionary mapping ImageNet class index (0‑999) to
    WordNet synset ID (e.g. 'n01440764').
    """
    with open(IMAGENET_MAP_FILE, "r") as f:
        mapping = json.load(f)
    # Convert keys to int
    return {int(k): v for k, v in mapping.items()}

# Load once at module import
IMAGENET_MAP = load_imagenet_mapping()

# -------------------------------------------------------------
# Helper: compute depth of a synset in the WordNet hierarchy
# -------------------------------------------------------------
_synset_depth_cache: Dict[str, int] = {}
def synset_depth(synset_id: str) -> int:
    """
    Return the depth of the synset in the WordNet hierarchy.
    Depth is defined as the number of edges from the root
    to the synset. Caches results for speed.
    """
    if synset_id in _synset_depth_cache:
        return _synset_depth_cache[synset_id]
    syn = wn.synset(synset_id)
    depth = max([len(path) for path in syn.hypernym_paths()]) - 1
    _synset_depth_cache[synset_id] = depth
    return depth

# -------------------------------------------------------------
# Compute LCA distance between two class indices
# -------------------------------------------------------------
def get_lca_distance(pred_idx: int, target_idx: int) -> float:
    """
    Compute the LCA distance between predicted and target class indices.
    """
    if pred_idx == target_idx:
        return 0.0

    # Get synset IDs
    pred_syn = IMAGENET_MAP.get(pred_idx)
    target_syn = IMAGENET_MAP.get(target_idx)

    if pred_syn is None or target_syn is None:
        # If mapping missing, fall back to zero distance
        return 0.0

    # Find lowest common ancestor
    pred_synset = wn.synset(pred_syn)
    target_synset = wn.synset(target_syn)

    # Compute all ancestors
    pred_anc = set(pred_synset.closure(lambda s: s.hypernyms()))
    target_anc = set(target_synset.closure(lambda s: s.hypernyms()))

    lca_candidates = pred_anc.intersection(target_anc)
    if not lca_candidates:
        # No common ancestor found; treat as root
        lca_depth = 0
    else:
        # Choose the one with maximum depth (lowest in the tree)
        lca_depth = max(synset_depth(c.name()) for c in lca_candidates)

    depth_pred = synset_depth(pred_syn)
    depth_target = synset_depth(target_syn)

    # Depth‑based LCA distance
    distance = (depth_pred - lca_depth) + (depth_target - lca_depth)
    return float(distance)