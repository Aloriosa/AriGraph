"""
Utility module for computing Lowest Common Ancestor (LCA) distance on a toy taxonomy.
The taxonomy is a simple binary tree over the 10 CIFAR‑10 classes.
"""

from collections import defaultdict

# Define the toy taxonomy as a parent map
# Leaf nodes are the 10 CIFAR‑10 classes (by integer index)
# Internal nodes are named with uppercase letters
PARENT_MAP = {
    # Leaf nodes
    0: 'AA',  # airplane
    1: 'AA',  # automobile
    2: 'AB',  # bird
    3: 'AB',  # cat
    4: 'BA',  # deer
    5: 'BA',  # dog
    6: 'BB',  # frog
    7: 'BB',  # horse
    8: 'BB',  # ship
    9: 'BB',  # truck

    # Internal nodes
    'AA': 'A',
    'AB': 'A',
    'BA': 'B',
    'BB': 'B',
    'A': 'root',
    'B': 'root',
    'root': None
}

# Precompute depth for each node
DEPTH_MAP = {}

def _compute_depth(node):
    if node is None:
        return -1
    if node in DEPTH_MAP:
        return DEPTH_MAP[node]
    parent = PARENT_MAP[node]
    depth = 1 + _compute_depth(parent)
    DEPTH_MAP[node] = depth
    return depth

for node in PARENT_MAP:
    _compute_depth(node)

def lca_distance(pred, true):
    """
    Compute the LCA distance between two class indices.
    LCA distance = (depth(true) - depth(lca)) + (depth(pred) - depth(lca))
    """
    # Build ancestor sets for true
    ancestors_true = set()
    cur = true
    while cur is not None:
        ancestors_true.add(cur)
        cur = PARENT_MAP[cur]
    # Traverse pred ancestors until find common
    cur = pred
    while cur not in ancestors_true:
        cur = PARENT_MAP[cur]
    lca = cur
    depth_lca = DEPTH_MAP[lca]
    depth_true = DEPTH_MAP[true]
    depth_pred = DEPTH_MAP[pred]
    return (depth_true - depth_lca) + (depth_pred - depth_lca)

def mean_lca_distance(preds, trues):
    """
    Compute the mean LCA distance over a list of predictions and ground‑truth labels.
    """
    total = 0.0
    for p, t in zip(preds, trues):
        total += lca_distance(p, t)
    return total / len(preds)