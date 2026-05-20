import torch
import random

def random_mapping(num_source_classes, num_target_classes, seed=0):
    """
    Returns a mapping from target classes [0..num_target_classes-1]
    to a random subset of source classes.
    """
    rng = random.Random(seed)
    source_indices = list(range(num_source_classes))
    rng.shuffle(source_indices)
    mapping = torch.tensor(source_indices[:num_target_classes], dtype=torch.long)
    return mapping