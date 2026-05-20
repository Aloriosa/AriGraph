"""Utility functions for reproducibility and logging."""
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42) -> None:
    """Set random seed for all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def discounted_returns(rewards, gamma=0.99):
    """Compute discounted returns for a trajectory."""
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns