#!/usr/bin/env python3
"""
Utility functions for the CompoNet implementation.
"""
import torch
import numpy as np
import random
import os

class Identity(nn.Module):
    """
    Identity function as a placeholder for no encoder.
    """
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

def get_position_encoding(seq_len: int, d: int) -> np.ndarray:
    """
    Generate positional encoding as described in the paper.
    
    Args:
        seq_len: Length of sequence
        d: Dimension of encoding
        
    Returns:
        Positional encoding matrix
    """
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d, 2) * -(np.log(10000.0) / d))
    pe = np.zeros((seq_len, d))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

def logit2prob(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to probabilities using softmax.
    
    Args:
        logits: Logits tensor
        
    Returns:
        Probabilities tensor
    """
    return torch.softmax(logits, dim=-1)

def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_env(task):
    """
    Create a MetaWorld environment from a task.
    
    Args:
        task: MetaWorld task
        
    Returns:
        Gym environment
    """
    env = task
    return env

def create_writer(log_dir: str):
    """
    Create a tensorboard writer.
    
    Args:
        log_dir: Directory for logs
        
    Returns:
        Tensorboard writer
    """
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir)