"""
Utility functions used by the training scripts.
"""

import math
import json
import os
from pathlib import Path
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 0):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(tensor, device):
    """Move a tensor to the specified device."""
    return tensor.to(device)


def log_results(
    log_file: str,
    task_id: int,
    success_rate: float,
    episode_rewards: list,
    extra: Dict = None,
):
    """Append a line to the log file."""
    with open(log_file, "a") as f:
        f.write(
            f"Task {task_id}: Success rate = {success_rate:.4f}, "
            f"Avg reward = {np.mean(episode_rewards):.2f}\n"
        )
    if extra:
        # Store JSON per task for later aggregation
        Path("tmp").mkdir(exist_ok=True)
        with open(Path("tmp") / f"task_{task_id}.json", "w") as f:
            json.dump(extra, f, indent=2)


def get_action_distribution(action_logits: torch.Tensor, action_dim: int, discrete: bool):
    """
    Return a distribution object used for sampling actions.
    For discrete actions we use Categorical, for continuous we use Normal.
    """
    if discrete:
        dist = torch.distributions.Categorical(logits=action_logits)
    else:
        # Assume action_logits are the means of a Gaussian with fixed std
        std = torch.full_like(action_logits, 0.1)
        dist = torch.distributions.Normal(action_logits, std)
    return dist