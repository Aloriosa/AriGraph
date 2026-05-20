import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Flatten(nn.Module):
    """Simple flatten layer for MLPs."""
    def forward(self, x):
        return x.view(x.size(0), -1)