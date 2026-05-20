import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MLP(nn.Module):
    """Simple 2‑layer MLP."""
    def __init__(self, input_dim, hidden_dim, output_dim, activation=nn.Tanh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

def get_action_space_dims(env):
    """Return observation and action dimensions."""
    if isinstance(env.observation_space, spaces.Box):
        obs_dim = env.observation_space.shape[0]
    else:
        raise ValueError("Only continuous observation spaces are supported.")
    if isinstance(env.action_space, spaces.Box):
        act_dim = env.action_space.shape[0]
    else:
        raise ValueError("Only continuous action spaces are supported.")
    return obs_dim, act_dim