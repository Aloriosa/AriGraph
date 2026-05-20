import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IQLQ(nn.Module):
    """Simple 3‑layer MLP Q‑network conditioned on state and latent z."""
    def __init__(self, state_dim, action_dim, latent_dim=32, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state, action, z):
        x = torch.cat([state, action, z], dim=-1)
        return self.net(x).squeeze(-1)

class IQLPolicy(nn.Module):
    """Deterministic policy: a 3‑layer MLP conditioned on state and z."""
    def __init__(self, state_dim, action_dim, latent_dim=32, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )

    def forward(self, state, z):
        x = torch.cat([state, z], dim=-1)
        return self.net(x)