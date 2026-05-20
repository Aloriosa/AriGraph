import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    """Deterministic policy conditioned on state and latent z."""
    def __init__(self, state_dim: int, latent_dim: int = 64,
                 action_dim: int = 1, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, state: torch.Tensor, z: torch.Tensor):
        return self.net(torch.cat([state, z], dim=-1))