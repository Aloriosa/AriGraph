import torch
import torch.nn as nn
import torch.optim as optim
from .utils import MLP

class MaskNetwork(nn.Module):
    """
    Predicts the probability of masking (blinding) a time‑step.
    Input: state observation.
    Output: probability in (0,1).
    """
    def __init__(self, obs_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # shape: (batch,)