# Definition of the velocity network.

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_timestep_embedding


class VelocityNet(nn.Module):
    """
    Simple MLP that predicts the velocity vector for an image
    conditioned on a time embedding.
    Input: flattened image of shape [batch, 784]
    Time embedding: sinusoidal embedding of shape [batch, emb_dim]
    Output: velocity of shape [batch, 784]
    """

    def __init__(self, img_dim=784, emb_dim=256, hidden_dim=512):
        super().__init__()
        self.img_dim = img_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(img_dim + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, img_dim),
        )

    def forward(self, x, t):
        """
        x: [batch, img_dim]   - flattened image
        t: [batch]            - time scalar in [0,1]
        """
        emb = get_timestep_embedding(t, self.emb_dim)
        inp = torch.cat([x, emb], dim=1)
        return self.net(inp)