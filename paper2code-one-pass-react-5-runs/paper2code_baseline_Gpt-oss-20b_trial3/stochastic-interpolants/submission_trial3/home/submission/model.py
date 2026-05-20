# model.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import time_embedding

class VelocityMLP(nn.Module):
    """
    Simple MLP that predicts a velocity vector of the same shape as the image.
    Input: concatenation of image pixels and time embedding.
    """
    def __init__(self, img_size=28, time_emb_dim=64, hidden_dim=512):
        super().__init__()
        self.img_size = img_size
        self.input_dim = img_size * img_size + time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, img_size * img_size),
        )

    def forward(self, x, t):
        """
        x: (B, C, H, W) – we assume C=1 for MNIST
        t: (B,) – time scalar in [0,1]
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, -1)  # (B, H*W)
        t_emb = time_embedding(t, dim=64)  # (B, time_emb_dim)
        inp = torch.cat([x_flat, t_emb], dim=1)  # (B, input_dim)
        out = self.net(inp)  # (B, H*W)
        out = out.view(B, C, H, W)
        return out