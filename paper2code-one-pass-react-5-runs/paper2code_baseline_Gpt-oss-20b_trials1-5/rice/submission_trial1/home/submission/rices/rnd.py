#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomNetwork(nn.Module):
    """
    Fixed random network used as the target for RND.
    """
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Freeze parameters
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.layers(x)

class PredictorNetwork(nn.Module):
    """
    Trainable network that predicts the output of the random network.
    """
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)