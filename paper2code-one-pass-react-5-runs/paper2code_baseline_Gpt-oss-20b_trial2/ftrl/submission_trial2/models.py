"""
Simple linear policy and value network for REINFORCE.
The policy outputs a probability of moving right (action 1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearPolicy(nn.Module):
    def __init__(self, obs_dim=1, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # probability of action 1 (right)
        )

    def forward(self, obs):
        # obs: [batch, obs_dim]
        return self.net(obs)

    def act(self, obs, deterministic=False):
        probs = self.forward(obs).detach().cpu().numpy().squeeze()
        if deterministic:
            return (probs > 0.5).astype(int)
        else:
            return (self.rng.binomial(1, probs)).astype(int)

class LinearBaseline(nn.Module):
    def __init__(self, obs_dim=1):
        super().__init__()
        self.net = nn.Linear(obs_dim, 1)

    def forward(self, obs):
        return self.net(obs)