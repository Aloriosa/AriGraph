import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------------------------- #
#                     Neural network building blocks                         #
# --------------------------------------------------------------------------- #
class SharedBackbone(nn.Module):
    """Shared MLP backbone for all policies."""
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class Policy(nn.Module):
    """Discrete policy with a shared backbone and a per‑policy embedding."""
    def __init__(self, obs_dim, act_dim, shared_backbone, embed_dim=8):
        super().__init__()
        self.shared = shared_backbone
        self.embed = nn.Parameter(torch.zeros(embed_dim))
        self.fc = nn.Linear(64 + embed_dim, act_dim)

    def forward(self, x):
        h = self.shared(x)
        h = torch.cat([h, self.embed.expand_as(h)], dim=-1)
        logits = self.fc(h)
        return logits

    def sample(self, x):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, dist

    def log_prob(self, x, a):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(a)

class ValueNet(nn.Module):
    """Shared backbone + per‑policy head for value estimation."""
    def __init__(self, obs_dim, shared_backbone, embed_dim=8):
        super().__init__()
        self.shared = shared_backbone
        self.embed = nn.Parameter(torch.zeros(embed_dim))
        self.fc = nn.Linear(64 + embed_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        h = torch.cat([h, self.embed.expand_as(h)], dim=-1)
        val = self.fc(h)
        return val.squeeze(-1)