#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class MaskNet(nn.Module):
    """
    A tiny neural network that maps an observation to a probability
    of masking the underlying agent's action (mask=1) or not (mask=0).
    """
    def __init__(self, obs_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)  # logits for {mask, keep}

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits

    def act(self, obs, deterministic=False):
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            m = Categorical(probs)
            action = m.sample()
        return action.item(), probs.detach().cpu().numpy()