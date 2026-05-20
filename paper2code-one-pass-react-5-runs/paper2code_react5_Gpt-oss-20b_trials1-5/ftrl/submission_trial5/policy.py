#!/usr/bin/env python3
"""
Simple linear policy for AppleRetrieval.
Maps the single scalar observation to action logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, input_dim=1, output_dim=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        x: Tensor of shape (batch, input_dim)
        Returns logits of shape (batch, output_dim)
        """
        return self.linear(x)

    def get_action(self, obs, deterministic=False):
        """
        obs: numpy array of shape (input_dim,)
        Returns action (int)
        """
        with torch.no_grad():
            logits = self.forward(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                return probs.argmax().item()
            else:
                dist = torch.distributions.Categorical(probs)
                return dist.sample().item()

    def get_action_probs(self, obs):
        """
        Returns the probability distribution over actions as a numpy array.
        """
        logits = self.forward(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        probs = F.softmax(logits, dim=-1)
        return probs.squeeze(0).cpu().numpy()