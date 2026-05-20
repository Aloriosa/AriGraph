#!/usr/bin/env python3
"""
Simple agent implementation for baseline comparison.
This is a standard actor-critic network without any continual learning mechanisms.
"""
import os
import torch
import torch.nn as nn
from typing import Optional

class SimpleAgent(nn.Module):
    """
    Simple agent with a shared encoder and separate heads for mean and logstd.
    Used as a baseline for comparison with CompoNet.
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim

        # Shared encoder
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Will be created when calling `reset_heads`
        self.fc_mean = None
        self.fc_logstd = None
        self.reset_heads()

    def reset_heads(self):
        """Reset the policy heads."""
        self.fc_mean = nn.Linear(256, self.act_dim)
        self.fc_logstd = nn.Linear(256, self.act_dim)

    def forward(self, x):
        x = self.fc(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        return mean, log_std

    def save(self, dirname):
        """Save the model."""
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, f"{dirname}/model.pt")

    @staticmethod
    def load(dirname, map_location=None, reset_heads=False):
        """Load the model."""
        model = torch.load(f"{dirname}/model.pt", map_location=map_location)
        if reset_heads:
            model.reset_heads()
        return model