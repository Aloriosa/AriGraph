#!/usr/bin/env python3
"""
Finetune agent implementation for baseline comparison.
This agent fine-tunes a single network on each new task without any memory of previous tasks.
"""
import os
import torch
import torch.nn as nn
from typing import Optional

class FinetuneAgent(nn.Module):
    """
    Finetune agent that fine-tunes a single network on each new task.
    This serves as a baseline for catastrophic forgetting.
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Shared network for both mean and logstd
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # Policy heads
        self.mean_head = nn.Linear(256, act_dim)
        self.logstd_head = nn.Linear(256, act_dim)
        
    def forward(self, x):
        x = self.network(x)
        mean = self.mean_head(x)
        log_std = self.logstd_head(x)
        return mean, log_std
    
    def save(self, dirname):
        """Save the model."""
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, f"{dirname}/model.pt")
    
    @staticmethod
    def load(dirname, map_location=None):
        """Load the model."""
        model = torch.load(f"{dirname}/model.pt", map_location=map_location)
        return model