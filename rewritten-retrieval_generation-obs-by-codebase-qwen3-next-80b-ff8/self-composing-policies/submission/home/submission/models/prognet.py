#!/usr/bin/env python3
"""
Progressive Network implementation for baseline comparison.
This architecture adds new columns for each new task while preserving previous knowledge.
"""
import os
import torch
import torch.nn as nn
from typing import List

class ProgNet(nn.Module):
    """
    Progressive Network that adds new columns for each new task.
    This is a baseline for continual learning that avoids catastrophic forgetting
    by preserving previous task knowledge in separate columns.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        
        # Shared bottom layer
        self.bottom = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )
        
        # List of columns (one for each task)
        self.columns = nn.ModuleList()
        
        # Output heads for each task
        self.output_heads = nn.ModuleList()
        
        # Current task index
        self.current_task = 0
        
    def add_task(self):
        """Add a new task column and output head."""
        # Create new column
        column = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.columns.append(column)
        
        # Create output head for this task
        mean_head = nn.Linear(self.hidden_dim, self.act_dim)
        logstd_head = nn.Linear(self.hidden_dim, self.act_dim)
        self.output_heads.append(nn.ModuleList([mean_head, logstd_head]))
        
        self.current_task = len(self.columns) - 1
        
    def forward(self, x):
        """Forward pass for the current task."""
        # Pass through bottom layer
        bottom_out = self.bottom(x)
        
        # Get output from current task column
        column_out = self.columns[self.current_task](x)
        
        # Combine bottom and column outputs
        combined = bottom_out + column_out
        
        # Get mean and logstd from current task output head
        mean_head, logstd_head = self.output_heads[self.current_task]
        mean = mean_head(combined)
        log_std = logstd_head(combined)
        
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