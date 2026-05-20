"""
Router implementation for SEMA algorithm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ExpandableRouter(nn.Module):
    """
    Expandable weighting router for mixture of adapters.
    """
    def __init__(self, input_dim=768, max_adapters=10):
        super(ExpandableRouter, self).__init__()
        self.input_dim = input_dim
        self.max_adapters = max_adapters
        # Initial weights for 1 adapter
        self.router_weights = nn.Parameter(torch.ones(1, self.input_dim))
        self.router_bias = nn.Parameter(torch.zeros(1))
        
        # Track current number of adapters
        self.current_adapters = 1
        
    def expand(self):
        """Expand router to accommodate new adapter"""
        if self.current_adapters < self.max_adapters:
            # Expand weights matrix by adding a new column
            new_weights = torch.zeros_like(self.router_weights.data[0, 0].unsqueeze(0))
        self.current_adapters += 1
        
    def forward(self, x, adapter_outputs, adapter_weights):
        # Compute router weights
        router_weights = torch.softmax(x @ self.router_weights.T + self.router_bias, dim=-1)
        # Weighted mixture of adapter outputs
        weighted_output = torch.sum(adapter_weights.unsqueeze(-1) * adapter_outputs, dim=1)
        return weighted_output