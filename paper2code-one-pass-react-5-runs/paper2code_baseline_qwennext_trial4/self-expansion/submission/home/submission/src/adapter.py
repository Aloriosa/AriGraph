"""
Adapter implementation for SEMA algorithm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Adapter(nn.Module):
    """
    Modular adapter with representation descriptor (RD) as an autoencoder.
    """
    def __init__(self, input_dim=768, hidden_dim=64, reduction_ratio=8):
        super(Adapter, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.down_proj = nn.Linear(input_dim, input_dim // reduction_ratio)
        self.up_proj = nn.Linear(input_dim // reduction_ratio, input_dim)
        self.relu = nn.ReLU()
        
        # Representation descriptor: Autoencoder
        self.representation_descriptor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.zeros_(self.down_proj.bias)
        
    def forward(self, x):
        # Adapter branch
        adapter_out = self.relu(self.down_proj(x))
        adapter_out = self.up_proj(adapter_out)
        adapter_out = adapter_out + x  # Residual connection
        return adapter_out