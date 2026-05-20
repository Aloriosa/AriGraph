#!/usr/bin/env python3
"""
Implementation of lightweight adapter module for SEMA
Based on paper_card_0001 and paper_card_0006
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    """
    Lightweight adapter module for Vision Transformer
    Implements the functional adapter as described in paper_card_0006
    - Dimension: 64 (as specified)
    - Activation: GELU
    - Dropout: 0.1
    - Inserted as side branch of MLP
    """
    
    def __init__(self, input_dim, adapter_dim=64, dropout=0.1):
        super(Adapter, self).__init__()
        self.input_dim = input_dim
        self.adapter_dim = adapter_dim
        
        # Down projection
        self.down_proj = nn.Linear(input_dim, adapter_dim)
        # Up projection
        self.up_proj = nn.Linear(adapter_dim, input_dim)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Activation
        self.activation = nn.GELU()
        
        # Initialize weights (as in paper)
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        """
        Apply adapter to input x
        x: (batch_size, seq_len, input_dim)
        Returns: (batch_size, seq_len, input_dim)
        """
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return x + residual  # Residual connection