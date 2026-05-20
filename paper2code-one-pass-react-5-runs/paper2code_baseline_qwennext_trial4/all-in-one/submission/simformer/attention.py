"""
Attention mechanisms for Simformer
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Attention(nn.Module):
    """
    Attention mechanism for Simformer.
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 num_heads: int,
                 dropout: float = 0.1):
        """
        Initialize attention mechanism.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(Attention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Attention mask
        Returns:
            output: Output tensor
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Project to hidden dimensions
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        
        # Split into multiple heads
        query = query.view(batch_size, seq_len, self.num_heads, -1)
        key = key.view(batch_size, seq_len, self.num_heads, -1)
        value = value.view(batch_size, value.size(1), self.num_heads, -1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.hidden_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        # Compute attention weights
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout_layer(weights)
        
        # Apply attention
        output = torch.matmul(weights, value)
        
        # Combine heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Apply output projection
        output = self.out_proj(output)
        
        return output