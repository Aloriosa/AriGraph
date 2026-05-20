#!/usr/bin/env python3
"""
Implementation of expandable weighting router for SEMA
Based on paper_card_0004 and paper_card_0046
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpandableWeightingRouter(nn.Module):
    """
    Expandable weighting router that dynamically combines multiple adapters
    Implements the soft weighting mechanism as described in paper_card_0004
    - Uses learned weights to combine adapters
    - Outputs probability distribution over adapters
    - Implemented as linear mapping + softmax
    - Expands when new adapter is added
    """
    
    def __init__(self, input_dim, num_adapters=1, hidden_dim=128, dropout=0.1):
        super(ExpandableWeightingRouter, self).__init__()
        self.input_dim = input_dim
        self.num_adapters = num_adapters
        self.hidden_dim = hidden_dim
        
        # Router network: maps input features to adapter weights
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_adapters)
        )
        
        # Initialize weights
        for module in self.router.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
    
    def forward(self, x, adapter_outputs):
        """
        Forward pass through the router
        x: (batch_size, seq_len, input_dim) - input features
        adapter_outputs: list of tensors, each (batch_size, seq_len, input_dim)
        
        Returns: weighted combination of adapter outputs
        """
        # Get weights for each adapter
        weights = self.router(x)  # (batch_size, seq_len, num_adapters)
        weights = F.softmax(weights, dim=-1)  # Softmax to get probabilities
        
        # Combine adapter outputs with weights
        # adapter_outputs: list of tensors of shape (batch_size, seq_len, input_dim)
        # weights: (batch_size, seq_len, num_adapters)
        
        # Stack adapter outputs
        adapter_stack = torch.stack(adapter_outputs, dim=-1)  # (batch_size, seq_len, input_dim, num_adapters)
        
        # Expand weights to match adapter_stack dimensions
        weights_expanded = weights.unsqueeze(-2)  # (batch_size, seq_len, 1, num_adapters)
        
        # Weighted combination
        weighted_output = torch.sum(adapter_stack * weights_expanded, dim=-1)  # (batch_size, seq_len, input_dim)
        
        return weighted_output, weights
    
    def add_adapter(self):
        """
        Add a new adapter to the router
        """
        # Get current weights
        current_weights = self.router[-1].weight.data  # (num_adapters, hidden_dim)
        current_bias = self.router[-1].bias.data  # (num_adapters,)
        
        # Increase number of adapters
        self.num_adapters += 1
        
        # Create new router with increased output dimension
        new_router = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.num_adapters)
        )
        
        # Initialize new weights
        nn.init.normal_(new_router[0].weight, std=0.02)
        nn.init.zeros_(new_router[0].bias)
        nn.init.normal_(new_router[3].weight, std=0.02)
        nn.init.zeros_(new_router[3].bias)
        
        # Copy old weights
        new_router[0].weight.data = self.router[0].weight.data.clone()
        new_router[0].bias.data = self.router[0].bias.data.clone()
        new_router[1].weight.data = self.router[1].weight.data.clone()
        new_router[1].bias.data = self.router[1].bias.data.clone()
        new_router[2].weight.data = self.router[2].weight.data.clone()
        new_router[2].bias.data = self.router[2].bias.data.clone()
        
        # Copy old weights to new router (first num_adapters-1)
        new_router[3].weight.data[:-1, :] = current_weights
        new_router[3].bias.data[:-1] = current_bias
        
        # Initialize new adapter weight to 1/num_adapters (uniform)
        new_router[3].weight.data[-1, :] = torch.randn_like(new_router[3].weight.data[-1, :]) * 0.01
        new_router[3].bias.data[-1] = torch.log(torch.tensor(1.0 / self.num_adapters))
        
        self.router = new_router