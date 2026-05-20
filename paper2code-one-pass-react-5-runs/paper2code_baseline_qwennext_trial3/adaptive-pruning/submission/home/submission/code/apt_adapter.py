#!/usr/bin/env python3
"""
APT Adapter implementation based on the paper
Adaptive Pruning and Tuning (APT) for efficient fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

class APTAdapter(nn.Module):
    """
    APT Adapter: Adaptive Pruning and Tuning Adapter
    Extends LoRA with dynamic pruning masks and adaptive rank adjustment
    """
    
    def __init__(self, input_dim: int, output_dim: int, rank: int = 8, 
                 dropout: float = 0.0, scaling: float = 1.0, 
                 use_pruning: bool = True):
        super(APTAdapter, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.scaling = scaling
        self.use_pruning = use_pruning
        
        # Initialize low-rank matrices (LoRA style)
        self.W_A = nn.Parameter(torch.zeros(rank, input_dim))
        self.W_B = nn.Parameter(torch.zeros(output_dim, rank))
        
        # Initialize pruning masks
        self.input_mask = None
        self.output_mask = None
        
        if use_pruning:
            # Initialize masks to all ones (no pruning initially)
            self.input_mask = nn.Parameter(torch.ones(input_dim), requires_grad=False)
            self.output_mask = nn.Parameter(torch.ones(output_dim), requires_grad=False)
        
        # Initialize weights (similar to LoRA)
        nn.init.kaiming_uniform_(self.W_A, a=np.sqrt(5))
        nn.init.zeros_(self.W_B)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, x: torch.Tensor, frozen_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with adaptive pruning and tuning
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            frozen_weight: Original frozen weights (if available)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # Apply input pruning mask if enabled
        if self.use_pruning and self.input_mask is not None:
            x_masked = x * self.input_mask.unsqueeze(0).unsqueeze(0)
        else:
            x_masked = x
            
        # Compute low-rank adaptation
        adapter_output = torch.matmul(x_masked, self.W_A.T)  # (batch, seq, rank)
        adapter_output = torch.matmul(adapter_output, self.W_B.T)  # (batch, seq, output_dim)
        
        # Apply output pruning mask if enabled
        if self.use_pruning and self.output_mask is not None:
            adapter_output = adapter_output * self.output_mask.unsqueeze(0).unsqueeze(0)
            
        # Apply scaling
        adapter_output = adapter_output * self.scaling
        
        # Apply dropout if enabled
        if self.dropout is not None:
            adapter_output = self.dropout(adapter_output)
            
        # Add to frozen weights if provided
        if frozen_weight is not None:
            return frozen_weight + adapter_output
        else:
            return adapter_output
            
    def update_rank(self, new_rank: int):
        """
        Dynamically update the rank of the adapter
        """
        if new_rank == self.rank:
            return
            
        # Create new weight matrices with expanded dimensions
        new_W_A = torch.zeros(new_rank, self.input_dim)
        new_W_B = torch.zeros(self.output_dim, new_rank)
        
        # Copy existing weights
        new_W_A[:self.rank, :] = self.W_A.data
        new_W_B[:, :self.rank] = self.W_B.data
        
        # Initialize new weights with small random values (LoRA initialization)
        nn.init.kaiming_uniform_(new_W_A[self.rank:, :], a=np.sqrt(5))
        nn.init.zeros_(new_W_B[:, self.rank:])
        
        # Update parameters
        self.W_A = nn.Parameter(new_W_A)
        self.W_B = nn.Parameter(new_W_B)
        self.rank = new_rank
        
    def get_pruning_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current pruning masks"""
        return self.input_mask, self.output_mask
        
    def set_pruning_masks(self, input_mask: torch.Tensor, output_mask: torch.Tensor):
        """Set pruning masks"""
        if self.use_pruning:
            self.input_mask.data = input_mask.clone().to(self.input_mask.device)
            self.output_mask.data = output_mask.clone().to(self.output_mask.device)
            
    def get_active_parameters(self) -> int:
        """Count active parameters (considering pruning masks)"""
        active_input = self.input_mask.sum().item() if self.use_pruning and self.input_mask is not None else self.input_dim
        active_output = self.output_mask.sum().item() if self.use_pruning and self.output_mask is not None else self.output_dim
        
        # Active parameters in W_A and W_B
        active_params = active_input * self.rank + active_output * self.rank
        
        return active_params
        
    def get_total_parameters(self) -> int:
        """Get total parameters (ignoring pruning)"""
        return self.input_dim * self.rank + self.output_dim * self.rank
        
    def prune_by_salience(self, input_salience: torch.Tensor, output_salience: torch.Tensor, 
                         target_sparsity: float, salience_threshold: float = 0.1):
        """
        Prune parameters based on salience scores
        """
        if not self.use_pruning:
            return
            
        # Normalize salience scores
        input_salience = input_salience / (input_salience.max() + 1e-8)
        output_salience = output_salience / (output_salience.max() + 1e-8)
        
        # Calculate number of parameters to prune
        input_to_prune = int(self.input_dim * target_sparsity)
        output_to_prune = int(self.output_dim * target_sparsity)
        
        # Get indices of least salient parameters
        input_indices = torch.argsort(input_salience)[:input_to_prune]
        output_indices = torch.argsort(output_salience)[:output_to_prune]
        
        # Create new masks
        new_input_mask = torch.ones_like(self.input_mask)
        new_output_mask = torch.ones_like(self.output_mask)
        
        # Set least salient parameters to 0
        new_input_mask[input_indices] = 0.0
        new_output_mask[output_indices] = 0.0
        
        # Apply gradual pruning (as described in paper)
        alpha = 0.01
        self.input_mask.data = torch.max(self.input_mask.data - alpha, new_input_mask)
        self.output_mask.data = torch.max(self.output_mask.data - alpha, new_output_mask)
        
        # Ensure masks are binary after pruning
        self.input_mask.data = (self.input_mask.data > salience_threshold).float()
        self.output_mask.data = (self.output_mask.data > salience_threshold).float()
        
    def get_salience_scores(self, gradients: torch.Tensor, activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate outlier-aware salience scores for pruning decisions
        """
        # Calculate salience as |W * dL/dW|, but since we don't have direct access to W
        # we use activation * gradient as proxy (as described in paper)
        
        # For input dimension salience
        input_salience = torch.sum(activations * gradients, dim=0)
        
        # For output dimension salience
        # We need to compute gradient w.r.t. output
        output_salience = torch.sum(activations * gradients, dim=1)
        
        # Add kurtosis component for outlier-aware scoring
        # This is a simplified version - in practice we'd compute kurtosis of activations
        input_kurtosis = self._compute_kurtosis(activations)
        output_kurtosis = self._compute_kurtosis(activations)
        
        # Combine salience with kurtosis
        input_salience = input_salience + input_kurtosis
        output_salience = output_salience + output_kurtosis
        
        return input_salience, output_salience
        
    def _compute_kurtosis(self, x: torch.Tensor) -> float:
        """
        Compute kurtosis of activations (simplified version)
        """
        # Simplified kurtosis calculation
        mean = torch.mean(x)
        std = torch.std(x) + 1e-8
        if std == 0:
            return 0.0
            
        # Fourth moment
        fourth_moment = torch.mean(((x - mean) / std) ** 4)
        # Kurtosis = fourth_moment - 3 (excess kurtosis)
        kurtosis = fourth_moment - 3.0
        
        return float(kurtosis)