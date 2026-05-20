import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

class APTAdapter(nn.Module):
    """
    Adaptive Pruning and Tuning (APT) Adapter implementation.
    Extends LoRA with dynamic pruning masks and adaptive rank tuning.
    """
    
    def __init__(self, input_dim: int, output_dim: int, rank: int = 8, 
                 alpha: float = 2.0, dropout: float = 0.0, 
                 pruning_threshold: float = 0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Initialize low-rank matrices (tuning parameters)
        self.W_A = nn.Parameter(torch.zeros(rank, input_dim))
        self.W_B = nn.Parameter(torch.zeros(output_dim, rank))
        
        # Initialize pruning masks (binary, but initialized as continuous for gradient flow)
        self.mask_input = nn.Parameter(torch.ones(input_dim))
        self.mask_output = nn.Parameter(torch.ones(output_dim))
        
        # Initialize with LoRA initialization
        nn.init.kaiming_uniform_(self.W_A, a=math.sqrt(5))
        nn.init.zeros_(self.W_B)
        
        self.dropout = nn.Dropout(dropout)
        
        # Store current effective rank and mask values
        self.current_rank = rank
        self.pruning_threshold = pruning_threshold
        self.is_pruned = False
        
    def forward(self, x: torch.Tensor, frozen_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with adaptive pruning and tuning.
        x: input tensor of shape (batch_size, seq_len, input_dim)
        frozen_weights: original weights (for residual connection)
        """
        # Apply input pruning mask
        x_masked = x * self.mask_input.unsqueeze(0).unsqueeze(0)
        
        # Compute low-rank adaptation
        # W_A: (rank, input_dim), x: (batch, seq, input_dim)
        # result: (batch, seq, rank)
        low_rank_output = torch.matmul(x_masked, self.W_A.T)
        
        # Apply dropout
        low_rank_output = self.dropout(low_rank_output)
        
        # W_B: (output_dim, rank), low_rank_output: (batch, seq, rank)
        # result: (batch, seq, output_dim)
        adaptation = torch.matmul(low_rank_output, self.W_B.T)
        
        # Apply output pruning mask
        adaptation = adaptation * self.mask_output.unsqueeze(0).unsqueeze(0)
        
        # Scale by alpha/rank
        adaptation = adaptation * self.scaling
        
        # Add to frozen weights if provided
        if frozen_weights is not None:
            return frozen_weights + adaptation
        else:
            return adaptation
    
    def get_pruning_scores(self, gradients: torch.Tensor, activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute outlier-aware salience scores for pruning.
        Uses the formula: S = |activation * gradient| + sqrt(kurtosis)
        """
        # Calculate activation-gradient product (compressed across batch)
        # activations: (batch, seq, dim) -> (dim)
        # gradients: (batch, seq, dim) -> (dim)
        avg_activation = torch.mean(torch.abs(activations), dim=[0, 1])
        avg_gradient = torch.mean(torch.abs(gradients), dim=[0, 1])
        
        # Salience score from paper: |W * dL/dW| but using activations and gradients
        salience = avg_activation * avg_gradient
        
        # Calculate kurtosis of activations for outlier detection
        # For simplicity, we use a simplified kurtosis approximation
        # Kurtosis = E[(X - mu)^4] / (E[(X - mu)^2])^2 - 3
        mean_act = torch.mean(activations, dim=[0, 1])
        centered_act = activations - mean_act.unsqueeze(0).unsqueeze(0)
        var_act = torch.mean(centered_act ** 2, dim=[0, 1])
        kurtosis = torch.mean(centered_act ** 4, dim=[0, 1]) / (var_act ** 2 + 1e-8) - 3
        
        # Combine salience with kurtosis (as in paper)
        outlier_aware_salience = salience + torch.sqrt(torch.relu(kurtosis))
        
        return outlier_aware_salience, salience
    
    def get_adapter_salience(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Calculate salience for the entire adapter (used for adaptive tuning).
        Uses the sum of salience scores of W_B parameters.
        """
        # For simplicity, we use the gradient magnitude of W_B
        # In practice, this would be computed during forward/backward pass
        if gradients is not None:
            return torch.sum(torch.abs(gradients))
        else:
            return torch.sum(torch.abs(self.W_B))
    
    def increase_rank(self, new_rank: int):
        """
        Dynamically increase the rank of the adapter.
        """
        if new_rank <= self.current_rank:
            return
            
        # Create new matrices with expanded dimensions
        new_W_A = torch.zeros(new_rank, self.input_dim)
        new_W_B = torch.zeros(self.output_dim, new_rank)
        
        # Copy existing weights
        new_W_A[:self.current_rank, :] = self.W_A.data
        new_W_B[:, :self.current_rank] = self.W_B.data
        
        # Initialize new weights with Gaussian noise (as in LoRA)
        nn.init.normal_(new_W_A[self.current_rank:, :], std=0.02)
        nn.init.zeros_(new_W_B[:, self.current_rank:])
        
        # Update parameters
        self.W_A = nn.Parameter(new_W_A)
        self.W_B = nn.Parameter(new_W_B)
        self.current_rank = new_rank
        self.rank = new_rank
        self.scaling = self.alpha / self.rank
    
    def apply_pruning(self, input_mask: torch.Tensor, output_mask: torch.Tensor):
        """
        Apply pruning masks to reduce model size.
        """
        self.mask_input.data = input_mask.float()
        self.mask_output.data = output_mask.float()
        self.is_pruned = True
    
    def get_pruned_parameters(self) -> Tuple[int, int]:
        """
        Return number of pruned input and output dimensions.
        """
        input_pruned = torch.sum(self.mask_input < 0.5).item()
        output_pruned = torch.sum(self.mask_output < 0.5).item()
        return int(input_pruned), int(output_pruned)
    
    def get_effective_parameters(self) -> int:
        """
        Calculate number of effective parameters after pruning.
        """
        # Count parameters in W_A and W_B that are not pruned
        input_active = torch.sum(self.mask_input >= 0.5).item()
        output_active = torch.sum(self.mask_output >= 0.5).item()
        
        # Effective parameters = active_input * rank + rank * active_output
        # But we need to account for the fact that we're using low-rank
        return int(input_active * self.current_rank + self.current_rank * output_active)