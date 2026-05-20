"""
Tokenizer for Simulation-Based Inference (SBI)
"""
import torch
import numpy as np
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SBITokenizer:
    """
    Tokenizer for SBI tasks that represents each variable as:
    - Identifier: Unique identifier for the variable
    - Value: Value of the variable
    - Condition state: Binary state indicating if variable is conditioned on
    """
    
    def __init__(self, 
                 max_vars: int = 20,
                 id_dim: int = 10,
                 value_dim: int = 10,
                 use_finite: bool = True):
        """
        Initialize tokenizer.
        
        Args:
            max_vars: Maximum number of variables (parameters + data)
            id_dim: Dimension of identifier embedding
            value_dim: Dimension of value embedding
            use_finite: Use finite-dimensional representation
        """
        self.max_vars = max_vars
        self.id_dim = id_dim
        self.value_dim = value_dim
        self.use_finite = use_finite
        
        # Learnable identifier embeddings
        self.id_embeddings = torch.nn.Parameter(torch.randn(max_vars, id_dim))
        
        # Learnable condition state embeddings
        self.condition_embeddings = torch.nn.Parameter(torch.randn(2, value_dim))
        
        # Fourier embeddings for function-valued parameters
        self.fourier_basis = self._create_fourier_basis()
        
    def _create_fourier_basis(self) -> torch.Tensor:
        """Create Fourier basis for function-valued parameters."""
        # Use 10 Fourier basis functions for function representation
        n_basis = 10
        basis = torch.zeros(self.max_vars, n_basis)
        for i in range(self.max_vars):
            for j in range(n_basis):
                basis[i, j] = np.sin((i + 1) * (j + 1) * np.pi / (self.max_vars + 1))
        return basis
    
    def tokenize(self, 
                 parameters: torch.Tensor, 
                 data: torch.Tensor, 
                 condition_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize parameters and data.
        
        Args:
            parameters: Parameters tensor of shape (batch_size, n_params)
            data: Data tensor of shape (batch_size, n_data)
            condition_mask: Condition mask tensor of shape (batch_size, n_vars)
        Returns:
            tokens: Token tensor of shape (batch_size, n_vars, 3)
            mask: Attention mask tensor of shape (batch_size, n_vars, n_vars)
        """
        batch_size = parameters.size(0)
        n_params = parameters.size(1)
        n_data = data.size(1)
        n_vars = n_params + n_data
        
        # Create tokens: [identifier, value, condition_state]
        tokens = torch.zeros(batch_size, n_vars, 3)
        
        # Set identifiers
        for i in range(n_vars):
            tokens[:, i, 0] = i  # Identifier (0 to n_vars-1)
        
        # Set values
        # First n_params variables are parameters
        tokens[:, :n_params, 1] = parameters
        # Next n_data variables are data
        tokens[:, n_params:, 1] = data
        
        # Set condition states
        tokens[:, :, 2] = condition_mask
        
        # Apply embeddings
        # Convert identifiers to embeddings
        id_embed = torch.nn.functional.embedding(tokens[:, :, 0].long(), self.id_embeddings)
        value_embed = tokens[:, :, 1].unsqueeze(-1)
        condition_embed = torch.nn.functional.embedding(tokens[:, :, 2].long(), self.condition_embeddings)
        
        # Combine embeddings
        token_embed = torch.cat([id_embed, value_embed, condition_embed], dim=2)
        
        # Create attention mask
        # For now, use a simple mask based on condition states
        mask = torch.zeros(batch_size, n_vars, n_vars)
        for i in range(n_vars):
            for j in range(n_vars):
                # If both are conditioned, allow attention
                if condition_mask[i] and condition_mask[j]:
                    mask[:, i, j] = 1
                # If one is conditioned and the other is not, allow attention
                elif condition_mask[i] or condition_mask[j]:
                    mask[:, i, j] = 1
                # If both are not conditioned, allow attention
                else:
                    mask[:, i, j] = 1
        
        return tokens, mask
    
    def tokenize_function(self, 
                         parameters: torch.Tensor, 
                         data: torch.Tensor, 
                         condition_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize function-valued parameters and data.
        
        Args:
            parameters: Parameters tensor of shape (batch_size, n_params)
            data: Data tensor of shape (batch_size, n_data)
            condition_mask: Condition mask tensor of shape (batch, n_vars)
        Returns:
            tokens: Token tensor of shape (batch_size, n_vars, 3)
            mask: Attention mask tensor of shape (batch_size, n_vars, n_vars)
        """
        batch_size = parameters.size(0)
        n_params = parameters.size(1)
        n_data = data.size(1)
        n_vars = n_params + n_data
        
        # Create tokens: [identifier, value, condition_state]
        tokens = torch.zeros(batch_size, n_vars, 3)
        
        # Set identifiers
        for i in range(n_vars):
            tokens[:, i, 0] = i  # Identifier (0 to n_vars-1)
        
        # Set values
        # First n_params variables are parameters
        tokens[:, :n_params, 1] = parameters
        # Next n_data variables are data
        tokens[:, n_params:, 1] = data
        
        # Set condition states
        tokens[:, :, 2] = condition_mask
        
        # Apply embeddings
        # Convert identifiers to embeddings
        id_embed = torch.nn.functional.embedding(tokens[:, :, 0].long(), self.id_embeddings)
        value_embed = tokens[:, :, 1].unsqueeze(-1)
        condition_embed = torch.nn.functional.embedding(tokens[:, :, 2].long(), self.condition_embeddings)
        
        # Combine embeddings
        token_embed = torch.cat([id_embed, value_embed, condition_embed], dim=2)
        
        # Create attention mask
        # For function-valued parameters, use a mask based on time dependencies
        mask = torch.zeros(batch_size, n_vars, n_vars)
        for i in range(n_vars):
            for j in range(n_vars):
                # For function-valued parameters, allow attention based on time
                # This is a simple example - in practice, this would be based on the simulator structure
                if i == j:
                    mask[:, i, j] = 1
                elif abs(i - j) <= 1:
                    mask[:, i, j] = 1
        
        return tokens, mask