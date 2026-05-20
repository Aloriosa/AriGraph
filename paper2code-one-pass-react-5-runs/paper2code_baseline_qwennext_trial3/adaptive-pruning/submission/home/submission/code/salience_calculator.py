#!/usr/bin/env python3
"""
Salience calculator for APT - outlier-aware scoring
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import logging

class SalienceCalculator:
    """
    Outlier-aware salience calculator for APT
    Implements the scoring function from Section 4.2 of the paper
    """
    
    def __init__(self, model: torch.nn.Module, use_kurtosis: bool = True):
        self.model = model
        self.use_kurtosis = use_kurtosis
        self.salience_cache = {}
        self.gradient_cache = {}
        
    def compute_block_salience(self, 
                             layer_name: str,
                             activations: torch.Tensor,
                             gradients: torch.Tensor,
                             block_type: str = 'head') -> torch.Tensor:
        """
        Compute outlier-aware salience for a block of parameters
        
        Args:
            layer_name: Name of the layer
            activations: Activations from forward pass
            gradients: Gradients from backward pass
            block_type: Type of block ('head', 'neuron', 'dimension')
            
        Returns:
            Salience scores for each block
        """
        
        # Compute basic salience: |activation * gradient|
        # This corresponds to Equation 5 in the paper
        salience = torch.sum(torch.abs(activations * gradients), dim=0)
        
        if self.use_kurtosis:
            # Compute kurtosis of activations for outlier-aware scoring
            kurtosis = self._compute_kurtosis(activations)
            
            # Add kurtosis component as described in paper
            # In the paper, this is added as (Kurt(O_j,:))^(1/2)
            salience = salience + torch.sqrt(torch.tensor(kurtosis))
            
        return salience
        
    def _compute_kurtosis(self, x: torch.Tensor) -> float:
        """
        Compute kurtosis of a tensor
        Kurtosis measures the "tailedness" of the distribution
        Higher kurtosis = more outliers
        """
        if x.numel() == 0:
            return 0.0
            
        # Convert to numpy for easier computation
        x_np = x.detach().cpu().numpy().flatten()
        
        # Calculate mean and standard deviation
        mean = np.mean(x_np)
        std = np.std(x_np) + 1e-8  # Avoid division by zero
        
        # Calculate fourth moment
        fourth_moment = np.mean(((x_np - mean) / std) ** 4)
        
        # Excess kurtosis (subtract 3 for normal distribution)
        kurtosis = fourth_moment - 3.0
        
        return float(kurtosis)
        
    def compute_layer_salience(self, 
                             layer_name: str,
                             activations: torch.Tensor,
                             gradients: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute salience for different blocks in a layer
        
        Returns a dictionary with salience scores for different block types
        """
        salience_scores = {}
        
        # For attention heads (MHA)
        if 'attention' in layer_name.lower():
            # Assuming shape: (batch, seq, num_heads, head_dim)
            if len(activations.shape) == 4:
                # Average over batch and sequence dimensions
                head_activations = torch.mean(activations, dim=[0, 1])  # (num_heads, head_dim)
                head_gradients = torch.mean(gradients, dim=[0, 1])      # (num_heads, head_dim)
                
                # Compute salience per head
                head_salience = torch.sum(torch.abs(head_activations * head_gradients), dim=1)
                salience_scores['head'] = head_salience
                
                # Compute salience per dimension
                dim_salience = torch.sum(torch.abs(head_activations * head_gradients), dim=0)
                salience_scores['dimension'] = dim_salience
                
        # For FFN layers
        elif 'ffn' in layer_name.lower() or 'feedforward' in layer_name.lower():
            # Assuming shape: (batch, seq, hidden_dim)
            if len(activations.shape) == 3:
                # Average over batch and sequence dimensions
                neuron_activations = torch.mean(activations, dim=[0, 1])  # (hidden_dim,)
                neuron_gradients = torch.mean(gradients, dim=[0, 1])      # (hidden_dim,)
                
                # Compute salience per neuron
                neuron_salience = torch.abs(neuron_activations * neuron_gradients)
                salience_scores['neuron'] = neuron_salience
                
        return salience_scores
        
    def get_global_salience(self, layer_salience_dict: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate salience scores across all layers
        """
        global_salience = {}
        
        for layer_name, layer_salience in layer_salience_dict.items():
            for block_type, scores in layer_salience.items():
                if block_type not in global_salience:
                    global_salience[block_type] = torch.zeros_like(scores)
                global_salience[block_type] += scores
                
        return global_salience
        
    def compute_salience_density(self, global_salience: Dict[str, torch.Tensor], 
                                block_sizes: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """
        Compute salience density: salience per parameter
        This is used for the latency-saliency knapsack problem
        """
        salience_density = {}
        
        for block_type, salience_scores in global_salience.items():
            if block_type in block_sizes:
                # Density = salience / number of parameters in block
                block_size = block_sizes[block_type]
                if block_size > 0:
                    salience_density[block_type] = salience_scores / block_size
                else:
                    salience_density[block_type] = salience_scores
            else:
                salience_density[block_type] = salience_scores
                
        return salience_density