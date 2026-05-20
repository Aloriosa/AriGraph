#!/usr/bin/env python3
"""
Pruning scheduler for APT - implements adaptive pruning logic
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class PruningScheduler:
    """
    Adaptive pruning scheduler for APT
    Implements the latency-saliency knapsack algorithm from Section 4.2
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 target_sparsity: float = 0.6,
                 initial_sparsity: float = 0.0,
                 total_steps: int = 1000,
                 schedule_type: str = 'cubic'):
        """
        Initialize pruning scheduler
        
        Args:
            model: The model to be pruned
            target_sparsity: Target sparsity level (0.0 to 1.0)
            initial_sparsity: Initial sparsity level
            total_steps: Total training steps
            schedule_type: Type of sparsity schedule ('linear', 'cubic', 'exponential')
        """
        self.model = model
        self.target_sparsity = target_sparsity
        self.initial_sparsity = initial_sparsity
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.current_sparsity = initial_sparsity
        self.step = 0
        
        # Store block information for pruning
        self.block_info = self._extract_block_info()
        
        # Store current pruning masks
        self.pruning_masks = {}
        
    def _extract_block_info(self) -> Dict:
        """
        Extract information about prunable blocks in the model
        """
        block_info = {}
        
        # For RoBERTa/T5/LLaMA models
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                # Identify attention layers and FFN layers
                if 'attention' in name.lower() or 'self_attn' in name.lower():
                    # Attention layers have 4 weight matrices: q, k, v, o
                    # We'll consider heads as blocks
                    if hasattr(module, 'num_attention_heads'):
                        num_heads = module.num_attention_heads
                        head_dim = module.head_dim if hasattr(module, 'head_dim') else module.hidden_size // num_heads
                        block_info[f"{name}.head"] = {
                            'type': 'head',
                            'size': num_heads,
                            'params_per_block': module.hidden_size * head_dim * 4  # Approximate
                        }
                elif 'feedforward' in name.lower() or 'ffn' in name.lower():
                    # FFN layers have up, gate, down projections
                    if hasattr(module, 'intermediate_size'):
                        num_neurons = module.intermediate_size
                        block_info[f"{name}.neuron"] = {
                            'type': 'neuron',
                            'size': num_neurons,
                            'params_per_block': module.hidden_size * 2  # Approximate
                        }
                elif 'embeddings' in name.lower():
                    # Embedding layers
                    if hasattr(module, 'weight'):
                        vocab_size = module.weight.shape[0]
                        hidden_size = module.weight.shape[1]
                        block_info[f"{name}.dimension"] = {
                            'type': 'dimension',
                            'size': hidden_size,
                            'params_per_block': vocab_size
                        }
                        
        return block_info
        
    def get_sparsity_at_step(self, step: int) -> float:
        """
        Get target sparsity at given step using scheduling function
        Implements cubic scheduling as described in paper
        """
        if self.schedule_type == 'cubic':
            # γ_t = γ_T + (1 - γ_T) * (1 - t/T)^3
            ratio = step / self.total_steps
            return self.target_sparsity + (1 - self.target_sparsity) * (1 - ratio) ** 3
        elif self.schedule_type == 'linear':
            # Linear increase
            return self.initial_sparsity + (self.target_sparsity - self.initial_sparsity) * (step / self.total_steps)
        else:  # exponential
            # Exponential increase
            return self.initial_sparsity + (self.target_sparsity - self.initial_sparsity) * (1 - np.exp(-5 * step / self.total_steps))
            
    def update(self, step: int, salience_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Update pruning masks based on salience scores and current step
        Implements the latency-saliency knapsack algorithm
        """
        self.step = step
        self.current_sparsity = self.get_sparsity_at_step(step)
        
        # Get current pruning masks
        new_masks = {}
        
        # Sort blocks by salience density
        salience_density = {}
        for block_name, salience in salience_scores.items():
            if block_name in self.block_info:
                block_size = self.block_info[block_name]['size']
                params_per_block = self.block_info[block_name]['params_per_block']
                density = salience / params_per_block if params_per_block > 0 else salience
                salience_density[block_name] = density
                
        # Sort blocks by density (ascending - least salient first)
        sorted_blocks = sorted(salience_density.items(), key=lambda x: x[1].mean().item())
        
        # Binary search to find optimal number of blocks to retain
        total_blocks = len(sorted_blocks)
        left, right = 0, total_blocks
        best_retain = 0
        
        while left <= right:
            mid = (left + right) // 2
            retained_blocks = sorted_blocks[mid:]
            
            # Calculate current sparsity
            total_params = sum(self.block_info[b[0]]['params_per_block'] * self.block_info[b[0]]['size'] 
                             for b in sorted_blocks)
            retained_params = sum(self.block_info[b[0]]['params_per_block'] * self.block_info[b[0]]['size'] 
                                for b in retained_blocks)
            
            current_sparsity = 1 - (retained_params / total_params) if total_params > 0 else 0
            
            if current_sparsity <= self.current_sparsity:
                best_retain = mid
                left = mid + 1
            else:
                right = mid - 1
                
        # Set pruning masks
        retained_blocks = sorted_blocks[best_retain:]
        retained_block_names = set([b[0] for b in retained_blocks])
        
        for block_name in self.block_info:
            if block_name in retained_block_names:
                # Keep this block
                new_masks[block_name] = torch.ones(self.block_info[block_name]['size'])
            else:
                # Prune this block
                new_masks[block_name] = torch.zeros(self.block_info[block_name]['size'])
                
        self.pruning_masks = new_masks
        return new_masks
        
    def get_current_sparsity(self) -> float:
        """Get current sparsity level"""
        return self.current_sparsity
        
    def get_pruning_masks(self) -> Dict[str, torch.Tensor]:
        """Get current pruning masks"""
        return self.pruning_masks