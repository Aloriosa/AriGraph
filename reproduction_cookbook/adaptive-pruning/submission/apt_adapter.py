import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from typing import Optional, Dict, Any
import math

class APTAdapter(nn.Module):
    """
    APT Adapter implementation that combines adaptive pruning and tuning.
    Based on LoRA but with dynamic pruning masks and adaptive rank adjustment.
    """
    
    def __init__(self, in_features: int, out_features: int, r: int = 8, 
                 alpha: float = 16.0, dropout: float = 0.0, 
                 pruning_ratio: float = 0.0, device=None, dtype=None):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(r, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))
        
        # Initialize pruning masks (binary masks for pruning)
        self.prune_mask_A = torch.ones(r, in_features, device=device, dtype=torch.bool)
        self.prune_mask_B = torch.ones(out_features, r, device=device, dtype=torch.bool)
        
        # Initialize salience scores for adaptive pruning
        self.salience_A = torch.zeros(r, in_features, device=device, dtype=torch.float32)
        self.salience_B = torch.zeros(out_features, r, device=device, dtype=torch.float32)
        
        # Initialize dynamic rank tracking
        self.current_rank = r
        self.max_rank = r
        self.rank_increment = 1
        
        # Initialize pruning parameters
        self.pruning_ratio = pruning_ratio
        self.pruning_step = 0
        self.pruning_schedule = "cubic"  # cubic scheduling as in paper
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters as in LoRA"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive pruning and tuning"""
        # Apply pruning masks to LoRA parameters
        masked_lora_A = self.lora_A * self.prune_mask_A.float()
        masked_lora_B = self.lora_B * self.prune_mask_B.float()
        
        # Apply LoRA transformation
        result = x @ masked_lora_A.T @ masked_lora_B.T * self.scaling
        result = self.dropout(result)
        
        return result
    
    def update_pruning_masks(self, salience_scores_A: torch.Tensor, 
                           salience_scores_B: torch.Tensor, 
                           target_sparsity: float):
        """
        Update pruning masks based on salience scores using cubic scheduling.
        This implements the adaptive pruning mechanism from the paper.
        """
        # Update salience scores with exponential moving average
        self.salience_A = 0.9 * self.salience_A + 0.1 * salience_scores_A
        self.salience_B = 0.9 * self.salience_B + 0.1 * salience_scores_B
        
        # Apply cubic scheduling for sparsity growth
        # t = _t + (1 - _t)(1 - t/t)^3 as in paper
        # We'll use a simple cubic function for demonstration
        if self.pruning_step > 0:
            progress = min(self.pruning_step / 1000, 1.0)  # Normalize to 0-1
            current_sparsity = target_sparsity * (1 - (1 - progress) ** 3)
        else:
            current_sparsity = 0.0
            
        # Update pruning masks based on salience scores
        # Keep top (1 - current_sparsity) fraction of parameters
        self._update_mask_with_salience(self.salience_A, self.prune_mask_A, current_sparsity)
        self._update_mask_with_salience(self.salience_B, self.prune_mask_B, current_sparsity)
        
        # Update current rank based on remaining parameters
        self.current_rank = min(self.max_rank, int(self.prune_mask_A.sum().item() / self.in_features))
        
        self.pruning_step += 1
        
    def _update_mask_with_salience(self, salience_scores: torch.Tensor, 
                                 mask: torch.Tensor, target_sparsity: float):
        """Update mask based on salience scores using binary search for target sparsity"""
        if target_sparsity >= 1.0:
            mask.fill_(False)
            return
            
        # Flatten salience scores
        flat_scores = salience_scores.flatten()
        n = flat_scores.numel()
        
        # Find threshold for target sparsity
        if n == 0:
            return
            
        # Sort scores in descending order (higher salience = more important)
        sorted_scores, indices = torch.sort(flat_scores, descending=True)
        
        # Find the threshold index for target sparsity
        keep_count = int(n * (1 - target_sparsity))
        keep_count = max(1, keep_count)  # Ensure at least one parameter is kept
        
        # Set threshold
        threshold = sorted_scores[keep_count - 1] if keep_count > 0 else float('-inf')
        
        # Update mask
        mask_flat = (salience_scores >= threshold)
        mask.copy_(mask_flat.view_as(mask))
        
    def update_tuning_ranks(self, salience_scores: torch.Tensor, 
                          target_rank: int, layer_importance: float):
        """
        Dynamically adjust tuning ranks based on layer importance.
        This implements the adaptive tuning mechanism from the paper.
        """
        # Calculate layer importance based on salience
        layer_salience = salience_scores.mean().item()
        
        # Increase rank for salient layers
        if layer_salience > layer_importance:
            # Increase rank by rank_increment
            new_rank = min(self.max_rank, self.current_rank + self.rank_increment)
            if new_rank > self.current_rank:
                self._expand_rank(new_rank)
                self.current_rank = new_rank
                
    def _expand_rank(self, new_rank: int):
        """Expand the rank of the LoRA adapter"""
        if new_rank <= self.current_rank:
            return
            
        # Create new parameters with expanded rank
        new_lora_A = torch.zeros(new_rank, self.in_features, device=self.lora_A.device, dtype=self.lora_A.dtype)
        new_lora_B = torch.zeros(self.out_features, new_rank, device=self.lora_B.device, dtype=self.lora_B.dtype)
        
        # Copy existing parameters
        new_lora_A[:self.current_rank, :] = self.lora_A
        new_lora_B[:, :self.current_rank] = self.lora_B
        
        # Initialize new parameters with small values
        nn.init.kaiming_uniform_(new_lora_A[self.current_rank:, :], a=math.sqrt(5))
        nn.init.zeros_(new_lora_B[:, self.current_rank:])
        
        # Update parameters
        self.lora_A = nn.Parameter(new_lora_A)
        self.lora_B = nn.Parameter(new_lora_B)
        
        # Update pruning masks
        new_prune_mask_A = torch.ones(new_rank, self.in_features, device=self.prune_mask_A.device, dtype=torch.bool)
        new_prune_mask_B = torch.ones(self.out_features, new_rank, device=self.prune_mask_B.device, dtype=torch.bool)
        
        new_prune_mask_A[:self.current_rank, :] = self.prune_mask_A
        new_prune_mask_B[:, :self.current_rank] = self.prune_mask_B
        
        self.prune_mask_A = new_prune_mask_A
        self.prune_mask_B = new_prune_mask_B
        
        # Update salience scores
        new_salience_A = torch.zeros(new_rank, self.in_features, device=self.salience_A.device, dtype=self.salience_A.dtype)
        new_salience_B = torch.zeros(self.out_features, new_rank, device=self.salience_B.device, dtype=self.salience_B.dtype)
        
        new_salience_A[:self.current_rank, :] = self.salience_A
        new_salience_B[:, :self.current_rank] = self.salience_B
        
        self.salience_A = new_salience_A
        self.salience_B = new_salience_B
        
        self.max_rank = new_rank
        self.current_rank = new_rank

class APTTransformerLayer(nn.Module):
    """
    APT-enhanced transformer layer that integrates adaptive pruning and tuning.
    """
    
    def __init__(self, base_layer: nn.Module, r: int = 8, alpha: float = 16.0,
                 pruning_ratio: float = 0.0, device=None, dtype=None):
        super().__init__()
        
        self.base_layer = base_layer
        self.apr = APTAdapter(base_layer.in_features, base_layer.out_features, 
                             r=r, alpha=alpha, pruning_ratio=pruning_ratio,
                             device=device, dtype=dtype)
        
        # Initialize self-distillation components
        self.teacher_layer = None
        self.distillation_weight = 0.0
        self.distillation_temperature = 2.0
        
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass with APT adaptation"""
        # Apply base layer
        base_output = self.base_layer(x, *args, **kwargs)
        
        # Apply APT adapter
        apt_output = self.apr(x)
        
        # Combine outputs
        output = base_output + apt_output
        
        # Apply self-distillation if teacher layer is available
        if self.teacher_layer is not None:
            teacher_output = self.teacher_layer(x, *args, **kwargs)
            # Compute distillation loss (MSE between outputs)
            distillation_loss = F.mse_loss(output, teacher_output)
            # Apply distillation weight (increases over training)
            output = output + self.distillation_weight * (teacher_output - output)
            
        return output
    
    def set_teacher_layer(self, teacher_layer: nn.Module):
        """Set the teacher layer for self-distillation"""
        self.teacher_layer = teacher_layer
        
    def update_distillation_weight(self, step: int, total_steps: int):
        """Update distillation weight using linear schedule"""
        self.distillation_weight = min(step / total_steps, 1.0)
        
    def update_pruning_masks(self, salience_scores_A: torch.Tensor, 
                           salience_scores_B: torch.Tensor, 
                           target_sparsity: float):
        """Update pruning masks for the APT adapter"""
        self.apr.update_pruning_masks(salience_scores_A, salience_scores_B, target_sparsity)
        
    def update_tuning_ranks(self, salience_scores: torch.Tensor, 
                          target_rank: int, layer_importance: float):
        """Update tuning ranks for the APT adapter"""
        self.apr.update_tuning_ranks(salience_scores, target_rank, layer_importance)