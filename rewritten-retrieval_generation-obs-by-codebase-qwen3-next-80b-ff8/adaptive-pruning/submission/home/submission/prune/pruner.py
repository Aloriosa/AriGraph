import torch
import torch.nn as nn
from transformers import PreTrainedModel
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class BasePruner:
    def __init__(self, model: PreTrainedModel, mask_required: List[str]):
        self.model = model
        self.mask_required = mask_required
        self.mask_to_shape = T5_MASK_TO_SHAPE if model.base_model_prefix == 'transformer' else MASK_TO_SHAPE
        self.mask_shapes = {
            key: self.mask_to_shape[key](model)
            for key in mask_required
        }
    
    def generate_mask(self):
        raise NotImplementedError

class AdapterPruner(BasePruner):
    def __init__(self, model: PreTrainedModel, dataloader, mask_required=None):
        if mask_required is None:
            mask_required = ['head_mask', 'intermediate_mask']
        super().__init__(model, mask_required)
        self.dataloader = dataloader
        self.scorer = None
        
    def generate_mask(self, sparsity_level=0.7, step=0):
        """Generate pruning mask based on salience scoring"""
        if self.scorer is None:
            self.scorer = SalienceScorer(self.model, self.dataloader)
        
        # Calculate salience scores
        head_scores, intermediate_scores = self.scorer.calculate_saliency()
        
        # Generate masks based on salience scores
        head_mask = self._generate_head_mask(head_scores, sparsity_level)
        intermediate_mask = self._generate_intermediate_mask(intermediate_scores, sparsity_level)
        
        return {
            'head_mask': head_mask,
            'intermediate_mask': intermediate_mask
        }
    
    def _generate_head_mask(self, scores, sparsity_level):
        """Generate head pruning mask"""
        head_mask = []
        total_heads = 0
        for layer_scores in scores:
            if layer_scores is None:
                head_mask.append(None)
                continue
                
            # Sort scores in ascending order (lowest salience first)
            sorted_scores, indices = torch.sort(layer_scores, descending=False)
            
            # Calculate number of heads to prune
            n_heads = len(layer_scores)
            n_prune = int(n_heads * sparsity_level)
            
            # Create mask (1 = keep, 0 = prune)
            mask = torch.ones(n_heads, device=layer_scores.device)
            if n_prune > 0:
                mask[indices[:n_prune]] = 0
                
            head_mask.append(mask)
            total_heads += n_heads
            
        return head_mask
    
    def _generate_intermediate_mask(self, scores, sparsity_level):
        """Generate intermediate pruning mask"""
        intermediate_mask = []
        total_neurons = 0
        for layer_scores in scores:
            if layer_scores is None:
                intermediate_mask.append(None)
                continue
                
            # Sort scores in ascending order (lowest salience first)
            sorted_scores, indices = torch.sort(layer_scores, descending=False)
            
            # Calculate number of neurons to prune
            n_neurons = len(layer_scores)
            n_prune = int(n_neurons * sparsity_level)
            
            # Create mask (1 = keep, 0 = prune)
            mask = torch.ones(n_neurons, device=layer_scores.device)
            if n_prune > 0:
                mask[indices[:n_prune]] = 0
                
            intermediate_mask.append(mask)
            total_neurons += n_neurons
            
        return intermediate_mask

class SalienceScorer:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.device = next(model.parameters()).device
        
    def calculate_saliency(self):
        """Calculate salience scores using gradient-based method"""
        # Reset gradients
        self.model.zero_grad()
        
        # Collect gradients for all parameters
        head_grads = []
        intermediate_grads = []
        
        # Sample a batch
        batch = next(iter(self.dataloader))
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss
        loss.backward()
        
        # Extract gradients for attention heads and intermediate layers
        for name, param in self.model.named_parameters():
            if 'self_attn.q_proj' in name or 'self_attn.k_proj' in name or 'self_attn.v_proj' in name:
                # For attention heads
                if param.grad is not None:
                    # Calculate salience score as gradient * parameter
                    salience = (param.grad * param).abs().mean().item()
                    head_grads.append(salience)
                    
            elif 'mlp.gate_proj' in name or 'mlp.up_proj' in name:
                # For intermediate layers
                if param.grad is not None:
                    # Calculate salience score as gradient * parameter
                    salience = (param.grad * param).abs().mean().item()
                    intermediate_grads.append(salience)
        
        # Group by layer
        head_scores = self._group_by_layer(head_grads, 'head')
        intermediate_scores = self._group_by_layer(intermediate_grads, 'intermediate')
        
        return head_scores, intermediate_scores
    
    def _group_by_layer(self, scores, type_):
        """Group scores by layer"""
        if type_ == 'head':
            num_layers = self.model.config.num_hidden_layers
            heads_per_layer = self.model.config.num_attention_heads
            grouped_scores = []
            for i in range(num_layers):
                start_idx = i * heads_per_layer
                end_idx = start_idx + heads_per_layer
                if end_idx <= len(scores):
                    layer_scores = torch.tensor(scores[start_idx:end_idx])
                else:
                    layer_scores = torch.tensor(scores[start_idx:])
                grouped_scores.append(layer_scores)
            return grouped_scores
        else:  # intermediate
            num_layers = self.model.config.num_hidden_layers
            intermediate_size = self.model.config.intermediate_size
            grouped_scores = []
            for i in range(num_layers):
                start_idx = i * intermediate_size
                end_idx = start_idx + intermediate_size
                if end_idx <= len(scores):
                    layer_scores = torch.tensor(scores[start_idx:end_idx])
                else:
                    layer_scores = torch.tensor(scores[start_idx:])
                grouped_scores.append(layer_scores)
            return grouped_scores

# Mask to shape mapping
MASK_TO_SHAPE = {
    'head_mask': lambda model: (model.config.num_hidden_layers, model.config.num_attention_heads),
    'intermediate_mask': lambda model: (model.config.num_hidden_layers, model.config.intermediate_size),
    'hidden_mask': lambda model: (model.config.hidden_size,),
}

T5_MASK_TO_SHAPE = {
    'head_mask': lambda model: (model.config.num_hidden_layers, model.config.num_attention_heads),
    'intermediate_mask': lambda model: (model.config.num_hidden_layers, model.config.d_ff),
    'hidden_mask': lambda model: (model.config.d_model,),
}