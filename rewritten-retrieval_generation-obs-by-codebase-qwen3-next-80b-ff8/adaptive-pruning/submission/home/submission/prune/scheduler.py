import torch
import torch.nn as nn
from typing import List, Dict, Optional
import logging
import math

logger = logging.getLogger(__name__)

class BasePruningScheduler:
    def __init__(self, model, head_mask, intermediate_mask, head_grads, intermediate_grads):
        self.model = model
        self.head_mask = head_mask
        self.intermediate_mask = intermediate_mask
        self.head_grads = head_grads
        self.intermediate_grads = intermediate_grads
        self.current_head_score = None
        self.current_intermediate_score = None
        
    def gen_priority(self, num_steps: int = 5):
        raise NotImplementedError
        
    def gen_schedule(self, num_steps: int = 5):
        raise NotImplementedError
        
    def calculate_saliency(self):
        raise NotImplementedError
        
    def gen_next_mask(self):
        raise NotImplementedError

class SaliencyPruningScheduler(BasePruningScheduler):
    def __init__(self, model, head_mask, intermediate_mask, head_grads, intermediate_grads, dataloader, mac_constraints, seq_len: int = 128):
        super().__init__(model, head_mask, intermediate_mask, head_grads, intermediate_grads)
        self.dataloader = dataloader
        self.current_head_score = None
        self.current_intermediate_score = None
        self.mac_constraints = mac_constraints
        self.seq_len = seq_len
        self.masks_schedule = None
        
    def gen_priority(self, num_steps: int = 5):
        return super().gen_priority()
    
    def gen_schedule(self, num_steps: int = 5):
        self.masks_schedule = self.mac_constraints
        
    def calculate_saliency(self):
        self.model.reset_masks()
        head_grads, intermediate_grads = collect_mask_grads(self.model, self.dataloader)
        self.current_head_score = compute_fisher_info(head_grads)
        self.current_intermediate_score = compute_fisher_info(intermediate_grads)
        
    def gen_next_mask(self):
        mac_constraint = self.masks_schedule.pop(0)
        assert 0 < mac_constraint < 1
        if self.current_head_score is None or self.current_intermediate_score is None:
            self.calculate_saliency()
            
        # Reversed search, compared with the original mask-tuning paper
        gen_head_mask, gen_intermediate_mask = search_mac_reverse(
            self.model.config,
            self.current_head_score,
            self.current_intermediate_score,
            self.seq_len,
            mac_constraint,
            head_mask_condition=self.current_head_mask,
            neuron_mask_condition=self.current_intermediate_mask,
        )
        
        self.current_head_mask = [
            current_mask[gen_mask.nonzero().squeeze()].view(-1).contiguous().clone() if current_mask is not None and current_mask.size() else torch.tensor([]).to(self.model.device)
            for gen_mask, current_mask in zip(gen_head_mask, self.current_head_mask)
        ] # Using .view(-1) to convert potential 0D tensor to 1D tensor
        self.current_intermediate_mask = [
            current_mask[gen_mask.nonzero().squeeze()].view(-1).contiguous().clone() if current_mask is not None and current_mask.size() else torch.tensor([]).to(self.model.device)
            for gen_mask, current_mask in zip(gen_intermediate_mask, self.current_intermediate_mask)
        ] # Using .view(-1) to convert potential 0D tensor to 1D tensor
        
        self.current_head_score = None
        self.current_intermediate_score = None
        return {
            'head_mask': gen_head_mask,
            'intermediate_mask': gen_intermediate_mask,
        }
    
    def gen_next_z(self):
        pass

# Helper functions (imported from other files)
def collect_mask_grads(model, dataloader):
    """Collect gradients for mask computation"""
    model.zero_grad()
    device = next(model.parameters()).device
    
    head_grads = []
    intermediate_grads = []
    
    # Sample a batch
    batch = next(iter(dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    
    # Collect gradients for attention heads
    for name, param in model.named_parameters():
        if 'self_attn.q_proj' in name or 'self_attn.k_proj' in name or 'self_attn.v_proj' in name:
            if param.grad is not None:
                head_grads.append(param.grad.detach().clone())
        elif 'mlp.gate_proj' in name or 'mlp.up_proj' in name:
            if param.grad is not None:
                intermediate_grads.append(param.grad.detach().clone())
    
    return head_grads, intermediate_grads

def compute_fisher_info(grads):
    """Compute Fisher information as a proxy for salience"""
    if len(grads) == 0:
        return []
    
    # Compute Fisher information as gradient variance
    fisher_info = []
    for grad in grads:
        # Use gradient magnitude as salience score
        if grad.dim() == 2:
            # For linear layers
            fisher = (grad ** 2).mean(dim=1)
        elif grad.dim() == 1:
            # For bias terms
            fisher = grad ** 2
        else:
            # For other dimensions
            fisher = (grad ** 2).mean()
        fisher_info.append(fisher)
    
    return fisher_info

def search_mac_reverse(config, head_scores, intermediate_scores, seq_len, mac_constraint, head_mask_condition=None, neuron_mask_condition=None):
    """Search for optimal mask that meets MAC constraint"""
    # This is a simplified version - actual implementation would be more complex
    # We'll use a greedy approach to find the optimal mask
    
    # Initialize masks
    head_mask = []
    intermediate_mask = []
    
    # Process attention heads
    for layer in range(config.num_hidden_layers):
        if head_scores[layer] is None:
            head_mask.append(None)
            continue
            
        # Sort scores in ascending order (lowest salience first)
        sorted_scores, indices = torch.sort(head_scores[layer], descending=False)
        
        # Start with all heads kept
        current_mask = torch.ones_like(head_scores[layer])
        
        # Prune heads until MAC constraint is met
        remaining_constraint = mac_constraint
        pruned_count = 0
        total_heads = len(head_scores[layer])
        
        # Prune from least salient heads
        for i in range(total_heads):
            if pruned_count / total_heads >= 1 - remaining_constraint:
                break
            current_mask[indices[i]] = 0
            pruned_count += 1
            
        head_mask.append(current_mask)
    
    # Process intermediate layers
    for layer in range(config.num_hidden_layers):
        if intermediate_scores[layer] is None:
            intermediate_mask.append(None)
            continue
            
        # Sort scores in ascending order (lowest salience first)
        sorted_scores, indices = torch.sort(intermediate_scores[layer], descending=False)
        
        # Start with all neurons kept
        current_mask = torch.ones_like(intermediate_scores[layer])
        
        # Prune neurons until MAC constraint is met
        remaining_constraint = mac_constraint
        pruned_count = 0
        total_neurons = len(intermediate_scores[layer])
        
        # Prune from least salient neurons
        for i in range(total_neurons):
            if pruned_count / total_neurons >= 1 - remaining_constraint:
                break
            current_mask[indices[i]] = 0
            pruned_count += 1
            
        intermediate_mask.append(current_mask)
    
    return head_mask, intermediate_mask