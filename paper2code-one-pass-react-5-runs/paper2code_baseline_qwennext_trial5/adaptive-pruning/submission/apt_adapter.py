#!/usr/bin/env python3
"""
APT Adapter implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel

class APTAdapter(nn.Module):
    """
    APT Adapter: Adaptive Pruning and Tuning Adapter
    Extends LoRA with dynamic pruning masks and adaptive tuning ranks
    """
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 rank: int = 8, 
                 alpha: float = 1.0,
                 dropout: float = 0.0,
                 device: str = "cuda"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Pruning masks (binary, 1 = keep, 0 = prune)
        self.input_mask = torch.ones(input_dim, device=device)
        self.output_mask = torch.ones(output_dim, device=device)
        
        # LoRA parameters
        self.W_A = nn.Parameter(torch.zeros(rank, input_dim, device=device))
        self.W_B = nn.Parameter(torch.zeros(output_dim, rank, device=device))
        
        # Initialize weights (same as LoRA)
        nn.init.kaiming_uniform_(self.W_A, a=np.sqrt(5))
        nn.init.zeros_(self.W_B)
        
        # Track current rank (can be increased adaptively)
        self.current_rank = rank
        
        # Track pruning status
        self.is_pruned = False
        self.pruned_input = 0
        self.pruned_output = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive pruning and tuning"""
        # Apply input mask
        x_masked = x * self.input_mask.unsqueeze(0)
        
        # Apply LoRA
        result = torch.matmul(x_masked, (self.W_B * self.output_mask.unsqueeze(1)) @ self.W_A.T)
        result = result * self.scaling
        
        # Apply dropout
        result = self.dropout(result)
        
        return result
    
    def update_rank(self, new_rank: int):
        """Dynamically increase the rank of the adapter"""
        if new_rank <= self.current_rank:
            return
            
        # Create new parameter matrices with the increased rank
        new_W_A = torch.zeros(new_rank, self.input_dim, device=self.W_A.device)
        new_W_B = torch.zeros(self.output_dim, new_rank, device=self.W_B.device)
        
        # Copy existing weights
        new_W_A[:self.current_rank, :] = self.W_A.data
        new_W_B[:, :self.current_rank] = self.W_B.data
        
        # Initialize new weights (zeros for W_B, Gaussian for W_A)
        nn.init.normal_(new_W_A[self.current_rank:, :], std=0.01)
        # W_B new weights are already zeros (as per paper)
        
        # Update parameters
        self.W_A = nn.Parameter(new_W_A)
        self.W_B = nn.Parameter(new_W_B)
        self.current_rank = new_rank
        self.scaling = self.alpha / new_rank
        
    def prune(self, input_prune_mask: torch.Tensor, output_prune_mask: torch.Tensor):
        """Prune the adapter by setting masks to 0"""
        self.input_mask = self.input_mask * input_prune_mask
        self.output_mask = self.output_mask * output_prune_mask
        
        # Update pruned counts
        self.pruned_input = (1 - input_prune_mask).sum().item()
        self.pruned_output = (1 - output_prune_mask).sum().item()
        self.is_pruned = True
    
    def get_pruning_info(self) -> Dict:
        """Get information about pruning status"""
        return {
            "input_mask": self.input_mask.clone(),
            "output_mask": self.output_mask.clone(),
            "pruned_input": self.pruned_input,
            "pruned_output": self.pruned_output,
            "current_rank": self.current_rank,
            "original_rank": self.rank
        }
    
    def get_salience_score(self, gradients: torch.Tensor, activations: torch.Tensor) -> float:
        """Calculate salience score for this adapter (as in paper)"""
        # Calculate gradient-activation product for output weights
        # S = |W * ∂L/∂W| for each parameter
        # For W_B: we use the gradient with respect to W_B
        # But since we don't have direct access to gradients of W_B in forward pass,
        # we approximate using activations and output gradients
        
        # Approximate salience as sum of |activation * gradient| for output dimension
        # This is a simplification of the paper's approach
        if gradients is None or activations is None:
            return 0.0
            
        # For W_B, salience is related to output activations and gradients
        # We approximate: salience ≈ sum(|activation * gradient|) over batch
        # where activation is the input to W_B (which is W_A * x)
        # and gradient is the gradient of loss w.r.t. output
        
        # Simplified version: use the magnitude of gradients w.r.t. W_B
        # This is what we would compute in backward pass
        return torch.norm(gradients).item()
    
    def get_layer_salience(self, gradients: torch.Tensor, activations: torch.Tensor) -> float:
        """Calculate salience for the entire adapter layer"""
        # Sum of salience scores for all parameters in W_B
        # As defined in Section 4.3
        if gradients is None:
            return 0.0
            
        # Use the gradients with respect to W_B
        # In practice, we would compute this during backward pass
        # Here we simulate it
        return torch.norm(gradients).item()


class APTModel(nn.Module):
    """
    APT-enabled model wrapper
    """
    def __init__(self, base_model: PreTrainedModel, config: Dict):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.adapters = {}
        self.device = config.get("device", "cuda")
        
        # Initialize adapters for attention and FFN layers
        self._add_adapters()
        
        # Store original parameters for distillation
        self.teacher_model = None
        
    def _add_adapters(self):
        """Add APT adapters to appropriate layers"""
        model_type = self.config["model_type"]
        
        if model_type == "roberta":
            self._add_roberta_adapters()
        elif model_type == "t5":
            self._add_t5_adapters()
        elif model_type == "llama":
            self._add_llama_adapters()
            
    def _add_roberta_adapters(self):
        """Add APT adapters to RoBERTa model"""
        roberta = self.base_model.roberta
        
        # Add adapters to attention layers
        for i, layer in enumerate(roberta.encoder.layer):
            # Attention Q and V projections
            for proj_name in ["query", "value"]:
                proj_layer = getattr(layer.attention.self, proj_name)
                input_dim = proj_layer.in_features
                output_dim = proj_layer.out_features
                
                adapter = APTAdapter(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    rank=self.config.get("initial_rank", 8),
                    alpha=self.config.get("alpha", 1.0),
                    device=self.device
                )
                
                # Store adapter
                adapter_name = f"adapter_{i}_{proj_name}"
                self.adapters[adapter_name] = adapter
                
                # Replace the original layer with a combined layer
                # We'll handle this in forward pass
                setattr(layer.attention.self, f"{proj_name}_adapter", adapter)
        
        # Add adapters to FFN layers
        for i, layer in enumerate(roberta.encoder.layer):
            # FFN layers
            for proj_name in ["intermediate", "output"]:
                if proj_name == "intermediate":
                    proj_layer = layer.intermediate.dense
                    input_dim = proj_layer.in_features
                    output_dim = proj_layer.out_features
                else:
                    proj_layer = layer.output.dense
                    input_dim = proj_layer.in_features
                    output_dim = proj_layer.out_features
                
                adapter = APTAdapter(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    rank=self.config.get("initial_rank", 8),
                    alpha=self.config.get("alpha", 1.0),
                    device=self.device
                )
                
                adapter_name = f"adapter_{i}_ffn_{proj_name}"
                self.adapters[adapter_name] = adapter
                setattr(layer, f"{proj_name}_adapter", adapter)
                
    def _add_t5_adapters(self):
        """Add APT adapters to T5 model"""
        t5 = self.base_model.transformer
        
        # Add adapters to attention layers
        for i, layer in enumerate(t5.encoder.block):
            # Self-attention
            for proj_name in ["q", "v"]:
                proj_layer = layer.layer[0].SelfAttention[proj_name]
                input_dim = proj_layer.in_features
                output_dim = proj_layer.out_features
                
                adapter = APTAdapter(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    rank=self.config.get("initial_rank", 8),
                    alpha=self.config.get("alpha", 1.0),
                    device=self.device
                )
                
                adapter_name = f"encoder_adapter_{i}_{proj_name}"
                self.adapters[adapter_name] = adapter
                setattr(layer.layer[0].SelfAttention, f"{proj_name}_adapter", adapter)
                
        # Add adapters to FFN layers
        for i, layer in enumerate(t5.encoder.block):
            proj_layer = layer.layer[1].DenseReluDense.wo
            input_dim = proj_layer.in_features
            output_dim = proj_layer.out_features
            
            adapter = APTAdapter(
                input_dim=input_dim,
                output_dim=output_dim,
                rank=self.config.get("initial_rank", 8),
                alpha=self.config.get("alpha", 1.0),
                device=self.device
            )
            
            adapter_name = f"encoder_adapter_{i}_ffn"
            self.adapters[adapter_name] = adapter
            setattr(layer.layer[1].DenseReluDense, "wo_adapter", adapter)
            
        # Add adapters to decoder layers
        for i, layer in enumerate(t5.decoder.block):
            # Self-attention
            for proj_name in ["q", "v"]:
                proj_layer = layer.layer[0].SelfAttention[proj_name]
                input_dim = proj_layer.in_features
                output_dim = proj_layer.out_features
                
                adapter = APTAdapter(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    rank=self.config.get("initial_rank", 8),
                    alpha=self.config.get("alpha", 1.0),
                    device=self.device
                )
                
                adapter_name = f"decoder_adapter_{i}_{proj_name}"
                self.adapters[adapter_name] = adapter
                setattr(layer.layer[0].SelfAttention, f"{proj_name}_adapter", adapter)
                
            # Cross-attention
            for proj_name in ["q", "v"]:
                proj_layer = layer.layer[1].EncDecAttention[proj_name]
                input_dim = proj_layer.in_features
                output_dim = proj_layer.out_features
                
                adapter = APTAdapter(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    rank=self.config.get("initial_rank", 8),
                    alpha=self.config.get("alpha", 1.0),
                    device=self.device
                )
                
                adapter_name = f"decoder_adapter_{i}_cross_{proj_name}"
                self.adapters[adapter_name] = adapter
                setattr(layer.layer[1].EncDecAttention, f"{proj_name}_adapter", adapter)
                
            # FFN
            proj_layer = layer.layer[2].DenseReluDense.wo
            input_dim = proj_layer.in_features
            output_dim = proj_layer.out_features
            
            adapter = APTAdapter(
                input_dim=input_dim,
                output_dim=output_dim,
                rank=self.config.get("initial_rank", 8),
                alpha=self.config.get("alpha", 1.0),
                device=self.device
            )
            
            adapter_name = f"decoder_adapter_{i}_ffn"
            self.adapters[adapter_name] = adapter
            setattr(layer.layer[2].DenseReluDense, "wo_adapter", adapter)
            
    def _add_llama_adapters(self):
        """Add APT adapters to LLaMA model"""
        llama = self.base_model.model
        
        # Add adapters to attention layers
        for i, layer in enumerate(llama.layers):
            # Attention Q and V projections
            for proj_name in ["q_proj", "v_proj"]:
                proj_layer = getattr(layer.self_attn, proj_name)
                input_dim = proj_layer.in_features
                output_dim = proj_layer.out_features
                
                adapter = APTAdapter(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    rank=self.config.get("initial_rank", 8),
                    alpha=self.config.get("alpha", 1.0),
                    device=self.device
                )
                
                adapter_name = f"adapter_{i}_{proj_name}"
                self.adapters[adapter_name] = adapter
                setattr(layer.self_attn, f"{proj_name}_adapter", adapter)
                
        # Add adapters to FFN layers
        for i, layer in enumerate(llama.layers):
            # FFN layers: gate_proj, up_proj, down_proj
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                proj_layer = getattr(layer.mlp, proj_name)
                input_dim = proj_layer.in_features
                output_dim = proj_layer.out_features
                
                adapter = APTAdapter(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    rank=self.config.get("initial_rank", 8),
                    alpha=self.config.get("alpha", 1.0),
                    device=self.device
                )
                
                adapter_name = f"adapter_{i}_ffn_{proj_name}"
                self.adapters[adapter_name] = adapter
                setattr(layer.mlp, f"{proj_name}_adapter", adapter)
                
    def forward(self, **kwargs):
        """Forward pass with APT adapters"""
        # Get base model output
        outputs = self.base_model(**kwargs)
        
        # Add adapter outputs to the base model outputs
        # This is a simplified version - in practice we would modify the attention/FFN computations
        return outputs
    
    def get_adapters_salience(self, gradients: Dict[str, torch.Tensor], activations: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate salience scores for all adapters"""
        salience_scores = {}
        
        for adapter_name, adapter in self.adapters.items():
            # Get gradients and activations for this adapter
            # This is a simplified version - in practice we would capture these during backward pass
            grad = gradients.get(adapter_name, torch.tensor(0.0))
            act = activations.get(adapter_name, torch.tensor(0.0))
            
            salience = adapter.get_layer_salience(grad, act)
            salience_scores[adapter_name] = salience
            
        return salience_scores
    
    def adaptive_pruning(self, salience_scores: Dict[str, float], target_sparsity: float):
        """Adaptive pruning based on salience scores"""
        # Get all adapter parameters and their salience
        adapter_info = []
        
        for adapter_name, adapter in self.adapters.items():
            # Calculate salience density (salience / number of parameters)
            # In practice, we would use the paper's method with kurtosis
            # Here we use a simplified version
            total_params = adapter.input_dim * adapter.current_rank + adapter.output_dim * adapter.current_rank
            salience = salience_scores.get(adapter_name, 0.0)
            salience_density = salience / (total_params + 1e-8)  # Avoid division by zero
            
            adapter_info.append({
                "name": adapter_name,
                "salience": salience,
                "salience_density": salience_density,
                "total_params": total_params,
                "adapter": adapter
            })
        
        # Sort by salience density (ascending)
        adapter_info.sort(key=lambda x: x["salience_density"])
        
        # Calculate total parameters and target pruned parameters
        total_params = sum(info["total_params"] for info in adapter_info)
        target_pruned = total_params * target_sparsity
        
        # Prune from least salient
        pruned = 0
        for info in adapter_info:
            if pruned >= target_pruned:
                break
                
            adapter = info["adapter"]
            # Simple pruning: prune 50% of input and output dimensions
            # In practice, we would use the knapsack algorithm from the paper
            input_prune_ratio = min(0.5, (target_pruned - pruned) / adapter.input_dim / 2)
            output_prune_ratio = min(0.5, (target_pruned - pruned) / adapter.output_dim / 2)
            
            # Create pruning masks
            input_prune_mask = torch.ones(adapter.input_dim, device=adapter.W_A.device)
            output_prune_mask = torch.ones(adapter.output_dim, device=adapter.W_A.device)
            
            # Prune input dimension
            n_prune_input = int(adapter.input_dim * input_prune_ratio)
            if n_prune_input > 0:
                input_prune_mask[:n_prune_input] = 0
                
            # Prune output dimension
            n_prune_output = int(adapter.output_dim * output_prune_ratio)
            if n_prune_output > 0:
                output_prune_mask[:n_prune_output] = 0
                
            # Apply pruning
            adapter.prune(input_prune_mask, output_prune_mask)
            pruned += n_prune_input * adapter.current_rank + n_prune_output * adapter.current_rank
            
    def adaptive_tuning(self, salience_scores: Dict[str, float], target_tuning: float):
        """Adaptive tuning: increase ranks in most salient layers"""
        # Get all adapter parameters and their salience
        adapter_info = []
        
        for adapter_name, adapter in self.adapters.items():
            salience = salience_scores.get(adapter_name, 0.0)
            adapter_info.append({
                "name": adapter_name,
                "salience": salience,
                "adapter": adapter,
                "current_rank": adapter.current_rank
            })
        
        # Sort by salience (descending)
        adapter_info.sort(key=lambda x: x["salience"], reverse=True)
        
        # Calculate total tuning parameters and target
        total_tuning = sum(info["current_rank"] * (info["adapter"].input_dim + info["adapter"].output_dim) 
                          for info in adapter_info)
        target_tuning_params = total_tuning * target_tuning
        
        # Increase ranks in top half of salient adapters
        top_half = len(adapter_info) // 2
        for i in range(top_half):
            info = adapter_info[i]
            adapter = info["adapter"]
            current_rank = adapter.current_rank
            new_rank = int(current_rank * 1.5)  # Increase by 50%
            
            # Don't exceed a maximum rank
            max_rank = 32
            new_rank = min(new_rank, max_rank)
            
            if new_rank > current_rank:
                adapter.update_rank(new_rank)
                
    def enable_distillation(self):
        """Enable self-distillation by creating a teacher copy"""
        # In practice, we would copy the model weights
        # Here we'll create a simple copy
        self.teacher_model = APTModel(self.base_model, self.config)
        self.teacher_model.load_state_dict(self.state_dict())
        
    def distillation_loss(self, student_outputs, teacher_outputs, layer_mapping=None):
        """Self-distillation loss"""
        if self.teacher_model is None:
            return torch.tensor(0.0, device=student_outputs.logits.device)
            
        # Calculate MSE between student and teacher outputs
        # This is a simplified version of the paper's layer-wise distillation
        if hasattr(student_outputs, "logits") and hasattr(teacher_outputs, "logits"):
            student_logits = student_outputs.logits
            teacher_logits = teacher_outputs.logits
            
            # Only compute loss for non-padded tokens
            if hasattr(student_outputs, "attention_mask"):
                mask = student_outputs.attention_mask.unsqueeze(-1)
                loss = F.mse_loss(student_logits * mask, teacher_logits * mask, reduction="sum")
                loss = loss / mask.sum()
            else:
                loss = F.mse_loss(student_logits, teacher_logits)
                
            return loss
            
        return torch.tensor(0.0, device=student_outputs.logits.device)