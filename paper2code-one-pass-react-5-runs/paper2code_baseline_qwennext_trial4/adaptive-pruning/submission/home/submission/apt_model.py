import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig, RobertaForSequenceClassification
from transformers import T5Model, T5ForConditionalGeneration, T5Config
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

class APTRobertaForSequenceClassification(nn.Module):
    """
    APT-enhanced RoBERTa for sequence classification.
    Replaces some attention and FFN layers with APT adapters.
    """
    
    def __init__(self, model_name: str = "roberta-base", num_labels: int = 2, 
                 prune_ratio: float = 0.6, rank: int = 8, 
                 prune_layers: List[int] = None, use_ffn: bool = True):
        super().__init__()
        
        self.num_labels = num_labels
        self.prune_ratio = prune_ratio
        self.rank = rank
        self.use_ffn = use_ffn
        
        # Load base model
        self.roberta = RobertaModel.from_pretrained(model_name)
        
        # Configuration for APT adapters
        self.hidden_size = self.roberta.config.hidden_size
        self.num_attention_heads = self.roberta.config.num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        
        # Define which layers to apply APT to
        if prune_layers is None:
            self.prune_layers = list(range(self.roberta.config.num_hidden_layers))
        else:
            self.prune_layers = prune_layers
            
        # Initialize APT adapters for attention and FFN layers
        self.apts = nn.ModuleDict()
        
        for layer_idx in self.prune_layers:
            # For attention: apply to query and value projections
            # Query and value projections are both of size hidden_size
            self.apts[f"layer_{layer_idx}_query"] = APTAdapter(
                self.hidden_size, self.hidden_size, rank=rank, alpha=2.0
            )
            self.apts[f"layer_{layer_idx}_value"] = APTAdapter(
                self.hidden_size, self.hidden_size, rank=rank, alpha=2.0
            )
            
            # For FFN if enabled
            if use_ffn:
                intermediate_size = self.roberta.config.intermediate_size
                self.apts[f"layer_{layer_idx}_ffn_up"] = APTAdapter(
                    self.hidden_size, intermediate_size, rank=rank, alpha=2.0
                )
                self.apts[f"layer_{layer_idx}_ffn_down"] = APTAdapter(
                    intermediate_size, self.hidden_size, rank=rank, alpha=2.0
                )
        
        # Classification head
        self.dropout = nn.Dropout(self.roberta.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Initialize pruning masks and adaptive tuning parameters
        self.pruning_masks = {}
        self.adaptive_ranks = {}
        self.initialize_pruning_masks()
        
        # Set up for adaptive tuning
        self.current_rank = rank
        self.max_rank = rank * 2  # Maximum rank for adaptive tuning
        self.tuning_step = 0
        self.tuning_schedule = [0.2, 0.5, 0.8]  # When to increase ranks
        
        # For self-distillation
        self.teacher_model = None
        
    def initialize_pruning_masks(self):
        """Initialize pruning masks for all APT adapters."""
        for layer_idx in self.prune_layers:
            # Initialize masks for attention layers
            self.pruning_masks[f"layer_{layer_idx}_query"] = {
                'input': torch.ones(self.hidden_size),
                'output': torch.ones(self.hidden_size)
            }
            self.pruning_masks[f"layer_{layer_idx}_value"] = {
                'input': torch.ones(self.hidden_size),
                'output': torch.ones(self.hidden_size)
            }
            
            # Initialize masks for FFN layers if enabled
            if self.use_ffn:
                intermediate_size = self.roberta.config.intermediate_size
                self.pruning_masks[f"layer_{layer_idx}_ffn_up"] = {
                    'input': torch.ones(self.hidden_size),
                    'output': torch.ones(intermediate_size)
                }
                self.pruning_masks[f"layer_{layer_idx}_ffn_down"] = {
                    'input': torch.ones(intermediate_size),
                    'output': torch.ones(self.hidden_size)
                }
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None, 
                return_hidden_states: bool = False) -> Dict:
        
        # Get base model outputs
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply APT adapters to selected layers
        hidden_states = sequence_output
        all_hidden_states = [hidden_states] if return_hidden_states else None
        
        for layer_idx in self.prune_layers:
            # Get the original layer
            layer = self.roberta.encoder.layer[layer_idx]
            
            # Apply attention with APT adapters
            attention_output = self._apply_attention_with_apts(
                layer, hidden_states, attention_mask, layer_idx
            )
            
            # Apply FFN with APT adapters if enabled
            if self.use_ffn:
                ffn_output = self._apply_ffn_with_apts(
                    layer, attention_output, layer_idx
                )
            else:
                ffn_output = layer.output.dense(attention_output)
                ffn_output = layer.output.dropout(ffn_output)
                ffn_output = layer.output.LayerNorm(ffn_output + attention_output)
            
            hidden_states = ffn_output
            
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
        
        # Pooling and classification
        pooled_output = hidden_states[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        result = {
            'logits': logits,
            'loss': loss,
            'hidden_states': all_hidden_states
        }
        
        return result
    
    def _apply_attention_with_apts(self, layer, hidden_states, attention_mask, layer_idx):
        """Apply APT adapters to attention layers."""
        # Self-attention
        attention_output = layer.attention.self(
            hidden_states, attention_mask, None, None, None, None, None, None
        )[0]
        
        # Apply APT adapters to query and value projections
        # Note: This is a simplified version - in practice, we'd need to access the weights
        # For this implementation, we'll assume the attention weights are accessible
        
        # Apply to query
        query_adapter = self.apts[f"layer_{layer_idx}_query"]
        # Apply to value
        value_adapter = self.apts[f"layer_{layer_idx}_value"]
        
        # In a real implementation, we'd modify the attention weights directly
        # Here we simulate by adding the adapter output
        attention_output = attention_output + query_adapter(attention_output) + value_adapter(attention_output)
        
        # Apply output projection and layer norm
        attention_output = layer.attention.output.dense(attention_output)
        attention_output = layer.attention.output.dropout(attention_output)
        attention_output = layer.attention.output.LayerNorm(attention_output + hidden_states)
        
        return attention_output
    
    def _apply_ffn_with_apts(self, layer, attention_output, layer_idx):
        """Apply APT adapters to FFN layers."""
        # FFN layers
        intermediate_output = layer.intermediate.dense(attention_output)
        
        # Apply APT adapter to up projection
        ffn_up_adapter = self.apts[f"layer_{layer_idx}_ffn_up"]
        intermediate_output = intermediate_output + ffn_up_adapter(intermediate_output)
        
        # Apply activation
        intermediate_output = layer.intermediate.intermediate_act_fn(intermediate_output)
        
        # Apply APT adapter to down projection
        ffn_down_adapter = self.apts[f"layer_{layer_idx}_ffn_down"]
        output = layer.output.dense(intermediate_output)
        output = output + ffn_down_adapter(output)
        
        # Apply dropout and layer norm
        output = layer.output.dropout(output)
        output = layer.output.LayerNorm(output + attention_output)
        
        return output
    
    def calculate_salience_scores(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                                labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate outlier-aware salience scores for pruning.
        This is a simplified version that would be computed during training.
        """
        # We need to do a forward and backward pass to get gradients
        self.train()
        self.zero_grad()
        
        outputs = self.forward(input_ids, attention_mask, labels)
        loss = outputs['loss']
        loss.backward()
        
        salience_scores = {}
        
        # Calculate salience for each APT adapter
        for layer_idx in self.prune_layers:
            # For query adapter
            adapter = self.apts[f"layer_{layer_idx}_query"]
            # Get gradients and activations (simplified)
            # In practice, we'd capture these during forward/backward
            # For this implementation, we'll use placeholder values
            salience_scores[f"layer_{layer_idx}_query"] = torch.rand(self.hidden_size)
            salience_scores[f"layer_{layer_idx}_value"] = torch.rand(self.hidden_size)
            
            if self.use_ffn:
                intermediate_size = self.roberta.config.intermediate_size
                salience_scores[f"layer_{layer_idx}_ffn_up"] = torch.rand(intermediate_size)
                salience_scores[f"layer_{layer_idx}_ffn_down"] = torch.rand(self.hidden_size)
        
        return salience_scores
    
    def adaptive_pruning(self, salience_scores: Dict[str, torch.Tensor], target_sparsity: float = 0.6):
        """
        Apply adaptive pruning based on salience scores.
        Uses binary search on salience density to meet sparsity constraint.
        """
        # Collect all blocks to prune
        blocks = []
        block_types = []  # 'query', 'value', 'ffn_up', 'ffn_down'
        block_sizes = []  # size of each block
        block_names = []  # name of each block
        
        for layer_idx in self.prune_layers:
            # Query and value blocks (attention)
            for block_name in [f"layer_{layer_idx}_query", f"layer_{layer_idx}_value"]:
                if block_name in salience_scores:
                    blocks.append(salience_scores[block_name])
                    block_types.append(block_name.split('_')[-1])
                    block_sizes.append(self.hidden_size)
                    block_names.append(block_name)
            
            # FFN blocks if enabled
            if self.use_ffn:
                for block_name in [f"layer_{layer_idx}_ffn_up", f"layer_{layer_idx}_ffn_down"]:
                    if block_name in salience_scores:
                        blocks.append(salience_scores[block_name])
                        block_types.append(block_name.split('_')[-1])
                        if block_name.endswith('_up'):
                            block_sizes.append(self.roberta.config.intermediate_size)
                        else:
                            block_sizes.append(self.hidden_size)
                        block_names.append(block_name)
        
        # Calculate salience density (salience / parameter count)
        salience_densities = []
        for i, block in enumerate(blocks):
            # Average salience over the block
            avg_salience = torch.mean(block).item()
            # Salience density = salience / number of parameters in block
            salience_density = avg_salience / block_sizes[i]
            salience_densities.append((salience_density, i, block_sizes[i], block_names[i]))
        
        # Sort by salience density (ascending - we want to prune lowest salience first)
        salience_densities.sort(key=lambda x: x[0])
        
        # Calculate total parameters and target pruned count
        total_params = sum(block_sizes)
        target_pruned = int(total_params * target_sparsity)
        
        # Binary search to find the optimal pruning threshold
        # We'll prune blocks until we reach target sparsity
        pruned_count = 0
        pruned_blocks = []
        
        for salience_density, idx, block_size, block_name in salience_densities:
            if pruned_count + block_size <= target_pruned:
                pruned_count += block_size
                pruned_blocks.append(block_name)
            else:
                break
        
        # Apply pruning masks
        for block_name in pruned_blocks:
            if 'query' in block_name or 'value' in block_name:
                # Prune input and output dimensions equally
                mask_size = self.hidden_size
                # For simplicity, we'll prune a random subset
                # In practice, we'd use the salience scores to determine which dimensions to prune
                pruned_indices = torch.randperm(mask_size)[:int(mask_size * 0.5)]
                mask = torch.ones(mask_size)
                mask[pruned_indices] = 0.0
                
                if 'query' in block_name:
                    self.pruning_masks[block_name]['input'] = mask.clone()
                    self.pruning_masks[block_name]['output'] = mask.clone()
                else:
                    self.pruning_masks[block_name]['input'] = mask.clone()
                    self.pruning_masks[block_name]['output'] = mask.clone()
                    
            elif 'ffn_up' in block_name:
                # FFN up projection
                mask_size = self.roberta.config.intermediate_size
                pruned_indices = torch.randperm(mask_size)[:int(mask_size * 0.5)]
                mask = torch.ones(mask_size)
                mask[pruned_indices] = 0.0
                self.pruning_masks[block_name]['input'] = torch.ones(self.hidden_size)
                self.pruning_masks[block_name]['output'] = mask.clone()
                
            elif 'ffn_down' in block_name:
                # FFN down projection
                mask_size = self.hidden_size
                pruned_indices = torch.randperm(mask_size)[:int(mask_size * 0.5)]
                mask = torch.ones(mask_size)
                mask[pruned_indices] = 0.0
                self.pruning_masks[block_name]['input'] = torch.ones(self.roberta.config.intermediate_size)
                self.pruning_masks[block_name]['output'] = mask.clone()
        
        # Apply the masks to the adapters
        for block_name in self.pruning_masks:
            if block_name in self.apts:
                adapter = self.apts[block_name]
                input_mask = self.pruning_masks[block_name]['input']
                output_mask = self.pruning_masks[block_name]['output']
                adapter.apply_pruning(input_mask, output_mask)
    
    def adaptive_tuning(self, adapter_salience_scores: Dict[str, float], 
                       current_step: int, total_steps: int):
        """
        Dynamically increase ranks in salient layers.
        """
        if current_step < total_steps * self.tuning_schedule[0]:
            return  # Too early to start tuning
            
        # Sort adapters by salience
        sorted_adapters = sorted(adapter_salience_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Determine how many adapters to upgrade
        # Increase rank for top half of salient adapters
        num_to_upgrade = len(sorted_adapters) // 2
        
        # Increase rank for top salient adapters
        for i, (adapter_name, salience) in enumerate(sorted_adapters[:num_to_upgrade]):
            if adapter_name in self.apts and self.apts[adapter_name].current_rank < self.max_rank:
                new_rank = min(self.apts[adapter_name].current_rank * 2, self.max_rank)
                self.apts[adapter_name].increase_rank(new_rank)
                self.current_rank = new_rank
    
    def get_pruned_parameters_count(self) -> int:
        """Get total number of pruned parameters."""
        total_pruned = 0
        for block_name in self.apts:
            adapter = self.apts[block_name]
            input_pruned, output_pruned = adapter.get_pruned_parameters()
            # This is a simplified estimate
            effective_params = adapter.get_effective_parameters()
            # We'll use a rough estimate
            total_pruned += input_pruned + output_pruned
        return total_pruned
    
    def get_total_parameters(self) -> int:
        """Get total number of parameters in the model."""
        total = 0
        for name, param in self.named_parameters():
            if 'apts' in name:
                total += param.numel()
        return total