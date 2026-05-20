#!/usr/bin/env python3
"""
Implementation of SEMA (Self-Expanding Modular Adaptation) model
Based on paper_card_0000, paper_card_0001, paper_card_0002, paper_card_0004, paper_card_0006
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from adapter_module import Adapter
from representation_descriptor import RepresentationDescriptor
from expandable_router import ExpandableWeightingRouter

class SEMA(nn.Module):
    """
    Self-Expanding Modular Adaptation model for continual learning
    Implements the complete SEMA algorithm as described in the paper
    """
    
    def __init__(self, num_classes, model_name="google/vit-base-patch16-224-in21k", 
                 adapter_dim=64, expansion_threshold=0.1, max_adapters_per_layer=3):
        super(SEMA, self).__init__()
        
        self.num_classes = num_classes
        self.adapter_dim = adapter_dim
        self.expansion_threshold = expansion_threshold
        self.max_adapters_per_layer = max_adapters_per_layer
        
        # Load pre-trained ViT model
        self.vit_config = ViTConfig.from_pretrained(model_name)
        self.vit = ViTModel.from_pretrained(model_name)
        
        # Freeze ViT parameters
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Get transformer layer count
        self.num_layers = self.vit_config.num_hidden_layers
        
        # Create adapter modules for each layer
        self.adapters = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            # Each layer has a list of adapters
            self.adapters.append(nn.ModuleList())
        
        # Create representation descriptors for each layer
        self.representation_descriptors = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            # Use the hidden size of ViT as feature dimension
            feature_dim = self.vit_config.hidden_size
            self.representation_descriptors.append(
                RepresentationDescriptor(feature_dim, hidden_dim=128, dropout=0.1)
            )
        
        # Create expandable routers for each layer
        self.routers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            # Start with one dummy adapter (identity)
            self.routers.append(
                ExpandableWeightingRouter(
                    input_dim=self.vit_config.hidden_size,
                    num_adapters=1,
                    hidden_dim=128,
                    dropout=0.1
                )
            )
        
        # Create task-specific classifier head
        self.classifier = nn.Linear(self.vit_config.hidden_size, num_classes)
        
        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
        # Track adapter expansion history
        self.adapter_expansion_history = {}
        self.current_task = 0
        
        # Store task-to-adapter mapping
        self.task_adapter_mapping = {}
        
    def forward(self, pixel_values, task_id=None):
        """
        Forward pass through SEMA model
        """
        # Get ViT outputs
        outputs = self.vit(pixel_values=pixel_values, output_hidden_states=True)
        
        # Get hidden states from all layers
        hidden_states = outputs.hidden_states  # List of length num_layers+1
        
        # Process each transformer layer
        for layer_idx in range(self.num_layers):
            # Get input to transformer block
            x = hidden_states[layer_idx]  # (batch_size, seq_len, hidden_size)
            
            # Get representation descriptor for this layer
            rd = self.representation_descriptors[layer_idx]
            
            # Detect distribution shift
            # Only do this during training
            if self.training:
                # Update representation descriptor statistics
                rd.update_statistics(x.mean(dim=1))  # Use mean over sequence dimension
                
                # Check if we need to expand
                if task_id is not None and task_id > 0:
                    # Only expand if this is a new task
                    if len(self.adapters[layer_idx]) == 0:
                        # First task, add initial adapter
                        adapter = Adapter(self.vit_config.hidden_size, self.adapter_dim)
                        self.adapters[layer_idx].append(adapter)
                        self.routers[layer_idx].add_adapter()
                        self.adapter_expansion_history.setdefault(layer_idx, []).append(task_id)
                    else:
                        # Check if distribution shift detected
                        shift_detected = rd.detect_shift(x.mean(dim=1), self.expansion_threshold)
                        
                        # Only add new adapter if shift detected and we haven't reached max
                        if shift_detected and len(self.adapters[layer_idx]) < self.max_adapters_per_layer:
                            # Add new adapter
                            adapter = Adapter(self.vit_config.hidden_size, self.adapter_dim)
                            self.adapters[layer_idx].append(adapter)
                            self.routers[layer_idx].add_adapter()
                            self.adapter_expansion_history.setdefault(layer_idx, []).append(task_id)
            
            # Get adapter outputs
            adapter_outputs = []
            for adapter in self.adapters[layer_idx]:
                adapter_output = adapter(x)
                adapter_outputs.append(adapter_output)
            
            # Use router to combine adapter outputs
            if len(adapter_outputs) > 0:
                x, weights = self.routers[layer_idx](x, adapter_outputs)
            else:
                # Fallback: use original features
                x = x
            
            # Update hidden state for next layer
            hidden_states[layer_idx + 1] = x
        
        # Use the final hidden state for classification
        sequence_output = hidden_states[-1]
        cls_token = sequence_output[:, 0]  # CLS token
        
        # Apply classifier
        logits = self.classifier(cls_token)
        
        return logits
    
    def add_task(self, num_classes):
        """
        Add a new task to the model
        """
        self.current_task += 1
        self.task_adapter_mapping[self.current_task] = []
        
        # Update classifier if needed
        if num_classes != self.num_classes:
            self.num_classes = num_classes
            self.classifier = nn.Linear(self.vit_config.hidden_size, num_classes)
            nn.init.normal_(self.classifier.weight, std=0.02)
            nn.init.zeros_(self.classifier.bias)
    
    def get_adapter_count(self):
        """
        Get total number of adapters in the model
        """
        total = 0
        for layer_adapters in self.adapters:
            total += len(layer_adapters)
        return total
    
    def get_expansion_history(self):
        """
        Get adapter expansion history
        """
        return self.adapter_expansion_history