"""
Model definition for SEMA algorithm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class Adapter(nn.Module):
    """
    Modular adapter with representation descriptor (RD) as an autoencoder.
    """
    def __init__(self, input_dim=768, hidden_dim=64, reduction_ratio=8):
        super(Adapter, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.down_proj = nn.Linear(input_dim, input_dim // reduction_ratio)
        self.up_proj = nn.Linear(input_dim // reduction_ratio, input_dim)
        self.relu = nn.ReLU()
        
        # Representation descriptor: Autoencoder
        self.representation_descriptor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.zeros_(self.down_proj.bias)
        
    def forward(self, x):
        # Adapter branch
        adapter_out = self.relu(self.down_proj(x))
        adapter_out = self.up_proj(adapter_out)
        adapter_out = adapter_out + x  # Residual connection
        return adapter_out

class ExpandableRouter(nn.Module):
    """
    Expandable weighting router for mixture of adapters.
    """
    def __init__(self, input_dim=768, max_adapters=10):
        super(ExpandableRouter, self).__init__()
        self.input_dim = input_dim
        self.max_adapters = max_adapters
        # Initial weights for 1 adapter
        self.router_weights = nn.Parameter(torch.ones(1, self.input_dim))
        self.router_bias = nn.Parameter(torch.zeros(1))
        
        # Track current number of adapters
        self.current_adapters = 1
        
    def expand(self):
        """Expand router to accommodate new adapter"""
        if self.current_adapters < self.max_adapters:
            # Expand weights matrix by adding a new column
            new_weights = torch.zeros_like(self.router_weights.data[0, 0].unsqueeze(0))
        self.current_adapters += 1
        
    def forward(self, x, adapter_outputs, adapter_weights):
        # Compute router weights
        router_weights = torch.softmax(x @ self.router_weights.T + self.router_bias, dim=-1)
        # Weighted mixture of adapter outputs
        weighted_output = torch.sum(adapter_weights.unsqueeze(-1) * adapter_outputs, dim=1)
        return weighted_output

class SEMAModel(nn.Module):
    """
    SEMA model with self-expansion capability.
    """
    def __init__(self, num_classes=100, num_layers=12, hidden_dim=768):
        super(SEMAModel, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Load pre-trained ViT-B/16 model
        self.vit = models.vit_b_16(pretrained=True)
        
        # Freeze ViT parameters
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Initialize adapters for each layer
        self.adapters = nn.ModuleList([
            Adapter(input_dim=hidden_dim) for _ in range(num_layers)
        ])
        
        # Expandable router for each layer
        self.routers = nn.ModuleList([
            ExpandableRouter(input_dim=hidden_dim, max_adapters=10) for _ in range(num_layers)
        ])
        
        # Representation descriptors (autoencoders)
        self.representation_descriptors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
            ) for _ in range(num_layers)
        ]
        
        # Classifier head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        # ViT features
        x = self.vit._process_input(x)
        # ViT transformer blocks
        for i in range(self.num_layers):
            # Get adapter output
            adapter_out = self.adapters[i](x)
            # Get router output
        return x

class SEMAAdapter(nn.Module):
    """
    SEMA adapter implementation
    """
    def __init__(self, input_dim=768, hidden_dim=64):
        super(SEMAAdapter, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Adapter layers
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        
        # Representation descriptor (autoencoder)
        self.representation_descriptor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        # Adapter branch
        adapter_out = self.relu(self.down_proj(x))
        adapter_out = self.up_proj(adapter_out)
        adapter_out = adapter_out + x  # Residual connection
        return adapter_out

# Add this to the main file to use the model