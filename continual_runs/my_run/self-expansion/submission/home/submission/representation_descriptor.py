#!/usr/bin/env python3
"""
Implementation of representation descriptor for SEMA
Based on paper_card_0002 and paper_card_0046
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RepresentationDescriptor(nn.Module):
    """
    Autoencoder-based representation descriptor for distribution shift detection
    Implements the autoencoder-based detection as described in paper_card_0002
    - Input: intermediate layer features
    - Output: distribution similarity score
    - Trained via reconstruction loss
    - Detects novel feature patterns
    """
    
    def __init__(self, feature_dim, hidden_dim=128, dropout=0.1):
        super(RepresentationDescriptor, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # For distribution shift detection
        self.register_buffer('mean', torch.zeros(feature_dim))
        self.register_buffer('std', torch.ones(feature_dim))
        
        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through the representation descriptor
        x: (batch_size, feature_dim)
        Returns: reconstruction, reconstruction_error, score
        """
        # Encode
        encoded = self.encoder(x)
        # Decode
        reconstructed = self.decoder(encoded)
        
        # Calculate reconstruction error
        reconstruction_error = F.mse_loss(reconstructed, x, reduction='none').mean(dim=1)
        
        # Calculate distribution shift score (z-score based as in paper)
        # Normalize reconstruction error
        score = (reconstruction_error - self.mean.mean()) / (self.std.std() + 1e-8)
        
        return reconstructed, reconstruction_error, score
    
    def update_statistics(self, x):
        """
        Update mean and std of reconstruction errors for normalization
        """
        with torch.no_grad():
            reconstructed, reconstruction_error, _ = self.forward(x)
            self.mean = self.mean * 0.9 + reconstruction_error.mean() * 0.1
            self.std = self.std * 0.9 + reconstruction_error.std() * 0.1
    
    def detect_shift(self, x, threshold=0.1):
        """
        Detect distribution shift based on reconstruction error
        Returns True if shift detected
        """
        _, _, score = self.forward(x)
        return score.mean() > threshold