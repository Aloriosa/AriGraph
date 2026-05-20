"""
Model implementation for NPSE algorithm.
This implements the score network architecture described in the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class ScoreNetwork(nn.Module):
    """
    Score network for estimating the score of the posterior distribution.
    This implements the architecture described in the paper with MLP embeddings
    for θ and x, and sinusoidal embedding for time t.
    """
    
    def __init__(self, theta_dim, x_dim, hidden_dim=256, num_layers=3):
        super(ScoreNetwork, self).__init__()
        
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding networks for θ and x
        self.theta_embedding = self._create_embedding_network(theta_dim, hidden_dim)
        self.x_embedding = self._create_embedding_network(x_dim, hidden_dim)
        
        # Time embedding (sinusoidal)
        self.time_embed_dim = 64
        self.time_embedding = nn.Linear(1, self.time_embed_dim)
        
        # Final MLP for combining embeddings
        self.final_mlp = self._create_mlp(hidden_dim * 2 + self.time_embed_dim, hidden_dim, num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, theta_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _create_embedding_network(self, input_dim, hidden_dim):
        """Create a simple embedding network for θ or x"""
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU()
        ]
        
        return nn.Sequential(*layers)
    
    def _create_mlp(self, input_dim, hidden_dim, num_layers):
        """Create an MLP with SiLU activations"""
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim)
        # Add SiLU activation for all layers except the last
            if i < num_layers - 1:
                layers.append(nn.SiLU())
        
        return nn.Sequential(*layers)
    
    def forward(self, theta, x, t):
        """
        Forward pass of the score network.
        
        Args:
            theta: (batch_size, theta_dim)
            x: (batch, x_dim)
            t: (batch_size, 1)
            
        Returns:
            score: (batch_size, theta_dim)
        """
        # Embed θ and x
        theta_emb = self.theta_embedding(theta)
        x_emb = self.x_embedding(x)
        
        # Time embedding
        time_emb = torch.sin(self.time_embedding(t))
        
        # Concatenate embeddings
        combined = torch.cat([theta_emb, x_emb, time_emb], dim=-1)
        
        # Pass through final MLP
        output = self.final_mlp(combined)
        
        # Output layer
        score = self.output_layer(output)
        
        return score

class DiffusionProcess(nn.Module):
    """
    Implementation of the forward and reverse diffusion processes.
    This implements the variance exploding (VE) and variance preserving (VP) SDEs
    as described in the paper.
    """
    
    def __init__(self, beta_min=0.1, beta_max=11.0, T=1.0):
        super(DiffusionProcess, self).__init__()
        
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define the drift and diffusion coefficients for the SDE
        self.beta_t = lambda t: self.beta_min + t * (self.beta_max - self.beta_min)
        
        # For VP SDE
        self.alpha_bar_t = lambda t: torch.exp(-0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2))
        self.sigma_t = lambda t: torch.sqrt(1 - self.alpha_bar_t(t))
        
    def forward_sde(self, theta_0, t):
        """
        Forward SDE for the variance preserving (VP) SDE.
        
        Args:
            theta_0: Initial samples (batch_size, theta_dim)
            t: Time (batch_size, 1)
            
        Returns:
            theta_t: Samples at time t
        """
        # For VP SDE
        beta_t = self.beta_t(t)
        alpha_t = torch.exp(-0.5 * beta_t * t)
        sigma_t = torch.sqrt(1 - alpha_t)
        
        # Sample noise
        noise = torch.randn_like(theta_0)
        
        # Forward SDE
        theta_t = alpha_t * theta_0 + sigma_t * noise
        
        return theta_t

# Define the diffusion process for the reverse SDE
class ReverseDiffusionProcess(nn.Module):
    """
    Implementation of the reverse diffusion process.
    This implements the reverse SDE for the variance preserving (VP) SDE.
    """
    
    def __init__(self, beta_min=0.1, beta_max=11.0, T=1.0):
        super(ReverseDiffusionProcess, self).__init__()
        
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define the drift and diffusion coefficients for the SDE
        self.beta_t = lambda t: self.beta_min + t * (self.beta_max - self.beta_min)
        
        # For VP SDE
        self.alpha_bar_t = lambda t: torch.exp(-0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2))
        self.sigma_t = lambda t: torch.sqrt(1 - self.alpha_bar_t(t))
        
    def reverse_sde(self, theta_t, x, t, score_network):
        """
        Reverse SDE for the variance preserving (VP) SDE.
        
        Args:
            theta_t: Samples at time t (batch_size, theta_dim)
            x: Observation (batch_size, theta_dim)
            t: Time (batch_size, 1)
            score_network: Score network to estimate the score of the posterior
        """
        # For VP SDE
        beta_t = self.beta_t(t)
        alpha_t = torch.exp(-0.5 * beta_t * t)
        sigma_t = torch.sqrt(1 - alpha_t)
        
        # Estimate the score of the posterior
        score = score_network(theta_t, x, t)
        
        # Reverse SDE
        theta_t_minus_1 = theta_t + 0.5 * beta_t * score * t + torch.sqrt(beta_t) * torch.randn_like(theta_t)
        
        return theta_t_minus_1