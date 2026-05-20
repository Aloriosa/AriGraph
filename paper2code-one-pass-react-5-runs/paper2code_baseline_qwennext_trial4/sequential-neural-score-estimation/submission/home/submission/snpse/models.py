"""
Models for Neural Posterior Score Estimation (NPSE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging

# Set up logging
logger = logging.getLogger(__name__)

class ScoreNetwork(nn.Module):
    """
    Score network for estimating the score of the posterior distribution.
    This network takes as input the state theta, observation x, and time t, and outputs an estimate of the score.
    """
    
    def __init__(self, theta_dim, x_dim, time_dim=64, hidden_dim=256, num_layers=3):
        """
        Initialize the score network.
        
        Args:
            theta_dim: Dimension of the parameter space
            x_dim: Dimension of the observation space
            time_dim: Dimension of the time embedding
            hidden_dim: Hidden dimension of the MLP
            num_layers: Number of layers in the MLP
        """
        super(ScoreNetwork, self).__init__()
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding for theta
        self.theta_embedding = nn.Sequential(
            nn.Linear(theta_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Embedding for x
        self.x_embedding = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Embedding for time
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_dim),
        )
        
        # Main network
        self.main_network = nn.ModuleList()
        self.main_network.append(nn.Linear(hidden_dim * 2 + time_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.main_network.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.output_layer = nn.Linear(hidden_dim, theta_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for the network."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
    
    def forward(self, theta, x, t):
        """
        Forward pass of the score network.
        
        Args:
            theta: Parameters, shape (batch_size, theta_dim)
            x: Observations, shape (batch_size, x_dim)
            t: Time, shape (batch_size, 1)
        
        Returns:
            score: Estimated score, shape (batch_size, theta_dim)
        """
        # Embed theta
        theta_emb = self.theta_embedding(theta)
        
        # Embed x
        x_emb = self.x_embedding(x)
        
        # Embed time
        time_emb = self.time_embedding(t)
        
        # Concatenate embeddings
        combined = torch.cat([theta_emb, x_emb, time_emb], dim=-1)
        
        # Pass through main network
        for i in range(self.num_layers):
            if i == 0:
                hidden = self.main_network[i](combined)
            else:
                hidden = self.main_network[i](hidden)
            if i < self.num_layers - 1:
                hidden = F.silu(hidden)
        
        # Output layer
        score = self.output_layer(hidden)
        
        return score

class DiffusionModel(nn.Module):
    """
    Diffusion model for generating samples from the posterior distribution.
    This model implements the forward and reverse diffusion processes.
    """
    
    def __init__(self, score_network, timesteps=100, beta_start=0.0001, beta_end=0.02):
        """
        Initialize the diffusion model.
        
        Args:
            score_network: Score network for estimating the score of the posterior
            timesteps: Number of diffusion timesteps
            beta_start: Starting value for beta
            beta_end: Ending value for beta
        """
        super(DiffusionModel, self).__init__()
        self.score_network = score_network
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Define beta schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_minus_one = torch.sqrt(1.0 / self.alphas - 1.0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-8)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        
        # Move to device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_recip_alphas_minus_one = self.sqrt_recip_alphas_minus_one.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
    
    def forward(self, x, t, theta, return_loss=True):
        """
        Forward pass of the diffusion model.
        
        Args:
            x: Observations, shape (batch_size, x_dim)
            t: Time, shape (batch, 1)
            theta: Parameters, shape (batch, theta_dim)
        """
        # Sample noise
        noise = torch.randn_like(theta)
        
        # Forward diffusion process
        theta_t = self.q_sample(theta, t, noise)
        
        # Estimate score
        estimated_score = self.score_network(theta_t, x, t)
        
        if return_loss:
            # Compute loss
            loss = F.mse_loss(estimated_score, noise)
            return loss
        else:
            return estimated_score
    
    def q_sample(self, x_start, t, noise=None):
        """
        Sample from the forward diffusion process.
        
        Args:
            x_start: Initial state
            t: Time
            noise: Noise to add
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return self.sqrt_alphas_cumprod[t, None] * x_start + self.sqrt_one_minus_alphas_cumprod[t, None] * noise
    
    def p_sample(self, x, t, x_start, noise=None):
        """
        Sample from the reverse diffusion process.
        
        Args:
            x: Current state
            t: Time
            x_start: Initial state
        """
        if noise is None:
            noise = torch.randn_like(x)
        
        # Estimate score
        estimated_score = self.score_network(x, x, t)
        
        # Reverse diffusion process
        mean = self.posterior_mean_coef1[t, None] * x + self.posterior_mean_coef2[t, None] * x_start
        return mean + torch.exp(0.5 * self.posterior_log_variance_clipped[t, None]) * noise
    
    def sample(self, x, steps=None):
        """
        Generate samples from the posterior distribution.
        
        Args:
            x: Observation
        """
        if steps is None:
            steps = self.timesteps
        
        # Start from noise
        x_t = torch.randn_like(x)
        
        # Reverse diffusion process
        for t in range(steps - 1, -1, -1):
            t_batch = torch.full((x.shape[0], 1), t, dtype=torch.float32)
            x_t = self.p_sample(x_t, t_batch, x, noise=None)
        
        return x_t