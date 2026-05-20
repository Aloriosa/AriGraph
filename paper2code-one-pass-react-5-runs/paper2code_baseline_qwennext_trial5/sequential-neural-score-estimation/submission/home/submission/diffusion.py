"""
Diffusion process implementation for NPSE algorithm.
This implements the forward and reverse diffusion processes.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

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
    
    def reverse_sde(self, theta_t, x, t, score_network):
        """
        Reverse SDE for the variance preserving (VP) SDE.
        
        Args:
            theta_t: Samples at time t (batch_size, theta_dim)
            x: Observation (batch_size, x_dim)
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
            x: Observation (batch_size, x_dim)
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