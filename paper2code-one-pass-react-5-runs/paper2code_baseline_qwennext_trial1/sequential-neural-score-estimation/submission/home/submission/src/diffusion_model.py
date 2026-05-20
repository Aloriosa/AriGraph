import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ScoreNetwork(nn.Module):
    """
    Neural network to estimate the score function ∇θ log p_t(θ | x)
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=3):
        super(ScoreNetwork, self).__init__()
        
        # Input dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding networks for θ and x
        self.theta_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        )
        
        self.x_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        )
        
        # Time embedding using sinusoidal encoding
        self.time_embedding = nn.Sequential(
            nn.Linear(64, hidden_dim),
        )
        
        # Main network
        layers = []
        layers.append(nn.Linear(hidden_dim * 3, hidden_dim))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.network = nn.Sequential(*layers)
        
        # Final layer to output the score (same dimension as input)
        self.final_layer = nn.Linear(hidden_dim, input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, theta_t, x, t):
        # Embed θ_t
        theta_emb = self.theta_embedding(theta_t)
        
        # Embed x
        x_emb = self.x_embedding(x)
        
        # Embed time t using sinusoidal encoding
        t_emb = self._sinusoidal_encoding(t, 64)
        t_emb = self.time_embedding(t_emb)
        
        # Concatenate embeddings
        combined = torch.cat([theta_emb, x_emb, t_emb], dim=-1)
        
        # Pass through network
        out = self.network(combined)
        
        # Final layer
        score = self.final_layer(out)
        
        return score
    
    def _sinusoidal_encoding(self, t, d_model):
        """Create sinusoidal positional encoding for time t"""
        batch_size = t.size(0)
        pe = torch.zeros(batch_size, d_model)
        position = t.unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

class DiffusionModel:
    """
    Implementation of the diffusion model for score estimation
    """
    def __init__(self, input_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.input_dim = input_dim
        self.device = device
        self.score_network = ScoreNetwork(input_dim).to(device)
        self.timesteps = 100
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar = torch.cat([torch.tensor([1.0]).to(self.device), self.alpha_bar[:-1]])
        self.alpha_bar = self.alpha_bar.to(self.device)
        
    def forward_diffusion(self, x0, t):
        """Forward diffusion process: q(x_t | x0)"""
        noise = torch.randn_like(x0)
        alpha_t = self.alpha_bar[t].view(-1, 1)
        x_t = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        return x_t, noise
    
    def reverse_diffusion(self, x_t, x, t):
        """Reverse diffusion process using the score network"""
        # Use the score network to estimate the score of the posterior
        score = self.score_network(x_t, x, t)
        
        # Euler-Maruyama step
        dt = 1.0 / self.timesteps
        x_t = x_t + (self.beta[t] * score) * dt
        return x_t
    
    def loss(self, x0, x, t):
        """Loss function: score matching loss"""
        x_t, noise = self.forward_diffusion(x0, t)
        score = self.score_network(x_t, x, t)
        loss = torch.mean(torch.sum((score + noise) ** 2, dim=1))
        return loss
    
    def train(self, data_loader, epochs=10, lr=1e-4):
        """Train the diffusion model"""
        optimizer = torch.optim.Adam(self.score_network.parameters(), lr=lr)
        self.score_network.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (x0, x) in enumerate(data_loader):
                t = torch.randint(0, self.timesteps, (x.size(0),), device=self.device)
                optimizer.zero_grad()
                loss = self.loss(x0, x, t)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader)}")
    
    def sample(self, x, num_samples=100):
        """Generate samples from the posterior using reverse diffusion"""
        self.score_network.eval()
        samples = []
        x_t = torch.randn(x.size(0), self.input_dim).to(self.device)
        
        for t in reversed(range(self.timesteps)):
            with torch.no_grad():
                score = self.score_network(x_t, x, torch.tensor([t]).to(self.device))
            dt = 1.0 / self.timesteps
            x_t = x_t + (self.beta[t] * score) * dt
        
        return x_t

# This is a simplified implementation of the core algorithm described in the paper.
# The full implementation would require additional components like the sequential training procedure,
# truncated proposals, and the full SDE solvers as described in the paper.