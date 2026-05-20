import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class Simformer(nn.Module):
    """
    Simformer: A unified framework for simulation-based inference using transformers
    and probabilistic diffusion models.
    """
    
    def __init__(self, 
                 token_dim=64, 
                 hidden_dim=256, 
                 n_layers=4, 
                 n_heads=4, 
                 dropout=0.1):
        super(Simformer, self).__init__()
        
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Token embeddings for variable identifiers
        self.id_embedding = nn.Embedding(100, token_dim)
        self.value_embedding = nn.Linear(1, token_dim)
        self.condition_embedding = nn.Embedding(2, token_dim)
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation='relu',
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        # Score network (MLP)
        self.score_net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Learnable time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x, t, mask=None):
        """
        Forward pass of the Simformer model.
        
        Args:
            x: Input tensor of shape (batch_size, n_vars, 1)
            t: Time step tensor of shape (batch_size, 1)
            mask: Attention mask of shape (batch_size, n_vars)
            
        Returns:
            scores: Predicted scores of shape (batch_size, n_vars, 1)
        """
        batch_size, n_vars = x.shape[0], x.shape[1]
        
        # Embed the input tokens
        # Each token consists of: variable_id, value, condition_state
        variable_ids = torch.arange(n_vars).unsqueeze(0).expand(batch_size, n_vars)
        variable_ids = variable_ids.to(x.device)
        
        # Embed variable identifiers
        id_embed = self.id_embedding(variable_ids)  # (batch_size, n_vars, token_dim)
        
        # Embed values
        value_embed = self.value_embedding(x)  # (batch_size, n_vars, token_dim)
        
        # Embed condition states
        condition_embed = self.condition_embedding(mask)  # (batch_size, n_vars, token_dim)
        
        # Combine embeddings
        token_embeddings = id_embed + value_embed + condition_embed
        
        # Add time embedding
        time_embed = self.time_embedding(t)  # (batch_size, hidden_dim)
        time_embed = time_embed.unsqueeze(1).expand(-1, n_vars, -1)
        
        # Add time embedding to each token
        token_embeddings = token_embeddings + time_embed
        
        # Apply transformer
        transformer_out = self.transformer(token_embeddings)  # (batch_size, n_vars, token_dim)
        
        # Predict scores
        scores = self.score_net(transformer_out)  # (batch_size, n_vars, 1)
        
        return scores

class DiffusionModel(nn.Module):
    """
    Score-based Diffusion Model for simulation-based inference.
    """
    
    def __init__(self, simformer, n_timesteps=100, beta_start=0.0001, beta_end=0.02):
        super(DiffusionModel, self).__init__()
        
        self.simformer = simformer
        self.n_timesteps = n_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Define noise schedule
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_one_minus_alphas_cumprod = torch.sqrt(1.0 / (1.0 - self.alphas_cumprod))
        self.sqrt_recip_one_minus_alphas_cumprod = torch.sqrt(1.0 / (1.0 - self.alphas_cumprod))
        
        # Initialize parameters
        self.init_parameters()
    
    def init_parameters(self):
        """Initialize model parameters."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, t, mask=None):
        """
        Forward pass of the diffusion model.
        
        Args:
            x: Input tensor of shape (batch_size, n_vars)
            t: Time step tensor of shape (batch_size, 1)
            mask: Attention mask of shape (batch_size, n_vars)
            
        Returns:
            noise: Predicted noise of shape (batch_size, n_vars)
        """
        # Add noise to input
        noise = torch.randn_like(x)
        
        # Get predicted noise
        noise_pred = self.simformer(x, t, mask)
        
        return noise_pred
    
    def sample(self, n_samples=100, n_vars=10, timesteps=None):
        """
        Sample from the diffusion model.
        
        Args:
            n_samples: Number of samples to generate
            n_vars: Number of variables
            timesteps: Number of timesteps for sampling
            
        Returns:
            samples: Generated samples of shape (n_samples, n_vars)
        """
        if timesteps is None:
            timesteps = self.n_timesteps
        
        # Initialize samples from noise
        samples = torch.randn(n_samples, n_vars)
        
        # Reverse diffusion process
        for t in range(timesteps, 0, -1):
            # Get current time step
            t_tensor = torch.full((n_samples, 1), t / timesteps)
            
            # Predict noise
            noise_pred = self.forward(samples, t_tensor)
            
            # Reverse diffusion step
            alpha_t = self.alphas[t-1]
            alpha_t_cumprod = self.alphas_cumprod[t-1]
            beta_t = self.betas[t-1]
            
            # Reverse diffusion equation
            samples = (1.0 / torch.sqrt(alpha_t)) * (samples - (beta_t / torch.sqrt(1.0 - alpha_t_cumprod)) * noise_pred)
            
            # Add noise
            if t > 1:
                noise = torch.randn_like(samples)
            else:
                noise = torch.zeros_like(samples)
            
            samples = samples + torch.sqrt(beta_t) * noise
        
        return samples

class SimformerInference:
    """
    Inference class for the Simformer model.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
    
    def infer_posterior(self, data, n_samples=1000, timesteps=100):
        """
        Infer the posterior distribution.
        
        Args:
            data: Observed data
            n_samples: Number of samples
            timesteps: Number of timesteps
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Sample from the diffusion model
        samples = self.model.sample(n_samples=n_samples, n_vars=data.shape[1], timesteps=timesteps)
        
        return samples
    
    def infer_likelihood(self, data, n_samples=1000, timesteps=100):
        """
        Infer the likelihood function.
        
        Args:
            data: Observed data
            n_samples: Number of samples
            timesteps: Number of timesteps
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Sample from the diffusion model
        samples = self.model.sample(n_samples=n_samples, n_vars=data.shape[1], timesteps=timesteps)
        
        return samples
    
    def infer_conditional(self, data, condition_vars, n_samples=1000, timesteps=100):
        """
        Infer arbitrary conditionals of the joint distribution.
        
        Args:
            data: Observed data
            condition_vars: Variables to condition on
            n_samples: Number of samples
            timesteps: Number of timesteps
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Sample from the diffusion model
        samples = self.model.sample(n_samples=n_samples, n_vars=data.shape[1], timesteps=timesteps)
        
        return samples
    
    def infer_with_guidance(self, data, constraints, n_samples=1000, timesteps=100, guidance_strength=1.0):
        """
        Infer with guided diffusion.
        
        Args:
            data: Observed data
            constraints: Constraints to apply
            n_samples: Number of samples
            timesteps: Number of timesteps
            guidance_strength: Strength of guidance
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Sample from the diffusion model with guidance
        samples = self.model.sample(n_samples=n_samples, n_vars=data.shape[1], timesteps=timesteps)
        
        # Apply guidance
        if constraints is not None:
            for constraint in constraints:
                # Apply constraint
                samples = self._apply_constraint(samples, constraint, guidance_strength)
        
        return samples
    
    def _apply_constraint(self, samples, constraint, strength):
        """Apply constraint to samples."""
        # Apply constraint
        constraint_type = constraint['type']
        constraint_value = constraint['value']
        
        if constraint_type == 'interval':
            # Apply interval constraint
            lower_bound = constraint_value[0]
            upper_bound = constraint_value[1]
            
            # Apply constraint
            samples = torch.clamp(samples, min=lower_bound, max=upper_bound)
        
        elif constraint_type == 'linear':
            # Apply linear constraint
            coefficients = constraint_value['coefficients']
            bound = constraint_value['bound']
            
            # Apply constraint
            constraint_value = torch.sum(samples * torch.tensor(coefficients), dim=1)
            constraint_value = constraint_value - bound
            constraint_value = torch.abs(constraint_value)
            constraint_value = torch.clamp(constraint_value, min=0)
            constraint_value = constraint_value * strength
            samples = samples - constraint_value.unsqueeze(1)
        
        return samples