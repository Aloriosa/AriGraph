import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class TransformerRewardEncoder(nn.Module):
    """
    Transformer-based Variational Autoencoder for encoding reward functions.
    
    This model takes state-reward pairs and learns a latent representation of the reward function.
    The encoder maps state-reward pairs to a latent vector, and the decoder reconstructs rewards
    from the latent vector and states.
    """
    
    def __init__(self, state_dim: int, latent_dim: int = 128, num_layers: int = 4, 
                 num_heads: int = 8, hidden_dim: int = 256, max_seq_len: int = 100):
        super(TransformerRewardEncoder, self).__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        
        # Embedding layer for state features
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        
        # Transformer encoder for processing state-reward pairs
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Latent space parameters (for VAE)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder for reconstructing rewards
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        
    def forward(self, states: torch.Tensor, rewards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VAE.
        
        Args:
            states: Tensor of shape (batch_size, seq_len, state_dim)
            rewards: Tensor of shape (batch_size, seq_len)
            
        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
            reconstructed_rewards: Reconstructed rewards (batch_size, seq_len)
        """
        batch_size, seq_len, _ = states.shape
        
        # Combine states and rewards into input sequence
        # Shape: (batch_size, seq_len, state_dim + 1)
        state_reward_pairs = torch.cat([states, rewards.unsqueeze(-1)], dim=-1)
        
        # Embed state-reward pairs
        # Shape: (batch_size, seq_len, hidden_dim)
        embedded = self.state_embed(state_reward_pairs)
        
        # Add positional encoding
        embedded = embedded + self.pos_encoding[:, :seq_len, :]
        
        # Transformer encoder processes the sequence
        # Shape: (seq_len, batch_size, hidden_dim)
        embedded = embedded.permute(1, 0, 2)
        transformer_output = self.transformer_encoder(embedded)
        
        # Average over sequence to get global representation
        # Shape: (batch_size, hidden_dim)
        global_rep = transformer_output.mean(dim=0)
        
        # Compute latent distribution parameters
        mu = self.mu_layer(global_rep)
        logvar = self.logvar_layer(global_rep)
        
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Decode rewards for each state
        # Expand z to match sequence length
        z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate latent vector with states for reconstruction
        decoder_input = torch.cat([states, z_expanded], dim=-1)
        
        # Decode rewards
        reconstructed_rewards = self.decoder(decoder_input).squeeze(-1)
        
        return mu, logvar, reconstructed_rewards
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sample from latent distribution
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, states: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """
        Encode state-reward pairs to latent representation without sampling.
        
        Args:
            states: Tensor of shape (batch_size, seq_len, state_dim)
            rewards: Tensor of shape (batch_size, seq_len)
            
        Returns:
            Latent representation (batch_size, latent_dim)
        """
        batch_size, seq_len, _ = states.shape
        
        # Combine states and rewards into input sequence
        state_reward_pairs = torch.cat([states, rewards.unsqueeze(-1)], dim=-1)
        
        # Embed state-reward pairs
        embedded = self.state_embed(state_reward_pairs)
        
        # Add positional encoding
        embedded = embedded + self.pos_encoding[:, :seq_len, :]
        
        # Transformer encoder processes the sequence
        embedded = embedded.permute(1, 0, 2)
        transformer_output = self.transformer_encoder(embedded)
        
        # Average over sequence to get global representation
        global_rep = transformer_output.mean(dim=0)
        
        # Return mean of latent distribution (no sampling)
        mu = self.mu_layer(global_rep)
        return mu
    
    def decode(self, z: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to predict rewards for given states.
        
        Args:
            z: Latent representation (batch_size, latent_dim)
            states: States to predict rewards for (batch_size, seq_len, state_dim)
            
        Returns:
            Predicted rewards (batch_size, seq_len)
        """
        batch_size, seq_len, _ = states.shape
        
        # Expand latent vector to match sequence length
        z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate latent vector with states
        decoder_input = torch.cat([states, z_expanded], dim=-1)
        
        # Decode rewards
        rewards = self.decoder(decoder_input).squeeze(-1)
        
        return rewards