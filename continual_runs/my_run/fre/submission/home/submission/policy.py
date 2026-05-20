import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class IQLPolicy(nn.Module):
    """
    Implicit Q-Learning (IQL) policy conditioned on latent reward encodings.
    
    This policy takes as input the current state and a latent reward encoding,
    and outputs an action. The policy is trained using IQL on offline data.
    """
    
    def __init__(self, state_dim: int, latent_dim: int = 128, action_dim: int = 4, 
                 hidden_dim: int = 256):
        super(IQLPolicy, self).__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Policy network that takes state and latent reward encoding as input
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value network for IQL (predicts state value)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q-network for IQL (predicts state-action value)
        self.q_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict actions given states and latent reward encoding.
        
        Args:
            states: Current states (batch_size, state_dim)
            z: Latent reward encoding (batch_size, latent_dim)
            
        Returns:
            Predicted actions (batch_size, action_dim)
        """
        # Concatenate state and latent encoding
        policy_input = torch.cat([states, z], dim=-1)
        
        # Predict actions
        actions = self.policy_net(policy_input)
        
        return actions
    
    def get_value(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get state value estimate.
        
        Args:
            states: Current states (batch_size, state_dim)
            
        Returns:
            State value estimate (batch_size, 1)
        """
        return self.value_net(states)
    
    def get_q_value(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Get state-action value estimate.
        
        Args:
            states: Current states (batch_size, state_dim)
            actions: Actions taken (batch_size, action_dim)
            
        Returns:
            State-action value estimate (batch_size, 1)
        """
        q_input = torch.cat([states, actions], dim=-1)
        return self.q_net(q_input)
    
    def get_action(self, states: torch.Tensor, z: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get action for given state and latent encoding.
        
        Args:
            states: Current states (batch_size, state_dim)
            z: Latent reward encoding (batch_size, latent_dim)
            deterministic: Whether to use deterministic policy
            
        Returns:
            Predicted actions (batch_size, action_dim)
        """
        actions = self.forward(states, z)
        
        if deterministic:
            return actions
        else:
            # Add some noise for exploration (in training)
            noise = torch.randn_like(actions) * 0.1
            return actions + noise