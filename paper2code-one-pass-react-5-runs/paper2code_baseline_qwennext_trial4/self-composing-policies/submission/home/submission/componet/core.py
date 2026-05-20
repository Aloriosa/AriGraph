"""
Core CompoNet implementation with self-composing policy modules
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SelfComposingPolicyModule(nn.Module):
    """
    Self-composing policy module for CompoNet architecture.
    
    Implements the three main components of the self-composing policy module:
    1. Output attention head: proposes an output based on preceding policies and current state
    2. Input attention head: retrieves relevant information from previous policies and output head
    3. Internal policy: adjusts the tentative output from the output attention head
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 model_dim: int = 256,
                 use_layer_norm: bool = True):
        """
        Initialize the self-composing policy module.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            model_dim: Dimension of the model hidden state
            use_layer_norm: Whether to use layer normalization
        """
        super(SelfComposingPolicyModule, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_dim = model_dim
        self.use_layer_norm = use_layer_norm
        
        # Output attention head parameters
        self.W_out_Q = nn.Linear(state_dim, model_dim)
        self.W_out_K = nn.Linear(action_dim, model_dim)
        
        # Input attention head parameters
        self.W_in_Q = nn.Linear(state_dim, model_dim)
        self.W_in_K = nn.Linear(action_dim, model_dim)
        self.W_in_V = nn.Linear(action_dim, model_dim)
        
        # Internal policy parameters
        self.internal_policy = nn.Sequential(
            nn.Linear(state_dim + action_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, action_dim)
        )
        
        # Layer normalization
        if self.use_layer_norm:
            self.norm = nn.LayerNorm(model_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, 
                state: torch.Tensor, 
                prev_outputs: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through the self-composing policy module.
        
        Args:
            state: Current state tensor [batch_size, state_dim]
            prev_outputs: Outputs from previous modules [batch_size, num_prev_modules, action_dim]
        
        Returns:
            output: Final output tensor [batch_size, action_dim]
            attention_info: Dictionary with attention weights for analysis
        """
        batch_size = state.shape[0]
        num_prev_modules = prev_outputs.shape[1] if prev_outputs.dim() > 2 else 0
        
        # Output attention head
        q_out = self.W_out_Q(state)  # [batch_size, model_dim]
        k_out = self.W_out_K(prev_outputs)  # [batch_size, num_prev_modules, model_dim]
        # Compute attention weights
        attn_scores_out = torch.bmm(q_out.unsqueeze(1), k_out.transpose(1, 2))  # [batch_size, 1, num_prev_modules]
        attn_weights_out = F.softmax(attn_scores_out, dim=-1)  # [batch_size, 1, num_prev_modules]
        # Compute output
        output_head_out = torch.bmm(attn_weights_out, prev_outputs)  # [batch_size, 1, action_dim]
        output_head_out = output_head_out.squeeze(1)  # [batch_size, action_dim]
        
        # Input attention head
        q_in = self.W_in_Q(state)  # [batch_size, model_dim]
        # Concatenate output head output with previous outputs
        attn_input = torch.cat([output_head_out.unsqueeze(1), prev_outputs], dim=1)  # [batch_size, num_prev_modules+1, action_dim]
        k_in = self.W_in_K(attn_input)  # [batch_size, num_prev_modules+1, model_dim]
        v_in = self.W_in_V(attn_input)  # [batch_size, num_prev_modules+1, model_dim]
        # Compute attention weights
        attn_scores_in = torch.bmm(q_in.unsqueeze(1), k_in.transpose(1, 2))  # [batch_size, 1, num_prev_modules+1]
        attn_weights_in = F.softmax(attn_scores_in, dim=-1)  # [batch_size, 1, num_prev_modules+1]
        # Compute input attention output
        input_head_out = torch.bmm(attn_weights_in, v_in)  # [batch, 1, model_dim]
        input_head_out = input_head_out.squeeze(1)  # [batch, model_dim]
        
        # Internal policy
        internal_input = torch.cat([state, output_head_out], dim=1)  # [batch_size, state_dim + action_dim]
        internal_output = self.internal_policy(internal_input)  # [batch_size, action_dim]
        
        # Final output: add internal policy output to output head output
        final_output = output_head_out + internal_output  # [batch_size, action_dim]
        
        # Return final output and attention information
        attention_info = {
            'output_attention_weights': attn_weights_out.squeeze(1),  # [batch_size, num_prev_modules]
            'input_attention_weights': attn_weights_in.squeeze(1)  # [batch_size, num_prev_modules+1]
        }
        
        return final_output, attention_info

class CompoNet(nn.Module):
    """
    CompoNet: Composable Network for Continual Learning
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 model_dim: int = 256,
                 use_layer_norm: bool = True):
        """
        Initialize CompoNet.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            model_dim: Dimension of the model hidden state
            use_layer_norm: Whether to use layer normalization
        """
        super(CompoNet, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_dim = model_dim
        self.use_layer_norm = use_layer_norm
        
        # Create the first module
        self.modules = nn.ModuleList([SelfComposingPolicyModule(state_dim, action_dim, model_dim, use_layer_norm)]
        )
        
        # For state encoding (if needed)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def add_module(self):
        """Add a new module to the network"""
        new_module = SelfComposingPolicyModule(self.state_dim, self.action_dim, self.model_dim, self.use_layer_norm)
        self.modules.append(new_module)
        return len(self.modules) - 1
    
    def forward(self, state: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through CompoNet.
        
        Args:
            state: Current state tensor [batch_size, state_dim]
            task_id: Current task ID
        """
        batch_size = state.shape[0]
        
        # Encode state
        encoded_state = self.state_encoder(state)
        
        # Get output from current module
        if task_id == 0:
            # First module
            output, attention_info = self.modules[0](state, torch.empty(batch_size, 0, self.action_dim))
        else:
            # Get outputs from previous modules
            prev_outputs = torch.zeros(batch_size, task_id, self.action_dim)
        for i in range(task_id):
            prev_outputs[:, i, :] = self.modules[i](state, torch.empty(batch_size, 0, self.action=0))
        
        # Get output from current module
        output, attention_info = self.modules[task_id](state, prev_outputs)
        
        return output, attention_info