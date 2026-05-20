"""
CompoNet implementation for a toy continual RL setting.
Each module is a small neural network that attends to the outputs
of all previously frozen modules and to its own internal policy.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CompoModule(nn.Module):
    """
    A single self‑composing policy module.

    Args:
        state_dim (int): dimensionality of the state representation.
        action_dim (int): number of discrete actions.
        hidden_dim (int): hidden size for attention and internal policy.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Output attention head
        self.w_q_out = nn.Linear(state_dim, hidden_dim, bias=False)
        self.w_k_out = nn.Linear(action_dim, hidden_dim, bias=False)

        # Input attention head
        self.w_q_in = nn.Linear(state_dim, hidden_dim, bias=False)
        self.w_k_in = nn.Linear(action_dim * 2, hidden_dim, bias=False)
        self.w_v_in = nn.Linear(action_dim * 2, action_dim, bias=False)

        # Internal policy (MLP)
        self.internal = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor,
                prev_outputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for a single module.

        Args:
            state: (state_dim,)
            prev_outputs: (N_prev, action_dim) or None

        Returns:
            logits: (action_dim,)
        """
        if prev_outputs is None or prev_outputs.size(0) == 0:
            out_head = torch.zeros(self.action_dim, device=state.device)
            inp_head = torch.zeros(self.action_dim, device=state.device)
        else:
            # ---------- Output attention ----------
            q_out = self.w_q_out(state)  # (hidden_dim,)
            k_out = self.w_k_out(prev_outputs)  # (N_prev, hidden_dim)
            attn_out = F.softmax(
                q_out @ k_out.t() / math.sqrt(self.hidden_dim), dim=1
            )  # (N_prev,)
            out_head = attn_out @ prev_outputs  # (action_dim,)

            # ---------- Input attention ----------
            # concatenate prev_outputs (N_prev, action_dim) with out_head repeated
            repeated_out_head = out_head.unsqueeze(0).repeat(prev_outputs.size(0), 1)
            keys_in = torch.cat([prev_outputs, repeated_out_head], dim=1)  # (N_prev, 2*action_dim)
            values_in = self.w_v_in(keys_in)  # (N_prev, action_dim)
            q_in = self.w_q_in(state)  # (hidden_dim,)
            k_in = self.w_k_in(keys_in)  # (N_prev, hidden_dim)
            attn_in = F.softmax(
                q_in @ k_in.t() / math.sqrt(self.hidden_dim), dim=1
            )  # (N_prev,)
            inp_head = attn_in @ values_in  # (action_dim,)

        # ---------- Internal policy ----------
        internal_in = torch.cat([state, inp_head], dim=0)  # (state_dim + action_dim,)
        internal_out = self.internal(internal_in)  # (action_dim,)

        # Final output: residual connection from output head
        logits = out_head + internal_out
        return logits


class CompoNet(nn.Module):
    """
    Growing network that manages a list of CompoModule instances.
    Each new task adds a new module; all previous modules are frozen.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.modules: List[CompoModule] = []
        self.hidden_dim = hidden_dim

    def add_module(self):
        """Create a new module and freeze all previous ones."""
        for m in self.modules:
            for p in m.parameters():
                p.requires_grad = False
        new_mod = CompoModule(self.state_dim, self.action_dim, self.hidden_dim)
        self.modules.append(new_mod)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that sequentially processes all modules.
        Returns the logits of the *last* module (the current task).
        """
        prev_outputs: Optional[torch.Tensor] = None
        for mod in self.modules:
            logits = mod(state, prev_outputs)
            out = F.softmax(logits, dim=0)  # (action_dim,)
            if prev_outputs is None:
                prev_outputs = out.unsqueeze(0)
            else:
                prev_outputs = torch.cat([prev_outputs, out.unsqueeze(0)], dim=0)
        return logits  # logits of the last module


def get_policy(net: CompoNet, device: torch.device):
    """Wrap the network in a callable that returns action and log_prob."""
    def policy(state_np):
        state = torch.tensor(state_np, dtype=torch.float32, device=device)
        logits = net(state)
        probs = F.softmax(logits, dim=0)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return int(action.item()), m.log_prob(action).item()
    return policy