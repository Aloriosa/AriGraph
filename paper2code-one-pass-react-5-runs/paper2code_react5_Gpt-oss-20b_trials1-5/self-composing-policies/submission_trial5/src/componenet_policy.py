# src/componenet_policy.py
"""
Implementation of the CompoNet architecture as described in the paper.
Each module is a self‑composing policy that can attend to the outputs
of all previously frozen modules.
"""

import torch
import torch.nn as nn
import math
from typing import List


class SelfComposeModule(nn.Module):
    """
    A single self‑composing policy module.
    """
    def __init__(self, state_dim: int, action_dim: int, d_model: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Output attention head
        self.W_q_out = nn.Linear(state_dim, d_model, bias=False)
        self.W_k_out = nn.Linear(action_dim, d_model, bias=False)

        # Input attention head
        self.W_q_in = nn.Linear(state_dim, d_model, bias=False)
        self.W_k_in = nn.Linear(action_dim, d_model, bias=False)
        self.W_v_in = nn.Linear(action_dim, d_model, bias=False)

        # Internal policy (MLP)
        self.internal_policy = nn.Sequential(
            nn.Linear(state_dim + d_model, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state: torch.Tensor, prev_logits: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the logits for the current module.

        Args:
            state: Tensor of shape (state_dim,)
            prev_logits: list of tensors of shape (action_dim,) from previous modules
        Returns:
            logits: Tensor of shape (action_dim,)
        """
        if len(prev_logits) == 0:
            # No previous modules – just use the internal policy
            return self.internal_policy(state)

        # Stack previous logits -> (n_prev, action_dim)
        Phi = torch.stack(prev_logits, dim=0)  # shape: (n, A)

        # Output attention head
        q_out = self.W_q_out(state)          # (d_model,)
        K_out = self.W_k_out(Phi)            # (n, d_model)
        attn_out = torch.softmax(
            q_out @ K_out.t() / math.sqrt(self.d_model), dim=-1
        )  # (n,)
        v = attn_out @ Phi                  # (action_dim,)

        # Input attention head
        P = torch.cat([v.unsqueeze(0), Phi], dim=0)  # (n+1, A)
        q_in = self.W_q_in(state)                   # (d_model,)
        K_in = self.W_k_in(P)                       # (n+1, d_model)
        V_in = self.W_v_in(P)                       # (n+1, d_model)
        attn_in = torch.softmax(
            q_in @ K_in.t() / math.sqrt(self.d_model), dim=-1
        )  # (n+1,)
        context = attn_in @ V_in                  # (d_model,)

        # Internal policy
        internal = self.internal_policy(
            torch.cat([state, context], dim=-1)
        )  # (action_dim,)

        logits = v + internal
        return logits


class CompoNet(nn.Module):
    """
    CompoNet: a stack of self‑composing modules.
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.modules = nn.ModuleList()

    def add_module(self) -> SelfComposeModule:
        """
        Add a new module to the network and return it.
        """
        module = SelfComposeModule(self.state_dim, self.action_dim)
        self.modules.append(module)
        return module

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all modules. The output of the last module
        is returned as the action logits.
        """
        prev_logits = []
        for module in self.modules:
            logits = module(state, prev_logits)
            prev_logits.append(logits)
        return prev_logits[-1]