"""Implementation of the CompoNet architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class PolicyModule(nn.Module):
    """
    A single self‑composing policy module.
    - state_dim: dimensionality of the state representation.
    - action_dim: number of discrete actions.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Output attention head: query & key/value projections
        self.q_proj = nn.Linear(state_dim, hidden_dim)
        self.k_proj = nn.Linear(action_dim, hidden_dim)
        self.v_proj = nn.Linear(action_dim, hidden_dim)

        # Internal policy MLP
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor, prev_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch, state_dim]
            prev_logits: [batch, N_prev, action_dim] or None
        Returns:
            logits: [batch, action_dim]
        """
        batch = state.size(0)

        # Output attention head
        if prev_logits is None or prev_logits.size(1) == 0:
            att_out = torch.zeros(batch, self.action_dim, device=state.device)
        else:
            q = self.q_proj(state)  # [batch, hidden]
            k = self.k_proj(prev_logits)  # [batch, N_prev, hidden]
            v = self.v_proj(prev_logits)  # [batch, N_prev, hidden]

            # Compute attention weights
            scores = torch.bmm(k, q.unsqueeze(2)).squeeze(2) / (self.hidden_dim ** 0.5)  # [batch, N_prev]
            attn = F.softmax(scores, dim=1).unsqueeze(2)  # [batch, N_prev, 1]

            # Weighted sum of values
            att_out = torch.sum(attn * v, dim=1)  # [batch, hidden]

            # Project back to action space
            att_out = self.mlp(att_out)  # reuse same MLP for projection

        # Internal policy
        internal_out = self.mlp(state)  # [batch, action_dim]

        # Final logits
        logits = att_out + internal_out
        return logits


class CompoNet(nn.Module):
    """
    A list of policy modules.  Each new task adds a new module.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.modules: List[PolicyModule] = []

    def add_module(self):
        """Instantiate and append a new policy module."""
        mod = PolicyModule(self.state_dim, self.action_dim, self.hidden_dim)
        self.modules.append(mod)
        return mod

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the chain of modules.
        Returns logits from the last module.
        """
        prev_logits = None
        for mod in self.modules:
            logits = mod(state, prev_logits)
            prev_logits = logits.unsqueeze(1)  # [batch, 1, action_dim]
        return logits  # logits from the last module