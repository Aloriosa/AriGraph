"""Implementation of the self‑composing policy network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompoNet(nn.Module):
    """
    A simple compositional policy network.

    The network stores a list of MLP modules.  Each module outputs a logit
    vector for the discrete action space.  The final action logits are a
    weighted sum of all module logits, where the weights are learnable
    and shared across actions.  All modules except the most recently
    added one are frozen after the first training episode.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.modules = nn.ModuleList()
        self.weights = nn.ParameterList()

        # Create the first module
        self.add_module()

    def add_module(self):
        """Append a new trainable module."""
        mod = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.modules.append(mod)
        # Weight for combining this module's logits
        w = nn.Parameter(torch.ones(1))
        self.weights.append(w)

    def freeze_modules(self):
        """Freeze all modules except the most recent one."""
        for mod in self.modules[:-1]:
            for p in mod.parameters():
                p.requires_grad = False
        for w in self.weights[:-1]:
            w.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute action logits as a weighted sum of module logits.

        Args:
            x: Observation tensor of shape (batch, obs_dim)
        Returns:
            logits: Tensor of shape (batch, action_dim)
        """
        logits_list = []
        for mod in self.modules:
            logits_list.append(mod(x))
        logits_stack = torch.stack(logits_list, dim=0)  # (num_mod, batch, act)
        weight_stack = torch.stack(self.weights, dim=0)  # (num_mod)
        # Softmax over modules to obtain a distribution that sums to 1
        weight_norm = torch.softmax(weight_stack, dim=0)  # (num_mod)
        # Reshape for broadcasting
        weight_norm = weight_norm.view(-1, 1, 1)  # (num_mod, 1, 1)
        # Weighted sum over modules
        combined = torch.sum(logits_stack * weight_norm, dim=0)  # (batch, act)
        return combined