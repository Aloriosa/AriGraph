"""
Simple linear policy implemented in PyTorch.
Policy: p_right = sigmoid(w * x + b)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearPolicy(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1) * 0.1)
        self.b = nn.Parameter(torch.randn(1) * 0.1)
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def forward(self, state: torch.Tensor):
        """
        :param state: [batch, 2] where state[:,0] = position, state[:,1] = phase
        :return: action probabilities, shape [batch, 2]
        """
        x = state[:, 0]
        logits = self.w * x + self.b
        p_right = torch.sigmoid(logits)
        probs = torch.stack([1 - p_right, p_right], dim=-1)
        return probs

    def action(self, state: torch.Tensor):
        """
        Sample action according to the policy.
        :param state: [2] tensor
        :return: action (0 or 1)
        """
        probs = self.forward(state.unsqueeze(0)).squeeze(0)
        action = torch.multinomial(probs, 1, generator=self.rng).item()
        return action, probs[action]

    def log_prob(self, state: torch.Tensor, action: int):
        probs = self.forward(state.unsqueeze(0)).squeeze(0)
        return torch.log(probs[action] + 1e-8)