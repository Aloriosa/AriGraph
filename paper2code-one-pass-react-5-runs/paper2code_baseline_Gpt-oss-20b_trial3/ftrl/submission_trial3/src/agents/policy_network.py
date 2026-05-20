import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """
    Simple linear policy: logits = w * obs + b
    Output: probability of action 1 (right); action 0 probability = 1 - p
    """
    def __init__(self, obs_dim, action_dim=2):
        super().__init__()
        assert action_dim == 2, "Only binary actions supported"
        self.linear = nn.Linear(obs_dim, 1)

    def forward(self, obs):
        logits = self.linear(obs)  # shape (..., 1)
        probs = torch.sigmoid(logits).squeeze(-1)  # shape (...)
        return probs

    def act(self, obs):
        with torch.no_grad():
            probs = self.forward(obs)
            return torch.bernoulli(probs).long()

    def sample_action(self, obs):
        probs = self.forward(obs)
        return torch.bernoulli(probs).long().item()