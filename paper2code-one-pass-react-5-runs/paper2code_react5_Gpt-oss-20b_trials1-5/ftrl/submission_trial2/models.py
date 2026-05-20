import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNet(nn.Module):
    """
    A simple two‑layer MLP that outputs policy logits and a scalar value.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value