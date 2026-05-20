import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedPolicy(nn.Module):
    """
    Shared backbone + conditioning vector.
    The policy outputs the mean of a Gaussian action distribution.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=64, cond_dim=32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        # Fixed log std (learnable but independent of obs)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs, cond):
        x = torch.cat([obs, cond], dim=-1)
        x = self.shared(x)
        mean = self.mean_head(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_log_prob(self, obs, act, cond):
        mean, std = self.forward(obs, cond)
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(act).sum(-1)
        return logp