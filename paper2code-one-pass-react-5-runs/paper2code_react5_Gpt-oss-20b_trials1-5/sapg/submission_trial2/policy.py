import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """
    Simple MLP policy with shared mean network and separate learnable log_std.
    Supports Gaussian action distributions.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ELU())
            last_dim = h
        self.mean_net = nn.Sequential(*layers, nn.Linear(last_dim, act_dim))
        # Log std is a learnable parameter per action dimension
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(self, obs, deterministic=False):
        """Sample action and return log probability."""
        mean, std = self.forward(obs)
        if deterministic:
            action = mean
            logp = torch.zeros_like(mean)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
            logp = dist.log_prob(action).sum(dim=-1)
        return action, logp

class ValueNetwork(nn.Module):
    """Value function approximator."""
    def __init__(self, obs_dim, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ELU())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.value_net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.value_net(obs).squeeze(-1)