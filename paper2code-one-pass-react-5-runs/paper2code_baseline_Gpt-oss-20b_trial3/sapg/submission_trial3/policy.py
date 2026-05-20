import torch
import torch.nn as nn
import torch.nn.functional as F


class SAPGPolicy(nn.Module):
    """
    Shared backbone + per-policy latent vector.
    Supports both actor and critic heads.
    """
    def __init__(self, obs_dim, act_dim, latent_dim=32, hidden_dim=128):
        super().__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        # Actor head
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        # Log std per action dimension (learnable, shared across policies)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        # Critic head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, latent):
        """
        obs: [batch, obs_dim]
        latent: [batch, latent_dim]
        """
        x = torch.cat([obs, latent], dim=-1)
        h = self.backbone(x)
        mean = self.mean_head(h)
        std = torch.exp(self.log_std)
        value = self.value_head(h).squeeze(-1)
        return mean, std, value

    def get_action(self, obs, latent):
        mean, std, value = self.forward(obs, latent)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value

    def evaluate_actions(self, obs, latent, actions):
        mean, std, value = self.forward(obs, latent)
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, value