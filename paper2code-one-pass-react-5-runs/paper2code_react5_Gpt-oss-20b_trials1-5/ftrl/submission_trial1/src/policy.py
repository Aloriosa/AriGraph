# src/policy.py
"""
Actor‑Critic network for AppleRetrieval.
The policy outputs logits for 2 discrete actions.
The critic outputs a scalar value estimate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=2, hidden_dim=64):
        super().__init__()
        # Shared encoder
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Policy head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 2),
        )
        # Value head
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        """Return action logits and value."""
        h = self.net(obs)
        logits = self.actor(h)
        value = self.value(h).squeeze(-1)
        return logits, value

    def get_action(self, obs, deterministic=False):
        """Sample or choose the best action."""
        logits, _ = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp

    def get_value(self, obs):
        _, value = self.forward(obs)
        return value

    def get_log_prob(self, obs, actions):
        logits, _ = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions)

    def get_probs(self, obs):
        logits, _ = self.forward(obs)
        return F.softmax(logits, dim=-1)