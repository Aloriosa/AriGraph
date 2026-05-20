"""
Simple offline RL using IQL-style updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import random


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, action_dim: int,
                 hidden_dims=[512,512,512]):
        super().__init__()
        dims = [state_dim + latent_dim + action_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor, z: torch.Tensor):
        inp = torch.cat([s, z, a], dim=-1)
        return self.net(inp)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int,
                 hidden_dims=[512,512,512]):
        super().__init__()
        dims = [state_dim + latent_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, z: torch.Tensor):
        inp = torch.cat([s, z], dim=-1)
        return self.net(inp)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, action_dim: int,
                 hidden_dims=[512,512,512]):
        super().__init__()
        dims = [state_dim + latent_dim] + hidden_dims + [action_dim]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, z: torch.Tensor):
        inp = torch.cat([s, z], dim=-1)
        return self.net(inp)  # mean action

class IQLAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 latent_dim: int,
                 device: torch.device = torch.device("cpu"),
                 gamma: float = 0.99,
                 lr: float = 3e-4,
                 expectile: float = 0.8):
        self.device = device
        self.gamma = gamma

        self.q1 = QNetwork(state_dim, latent_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, latent_dim, action_dim).to(device)
        self.value = ValueNetwork(state_dim, latent_dim).to(device)
        self.policy = PolicyNetwork(state_dim, latent_dim, action_dim).to(device)

        self.q1_target = QNetwork(state_dim, latent_dim, action_dim).to(device)
        self.q2_target = QNetwork(state_dim, latent_dim, action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=lr)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.expectile = expectile

    @torch.no_grad()
    def update_targets(self, tau=0.005):
        for p_t, p_s in zip(self.q1_target.parameters(), self.q1.parameters()):
            p_t.data.copy_(tau * p_s.data + (1 - tau) * p_t.data)
        for p_t, p_s in zip(self.q2_target.parameters(), self.q2.parameters()):
            p_t.data.copy_(tau * p_s.data + (1 - tau) * p_t.data)

    def train_step(self,
                   s: torch.Tensor,
                   a: torch.Tensor,
                   r: torch.Tensor,
                   s_next: torch.Tensor,
                   z: torch.Tensor):
        """
        One training step for IQL.
        s, a, r, s_next: (B, ...). z: (B, latent_dim)
        """
        with torch.no_grad():
            # Target Q
            a_next = self.policy(s_next, z)
            q1_next = self.q1_target(s_next, a_next, z).squeeze(-1)
            q2_next = self.q2_target(s_next, a_next, z).squeeze(-1)
            q_next = torch.min(q1_next, q2_next)
            target_q = r + self.gamma * q_next

        # Q loss
        q1_pred = self.q1(s, a, z).squeeze(-1)
        q2_pred = self.q2(s, a, z).squeeze(-1)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        # Value loss (expectile regression)
        v_pred = self.value(s, z).squeeze(-1)
        q_tilde = torch.min(q1_pred, q2_pred)
        weight = (q_tilde > v_pred).float()
        value_loss = (weight * (v_pred - q_tilde).pow(2) +
                      (1 - weight) * (v_pred - q_tilde).pow(2) * self.expectile).mean()

        # Policy loss (advantage weighting)
        advantage = q1_pred - v_pred
        # Advantage weighting: exp(advantage)
        weight_policy = torch.exp(advantage).clamp(max=10.0)
        policy_loss = -(advantage * weight_policy).mean()

        # Optimize
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.update_targets()