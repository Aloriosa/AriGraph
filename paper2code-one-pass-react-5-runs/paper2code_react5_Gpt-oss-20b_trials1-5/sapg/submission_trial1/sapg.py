"""
Core implementation of SAPG (Split and Aggregate Policy Gradients).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class SharedMLP(nn.Module):
    """Shared backbone for policy and value networks."""
    def __init__(self, obs_dim, latent_dim=16, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        in_dim = obs_dim + latent_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PolicyHead(nn.Module):
    """Head that outputs mean and log_std for Gaussian policy."""
    def __init__(self, hidden_size, act_dim):
        super().__init__()
        self.mean = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        mean = self.mean(x)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std


class ValueHead(nn.Module):
    """Value network head."""
    def __init__(self, hidden_size):
        super().__init__()
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.value(x).squeeze(-1)


class SAPGAgent:
    """
    SAPG agent with M policies (leader + M-1 followers).
    Each policy shares a backbone but has its own latent vector.
    """
    def __init__(
        self,
        obs_dim,
        act_dim,
        M=4,
        horizon=64,
        gamma=0.99,
        tau=0.95,
        lr=5e-4,
        eps_clip=0.2,
        lambda_off=1.0,
        entropy_coef=None,
        device="cpu",
        latent_dim=16,
        hidden_sizes=(64, 64),
    ):
        self.M = M
        self.horizon = horizon
        self.gamma = gamma
        self.tau = tau
        self.eps_clip = eps_clip
        self.lambda_off = lambda_off
        self.device = device

        # Shared backbone
        self.backbone = SharedMLP(obs_dim, latent_dim, hidden_sizes).to(device)

        # Separate latent vectors for each policy
        self.latents = nn.ParameterList([
            nn.Parameter(torch.randn(latent_dim)) for _ in range(M)
        ])

        # Policy and value heads for each policy
        hidden_size = hidden_sizes[-1]
        self.policy_heads = nn.ModuleList([
            PolicyHead(hidden_size, act_dim).to(device) for _ in range(M)
        ])
        self.value_heads = nn.ModuleList([
            ValueHead(hidden_size).to(device) for _ in range(M)
        ])

        # Optimizer
        params = list(self.backbone.parameters()) + \
                 list(self.policy_heads.parameters()) + \
                 list(self.value_heads.parameters()) + \
                 list(self.latents.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        # Entropy coefficients (leader gets 0)
        if entropy_coef is None:
            entropy_coef = [0.0] * M
        self.entropy_coef = entropy_coef

    def act(self, obs, policy_idx=0, deterministic=False):
        """
        Sample action from policy `policy_idx` given observation.
        `obs` is a torch tensor of shape (batch, obs_dim).
        Returns (actions, log_probs, values)
        """
        obs = obs.to(self.device)
        latent = self.latents[policy_idx]
        x = torch.cat([obs, latent.expand(obs.size(0), -1)], dim=-1)
        h = self.backbone(x)
        mean, log_std = self.policy_heads[policy_idx](h)
        std = log_std.exp()

        if deterministic:
            action = mean
        else:
            action = torch.normal(mean, std)

        # Log probability of the action
        log_prob = (-0.5 * (((action - mean) / std) ** 2 + 2 * log_std + np.log(2 * np.pi))).sum(-1)

        value = self.value_heads[policy_idx](h)
        return action, log_prob, value

    def _compute_gae(self, rewards, values, dones):
        """
        Compute GAE advantages and returns.
        All inputs are torch tensors.
        """
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.tau * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def _train_on_policy(self, batch, policy_idx):
        """
        Perform one gradient step on an on‑policy batch for policy `policy_idx`.
        """
        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.float32, device=self.device)
        old_logp = torch.tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(batch["advantages"], dtype=torch.float32, device=self.device)

        num_samples = states.size(0)
        perm = torch.randperm(num_samples)
        mini_batch_size = 256

        for start in range(0, num_samples, mini_batch_size):
            end = start + mini_batch_size
            idx = perm[start:end]

            s = states[idx]
            a = actions[idx]
            old_logp_i = old_logp[idx]
            ret = returns[idx]
            adv = advantages[idx]

            latent = self.latents[policy_idx]
            x = torch.cat([s, latent.expand(s.size(0), -1)], dim=-1)
            h = self.backbone(x)

            mean, log_std = self.policy_heads[policy_idx](h)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            logp = dist.log_prob(a).sum(-1)

            ratio = torch.exp(logp - old_logp_i)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            val = self.value_heads[policy_idx](h)
            val_loss = F.mse_loss(val, ret)

            entropy = dist.entropy().sum(-1).mean()
            entropy_loss = -self.entropy_coef[policy_idx] * entropy

            loss = policy_loss + 0.5 * val_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.policy_heads[policy_idx].parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_heads[policy_idx].parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.latents, 0.5)
            self.optimizer.step()

    def _train_off_policy_leader(self, leader_on, off_data):
        """
        Leader update that includes off‑policy samples from followers.
        `leader_on` is the on‑policy batch for the leader.
        `off_data` is a list of follower batches.
        """
        # Concatenate off-policy samples
        states_off = np.concatenate([d["states"] for d in off_data], axis=0)
        actions_off = np.concatenate([d["actions"] for d in off_data], axis=0)
        old_logp_off = np.concatenate([d["log_probs"] for d in off_data], axis=0)
        rewards_off = np.concatenate([d["rewards"] for d in off_data], axis=0)
        dones_off = np.concatenate([d["dones"] for d in off_data], axis=0)

        # Compute leader's value estimates for off-policy states
        states_off_tensor = torch.tensor(states_off, dtype=torch.float32, device=self.device)
        latent = self.latents[0]
        x_off = torch.cat([states_off_tensor, latent.expand(states_off_tensor.size(0), -1)], dim=-1)
        h_off = self.backbone(x_off)
        values_off = self.value_heads[0](h_off).detach()

        # Bootstrap next value as 0 (simplification)
        values_off_np = values_off.cpu().numpy()
        values_off_np = np.concatenate([values_off_np, [0.0]])

        rewards_off_tensor = torch.tensor(rewards_off, dtype=torch.float32, device=self.device)
        dones_off_tensor = torch.tensor(dones_off, dtype=torch.float32, device=self.device)
        values_off_tensor = torch.tensor(values_off_np, dtype=torch.float32, device=self.device)

        adv_off, ret_off = self._compute_gae(rewards_off_tensor, values_off_tensor, dones_off_tensor)

        # Compute log probabilities under leader's current policy
        mean_leader, log_std_leader = self.policy_heads[0](h_off)
        std_leader = log_std_leader.exp()
        dist_leader = torch.distributions.Normal(mean_leader, std_leader)
        logp_new_off = dist_leader.log_prob(torch.tensor(actions_off, dtype=torch.float32, device=self.device)).sum(-1)

        # Importance sampling ratio
        ratio = torch.exp(logp_new_off - torch.tensor(old_logp_off, dtype=torch.float32, device=self.device))

        # μ scaling factor (here approximated as 1.0)
        mu = 1.0
        clip_low = mu * (1 - self.eps_clip)
        clip_high = mu * (1 + self.eps_clip)

        surr1 = ratio * adv_off
        surr2 = torch.clamp(ratio, clip_low, clip_high) * adv_off
        off_policy_loss = -torch.min(surr1, surr2).mean()

        # On‑policy loss for leader
        # Re‑compute on‑policy loss in the same mini‑batch loop
        states = torch.tensor(leader_on["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(leader_on["actions"], dtype=torch.float32, device=self.device)
        old_logp = torch.tensor(leader_on["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.tensor(leader_on["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(leader_on["advantages"], dtype=torch.float32, device=self.device)

        num_samples = states.size(0)
        perm = torch.randperm(num_samples)
        mini_batch_size = 256

        for start in range(0, num_samples, mini_batch_size):
            end = start + mini_batch_size
            idx = perm[start:end]

            s = states[idx]
            a = actions[idx]
            old_logp_i = old_logp[idx]
            ret = returns[idx]
            adv = advantages[idx]

            latent = self.latents[0]
            x = torch.cat([s, latent.expand(s.size(0), -1)], dim=-1)
            h = self.backbone(x)

            mean, log_std = self.policy_heads[0](h)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            logp = dist.log_prob(a).sum(-1)

            ratio_on = torch.exp(logp - old_logp_i)

            surr1_on = ratio_on * adv
            surr2_on = torch.clamp(ratio_on, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            policy_loss = -torch.min(surr1_on, surr2_on).mean()

            val = self.value_heads[0](h)
            val_loss = F.mse_loss(val, ret)

            entropy = dist.entropy().sum(-1).mean()
            entropy_loss = -self.entropy_coef[0] * entropy

            # Combine on‑policy and off‑policy losses
            loss = policy_loss + 0.5 * val_loss + entropy_loss + self.lambda_off * off_policy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.policy_heads[0].parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_heads[0].parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.latents, 0.5)
            self.optimizer.step()

    def update(self, all_data):
        """
        Update agent given data from all M policy blocks.
        `all_data` is a list of dicts, one per policy block.
        """
        # Compute GAE for each block
        for data in all_data:
            # Append a bootstrap value of 0 for the last step
            values = np.concatenate([data["values"], [0.0]])  # add last value as 0
            rewards = data["rewards"]
            dones = data["dones"]
            values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
            advantages, returns = self._compute_gae(rewards_tensor, values_tensor, dones_tensor)
            data["advantages"] = advantages.detach().cpu().numpy()
            data["returns"] = returns.detach().cpu().numpy()

        # Leader update (policy 0) with off‑policy data from others
        leader_on = all_data[0]
        off_data = all_data[1:]  # followers
        self._train_off_policy_leader(leader_on, off_data)

        # Followers update (on‑policy only)
        for j in range(1, self.M):
            self._train_on_policy(all_data[j], policy_idx=j)

    def get_leader_state_dict(self):
        """
        Return state dict of the leader policy (index 0) for saving.
        """
        return {
            "backbone": self.backbone.state_dict(),
            "policy_head": self.policy_heads[0].state_dict(),
            "value_head": self.value_heads[0].state_dict(),
            "latent": self.latents[0].detach().cpu().clone(),
        }