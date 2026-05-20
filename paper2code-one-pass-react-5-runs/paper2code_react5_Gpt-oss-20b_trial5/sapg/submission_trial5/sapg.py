"""
SAPG – Split and Aggregate Policy Gradients
===========================================

This module implements the core algorithmic components of SAPG:
- Shared backbone with per‑policy latent conditioning
- Leader‑follower aggregation scheme
- On‑policy and off‑policy clipped surrogate loss
- GAE advantage estimation
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------- #
#          Utility helpers (seeding, device, etc.)                #
# --------------------------------------------------------------- #
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------- #
#                   Policy / Value Network                        #
# --------------------------------------------------------------- #
class SAPGActorCritic(nn.Module):
    """
    Shared backbone with per‑policy latent vector.
    Supports continuous actions (Gaussian policy).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        latent_dim: int = 32,
        num_policies: int = 4,
        hidden_sizes: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.num_policies = num_policies
        self.latent_dim = latent_dim

        # Shared backbone
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ELU())
            last_dim = h
        self.backbone = nn.Sequential(*layers)

        # Latent vectors (one per policy)
        self.latents = nn.Parameter(
            torch.randn(num_policies, latent_dim)
        )  # [M, latent_dim]

        # Output heads
        self.mean_head = nn.Linear(last_dim + latent_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # shared log_std
        self.value_head = nn.Linear(last_dim + latent_dim, 1)

    def forward(
        self, obs: torch.Tensor, policy_idx: int
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Returns action distribution and state value for the given policy.
        """
        h = self.backbone(obs)
        latent = self.latents[policy_idx]
        h = torch.cat([h, latent], dim=-1)
        mean = self.mean_head(h)
        log_std = self.log_std.expand_as(mean)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        value = self.value_head(h).squeeze(-1)
        return dist, value

    def act(
        self, obs: torch.Tensor, policy_idx: int
    ) -> Tuple[np.ndarray, float, float, torch.Tensor]:
        """
        Sample an action and return it along with log_prob and value.
        """
        dist, value = self.forward(obs, policy_idx)
        action = dist.rsample()
        logp = dist.log_prob(action).sum(-1)
        return (
            action.cpu().numpy(),
            logp.item(),
            value.item(),
            dist,
        )

    def log_prob(
        self, obs: torch.Tensor, actions: torch.Tensor, policy_idx: int
    ) -> torch.Tensor:
        dist, _ = self.forward(obs, policy_idx)
        return dist.log_prob(actions).sum(-1)


# --------------------------------------------------------------- #
#                GAE Advantage Estimation                         #
# --------------------------------------------------------------- #
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE advantages and discounted returns.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(T)):
        delta = (
            rewards[t]
            + gamma * values[t + 1] * (1.0 - dones[t])
            - values[t]
        )
        advantages[t] = last_adv = delta + gamma * lam * (1.0 - dones[t]) * last_adv
    returns = advantages + values[:-1]
    return advantages, returns


# --------------------------------------------------------------- #
#                    SAPG training loop                           #
# --------------------------------------------------------------- #
@dataclass
class Transition:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray
    logp: np.ndarray
    values: np.ndarray


class SAPGAgent:
    """
    SAPG training agent handling multiple policies,
    leader‑follower aggregation, and off‑policy updates.
    """

    def __init__(
        self,
        envs: List[gym.vector.SyncVectorEnv],
        model: SAPGActorCritic,
        policy_idx: int,
        steps_per_update: int,
        gamma: float,
        lam: float,
        eps_clip: float,
        lambda_off: float,
    ):
        self.envs = envs
        self.model = model
        self.policy_idx = policy_idx
        self.steps_per_update = steps_per_update
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.lambda_off = lambda_off
        self.num_envs = self.envs.num_envs

        # Storage buffers (list of transitions)
        self.reset_buffers()

    def reset_buffers(self):
        self.obs_buf = []
        self.action_buf = []
        self.reward_buf = []
        self.next_obs_buf = []
        self.done_buf = []
        self.logp_buf = []
        self.value_buf = []

    def rollout(self):
        """Collect a rollout of `steps_per_update` steps."""
        self.reset_buffers()
        obs, _ = self.envs.reset()
        for _ in range(self.steps_per_update):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                dist, value = self.model.forward(obs_t, self.policy_idx)
                action = dist.sample().cpu().numpy()
                logp = dist.log_prob(action).sum(-1).cpu().numpy()
                value = value.cpu().numpy()

            next_obs, reward, terminated, truncated, info = self.envs.step(action)
            done = np.logical_or(terminated, truncated)

            # Store
            self.obs_buf.append(obs)
            self.action_buf.append(action)
            self.reward_buf.append(reward)
            self.next_obs_buf.append(next_obs)
            self.done_buf.append(done)
            self.logp_buf.append(logp)
            self.value_buf.append(value)

            obs = next_obs

        # Convert to numpy arrays
        self.obs_buf = np.concatenate(self.obs_buf, axis=0)
        self.action_buf = np.concatenate(self.action_buf, axis=0)
        self.reward_buf = np.concatenate(self.reward_buf, axis=0)
        self.next_obs_buf = np.concatenate(self.next_obs_buf, axis=0)
        self.done_buf = np.concatenate(self.done_buf, axis=0)
        self.logp_buf = np.concatenate(self.logp_buf, axis=0)
        self.value_buf = np.concatenate(self.value_buf, axis=0)

    def compute_advantages(self):
        """Compute GAE advantages and returns."""
        # Append bootstrap value for the final state
        next_value = torch.as_tensor(
            self.model.forward(
                torch.as_tensor(
                    self.next_obs_buf[-1], dtype=torch.float32, device=device
                ),
                self.policy_idx,
            )[1].item(),
            dtype=torch.float32,
            device=device,
        ).cpu().numpy()

        values = np.concatenate([self.value_buf, [next_value]], axis=0)
        adv, returns = compute_gae(
            self.reward_buf, values, self.done_buf, self.gamma, self.lam
        )
        self.advantages = adv
        self.returns = returns

    def on_policy_loss(self, logp_new, adv):
        """Clipped surrogate (on‑policy)."""
        ratio = torch.exp(logp_new - logp_new)  # ratio == 1
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv
        loss = -torch.min(surr1, surr2).mean()
        return loss

    def off_policy_loss(self, logp_new, logp_old, adv):
        """Clipped surrogate (off‑policy) with μ = 1."""
        ratio = torch.exp(logp_new - logp_old)
        surr1 = ratio * adv
        surr2 = torch.clamp(
            ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip
        ) * adv
        loss = -torch.min(surr1, surr2).mean()
        return loss

    def update(
        self,
        leader_off_policy_data: List[Transition] = None,
    ):
        """Perform a policy and value update."""
        # Convert buffers to tensors
        obs = torch.as_tensor(self.obs_buf, dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.action_buf, dtype=torch.float32, device=device)
        advantages = torch.as_tensor(self.advantages, dtype=torch.float32, device=device)
        returns = torch.as_tensor(self.returns, dtype=torch.float32, device=device)
        old_logp = torch.as_tensor(self.logp_buf, dtype=torch.float32, device=device)

        # Current log prob
        dist, value = self.model.forward(obs, self.policy_idx)
        logp = dist.log_prob(actions).sum(-1)

        # On‑policy loss
        loss_policy = self.on_policy_loss(logp, advantages)

        # Off‑policy loss (if leader)
        if leader_off_policy_data is not None:
            off_losses = []
            for data in leader_off_policy_data:
                # Compute logp under leader (current) and follower (old)
                follower_actions = torch.as_tensor(
                    data.actions, dtype=torch.float32, device=device
                )
                follower_obs = torch.as_tensor(
                    data.obs, dtype=torch.float32, device=device
                )
                follower_logp_old = torch.as_tensor(
                    data.logp, dtype=torch.float32, device=device
                )
                # Leader logp on follower data
                leader_dist, _ = self.model.forward(follower_obs, self.policy_idx)
                leader_logp_new = leader_dist.log_prob(follower_actions).sum(-1)
                # Advantage under leader (use leader's advantage on follower data)
                # For simplicity, we use the follower's advantage as a proxy
                adv = torch.as_tensor(
                    data.advantages, dtype=torch.float32, device=device
                )
                off_loss = self.off_policy_loss(
                    leader_logp_new, follower_logp_old, adv
                )
                off_losses.append(off_loss)
            loss_off = torch.stack(off_losses).mean()
            loss_policy += self.lambda_off * loss_off

        # Value loss
        value_loss = F.mse_loss(value, returns)

        # Total loss
        loss = loss_policy + value_loss

        # Backprop
        loss.backward()
        return loss.item()


# --------------------------------------------------------------- #
#                     Helper functions                            #
# --------------------------------------------------------------- #
def make_vec_env(env_name: str, num_envs: int, seed: int = 0):
    """Create a synchronous vectorized environment."""
    def _thunk():
        env = gym.make(env_name)
        env.seed(seed)
        return env

    envs = gym.vector.SyncVectorEnv([_thunk for _ in range(num_envs)])
    return envs