"""
Minimal SAPG implementation on a toy environment (CartPole).
The code follows the algorithmic structure described in the paper:
- Multiple policies (leader + followers) share a backbone but have individual
  latent conditioning vectors.
- The leader aggregates off‑policy data from followers via importance‑sampling.
- Followers only use their own on‑policy data.
- Entropy regularisation is applied to followers only.
"""

import math
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from env_wrapper import CartPoleContinuous

# ----------------------------
# Helper modules
# ----------------------------

class SharedMLP(nn.Module):
    """Small shared backbone that transforms observations."""
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Policy(nn.Module):
    """
    Policy network with shared backbone + per‑policy latent vector phi.
    Outputs mean and log‑std for Gaussian policy.
    """
    def __init__(self, obs_dim, act_dim, shared_backbone, phi_dim=16):
        super().__init__()
        self.backbone = shared_backbone
        self.phi = nn.Parameter(torch.randn(phi_dim))
        # MLP that takes [backbone_output, phi] -> [mean, log_std]
        self.head = nn.Sequential(
            nn.Linear(64 + phi_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim * 2)  # mean + log_std
        )

    def forward(self, obs):
        x = self.backbone(obs)
        x = torch.cat([x, self.phi.expand_as(x)], dim=-1)
        out = self.head(x)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        std = torch.exp(log_std)
        return mean, std

    def get_action(self, obs):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action.detach(), logp.detach()

    def log_prob(self, obs, action):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(-1)


class ValueNetwork(nn.Module):
    """Critic network sharing the same backbone."""
    def __init__(self, obs_dim, shared_backbone):
        super().__init__()
        self.backbone = shared_backbone
        self.head = nn.Linear(64, 1)

    def forward(self, obs):
        x = self.backbone(obs)
        return self.head(x).squeeze(-1)


# ----------------------------
# Training utilities
# ----------------------------
def compute_returns(rewards, dones, gamma=0.99):
    """Compute discounted returns."""
    returns = []
    R = 0
    for r, d in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1 - d)
        returns.insert(0, R)
    return returns


def train_one_step(policy, value_net, optimizer, batch, entropy_coef, clip_eps=0.2, lam=0.95):
    """
    On‑policy PPO update (clipped surrogate + value loss).
    batch: dict of tensors
    """
    obs = batch["obs"]
    actions = batch["actions"]
    old_logp = batch["logp"]
    returns = batch["returns"]

    # Policy loss
    new_logp = policy.log_prob(obs, actions)
    ratio = torch.exp(new_logp - old_logp)
    adv = returns - value_net(obs).detach()
    clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    policy_loss = -torch.min(ratio * adv, clip_adv).mean()

    # Value loss
    value_pred = value_net(obs)
    value_loss = nn.functional.mse_loss(value_pred, returns)

    # Entropy
    mean, std = policy.forward(obs)
    dist = torch.distributions.Normal(mean, std)
    entropy = dist.entropy().sum(-1).mean()
    loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_net.parameters()), 0.5)
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()


def aggregate_off_policy(leader_policy, leader_value, follower_policies,
                         follower_datasets, optimizer, entropy_coef, clip_eps=0.2):
    """
    Off‑policy update for the leader using data from all followers.
    follower_datasets: list of dicts (each a batch from a follower)
    """
    # Concatenate all follower data
    obs = torch.cat([d["obs"] for d in follower_datasets], dim=0)
    actions = torch.cat([d["actions"] for d in follower_datasets], dim=0)
    logp_old = torch.cat([d["logp"] for d in follower_datasets], dim=0)
    returns = torch.cat([d["returns"] for d in follower_datasets], dim=0)

    # Importance weight w = pi_leader / pi_follower
    # We use the old policy of the follower (stored in logp_old)
    # Compute logp for leader
    logp_leader = leader_policy.log_prob(obs, actions)
    w = torch.exp(logp_leader - logp_old)  # importance ratio

    # Clip ratio similar to PPO
    ratio = w
    adv = returns - leader_value(obs).detach()
    clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    policy_loss = -torch.min(ratio * adv, clip_adv).mean()

    # Value loss
    value_pred = leader_value(obs)
    value_loss = nn.functional.mse_loss(value_pred, returns)

    # Entropy (no entropy for leader)
    mean, std = leader_policy.forward(obs)
    dist = torch.distributions.Normal(mean, std)
    entropy = dist.entropy().sum(-1).mean()

    loss = policy_loss + 0.5 * value_loss - 0 * entropy  # no entropy term

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(list(leader_policy.parameters()) + list(leader_value.parameters()), 0.5)
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item()


# ----------------------------
# Main training loop
# ----------------------------
def main():
    # Hyper‑parameters
    NUM_ENVS = 64      # total number of parallel environments
    NUM_POLICIES = 4   # 1 leader + 3 followers
    NUM_STEPS = 200    # rollout length per env
    UPDATE_EPOCHS = 5
    BATCH_SIZE = 256   # per‑policy minibatch
    GAMMA = 0.99
    ENTROPY_COEF = 0.01  # applied to followers only
    CLIP_EPS = 0.2
    LR = 5e-4

    # Create environments
    envs = [CartPoleContinuous() for _ in range(NUM_ENVS)]

    # Shared backbone
    shared_backbone = SharedMLP(obs_dim=envs[0].observation_space.shape[0], hidden_dim=64)

    # Instantiate policies and critics
    policies = []
    critics = []
    optimizers = []
    for i in range(NUM_POLICIES):
        policy = Policy(obs_dim=envs[0].observation_space.shape[0],
                        act_dim=envs[0].action_space.shape[0],
                        shared_backbone=shared_backbone,
                        phi_dim=16)
        critic = ValueNetwork(obs_dim=envs[0].observation_space.shape[0],
                              shared_backbone=shared_backbone)
        policies.append(policy)
        critics.append(critic)
        optimizers.append(optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=LR))

    # Random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Main loop
    NUM_UPDATES = 30
    results = []

    for upd in trange(NUM_UPDATES, desc="Training"):
        # Rollouts per policy
        rollouts = [{}, {}]  # placeholder for leader and follower batches
        # Collect data for each policy
        for pid, policy in enumerate(policies):
            batch_obs = []
            batch_actions = []
            batch_logp = []
            batch_rewards = []
            batch_dones = []

            # Assign a slice of envs to this policy
            start = pid * (NUM_ENVS // NUM_POLICIES)
            end = (pid + 1) * (NUM_ENVS // NUM_POLICIES)
            env_subset = envs[start:end]

            # Reset environments
            obs_batch = []
            for env in env_subset:
                obs, _ = env.reset()
                obs_batch.append(obs)
            obs_batch = np.array(obs_batch)

            for step in range(NUM_STEPS):
                obs_tensor = torch.tensor(obs_batch, dtype=torch.float32)
                actions, logp = policy.get_action(obs_tensor)
                actions_np = actions.cpu().numpy()

                # Step all environments in the subset
                next_obs = []
                rewards = []
                dones = []
                for env, act in zip(env_subset, actions_np):
                    o, r, d, _, _ = env.step(act)
                    next_obs.append(o)
                    rewards.append(r)
                    dones.append(d)

                next_obs = np.array(next_obs)
                batch_obs.append(obs_batch)
                batch_actions.append(actions_np)
                batch_logp.append(logp.detach().cpu().numpy())
                batch_rewards.append(rewards)
                batch_dones.append(dones)

                # Prepare for next step
                obs_batch = next_obs

            # Convert to tensors
            obs_tensor = torch.tensor(np.concatenate(batch_obs), dtype=torch.float32)
            actions_tensor = torch.tensor(np.concatenate(batch_actions), dtype=torch.float32)
            logp_tensor = torch.tensor(np.concatenate(batch_logp), dtype=torch.float32)
            rewards_list = [r for rl in batch_rewards for r in rl]
            dones_list = [d for dl in batch_dones for d in dl]
            returns = compute_returns(rewards_list, dones_list, gamma=GAMMA)
            returns_tensor = torch.tensor(returns, dtype=torch.float32)

            rollouts[pid] = {
                "obs": obs_tensor,
                "actions": actions_tensor,
                "logp": logp_tensor,
                "returns": returns_tensor,
            }

        # ----- Update leader (policy 0) with on‑ and off‑policy data -----
        leader_policy = policies[0]
        leader_critic = critics[0]
        leader_opt = optimizers[0]

        # On‑policy loss
        loss, p_loss, v_loss, ent = train_one_step(
            leader_policy, leader_critic, leader_opt,
            rollouts[0], ENTROPY_COEF, clip_eps=CLIP_EPS
        )

        # Off‑policy aggregation from followers
        follower_datasets = [rollouts[j] for j in range(1, NUM_POLICIES)]
        off_loss, off_p_loss, off_v_loss = aggregate_off_policy(
            leader_policy, leader_critic, policies[1:],
            follower_datasets, leader_opt, ENTROPY_COEF, clip_eps=CLIP_EPS
        )

        # ----- Update followers (only on‑policy) -----
        for pid in range(1, NUM_POLICIES):
            train_one_step(
                policies[pid], critics[pid], optimizers[pid],
                rollouts[pid], ENTROPY_COEF, clip_eps=CLIP_EPS
            )

        # ----- Logging -----
        # Compute average return per policy
        for pid in range(NUM_POLICIES):
            returns = rollouts[pid]["returns"].numpy()
            avg_ret = returns.mean()
            results.append((pid, avg_ret))

    # Write results to CSV
    import csv
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy_id", "avg_return"])
        for pid, avg_ret in results:
            writer.writerow([pid, avg_ret])

    print("\nTraining finished. Results written to results.csv.")


if __name__ == "__main__":
    main()