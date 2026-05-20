#!/usr/bin/env python3
"""
Training routine for the ALE sequences (SpaceInvaders, Freeway) using PPO.
"""

import argparse
import os
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from compo import CompoNet
from utils import set_seed, log_results, get_action_distribution

# ----------------------------------------------------------------------
# 1. Hyper‑parameters (minimal, not tuned)
# ----------------------------------------------------------------------
BATCH_SIZE = 256
LR = 2.5e-4
GAMMA = 0.99
TARGET_UPDATE_RATE = 0.005
MAX_STEPS_PER_TASK = 1_000_000
DISCRETE = True
ACTION_DIM = 6  # SpaceInvaders (6 actions), Freeway (3 actions)
INPUT_DIM = 512  # Output of the encoder (see encoder below)
HIDDEN_DIM = 256

# ----------------------------------------------------------------------
# 2. Simple CNN encoder for Atari frames
# ----------------------------------------------------------------------
class AtariEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, INPUT_DIM),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# 3. Simple PPO implementation
# ----------------------------------------------------------------------
class PPOAgent:
    def __init__(self, device, action_dim):
        self.device = device
        self.actor = CompoNet(INPUT_DIM, action_dim, HIDDEN_DIM).to(device)
        self.critic = nn.Sequential(
            nn.Linear(INPUT_DIM + action_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        ).to(device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR)

        self.buffer = []

    def collect(self, env, encoder, steps=1024):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        for _ in range(steps):
            h_s = encoder(state[None, ...])  # (1, INPUT_DIM)
            logits = self.actor(h_s)
            dist = get_action_distribution(logits, self.actor.action_dim, DISCRETE)
            action = dist.sample()
            logp = dist.log_prob(action)
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated
            self.buffer.append(
                (state, action, logp, reward, torch.tensor(done, dtype=torch.float32))
            )
            state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            if done:
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=self.device)

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        # Sample a minibatch
        batch = random.sample(self.buffer, BATCH_SIZE)
        states, actions, logp_old, rewards, dones = map(
            torch.stack, zip(*batch)
        )
        # Compute returns
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + GAMMA * R * (1 - d)
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Critic loss
        critic_vals = self.critic(torch.cat([states, actions], dim=1)).squeeze()
        critic_loss = ((critic_vals - returns) ** 2).mean()

        # Actor loss
        logits = self.actor(states)
        dist = get_action_distribution(logits, self.actor.action_dim, DISCRETE)
        logp_new = dist.log_prob(actions)
        ratio = torch.exp(logp_new - logp_old)
        surrogate = ratio * (returns - critic_vals.detach())
        actor_loss = -surrogate.mean()

        # Optimize
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Clear buffer
        self.buffer = []


# ----------------------------------------------------------------------
# 4. Main training loop
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["SpaceInvaders-v5", "Freeway-v5"], required=True)
    parser.add_argument("--log_dir", default="ale_logs")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"{args.env.lower()}.log")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(device, ACTION_DIM)
    encoder = AtariEncoder().to(device)

    env = gym.make(args.env, frameskip=1, repeat_action_probability=0.0)
    env.reset(seed=args.seed)
    steps = 0
    episode_rewards = []
    success_count = 0

    while steps < MAX_STEPS_PER_TASK:
        agent.collect(env, encoder, steps=1024)
        agent.update()
        # Evaluate one episode
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done and steps < MAX_STEPS_PER_TASK:
            state_t = torch.tensor(state, dtype=torch.float32, device=device)[None, ...]
            h_s = encoder(state_t)
            logits = agent.actor(h_s)
            dist = get_action_distribution(logits, agent.actor.action_dim, DISCRETE)
            action = dist.sample()
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated
            ep_reward += reward
            steps += 1
            state = next_state
        episode_rewards.append(ep_reward)
        if ep_reward >= 1000:  # dummy threshold
            success_count += 1

    success_rate = success_count / len(episode_rewards)
    log_results(log_file, 0, success_rate, episode_rewards)

    env.close()
    print(f"=== Completed {args.env} ===")
    print("ALE training finished.")


if __name__ == "__main__":
    main()