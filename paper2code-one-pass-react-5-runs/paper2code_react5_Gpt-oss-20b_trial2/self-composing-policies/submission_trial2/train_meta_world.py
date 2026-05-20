#!/usr/bin/env python3
"""
Training routine for the Meta‑World sequence using SAC.
"""

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import metaworld
import torch
import torch.nn as nn
import torch.optim as optim

from compo import CompoNet
from utils import set_seed, log_results, get_action_distribution

# ----------------------------------------------------------------------
# 1. Hyper‑parameters (high‑level, not tuned to exact paper values)
# ----------------------------------------------------------------------
BATCH_SIZE = 128
LR = 3e-4
GAMMA = 0.99
POLICY_LR = 3e-4
Q_LR = 3e-4
TARGET_UPDATE_RATE = 0.005
MAX_STEPS_PER_TASK = 1_000_000
NUM_TASKS = 20   # 10 tasks repeated twice (CW20)
ACTION_DIM = 4
INPUT_DIM = 39   # State size for Meta‑World tasks
HIDDEN_DIM = 256
DISCRETE = False  # Continuous actions

# ----------------------------------------------------------------------
# 2. Simple SAC implementation (actor + two critics)
# ----------------------------------------------------------------------
class SACAgent:
    def __init__(self, device):
        self.device = device
        self.actor = CompoNet(INPUT_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
        self.critic1 = nn.Sequential(
            nn.Linear(INPUT_DIM + ACTION_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        ).to(device)
        self.critic2 = nn.Sequential(
            nn.Linear(INPUT_DIM + ACTION_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        ).to(device)
        self.target_critic1 = nn.Sequential(
            nn.Linear(INPUT_DIM + ACTION_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        ).to(device)
        self.target_critic2 = nn.Sequential(
            nn.Linear(INPUT_DIM + ACTION_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        ).to(device)

        # Copy parameters
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=POLICY_LR)
        self.q_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=Q_LR
        )
        self.alpha = 0.2  # Entropy coefficient

        self.replay_buffer = []

    def sample_action(self, state):
        h_s = torch.tensor(state, dtype=torch.float32, device=self.device)
        logits = self.actor(h_s.unsqueeze(0))
        dist = get_action_distribution(logits, ACTION_DIM, DISCRETE)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action.detach().cpu().numpy(), logp, dist.entropy()

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = map(
            torch.stack, zip(*batch)
        )

        # Critic update
        with torch.no_grad():
            next_logits = self.actor(next_states)
            next_dist = get_action_distribution(next_logits, ACTION_DIM, DISCRETE)
            next_actions = next_dist.sample()
            next_logp = next_dist.log_prob(next_actions)
            next_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=1))
            next_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=1))
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_logp
            target_q = rewards + (1 - dones) * GAMMA * next_q

        q1 = self.critic1(torch.cat([states, actions], dim=1))
        q2 = self.critic2(torch.cat([states, actions], dim=1))
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # Actor update
        logits = self.actor(states)
        dist = get_action_distribution(logits, ACTION_DIM, DISCRETE)
        actions_new = dist.sample()
        logp_new = dist.log_prob(actions_new)
        q1_new = self.critic1(torch.cat([states, actions_new], dim=1))
        q2_new = self.critic2(torch.cat([states, actions_new], dim=1))
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * logp_new - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Target update
        for target, source in zip(
            [self.target_critic1, self.target_critic2],
            [self.critic1, self.critic2],
        ):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    target_param.data * (1 - TARGET_UPDATE_RATE) + param.data * TARGET_UPDATE_RATE
                )

    def store_transition(self, transition):
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > 1_000_000:
            self.replay_buffer.pop(0)


# ----------------------------------------------------------------------
# 3. Main training loop
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="meta_world_logs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, "meta_world.log")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(device)

    # Meta‑World task list (the 10 tasks used in CW20)
    task_names = [
        "hammer-v2",
        "push-wall-v2",
        "faucet-close-v2",
        "push-back-v2",
        "stick-pull-v2",
        "handle-press-side-v2",
        "push-v2",
        "shelf-place-v2",
        "window-close-v2",
        "peg-unplug-side-v2",
    ]
    # Repeat twice to get 20 tasks
    task_names = task_names * 2

    for task_id, task_name in enumerate(task_names):
        env = gym.make(task_name)
        env.reset(seed=args.seed + task_id)
        agent.actor.add_module()  # new module for this task

        episode_rewards = []
        success_count = 0
        steps = 0
        while steps < MAX_STEPS_PER_TASK:
            state, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done and steps < MAX_STEPS_PER_TASK:
                action, logp, entropy = agent.sample_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.store_transition(
                    (
                        torch.tensor(state, dtype=torch.float32, device=device),
                        torch.tensor(action, dtype=torch.float32, device=device),
                        torch.tensor(reward, dtype=torch.float32, device=device),
                        torch.tensor(
                            next_state, dtype=torch.float32, device=device
                        ),
                        torch.tensor(float(done), dtype=torch.float32, device=device),
                    )
                )
                agent.update()
                state = next_state
                ep_reward += reward
                steps += 1

            episode_rewards.append(ep_reward)
            # Success is defined as reaching a reward threshold
            if ep_reward >= 0.5:  # dummy threshold for demo
                success_count += 1

        success_rate = success_count / len(episode_rewards)
        log_results(log_file, task_id, success_rate, episode_rewards)

        env.close()
        print(f"=== Completed Task {task_id} ({task_name}) ===")
    print("Meta‑World training finished.")


if __name__ == "__main__":
    main()