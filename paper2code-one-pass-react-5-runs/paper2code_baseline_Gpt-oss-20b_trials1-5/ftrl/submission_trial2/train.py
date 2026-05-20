"""
Training script that reproduces the toy experiments of the paper.

1. Pre‑train a policy on Phase 2 only (AppleRetrieval phase 2).
2. Fine‑tune the policy on the full task using:
   - Vanilla
   - BC (behavioral cloning)
   - KS (kick‑starting)
   - EWC (elastic weight consolidation)
3. Log the average discounted return over 200 episodes for each method.
"""

import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Bernoulli
from apple_retrieval import AppleRetrieval
from models import LinearPolicy, LinearBaseline
from bc import bc_loss
from ewc import compute_fisher, ewc_loss

# Fix seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Hyper‑parameters
M = 30                      # distance to apple
max_steps = 100
pretrain_episodes = 200
finetune_episodes = 200
batch_size = 32
lr = 1e-3
gamma = 0.99

# Helper: compute discounted returns
def discounted_returns(rewards, gamma):
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

# 1. Pre‑training on Phase 2 only
env = AppleRetrieval(M=M, max_steps=max_steps, seed=42)
policy = LinearPolicy()
baseline = LinearBaseline()
optimizer = optim.Adam(list(policy.parameters()) + list(baseline.parameters()), lr=lr)

def pretrain_policy():
    policy.train()
    baseline.train()
    for ep in range(pretrain_episodes):
        obs = env.reset()
        if env.phase != 2:
            # force start in phase 2
            env.phase = 2
            env.x = M
            env.done = False
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        log_probs = []
        rewards = []
        values = []
        for t in range(max_steps):
            probs = policy(obs)
            m = Bernoulli(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            next_obs, reward, done, _ = env.step(int(action.item()))
            value = baseline(obs)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            if done:
                break
        returns = discounted_returns(rewards, gamma)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
        values = torch.cat(values)
        advantage = returns - values.detach()
        policy_loss = -(torch.stack(log_probs) * advantage).mean()
        baseline_loss = ((returns - values).pow(2).mean())
        loss = policy_loss + baseline_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save teacher policy
    torch.save(policy.state_dict(), "teacher.pt")
    torch.save(baseline.state_dict(), "baseline_teacher.pt")
    return policy, baseline

teacher_policy, teacher_baseline = pretrain_policy()

# 2. Fine‑tuning
def finetune(method="vanilla", buffer=None, fisher=None, theta_star=None):
    """
    method: 'vanilla', 'bc', 'ks', 'ewc'
    buffer: list of pre‑trained states for BC
    fisher, theta_star: for EWC
    """
    # load teacher weights
    policy.load_state_dict(torch.load("teacher.pt"))
    baseline.load_state_dict(torch.load("baseline_teacher.pt"))

    policy.train()
    baseline.train()
    optimizer = optim.Adam(list(policy.parameters()) + list(baseline.parameters()), lr=lr)

    # Buffer for BC
    if buffer is not None:
        buffer_tensor = torch.stack(buffer).to("cpu")

    # Data loader for EWC (few samples)
    if fisher is not None:
        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(buffer_tensor),
            batch_size=batch_size, shuffle=False)

    all_returns = []

    for ep in range(finetune_episodes):
        env.phase = 1
        env.x = 0
        env.done = False
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        log_probs = []
        rewards = []
        values = []
        for t in range(max_steps):
            probs = policy(obs)
            m = Bernoulli(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            next_obs, reward, done, _ = env.step(int(action.item()))
            value = baseline(obs)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            if done:
                break

        returns = discounted_returns(rewards, gamma)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
        values = torch.cat(values)
        advantage = returns - values.detach()
        policy_loss = -(torch.stack(log_probs) * advantage).mean()
        baseline_loss = ((returns - values).pow(2).mean())
        loss = policy_loss + baseline_loss

        # Knowledge‑retention losses
        if method == "bc":
            loss += bc_loss(policy, teacher_policy, buffer_tensor)
        elif method == "ks":
            # sample online states for KL
            online_states = torch.cat([obs.detach() for obs in [torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0) for _ in range(batch_size)]])[:buffer_tensor.size(0)]
            loss += bc_loss(policy, teacher_policy, online_states)
        elif method == "ewc":
            loss += ewc_loss(policy, fisher, theta_star)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ep_return = sum(rewards)
        all_returns.append(ep_return)

    avg_return = np.mean(all_returns)
    return avg_return

# Build BC buffer (pre‑trained states)
print("Collecting BC buffer (pre‑trained phase 2 states)...")
teacher_policy.eval()
buffer_states = []
for _ in range(500):  # 500 samples
    env.phase = 2
    env.x = M
    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    buffer_states.append(obs)
buffer_tensor = torch.cat(buffer_states)

# Compute Fisher for EWC
print("Estimating Fisher for EWC...")
fisher = compute_fisher(teacher_policy, torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(buffer_tensor), batch_size=32))

# Save theta_star for EWC
theta_star = [p.clone().detach() for p in teacher_policy.parameters()]

# Fine‑tune with different methods
methods = ["vanilla", "bc", "ks", "ewc"]
results = {}
for m in methods:
    print(f"Fine‑tuning with {m}...")
    avg = finetune(method=m,
                   buffer=buffer_states if m=="bc" else None,
                   fisher=fisher if m=="ewc" else None,
                   theta_star=theta_star if m=="ewc" else None)
    results[m] = round(float(avg), 2)
    print(f"  Avg return: {results[m]}")

# Save results
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== Results ===")
print(json.dumps(results, indent=2))