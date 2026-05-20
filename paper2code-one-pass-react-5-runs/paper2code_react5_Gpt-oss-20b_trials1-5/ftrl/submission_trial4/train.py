#!/usr/bin/env python3
"""
Minimal implementation of the fine‑tuning pipeline from
“Fine‑tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem”.
"""

import argparse
import csv
import os
import random
import sys
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discount(x, gamma):
    """Apply discount to a 1‑D array."""
    return np.array(
        [sum(gamma**i * y for i, y in enumerate(x[idx:])) for idx in range(len(x))]
    )


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Return advantages and returns using GAE."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns


# --------------------------------------------------------------------------- #
# Policy network (actor + critic)
# --------------------------------------------------------------------------- #
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128)):
        super().__init__()
        # Actor
        actor_layers = []
        last = obs_dim
        for h in hidden_sizes:
            actor_layers.append(nn.Linear(last, h))
            actor_layers.append(nn.ReLU())
            last = h
        actor_layers.append(nn.Linear(last, act_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Critic
        critic_layers = []
        last = obs_dim
        for h in hidden_sizes:
            critic_layers.append(nn.Linear(last, h))
            critic_layers.append(nn.ReLU())
            last = h
        critic_layers.append(nn.Linear(last, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x):
        """Return action logits and value."""
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


# --------------------------------------------------------------------------- #
# Replay buffer for offline data
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*batch)
        )
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# --------------------------------------------------------------------------- #
# Rule‑based policy (teacher)
# --------------------------------------------------------------------------- #
def rule_policy(state, env):
    """
    Simple deterministic policy for CartPole:
    Push right (action 1) if pole angle > 0 else left (action 0).
    """
    # state shape: (4,)
    # For CartPole, state[2] is pole angle
    angle = state[2]
    if angle > 0:
        return 1
    else:
        return 0


# --------------------------------------------------------------------------- #
# Offline dataset generation
# --------------------------------------------------------------------------- #
def generate_offline_dataset(env, buffer, n_steps=50000):
    """Run the rule‑based policy to collect a large offline dataset."""
    obs = env.reset()
    done = False
    steps = 0
    while steps < n_steps:
        action = rule_policy(obs, env)
        next_obs, reward, done, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        steps += 1
        if done:
            obs = env.reset()
            done = False
    return buffer


# --------------------------------------------------------------------------- #
# BC pre‑training
# --------------------------------------------------------------------------- #
def bc_pretrain(env, buffer, device, hidden_sizes=(128, 128),
                epochs=5, batch_size=64, lr=1e-3):
    """Train a new policy to mimic the rule‑based teacher on the offline buffer."""
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy = PolicyNetwork(obs_dim, act_dim, hidden_sizes).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Precompute teacher logits for the buffer (one‑hot actions)
    states = torch.tensor(np.array([s for s, _, _, _, _ in buffer.buffer]),
                          dtype=torch.float32, device=device)
    actions = torch.tensor([a for _, a, _, _, _ in buffer.buffer],
                           dtype=torch.long, device=device)
    # Teacher distribution: deterministic one‑hot
    teacher_log_probs = torch.zeros_like(states, device=device)
    teacher_log_probs = F.one_hot(actions, num_classes=act_dim).float()

    for epoch in range(epochs):
        idxs = np.random.permutation(len(buffer))
        for i in range(0, len(buffer), batch_size):
            batch_idxs = idxs[i : i + batch_size]
            batch_states = states[batch_idxs]
            batch_teacher_log_probs = teacher_log_probs[batch_idxs]
            # Current policy
            logits, _ = policy(batch_states)
            log_probs = F.log_softmax(logits, dim=-1)
            # KL divergence (teacher -> student) reduces to negative log prob of teacher action
            kl = -torch.sum(
                batch_teacher_log_probs * log_probs, dim=-1
            ).mean()
            loss = kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return policy


# --------------------------------------------------------------------------- #
# Fisher estimation for EWC
# --------------------------------------------------------------------------- #
def estimate_fisher(policy, buffer, device, samples=512):
    """Approximate diagonal Fisher information matrix for the actor."""
    fisher = {}
    params = list(policy.parameters())
    for name, param in zip([n for n, _ in policy.named_parameters()], params):
        fisher[name] = torch.zeros_like(param, device=device)

    # Use a subset of buffer for estimation
    idxs = random.sample(range(len(buffer)), min(samples, len(buffer)))
    for idx in idxs:
        state, action, _, _, _ = buffer.buffer[idx]
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, _ = policy(state_t)
        log_probs = F.log_softmax(logits, dim=-1)
        log_prob = log_probs[0, action]
        grads = torch.autograd.grad(log_prob, params, retain_graph=False)
        for (name, _), grad in zip(zip([n for n, _ in policy.named_parameters()], params), grads):
            if grad is not None:
                fisher[name] += grad.pow(2)

    for name in fisher:
        fisher[name] /= len(idxs)
    return fisher


# --------------------------------------------------------------------------- #
# Fine‑tuning with optional knowledge‑retention losses
# --------------------------------------------------------------------------- #
def fine_tune(env, policy, buffer, device,
              total_timesteps=200000, eval_interval=20000,
              output_csv="output.csv",
              use_bc=False, use_ks=False, use_ewc=False, use_em=False,
              bc_coef=2.0, ks_coef=0.5, ewc_coef=1e-4,
              gamma=0.99, lam=0.95, batch_steps=2048):
    """Fine‑tune the policy with optional losses."""
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Precompute Fisher if needed
    if use_ewc:
        fisher = estimate_fisher(policy, buffer, device)
        mean_params = {name: param.detach().clone()
                       for name, param in policy.named_parameters()}
    else:
        fisher = {}
        mean_params = {}

    # Evaluation helper
    def evaluate():
        eval_env = gym.make(env.unwrapped.spec.id)
        n_eps = 10
        rewards = []
        for _ in range(n_eps):
            o = eval_env.reset()
            done = False
            ep_r = 0.0
            while not done:
                o_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = policy(o_t)
                probs = F.softmax(logits, dim=-1)
                action = probs.argmax().item()
                o, r, done, _ = eval_env.step(action)
                ep_r += r
            rewards.append(ep_r)
        eval_env.close()
        return np.mean(rewards), np.std(rewards)

    # Main training loop
    obs = env.reset()
    done = False
    step = 0
    results = []

    # Buffer for online transitions (for EM)
    online_buffer = ReplayBuffer(max_size=100000)

    # Store episode data for GAE
    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_values = []
    episode_dones = []

    while step < total_timesteps:
        state = obs
        done = False
        ep_reward = 0.0
        ep_len = 0
        while not done and step < total_timesteps:
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = policy(state_t)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
            next_state, reward, done, _ = env.step(action)

            # Store for GAE
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_values.append(value.item())
            episode_dones.append(done)

            # Store in online buffer for EM
            online_buffer.add(state, action, reward, next_state, done)

            ep_reward += reward
            ep_len += 1
            step += 1
            state = next_state

        # After episode, compute GAE and update
        # Append a dummy value for the last state
        episode_values.append(0.0)
        returns, advantages = compute_gae(
            episode_rewards, np.array(episode_values), gamma=gamma, lam=lam
        )
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # Convert episode data to tensors
        states_tensor = torch.tensor(episode_states, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(episode_actions, dtype=torch.long, device=device)
        # Compute policy loss
        logits, _ = policy(states_tensor)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        policy_loss = -(advantages * action_log_probs).mean()

        # Compute value loss
        values_pred = policy.states_tensor  # placeholder, replaced below
        # Since we already have logits, recompute values
        _, values_pred = policy(states_tensor)
        value_loss = F.mse_loss(returns, values_pred)

        # Extra losses
        extra_loss = 0.0

        # BC loss on offline buffer
        if use_bc:
            if len(buffer) >= 64:
                buf_states, buf_actions, _, _, _ = buffer.sample(64)
                # Teacher is deterministic one‑hot
                teacher_log_probs = F.one_hot(buf_actions, num_classes=act_dim).float()
                logits_buf, _ = policy(buf_states)
                log_probs_buf = F.log_softmax(logits_buf, dim=-1)
                kl = -torch.sum(teacher_log_probs * log_probs_buf, dim=-1).mean()
                extra_loss += bc_coef * kl

        # KS loss on online states
        if use_ks:
            if len(online_buffer) >= 64:
                on_states, on_actions, _, _, _ = online_buffer.sample(64)
                teacher_log_probs = F.one_hot(on_actions, num_classes=act_dim).float()
                logits_on, _ = policy(on_states)
                log_probs_on = F.log_softmax(logits_on, dim=-1)
                kl = -torch.sum(teacher_log_probs * log_probs_on, dim=-1).mean()
                extra_loss += ks_coef * kl

        # EWC loss
        if use_ewc:
            ewc_loss = 0.0
            for name, param in policy.named_parameters():
                if name in fisher:
                    ewc_loss += (
                        fisher[name] * (param - mean_params[name]).pow(2)
                    ).sum()
            extra_loss += ewc_coef * ewc_loss

        # EM loss (replay of offline buffer)
        if use_em:
            if len(buffer) >= 64:
                em_states, em_actions, _, _, _ = buffer.sample(64)
                teacher_log_probs = F.one_hot(em_actions, num_classes=act_dim).float()
                logits_em, _ = policy(em_states)
                log_probs_em = F.log_softmax(logits_em, dim=-1)
                kl = -torch.sum(teacher_log_probs * log_probs_em, dim=-1).mean()
                extra_loss += bc_coef * kl  # reuse BC coefficient

        total_loss = policy_loss + value_loss + extra_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Reset episode buffers
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_values = []
        episode_dones = []

        # Evaluation
        if step % eval_interval == 0:
            mean_r, std_r = evaluate()
            results.append((step, mean_r, std_r))
            print(
                f"Step {step:>7} | Eval mean: {mean_r:.2f} ± {std_r:.2f} | "
                f"Episode reward: {ep_reward:.2f}"
            )

    # Final evaluation
    mean_r, std_r = evaluate()
    results.append((step, mean_r, std_r))
    print(f"Training finished. Final eval mean: {mean_r:.2f} ± {std_r:.2f}")

    # Write results
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "mean_reward", "std_reward"])
        writer.writerows(results)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Fine‑tune RL policy with optional forgetting mitigation."
    )
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--eval-interval", type=int, default=20000)
    parser.add_argument("--output", type=str, default="output.csv")
    parser.add_argument("--use_bc", action="store_true")
    parser.add_argument("--use_ks", action="store_true")
    parser.add_argument("--use_ewc", action="store_true")
    parser.add_argument("--use_em", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(args.env)

    print("=== Phase 1: Offline dataset generation (rule‑based) ===")
    buffer = ReplayBuffer(max_size=50000)
    generate_offline_dataset(env, buffer, n_steps=50000)

    print("=== Phase 2: Behavioural‑Cloning pre‑training ===")
    bc_policy = bc_pretrain(env, buffer, device)

    print("=== Phase 3: Fine‑tuning ===")
    fine_tune(
        env,
        bc_policy,
        buffer,
        device,
        total_timesteps=args.timesteps,
        eval_interval=args.eval_interval,
        output_csv=args.output,
        use_bc=args.use_bc,
        use_ks=args.use_ks,
        use_ewc=args.use_ewc,
        use_em=args.use_em,
    )


if __name__ == "__main__":
    main()