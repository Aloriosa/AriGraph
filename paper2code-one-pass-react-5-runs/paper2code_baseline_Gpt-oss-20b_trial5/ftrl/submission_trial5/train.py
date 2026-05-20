"""
Full training pipeline:
 1. Pre‑train on phase 1 (Far) only.
 2. Fine‑tune on full task without BC (vanilla).
 3. Fine‑tune on full task with BC (knowledge retention).
 4. Evaluate all policies on both phases.
 5. Save results to results.csv.
"""
import csv
import json
import math
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from bc import bc_loss
from env import TwoPhaseGridWorld
from policy import LinearPolicy

# --------------------------------------------------------------------------- #
# Hyper‑parameters
# --------------------------------------------------------------------------- #
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

M = 10
GAMMA = 0.99
LR = 1e-2
BATCH_SIZE = 64
PRETRAIN_EPISODES = 200
FINETUNE_EPISODES = 200
EVAL_EPISODES = 100
BC_LAMBDA = 0.5
BUFFER_SIZE = 500  # number of pre‑trained states to keep for BC


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def run_episode(env, policy, max_steps):
    """Run one episode, return trajectory data."""
    state = env.reset()
    traj = []
    for _ in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action, prob = policy.action(state_tensor)
        next_state, reward, done, _ = env.step(action)
        traj.append((state, action, reward, prob))
        state = next_state
        if done:
            break
    return traj


def compute_returns(rewards, gamma):
    """Compute discounted returns."""
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def update_policy(policy, traj, optimizer, teacher=None, buffer_states=None, bc_lambda=0.0):
    """Policy gradient update (REINFORCE) with optional BC loss."""
    states, actions, rewards, probs = zip(*traj)
    returns = compute_returns(rewards, GAMMA)
    loss = 0.0
    for s, a, R, p in zip(states, actions, returns, probs):
        state_tensor = torch.tensor(s, dtype=torch.float32)
        logp = policy.log_prob(state_tensor, a)
        loss -= logp * R  # negative for gradient descent

    loss = loss / len(traj)  # average

    # BC loss
    if bc_lambda > 0.0 and buffer_states is not None:
        bc_l = bc_loss(policy, teacher, buffer_states)
        loss += bc_lambda * bc_l

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(env, policy, episodes):
    """Return success rate on given environment."""
    successes = 0
    for _ in range(episodes):
        state = env.reset()
        for _ in range(2 * M):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action, _ = policy.action(state_tensor)
            state, _, done, _ = env.step(action)
            if done:
                # success if reached goal
                if state[0] == (M if env.phase == 0 else 0):
                    successes += 1
                break
    return successes / episodes


# --------------------------------------------------------------------------- #
# Main training pipeline
# --------------------------------------------------------------------------- #
def main():
    # Pre‑training on phase 1 (Far)
    print("Pre‑training on phase 1 (Far)...")
    pretrain_policy = LinearPolicy(seed=SEED)
    optimizer = torch.optim.Adam(pretrain_policy.parameters(), lr=LR)
    buffer = deque(maxlen=BUFFER_SIZE)

    for ep in range(PRETRAIN_EPISODES):
        env = TwoPhaseGridWorld(M=M, phase=1, seed=SEED + ep)
        traj = run_episode(env, pretrain_policy, max_steps=2 * M)
        # Store states for BC
        for s, a, r, p in traj:
            buffer.append(s)
        loss = update_policy(pretrain_policy, traj, optimizer)
        if (ep + 1) % 50 == 0:
            print(f"  pre‑train ep {ep+1}/{PRETRAIN_EPISODES} loss {loss:.4f}")

    pretrain_buffer = torch.tensor(list(buffer), dtype=torch.float32)

    # Fine‑tune without BC
    print("\nFine‑tuning without BC (vanilla)...")
    finetune_policy = LinearPolicy(seed=SEED + 1000)
    finetune_policy.load_state_dict(pretrain_policy.state_dict())
    opt_ft = torch.optim.Adam(finetune_policy.parameters(), lr=LR)
    for ep in range(FINETUNE_EPISODES):
        env = TwoPhaseGridWorld(M=M, phase=0, seed=SEED + ep + 2000)
        traj = run_episode(env, finetune_policy, max_steps=2 * M)
        loss = update_policy(finetune_policy, traj, opt_ft)
        if (ep + 1) % 50 == 0:
            print(f"  finetune ep {ep+1}/{FINETUNE_EPISODES} loss {loss:.4f}")

    # Fine‑tune with BC
    print("\nFine‑tuning with BC (knowledge retention)...")
    finetune_bc_policy = LinearPolicy(seed=SEED + 2000)
    finetune_bc_policy.load_state_dict(pretrain_policy.state_dict())
    opt_bc = torch.optim.Adam(finetune_bc_policy.parameters(), lr=LR)
    for ep in range(FINETUNE_EPISODES):
        env = TwoPhaseGridWorld(M=M, phase=0, seed=SEED + ep + 3000)
        traj = run_episode(env, finetune_bc_policy, max_steps=2 * M)
        loss = update_policy(
            finetune_bc_policy,
            traj,
            opt_bc,
            teacher=pretrain_policy,
            buffer_states=pretrain_buffer,
            bc_lambda=BC_LAMBDA,
        )
        if (ep + 1) % 50 == 0:
            print(f"  finetune_bc ep {ep+1}/{FINETUNE_EPISODES} loss {loss:.4f}")

    # Evaluation
    print("\nEvaluation:")
    eval_env_close = TwoPhaseGridWorld(M=M, phase=0, seed=SEED + 4000)
    eval_env_far = TwoPhaseGridWorld(M=M, phase=1, seed=SEED + 4001)

    results = []

    # Pre‑train
    sr = evaluate(eval_env_far, pretrain_policy, EVAL_EPISODES)
    results.append(("far", "pretrain", sr))
    sr = evaluate(eval_env_close, pretrain_policy, EVAL_EPISODES)
    results.append(("close", "pretrain", sr))

    # Finetune vanilla
    sr = evaluate(eval_env_close, finetune_policy, EVAL_EPISODES)
    results.append(("close", "finetune", sr))
    sr = evaluate(eval_env_far, finetune_policy, EVAL_EPISODES)
    results.append(("far", "finetune", sr))

    # Finetune BC
    sr = evaluate(eval_env_close, finetune_bc_policy, EVAL_EPISODES)
    results.append(("close", "finetune_bc", sr))
    sr = evaluate(eval_env_far, finetune_bc_policy, EVAL_EPISODES)
    results.append(("far", "finetune_bc", sr))

    # Save to CSV
    csv_path = "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "method", "success_rate"])
        for phase, method, sr in results:
            writer.writerow([phase, method, f"{sr:.4f}"])

    print(f"\nResults written to {csv_path}")

    # Also output a quick JSON summary
    summary = {f"{p}_{m}": sr for p, m, sr in results}
    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()