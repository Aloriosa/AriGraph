"""Training script for the minimal CompoNet implementation."""
import os
import random
import numpy as np
import torch
import torch.optim as optim
import gym
from tqdm import trange

from .utils import set_global_seed, discounted_returns
from .envs import get_task_sequence, make_env
from .models import CompoNet

# ---------------------------
# Hyper‑parameters
# ---------------------------
SEED = 42
NUM_EPISODES_PER_TASK = 200   # keep small for quick demo
MAX_STEPS_PER_EPISODE = 200
GAMMA = 0.99
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
LOG_INTERVAL = 20

# ---------------------------
# Main training routine
# ---------------------------

def main():
    set_global_seed(SEED)

    task_sequence = get_task_sequence()
    results = []

    for task_idx, env_id in enumerate(task_sequence):
        print(f"\n=== Training on Task {task_idx}: {env_id} ===")
        env = make_env(env_id, seed=SEED + task_idx)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Create or extend the CompoNet
        if task_idx == 0:
            compo = CompoNet(state_dim, action_dim, HIDDEN_DIM)
        else:
            # Ensure state/action dims match across tasks
            assert state_dim == compo.state_dim
            assert action_dim == compo.action_dim

        # Add new module for the current task
        current_module = compo.add_module()
        optimizer = optim.Adam(current_module.parameters(), lr=LEARNING_RATE)

        # Freeze previous modules
        for mod in compo.modules[:-1]:
            for p in mod.parameters():
                p.requires_grad = False

        episode_returns = []

        # Training loop
        for ep in trange(NUM_EPISODES_PER_TASK, desc="Episodes", leave=False):
            state = env.reset()
            ep_rewards = []
            log_probs = []

            for step in range(MAX_STEPS_PER_EPISODE):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits = compo(state_tensor)
                probs = torch.softmax(logits, dim=-1)
                m = torch.distributions.Categorical(probs)
                action = m.sample()
                log_prob = m.log_prob(action)

                next_state, reward, done, _ = env.step(action.item())
                ep_rewards.append(reward)
                log_probs.append(log_prob)

                state = next_state

                if done:
                    break

            # Compute discounted returns
            returns = discounted_returns(ep_rewards, gamma=GAMMA)
            returns = torch.tensor(returns, dtype=torch.float32)

            # REINFORCE update (only current module)
            loss = 0
            for log_p, R in zip(log_probs, returns):
                loss -= log_p * R
            loss = loss / len(log_probs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_returns.append(sum(ep_rewards))

            if (ep + 1) % LOG_INTERVAL == 0:
                avg_ret = np.mean(episode_returns[-LOG_INTERVAL:])
                print(f"Ep {ep+1} avg return (last {LOG_INTERVAL}): {avg_ret:.2f}")

        # Evaluate final policy
        eval_returns = []
        for _ in range(10):
            state = env.reset()
            ep_ret = 0
            for _ in range(MAX_STEPS_PER_EPISODE):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits = compo(state_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()
                state, reward, done, _ = env.step(action)
                ep_ret += reward
                if done:
                    break
            eval_returns.append(ep_ret)

        avg_eval = np.mean(eval_returns)
        print(f"=== Task {task_idx} ({env_id}) avg eval return: {avg_eval:.2f} ===")
        results.append((env_id, avg_eval))

    # Save results to a text file
    out_path = "results.txt"
    with open(out_path, "w") as f:
        for env_id, avg_ret in results:
            f.write(f"{env_id}\t{avg_ret:.2f}\n")
    print(f"\nResults written to {out_path}")

if __name__ == "__main__":
    main()