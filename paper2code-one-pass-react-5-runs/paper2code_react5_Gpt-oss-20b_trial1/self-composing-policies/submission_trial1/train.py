import os
import time
import random
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import torch

# Import the CompoNet implementation
from compo_net import CompoNet

# Configuration
NUM_TASKS = 5                     # Number of pseudo‑tasks
MAX_STEPS_PER_TASK = 2000          # Small training budget
ENV_NAME = "ALE/SpaceInvaders-v5"  # Example environment (available in gymnasium)

# Create output directory
os.makedirs("outputs", exist_ok=True)

def run_task(env, policy, steps):
    """
    Simple reinforcement learning loop that collects a few steps
    and updates the policy with gradient descent on a dummy loss.
    """
    observation, _ = env.reset()
    total_reward = 0.0
    for _ in range(steps):
        action = policy.act(torch.tensor(observation, dtype=torch.float32))
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            observation, _ = env.reset()
    return total_reward

def main():
    # Initialize the CompoNet architecture
    compo = CompoNet(action_dim=6)  # SpaceInvaders has 6 discrete actions

    # Store results
    task_rewards = []

    for task_id in tqdm(range(NUM_TASKS), desc="Tasks"):
        # Create a fresh environment for each task
        env = gym.make(ENV_NAME)

        # Train on the current task
        reward = run_task(env, compo, MAX_STEPS_PER_TASK)
        task_rewards.append(reward)

        # Freeze the current module and add a new one for the next task
        compo.freeze_current()
        compo.add_module()

        env.close()

    # Save results
    results_path = os.path.join("outputs", "results.txt")
    with open(results_path, "w") as f:
        for i, r in enumerate(task_rewards, 1):
            f.write(f"Task {i} reward: {r:.2f}\n")
    print(f"Results written to {results_path}")

if __name__ == "__main__":
    main()