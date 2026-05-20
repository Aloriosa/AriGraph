"""
Training script for FRE and policy on a D4RL offline dataset.
Usage: python train_fre_and_policy.py --env antmaze-large-diverse-v2
"""

import argparse
import os
import random
import numpy as np
import torch
import gymnasium as gym
import d4rl
from tqdm import tqdm
from fre.functional_reward_encoding import FRE, RewardPrior
from rl.offline_rl import IQLAgent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(env_name: str):
    env = gym.make(env_name)
    dataset = env.get_dataset()
    return env, dataset


def sample_minibatch(dataset, batch_size: int):
    idx = np.random.choice(len(dataset['observations']), size=batch_size, replace=False)
    s = torch.tensor(dataset['observations'][idx], dtype=torch.float32)
    a = torch.tensor(dataset['actions'][idx], dtype=torch.float32)
    s_next = torch.tensor(dataset['next_observations'][idx], dtype=torch.float32)
    return s, a, s_next


def reward_prior_sample(prior: RewardPrior, dataset, batch: int, K: int):
    """
    Sample K context states from the dataset for each batch element,
    evaluate the sampled reward function on them.
    """
    state_dim = prior.state_dim
    all_idx = np.random.choice(len(dataset['observations']), size=batch * K, replace=False)
    states = torch.tensor(dataset['observations'][all_idx], dtype=torch.float32)
    states = states.reshape(batch, K, state_dim)

    rewards = torch.empty(batch, K, device=states.device)
    for b in range(batch):
        fn, _ = prior.sample()
        rewards[b] = fn(states[b])

    return states, rewards


def train_fre(env_name: str, device: torch.device):
    print(f"Training FRE on {env_name}")
    env, dataset = load_dataset(env_name)
    state_dim = dataset['observations'].shape[1]
    fre = FRE(state_dim, latent_dim=32, n_layers=4, n_heads=4, dim_feedforward=256).to(device)
    prior = RewardPrior(state_dim, device)
    optimizer = torch.optim.Adam(fre.parameters(), lr=1e-4)
    beta = 0.01
    K_enc = 32
    K_dec = 8
    steps = 150000
    batch_size = 64
    for step in tqdm(range(steps)):
        # Encoder-decoder batch
        s_enc, r_enc = reward_prior_sample(prior, dataset, batch_size, K_enc)
        s_dec, r_dec = reward_prior_sample(prior, dataset, batch_size, K_dec)
        s_enc, r_enc = s_enc.to(device), r_enc.to(device)
        s_dec, r_dec = s_dec.to(device), r_dec.to(device)

        preds, mu, logvar = fre(s_enc, r_enc, s_dec)
        mse = F.mse_loss(preds, r_dec)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = mse + beta * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10000 == 0:
            print(f"Step {step} loss {loss.item():.4f}")

    torch.save(fre.state_dict(), "fre_checkpoint.pt")
    return fre, env, dataset


def train_policy(fre: FRE, env, dataset, device: torch.device):
    print("Training policy")
    state_dim = dataset['observations'].shape[1]
    action_dim = dataset['actions'].shape[1]
    agent = IQLAgent(state_dim, action_dim, latent_dim=32, device=device)
    prior = RewardPrior(state_dim, device)
    batch_size = 256
    steps = 850000
    K_enc = 32
    for step in tqdm(range(steps)):
        # Sample reward function
        fn, _ = prior.sample()
        # Sample batch from dataset
        s, a, s_next = sample_minibatch(dataset, batch_size)
        s, a, s_next = s.to(device), a.to(device), s_next.to(device)
        # Evaluate reward
        r = fn(s).to(device)

        # Encode context: use K_enc random states from dataset
        idx = np.random.choice(len(dataset['observations']), size=K_enc, replace=False)
        s_enc = torch.tensor(dataset['observations'][idx], dtype=torch.float32, device=device)
        r_enc = fn(s_enc).to(device)
        with torch.no_grad():
            z, _, _ = fre.encode(s_enc.unsqueeze(0), r_enc.unsqueeze(0))
            z = z.squeeze(0).detach()

        agent.train_step(s, a, r, s_next, z)

        if step % 5000 == 0:
            print(f"Policy step {step}")

    torch.save({
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "value": agent.value.state_dict(),
        "policy": agent.policy.state_dict(),
    }, "policy_checkpoint.pt")
    return agent


def evaluate_agent(agent, fre, env, dataset, device, n_episodes=10):
    fre.eval()
    env = gym.make(env.unwrapped.spec.id)  # fresh env
    results = []
    prior = RewardPrior(env.observation_space.shape[0], device)
    for ep in range(n_episodes):
        # Sample a downstream task: e.g., goal-reaching
        fn, _ = prior.sample()
        # Sample 32 context states from dataset
        idx = np.random.choice(len(dataset['observations']), size=32, replace=False)
        s_ctx = torch.tensor(dataset['observations'][idx], dtype=torch.float32, device=device)
        r_ctx = fn(s_ctx).to(device)
        with torch.no_grad():
            z, _, _ = fre.encode(s_ctx.unsqueeze(0), r_ctx.unsqueeze(0))
            z = z.squeeze(0).detach()

        obs, _ = env.reset()
        total_r = 0.0
        for t in range(2000):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a = agent.policy(obs_t, z)
                a = a.squeeze(0).cpu().numpy()
            # clip action to bounds
            a = np.clip(a, env.action_space.low, env.action_space.high)
            obs, rew, terminated, truncated, _ = env.step(a)
            total_r += rew
            if terminated or truncated:
                break
        results.append(total_r)
    return np.mean(results), np.std(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="antmaze-large-diverse-v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fre, env, dataset = train_fre(args.env, device)
    agent = train_policy(fre, env, dataset, device)
    mean, std = evaluate_agent(agent, fre, env, dataset, device)
    print(f"Zero-shot evaluation mean reward: {mean:.2f} ± {std:.2f}")
    with open("output.txt", "w") as f:
        f.write(f"Zero-shot mean reward: {mean:.2f} +/- {std:.2f}\n")


if __name__ == "__main__":
    main()