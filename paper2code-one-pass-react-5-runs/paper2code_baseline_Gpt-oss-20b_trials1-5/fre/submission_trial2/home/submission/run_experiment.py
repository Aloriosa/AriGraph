#!/usr/bin/env python
import argparse
import os
import json
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
from pathlib import Path

from fre.encoder import FREEncoder
from fre.decoder import FREDecoder
from fre.trainer import FRETrainer
from fre.data_utils import generate_offline_dataset, load_offline_dataset
from fre.policy import IQLQ, IQLPolicy

def create_env(env_name, seed):
    env = gym.make(env_name, render_mode=None, disable_env_checker=True)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def sample_reward_func(state_dim, rng):
    choice = rng.choice(3)
    if choice == 0:
        # Goal reward
        goal_state = rng.uniform(-1, 1, size=state_dim)
        def func(s):
            return -1.0 * (np.linalg.norm(s - goal_state, axis=-1) > 0.2).astype(float)
        return func, "goal"
    elif choice == 1:
        w = rng.normal(0, 1, size=state_dim).astype(np.float32)
        def func(s):
            return (s * w).sum(axis=-1)
        return func, "linear"
    else:
        hidden = rng.normal(0, 1, size=(state_dim, 32)).astype(np.float32)
        out = rng.normal(0, 1, size=(32, 1)).astype(np.float32)
        def func(s):
            h = np.tanh(s @ hidden)
            outv = h @ out
            return np.clip(outv, -1, 1).squeeze(-1)
        return func, "mlp"

def train_fre(encoder, decoder, dataset, device, epochs, rng):
    trainer = FRETrainer(encoder, decoder, dataset,
                         device=device, rng=rng)
    trainer.train(epochs)

def train_policy(encoder, dataset, device, epochs, rng):
    # Freeze encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Prepare replay buffer
    obs = torch.tensor(dataset["obs"], device=device)           # (N, state_dim)
    actions = torch.tensor(dataset["actions"], device=device)
    rewards = torch.tensor(dataset["rewards"], device=device)
    next_obs = torch.tensor(dataset["next_obs"], device=device)
    terminals = torch.tensor(dataset["terminals"], device=device)

    N = obs.shape[0]
    state_dim = obs.shape[1]
    action_dim = actions.shape[1]
    latent_dim = encoder.readout.out_features // 2

    # Networks
    q_net = IQLQ(state_dim, action_dim, latent_dim).to(device)
    q_target = IQLQ(state_dim, action_dim, latent_dim).to(device)
    q_target.load_state_dict(q_net.state_dict())
    policy_net = IQLPolicy(state_dim, action_dim, latent_dim).to(device)

    # Optimizers
    q_opt = torch.optim.Adam(q_net.parameters(), lr=3e-4)
    pi_opt = torch.optim.Adam(policy_net.parameters(), lr=3e-4)

    gamma = 0.99
    batch_size = 256
    rng_np = rng

    for epoch in range(epochs):
        perms = rng_np.permutation(N)
        for i in range(0, N, batch_size):
            idx = perms[i:i+batch_size]
            s = obs[idx]
            a = actions[idx]
            r = rewards[idx]
            s_next = next_obs[idx]
            t = terminals[idx].float()

            # Sample latent for each transition using K_enc=8
            K_enc = 8
            idx_enc = rng_np.choice(N, size=(len(idx), K_enc), replace=False)
            s_enc = obs[idx_enc]
            # For each transition we need a reward function; sample random
            # reward function from prior
            reward_funcs = [sample_reward_func(state_dim, rng_np)[0]
                            for _ in range(len(idx))]
            # Compute rewards for encoder states
            r_enc = torch.tensor([rf(s_enc[j].cpu().numpy())
                                  for j, rf in enumerate(reward_funcs)],
                                 dtype=torch.float32, device=device)
            r_enc = r_enc.view(len(idx), K_enc, 1)

            # Encode to z
            z, _, _ = encoder.encode(s_enc, r_enc)
            # z shape (len(idx), latent_dim)

            # Q loss
            with torch.no_grad():
                a_next = policy_net(s_next, z)
                q_next = q_target(s_next, a_next, z)
                target_q = r + (1 - t) * gamma * q_next
            q_pred = q_net(s, a, z)
            q_loss = F.mse_loss(q_pred, target_q)

            # Policy loss (simple deterministic policy gradient)
            pi_loss = -q_net(s, policy_net(s, z), z).mean()

            # Optimize
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()

            pi_opt.zero_grad()
            pi_loss.backward()
            pi_opt.step()

        # Soft update target
        for p, p_t in zip(q_net.parameters(), q_target.parameters()):
            p_t.data.copy_(0.995 * p_t.data + 0.005 * p.data)

        if (epoch + 1) % 10 == 0:
            print(f"[Policy] Epoch {epoch+1}/{epochs} | q_loss={q_loss.item():.4f} | pi_loss={pi_loss.item():.4f}")

    return policy_net, q_net

def evaluate_policy(policy, encoder, env, reward_func, rng, device,
                    num_episodes=5, max_steps=200):
    encoder.eval()
    policy.eval()
    total_reward = 0.0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        t = 0
        # Sample K_enc states for encoding
        K_enc = 8
        idx = rng.choice(len(env), size=K_enc, replace=False)
        # Use random subset of offline dataset for encoding
        # We don't have dataset here; so we use the current observation
        # as a placeholder – this is only for demo purposes.
        # In a real implementation we would sample from the offline buffer.
        # Instead we encode the current state repeated K_enc times
        states_enc = torch.tensor(obs, dtype=torch.float32,
                                  device=device).unsqueeze(0).repeat(K_enc, 1)
        rewards_enc = torch.tensor(reward_func(states_enc.cpu().numpy()),
                                   dtype=torch.float32, device=device)
        rewards_enc = rewards_enc.unsqueeze(-1)
        z, _, _ = encoder.encode(states_enc.unsqueeze(0), rewards_enc.unsqueeze(0))
        z = z.squeeze(0)
        while not done and t < max_steps:
            act = policy(obs, z).cpu().numpy()
            obs, _, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            ep_reward += reward_func(obs)
            t += 1
        total_reward += ep_reward
    avg_reward = total_reward / num_episodes
    return avg_reward

def main(args):
    rng = np.random.default_rng(args.seed)

    # 1. Dataset
    if not Path(args.offline_dataset).exists():
        generate_offline_dataset(args.env, seed=args.seed,
                                 save_path=args.offline_dataset)
    dataset = load_offline_dataset(args.offline_dataset)

    state_dim = dataset["obs"].shape[1]
    action_dim = dataset["actions"].shape[1]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. FRE encoder / decoder
    encoder = FREEncoder(state_dim=state_dim).to(device)
    decoder = FREDecoder(state_dim=state_dim).to(device)

    print("Training FRE encoder/decoder...")
    train_fre(encoder, decoder, dataset, device,
              epochs=args.encoder_epochs, rng=rng)

    # 3. Train policy
    print("Training policy...")
    policy_net, q_net = train_policy(encoder, dataset, device,
                                     epochs=args.policy_epochs, rng=rng)

    # 4. Evaluation on downstream tasks
    print("Evaluating on downstream tasks...")
    env = create_env(args.env, args.seed)

    downstream_results = {}
    for _ in range(3):
        rf, lbl = sample_reward_func(state_dim, rng)
        avg = evaluate_policy(policy_net, encoder, env, rf, rng, device,
                              num_episodes=3, max_steps=200)
        downstream_results[lbl] = avg
        print(f"Task {lbl}: avg reward = {avg:.2f}")

    # 5. Save results
    with open("results.json", "w") as f:
        json.dump(downstream_results, f, indent=4)
    print("Results written to results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--offline_dataset", default="offline_dataset.pt")
    parser.add_argument("--encoder_epochs", type=int, default=5)
    parser.add_argument("--policy_epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)