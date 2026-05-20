# -*- coding: utf-8 -*-
"""
Standalone evaluation script for a saved FRE encoder.
"""

import argparse
import os

import torch
import gymnasium as gym

from fre import FREEncoder, FREDecoder, DEVICE

def load_encoder(decoder, path):
    encoder = FREEncoder(decoder.state_dim).to(DEVICE)
    encoder.load_state_dict(torch.load(path, map_location=DEVICE))
    encoder.eval()
    return encoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-path", type=str, required=True)
    parser.add_argument("--env", type=str, default="antmaze-large-diverse-v2")
    args = parser.parse_args()

    env = gym.make(args.env)
    dataset = env.get_dataset()
    obs = dataset["observations"]
    state_dim = obs.shape[1]

    # Dummy decoder to get state_dim
    decoder = FREDecoder(state_dim).to(DEVICE)
    encoder = load_encoder(decoder, args.encoder_path)

    # Evaluate
    eval_results = {}
    for name in ["Goal‑reaching", "Linear reward", "MLP reward"]:
        # Sample reward function from prior
        reward_prior = RewardPrior(obs)
        if name == "Goal‑reaching":
            reward_fn = reward_prior.sample_goal_reward()
        elif name == "Linear reward":
            reward_fn = reward_prior.sample_linear_reward()
        else:
            reward_fn = reward_prior.sample_mlp_reward()

        # Sample 64 states
        eval_idx = np.random.choice(len(obs), size=64, replace=False)
        eval_states = torch.tensor(obs[eval_idx], dtype=torch.float32, device=DEVICE)
        true_rewards = torch.tensor(reward_fn(eval_states.cpu().numpy()), dtype=torch.float32, device=DEVICE)

        # Encode with random context
        ctx_idx = np.random.choice(len(obs), size=32, replace=False)
        ctx_states = torch.tensor(obs[ctx_idx], dtype=torch.float32, device=DEVICE)
        ctx_rewards_np = reward_fn(ctx_states.cpu().numpy())
        ctx_rewards = torch.tensor(ctx_rewards_np, dtype=torch.float32, device=DEVICE)

        mean, logvar = encoder(ctx_states.unsqueeze(0), ctx_rewards.unsqueeze(0))
        z = reparameterise(mean, logvar).squeeze(0)

        # Predict
        pred_rewards = decoder(eval_states.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
        mse = torch.mean((pred_rewards - true_rewards) ** 2).item()
        eval_results[name] = mse

    print("=== Evaluation Results ===")
    for name, mse in eval_results.items():
        print(f"{name:15s} : {mse:.4f}")

if __name__ == "__main__":
    main()