import gymnasium as gym
import torch
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.utils import set_seed, get_action_space_dims
from src.mask_network import MaskNetwork
from tqdm import tqdm

set_seed(43)

env_id = "Hopper-v3"
env = DummyVecEnv([lambda: gym.make(env_id, render_mode=None)])
obs_dim, act_dim = get_action_space_dims(env)
mask_net = MaskNetwork(obs_dim).to("cpu")
optimizer = torch.optim.Adam(mask_net.parameters(), lr=1e-3)

# Load pre‑trained policy
pretrained = PPO.load("pretrained.zip", env=env, device="cpu")

# Hyper‑parameters
episodes = 200
alpha = 0.001      # intrinsic bonus for masking
gamma = 0.99

# Running mean of total reward (baseline)
baseline = 0.0
alpha_baseline = 0.0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train():
    global baseline, alpha_baseline
    mask_net.train()
    all_mask_probs = []
    for ep in tqdm(range(episodes), desc="Mask training"):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_alpha = 0.0
        log_probs = []
        masks = []
        states = []
        while not done:
            state = torch.tensor(obs, dtype=torch.float32)
            p_mask = mask_net(state)
            mask_action = torch.bernoulli(p_mask).item()
            # Sample action from pre‑trained policy
            act, _ = pretrained.predict(obs, deterministic=True)
            if mask_action == 1:
                # blind – replace with random action
                act = env.action_space.sample()
            # Step
            next_obs, reward, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            ep_reward += reward
            ep_alpha += alpha * mask_action
            # Log probability
            log_prob = torch.log(p_mask + 1e-8) if mask_action==1 else torch.log(1-p_mask + 1e-8)
            log_probs.append(log_prob)
            masks.append(mask_action)
            states.append(state)
            obs = next_obs
        # Compute advantage
        total = ep_reward + ep_alpha
        baseline = 0.9 * baseline + 0.1 * total
        advantage = total - baseline
        # Policy gradient update
        loss = 0
        for lp, m in zip(log_probs, masks):
            loss -= lp * advantage
        loss /= len(log_probs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Record mask probs
        probs = [p_mask.item() for p_mask in mask_net(torch.tensor(obs, dtype=torch.float32)).unsqueeze(0)]
        all_mask_probs.append(np.mean(probs))
    print(f"Mask training finished: mean mask prob = {np.mean(all_mask_probs):.3f}")
    torch.save(mask_net.state_dict(), "mask.pt")

if __name__ == "__main__":
    train()