import gymnasium as gym
import torch
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.env_wrapper import RNDWrapper
from src.utils import set_seed, get_action_space_dims
from tqdm import tqdm

set_seed(44)

env_id = "Hopper-v3"
# Wrap env with RND
base_env = gym.make(env_id, render_mode=None)
wrapped_env = RNDWrapper(base_env, intrinsic_coef=0.01, device="cpu")
# DummyVecEnv for SB3
env = DummyVecEnv([lambda: wrapped_env])

obs_dim, act_dim = get_action_space_dims(env)

# Load pre‑trained policy and mask network
pretrained = PPO.load("pretrained.zip", env=env, device="cpu")
mask_net = torch.load("mask.pt", map_location="cpu")
# For simplicity we only use mask_net for selecting critical states

# Generate a pool of critical states
critical_states = []
mask_net.eval()
episodes_for_crit = 10
for _ in range(episodes_for_crit):
    obs, _ = env.reset()
    done = False
    while not done:
        state = torch.tensor(obs, dtype=torch.float32)
        p_mask = mask_net(state).item()
        # If mask probability high, store state
        if p_mask > 0.7:
            # Store MuJoCo state (qpos, qvel)
            critical_states.append(base_env.env.get_state())
        # Step with pre‑trained policy
        act, _ = pretrained.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(act)
        done = terminated or truncated

print(f"Collected {len(critical_states)} critical states.")

# Training refined agent with mixed initial distribution
model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device="cpu"
)

# Custom reset logic: we wrap the env's reset method
original_reset = env.env.reset
def mixed_reset(seed=None):
    if random.random() < 0.5 and critical_states:
        state = random.choice(critical_states)
        return env.env.reset_to_state(state), {}
    else:
        return original_reset(seed=seed)
env.env.reset = mixed_reset

timesteps = 200_000
model.learn(total_timesteps=timesteps, log_interval=10)
model.save("refined.zip")

# Evaluate refined agent
mean_reward = 0
for _ in range(10):
    obs, _ = env.reset()
    done = False
    total = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += reward
    mean_reward += total
print(f"Refinement finished: mean reward = {mean_reward/10:.2f}")