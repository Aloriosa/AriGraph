#!/usr/bin/env python3
"""
Training driver for SAPG on HalfCheetah-v4.
"""

import os
import sys
import time
import numpy as np
from collections import deque

import torch
import gymnasium as gym
import tqdm

from sapg import (
    SAPGActorCritic,
    SAPGAgent,
    make_vec_env,
    set_seed,
)

# --------------------------------------------------------------------------- #
#                       Configuration and hyper‑parameters                    #
# --------------------------------------------------------------------------- #
SEED = 42
set_seed(SEED)

ENV_NAME = "HalfCheetah-v4"
NUM_POLICIES = 4          # M
NUM_ENVS = 256            # N
STEPS_PER_UPDATE = 16     # rollout horizon
BATCH_SIZE = 64
GAMMA = 0.99
LAMBDA = 0.95
EPS_CLIP = 0.2
LAMBDA_OFF = 1.0
LR = 5e-4
EPOCHS = 10
EVAL_INTERVAL = 2        # evaluate every 2 epochs
NUM_EVAL_EPISODES = 5

# --------------------------------------------------------------------------- #
#                          Create environments                              #
# --------------------------------------------------------------------------- #
print("Creating vectorized environments...")
envs = []
envs_per_policy = NUM_ENVS // NUM_POLICIES
for i in range(NUM_POLICIES):
    env = make_vec_env(ENV_NAME, envs_per_policy, seed=SEED + i)
    envs.append(env)

# Get observation and action dimensions
obs_space = envs[0].observation_space
act_space = envs[0].action_space
obs_dim = obs_space.shape[0]
act_dim = act_space.shape[0]

# --------------------------------------------------------------------------- #
#                          Build model & agents                            #
# --------------------------------------------------------------------------- #
print("Initializing SAPG model...")
model = SAPGActorCritic(
    obs_dim=obs_dim,
    act_dim=act_dim,
    latent_dim=32,
    num_policies=NUM_POLICIES,
).to("cuda" if torch.cuda.is_available() else "cpu")

optimizer = torch.optim.Adam(
    list(model.parameters()), lr=LR, eps=1e-5
)

agents = []
for i in range(NUM_POLICIES):
    agent = SAPGAgent(
        envs=[envs[i]],
        model=model,
        policy_idx=i,
        steps_per_update=STEPS_PER_UPDATE,
        gamma=GAMMA,
        lam=LAMBDA,
        eps_clip=EPS_CLIP,
        lambda_off=LAMBDA_OFF,
    )
    agents.append(agent)

# --------------------------------------------------------------------------- #
#                          Training loop                                   #
# --------------------------------------------------------------------------- #
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", "train.log")
log_file = open(log_path, "w")

def log(msg: str):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    epoch_losses = []

    # ----- Rollout for each policy ------------------------------------- #
    for agent in agents:
        agent.rollout()

    # ----- Compute advantages ------------------------------------------ #
    for agent in agents:
        agent.compute_advantages()

    # ----- Prepare off‑policy data for leader --------------------------- #
    leader_off_policy_data = []
    for follower in agents[1:]:
        # Build Transition object for follower data
        follower_data = {
            "obs": follower.obs_buf,
            "actions": follower.action_buf,
            "rewards": follower.reward_buf,
            "next_obs": follower.next_obs_buf,
            "dones": follower.done_buf,
            "logp": follower.logp_buf,
            "advantages": follower.advantages,
        }
        leader_off_policy_data.append(follower_data)

    # ----- Update each policy ------------------------------------------ #
    for idx, agent in enumerate(agents):
        optimizer.zero_grad()
        if idx == 0:
            loss = agent.update(leader_off_policy_data)
        else:
            loss = agent.update()
        optimizer.step()
        epoch_losses.append(loss)

    epoch_time = time.time() - epoch_start
    avg_loss = np.mean(epoch_losses)

    log(f"Epoch {epoch:02d}/{EPOCHS} | Avg loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")

    # ----- Evaluation ----------------------------------------------- #
    if epoch % EVAL_INTERVAL == 0:
        eval_env = gym.make(ENV_NAME)
        eval_env.reset(seed=SEED + 100)
        returns = []
        for _ in range(NUM_EVAL_EPISODES):
            obs, _ = eval_env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=model.device
                ).unsqueeze(0)
                with torch.no_grad():
                    dist, _ = model.forward(obs_t, policy_idx=0)
                    action = dist.mean.squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                ep_ret += reward
            returns.append(ep_ret)
        mean_ret = np.mean(returns)
        log(f"  Evaluation: mean return = {mean_ret:.2f}")

log_file.close()
print(f"Training completed. Log written to {log_path}")