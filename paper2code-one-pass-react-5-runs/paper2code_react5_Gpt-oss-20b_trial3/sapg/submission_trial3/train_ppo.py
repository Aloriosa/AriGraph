#!/usr/bin/env python3
"""
Vanilla PPO baseline for comparison with SAPG.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
from utils import SharedBackbone, Policy, ValueNet

# --------------------------------------------------------------------------- #
#                     Hyper‑parameters and constants                        #
# --------------------------------------------------------------------------- #
SEED = 43  # Different seed to avoid identical runs
NUM_ENVS = 8192
ENV_PER_POLICY = NUM_ENVS
HORIZON = 20
TOTAL_ITERS = 200
GAMMA = 0.99
LAMBDA_GAE = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.0
LR = 3e-4
GRAD_CLIP = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
#                     Utility functions                                    #
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_gae(rewards, dones, values, next_values):
    gae = 0.0
    advantages = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * LAMBDA_GAE * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages

def ppo_surrogate(logp, old_logp, adv, clip_eps):
    ratio = torch.exp(logp - old_logp)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    return -torch.min(ratio * adv, clipped * adv).mean()

# --------------------------------------------------------------------------- #
#                     Training loop                                         #
# --------------------------------------------------------------------------- #
def main():
    set_seed(SEED)
    os.makedirs("outputs", exist_ok=True)

    env = gym.vector.make("CartPole-v1", num_envs=NUM_ENVS, asynchronous=False)
    env.seed(SEED)
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.n

    shared_backbone = SharedBackbone(obs_dim).to(DEVICE)
    policy = Policy(obs_dim, act_dim, shared_backbone).to(DEVICE)
    critic = ValueNet(obs_dim, shared_backbone).to(DEVICE)
    opt = optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=LR)

    episode_returns = []

    for itr in tqdm(range(TOTAL_ITERS), desc="Training PPO"):
        obs, _ = env.reset(seed=SEED + itr)
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

        data = []

        for step in range(HORIZON):
            act, logp, _ = policy.sample(obs)
            val = critic(obs)

            data.append((obs, act, logp, val))

            obs_new, rew, done, trunc, _ = env.step(act.cpu().numpy())
            obs_new = torch.tensor(obs_new, dtype=torch.float32, device=DEVICE)
            rew = torch.tensor(rew, dtype=torch.float32, device=DEVICE)
            done = torch.tensor(done, dtype=torch.float32, device=DEVICE)

            data.append((obs_new, act, logp, val, rew, done))
            obs = obs_new

        # Compute returns & advantages
        obs_list, act_list, logp_list, val_list, rew_list, done_list = zip(*data)
        val_list = torch.stack(val_list)
        rew_list = torch.stack(rew_list)
        done_list = torch.stack(done_list)
        val_next = critic(obs[-ENV_PER_POLICY:]).detach()
        adv = compute_gae(rew_list.cpu().numpy(),
                          done_list.cpu().numpy(),
                          val_list.cpu().numpy(),
                          torch.cat([val_list[1:], val_next.unsqueeze(0)]).cpu().numpy())
        adv = torch.tensor(adv, dtype=torch.float32, device=DEVICE)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Policy update
        on_obs = torch.cat([d[0] for d in data])
        on_act = torch.cat([d[1] for d in data])
        on_logp = torch.cat([d[2] for d in data])
        on_adv = adv

        policy_loss = ppo_surrogate(policy.log_prob(on_obs, on_act),
                                    on_logp,
                                    on_adv,
                                    CLIP_EPS)

        val_pred = critic(on_obs)
        val_loss = ((val_pred - (on_adv + val_pred).detach()) ** 2).mean()

        entropy = -torch.mean(policy.log_prob(on_obs, on_act))
        total_loss = policy_loss + val_loss + ENTROPY_COEF * entropy

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
        opt.step()

        # Evaluation
        eval_env = gym.make("CartPole-v1")
        eval_obs, _ = eval_env.reset(seed=SEED + 9999)
        done = False
        ep_ret = 0.0
        while not done:
            eval_obs_t = torch.tensor(eval_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            act, _, _ = policy.sample(eval_obs_t)
            act_np = act.cpu().numpy()[0]
            eval_obs, reward, done, trunc, _ = eval_env.step(act_np)
            ep_ret += reward
        episode_returns.append(ep_ret)

        if itr % 20 == 0:
            avg_ret = np.mean(episode_returns[-20:])
            print(f"[Iter {itr}] Avg return (last 20) = {avg_ret:.2f}")

    # Write results
    with open("outputs/results_ppo.txt", "w") as f:
        f.write("PPO training complete.\n")
        f.write(f"Final average return over last 20 episodes: {np.mean(episode_returns[-20:]):.2f}\n")
        f.write(f"All episode returns: {episode_returns}\n")
    print("Training finished. Results written to outputs/results_ppo.txt")

if __name__ == "__main__":
    main()