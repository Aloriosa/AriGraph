#!/usr/bin/env python3
"""
SAPG training loop for the discrete CartPole‑v1 environment.
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
SEED = 42
NUM_ENVS = 8192            # Large batch size (≈8k environments)
NUM_POLICIES = 2           # 1 leader + 1 follower
ENV_PER_POLICY = NUM_ENVS // NUM_POLICIES
HORIZON = 20               # Steps per rollout
TOTAL_ITERS = 200          # Training iterations
GAMMA = 0.99
LAMBDA_GAE = 0.95
CLIP_EPS = 0.2
LEADER_OFFPOL = True
ENTROPY_COEF_FOLLOWER = 0.01  # Only follower gets entropy bonus
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
    """Compute Generalised Advantage Estimation."""
    gae = 0.0
    advantages = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * LAMBDA_GAE * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages

def ppo_surrogate(logp, old_logp, adv, clip_eps):
    """Standard clipped PPO surrogate."""
    ratio = torch.exp(logp - old_logp)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    return -torch.min(ratio * adv, clipped * adv).mean()

def offpolicy_surrogate(policy, obs, act, old_logp, adv, clip_eps):
    """
    Importance‑weighted off‑policy PPO loss with clipping rule
    clip(r, μ(1‑ε), μ(1+ε))
    """
    new_logp = policy.log_prob(obs, act)
    mu = torch.exp(new_logp - old_logp)          # μ = π_i(a|s) / π_j(a|s)
    clipped_mu = torch.clamp(mu, 1 - clip_eps, 1 + clip_eps)
    return -torch.min(mu * adv, clipped_mu * adv).mean()

# --------------------------------------------------------------------------- #
#                     Training loop                                         #
# --------------------------------------------------------------------------- #
def main():
    set_seed(SEED)
    os.makedirs("outputs", exist_ok=True)

    # Vectorised environment
    env = gym.vector.make("CartPole-v1", num_envs=NUM_ENVS, asynchronous=False)
    env.seed(SEED)
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.n

    # Shared backbone
    shared_backbone = SharedBackbone(obs_dim).to(DEVICE)

    # Initialise policies and critics
    policies = [Policy(obs_dim, act_dim, shared_backbone).to(DEVICE) for _ in range(NUM_POLICIES)]
    critics  = [ValueNet(obs_dim, shared_backbone).to(DEVICE) for _ in range(NUM_POLICIES)]
    optimizers = [optim.Adam(list(p.parameters()) + list(c.parameters()), lr=LR)
                  for p, c in zip(policies, critics)]

    episode_returns = []

    for itr in tqdm(range(TOTAL_ITERS), desc="Training SAPG"):
        # Reset environments
        obs, _ = env.reset(seed=SEED + itr)
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

        # Storage per policy
        data = [[] for _ in range(NUM_POLICIES)]

        # Rollout
        for step in range(HORIZON):
            for pid in range(NUM_POLICIES):
                start = pid * ENV_PER_POLICY
                end   = (pid + 1) * ENV_PER_POLICY
                batch_obs = obs[start:end]

                # Policy forward
                act, logp, _ = policies[pid].sample(batch_obs)
                val = critics[pid](batch_obs)

                # Store
                data[pid].append((batch_obs, act, logp, val))

                # Step env for this slice
                # Convert actions to numpy (CartPole is discrete)
                env_actions = act.cpu().numpy()
                obs_new, rew, done, trunc, _ = env.step(env_actions)

                # Convert to tensors
                obs_new = torch.tensor(obs_new, dtype=torch.float32, device=DEVICE)
                rew = torch.tensor(rew, dtype=torch.float32, device=DEVICE)
                done = torch.tensor(done, dtype=torch.float32, device=DEVICE)

                # Push to storage
                for idx, (o, a, lp, v) in enumerate(data[pid][-1]):
                    data[pid][-1] = (o, a, lp, v, rew[idx], done[idx])

                # Update observation slice
                obs[start:end] = obs_new[start:end]

        # Compute returns & advantages per policy
        for pid in range(NUM_POLICIES):
            # Unzip
            obs_list, act_list, logp_list, val_list, rew_list, done_list = zip(*data[pid])
            val_next = critics[pid](obs[-ENV_PER_POLICY:])  # value of last obs
            # Convert to tensors
            val_list = torch.stack(val_list)
            rew_list = torch.stack(rew_list)
            done_list = torch.stack(done_list)
            val_next = val_next.detach()
            # GAE
            adv = compute_gae(rew_list.cpu().numpy(),
                              done_list.cpu().numpy(),
                              val_list.cpu().numpy(),
                              torch.cat([val_list[1:], val_next.unsqueeze(0)]).cpu().numpy())
            adv = torch.tensor(adv, dtype=torch.float32, device=DEVICE)
            # Normalise
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            # Replace reward with advantage for simplicity
            data[pid] = [(o, a, lp, v, adv[i]) for i, (o, a, lp, v, _, _) in enumerate(data[pid])]

        # Policy updates
        for pid in range(NUM_POLICIES):
            policy = policies[pid]
            critic = critics[pid]
            opt = optimizers[pid]

            # On‑policy data
            on_obs = torch.cat([d[0] for d in data[pid]])
            on_act = torch.cat([d[1] for d in data[pid]])
            on_logp = torch.cat([d[2] for d in data[pid]])
            on_val = torch.cat([d[3] for d in data[pid]])
            on_adv = torch.cat([d[4] for d in data[pid]])

            # Surrogate loss
            policy_loss = ppo_surrogate(policy.log_prob(on_obs, on_act),
                                        on_logp,
                                        on_adv,
                                        CLIP_EPS)

            # Value loss
            val_pred = critic(on_obs)
            val_loss = ((val_pred - (on_adv + on_val).detach()) ** 2).mean()

            total_loss = policy_loss + val_loss

            # Entropy bonus for follower only
            if pid == 1:   # follower
                entropy = -torch.mean(policy.log_prob(on_obs, on_act))
                total_loss += ENTROPY_COEF_FOLLOWER * entropy

            # Off‑policy update for leader
            if LEADER_OFFPOL and pid == 0:
                # Gather follower data
                off_obs = torch.cat([d[0] for d in data[1]])
                off_act = torch.cat([d[1] for d in data[1]])
                off_old_logp = torch.cat([d[2] for d in data[1]])
                off_adv = torch.cat([d[4] for d in data[1]])

                # Subsample to match on‑policy batch size
                if len(off_obs) > len(on_obs):
                    idx = torch.randperm(len(off_obs))[:len(on_obs)]
                    off_obs = off_obs[idx]
                    off_act = off_act[idx]
                    off_old_logp = off_old_logp[idx]
                    off_adv = off_adv[idx]

                off_loss = offpolicy_surrogate(policy,
                                               off_obs,
                                               off_act,
                                               off_old_logp,
                                               off_adv,
                                               CLIP_EPS)

                # Off‑policy critic loss
                off_val_pred = critic(off_obs)
                off_val_target = off_adv + off_val_pred
                off_val_loss = ((off_val_pred - off_val_target.detach()) ** 2).mean()

                total_loss += off_loss + off_val_loss

            # Backprop
            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
            opt.step()

        # Evaluation with leader policy
        eval_env = gym.make("CartPole-v1")
        eval_obs, _ = eval_env.reset(seed=SEED + 9999)
        done = False
        ep_ret = 0.0
        while not done:
            eval_obs_t = torch.tensor(eval_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            act, _, _ = policies[0].sample(eval_obs_t)
            act_np = act.cpu().numpy()[0]
            eval_obs, reward, done, trunc, _ = eval_env.step(act_np)
            ep_ret += reward
        episode_returns.append(ep_ret)

        if itr % 20 == 0:
            avg_ret = np.mean(episode_returns[-20:])
            print(f"[Iter {itr}] Avg return (last 20) = {avg_ret:.2f}")

    # Write results
    with open("outputs/results_sapg.txt", "w") as f:
        f.write("SAPG training complete.\n")
        f.write(f"Final average return over last 20 episodes: {np.mean(episode_returns[-20:]):.2f}\n")
        f.write(f"All episode returns: {episode_returns}\n")
    print("Training finished. Results written to outputs/results_sapg.txt")

if __name__ == "__main__":
    main()