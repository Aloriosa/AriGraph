#!/usr/bin/env python3
"""
Minimal SAPG training script on CartPole-v1.
"""

import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch
from torch import optim

from env_utils import make_cartpole_vec_env
from policy import SAPGPolicy
from utils import RolloutBuffer, trw_clip

# ---------- Hyperparameters ----------
DEFAULT_NUM_POLICIES = 4
DEFAULT_BLOCK_SIZE = 16
DEFAULT_HORIZON = 128          # steps per policy per update
DEFAULT_GAMMA = 0.99
DEFAULT_LAMBDA = 0.95
DEFAULT_CLIP = 0.2
DEFAULT_LR = 3e-4
DEFAULT_EPOCHS = 4
DEFAULT_BATCH_SIZE = 64
DEFAULT_OFF_RATIO = 1.0        # fraction of off‑policy data to use
DEFAULT_ENTROPY_COEF = 0.0
DEFAULT_VALUE_COEF = 0.5
DEFAULT_CLIP_EPS = 0.2
DEFAULT_ENTROPY_COEF_FOLLOWER = 0.0
DEFAULT_LEADER_FOLLOWER = True
DEFAULT_LOG_INTERVAL = 10

torch.set_num_threads(2)  # keep CPU usage moderate


def parse_args():
    parser = argparse.ArgumentParser(description="SAPG on CartPole")
    parser.add_argument("--num_policies", type=int, default=DEFAULT_NUM_POLICIES,
                        help="Number of sub‑policies (leader + followers).")
    parser.add_argument("--off_ratio", type=float, default=DEFAULT_OFF_RATIO,
                        help="Fraction of off‑policy data used in leader update.")
    parser.add_argument("--entropy_coef", type=float, default=DEFAULT_ENTROPY_COEF,
                        help="Entropy coefficient for all policies.")
    parser.add_argument("--entropy_coef_follower", type=float,
                        default=DEFAULT_ENTROPY_COEF_FOLLOWER,
                        help="Entropy coefficient for follower policies.")
    parser.add_argument("--leader_follower", action="store_true",
                        default=DEFAULT_LEADER_FOLLOWER,
                        help="Enable leader‑follower aggregation.")
    parser.add_argument("--iterations", type=int, default=2000,
                        help="Number of training updates.")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help="Steps per policy per update.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Mini‑batch size for optimizer.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Epochs per update.")
    parser.add_argument("--log_interval", type=int, default=DEFAULT_LOG_INTERVAL,
                        help="Logging interval.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create vectorised environment
    total_envs = args.num_policies * DEFAULT_BLOCK_SIZE
    env = make_cartpole_vec_env(total_envs)
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.n  # discrete; we treat as 1‑dim Gaussian for demo

    # Convert discrete actions to continuous by embedding (0 or 1)
    act_dim_cont = 1

    # Initialize policies
    latent_dim = 16
    policies = []
    latents = []
    for _ in range(args.num_policies):
        pol = SAPGPolicy(obs_dim, act_dim_cont, latent_dim=latent_dim).to(device)
        policies.append(pol)
        lat = torch.randn(latent_dim, device=device, requires_grad=False)
        latents.append(lat)

    # Optimizer collects all parameters
    all_params = []
    for pol in policies:
        all_params += list(pol.parameters())
    optimizer = optim.Adam(all_params, lr=args.lr)

    # Training loop
    start_time = time.time()
    for it in range(1, args.iterations + 1):
        # Collect data for each policy
        buffers = [RolloutBuffer(obs_dim, act_dim_cont) for _ in range(args.num_policies)]

        for step in range(args.horizon):
            # Sample actions for all policies in their blocks
            obs = env.reset()[0] if step == 0 else obs
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

            # Split envs among policies
            obs_blocks = torch.split(obs_tensor, DEFAULT_BLOCK_SIZE, dim=0)
            actions_list = []
            logp_list = []
            values_list = []
            latents_list = []

            for i, (pol, lat, ob_block) in enumerate(zip(policies, latents, obs_blocks)):
                mean, std, val = pol.forward(ob_block, lat.expand(DEFAULT_BLOCK_SIZE, -1))
                dist = torch.distributions.Normal(mean, std)
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
                actions_list.append(act.cpu())
                logp_list.append(logp.cpu())
                values_list.append(val.cpu())
                latents_list.append(lat.expand(DEFAULT_BLOCK_SIZE, -1).cpu())

                # Write to buffer
                buffers[i].add(
                    obs[step * DEFAULT_BLOCK_SIZE: (step + 1) * DEFAULT_BLOCK_SIZE],
                    act.cpu().numpy(),
                    0.0,  # CartPole reward handled separately
                    np.zeros(DEFAULT_BLOCK_SIZE, dtype=np.float32),
                    val.cpu().numpy(),
                    logp.cpu().numpy()
                )

            # Step the environments
            actions_np = np.stack([a.squeeze(-1).numpy() for a in actions_list], axis=1)
            actions_np = actions_np.astype(int)  # discrete action
            next_obs, rewards, dones, infos = env.step(actions_np)
            # Record rewards and dones
            for i in range(args.num_policies):
                buffers[i].rewards[-1] = rewards[i * DEFAULT_BLOCK_SIZE:(i + 1) * DEFAULT_BLOCK_SIZE]
                buffers[i].dones[-1] = dones[i * DEFAULT_BLOCK_SIZE:(i + 1) * DEFAULT_BLOCK_SIZE]

        # After horizon, compute advantages and returns
        for buf in buffers:
            buf.finish_episode(gamma=args.gamma, lam=args.lambda)

        # Build combined dataset
        all_obs = torch.cat([buf.obs for buf in buffers], dim=0)
        all_actions = torch.cat([buf.actions for buf in buffers], dim=0)
        all_logp = torch.cat([buf.log_probs for buf in buffers], dim=0)
        all_ret = torch.cat([buf.returns for buf in buffers], dim=0)
        all_adv = torch.cat([buf.advantages for buf in buffers], dim=0)

        # Split indices by policy block
        idxs = torch.arange(all_obs.size(0))
        block_size = DEFAULT_BLOCK_SIZE * args.horizon
        policy_blocks = torch.split(idxs, block_size, dim=0)

        # Optimization steps
        for epoch in range(args.epochs):
            # Shuffle indices
            perm = torch.randperm(all_obs.size(0))
            for start in range(0, all_obs.size(0), args.batch_size):
                end = start + args.batch_size
                batch_idx = perm[start:end]

                obs_batch = all_obs[batch_idx]
                act_batch = all_actions[batch_idx]
                logp_old_batch = all_logp[batch_idx]
                ret_batch = all_ret[batch_idx]
                adv_batch = all_adv[batch_idx]

                # Compute current logp, entropy, value
                # Determine which policy block this batch belongs to
                # For simplicity we compute loss for each policy separately
                loss = 0.0
                for i, pol in enumerate(policies):
                    block_idx = policy_blocks[i]
                    mask = torch.isin(batch_idx, block_idx)
                    if mask.sum() == 0:
                        continue
                    obs_p = obs_batch[mask]
                    act_p = act_batch[mask]
                    logp_old_p = logp_old_batch[mask]
                    ret_p = ret_batch[mask]
                    adv_p = adv_batch[mask]

                    lat = latents[i].expand(obs_p.size(0), -1).to(device)

                    logp, entropy, val = pol.evaluate_actions(obs_p, lat, act_p)

                    # PPO surrogate loss
                    ratio = torch.exp(logp - logp_old_p)
                    surr1 = ratio * adv_p
                    surr2 = trw_clip(ratio, args.clip) * adv_p
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = F.mse_loss(val, ret_p)

                    # Entropy bonus
                    entropy_loss = -entropy.mean() * args.entropy_coef
                    if i != 0:  # follower
                        entropy_loss = -entropy.mean() * args.entropy_coef_follower

                    # Off‑policy loss for leader
                    if args.leader_follower and i == 0:
                        # Aggregate data from all followers
                        off_logp_sum = 0.0
                        off_count = 0
                        for j in range(1, args.num_policies):
                            # follower j data
                            idx_j = policy_blocks[j]
                            mask_j = torch.isin(batch_idx, idx_j)
                            if mask_j.sum() == 0:
                                continue
                            obs_j = obs_batch[mask_j]
                            act_j = act_batch[mask_j]
                            lat_j = latents[j].expand(obs_j.size(0), -1).to(device)

                            # Importance weight: pi_leader / pi_follower
                            logp_leader, _, _ = pol.evaluate_actions(obs_j, lat, act_j)
                            logp_follower, _, _ = policies[j].evaluate_actions(obs_j, lat_j, act_j)
                            ratio_off = torch.exp(logp_leader - logp_follower)

                            # Clip ratio
                            ratio_off_clipped = trw_clip(ratio_off, args.clip)
                            adv_j = adv_batch[mask_j]

                            off_logp_sum += (ratio_off_clipped * adv_j).mean()
                            off_count += 1
                        if off_count > 0:
                            off_policy_loss = -off_logp_sum / off_count
                            policy_loss += args.off_ratio * off_policy_loss

                    # Total loss for this policy
                    total_policy_loss = policy_loss + args.value_coef * value_loss + entropy_loss
                    loss += total_policy_loss

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 0.5)
                optimizer.step()

        # Logging
        if it % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"[{it}/{args.iterations}] "
                  f"Elapsed: {elapsed:.1f}s "
                  f"Mean Return: {all_ret.mean().item():.2f}")

    # Final evaluation
    eval_rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        while not done:
            # Use leader policy for evaluation
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            mean, std, _ = policies[0].forward(obs_t, latents[0].expand(1, -1))
            dist = torch.distributions.Normal(mean, std)
            act = dist.sample()
            act_np = int(act.squeeze().item())
            obs, r, done, _ = env.step([act_np] * total_envs)
            total_r += r[0]
        eval_rewards.append(total_r)
    mean_eval = np.mean(eval_rewards)
    print(f"Training finished. Final mean episode reward: {mean_eval:.2f}")

    # Save results
    results = {
        "final_mean_reward": float(mean_eval),
        "iterations": args.iterations,
        "env_name": "CartPole-v1",
        "num_policies": args.num_policies
    }
    with open("results.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print("Results written to results.json")


if __name__ == "__main__":
    main()