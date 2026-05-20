#!/usr/bin/env python3
"""
Evaluate all fine-tuned policies against baselines
"""
import torch
import numpy as np
import gymnasium as gym
import argparse
import os
import csv
import random
from stable_baselines3 import PPO

def evaluate_policy(env_name, policy_path, num_episodes=100, seed=42, deterministic=True):
    """
    Evaluate a policy on the specified environment.
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Load environment
    env = gym.make(env_name)
    
    # Load policy
    policy = PPO.load(policy_path)
    
    # Evaluate
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = policy.predict(state, deterministic=deterministic)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    return mean_reward, std_reward, episode_rewards

def evaluate_baselines(env_name, 
                      pretrained_path, 
                      vanilla_path, 
                      bc_path, 
                      ewc_path, 
                      ks_path, 
                      scratch_path,
                      num_episodes=100, 
                      num_seeds=5, 
                      output='results/baseline_comparison.csv'):
    """
    Evaluate all baseline methods
    """
    results = []
    
    # Define the policies to evaluate
    policies = [
        ("Pretrained", pretrained_path),
        ("Vanilla Fine-tuning", vanilla_path),
        ("BC Fine-tuning", bc_path),
        ("EWC Fine-tuning", ewc_path),
        ("KS Fine-tuning", ks_path),
        ("Training from Scratch", scratch_path)
    ]
    
    # Evaluate each policy with multiple seeds
    for method_name, policy_path in policies:
        if not os.path.exists(policy_path):
            print(f"Warning: Policy file not found: {policy_path}")
            continue
            
        all_mean_rewards = []
        all_std_rewards = []
        
        for seed in range(num_seeds):
            mean_reward, std_reward, _ = evaluate_policy(
                env_name, policy_path, num_episodes, seed, deterministic=True
            )
            all_mean_rewards.append(mean_reward)
            all_std_rewards.append(std_reward)
        
        # Compute mean and std across seeds
        mean_mean_reward = np.mean(all_mean_rewards)
        std_mean_reward = np.std(all_mean_rewards)
        
        # Use the mean of stds as the std for the mean
        mean_std_reward = np.mean(all_std_rewards)
        
        results.append({
            'method': method_name,
            'mean_reward': mean_mean_reward,
            'std_mean_reward': std_mean_reward,
            'mean_std_reward': mean_std_reward,
            'all_rewards': all_mean_rewards
        })
        
        print(f"{method_name}: {mean_mean_reward:.2f} ± {std_mean_reward:.2f}")
    
    # Save results to CSV
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'method', 
            'mean_reward', 
            'std_mean_reward', 
            'mean_std_reward',
            'all_rewards'
        ])
        
        for result in results:
            writer.writerow([
                result['method'],
                result['mean_reward'],
                result['std_mean_reward'],
                result['mean_std_reward'],
                ','.join([str(r) for r in result['all_rewards']])
            ])
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MontezumaRevenge-v4')
    parser.add_argument('--pretrained_path', type=str, required=True)
    parser.add_argument('--vanilla_path', type=str, required=True)
    parser.add_argument('--bc_path', type=str, required=True)
    parser.add_argument('--ewc_path', type=str, required=True)
    parser.add_argument('--ks_path', type=str, required=True)
    parser.add_argument('--scratch_path', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--output', type=str, default='results/baseline_comparison.csv')
    
    args = parser.parse_args()
    
    evaluate_baselines(
        args.env_name,
        args.pretrained_path,
        args.vanilla_path,
        args.bc_path,
        args.ewc_path,
        args.ks_path,
        args.scratch_path,
        args.num_episodes,
        args.num_seeds,
        args.output
    )