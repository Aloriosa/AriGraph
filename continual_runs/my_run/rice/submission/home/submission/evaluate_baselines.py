import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import random
import pickle
import csv

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
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Load policy
    policy = torch.load(policy_path, map_location='cpu')
    policy.eval()
    
    # Evaluate
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                if hasattr(policy, 'get_action'):
                    action = policy.get_action(state_tensor, deterministic=deterministic)
                else:
                    # For PPO policy
                    mean, std = policy(state_tensor)
                    if deterministic:
                        action = mean
                    else:
                        action_dist = torch.distributions.Normal(mean, std)
                        action = action_dist.sample()
            
            action = action.squeeze().numpy()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    return mean_reward, std_reward, episode_rewards

def evaluate_baselines(env_name, pretrained_policy_path, rice_policy_path, num_episodes=100, num_seeds=5, output="results/baseline_comparison.csv"):
    """
    Evaluate RICE against baselines: no refinement, random sampling, pure critical states.
    """
    baselines = [
        ("No Refinement", pretrained_policy_path),
        ("RICE", rice_policy_path)
    ]
    
    results = []
    
    for name, policy_path in baselines:
        mean_rewards = []
        std_rewards = []
        
        for seed in range(num_seeds):
            mean_reward, std_reward, _ = evaluate_policy(env_name, policy_path, num_episodes, seed)
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)
        
        overall_mean = np.mean(mean_rewards)
        overall_std = np.std(mean_rewards)
        
        results.append({
            'method': name,
            'mean_reward': overall_mean,
            'std_reward': overall_std,
            'mean_rewards': mean_rewards,
            'std_rewards': std_rewards
        })
    
    # Write results to CSV
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'mean_reward', 'std_reward', 'mean_rewards', 'std_rewards'])
        
        for result in results:
            writer.writerow([
                result['method'],
                result['mean_reward'],
                result['std_reward'],
                ','.join([f"{r:.4f}" for r in result['mean_rewards']]),
                ','.join([f"{r:.4f}" for r in result['std_rewards']])
            ])
    
    # Print summary
    print("\nBaseline Comparison Results:")
    print("-" * 50)
    for result in results:
        print(f"{result['method']}: {result['mean_reward']:.4f} ± {result['std_reward']:.4f}")
    
    # Calculate statistical significance (t-test between RICE and No Refinement)
    rice_rewards = [r['mean_rewards'] for r in results if r['method'] == 'RICE'][0]
    no_refinement_rewards = [r['mean_rewards'] for r in results if r['method'] == 'No Refinement'][0]
    
    # Simple t-test approximation
    diff_mean = np.mean(rice_rewards) - np.mean(no_refinement_rewards)
    diff_std = np.sqrt(np.var(rice_rewards)/len(rice_rewards) + np.var(no_refinement_rewards)/len(no_refinement_rewards))
    t_stat = diff_mean / diff_std if diff_std > 0 else 0
    p_value = 2 * (1 - 0.5 * (1 + np.erf(np.abs(t_stat) / np.sqrt(2))))
    
    print(f"\nStatistical Significance (RICE vs No Refinement):")
    print(f"Mean difference: {diff_mean:.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Significant (p < 0.05): {p_value < 0.05}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--pretrained_policy_path', type=str, default='models/pretrained_ppo.pth')
    parser.add_argument('--rice_policy_path', type=str, default='results/rice_refined_policy.pth')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--output', type=str, default='results/baseline_comparison.csv')
    
    args = parser.parse_args()
    
    evaluate_baselines(
        args.env_name,
        args.pretrained_policy_path,
        args.rice_policy_path,
        args.num_episodes,
        args.num_seeds,
        args.output
    )