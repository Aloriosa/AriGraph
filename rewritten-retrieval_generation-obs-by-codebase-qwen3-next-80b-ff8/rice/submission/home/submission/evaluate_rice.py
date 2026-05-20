#!/usr/bin/env python3
"""
Evaluation script for RICE algorithm
"""
import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pickle
import csv
import time

def load_env(env_name, normalize=True):
    """Load environment with normalization"""
    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])
    
    if normalize:
        # Try to load saved normalization parameters
        if os.path.exists('env_stats.pkl'):
            env = VecNormalize.load('env_stats.pkl', env)
            env.training = False
            env.norm_reward = False
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
            
    return env

def evaluate_model(model_path, env_name, episodes=100, output_file="./evaluation_results.csv"):
    """Evaluate RICE model"""
    # Load environment
    env = load_env(env_name)
    
    # Load model
    model = PPO.load(model_path, env=env)
    
    # Evaluate policy
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Handle VecEnv
            if isinstance(done, (list, np.ndarray)):
                done = done[0]
                if isinstance(info, list):
                    info = info[0]
                    
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if (episode + 1) % 10 == 0:
            print(f"Completed {episode + 1}/{episodes} episodes")
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    # Save results
    results = {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'std_length': std_length,
        'episodes': episodes,
        'model_path': model_path,
        'env_name': env_name,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['mean_reward', mean_reward])
        writer.writerow(['std_reward', std_reward])
        writer.writerow(['mean_length', mean_length])
        writer.writerow(['std_length', std_length])
        writer.writerow(['episodes', episodes])
        
        # Write episode-by-episode results
        writer.writerow([''])
        writer.writerow(['episode', 'reward', 'length'])
        for i in range(len(episode_rewards)):
            writer.writerow([i+1, episode_rewards[i], episode_lengths[i]])
    
    print(f"Evaluation completed:")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean length: {mean_length:.2f} +/- {std_length:.2f}")
    print(f"Results saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate RICE model')
    parser.add_argument('--env', type=str, default='HalfCheetah-v4', help='Environment name')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--output', type=str, default='./evaluation_results.csv', help='Output file path')
    
    args = parser.parse_args()
    
    # Evaluate model
    results = evaluate_model(args.model_path, args.env, args.episodes, args.output)
    
if __name__ == "__main__":
    main()