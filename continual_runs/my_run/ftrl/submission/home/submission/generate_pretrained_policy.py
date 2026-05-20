#!/usr/bin/env python3
"""
Train a pre-trained policy on a source task (simplified version of Montezuma's Revenge)
This simulates the pre-training phase described in the paper
"""
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import json

class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for Atari games
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

def train_pretrained_policy(env_name, num_steps, seed, model_path):
    """
    Train a pre-trained policy on the source task
    """
    # Create environment
    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    
    # Create policy with custom CNN feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )
    
    # Initialize PPO agent
    model = PPO(
        "CnnPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        verbose=1
    )
    
    # Train the model
    print(f"Training pre-trained policy on {env_name} for {num_steps} steps...")
    model.learn(total_timesteps=num_steps)
    
    # Save the model
    model.save(model_path)
    print(f"Pre-trained model saved to {model_path}")
    
    # Save training metadata
    metadata = {
        "env_name": env_name,
        "num_steps": num_steps,
        "seed": seed,
        "model_path": model_path
    }
    
    metadata_path = model_path.replace(".pth", ".json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MontezumaRevenge-v4')
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_path', type=str, default='models/pretrained_montezuma.pth')
    
    args = parser.parse_args()
    
    train_pretrained_policy(args.env_name, args.num_steps, args.seed, args.model_path)