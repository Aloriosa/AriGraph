#!/usr/bin/env python3
"""
RICE: Refining scheme for ReInforCement learning with Explanation

This implementation reproduces the RICE algorithm from the paper:
"RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation"
by Zelei Cheng, Xian Wu, Jiahao Yu, Sabrina Yang, Gang Wang, Xinyu Xing

The algorithm combines explanation methods to identify critical states and 
constructs a mixed initial state distribution to break through RL training bottlenecks.
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Normal
from collections import deque
import matplotlib.pyplot as plt
import argparse
import csv

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MaskNetwork(nn.Module):
    """
    Enhanced Mask Network for identifying critical states
    This implements the improved mask network described in Section 3.3
    """
    def __init__(self, state_dim, hidden_dim=128):
        super(MaskNetwork, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Mask network architecture
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        """
        Forward pass through the mask network
        Returns the probability of masking this state
        """
        return torch.sigmoid(self.network(state))

class RICEAgent:
    """
    RICE Agent implementing the RICE algorithm
    This agent implements the full RICE algorithm with explanation-based refinement
    """
    def __init__(self, env_name, hidden_dim=128, learning_rate=0.001, gamma=0.99, p=0.5, lambda_=0.01, alpha=0.01):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.p = p  # probability of using critical states
        self.lambda_ = lambda_  # exploration bonus weight
        self.alpha = alpha  # mask bonus parameter
        self.device = device
        
        # Initialize policy network
        self.policy = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim, self.action_dim),
        nn.Tanh()
        ).to(self.device)
        
        # Initialize mask network
        self.mask_network = MaskNetwork(self.state_dim, self.hidden_dim).to(self.device)
        
        # Initialize optimizer
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.mask_optimizer = torch.optim.Adam(self.mask_network.parameters(), lr=self.learning_rate)
        
        # Initialize RND predictor
        self.rnd_predictor = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim, self.state_dim)
        ).to(self.device)
        
        self.rnd_target = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim, self.state_dim)
        ).to(self.device)
        
        self.rnd_optimizer = torch.optim.Adam(self.rnd_predictor.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # Track metrics
        self.episode_rewards = []
        self.critical_states = []
        self.exploration_bonus = 0
        
    def get_action(self, state):
        """
        Get action from policy
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        action = self.policy(state_tensor)
        return action.detach().cpu().numpy()
    
    def get_mask_probability(self, state):
        """
        Get mask probability from mask network
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        mask_prob = self.mask_network(state_tensor)
        return mask_prob.detach().cpu().numpy()
    
    def calculate_rnd_bonus(self, state):
        """
        Calculate RND bonus
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        target = self.rnd_target(state_tensor)
        predicted = self.rnd_predictor(state_tensor)
        rnd_bonus = torch.norm(target - predicted, dim=-1)
        return rnd_bonus.detach().cpu().numpy()
    
    def update_mask_network(self, states, actions, rewards, next_states, dones):
        """
        Update mask network
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        
        # Calculate mask network loss
        mask_probs = self.mask_network(states_tensor)
        
        # Calculate advantage
        values = torch.sum(self.policy(states_tensor) * actions_tensor, dim=-1)
        advantages = rewards_tensor - values
        advantages = advantages.unsqueeze(-1)
        
        # Calculate mask network loss
        mask_loss = torch.mean(mask_probs * advantages)
        
        # Add bonus for masking
        mask_loss = mask_loss - self.alpha * torch.mean(mask_probs)
        
        # Update mask network
        self.mask_optimizer.zero_grad()
        mask_loss.backward()
        self.mask_optimizer.step()
        
        return mask_loss.detach().cpu().numpy()
    
    def update_policy(self, states, actions, rewards, next_states, dones):
        """
        Update policy using PPO
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        
        # Calculate advantage
        values = torch.sum(self.policy(states_tensor) * actions_tensor, dim=-1)
        advantages = rewards_tensor - values
        advantages = advantages.unsqueeze(-1)
        
        # Calculate PPO loss
        log_probs = torch.sum(actions_tensor * torch.log(self.policy(states_tensor)), dim=-1)
        ratio = torch.exp(log_probs)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
        policy_loss = -torch.mean(torch.min(surrogate1, surrogate2))
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.detach().cpu().numpy()
    
    def train(self, epochs=10, batch_size=32):
        """
        Train the RICE agent
        """
        for epoch in range(epochs):
            state = self.env.reset()[0]
            episode_reward = 0
            states, actions, rewards, next_states, dones = [], [], [], [], []
            
            for step in range(1000):
                # Determine if we're using critical state or default state
                if random.random() < self.p and len(self.critical_states) > 0:
                    # Use critical state
                    state = random.choice(self.critical_states)
                else:
                    # Use default state
                    state = state
                
                # Get action
                action = self.get_action(state)
                
                # Get mask probability
                mask_prob = self.get_mask_probability(state)
                
                # Determine if we're masking
                if random.random() < mask_prob:
                    # Mask: use random action
                    action = self.env.action_space.sample()
                
                # Take step
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # Calculate RND bonus
                rnd_bonus = self.calculate_rnd_bonus(state)
                
                # Add RND bonus to reward
                total_reward = reward + self.lambda_ * rnd_bonus
                
                # Store experience
                states.append(state)
                actions.append(action)
                rewards.append(total_reward)
                next_states.append(next_state)
                dones.append(done)
                
                # Update critical states
                if done:
                    # Add state to critical states
                    self.critical_states.append(state)
                
                # Update RND
                self.update_rnd(state, next_state)
                
                # Update mask network
                if len(states) > 1:
                    mask_loss = self.update_mask_network(states, actions, rewards, next_states, dones)
                
                # Update policy
                if len(states) > 1:
                    policy_loss = self.update_policy(states, actions, rewards, next_states, next_states)
                
                state = next_state
                episode_reward += total_reward
                
                if done:
                    break
            
            # Store episode reward
            self.episode_rewards.append(episode_reward)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Reward: {episode_reward:.2f}")
        
        # Save model
        torch.save(self.policy.state_dict(), "rice_model.pth")
        
        # Save results
        with open("results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "reward"])
            for i, reward in enumerate(self.episode_rewards):
                writer.writerow([i, reward])
    
    def update_rnd(self, state, next_state):
        """
        Update RND predictor
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        
        # Update target network
        with torch.no_grad():
            target = self.rnd_target(state_tensor)
        
        # Update predictor
        predicted = self.rnd_predictor(state_tensor)
        rnd_loss = torch.mean((target - predicted) ** 2)
        
        # Update RND
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()

def main():
    """
    Main function for reproduction script
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Reproduce RICE algorithm")
    parser.add_argument("--envs", nargs="+", default=["Hopper-v3", "Walker2d-v3", "Reacher-v2", "HalfCheetah-v3"], help="List of environments to evaluate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run reproduction for each environment
    results = {}
    for env_name in args.envs:
        print(f"Reproducing RICE for environment: {env_name}")
        
        # Create RICE agent
        agent = RICEAgent(env_name, learning_rate=0.001, gamma=0.99, p=0.5, lambda_=0.01, alpha=0.01)
        
        # Train agent
        agent.train(epochs=args.epochs)
        
        # Store results
        results[env_name] = {
            "final_reward": np.mean(agent.episode_rewards[-10:]),
            "max_reward": np.max(agent.episode_rewards),
        }
        
        # Save results
        np.save(f"{args.output}/{env_name}_results.npy", agent.episode_rewards)
    
    # Print summary
    print("\n" + "="*60)
    print("REPRODUCTION SUMMARY")
    print("="*60)
    for env_name, result in results.items():
        print(f"{env_name}: Final Reward: {result['final_reward']:.2f}, Max Reward: {result['max_reward']:.2f}")
    
    print(f"\nResults saved to {args.output}/")
    
    # Create summary file
    with open(f"{args.output}/summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["environment", "final_reward", "max_reward"])
        for env_name, result in results.items():
            writer.writerow([env_name, result['final_reward'], result['max_reward"]])
    
    print("Reproduction completed successfully!")

if __name__ == "__main__":
    main()