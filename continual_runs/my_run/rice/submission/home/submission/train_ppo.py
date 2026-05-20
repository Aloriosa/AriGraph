import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import argparse
import os
import random
from collections import deque
import pickle

class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PPOPolicy, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Log standard deviation for action distribution (parameterized)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, x):
        mean = self.actor(x)
        std = torch.exp(self.log_std)
        return mean, std
    
    def get_action(self, x, deterministic=False):
        mean, std = self.forward(x)
        
        if deterministic:
            return mean
        else:
            action_dist = torch.distributions.Normal(mean, std)
            action = action_dist.sample()
            return action
    
    def get_log_prob(self, x, actions):
        mean, std = self.forward(x)
        action_dist = torch.distributions.Normal(mean, std)
        log_prob = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        return log_prob
    
    def get_value(self, x):
        return self.critic(x)

def train_ppo(env_name, num_steps=300000, seed=42, model_path="models/pretrained_ppo.pth"):
    """
    Train a PPO policy on the specified environment
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize policy
    policy = PPOPolicy(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    
    # PPO hyperparameters
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    batch_size = 64
    epochs = 10
    
    # Training variables
    state, _ = env.reset(seed=seed)
    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    episode_reward = 0
    episode_length = 0
    
    # Buffer for storing trajectories
    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []
    
    print(f"Training PPO on {env_name} for {num_steps} steps...")
    
    for step in range(num_steps):
        # Select action
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mean, std = policy(state_tensor)
            action_dist = torch.distributions.Normal(mean, std)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1, keepdim=True)
            value = policy.get_value(state_tensor)
        
        # Take action in environment
        next_state, reward, terminated, truncated, _ = env.step(action.squeeze().numpy())
        done = terminated or truncated
        
        # Store transition
        states.append(state)
        actions.append(action.squeeze().numpy())
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.squeeze().numpy())
        values.append(value.squeeze().numpy())
        
        # Update episode statistics
        episode_reward += reward
        episode_length += 1
        
        # If episode is done, reset environment
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
        else:
            state = next_state
        
        total_steps += 1
        
        # Update policy if we have enough data
        if len(states) >= batch_size:
            # Compute advantages using GAE
            states_tensor = torch.FloatTensor(np.array(states))
            actions_tensor = torch.FloatTensor(np.array(actions))
            rewards_tensor = torch.FloatTensor(np.array(rewards))
            dones_tensor = torch.FloatTensor(np.array(dones))
            old_log_probs_tensor = torch.FloatTensor(np.array(log_probs))
            values_tensor = torch.FloatTensor(np.array(values))
            
            # Compute next value for GAE
            with torch.no_grad():
                next_value = policy.get_value(torch.FloatTensor(next_state).unsqueeze(0)).squeeze().numpy()
            
            # Compute advantages
            advantages = []
            gae = 0
            for i in reversed(range(len(rewards))):
                delta = rewards[i] + gamma * (1 - dones[i]) * next_value - values_tensor[i]
                gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
                advantages.insert(0, gae)
                next_value = values_tensor[i]
            
            advantages_tensor = torch.FloatTensor(advantages)
            returns_tensor = advantages_tensor + values_tensor
            
            # Normalize advantages
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
            
            # PPO update
            for _ in range(epochs):
                # Sample mini-batch
                indices = np.random.permutation(len(states))
                for i in range(0, len(states), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    
                    # Get current policy values
                    current_mean, current_std = policy(states_tensor[batch_indices])
                    current_action_dist = torch.distributions.Normal(current_mean, current_std)
                    current_log_probs = current_action_dist.log_prob(actions_tensor[batch_indices]).sum(dim=-1, keepdim=True)
                    current_values = policy.get_value(states_tensor[batch_indices])
                    
                    # Compute ratio
                    ratio = torch.exp(current_log_probs - old_log_probs_tensor[batch_indices])
                    
                    # Compute surrogate loss
                    surr1 = ratio * advantages_tensor[batch_indices]
                    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_tensor[batch_indices]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Compute value loss
                    value_loss = nn.MSELoss()(current_values, returns_tensor[batch_indices].unsqueeze(1))
                    
                    # Compute entropy loss
                    entropy_loss = current_action_dist.entropy().mean()
                    
                    # Total loss
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
                    
                    # Optimize
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()
            
            # Clear buffer
            states = []
            actions = []
            rewards = []
            dones = []
            log_probs = []
            values = []
        
        # Print progress
        if step % 10000 == 0:
            if len(episode_rewards) > 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Step {step}/{num_steps}, Avg Reward: {avg_reward:.2f}")
    
    # Save policy
    torch.save(policy, model_path)
    print(f"Saved trained policy to {model_path}")
    
    # Print final statistics
    if len(episode_rewards) > 0:
        print(f"Final average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    
    return policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--num_steps', type=int, default=300000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_path', type=str, default='models/pretrained_ppo.pth')
    
    args = parser.parse_args()
    
    train_ppo(args.env_name, args.num_steps, args.seed, args.model_path)