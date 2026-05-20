import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import argparse
import os
import random
import pickle
from collections import deque

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

def refine_policy_rice(env_name, pretrained_policy_path, critical_states_path, mixing_ratio=0.5, 
                      num_steps=100000, seed=42, output_path="results/rice_refined_policy.pth", 
                      results_file="results/rice_results.csv"):
    """
    Refine a pre-trained policy using the RICE algorithm.
    
    RICE creates a mixed initial state distribution combining default states and critical states.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Load environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Load pre-trained policy
    pretrained_policy = torch.load(pretrained_policy_path, map_location='cpu')
    policy = PPOPolicy(state_dim, action_dim)
    
    # Initialize policy with pre-trained weights
    policy.load_state_dict(pretrained_policy.state_dict())
    
    # Load critical states
    with open(critical_states_path, 'rb') as f:
        critical_data = pickle.load(f)
    critical_states = critical_data['critical_states']
    
    # Initialize optimizer
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
    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    
    print(f"Refining policy using RICE with mixing ratio {mixing_ratio}...")
    
    # Buffer for storing trajectories
    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []
    
    # Initialize environment
    state, _ = env.reset(seed=seed)
    
    for step in range(num_steps):
        # Select initial state using mixed distribution
        if random.random() < mixing_ratio:
            # Sample from critical states
            critical_idx = random.randint(0, len(critical_states) - 1)
            state = critical_states[critical_idx]
        else:
            # Sample from default distribution
            state, _ = env.reset()
        
        episode_reward = 0
        episode_length = 0
        
        # Run episode
        for _ in range(1000):  # Max episode length
            states.append(state)
            
            # Select action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                mean, std = policy(state_tensor)
                action_dist = torch.distributions.Normal(mean, std)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum().item()
                value = policy.get_value(state_tensor).item()
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action.squeeze().numpy())
            done = terminated or truncated
            
            actions.append(action.squeeze().numpy())
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        total_steps += episode_length
        
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
                next_value = policy.get_value(torch.FloatTensor(state).unsqueeze(0)).item()
            
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
        if step % 1000 == 0:
            if len(episode_rewards) > 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Step {step}/{num_steps}, Avg Reward: {avg_reward:.2f}")
        
        # Save policy periodically
        if step % 10000 == 0:
            torch.save(policy, output_path.replace(".pth", f"_step_{step}.pth"))
    
    # Save final policy
    torch.save(policy, output_path)
    print(f"Saved refined policy to {output_path}")
    
    # Save results
    results = {
        'final_mean_reward': np.mean(episode_rewards),
        'final_std_reward': np.std(episode_rewards),
        'total_steps': total_steps,
        'num_episodes': len(episode_rewards),
        'mixing_ratio': mixing_ratio,
        'critical_states_count': len(critical_states),
        'episode_rewards': episode_rewards
    }
    
    with open(results_file, 'w') as f:
        f.write("mean_reward,std_reward,total_steps,num_episodes,mixing_ratio,critical_states_count\n")
        f.write(f"{results['final_mean_reward']},{results['final_std_reward']},{results['total_steps']},{results['num_episodes']},{results['mixing_ratio']},{results['critical_states_count']}\n")
    
    print(f"Saved results to {results_file}")
    
    return policy, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--pretrained_policy_path', type=str, default='models/pretrained_ppo.pth')
    parser.add_argument('--critical_states_path', type=str, default='data/critical_states.pkl')
    parser.add_argument('--mixing_ratio', type=float, default=0.5)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_path', type=str, default='results/rice_refined_policy.pth')
    parser.add_argument('--results_file', type=str, default='results/rice_results.csv')
    
    args = parser.parse_args()
    
    refine_policy_rice(
        args.env_name,
        args.pretrained_policy_path,
        args.critical_states_path,
        args.mixing_ratio,
        args.num_steps,
        args.seed,
        args.output_path,
        args.results_file
    )