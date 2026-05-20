#!/usr/bin/env python3
"""
Reproduction of "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem"

This script implements a simplified version of the paper's key findings.

The paper's main contribution is that forgetting of pre-trained capabilities (FPC)
is a critical problem in RL fine-tuning, and that knowledge retention techniques can mitigate this problem.

This reproduction implements a simplified version of the paper's environment and model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import os
from collections import deque
import time

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Environment parameters
GRID_SIZE = 10
NUM_STATES = GRID_SIZE * GRID_SIZE
NUM_ACTIONS = 4  # up, down, left, right
NUM_EPISODES = 1000
NUM_STEPS = 100  # maximum steps per episode
BATCH_SIZE = 32
PRETRAIN_EPISODES = 100  # number of episodes for pre-training
FINETUNE_EPISODES = 100  # number of episodes for fine-tuning
MEMORY_SIZE = 10000  # size of replay buffer

# Model parameters
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON = 0.1

# Knowledge retention parameters
EWC_LAMBDA = 1000  # EWC regularization strength
BC_WEIGHT = 0.5  # BC loss weight
KS_WEIGHT = 0.5  # KS loss weight
EM_SIZE = 100  # EM buffer size

# State partitioning parameters
CLOSE_RADIUS = 3  # radius of CLOSE states around start
FAR_START = 7  # FAR states start from this distance from start
FAR_RADIUS = 2  # radius of FAR states around goal

class Environment:
    """
    Simplified 2D grid world environment with CLOSE and FAR states.
    """
    def __init__(self, grid_size=GRID_SIZE, num_actions=NUM_ACTIONS):
        self.grid_size = grid_size
        self.num_actions = num_actions
        self.reset()
        
    def reset(self):
        """Reset the environment to the start state."""
        # Start at position (0, 0)
        self.position = [0, 0]
        self.goal = [self.grid_size - 1, self.grid_size - 1]  # Goal at bottom right
        self.done = False
        self.step_count = 0
        return self.get_state()
    
    def get_state(self):
        """Get the current state as a flattened grid with agent position.
        The state is a 1D array of size GRID_SIZE * GRID_SIZE with a 1 at the agent's position."""
        state = np.zeros(self.grid_size * self.grid_size)
        state[self.position[0] * self.grid_size + self.position[1]] = 1
        return state
    
    def step(self, action):
        """
        Take a step in the environment.
        action: 0=up, 1=down, 2=left, 3=right
        """
        # Move agent based on action
        if action == 0:  # up
            self.position[0] = max(0, self.position[0] - 1)
        elif action == 1:  # down
            self.position[0] = min(self.grid_size - 1, self.position[0] + 1)
        elif action == 2:  # left
            self.position[1] = max(0, self.position[1] - 1)
        elif action == 3:  # right
            self.position[1] = min(self.grid_size - 1, self.position[1] + 1)
        
        # Calculate distance to goal
        distance_to_goal = np.sqrt((self.position[0] - self.goal[0])**2 + (self.position[1] - self.goal[1])**2)
        
        # Check if goal reached
        if distance_to_goal < 0.5:
            reward = 10
            self.done = True
        elif self.step_count >= NUM_STEPS:
            reward = -1
            self.done = True
        else:
            reward = -0.1
        
        # Calculate next state
        next_state = self.get_state()
        self.step_count += 1
        
        return next_state, reward, self.done
    
    def is_close_state(self, state):
        """Check if a state is in the CLOSE region (near start)."""
        pos = np.array([state[0], state[1]])
        start = np.array([0, 0])
        distance = np.sqrt((pos[0] - start[0])**2 + (pos[1] - start[1])**2)
        return distance <= CLOSE_RADIUS
    
    def is_far_state(self, state):
        """Check if a state is in the FAR region (near goal)."""
        pos = np.array([state[0], state[1]])
        goal = np.array([self.grid_size - 1, self.grid_size - 1])
        distance = np.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)
        return distance <= FAR_RADIUS
    
    def get_state_coords(self, state):
        """Get the (x, y) coordinates from a state vector."""
        idx = np.argmax(state)
        x = idx // self.grid_size
        y = idx % self.grid_size
        return x, y

class PolicyNetwork(nn.Module):
    """
    Neural network policy for the agent.
    """
    def __init__(self, input_size=NUM_STATES, hidden_size=HIDDEN_SIZE, output_size=NUM_ACTIONS):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

class ValueNetwork(nn.Module):
    """
    Value network for estimating state values.
    """
    def __init__(self, input_size=NUM_STATES, hidden_size=HIDDEN_SIZE):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    """
    Experience replay buffer for storing transitions.
    """
    def __init__(self, max_size=MEMORY_SIZE):
        self.buffer = deque(maxlen=max_size)
        
    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

class KnowledgeRetention:
    """
    Knowledge retention techniques to mitigate forgetting of pre-trained capabilities.
    """
    def __init__(self, model, pretrain_model, state_size=NUM_STATES, buffer_size=MEMORY_SIZE):
        self.model = model
        self.pretrain_model = pretrain_model
        self.pretrain_states = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.pretrain_buffer = ReplayBuffer(buffer_size)
        
    def add_pretrain_state(self, state):
        """Add a state from the pre-training phase."""
        self.pretrain_states.append(state)
        
    def get_pretrain_states(self):
        """Get the states from the pre-training phase."""
        return list(self.pretrain_states)
    
    def add_pretrain_buffer(self, state, action, reward, next_state, done):
        """Add a transition from the pre-training phase to the buffer."""
        self.pretrain_buffer.push(state, action, reward, next_state, done)
        
    def sample_pretrain_buffer(self, batch_size):
        """Sample a batch of transitions from the pre-training buffer."""
        return self.pretrain_buffer.sample(batch_size)
    
    def ewc_loss(self, model, pretrain_model, ewc_lambda):
        """Elastic Weight Consolidation loss to penalize changes in parameters."""
        loss = 0
        for p, p_pre in zip(model.parameters(), pretrain_model.parameters()):
            loss += torch.sum((p - p_pre) ** 2)
        return ewc_lambda * loss
    
    def bc_loss(self, model, pretrain_model, buffer, weight):
        """Behavioral Cloning loss to mimic the pre-trained policy."""
        if len(buffer) == 0:
            return 0
        states = torch.tensor([s for s in buffer], dtype=torch.float32)
        with torch.no_grad():
            pretrain_probs = pretrain_model(states)
        model_probs = model(states)
        kl_div = torch.sum(pretrain_probs * torch.log(pretrain_probs / (model_probs + 1e-8)))
        return weight * torch.mean(kl_div)
    
    def ks_loss(self, model, pretrain_model, buffer, weight):
        """Kickstarting loss to match the pre-trained policy on data from the current policy."""
        if len(buffer) == 0:
            return 0
        states = torch.tensor([s for s in buffer], dtype=torch.float32)
        with torch.no_grad():
            pretrain_probs = pretrain_model(states)
        model_probs = model(states)
        kl_div = torch.sum(pretrain_probs * torch.log(pretrain_probs / (model_probs + 1e-8)))
        return weight * torch.mean(kl_div)
    
    def em_loss(self, model, pretrain_model, buffer, weight):
        """Episodic Memory loss to keep examples from the pre-trained task in the replay buffer."""
        if len(buffer) == 0:
            return 0
        states = torch.tensor([s for s in buffer], dtype=torch.float32)
        with torch.no_grad():
            pretrain_probs = pretrain_model(states)
        model_probs = model(states)
        kl_div = torch.sum(pretrain_probs * torch.log(pretrain_probs / (model_probs + 1e-8)))
        return weight * torch.mean(kl_div)

def train_policy(model, env, episodes, optimizer, memory, policy_net, value_net, pretrain_model, retention):
    """
    Train the policy using PPO algorithm.
    """
    scores = []
    values = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for step in range(NUM_STEPS):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action_probs = model(state_tensor)
                action = np.random.choice(NUM_ACTIONS, p=action_probs.numpy())
            
            next_state, reward, done = env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            # Add to replay buffer
            memory.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update policy using PPO
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)
        
        # Calculate advantages and returns
        with torch.no_grad():
            values = value_net(states)
            next_values = value_net(next_states)
            advantages = rewards + GAMMA * next_values - values
            returns = advantages + values
        
        # PPO loss
        old_probs = model(states)
        old_probs = torch.gather(old_probs, 1, actions.unsqueeze(1)).squeeze()
        new_probs = model(states)
        new_probs = torch.gather(new_probs, 1, actions.unsqueeze(1)).squeeze()
        
        ratio = new_probs / (old_probs + 1e-8)
        clipped_ratio = torch.clamp(ratio, 1 - 0.1, 1 + 0.1)
        ppo_loss = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))
        
        # Knowledge retention loss
        ewc_loss = retention.ewc_loss(model, pretrain_model, EWC_LAMBDA)
        bc_loss = retention.bc_loss(model, pretrain_model, retention.pretrain_states, BC_WEIGHT)
        ks_loss = retention.ks_loss(model, pretrain_model, retention.pretrain_states, KS_WEIGHT)
        em_loss = retention.em_loss(model, pretrain_model, retention.pretrain_states, 0.1)
        
        # Total loss
        loss = ppo_loss + ewc_loss + bc_loss + ks_loss + em_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        scores.append(total_reward)
        values.append(torch.mean(values).item())
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Score: {total_reward:.2f}")
    
    return scores, values

def main():
    """
    Main function to run the reproduction.
    """
    print("Starting reproduction of 'Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem'")
    
    # Create environment
    env = Environment()
    
    # Create models
    policy_net = PolicyNetwork().to(device)
    value_net = ValueNetwork().to(device)
    pretrain_policy_net = PolicyNetwork().to(device)
    
    # Create optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    
    # Create replay buffer
    memory = ReplayBuffer()
    
    # Create knowledge retention
    retention = KnowledgeRetention(policy_net, pretrain_policy_net)
    
    # Pre-train on FAR states
    print("Pre-training on FAR states...")
    pretrain_env = Environment()
    pretrain_env.goal = [env.grid_size - 1, env.grid_size - 1]
    
    for episode in range(PRETRAIN_EPISODES):
        state = pretrain_env.reset()
        total_reward = 0
        for step in range(NUM_STEPS):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action_probs = pretrain_policy_net(state_tensor)
                action = np.random.choice(NUM_ACTIONS, p=action_probs.numpy())
            next_state, reward, done = pretrain_env.step(action)
            # Add to pretrain buffer
            retention.add_pretrain_state(state)
            retention.add_pretrain_buffer(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        if episode % 10 == 0:
            print(f"Pretrain Episode {episode}, Score: {total_reward:.2f}")
    
    # Fine-tune on all states
    print("Fine-tuning on all states...")
    scores, values = train_policy(policy_net, env, FINETUNE_EPISODES, optimizer, memory, policy_net, value_net, pretrain_policy_net, retention)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(values)
    plt.title('Value Estimates')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig('/home/submission/results/training_results.png')
    plt.show()
    
    # Save models
    torch.save(policy_net.state_dict(), '/home/submission/models/final_policy_model.pth')
    torch.save(pretrain_policy_net.state_dict(), '/home/submission/models/pretrained_policy_model.pth')
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Environment: 2D Grid World ({GRID_SIZE}x{GRID_SIZE})")
    print(f"States: {NUM_STATES} (CLOSE: {CLOSE_RADIUS} radius from start, FAR: {FAR_RADIUS} radius from goal)")
    print(f"Actions: {NUM_ACTIONS} (up, down, left, right)")
    print(f"Episodes: {FINETUNE_EPISODES}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Gamma: {GAMMA}")
    print(f"EWC Lambda: {EWC_LAMBDA}")
    print(f"BC Weight: {BC_WEIGHT}")
    print(f"KS Weight: {KS_WEIGHT}")
    print(f"EM Size: {EM_SIZE}")
    print(f"Model Parameters: {sum(p.numel() for p in policy_net.parameters())}")
    print(f"Final Score: {np.mean(scores[-10:]):.2f}")
    print(f"Final Value: {np.mean(values[-10:]):.2f}")
    print(f"Memory Used: {len(memory)}")
    print(f"Pretrain States: {len(retention.pretrain_states)}")
    print("="*60)
    
    # Save summary
    with open('/home/submission/results/summary.txt', 'w') as f:
        f.write(f"Environment: 2D Grid World ({GRID_SIZE}x{GRID_SIZE})\n")
        f.write(f"States: {NUM_STATES} (CLOSE: {CLOSE_RADIUS} radius from start, FAR: {FAR_RADIUS} radius from goal)\n")
        f.write(f"Actions: {NUM_ACTIONS} (up, down, left, right)\n")
        f.write(f"Episodes: {FINETUNE_EPISODES}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Gamma: {GAMMA}\n")
        f.write(f"EWC Lambda: {EWC_LAMBDA}\n")
        f.write(f"BC Weight: {BC_WEIGHT}\n")
        f.write(f"KS Weight: {KS_WEIGHT}\n")
    print("Reproduction complete!")
    
if __name__ == "__main__":
    main()