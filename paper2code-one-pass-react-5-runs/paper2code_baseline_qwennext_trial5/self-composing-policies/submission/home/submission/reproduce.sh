#!/bin/bash
set -e

# Update system and install dependencies
echo "Updating system and installing dependencies..."
apt-get update
apt-get install -y python3 python3-pip git

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install torch torchvision torchaudio numpy matplotlib gymnasium metaworld

# Create project structure
echo "Creating project structure..."
mkdir -p /home/submission/src
mkdir -p /home/submission/data
mkdir -p /home/submission/results

# Copy the CompoNet implementation
echo "Copying CompoNet implementation..."
cat > /home/submission/src/componet.py << 'EOF'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import metaworld
import random
from collections import deque
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class SelfComposingPolicyModule(nn.Module):
    """
    Self-Composing Policy Module as described in the CompoNet architecture.
    This module implements the core functionality of CompoNet with output attention head,
    input attention head, and internal policy.
    """
    
    def __init__(self, state_dim, action_dim, model_dim=256, num_heads=4):
        super(SelfComposingPolicyModule, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        
        # Output Attention Head
        self.output_query_proj = nn.Linear(state_dim, model_dim)
        self.output_key_proj = nn.Linear(action_dim, model_dim)
        self.output_value_proj = nn.Linear(action_dim, model_dim)
        
        # Input Attention Head
        self.input_query_proj = nn.Linear(state_dim, model_dim)
        self.input_key_proj = nn.Linear(model_dim, model_dim)
        self.input_value_proj = nn.Linear(model_dim, model_dim)
        
        # Internal Policy
        self.internal_policy = nn.Sequential(
            nn.Linear(model_dim + state_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, action_dim)
        )
        
        # Positional encoding for attention
        self.pos_encoding = nn.Parameter(torch.randn(model_dim, 1))
        
        # Layer normalization
        self.norm = nn.LayerNorm(model_dim)
        
    def forward(self, state, prev_outputs):
        """
        Forward pass of the self-composing policy module.
        
        Args:
            state: Current state representation (batch_size, state_dim)
            prev_outputs: Outputs from previous modules (batch_size, num_prev_modules, action_dim)
        
        Returns:
            output: Final output of the module (batch_size, action_dim)
        """
        batch_size = state.shape[0]
        num_prev_modules = prev_outputs.shape[1] if prev_outputs.dim() > 1 else 0
        
        # Output Attention Head
        # Query: state representation
        q_out = self.output_query_proj(state)  # (batch_size, model_dim)
        
        # Keys and Values: previous outputs
        if num_prev_modules > 0:
            # Add positional encoding to previous outputs
        else:
            # No previous outputs
            prev_outputs = torch.zeros(batch_size, 1, self.action_dim, device=state.device)
        
        # Compute keys and values
        k_out = self.output_key_proj(prev_outputs)  # (batch_size, num_prev_modules, model_dim)
        v_out = self.output_value_proj(prev_outputs)  # (batch_size, num_prev_modules, action_dim)
        
        # Compute attention weights
        attention_weights = torch.bmm(k_out, q_out.unsqueeze(-1)).squeeze(-1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        attention_output = torch.bmm(attention_weights.unsqueeze(1), v_out).squeeze(1)
        
        # Input Attention Head
        # Query: state representation
        q_in = self.input_query_proj(state)
        
        # Keys and Values: concatenated output from output attention head and previous outputs
        # Note: In practice, we would use the attention output from the output attention head
        k_in = self.input_key_proj(attention_output)
        v_in = self.input_value_proj(attention_output)
        
        # Compute attention weights
        attention_weights_in = torch.bmm(k_in.unsqueeze(1), q_in.unsqueeze(-1)).squeeze(-1)
        attention_weights_in = torch.softmax(attention_in, dim=1)
        
        # Apply attention
        attention_output_in = torch.bmm(attention_weights_in.unsqueeze(1), v_in.unsqueeze(1)).squeeze(1)
        
        # Internal Policy
        # Combine attention output and state
        combined = torch.cat([attention_output, state], dim=1)
        internal_output = self.internal_policy(combined)
        
        # Final output: internal policy output + attention output
        output = attention_output + internal_output
        
        return output

class CompoNet(nn.Module):
    """
    CompoNet: Composable Network for Continual Reinforcement Learning
    This is the main CompoNet architecture that consists of multiple self-composing
    policy modules arranged in a cascading structure.
    """
    
    def __init__(self, state_dim, action_dim, model_dim=256, max_modules=100):
        super(CompoNet, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_dim = model_dim
        self.max_modules = max_modules
        
        # Initialize the first module
        self.modules = nn.ModuleList()
        self.modules.append(SelfComposingPolicyModule(state_dim, action_dim, model_dim)
        
        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(action_dim)
        
    def add_module(self):
        """Add a new self-composing policy module to the network."""
        if len(self.modules) < self.max_modules:
            new_module = SelfComposingPolicyModule(self.state_dim, self.action_dim, self.model_dim)
            self.modules.append(new_module)
        else:
            print("Maximum number of modules reached!")
            
    def forward(self, state, task_id):
        """
        Forward pass of the CompoNet.
        
        Args:
            state: Current state representation
            task_id: Current task identifier
        """
        # Encode the state
        state_encoded = self.encoder(state)
        
        # Get the current module
        current_module = self.modules[task_id]
        
        # Get outputs from previous modules
        prev_outputs = torch.zeros(1, 1, self.action_dim, device=state.device)
        
        # Collect outputs from previous modules
        if task_id > 0:
            prev_outputs = torch.zeros(1, task_id, self.action_dim, device=state.device)
            
            for i in range(task_id):
                prev_module = self.modules[i]
            prev_outputs[0, i] = prev_module(state_encoded)
            
        # Forward pass through the current module
        output = current_module(state_encoded, prev_outputs)
        
        # Normalize output
        output = self.output_norm(output)
        
        return output

class SACAgent:
    """
    Soft Actor-Critic Agent for CompoNet
    """
    
    def __init__(self, state_dim, action_dim, model_dim=256, max_modules=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_dim = model_dim
        self.max_modules = max_modules
        
        # CompoNet model
        self.model = CompoNet(state_dim, action_dim, model_dim, max_modules)
        
        # Actor and Critic networks
        self.actor = self.model
        self.critic = self.model
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.target_update_freq = 1
        self.batch_size = 128
        self.buffer_size = 100000
        self.min_steps = 5000
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
        # Target networks
        self.target_actor = CompoNet(state_dim, action_dim, model_dim, max_modules)
        self.target_critic = CompoNet(state_dim, action_dim, model_dim, max_modules)
        self.update_target_networks()
        
        # Action noise
        self.action_noise = 0.1
        self.action_noise_decay = 0.9999
        self.action_noise_min = 0.01
        
        # Training step counter
        self.step_count = 0
        
    def update_target_networks(self):
        """Update target networks using Polyak averaging."""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
    def select_action(self, state, task_id):
        """Select action using the actor network with exploration noise."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor, task_id)
            action = action.squeeze().cpu().numpy()
            action = np.clip(action, -1, 1)
            return action
        
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def train(self):
        """Train the agent using SAC algorithm."""
        if len(self.replay_buffer) < self.min_steps:
            return
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
        next_q_values = self.target_critic(next_states)
        target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values
        
        current_q_values = self.critic(states)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions)
        actor_loss = actor_loss.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        if self.step_count % self.target_update_freq == 0:
            self.update_target_networks()
        
        self.step_count += 1
        
        return actor_loss.item(), critic_loss.item()

def main():
    """Main function to train CompoNet on Meta-World tasks."""
    print("Initializing Meta-World environment...")
    
    # Initialize Meta-World environment
    ml = metaworld.ML1('meta-world')
    tasks = ml.train_tasks
    print(f"Number of tasks: {len(tasks)}")
    
    # Initialize agent
    state_dim = 39  # Meta-World state dimension
    action_dim = 4  # Meta-World action dimension
    agent = SACAgent(state_dim, action_dim)
    
    # Training parameters
    total_timesteps = 1000000  # 1M timesteps per task
    num_tasks = 20  # Number of tasks in sequence
    timesteps_per_task = total_timesteps // num_tasks
    
    # Training loop
    print(f"Training on {num_tasks} tasks, {timesteps_per_task} timesteps per task")
    
    # Track performance
    success_rates = []
    episodic_returns = []
    
    for task_id in range(num_tasks):
        print(f"Training on task {task_id}...")
        
        # Add new module for the task
        agent.model.add_module()
        
        # Reset environment
        env = metaworld.MT1(tasks[task_id])
        state = env.reset()
        
        task_returns = []
        task_success = 0
        
        for step in range(timesteps_per_task):
            # Select action
            action = agent.select_action(state, task_id)
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.replay_buffer) >= agent.min_steps:
                agent.train()
            
            state = next_state
            task_returns.append(reward)
            
            # Check success
            if info.get('success', 0) == 1:
                task_success += 1
            
            # Print progress
            if step % 1000 == 0:
                print(f"Task {task_id}, Step {step}/{timesteps_per_task}")
        
        # Record performance
        success_rate = task_success / timesteps_per_task
        avg_return = np.mean(task_returns)
        
        success_rates.append(success_rate)
        episodic_returns.append(avg_return)
        
        print(f"Task {task_id} completed. Success rate: {success_rate:.4f}, Avg return: {avg_return:.4f}")
    
    # Save results
    print("Saving results...")
    np.save("/home/submission/results/success_rates.npy", np.array(success_rates))
    np.save("/home/submission/results/episodic_returns.npy", np.array(episodic_returns))
    
    print("Training completed!")
    
    # Print final results
    print(f"Final success rates: {success_rates}")
    print(f"Final episodic returns: {episodic_returns}")
    print(f"Average success rate: {np.mean(success_rates):.4f}")
    print(f"Average episodic return: {np.mean(episodic_returns):.4f}")

if __name__ == "__main__":
    main()
EOF