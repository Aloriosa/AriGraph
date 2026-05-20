import torch
import torch.nn as nn
import numpy as np
import random

class DQNAgent:
    """
    A simple DQN agent for the grid world environment.
    """
    
    def __init__(self, state_size=2, action_size=4, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Create Q-network
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        # Create target network
        self.target_network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Create memory
        self.memory = []
        self.memory_size = memory_size
        
        # Copy weights from q_network to target_network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Initialize
        self.reset()
    
    def reset(self):
        self.epsilon = 1.0
        self.memory = []
        self.q_network = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        self.target_network = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.memory = []
    
    def get_action(self, state):
        """Get action from epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def replay(self, batch_size=32):
        """Train on experience from memory"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute current Q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1))
        
        # Compute loss
        loss = self.criterion(q_values.squeeze(), target_q_values)
        
        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        return loss.item()
    
    def update_target_network(self, tau=0.01):
        """Update target network using soft update"""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save_model(self, path):
        """Save model weights"""
        torch.save(self.q_network.state_dict(), path)
    
    def load_model(self, path):
        """Load model weights"""
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())