"""
Trainer for CompoNet
"""
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import List, Dict, Optional
from tqdm import tqdm
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompoNetTrainer:
    """
    Trainer for CompoNet
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: CompoNet model
            optimizer: Optimizer
            device: Device to use
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Track metrics
        self.metrics = {
            'loss': [],
            'success_rate': [],
            'forward_transfer': [],
            'parameters': []
        }
        
    def train(self, 
              env: gym.Env,
              num_tasks: int,
              timesteps_per_task: int,
              task_sequence: List[str]) -> Dict[str, List]:
        """
        Train CompoNet on a sequence of tasks.
        
        Args:
            env: Environment
            num_tasks: Number of tasks
            timesteps_per_task: Timesteps per task
            task_sequence: Sequence of task names
        """
        logger.info(f"Training on {num_tasks} tasks with {timesteps_per_task} timesteps each")
        
        # Track metrics
        success_rates = []
        forward_transfers = []
        parameters = []
        
        # Training loop
        for task_id in range(num_tasks):
            logger.info(f"Training on task {task_id}: {task_sequence[task_id]}")
            
            # Add module for this task
            self.model.add_module()
            
            # Reset environment
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            
            # Train for timesteps
            task_losses = []
            task_success_rate = 0.0
            timesteps = 0
            
            # Training loop for this task
            while timesteps < timesteps_per_task:
                # Get action
                action, _ = self.model(state, task_id)
            
                # Take step
                next_state, reward, done, truncated, _ = env.step(action.detach().cpu().numpy())
                next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                
                # Calculate loss
                loss = self.loss_fn(action, torch.zeros_like(action))
                task_losses.append(loss.item())
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update state
                state = next_state
                timesteps += 1
                
                # Check success
                if reward > 0.5:
                    task_success_rate += 1
            
            # Calculate metrics
            avg_loss = np.mean(task_losses)
            success_rate = task_success_rate / timesteps
            parameters.append(sum(p.numel() for p in self.model.parameters()))
            
            # Update metrics
            self.metrics['loss'].append(avg_loss)
            self.metrics['success_rate'].append(success_rate)
            self.metrics['parameters'].append(parameters[-1])
            
            logger.info(f"Task {task_id}: Loss={avg_loss:.4f}, Success Rate={success_rate:.4f}, Parameters={parameters[-1]}")
        
        return self.metrics