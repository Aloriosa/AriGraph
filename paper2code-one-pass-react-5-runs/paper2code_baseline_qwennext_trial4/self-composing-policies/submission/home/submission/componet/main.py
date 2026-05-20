"""
Main script for CompoNet reproduction
"""
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import argparse
import os
import logging
from componet.core import CompoNet
from componet.trainer import CompoNetTrainer
from componet.utils import get_env, get_algorithm, get_task_sequence

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='CompoNet Reproduction')
    parser.add_argument('--env_name', type=str, default='MetaWorld', 
                      choices=['MetaWorld', 'SpaceInvaders', 'Freeway'])
    parser.add_argument('--algorithm', type=str, default='SAC', 
                      choices=['SAC', 'PPO'])
    parser.add_argument('--num_tasks', type=int, default=20)
    parser.add_argument('--timesteps_per_task', type=int, default=100000)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}')
    
    # Get environment
    env = get_env(args.env_name)
    
    # Get algorithm
    algorithm = get_algorithm(args.algorithm)
    
    # Get task sequence
    task_sequence = get_task_sequence(args.env_name, args.num_tasks)
    
    # Create model
    model = CompoNet(
        state_dim=env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 1,
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # Create trainer
    trainer = CompoNetTrainer(model, optimizer, device)
    
    # Train
    metrics = trainer.train(
        env=env,
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({
        'metrics': metrics,
    }, f'{args.output_dir}/results.pth')
    
    # Print results
    logger.info("Training completed!")
    logger.info(f"Final parameters: {metrics['parameters'][-1]}")
    logger.info(f"Final success rate: {metrics['success_rate'][-1]}")
    
    return metrics

if __name__ == '__main__':
    main()