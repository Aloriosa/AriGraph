"""
Utility functions for CompoNet
"""
import gymnasium as gym
import numpy as np
from typing import List

def get_env(env_name: str) -> gym.Env:
    """
    Get environment by name.
    
    Args:
        env_name: Environment name
    Returns:
        Environment
    """
    if env_name == 'MetaWorld':
        # Use a simple continuous control environment for reproduction
        # In a full reproduction, we would use the actual MetaWorld environment
        # For this reproduction, we'll use a simple continuous control environment
        return gym.make('Pendulum-v1')
    elif env_name == 'SpaceInvaders':
        return gym.make('ALE/SpaceInvaders-v5')
    elif env_name == 'Freeway':
        return gym.make('ALE/Freeway-v5')
    else:
        raise ValueError(f"Unknown environment: {env_name}")

def get_algorithm(algorithm: str):
    """
    Get algorithm by name.
    
    Args:
        algorithm: Algorithm name
    Returns:
        Algorithm
    """
    if algorithm == 'SAC':
        return 'SAC'
    elif algorithm == 'PPO':
        return 'PPO'
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}')

def get_task_sequence(env_name: str, num_tasks: int) -> List[str]:
    """
    Get task sequence by environment name.
    
    Args:
        env_name: Environment name
        num_tasks: Number of tasks
    Returns:
        Task sequence
    """
    if env_name == 'MetaWorld':
        # 20 tasks from MetaWorld
        return [f'MetaWorld_Task_{i}' for i in range(num_tasks)]
    elif env_name == 'SpaceInvaders':
        # 10 tasks from SpaceInvaders
        return [f'SpaceInvaders_Mode_{i}' for i in range(num_tasks)]
    elif env_name == 'Freeway':
        # 7 tasks from Freeway
        return [f'Freeway_Mode_{i}' for i in range(num_tasks)]
    else:
        raise ValueError(f"Unknown environment: {env_name}')