import numpy as np
import pickle
import random
import torch
from typing import List, Dict, Tuple

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

def generate_synthetic_robotic_trajectory(
    state_dim: int = 10, 
    action_dim: int = 4, 
    trajectory_length: int = 50,
    reward_function_type: str = "goal_reaching"
) -> Dict:
    """
    Generate a synthetic robotic trajectory with associated reward function.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        trajectory_length: Length of trajectory
        reward_function_type: Type of reward function
        
    Returns:
        Dictionary containing trajectory data and reward function
    """
    # Generate states (simulated robot states)
    states = np.random.randn(trajectory_length, state_dim).astype(np.float32)
    
    # Generate actions (simulated robot actions)
    actions = np.random.randn(trajectory_length, action_dim).astype(np.float32)
    
    # Generate reward function based on type
    if reward_function_type == "goal_reaching":
        # Reward based on distance to goal
        goal = np.random.randn(state_dim).astype(np.float32)
        rewards = -np.linalg.norm(states - goal, axis=1)
    elif reward_function_type == "velocity":
        # Reward based on velocity magnitude
        velocities = np.random.randn(trajectory_length, state_dim).astype(np.float32)
        rewards = np.linalg.norm(velocities, axis=1)
    elif reward_function_type == "energy_efficient":
        # Reward based on negative energy consumption
        energy = np.sum(np.abs(actions), axis=1)
        rewards = -energy
    elif reward_function_type == "balance":
        # Reward based on state stability
        rewards = -np.sum(np.square(states), axis=1)
    elif reward_function_type == "exploration":
        # Reward based on state novelty (distance from mean)
        mean_state = np.mean(states, axis=0)
        rewards = -np.linalg.norm(states - mean_state, axis=1)
    else:
        # Random reward function
        weights = np.random.randn(state_dim).astype(np.float32)
        rewards = np.dot(states, weights)
    
    # Add some noise to rewards
    rewards += np.random.normal(0, 0.1, size=rewards.shape)
    
    # Create reward function representation as a dictionary
    reward_function = {
        "type": reward_function_type,
        "parameters": {
            "goal": goal.tolist() if reward_function_type == "goal_reaching" else None,
            "weights": weights.tolist() if reward_function_type == "random" else None
        }
    }
    
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "reward_function": reward_function,
        "terminal": False
    }

def generate_offline_dataset(
    num_trajectories: int = 50,
    state_dim: int = 10,
    action_dim: int = 4,
    min_trajectory_length: int = 30,
    max_trajectory_length: int = 70
) -> List[Dict]:
    """
    Generate a comprehensive offline dataset of robotic trajectories.
    
    Args:
        num_trajectories: Number of trajectories to generate
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        min_trajectory_length: Minimum trajectory length
        max_trajectory_length: Maximum trajectory length
        
    Returns:
        List of trajectory dictionaries
    """
    # Define diverse reward function types
    reward_types = ["goal_reaching", "velocity", "energy_efficient", "balance", "exploration", "random"]
    
    dataset = []
    
    for i in range(num_trajectories):
        # Randomly select reward function type
        reward_type = random.choice(reward_types)
        
        # Random trajectory length
        trajectory_length = random.randint(min_trajectory_length, max_trajectory_length)
        
        # Generate trajectory
        trajectory = generate_synthetic_robotic_trajectory(
            state_dim=state_dim,
            action_dim=action_dim,
            trajectory_length=trajectory_length,
            reward_function_type=reward_type
        )
        
        dataset.append(trajectory)
    
    return dataset

def save_dataset(dataset: List[Dict], filepath: str = "data/offline_trajectories.pkl"):
    """
    Save the generated dataset to a pickle file.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset with {len(dataset)} trajectories saved to {filepath}")

if __name__ == "__main__":
    # Generate and save dataset
    dataset = generate_offline_dataset(
        num_trajectories=50,
        state_dim=10,
        action_dim=4,
        min_trajectory_length=30,
        max_trajectory_length=70
    )
    
    save_dataset(dataset, "data/offline_trajectories.pkl")