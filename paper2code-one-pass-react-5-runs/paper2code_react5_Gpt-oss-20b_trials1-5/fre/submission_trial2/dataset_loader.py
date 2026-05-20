import gymnasium as gym
import d4rl
import numpy as np

def load_dataset(env_name: str) -> dict:
    """
    Loads a D4RL offline dataset. Returns a dictionary with keys:
        observations, actions, rewards, next_observations, terminals.
    """
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    # The dataset contains 'rewards' which are the expert rewards;
    # we ignore them because FRE uses arbitrary reward functions.
    return {
        'observations': np.array(dataset['observations']),
        'actions': np.array(dataset['actions']),
        'next_observations': np.array(dataset['next_observations']),
        'terminals': np.array(dataset['terminals']),
        'dones': np.array(dataset['terminals'])
    }