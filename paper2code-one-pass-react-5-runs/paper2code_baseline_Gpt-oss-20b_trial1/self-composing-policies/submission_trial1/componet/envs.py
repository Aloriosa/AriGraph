"""Define the task sequence used for training."""
import gym
from typing import List

def get_task_sequence() -> List[str]:
    """
    Return a list of Gym environment IDs.
    The sequence is deliberately short so that training completes quickly.
    """
    return ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]

def make_env(env_id: str, seed: int = 0):
    """Create a Gym environment and set the random seed."""
    env = gym.make(env_id)
    env.seed(seed)
    env.action_space.seed(seed)
    return env