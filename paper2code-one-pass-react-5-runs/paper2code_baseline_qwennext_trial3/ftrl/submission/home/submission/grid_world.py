import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class GridWorld(gym.Env):
    """
    A 2D grid world with CLOSE and FAR states.
    The agent starts in CLOSE states and must learn to reach FAR states.
    The goal is to reach a specific target state in the FAR region.
    """
    
    def __init__(self, grid_size=10, close_threshold=3, far_threshold=7, max_steps=100):
        super(GridWorld, self).__init__()
        
        self.grid_size = grid_size
        self.close_threshold = close_threshold
        self.far_threshold = far_threshold
        self.max_steps = max_steps
        
        # Define action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Define observation space: agent position (x, y)
        self.observation_space = spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Start agent at a random position in CLOSE region
        self.agent_pos = np.array([random.randint(0, self.close_threshold), random.randint(0, self.close_threshold)])
        self.steps = 0
        self.far_state_visited = False
        self.far_state_achieved = False
        self.close_state_achieved = False
        
        return self.agent_pos, {}
    
    def step(self, action):
        # Move agent based on action
        if action == 0:  # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # down
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        
        # Calculate reward
        reward = -0.01  # Small penalty per step
        terminated = False
        truncated = False
        
        # Check if agent reached FAR state
        if self.agent_pos[0] >= self.far_threshold and not self.far_state_visited:
            self.far_state_visited = True
            reward += 1.0  # Reward for reaching FAR state
        
        # Check if agent reached target in FAR region
        if self.agent_pos[0] >= self.far_threshold and self.agent_pos[1] >= self.far_threshold and not self.far_state_achieved:
            self.far_state_achieved = True
            reward += 10.0  # Reward for reaching target
            terminated = True
        
        # Check if agent reached target in CLOSE region
        if self.agent_pos[0] <= self.close_threshold and self.agent_pos[1] <= self.close_threshold and not self.close_state_achieved:
            self.close_state_achieved = True
            reward += 5.0  # Reward for reaching CLOSE target
        
        # Check if max steps reached
        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True
        
        return self.agent_pos, reward, terminated, truncated, {}
    
    def get_state_category(self, pos):
        """Return 'CLOSE' or 'FAR' based on position"""
        if pos[0] <= self.close_threshold and pos[1] <= self.close_threshold:
            return 'CLOSE'
        elif pos[0] >= self.far_threshold and pos[1] >= self.far_threshold:
            return 'FAR'
        else:
            return 'TRANSITION'
    
    def get_state_info(self):
        """Return information about current state"""
        return {
            'agent_pos': self.agent_pos,
            'steps': self.steps,
            'far_visited': self.far_state_visited,
            'far_achieved': self.far_state_achieved,
            'close_achieved': self.close_state_achieved,
            'category': self.get_state_category(self.agent_pos)
        }