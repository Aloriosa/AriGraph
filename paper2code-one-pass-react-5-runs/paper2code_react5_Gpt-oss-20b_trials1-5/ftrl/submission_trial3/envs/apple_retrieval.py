import gymnasium as gym
import numpy as np

class AppleRetrieval(gym.Env):
    """
    1D grid world with two phases.
    Phase 0: 0 <= x < M  (retrieve apple)
    Phase 1: M <= x <= 2*M  (return home)
    Rewards: +1 for moving in the correct direction, -1 otherwise.
    Episode ends after max_steps or when the apple is retrieved and the agent returns home.
    Observation: [phase, position_norm]
    Action: 0 = left, 1 = right
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, M=5, max_steps=100, render_mode=None):
        super().__init__()
        self.M = M
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Observation: [phase, position_norm]
        low = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Action: 0 = left, 1 = right
        self.action_space = gym.spaces.Discrete(2)

        self.reset()

    def reset(self, seed=None, options=None):
        self.x = 0  # start at home
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        phase = 1.0 if self.x >= self.M else 0.0
        pos_norm = self.x / (2 * self.M)
        return np.array([phase, pos_norm], dtype=np.float32)

    def step(self, action):
        # action 0: left, 1: right
        prev_x = self.x
        if action == 0:
            self.x = max(self.x - 1, 0)
        else:
            self.x = min(self.x + 1, 2 * self.M)

        reward = 0.0
        # Determine if action is correct
        if self.x > prev_x:  # moved right
            if self.x <= self.M:  # still collecting apple
                reward = 1.0
            else:  # moving back home
                reward = -1.0
        else:  # moved left
            if self.x >= self.M:  # still collecting apple
                reward = -1.0
            else:  # moving back home
                reward = 1.0

        self.step_count += 1
        done = self.step_count >= self.max_steps
        # If agent has collected apple and returned home
        if self.x == 0 and self.x >= self.M:
            done = True

        return self._get_obs(), reward, done, False, {}

    def render(self, mode="human"):
        # Simple textual rendering
        grid = ['.'] * (2 * self.M + 1)
        grid[self.x] = 'A'
        print("".join(grid))