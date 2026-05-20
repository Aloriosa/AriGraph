"""Environment wrappers with task-specific variations."""

import gymnasium as gym


class CartPoleGravity(gym.Env):
    """
    CartPole environment with a configurable gravity value.
    The underlying CartPole-v1 environment is modified in-place.
    """

    def __init__(self, gravity: float = 9.8):
        super().__init__()
        self.env = gym.make("CartPole-v1")
        # The internal environment uses a Box2D physics engine where gravity
        # can be accessed as a float attribute.
        self.env.env.gravity = gravity

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        # gymnasium returns: (observation, reward, terminated, truncated, info)
        return self.env.step(action)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space