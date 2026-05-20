#!/usr/bin/env python3
import os
import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from rices.mask_network import MaskNet

SEED = 42
set_random_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ENV_ID = "CartPole-v1"
MASK_MODEL_DIR = "../models/mask_net.zip"
ALPHA = 0.01          # intrinsic bonus for masking
TIMESTEPS = 200_000   # how many steps to train the mask network

# Load pre‑trained policy
PRETRAINED_MODEL = "../models/pretrained_ppo.zip"

class MaskEnv(gym.Env):
    """
    Wrapper environment that decides whether to mask the action of the
    pre‑trained policy. The mask network outputs a discrete action
    (0: keep original action, 1: mask -> random action).
    The reward is the original reward + ALPHA * mask.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, base_env, pretrained_policy):
        super().__init__()
        self.base_env = base_env
        self.pretrained_policy = pretrained_policy
        self.observation_space = base_env.observation_space
        self.action_space = gym.spaces.Discrete(2)  # mask or not
        self.state = None

    def reset(self, **kwargs):
        self.state = self.base_env.reset(**kwargs)
        return self.state

    def step(self, mask_action):
        # mask_action: 0 (keep) or 1 (mask)
        if mask_action == 0:
            # Use pre‑trained policy action
            action, _ = self.pretrained_policy.predict(self.state, deterministic=True)
        else:
            # Mask: sample random action
            action = self.base_env.action_space.sample()
        next_state, reward, done, info = self.base_env.step(action)
        # Intrinsic bonus
        bonus = ALPHA * mask_action
        reward += bonus
        self.state = next_state
        return next_state, reward, done, info

    def render(self, mode="human"):
        return self.base_env.render(mode)

    def close(self):
        self.base_env.close()

# Create environments
def make_env():
    env = gym.make(ENV_ID)
    env.seed(SEED)
    return env

env = DummyVecEnv([make_env])
pretrained = PPO.load(PRETRAINED_MODEL)
mask_env = MaskEnv(env.envs[0], pretrained)  # unwrap vec env

# Build mask network policy
class MaskPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        # Replace policy net with our MaskNet
        self.policy_net = MaskNet(observation_space.shape[0])

    def _build_mlp_extractor(self):
        # Not used because we override policy_net
        pass

    def forward(self, obs, deterministic=False):
        logits = self.policy_net(obs)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            m = Categorical(probs)
            action = m.sample()
        return action, probs

# Train mask network with PPO
model = PPO(
    MaskPolicy,
    mask_env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    seed=SEED,
)

print("Training mask network...")
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=".", name_prefix="mask")
model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)

# Save the mask model
os.makedirs(os.path.dirname(MASK_MODEL_DIR), exist_ok=True)
model.save(MASK_MODEL_DIR)
print(f"Mask network saved to {MASK_MODEL_DIR}")