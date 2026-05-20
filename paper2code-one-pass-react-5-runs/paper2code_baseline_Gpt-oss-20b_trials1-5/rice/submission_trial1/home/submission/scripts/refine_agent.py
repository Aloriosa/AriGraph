#!/usr/bin/env python3
import os
import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import set_random_seed

from rices.mask_network import MaskNet
from rices.rnd import RandomNetwork, PredictorNetwork

SEED = 42
set_random_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ENV_ID = "CartPole-v1"
PRETRAINED_MODEL = "../models/pretrained_ppo.zip"
MASK_MODEL = "../models/mask_net.zip"
REFINED_MODEL = "../models/refined_ppo.zip"

# Hyper‑parameters for refinement
P = 0.25          # probability of sampling a critical state
LAMBDA = 0.01     # intrinsic bonus weight
N_EPISODES = 50   # refinement episodes
TIMESTEPS_PER_EPISODE = 2048

# ----------------------------------------------------------------------
# Helper: environment that can reset to an arbitrary state
# ----------------------------------------------------------------------
class CartPoleWithReset(gym.envs.classic_control.CartPoleEnv):
    def reset_to_state(self, state):
        self.state = np.array(state, dtype=np.float32)
        self.steps_beyond_done = None
        return self.state

# ----------------------------------------------------------------------
# Load models
# ----------------------------------------------------------------------
pretrained = PPO.load(PRETRAINED_MODEL, verbose=0)
mask_net = torch.load(MASK_MODEL, map_location="cpu")
mask_net.eval()

# ----------------------------------------------------------------------
# Collect critical states using the mask network
# ----------------------------------------------------------------------
def collect_critical_states(num_episodes=5, max_steps=500):
    env = CartPoleWithReset()
    critical_states = []
    for _ in range(num_episodes):
        state = env.reset()
        for _ in range(max_steps):
            # Get mask probability
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = mask_net(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            mask_prob = probs[0, 1].item()  # probability of masking
            # We consider states with high mask probability as *critical*
            if mask_prob > 0.8:   # threshold can be tuned
                critical_states.append(state.copy())
            # step with pretrained policy
            action, _ = pretrained.predict(state, deterministic=True)
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                break
    return critical_states

print("Collecting critical states...")
critical_states = collect_critical_states()
print(f"Collected {len(critical_states)} critical states.")

# ----------------------------------------------------------------------
# RND networks
# ----------------------------------------------------------------------
obs_dim = env.observation_space.shape[0]
rnd_target = RandomNetwork(obs_dim)
rnd_predictor = PredictorNetwork(obs_dim)
rnd_optimizer = torch.optim.Adam(rnd_predictor.parameters(), lr=1e-3)

# ----------------------------------------------------------------------
# Custom policy that uses pretrained actions and applies RND bonus
# ----------------------------------------------------------------------
class RefinePolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.policy_net = pretrained.policy.net   # reuse pretrained network

    def forward(self, obs, deterministic=False):
        # Use pretrained action
        action, _ = pretrained.predict(obs, deterministic=deterministic)
        return torch.tensor([action]), torch.ones(1, dtype=torch.float32)

# ----------------------------------------------------------------------
# Custom environment for refinement
# ----------------------------------------------------------------------
class RefinementEnv(gym.Env):
    def __init__(self, base_env, pretrained_policy, critical_states, p):
        super().__init__()
        self.base_env = base_env
        self.pretrained_policy = pretrained_policy
        self.critical_states = critical_states
        self.p = p
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space
        self.state = None

    def reset(self):
        if np.random.rand() < self.p and self.critical_states:
            # Sample a critical state
            self.state = np.array(np.random.choice(self.critical_states), dtype=np.float32)
        else:
            self.state = self.base_env.reset()
        return self.state

    def step(self, action=None):
        # Pick action from pretrained policy
        action, _ = self.pretrained_policy.predict(self.state, deterministic=True)
        next_state, reward, done, info = self.base_env.step(action)

        # RND intrinsic bonus
        state_tensor = torch.tensor(self.state, dtype=torch.float32)
        target = rnd_target(state_tensor)
        pred = rnd_predictor(state_tensor)
        intrinsic = torch.norm(target - pred, p=2).item() ** 2
        reward += LAMBDA * intrinsic

        self.state = next_state
        return next_state, reward, done, info

    def render(self, mode="human"):
        return self.base_env.render(mode)

    def close(self):
        self.base_env.close()

# ----------------------------------------------------------------------
# Train refined policy
# ----------------------------------------------------------------------
env = CartPoleWithReset()
refine_env = RefinementEnv(env, pretrained, critical_states, P)
vec_env = DummyVecEnv([lambda: refine_env])

model = PPO(
    RefinePolicy,
    vec_env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    seed=SEED,
)

print("Refining policy with RICE...")
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=".", name_prefix="refine")
model.learn(total_timesteps=N_EPISODES * TIMESTEPS_PER_EPISODE,
            callback=checkpoint_callback)

# Save refined model
os.makedirs(os.path.dirname(REFINED_MODEL), exist_ok=True)
model.save(REFINED_MODEL)
print(f"Refined policy saved to {REFINED_MODEL}")