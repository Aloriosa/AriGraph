import os
import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import set_global_seeds, evaluate_policy

# Configuration
ENV_ID = "CartPole-v1"
MODEL_PATH = "base.zip"
TOTAL_TIMESTEPS = 50_000
SEED = 42
EVAL_EPISODES = 10

def main():
    set_global_seeds(SEED)

    # Create env
    env = DummyVecEnv([lambda: gym.make(ENV_ID)])
    env.seed(SEED)
    env.action_space.seed(SEED)

    # Train base PPO agent
    model = PPO("MlpPolicy", env, verbose=0, seed=SEED)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH)

    # Evaluate
    env_unwrapped = gym.make(ENV_ID)
    env_unwrapped.seed(SEED)
    avg_reward = evaluate_policy(env_unwrapped, model, n_episodes=EVAL_EPISODES, seed=SEED)
    print(f"Base training finished. Avg reward over {EVAL_EPISODES} eval episodes: {avg_reward:.2f}")

if __name__ == "__main__":
    main()