import gymnasium as gym
import numpy as np
import os
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import set_global_seeds, evaluate_policy
from mixed_reset_env import MixedResetEnv
from rnd_wrapper import RNDWrapper
from constants import *

def main():
    set_global_seeds(REFINE_SEED)

    # Load baseline model and critical states
    base_model = PPO.load("results/baseline.zip")
    critical_states = np.load("results/critical_states.npy")

    # Environment with mixed initial states
    base_env = gym.make(ENV_NAME)
    mixed_env = MixedResetEnv(base_env, critical_states, P_MIX)

    # Wrap with RND intrinsic reward
    env = DummyVecEnv([lambda: RNDWrapper(mixed_env, LAMBDA_RND)])

    # Build new policy starting from the baseline weights
    model = PPO("MlpPolicy", env, verbose=1,
                tensorboard_log="./logs/refine/",
                learning_rate=3e-4,
                batch_size=64,
                n_steps=2048,
                gamma=0.99,
                seed=REFINE_SEED)

    # Copy weights from baseline
    model.set_parameters(base_model.get_parameters())

    # Train refinement
    model.learn(total_timesteps=REFINE_TIMESTEPS,
                log_interval=REFINE_LOG_FREQ)
    os.makedirs("results", exist_ok=True)
    model.save("results/refined.zip")
    print("\nRefined model saved to results/refined.zip")

    # Evaluation
    eval_env = gym.make(ENV_NAME)
    avg_ret = evaluate_policy(model, eval_env, n_episodes=10, seed=REFINE_SEED)
    print(f"\nEvaluation of refined policy over 10 episodes: average return = {avg_ret:.2f}")

if __name__ == "__main__":
    main()