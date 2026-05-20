import os
import gym
import json
import numpy as np
from stable_baselines3 import PPO
from utils import set_global_seeds, evaluate_policy
from env_wrappers import MixedInitWrapper, RNDWrapper, RND

# Configuration
ENV_ID = "CartPole-v1"
MODEL_PATH = "refine.zip"
BASE_MODEL_PATH = "base.zip"
CRITICAL_STATES_FILE = "critical_states.npy"
TOTAL_TIMESTEPS = 50_000
SEED = 44
EVAL_EPISODES = 10
P_MIX = 0.25        # probability to start from a critical state
LAMBDA_INTRINSIC = 0.01
HIDDEN_DIM = 128

def main():
    set_global_seeds(SEED)

    # Load base agent and critical states
    base_model = PPO.load(BASE_MODEL_PATH)
    critical_states = np.load(CRITICAL_STATES_FILE)
    print(f"Loaded {len(critical_states)} critical states")

    # Build mixed‑init environment
    env = gym.make(ENV_ID)
    env.seed(SEED)
    env.action_space.seed(SEED)
    mixed_env = MixedInitWrapper(env, critical_states, p=P_MIX, seed=SEED)

    # Build RND wrapper
    obs_dim = env.observation_space.shape[0]
    rnd = RND(obs_dim, hidden_dim=HIDDEN_DIM, device='cpu')
    rnd_env = RNDWrapper(mixed_env, rnd, intrinsic_scale=LAMBDA_INTRINSIC, device='cpu')

    # Train refined PPO agent on the wrapped environment
    model = PPO("MlpPolicy", rnd_env, verbose=0, seed=SEED)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH)

    # Evaluate
    eval_env = gym.make(ENV_ID)
    eval_env.seed(SEED)
    avg_reward = evaluate_policy(eval_env, model, n_episodes=EVAL_EPISODES, seed=SEED)
    print(f"Refinement training finished. Avg reward over {EVAL_EPISODES} eval episodes: {avg_reward:.2f}")

    # Also evaluate base agent for comparison
    base_eval_env = gym.make(ENV_ID)
    base_eval_env.seed(SEED)
    base_avg = evaluate_policy(base_eval_env, base_model, n_episodes=EVAL_EPISODES, seed=SEED)

    # Save results
    results = {
        "base_avg_reward": base_avg,
        "refine_avg_reward": avg_reward,
        "critical_states_count": len(critical_states)
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results written to results.json")

if __name__ == "__main__":
    main()