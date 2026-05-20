import os
import gym
import numpy as np
from stable_baselines3 import PPO
from utils import set_global_seeds, evaluate_policy

# Configuration
ENV_ID = "CartPole-v1"
MODEL_PATH = "base.zip"
CRITICAL_STATES_FILE = "critical_states.npy"
TOTAL_EPISODES = 20
HORIZON = 5
TOP_K = 200
SEED = 43  # Different seed for collection

def collect_states(env, model, n_episodes, horizon):
    """
    For each visited state, compute the sum of rewards obtained by following
    the policy for `horizon` steps starting from that state. This sum is used
    as an importance score.
    """
    states = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, _ = env.step(action)
            # Compute future horizon reward
            future_reward = 0.0
            tmp_obs = next_obs
            tmp_done = done
            for _ in range(horizon):
                if tmp_done:
                    break
                tmp_action, _ = model.predict(tmp_obs, deterministic=True)
                tmp_obs, r, tmp_done, _ = env.step(tmp_action)
                future_reward += r
            states.append((next_obs, future_reward))
            obs = next_obs
    return states

def main():
    set_global_seeds(SEED)

    # Load base agent
    model = PPO.load(MODEL_PATH)

    # Environment for collection
    env = gym.make(ENV_ID)
    env.seed(SEED)
    env.action_space.seed(SEED)

    # Collect states
    collected = collect_states(env, model, TOTAL_EPISODES, HORIZON)
    print(f"Collected {len(collected)} state samples")

    # Sort by importance and keep top_k
    collected.sort(key=lambda x: x[1], reverse=True)
    top_states = [s for s, _ in collected[:TOP_K]]
    np.save(CRITICAL_STATES_FILE, np.array(top_states))
    print(f"Saved {len(top_states)} critical states to {CRITICAL_STATES_FILE}")

if __name__ == "__main__":
    main()