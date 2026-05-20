import os
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.utils import set_global_seed
from src.rnd_wrapper import RNDWrapper

# -------------------- CONFIGURATION --------------------
SEED = 42
ENV_ID = "Hopper-v3"          # Change to "HalfCheetah-v3", "Walker2d-v3", etc.
BASE_TRAIN_TIMESTEPS = 200_000
REFINE_TRAIN_TIMESTEPS = 400_000
EVAL_EPISODES = 10
P_CRITIC = 0.25          # probability of sampling a critical state
LAMBDA_RND = 0.01
MODEL_DIR = "models"
CRITIC_TOP_K = 0.20      # percentage of states considered critical
os.makedirs(MODEL_DIR, exist_ok=True)
# -------------------------------------------------------


def evaluate_policy(env, policy, episodes=EVAL_EPISODES):
    """Run `episodes` episodes and return average return."""
    total_reward = 0.0
    for _ in range(episodes):
        obs, _ = env.reset(seed=SEED)
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_reward += ep_reward
    return total_reward / episodes


def collect_critical_states(env, policy, num_episodes=5, top_k=CRITIC_TOP_K):
    """
    Run `num_episodes` episodes with `policy` and collect states with the
    largest advantage estimates.  Advantage is computed as:
        A(s,a) = r + γ * V(s') - V(s)
    """
    states = []
    advantages = []

    gamma = 0.99

    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset(seed=SEED)
            done = False
            while not done:
                action, _ = policy.predict(obs, deterministic=True)
                # Value of current state
                v_s = policy.policy.value(torch.as_tensor(obs, dtype=torch.float32))

                # Take a step
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Value of next state
                v_next = policy.policy.value(torch.as_tensor(next_obs, dtype=torch.float32))

                # Advantage estimate
                adv = reward + gamma * v_next - v_s
                states.append(obs)
                advantages.append(adv.item())

                obs = next_obs

    # Select top_k fraction of states
    num_critics = max(1, int(len(states) * top_k))
    # Get indices of states with largest advantage
    idxs = np.argsort(advantages)[-num_critics:]
    critical_states = [states[i] for i in idxs]
    return critical_states


def main():
    set_global_seed(SEED)

    # ----------------- Base Policy -----------------
    base_env = gym.make(ENV_ID)
    base_env = DummyVecEnv([lambda: base_env])

    print("Training baseline PPO agent...")
    base_model = PPO(
        "MlpPolicy",
        base_env,
        verbose=0,
        seed=SEED,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
    )
    base_model.learn(total_timesteps=BASE_TRAIN_TIMESTEPS)
    base_model_path = os.path.join(MODEL_DIR, "base_model.zip")
    base_model.save(base_model_path)

    # Evaluate baseline
    base_eval_env = gym.make(ENV_ID)
    base_reward = evaluate_policy(base_eval_env, base_model)
    print(f"Base Reward: {base_reward:.2f}")

    # ----------------- Collect critical states ----------
    # Use the trained policy to generate trajectories and collect states
    collect_env = gym.make(ENV_ID)
    critical_states = collect_critical_states(collect_env, base_model, num_episodes=5)
    print(f"Collected {len(critical_states)} critical states.")

    # ----------------- Refine with RICE -----------------
    refine_env = gym.make(ENV_ID)
    rnd_wrapped_env = RNDWrapper(
        refine_env,
        lambda_rnd=LAMBDA_RND,
        critical_states=critical_states,
        p=P_CRITIC,
    )
    refine_env_vec = DummyVecEnv([lambda: rnd_wrapped_env])

    print("Fine‑tuning agent with RICE (RND + mixed init)...")
    # Load base policy weights into the new model
    refine_model = PPO(
        "MlpPolicy",
        refine_env_vec,
        verbose=0,
        seed=SEED,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
    )
    refine_model.policy.load_state_dict(base_model.policy.state_dict())
    refine_model.learn(total_timesteps=REFINE_TRAIN_TIMESTEPS)

    refine_reward = evaluate_policy(gym.make(ENV_ID), refine_model)
    print(f"Refined Reward: {refine_reward:.2f}")

    # ----------------- Save results -----------------
    with open("output.txt", "w") as f:
        f.write(f"Base Reward: {base_reward:.2f}\n")
        f.write(f"Refined Reward: {refine_reward:.2f}\n")

    print("Results written to output.txt")


if __name__ == "__main__":
    main()