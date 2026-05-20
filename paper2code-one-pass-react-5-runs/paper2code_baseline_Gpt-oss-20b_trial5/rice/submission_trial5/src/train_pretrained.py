"""
Train a baseline PPO agent on CartPole-v1.
The trained policy is saved as baseline_model.zip.
"""
import gymnasium as gym
from stable_baselines3 import PPO


def main():
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1, seed=42)
    model.learn(total_timesteps=20000)
    model.save("baseline_model.zip")
    env.close()


if __name__ == "__main__":
    main()