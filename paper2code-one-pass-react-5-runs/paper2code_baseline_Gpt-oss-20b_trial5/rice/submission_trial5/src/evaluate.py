"""
Evaluate the baseline and refined policies.
Prints the average return over a fixed number of episodes.
"""
import gymnasium as gym
from stable_baselines3 import PPO


def evaluate(model_path, env_name="CartPole-v1", n_episodes=10):
    env = gym.make(env_name)
    model = PPO.load(model_path)
    total_reward = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_reward += ep_reward
    avg_reward = total_reward / n_episodes
    print(f"Average reward for {model_path}: {avg_reward}")
    env.close()


if __name__ == "__main__":
    evaluate("baseline_model.zip")
    evaluate("refined_model.zip")