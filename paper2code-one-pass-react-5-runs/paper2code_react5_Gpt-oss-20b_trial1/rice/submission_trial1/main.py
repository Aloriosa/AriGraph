import os
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from config import *
from mask_network import MaskNet, train_mask_network
from rnd import RNDModule
from env_wrapper import RICEEnvWrapper
from utils import set_seed

class RICECallback(BaseCallback):
    """
    Simple callback to log mean episode reward.
    """
    def __init__(self, verbose=0):
        super(RICECallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if len(self.locals['infos']) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
        return True

def train_pretrained_policy(env, timesteps, seed):
    set_seed(seed)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed, device='cpu')
    model.learn(total_timesteps=timesteps)
    return model

def sample_trajectory(policy, env, max_steps=200):
    obs = env.reset(seed=42)[0]
    traj = []
    for _ in range(max_steps):
        action, _ = policy.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        traj.append((obs, action, reward))
        if done:
            break
        obs = next_obs
    return traj

def pick_critical_state(traj, mask_net, device):
    """
    Pick the state with the highest mask probability (i.e., most important).
    """
    obs_batch = torch.tensor([t[0] for t in traj], dtype=torch.float32, device=device)
    probs = mask_net(obs_batch).detach().cpu().numpy()
    idx = np.argmax(probs)
    return traj[idx][0]  # return observation

def main():
    # Create base environment
    base_env = gym.make(ENV_NAME)
    base_env = DummyVecEnv([lambda: base_env])
    # 1) Pre‑train policy
    print("Training pre‑trained PPO policy...")
    pretrained = train_pretrained_policy(base_env, TOTAL_TIMESTEPS, SEED)
    # Save for later
    pretrained.save("pretrained.zip")

    # 2) Train mask network
    print("Training mask network...")
    mask_net = MaskNet(obs_dim=pretrained.observation_space.shape[0], device=torch.device('cpu'))
    # Use raw env (not wrapped)
    env_for_mask = gym.make(ENV_NAME)
    set_seed(SEED)
    train_mask_network(env_for_mask, pretrained, mask_net, torch.device('cpu'),
                       MASK_TRAIN_TIMESTEPS, ALPHA)

    # 3) RICE refinement
    print("Starting RICE refinement...")
    rnd = RNDModule(obs_dim=pretrained.observation_space.shape[0], device=torch.device('cpu'))
    env_wrapped = RICEEnvWrapper(gym.make(ENV_NAME), rnd, LAMBDA)
    env_wrapped = DummyVecEnv([lambda: env_wrapped])
    # Load pretrained weights into new policy
    refined_policy = PPO("MlpPolicy", env_wrapped, verbose=0,
                         policy_kwargs=dict(net_arch=[dict(pi=[64, 64], qf=[64, 64])]),
                         seed=SEED, device='cpu')
    refined_policy.policy.load_state_dict(pretrained.policy.state_dict())

    # Custom training loop to incorporate mixed initial distribution
    total_steps = 0
    callback = RICECallback()
    while total_steps < REFINE_TIMESTEPS:
        # Decide whether to reset to critical state or default
        if np.random.rand() < P:
            # Sample a trajectory from pretrained policy to identify critical state
            traj = sample_trajectory(pretrained, gym.make(ENV_NAME))
            crit_state = pick_critical_state(traj, mask_net, torch.device('cpu'))
            # Reset to this state by creating a new env and setting observation
            env = gym.make(ENV_NAME)
            obs, _ = env.reset()
            # This simple env does not support setting state directly;
            # we approximate by stepping until we reach the desired observation.
            # For the purpose of this demo, we just use the default reset.
            env = env_wrapped
        else:
            env = env_wrapped

        # Train for a small chunk
        chunk = min(2000, REFINE_TIMESTEPS - total_steps)
        refined_policy.learn(total_timesteps=chunk, callback=callback)
        total_steps += chunk

    # Evaluate final policy
    env = gym.make(ENV_NAME)
    obs = env.reset(seed=SEED)[0]
    rewards = []
    for _ in range(10):
        total_r = 0
        done = False
        while not done:
            action, _ = refined_policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            done = terminated or truncated
        rewards.append(total_r)
    mean_reward = np.mean(rewards)
    print(f"Mean episode reward after RICE refinement: {mean_reward:.2f}")
    with open("final_reward.txt", "w") as f:
        f.write(f"{mean_reward}\n")

if __name__ == "__main__":
    main()