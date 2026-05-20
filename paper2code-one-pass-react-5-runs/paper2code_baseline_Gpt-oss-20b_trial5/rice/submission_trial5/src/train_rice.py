"""
Apply the RICE refinement pipeline:
  1. Collect a buffer of states from the baseline policy.
  2. Create a mixed‑initial‑state environment.
  3. Wrap the environment with RND intrinsic reward.
  4. Train a new PPO agent on the modified environment.
The refined policy is saved as refined_model.zip.
"""
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from .rnd import RNDTarget, RNDPredictor, rnd_bonus


# --------------------------------------------------------------------------- #
# 1. Buffer of critical states
# --------------------------------------------------------------------------- #
def collect_state_buffer(env, model, num_episodes=20):
    buffer = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            buffer.append(obs)
            obs = next_obs
            done = terminated or truncated
    return np.array(buffer)


# --------------------------------------------------------------------------- #
# 2. Mixed‑initial‑state wrapper
# --------------------------------------------------------------------------- #
class MixedInitEnv(gym.Wrapper):
    """
    Wraps an env so that on reset it either returns the default initial state
    or samples a state from a buffer with probability `p`.
    """
    def __init__(self, env, buffer_states, p=0.5):
        super().__init__(env)
        self.buffer_states = buffer_states
        self.p = p

    def reset(self, *, seed=None, options=None):
        if np.random.rand() < self.p and len(self.buffer_states) > 0:
            state = self.buffer_states[np.random.randint(len(self.buffer_states))]
            # CartPole exposes the underlying state as a numpy array
            self.env.state = state
            obs = state
        else:
            obs, _ = self.env.reset(seed=seed, options=options)
        return obs, {}


# --------------------------------------------------------------------------- #
# 3. RND wrapper
# --------------------------------------------------------------------------- #
class RndEnv(gym.Wrapper):
    """
    Adds an intrinsic reward based on RND to the environment.
    """
    def __init__(self, env, predictor, target, lam=0.01, device='cpu'):
        super().__init__(env)
        self.predictor = predictor
        self.target = target
        self.lam = lam
        self.device = device
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        bonus = rnd_bonus(self.predictor, self.target, obs, device=self.device)
        reward = reward + self.lam * bonus

        # Train predictor
        state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            target_out = self.target(state)
        pred_out = self.predictor(state)
        loss = F.mse_loss(pred_out, target_out)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return obs, reward, terminated, truncated, info


# --------------------------------------------------------------------------- #
# 4. Main training routine
# --------------------------------------------------------------------------- #
def main():
    # Load baseline policy
    baseline_env = gym.make("CartPole-v1")
    baseline_model = PPO.load("baseline_model.zip")

    # Collect buffer of states
    print("Collecting state buffer from baseline policy...")
    state_buffer = collect_state_buffer(baseline_env, baseline_model, num_episodes=20)
    baseline_env.close()

    # Create mixed‑init env
    mixed_env = MixedInitEnv(gym.make("CartPole-v1"), state_buffer, p=0.5)

    # RND networks
    obs_dim = mixed_env.observation_space.shape[0]
    target_net = RNDTarget(obs_dim).to("cpu")
    predictor_net = RNDPredictor(obs_dim).to("cpu")

    # Wrap with RND
    rnd_env = RndEnv(mixed_env, predictor_net, target_net, lam=0.01, device="cpu")

    # Vectorised env for SB3
    vec_env = DummyVecEnv([lambda: rnd_env])

    # Train refined policy
    print("Training refined policy with RICE...")
    refined_model = PPO("MlpPolicy", vec_env, verbose=1, seed=42)
    refined_model.learn(total_timesteps=20000)
    refined_model.save("refined_model.zip")
    vec_env.close()
    print("Refinement finished. Model saved as refined_model.zip.")


if __name__ == "__main__":
    main()