import gym
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import random
from typing import List, Tuple


# ---------- Environment wrappers ----------
class PenalisedCartPole(gym.Wrapper):
    """
    CartPole where taking the left action (action 0) incurs a heavy penalty.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action: int):
        obs, reward, done, info = self.env.step(action)
        if action == 0:  # left
            reward -= 10.0
        return obs, reward, done, info


# ---------- AppleRetrieval toy environment ----------
class AppleRetrieval(gym.Env):
    """
    1‑D grid world with two phases: go right to a goal at position M (CLOSE),
    then go left back to 0 (FAR).  Rewards are +1 for moving in the
    correct direction, -1 otherwise.  The episode ends when the goal is
    reached or after 100 steps.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, start: int = 0, M: int = 10, max_steps: int = 100):
        super().__init__()
        self.M = M
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Box(low=0, high=M, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)  # 0: left, 1: right
        self.start = start
        self.reset()

    def reset(self):
        self.pos = self.start
        self.phase = 1  # 1: CLOSE, 2: FAR
        self.steps = 0
        return np.array([self.pos], dtype=np.float32)

    def step(self, action: int):
        self.steps += 1
        # Determine if action is correct
        correct = False
        if self.phase == 1 and action == 1:  # move right in CLOSE
            correct = True
        elif self.phase == 2 and action == 0:  # move left in FAR
            correct = True

        reward = 1.0 if correct else -1.0

        # Update position
        if action == 0:
            self.pos = max(0, self.pos - 1)
        else:
            self.pos = min(self.M, self.pos + 1)

        # Check for phase transition
        if self.phase == 1 and self.pos == self.M:
            self.phase = 2

        done = False
        if self.pos == 0 and self.phase == 2:
            done = True
        if self.steps >= self.max_steps:
            done = True

        return np.array([self.pos], dtype=np.float32), reward, done, {}

    def render(self, mode="human"):
        grid = ['.'] * (self.M + 1)
        grid[self.pos] = 'A'
        print("".join(grid) + f"  Phase {self.phase}")


# ---------- Replay buffer ----------
class ReplayBuffer:
    """
    Simple buffer that stores (state, action) pairs.
    """
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []

    def push(self, state: np.ndarray, action: int):
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
        self.states.append(state)
        self.actions.append(action)

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.states), batch_size, replace=False)
        states = np.array([self.states[i] for i in idx], dtype=np.float32)
        actions = np.array([self.actions[i] for i in idx], dtype=np.int64)
        return states, actions

    def __len__(self):
        return len(self.states)


# ---------- KL divergence ----------
def kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """
    KL( p || q ) where p and q are distributions derived from logits.
    Returns the mean KL over the batch.
    """
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    logp = torch.log(p + 1e-12)
    logq = torch.log(q + 1e-12)
    kl = torch.sum(p * (logp - logq), dim=-1)
    return kl.mean()


# ---------- Fisher diagonal for EWC ----------
def compute_fisher(
    model: nn.Module,
    env: gym.Env,
    num_samples: int,
    device: torch.device,
    batch_size: int = 128,
) -> dict:
    """
    Estimate the diagonal Fisher Information Matrix for the given model.
    We sample `num_samples` trajectories from `env` and accumulate the squared
    gradients of the log‑probabilities of the taken actions.
    """
    fisher = {name: torch.zeros_like(p, device=device) for name, p in model.named_parameters()}
    model.eval()
    for _ in range(num_samples):
        # Sample a trajectory
        states = []
        actions = []
        obs = env.reset()
        done = False
        while not done:
            states.append(obs)
            with torch.no_grad():
                state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model(state_t)
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            actions.append(action)
            obs, _, done, _ = env.step(action)

        # Compute log‑prob gradients
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        logits, _ = model(states_t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze()
        loss = -selected_log_probs.sum()
        loss.backward()
        # Accumulate squared gradients
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach() ** 2
        model.zero_grad()
    # Average over samples
    for name in fisher:
        fisher[name] /= num_samples
    return fisher