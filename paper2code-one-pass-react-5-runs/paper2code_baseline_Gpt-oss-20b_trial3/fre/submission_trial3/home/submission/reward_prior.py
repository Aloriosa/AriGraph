import numpy as np
import torch

def sample_goal_reaching(state_batch, rng):
    """
    Random goal‑reaching reward: 0 if state is within 0.1 of goal, -1 otherwise.
    """
    goal = rng.uniform(0.0, 1.0, size=(1, state_batch.shape[1])).astype(np.float32)
    dist = np.linalg.norm(state_batch - goal, axis=1)
    rewards = np.where(dist < 0.1, 0.0, -1.0).astype(np.float32)
    return goal.squeeze(), rewards

def sample_linear(state_batch, rng):
    """
    Random linear reward: <w, state>
    """
    w = rng.uniform(-1.0, 1.0, size=(state_batch.shape[1],)).astype(np.float32)
    rewards = np.dot(state_batch, w).astype(np.float32)
    return w, rewards

def sample_mlp(state_batch, rng):
    """
    Small 2‑layer MLP reward: tanh( w2 * tanh(w1 * state) )
    """
    d_in = state_batch.shape[1]
    w1 = rng.normal(0, 1.0, size=(d_in, 32)).astype(np.float32)
    b1 = rng.normal(0, 0.1, size=(32,)).astype(np.float32)
    w2 = rng.normal(0, 1.0, size=(32, 1)).astype(np.float32)
    b2 = rng.normal(0, 0.1, size=(1,)).astype(np.float32)

    hidden = np.tanh(np.dot(state_batch, w1) + b1)
    out = np.tanh(np.dot(hidden, w2) + b2).squeeze()
    return (w1, b1, w2, b2), out

def sample_random_reward(state_batch, rng):
    """
    Uniformly sample one of the reward types.
    Returns a tuple (reward_function, rewards) where reward_function is a callable.
    """
    choice = rng.choice(['goal', 'linear', 'mlp'])
    if choice == 'goal':
        goal, rewards = sample_goal_reaching(state_batch, rng)
        def func(state):
            return np.where(np.linalg.norm(state - goal, axis=1) < 0.1, 0.0, -1.0).astype(np.float32)
        return func, rewards
    elif choice == 'linear':
        w, rewards = sample_linear(state_batch, rng)
        def func(state):
            return np.dot(state, w).astype(np.float32)
        return func, rewards
    else:
        (w1, b1, w2, b2), rewards = sample_mlp(state_batch, rng)
        def func(state):
            hidden = np.tanh(np.dot(state, w1) + b1)
            out = np.tanh(np.dot(hidden, w2) + b2).squeeze()
            return out.astype(np.float32)
        return func, rewards