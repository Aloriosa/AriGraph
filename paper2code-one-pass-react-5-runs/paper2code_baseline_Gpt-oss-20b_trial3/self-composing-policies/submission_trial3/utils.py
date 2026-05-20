import json
import os
from pathlib import Path
import torch
import numpy as np
import gymnasium as gym
import gymnasium_robotics  # for Meta‑World
from tqdm import tqdm


def load_sequences(path: str = "sequences.json"):
    with open(path, "r") as f:
        seq = json.load(f)
    return seq


def make_env(env_name: str, task_id: int = None, seed: int = 0):
    """
    Create a gymnasium environment.
    For Atari games we set `task_id` to the mode id.
    For Meta‑World we ignore `task_id`.
    """
    if env_name.startswith("ALE"):
        env = gym.make(env_name, render_mode=None, disable_env_checker=True)
        env.unwrapped.set_mode(task_id)
    else:  # Meta‑World
        env = gym.make(env_name, render_mode=None, disable_env_checker=True)
    env.seed(seed)
    env.action_space.seed(seed)
    return env


def get_success_score(env):
    """
    Return the success threshold for an environment.
    For Atari we use 90% of the mean final score of all methods
    (this is defined in the paper).
    For Meta‑World we use the environment's `is_success` flag.
    """
    if isinstance(env, gymnasium.envs.atari.AtariEnv):
        # placeholder – users should replace with actual threshold
        return 200.0
    else:
        return None  # not used


def save_curve(path: Path, curve: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, curve)