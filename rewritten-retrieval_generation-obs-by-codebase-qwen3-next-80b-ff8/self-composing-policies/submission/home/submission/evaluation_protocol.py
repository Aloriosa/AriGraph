#!/usr/bin/env python3
"""
Evaluation protocol for continual RL benchmarks.
Implements evaluation metrics as described in the paper.
"""
import numpy as np
import torch
import gymnasium as gym
from typing import List, Optional, Tuple
import time

def evaluate_agent(agent, env, num_evals: int, device: torch.device, model_type: str) -> float:
    """
    Evaluate the agent on a single task.
    
    Args:
        agent: The agent to evaluate
        env: The environment to evaluate on
        num_evals: Number of episodes to evaluate
        device: Device to run evaluation on
        model_type: Type of model being evaluated
        
    Returns:
        Average episodic return
    """
    returns = []
    
    for _ in range(num_evals):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                if model_type == "componet":
                    mean, logstd = agent(obs_tensor)
                else:
                    mean, logstd = agent(obs_tensor)
                
                # Use mean for deterministic evaluation
                action = mean.cpu().numpy()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        returns.append(total_reward)
    
    return np.mean(returns)

def save_evaluation_results(eval_results: List[dict], filepath: str):
    """
    Save evaluation results to a file.
    
    Args:
        eval_results: List of evaluation results
        filepath: Path to save the results
    """
    import pickle
    
    with open(filepath, 'wb') as f:
        pickle.dump(eval_results, f)