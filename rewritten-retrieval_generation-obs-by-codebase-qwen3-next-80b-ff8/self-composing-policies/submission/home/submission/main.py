#!/usr/bin/env python3
"""
Main training script for CompoNet (Growable Modular Neural Network) for Continual RL.
Implements the architecture described in the paper with attention-based policy composition.
"""
import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium as gym
import metaworld
from metaworld import MT1, MT10, MT50
import wandb
from collections import deque
import pickle
from typing import List, Optional, Tuple
from tqdm import tqdm

# Import local modules
from models.compo import CompoNetAgent
from models.simple import SimpleAgent
from models.packnet import PackNet
from models.prognet import ProgNet
from models.finetune import FinetuneAgent
from task_sequence_loader import get_task_sequence
from evaluation_protocol import evaluate_agent, save_evaluation_results
from utils import set_seed, create_env, create_writer

# Define model types
MODEL_TYPES = ["simple", "finetune", "componet", "packnet", "prognet"]

class Args:
    model_type: str
    """The name of the NN model to use for the agent"""
    save_dir: Optional[str] = None
    """If provided, the model will be saved in the given directory"""
    prev_units: Tuple[str, ...] = ()
    """Paths to the previous models. Not required when model_type is `simple` or `packnet` or `prognet`"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cw-sac"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    task_sequence: str = "meta_world_20"
    """The task sequence to use for training"""
    eval_every: int = 10000
    """Evaluate the agent in deterministic mode every X timesteps"""
    num_evals: int = 10
    """Number of times to evaluate the agent"""
    total_timesteps: int = int(1e6)
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5000
    """timestep to start learning"""
    random_actions_end: int = 10000
    """timesteps to take actions randomly"""
    policy_lr: float = 1e-3
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

def main():
    args = Args()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="componet", choices=MODEL_TYPES)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--prev_units", type=str, nargs='+', default=[])
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__)[:-3])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch_deterministic", action="store_true", default=True)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--track", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default="cw-sac")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--capture_video", action="store_true", default=False)
    parser.add_argument("--task_sequence", type=str, default="meta_world_20")
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--num_evals", type=int, default=10)
    parser.add_argument("--total_timesteps", type=int, default=int(1e6))
    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_starts", type=int, default=5000)
    parser.add_argument("--random_actions_end", type=int, default=10000)
    parser.add_argument("--policy_lr", type=float, default=1e-3)
    parser.add_argument("--q_lr", type=float, default=1e-3)
    parser.add_argument("--policy_frequency", type=int, default=2)
    parser.add_argument("--target_network_frequency", type=int, default=1)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--autotune", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # Initialize wandb tracking
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"{args.exp_name}_{args.seed}",
            monitor_gym=True,
            save_code=True,
        )
    
    # Get task sequence
    task_sequence = get_task_sequence(args.task_sequence)
    
    # Initialize agent
    if args.model_type == "simple":
        agent = SimpleAgent(task_sequence[0].observation_space.shape[0], task_sequence[0].action_space.shape[0])
    elif args.model_type == "finetune":
        agent = FinetuneAgent(task_sequence[0].observation_space.shape[0], task_sequence[0].action_space.shape[0])
    elif args.model_type == "packnet":
        agent = PackNet(
            hidden_dim=256,
            task_id=0,
            total_task_num=len(task_sequence),
            is_first_task=True
        )
    elif args.model_type == "prognet":
        agent = ProgNet(
            obs_dim=task_sequence[0].observation_space.shape[0],
            act_dim=task_sequence[0].action_space.shape[0],
            hidden_dim=256
        )
    elif args.model_type == "componet":
        # Initialize the first CompoNet module
        if len(args.prev_units) > 0:
            prev_units = []
            for prev_path in args.prev_units:
                # Load previous model
                model = torch.load(prev_path, map_location=device)
                prev_units.append(model)
            agent = CompoNetAgent(
                obs_dim=task_sequence[0].observation_space.shape[0],
                act_dim=task_sequence[0].action_space.shape[0],
                prev_paths=args.prev_units,
                map_location=device
            )
        else:
            # First task - initialize with simple network
            agent = CompoNetAgent(
                obs_dim=task_sequence[0].observation_space.shape[0],
                act_dim=task_sequence[0].action_space.shape[0],
                prev_paths=[],
                map_location=device
            )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    agent.to(device)
    
    # Initialize optimizer
    if args.model_type in ["simple", "finetune", "packnet", "prognet"]:
        optimizer = optim.Adam(agent.parameters(), lr=args.policy_lr)
    else:  # componet
        optimizer = optim.Adam(agent.net_mean.parameters(), lr=args.policy_lr)
        optimizer_logstd = optim.Adam(agent.net_logstd.parameters(), lr=args.policy_lr)
    
    # Initialize replay buffer
    replay_buffer = deque(maxlen=args.buffer_size)
    
    # Training loop
    global_step = 0
    episode_returns = []
    episode_lengths = []
    current_task_idx = 0
    task_start_time = time.time()
    
    # Initialize evaluation results storage
    eval_results = []
    
    # Create directories for saving
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize environments
    envs = []
    for task in task_sequence:
        env = create_env(task)
        envs.append(env)
    
    # Training loop over tasks
    for task_idx, task in enumerate(task_sequence):
        print(f"Starting training on task {task_idx + 1}/{len(task_sequence)}: {task.name}")
        
        # Reset environment
        obs, _ = envs[task_idx].reset(seed=args.seed)
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        
        # Training for current task
        for step in range(args.total_timesteps // len(task_sequence)):
            global_step += 1
            
            # Sample action
            with torch.no_grad():
                if args.model_type == "componet":
                    mean, logstd = agent(obs)
                    std = torch.exp(logstd)
                    action = mean + std * torch.randn_like(mean)
                    action = torch.clamp(action, -1, 1)
                else:
                    mean, logstd = agent(obs)
                    std = torch.exp(logstd)
                    action = mean + std * torch.randn_like(mean)
                    action = torch.clamp(action, -1, 1)
            
            # Execute action
            next_obs, reward, terminated, truncated, info = envs[task_idx].step(action.cpu().numpy())
            done = terminated or truncated
            
            # Store transition
            replay_buffer.append((obs.cpu().numpy(), action.cpu().numpy(), reward, next_obs, done))
            
            # Update observation
            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            
            # Train when enough samples are collected
            if global_step > args.learning_starts:
                # Sample batch
                batch = random.sample(replay_buffer, min(len(replay_buffer), args.batch_size))
                batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = zip(*batch)
                
                batch_obs = torch.tensor(batch_obs, dtype=torch.float32, device=device)
                batch_actions = torch.tensor(batch_actions, dtype=torch.float32, device=device)
                batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
                batch_next_obs = torch.tensor(batch_next_obs, dtype=torch.float32, device=device)
                batch_dones = torch.tensor(batch_dones, dtype=torch.float32, device=device)
                
                # Compute loss and update
                if args.model_type == "componet":
                    # CompoNet uses SAC-like updates
                    mean, logstd = agent(batch_obs)
                    std = torch.exp(logstd)
                    action_dist = torch.distributions.Normal(mean, std)
                    log_probs = action_dist.log_prob(batch_actions).sum(-1)
                    
                    # Compute value loss
                    # Note: In a full SAC implementation, we'd have a critic network here
                    # For simplicity, we're using the actor as both policy and value estimator
                    value = mean.sum(-1)
                    target_value = batch_rewards + args.gamma * (1 - batch_dones) * value
                    
                    # Actor loss
                    actor_loss = -(log_probs * (log_probs - args.alpha).detach()).mean()
                    
                    # Critic loss (simplified)
                    critic_loss = nn.MSELoss()(value, target_value)
                    
                    # Update
                    optimizer.zero_grad()
                    optimizer_logstd.zero_grad()
                    (actor_loss + critic_loss).backward()
                    optimizer.step()
                    optimizer_logstd.step()
                    
                    # Log attention weights if available
                    if global_step % args.eval_every == 0:
                        mean, phi, att_in, att_out, int_pol, head_out = agent(
                            batch_obs, return_atts=True, ret_int_pol=True, ret_head_out=True
                        )
                        
                        if args.track:
                            for i, v in enumerate(att_in.mean(0)[0].detach()):
                                wandb.log({f"charts/att_in_{i}": v.item()}, step=global_step)
                            for i, v in enumerate(att_out.mean(0)[0].detach()):
                                wandb.log({f"charts/att_out_{i}": v.item()}, step=global_step)
                
                else:
                    # Simple, finetune, packnet, prognet use standard SAC updates
                    # This is a simplified version for demonstration
                    mean, logstd = agent(batch_obs)
                    std = torch.exp(logstd)
                    action_dist = torch.distributions.Normal(mean, std)
                    log_probs = action_dist.log_prob(batch_actions).sum(-1)
                    
                    # Simplified loss
                    actor_loss = -(log_probs * (log_probs - args.alpha).detach()).mean()
                    critic_loss = torch.tensor(0.0, device=device)
                    
                    optimizer.zero_grad()
                    (actor_loss + critic_loss).backward()
                    optimizer.step()
            
            # Evaluate agent periodically
            if global_step % args.eval_every == 0:
                print(f"Evaluating at step {global_step}...")
                eval_return = evaluate_agent(agent, envs[task_idx], args.num_evals, device, args.model_type)
                eval_results.append({
                    'step': global_step,
                    'task_idx': task_idx,
                    'return': eval_return,
                    'model_type': args.model_type
                })
                
                if args.track:
                    wandb.log({"eval/episode_return": eval_return}, step=global_step)
                
                # Save model checkpoint
                if args.save_dir:
                    model_path = os.path.join(args.save_dir, f"model_{global_step}.pt")
                    if args.model_type == "componet":
                        # Save CompoNet components separately
                        torch.save(agent.net_mean, os.path.join(args.save_dir, f"net_mean_{global_step}.pt"))
                        torch.save(agent.net_logstd, os.path.join(args.save_dir, f"net_logstd_{global_step}.pt"))
                    else:
                        torch.save(agent, model_path)
            
            # Check if episode is done
            if done:
                obs, _ = envs[task_idx].reset()
                obs = torch.tensor(obs, dtype=torch.float32, device=device)
        
        # After completing a task, save the current model for next task
        if args.model_type == "componet" and task_idx < len(task_sequence) - 1:
            # Save current module for next task
            model_path = os.path.join(args.save_dir, f"module_{task_idx}.pt")
            torch.save(agent.net_mean, model_path)
            print(f"Saved module {task_idx} to {model_path}")
    
    # Save final evaluation results
    if args.save_dir:
        results_path = os.path.join(args.save_dir, "eval_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(eval_results, f)
    
    # Print final results
    print("Training completed!")
    print(f"Final evaluation results saved to {results_path}")

if __name__ == "__main__":
    main()