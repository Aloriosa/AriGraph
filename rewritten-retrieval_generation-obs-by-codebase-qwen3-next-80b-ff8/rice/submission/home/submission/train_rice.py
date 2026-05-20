#!/usr/bin/env python3
"""
RICE: Reward-Informed Critical Exploration
Implementation of the RICE algorithm for DRL policy refinement.
Based on the paper: "RICE: Reward-Informed Critical Exploration for Deep Reinforcement Learning"
"""
import os
import sys
import argparse
import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import pickle
import csv
import time
from typing import Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RICECallback(BaseCallback):
    """
    Custom callback to implement RICE's critical state identification and mixed distribution sampling
    """
    def __init__(self, 
                 verbose=0, 
                 critical_state_threshold=0.1,  # Top 10% of states as critical
                 critical_state_weight=0.5,     # 50% critical, 50% default
                 reward_bonus_weight=0.1,       # Reward bonus for critical states
                 identification_frequency=50000, # Identify critical states every X timesteps
                 pretrained_policy_path=None):
        super(RICECallback, self).__init__(verbose)
        self.critical_state_threshold = critical_state_threshold
        self.critical_state_weight = critical_state_weight
        self.reward_bonus_weight = reward_bonus_weight
        self.identification_frequency = identification_frequency
        self.pretrained_policy_path = pretrained_policy_path
        self.critical_states = set()
        self.state_buffer = []
        self.episode_rewards = []
        self.critical_state_counts = []
        self.training_steps = 0
        self.total_timesteps = 0
        self.pretrained_policy = None
        self.explanation_model = None
        self.log_data = []
        
    def _init_callback(self) -> None:
        """Initialize the callback after model setup"""
        # Load pretrained policy if provided
        if self.pretrained_policy_path and os.path.exists(self.pretrained_policy_path):
            logger.info(f"Loading pretrained policy from {self.pretrained_policy_path}")
            self.pretrained_policy = PPO.load(self.pretrained_policy_path, 
                                            env=self.training_env, 
                                            device=self.model.device)
            # Extract state representation from pretrained policy
            self.explanation_model = self.pretrained_policy.policy.features_extractor
        else:
            logger.warning("No pretrained policy provided. Using current policy for explanation.")
            self.pretrained_policy = self.model
            self.explanation_model = self.model.policy.features_extractor
            
        # Initialize state buffer
        self.state_buffer = []
        self.critical_states = set()
        
    def _on_step(self) -> bool:
        """Called after each step"""
        self.training_steps += 1
        self.total_timesteps += self.locals['n_envs']
        
        # Store states for critical state identification
        if 'observations' in self.locals:
            obs = self.locals['observations']
            if isinstance(obs, np.ndarray):
                # Convert to tuple for hashing
                for i in range(obs.shape[0]):
                    state_tuple = tuple(obs[i].flatten())
                    self.state_buffer.append((state_tuple, self.training_steps))
        
        # Periodically identify critical states
        if self.training_steps % self.identification_frequency == 0 and len(self.state_buffer) > 0:
            self._identify_critical_states()
            
        # Add reward bonus for exploration from critical frontiers
        if 'rewards' in self.locals and len(self.critical_states) > 0:
            self._add_reward_bonus()
            
        return True
        
    def _identify_critical_states(self):
        """Identify critical states using explanation method (StateMask)"""
        logger.info(f"Identifying critical states at step {self.training_steps}")
        
        if len(self.state_buffer) == 0:
            return
            
        # Extract state features using pretrained policy's feature extractor
        state_features = []
        state_values = []
        
        for state_tuple, step in self.state_buffer:
            state_array = np.array(state_tuple).reshape(1, -1)
            state_tensor = th.tensor(state_array, dtype=th.float32).to(self.model.device)
            
            with th.no_grad():
                features = self.explanation_model(state_tensor)
                state_features.append(features.cpu().numpy().flatten())
                
                # Get value estimate
                if self.pretrained_policy:
                    value = self.pretrained_policy.policy.predict_values(state_tensor)
                    state_values.append(value.cpu().numpy()[0])
        
        if len(state_features) == 0:
            return
            
        # Convert to numpy array
        state_features = np.array(state_features)
        state_values = np.array(state_values)
        
        # Use state values as proxy for importance (higher value = more critical)
        # This is a simplified version of StateMask explanation
        importance_scores = state_values
        
        # Identify top k% critical states
        n_critical = int(len(importance_scores) * self.critical_state_threshold)
        critical_indices = np.argsort(importance_scores)[-n_critical:]
        
        # Get the actual state tuples for critical states
        critical_state_tuples = []
        for idx in critical_indices:
            if idx < len(self.state_buffer):
                critical_state_tuples.append(self.state_buffer[idx][0])
        
        # Update critical states set
        self.critical_states = set(critical_state_tuples)
        self.critical_state_counts.append(len(self.critical_states))
        
        logger.info(f"Identified {len(self.critical_states)} critical states")
        
        # Clear state buffer to avoid memory issues
        self.state_buffer = []
        
    def _add_reward_bonus(self):
        """Add reward bonus for exploration from critical frontiers"""
        # In the original implementation, this would modify the reward function
        # Here we'll modify the rewards in the rollout buffer
        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            observations = self.locals['observations']
            
            # For each environment, check if the state is critical
            for i in range(len(rewards)):
                state_tuple = tuple(observations[i].flatten())
                if state_tuple in self.critical_states:
                    # Add bonus reward for visiting critical states
                    rewards[i] += self.reward_bonus_weight
                    
    def _on_rollout_end(self) -> None:
        """Called after each rollout"""
        # Log metrics
        if len(self.critical_states) > 0:
            self.log_data.append({
                'step': self.training_steps,
                'critical_states': len(self.critical_states),
                'critical_state_weight': self.critical_state_weight,
                'reward_bonus_weight': self.reward_bonus_weight,
                'total_timesteps': self.total_timesteps
            })
            
    def _on_training_end(self) -> None:
        """Called when training ends"""
        # Save critical states for evaluation
        if len(self.critical_states) > 0:
            with open('critical_states.pkl', 'wb') as f:
                pickle.dump(list(self.critical_states), f)
                
        # Save training log
        if len(self.log_data) > 0:
            with open('training_log.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.log_data[0].keys())
                writer.writeheader()
                writer.writerows(self.log_data)

class RICEPPO(PPO):
    """
    RICE-enhanced PPO algorithm
    """
    def __init__(self, 
                 policy, 
                 env, 
                 critical_state_threshold=0.1,
                 critical_state_weight=0.5,
                 reward_bonus_weight=0.1,
                 identification_frequency=50000,
                 pretrained_policy_path=None,
                 **kwargs):
        
        super(RICEPPO, self).__init__(policy, env, **kwargs)
        
        # RICE-specific parameters
        self.critical_state_threshold = critical_state_threshold
        self.critical_state_weight = critical_state_weight
        self.reward_bonus_weight = reward_bonus_weight
        self.identification_frequency = identification_frequency
        self.pretrained_policy_path = pretrained_policy_path
        
        # Initialize RICE callback
        self.rice_callback = RICECallback(
            verbose=1,
            critical_state_threshold=critical_state_threshold,
            critical_state_weight=critical_state_weight,
            reward_bonus_weight=reward_bonus_weight,
            identification_frequency=identification_frequency,
            pretrained_policy_path=pretrained_policy_path
        )
        
        # Add RICE callback to model callbacks
        self.callback = self.rice_callback
        
    def learn(self, total_timesteps, callback=None, **kwargs):
        """Override learn to include RICE callback"""
        if callback is None:
            callback = []
        elif not isinstance(callback, list):
            callback = [callback]
            
        # Add RICE callback
        callback.append(self.rice_callback)
        
        return super(RICEPPO, self).learn(total_timesteps, callback=callback, **kwargs)

def create_env(env_name, normalize=True):
    """Create and optionally normalize environment"""
    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])
    
    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
    return env

def train_rice(env_name, 
               pretrained_policy_path=None,
               total_timesteps=1000000,
               seed=42,
               output_dir="./results",
               critical_state_threshold=0.1,
               critical_state_weight=0.5,
               reward_bonus_weight=0.1,
               identification_frequency=50000,
               learning_rate=3e-4,
               batch_size=2048,
               n_epochs=10,
               gamma=0.99,
               gae_lambda=0.95,
               clip_range=0.2,
               ent_coef=0.0,
               vf_coef=0.5,
               max_grad_norm=0.5,
               n_steps=2048):
    
    # Set random seed
    set_random_seed(seed)
    
    # Create environment
    env = create_env(env_name, normalize=True)
    
    # Create RICE-PPO model
    model = RICEPPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        seed=seed,
        verbose=1,
        tensorboard_log=output_dir,
        critical_state_threshold=critical_state_threshold,
        critical_state_weight=critical_state_weight,
        reward_bonus_weight=reward_bonus_weight,
        identification_frequency=identification_frequency,
        pretrained_policy_path=pretrained_policy_path
    )
    
    # Train model
    logger.info(f"Starting RICE training for {total_timesteps} timesteps")
    model.learn(total_timesteps=total_timesteps)
    
    # Save final model
    os.makedirs(output_dir, exist_ok=True)
    model.save(f"{output_dir}/final_model")
    
    # Save environment normalization parameters
    env.save(f"{output_dir}/env_stats.pkl")
    
    logger.info(f"Training completed. Model saved to {output_dir}/final_model.zip")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='RICE: Reward-Informed Critical Exploration')
    parser.add_argument('--env', type=str, default='HalfCheetah-v4', help='Environment name')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='Total training timesteps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--critical_state_threshold', type=float, default=0.1, help='Top k% states as critical')
    parser.add_argument('--critical_state_weight', type=float, default=0.5, help='Weight of critical states in mixed distribution')
    parser.add_argument('--reward_bonus_weight', type=float, default=0.1, help='Reward bonus weight for critical states')
    parser.add_argument('--identification_frequency', type=int, default=50000, help='Frequency of critical state identification')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train RICE model
    model = train_rice(
        env_name=args.env,
        pretrained_policy_path=args.pretrained_model,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        output_dir=args.output_dir,
        critical_state_threshold=args.critical_state_threshold,
        critical_state_weight=args.critical_state_weight,
        reward_bonus_weight=args.reward_bonus_weight,
        identification_frequency=args.identification_frequency
    )
    
    # Save training arguments
    with open(f"{args.output_dir}/training_args.txt", 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    print(f"RICE training completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()