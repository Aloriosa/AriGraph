import numpy as np
import torch
import gymnasium as gym
import pickle
import argparse
import os

def generate_trajectories(env_name, num_trajectories=100, max_steps=1000, policy_path=None):
    """
    Generate offline trajectories using a pre-trained policy.
    If no policy is provided, use a random policy.
    """
    env = gym.make(env_name)
    
    trajectories = []
    
    for traj_idx in range(num_trajectories):
        obs, _ = env.reset()
        states = []
        actions = []
        rewards = []
        dones = []
        
        for step in range(max_steps):
            states.append(obs)
            
            # Use random policy if no pretrained policy provided
            if policy_path is None:
                action = env.action_space.sample()
            else:
                # Load policy and get action
                policy = torch.load(policy_path, map_location='cpu')
                policy.eval()
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action_dist = policy(obs_tensor)
                    action = action_dist.sample().squeeze().numpy()
            
            actions.append(action)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            dones.append(terminated or truncated)
            
            if terminated or truncated:
                break
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool_)
        
        trajectories.append({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'length': len(states)
        })
    
    # Save trajectories
    with open('data/offline_trajectories.pkl', 'wb') as f:
        pickle.dump(trajectories, f)
    
    print(f"Generated {len(trajectories)} trajectories with {sum(len(t['states']) for t in trajectories)} total steps")
    return trajectories

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--num_trajectories', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--policy_path', type=str, default=None)
    
    args = parser.parse_args()
    
    generate_trajectories(args.env_name, args.num_trajectories, args.max_steps, args.policy_path)