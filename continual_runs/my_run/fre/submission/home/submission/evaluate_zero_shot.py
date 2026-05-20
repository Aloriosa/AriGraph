import torch
import torch.nn as nn
import pickle
import numpy as np
import argparse
import os
from reward_encoder import TransformerRewardEncoder
from policy import IQLPolicy
import csv

def evaluate_zero_shot(
    reward_encoder_path: str,
    policy_path: str,
    num_eval_tasks: int = 10,
    episodes_per_task: int = 5,
    device: str = 'cuda',
    output: str = 'results/zero_shot_results.csv'
):
    """
    Evaluate zero-shot performance on unseen tasks.
    
    Args:
        reward_encoder_path: Path to trained reward encoder
        policy_path: Path to trained policy
        num_eval_tasks: Number of evaluation tasks
        episodes_per_task: Number of episodes per task
        device: Device to use ('cuda' or 'cpu')
        output: Path to save results
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load reward encoder and policy
    # We need to know the state dimension from the saved models
    # Load a sample trajectory to get state dimension
    with open('data/offline_trajectories.pkl', 'rb') as f:
        trajectories = pickle.load(f)
    
    state_dim = trajectories[0]['states'].shape[1]
    action_dim = trajectories[0]['actions'].shape[1]
    latent_dim = 128  # From training configuration
    
    # Load models
    reward_encoder = TransformerRewardEncoder(
        state_dim=state_dim,
        latent_dim=latent_dim
    ).to(device)
    reward_encoder.load_state_dict(torch.load(reward_encoder_path, map_location=device))
    reward_encoder.eval()
    
    policy = IQLPolicy(
        state_dim=state_dim,
        latent_dim=latent_dim,
        action_dim=action_dim
    ).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    
    # Create evaluation tasks (different reward functions)
    eval_tasks = []
    for i in range(num_eval_tasks):
        # Generate random reward function parameters
        task_type = np.random.choice(['goal_reaching', 'velocity', 'energy_efficient', 'balance'])
        if task_type == 'goal_reaching':
            goal = np.random.randn(state_dim).astype(np.float32)
            eval_tasks.append({
                'type': task_type,
                'goal': goal
            })
        elif task_type == 'velocity':
            weight = np.random.randn(state_dim).astype(np.float32)
            eval_tasks.append({
                'type': task_type,
                'weight': weight
            })
        elif task_type == 'energy_efficient':
            eval_tasks.append({
                'type': task_type
            })
        else:  # balance
            eval_tasks.append({
                'type': task_type
            })
    
    # Evaluate on each task
    results = []
    
    for task_idx, task in enumerate(eval_tasks):
        task_rewards = []
        
        for episode in range(episodes_per_task):
            # Reset environment (simulated)
            state = np.random.randn(state_dim).astype(np.float32)
            total_reward = 0.0
            
            # Encode the reward function for this task
            # In practice, we would have state-reward pairs from the task
            # Here we'll create a synthetic set of state-reward pairs to encode
            num_samples = 10
            states_for_encoding = np.random.randn(num_samples, state_dim).astype(np.float32)
            rewards_for_encoding = np.zeros(num_samples)
            
            # Compute rewards for these samples based on task
            for i in range(num_samples):
                if task['type'] == 'goal_reaching':
                    rewards_for_encoding[i] = -np.linalg.norm(states_for_encoding[i] - task['goal'])
                elif task['type'] == 'velocity':
                    rewards_for_encoding[i] = np.dot(states_for_encoding[i], task['weight'])
                elif task['type'] == 'energy_efficient':
                    rewards_for_encoding[i] = -np.random.rand()  # Random energy cost
                else:  # balance
                    rewards_for_encoding[i] = -np.sum(np.square(states_for_encoding[i]))
            
            # Encode the reward function
            states_tensor = torch.from_numpy(states_for_encoding).float().to(device)
            rewards_tensor = torch.from_numpy(rewards_for_encoding).float().to(device)
            
            with torch.no_grad():
                z = reward_encoder.encode(states_tensor.unsqueeze(0), rewards_tensor.unsqueeze(0))
            
            # Run episode
            for step in range(50):  # 50 steps per episode
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    action = policy.get_action(state_tensor, z, deterministic=True)
                    action = action.cpu().numpy().squeeze()
                
                # Simulate next state (simple dynamics)
                state = state + action * 0.1 + np.random.randn(state_dim) * 0.01
                
                # Compute reward based on task
                if task['type'] == 'goal_reaching':
                    reward = -np.linalg.norm(state - task['goal'])
                elif task['type'] == 'velocity':
                    reward = np.dot(state, task['weight'])
                elif task['type'] == 'energy_efficient':
                    reward = -np.sum(np.abs(action))
                else:  # balance
                    reward = -np.sum(np.square(state))
                
                total_reward += reward
                
                # Check if episode is done (simplified)
                if np.linalg.norm(state) > 10.0:
                    break
            
            task_rewards.append(total_reward)
        
        # Calculate average reward for this task
        avg_reward = np.mean(task_rewards)
        std_reward = np.std(task_rewards)
        
        results.append({
            'task_id': task_idx,
            'task_type': task['type'],
            'average_reward': avg_reward,
            'std_reward': std_reward,
            'rewards': task_rewards
        })
        
        print(f"Task {task_idx} ({task['type']}): Avg Reward = {avg_reward:.2f} ± {std_reward:.2f}")
    
    # Save results to CSV
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task_id', 'task_type', 'average_reward', 'std_reward'])
        
        for result in results:
            writer.writerow([
                result['task_id'],
                result['task_type'],
                result['average_reward'],
                result['std_reward']
            ])
    
    print(f"Results saved to {output}")
    
    # Print summary
    avg_overall = np.mean([r['average_reward'] for r in results])
    std_overall = np.std([r['average_reward'] for r in results])
    print(f"\nOverall Performance: {avg_overall:.2f} ± {std_overall:.2f}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate zero-shot performance')
    parser.add_argument('--reward_encoder_path', type=str, default='models/reward_encoder.pth',
                        help='Path to trained reward encoder')
    parser.add_argument('--policy_path', type=str, default='models/policy.pth',
                        help='Path to trained policy')
    parser.add_argument('--num_eval_tasks', type=int, default=10,
                        help='Number of evaluation tasks')
    parser.add_argument('--episodes_per_task', type=int, default=5,
                        help='Number of episodes per task')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output', type=str, default='results/zero_shot_results.csv',
                        help='Path to save results')
    
    args = parser.parse_args()
    
    evaluate_zero_shot(
        reward_encoder_path=args.reward_encoder_path,
        policy_path=args.policy_path,
        num_eval_tasks=args.num_eval_tasks,
        episodes_per_task=args.episodes_per_task,
        device=args.device,
        output=args.output
    )