import numpy as np
import torch
import torch.nn as nn
import pickle
import argparse
import os
from torch.autograd import grad

def integrated_gradients(state, action, policy, baseline=None, steps=50):
    """
    Compute Integrated Gradients for state-action pairs.
    
    Integrated Gradients computes the integral of gradients along a path from a baseline to the input.
    """
    if baseline is None:
        baseline = torch.zeros_like(state)
    
    # Scale the input
    scaled_inputs = [baseline + (float(i) / steps) * (state - baseline) for i in range(0, steps + 1)]
    
    # Compute gradients
    gradients = []
    for scaled_input in scaled_inputs:
        scaled_input.requires_grad_(True)
        mean, std = policy(scaled_input)
        action_dist = torch.distributions.Normal(mean, std)
        # Use log probability as the target for gradient computation
        log_prob = action_dist.log_prob(action).sum()
        gradient = grad(log_prob, scaled_input, retain_graph=True)[0]
        gradients.append(gradient)
    
    # Average gradients
    avg_gradients = torch.mean(torch.stack(gradients), dim=0)
    
    # Compute integrated gradients
    integrated_grad = (state - baseline) * avg_gradients
    
    return integrated_grad

def extract_critical_states(trajectory_path, pretrained_policy_path, critical_states_path, top_k=10, explanation_method="integrated_gradients"):
    """
    Extract critical states from trajectories using an explanation method.
    
    Critical states are defined as states with high explanatory significance (high integrated gradients magnitude).
    """
    # Load trajectories
    with open(trajectory_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    # Load pretrained policy
    policy = torch.load(pretrained_policy_path, map_location='cpu')
    policy.eval()
    
    # Extract all state-action pairs from trajectories
    state_action_pairs = []
    all_states = []
    all_actions = []
    
    for traj in trajectories:
        states = traj['states']
        actions = traj['actions']
        
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            all_states.append(state)
            all_actions.append(action)
            state_action_pairs.append((state, action))
    
    # Compute explanation scores for each state-action pair
    scores = []
    
    print(f"Computing {explanation_method} scores for {len(state_action_pairs)} state-action pairs...")
    
    for i, (state, action) in enumerate(state_action_pairs):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        
        if explanation_method == "integrated_gradients":
            # Compute Integrated Gradients
            ig = integrated_gradients(state_tensor, action_tensor, policy)
            # Use L2 norm of gradients as importance score
            score = torch.norm(ig).item()
        else:
            # Default to simple gradient
            state_tensor.requires_grad_(True)
            mean, std = policy(state_tensor)
            action_dist = torch.distributions.Normal(mean, std)
            log_prob = action_dist.log_prob(action_tensor).sum()
            gradient = grad(log_prob, state_tensor)[0]
            score = torch.norm(gradient).item()
        
        scores.append(score)
    
    # Convert to numpy array
    scores = np.array(scores)
    
    # Select top-k% critical states
    num_critical = max(1, int(len(scores) * top_k / 100))
    critical_indices = np.argsort(scores)[-num_critical:]
    
    # Extract critical states
    critical_states = [all_states[i] for i in critical_indices]
    critical_scores = [scores[i] for i in critical_indices]
    
    # Save critical states
    critical_data = {
        'critical_states': np.array(critical_states),
        'critical_scores': np.array(critical_scores),
        'total_states': len(all_states),
        'critical_count': len(critical_states),
        'top_k_percent': top_k
    }
    
    with open(critical_states_path, 'wb') as f:
        pickle.dump(critical_data, f)
    
    print(f"Extracted {len(critical_states)} critical states ({top_k}% of {len(all_states)} total states)")
    print(f"Critical state scores range: {np.min(critical_scores):.4f} to {np.max(critical_scores):.4f}")
    
    return critical_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_path', type=str, default='data/offline_trajectories.pkl')
    parser.add_argument('--pretrained_policy_path', type=str, default='models/pretrained_ppo.pth')
    parser.add_argument('--critical_states_path', type=str, default='data/critical_states.pkl')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--explanation_method', type=str, default='integrated_gradients')
    
    args = parser.parse_args()
    
    extract_critical_states(
        args.trajectory_path, 
        args.pretrained_policy_path, 
        args.critical_states_path, 
        args.top_k, 
        args.explanation_method
    )