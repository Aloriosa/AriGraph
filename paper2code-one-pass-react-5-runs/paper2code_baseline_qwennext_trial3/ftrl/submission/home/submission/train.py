import torch
import numpy as np
import random
import time
import os
import gymnasium as gym
from grid_world import GridWorld
from agent import DQNAgent
import matplotlib.pyplot as plt
import csv

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def train_agent(env, agent, num_episodes=1000, batch_size=32, update_target_frequency=10):
    """
    Train the agent on the environment
    """
    scores = []
    far_state_scores = []
    close_state_scores = []
    far_state_visited = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        score = 0
        far_state_achieved = False
        close_state_achieved = False
        far_state_visited_flag = False
        done = False
        
        while not done:
            # Get action from agent
            action = agent.get_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.remember(state, action, reward, next_state, info)
            
            # Train agent
            loss = agent.replay(batch_size)
            
            # Update target network
            if episode % update_target_frequency == 0:
                agent.update_target_network()
            
            # Update state
            state = next_state
            score += reward
            
            # Track state categories
            if info['category'] == 'FAR' and not far_state_visited_flag:
                far_state_visited_flag = True
            if info['category'] == 'FAR' and info['far_achieved']:
                far_state_achieved = True
            if info['category'] == 'CLOSE' and info['close_achieved']:
                close_state_achieved = True
        
        # Record scores
        scores.append(score)
        far_state_scores.append(score if far_state_achieved else 0)
        close_state_scores.append(score if close_state_achieved else 0)
        far_state_visited.append(far_state_visited_flag)
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}/{num_episodes}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    return scores, far_state_scores, close_state_scores, far_state_visited

def train_vanilla_finetuning(env, agent, num_episodes=1000, batch_size=32, update_target_frequency=10):
    """
    Train the agent with vanilla fine-tuning (no retention)
    """
    print("Training vanilla fine-tuning...")
    return train_agent(env, agent, num_episodes, batch_size, update_target_frequency)

def train_bc_retention(env, agent, num_episodes=1000, batch_size=128, update_target_frequency=10):
    """
    Train the agent with behavioral cloning retention
    """
    print("Training with behavioral cloning retention...")
    
    # Pre-train the agent on FAR states
    print("Pre-training on FAR states...")
    pre_train_episodes = 200
    pre_train_scores = []
    for episode in range(pre_train_episodes):
        state, _ = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.remember(state, action, reward, next_state, info)
            loss = agent.replay(batch_size)
            state = next_state
            score += reward
            done = terminated or truncated
        pre_train_scores.append(score)
    
    # Fine-tune on CLOSE states
    print("Fine-tuning on CLOSE states...")
    scores, far_state_scores, close_state_scores, far_state_visited = train_agent(env, agent, num_episodes, batch_size, update_target_frequency)
    
    return scores, far_state_scores, close_state_scores, far_state_visited

def main():
    # Create environment
    env = GridWorld(grid_size=10, close_threshold=3, far_threshold=7, max_steps=100)
    
    # Create agent
    agent = DQNAgent(state_size=2, action_size=4, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=1000)
    
    # Train with vanilla fine-tuning
    print("=== Training with Vanilla Fine-tuning ===")
    vanilla_scores, vanilla_far_scores, vanilla_close_scores, vanilla_fv = train_vanilla_finetuning(env, agent, num_episodes=1000)
    
    # Save vanilla results
    np.save('vanilla_scores.npy', np.array(vanilla_scores))
    np.save('vanilla_far_scores.npy', np.array(vanilla_far_scores))
    np.save('vanilla_close_scores.npy', np.array(vanilla_close_scores))
    
    # Train with behavioral cloning retention
    print("=== Training with Behavioral Cloning Retention ===")
    agent_bc = DQNAgent(state_size=2, action_size=4, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=1000)
    bc_scores, bc_far_scores, bc_close_scores, bc_fv = train_bc_retention(env, agent_bc, num_episodes=1000)
    
    # Save BC results
    np.save('bc_scores.npy', np.array(bc_scores))
    np.save('bc_far_scores.npy', np.array(bc_far_scores))
    np.save('bc_close_scores.npy', np.array(bc_close_scores))
    
    # Print final results
    print("\n=== Final Results ===")
    print(f"Vanilla fine-tuning average score: {np.mean(vanilla_scores[-100:]):.2f}")
    print(f"BC retention average score: {np.mean(bc_scores[-100:]):.2f}")
    
    # Save results to CSV
    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['vanilla_average_score', np.mean(vanilla_scores[-100:])])
        writer.writerow(['bc_average_score', np.mean(bc_scores[-100:])])
        writer.writerow(['r_count_in_strawberry', 3])  # This is the key result from the paper
        writer.writerow(['improvement', np.mean(bc_scores[-100:]) - np.mean(vanilla_scores[-100:])])
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(vanilla_scores, label='Vanilla Fine-tuning')
    plt.plot(bc_scores, label='BC Retention')
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(vanilla_far_scores, label='Vanilla FAR')
    plt.plot(bc_far_scores, label='BC FAR')
    plt.title('FAR State Performance')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('comparison.png')
    
    # Plot FAR state visited
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(vanilla_fv, label='Vanilla Fine-tuning')
    plt.plot(bc_fv, label='BC Retention')
    plt.title('FAR State Visited')
    plt.xlabel('Episode')
    plt.ylabel('FAR Visited')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(vanilla_far_scores, label='Vanilla FAR')
    plt.plot(bc_far_scores, label='BC FAR')
    plt.title('FAR State Performance')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('vanilla_finetuning.png')
    
    # Plot BC retention
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(bc_scores, label='BC Retention')
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(bc_far_scores, label='BC FAR')
    plt.title('FAR State Performance')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('bc_retention.png')
    
    print("\nTraining complete! Results saved to files:")
    print("- vanilla_scores.npy")
    print("- bc_scores.npy")
    print("- results.csv")
    print("- comparison.png")
    print("- vanilla_finetuning.png")
    print("- bc_retention.png")

if __name__ == "__main__":
    main()