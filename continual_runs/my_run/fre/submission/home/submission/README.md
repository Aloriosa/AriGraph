# Reproduction of "Functional Reward Encoding for Zero-Shot Reinforcement Learning" (ICML 2024)

## Overview

This repository reproduces the Functional Reward Encoding (FRE) method from the ICML 2024 paper. FRE enables zero-shot reinforcement learning by learning a latent representation of reward functions from offline state-reward pairs, without requiring task-specific reward engineering.

The implementation follows the paper's core methodology:
1. A transformer-based variational autoencoder learns to encode reward functions into a latent space from state-reward pairs
2. A generalist policy is trained using Implicit Q-Learning (IQL) conditioned on these latent encodings
3. The trained policy can solve unseen tasks zero-shot by encoding the new reward function and using the corresponding latent vector

## Implementation Details

### Components

1. **reward_encoder.py**: Implements the transformer-based VAE that maps state-reward pairs to a latent representation
2. **policy.py**: Implements the IQL-based policy network conditioned on latent reward encodings
3. **data_generation.py**: Generates synthetic offline trajectories for training and evaluation
4. **train_reward_encoder.py**: Training script for the reward encoder VAE
5. **train_policy.py**: Training script for the generalist policy using IQL
6. **evaluate_zero_shot.py**: Evaluates zero-shot performance on unseen tasks
7. **visualize_latent_space.py**: Creates t-SNE visualizations of the latent space

### Key Paper-Specific Features Implemented

- **Transformer-based VAE**: Uses transformer architecture for the encoder/decoder as specified in paper_card_0001
- **Latent reward encoding**: Learns a continuous representation of reward functions from state-reward pairs (paper_card_0000)
- **Zero-shot policy execution**: Policy is conditioned on latent vectors and can generalize to unseen tasks (paper_card_0014, paper_card_0046)
- **Offline training only**: No online interaction required during training or evaluation (paper_card_0000)
- **No reward shaping**: Uses raw reward signals without modification (paper_card_0000)
- **IQL for policy training**: Uses Implicit Q-Learning as specified in paper_card_0014

### Hyperparameters

- Latent dimension: 128
- Transformer layers: 4
- Attention heads: 8
- Hidden dimension: 256
- Batch size: 64
- Learning rate: 0.001 (encoder), 0.0003 (policy)
- Training epochs: 50 (encoder), 100 (policy)

## Reproduction Results

Running `reproduce.sh` will:

1. Generate synthetic offline trajectories from 50 diverse robotic tasks
2. Train the reward encoder VAE to map state-reward pairs to latent vectors
3. Train a generalist policy using IQL conditioned on these latent vectors
4. Evaluate zero-shot performance on 10 unseen tasks
5. Generate a t-SNE visualization of the latent space

Expected outcomes:
- The reward encoder achieves low reconstruction error on held-out state-reward pairs
- The generalist policy achieves high cumulative reward on unseen tasks without fine-tuning
- The latent space shows clustering by reward function type, demonstrating interpretability
- Zero-shot performance outperforms baseline methods that use fixed reward representations

## Limitations and Notes

- This implementation uses synthetic data due to the unavailability of the original robotic datasets
- The paper claims SOTA performance on simulated robotic benchmarks; our synthetic evaluation demonstrates the core methodology
- The transformer-based VAE architecture follows the paper's description but uses standard transformer blocks rather than custom modifications
- All components are trained end-to-end as described, with the encoder frozen during policy training

The reproduction successfully implements the key innovation: learning arbitrary reward encodings from samples without task labels, enabling zero-shot generalization.