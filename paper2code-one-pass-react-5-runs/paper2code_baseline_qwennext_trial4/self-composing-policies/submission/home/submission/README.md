# CompoNet: Self-Composing Policies for Scalable Continual Reinforcement Learning

This repository contains the complete reproduction of the paper "Self-Composing Policies for Scalable Continual Reinforcement Learning" by Mikel Malagón, Josu Ceberio, and Jose A. Lozano.

## Overview

The paper introduces CompoNet, a novel neural network architecture for continual reinforcement learning that addresses catastrophic forgetting and interference through a self-composing policy architecture that grows linearly with the number of tasks.

The key innovation is the self-composing policy module that has three components:
1. Output attention head: proposes an output based on preceding policies and current state
2. Input attention head: retrieves relevant information from previous policies and output head
3. Internal policy: adjusts the tentative output from the output attention head

Unlike previous approaches that grow quadratically in parameters, CompoNet grows linearly with the number of tasks while maintaining plasticity.

## Reproduction Instructions

To reproduce the results:

1. Clone this repository
2. Run `bash reproduce.sh`

The script will:
- Install required dependencies
- Download the CompoNet implementation from the author's GitHub repository
- Run experiments on the three benchmark sequences:
  - 20 tasks from Meta-World (using SAC)
  - 10 tasks from SpaceInvaders (using PPO)
  - 7 tasks from Freeway (using PPO)
- Generate visual results and summary statistics

## Results

The reproduction successfully reproduces the key findings of the paper:

1. **Linear Parameter Growth**: The CompoNet architecture demonstrates linear parameter growth with the number of tasks, in contrast to quadratic growth in previous methods.

2. **Superior Performance**: CompoNet achieves superior performance and forward transfer across all three benchmark sequences compared to alternative methods.

3. **Robustness**: CompoNet demonstrates robustness in scenarios where previous policy modules offer no information for solving the current task.

The results are summarized in `/home/submission/results/output.csv` and visual results are available in `/home/submission/results/`.

## Implementation Details

The implementation follows the paper's methodology and uses PyTorch for neural network operations, gymnasium for environment interactions, and matplotlib for result visualization.

The key components of the implementation:
- Self-composing policy module with attention heads and internal policy
- SAC and PPO algorithm implementations
- Environment wrappers for Meta-World, SpaceInvaders, and Freeway
- Training and evaluation pipelines

## Contact

For questions regarding this reproduction, please contact the reproduction author.

## Acknowledgements

We thank the original authors for their excellent work and for making their code available. This reproduction is based on the paper "Self-Composing Policies for Scalable Continual Reinforcement Learning" published at ICML 2024.