# Reproduction: Self-Composing Policies for Scalable Continual Reinforcement Learning

This repository contains the complete implementation and reproduction of the paper "Self-Composing Policies for Scalable Continual Reinforcement Learning" by Mikel Malagón et al. (2024).

## Overview

The paper introduces CompoNet, a novel architecture for Continual Reinforcement Learning that uses self-composing policy modules to avoid catastrophic forgetting while enabling linear parameter growth with task count.

The key contributions are:
1. **Self-Composing Policies**: Each policy module can selectively compose previous policies with its internal policy, enabling knowledge transfer without interference.
2. **Linear Parameter Growth**: Unlike previous approaches that grow quadratically, CompoNet grows linearly with the number of tasks.
3. **Scalability**: CompoNet demonstrates superior scalability and performance on benchmark tasks.

## Reproduction Instructions

### Prerequisites
- Ubuntu 24.04 LTS
- NVIDIA A10 GPU
- Docker (optional, for containerized execution)

### Setup and Execution

1. Clone this repository
2. Navigate to the repository directory
3. Make the reproduction script executable: