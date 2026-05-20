# RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation

This repository contains the complete reproduction of the RICE paper "RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation" (Cheng et al., 2024).

## Overview

The RICE (Refining scheme for ReInforCement learning with Explanation) framework aims to break through training bottlenecks in deep reinforcement learning by incorporating explanation methods to identify critical states and construct mixed initial state distributions.

The key innovations of RICE are:
1. **Explanation-based critical state identification**: Using a novel state mask network to identify the most critical states that contribute most to the final reward
2. **Mixed initial state distribution**: Combining default initial states with identified critical states to prevent overfitting
3. **Exploration incentive**: Encouraging exploration from these mixed initial states to expand state coverage

## Reproduction Instructions

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support
- Docker (optional)

### Installation