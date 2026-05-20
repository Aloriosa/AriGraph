# CompoNet: Self-Composing Policies for Scalable Continual Reinforcement Learning

This repository contains the complete reproduction of the paper "Self-Composing Policies for Scalable Continual Reinforcement Learning" (Malagón et al., 2024).

## Overview

The paper introduces CompoNet, a novel architecture for continual reinforcement learning that uses self-composing policy modules. Unlike previous approaches that grow quadratically in parameters with respect to the number of tasks, CompoNet grows linearly, making it highly scalable.

The key insight of CompoNet is that each module can selectively combine previous policies with its own internal policy. This is achieved through:
1. **Output attention head**: Proposes an output based on previous policies and the current state
2. **Input attention head**: Retrieves relevant information from previous policies and the output attention head
3. **Internal policy**: Adjusts the tentative output from the output attention head

This design allows CompoNet to:
- Reuse previous policies when they solve the current task
- Learn functions over previous policies when they can solve the current task
- Learn from scratch when previous policies provide no useful information

## Reproduction

To reproduce the results from the paper, run the following command: